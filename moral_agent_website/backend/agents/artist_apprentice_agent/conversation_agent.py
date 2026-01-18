from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
import sys

# Gemini 지원 (선택적)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_all_prompts, load_json_data

# 모듈화된 핸들러 임포트
from .stages import Stage1Handler, Stage2Handler, Stage3Handler
from .routing import ConversationRouter
from .persona_llm import PersonaLLM

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """LangGraph 상태"""
    stage: str  # stage1, stage2, stage3
    previous_stage: str  # 이전 스테이지 (첫 방문 감지용)
    messages: List[Dict[str, str]]  # 대화 기록 (딕셔너리 리스트로 관리)
    covered_topics: List[str]  # LLM이 판단한 다룬 주제들
    current_asking_topic: str  # ✨ 현재 물어보고 있는 주제 (무한 루프 방지용)
    message_count: int
    last_response: str
    artist_character_set: bool  # ✨ 화가 캐릭터 설정 완료 플래그 (Stage 1→2 전환)
    should_end: bool  # ✨ 대화 종료 플래그 (5가지 주제 모두 다루면 True)
    stage1_attempts: int  # ✨ Stage1 반복 횟수 (무한 루프 방지용)
    # ✨ Stage 2 관련 플래그들
    stage2_question_asked: bool  # Stage 2 질문을 했는지
    stage2_complete: bool  # Stage 2 완료 여부 (Stage 3 전환용)
    stage2_completed: bool  # Stage 2 완전히 끝남 (중복 방지용)
    # ✨ Stage 3 관련 플래그들
    current_question_index: int  # 현재 질문 인덱스 (0~4)
    variation_index: int  # 현재 질문의 변형 인덱스 (0~2)


class ConversationAgent:
    """
    LangGraph 기반 자연스러운 대화 에이전트
    - 3단계 구조: Stage1(화가 캐릭터 설정) → Stage2(AI 예술 입장) → Stage3(윤리 탐구)
    - 명확한 단계별 목표 + 페르소나 중심 일관된 캐릭터 유지
    """

    def __init__(self, api_key: str, model: str = "gpt-5.1"):
        # 프롬프트 로드 (페르소나를 먼저 로드)
        prompts_path = Path(__file__).parent / "prompts"
        self.prompts = load_all_prompts(str(prompts_path))
        logger.info(f"Loaded {len(self.prompts)} prompts: {list(self.prompts.keys())}")

        # 페르소나 프롬프트 로드
        persona_prompt = self.prompts.get("persona", "당신은 예의 바른 화가 지망생입니다.")
        logger.info(f"Loaded persona prompt: {persona_prompt[:50]}...")

        # 기본 LLM 초기화
        # gpt-5-nano는 reasoning 모델이므로 max_completion_tokens를 크게 설정해야 함
        # (reasoning_tokens ~400 + output_tokens ~100 합산, 여유있게 1000 설정)
        base_llm = ChatOpenAI(
            model=model,
            temperature=1.0,  # gpt-5 계열은 temperature=1 고정
            max_completion_tokens=1000,  # gpt-5 reasoning 모델용 (reasoning ~400 + output ~100)
            api_key=api_key
        )

        # ✨ 페르소나가 적용된 LLM 생성 (Wrapper)
        self.llm = PersonaLLM(base_llm, persona_prompt)
        logger.info("✅ Created PersonaLLM with embedded persona")

        # ✨ 의도 분석용 LLM - Gemini 우선 (빠르고 저렴)
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if HAS_GEMINI and gemini_api_key:
            self.analyzer = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,  # 의도 분석은 결정적이어야 함
                google_api_key=gemini_api_key
            )
            logger.info("✅ Using Gemini (gemini-2.0-flash) for intent detection")
        else:
            # Gemini 없으면 GPT fallback
            self.analyzer = ChatOpenAI(
                model=model,
                temperature=1.0,
                max_completion_tokens=500,
                api_key=api_key
            )
            logger.info("⚠️ Using GPT for intent detection (Gemini not available)")

        # 윤리 주제 JSON 로드
        self.ethics_topics = load_json_data("ethics_topics.json", str(prompts_path))
        logger.info(f"Loaded {len(self.ethics_topics)} questions from ethics_topics")

        # ✨ Stage 핸들러 초기화 (페르소나가 이미 적용된 LLM 전달)
        # persona_prompt=""로 전달 (페르소나는 이미 LLM에 베이킹됨)
        self.stage1_handler = Stage1Handler(
            self.llm,
            self.prompts,
            persona_prompt="",  # 페르소나는 이미 LLM에 적용됨
            analyzer=self.analyzer
        )
        self.stage2_handler = Stage2Handler(
            self.llm,
            self.prompts,
            persona_prompt="",  # 페르소나는 이미 LLM에 적용됨
            analyzer=self.analyzer
        )
        self.stage3_handler = Stage3Handler(
            self.llm,
            self.analyzer,
            self.prompts,
            self.ethics_topics,
            persona_prompt=""  # 페르소나는 이미 LLM에 적용됨
        )

        # Router 초기화
        self.router = ConversationRouter(self.analyzer)

        # ✨ MemorySaver로 세션 상태 자동 관리
        self.memory = MemorySaver()

        # LangGraph 워크플로우 구성 (checkpointer 연결)
        self.graph = self._build_graph()

        logger.info(f"ConversationAgent initialized with LangGraph + MemorySaver, model={model}")

    def _build_graph(self) -> StateGraph:
        """
        LangGraph 워크플로우 구성 - 명확한 3단계 구조
        ✨ 각 노드는 응답 생성 후 무조건 END로 종료 (무한 루프 방지)
        ✨ Stage 전환은 다음 메시지에서 entry point에서 판단
        """
        workflow = StateGraph(ConversationState)

        # 노드 정의
        workflow.add_node("stage1", self.stage1)
        workflow.add_node("stage2", self.stage2)
        workflow.add_node("stage3", self.stage3)

        # Entry point: 현재 state를 보고 어느 stage로 갈지 결정
        workflow.set_conditional_entry_point(
            self.determine_stage,
            {
                "stage1": "stage1",
                "stage2": "stage2",
                "stage3": "stage3",
                "end": END
            }
        )

        # 각 Stage는 응답 생성 후 무조건 END (무한 루프 방지!)
        workflow.add_edge("stage1", END)
        workflow.add_edge("stage2", END)
        workflow.add_edge("stage3", END)

        # ✨ checkpointer 연결 - thread_id로 세션 자동 관리
        return workflow.compile(checkpointer=self.memory)

    # ========= Stage 노드 함수들 (핸들러에 위임) =========

    def stage1(self, state: ConversationState) -> ConversationState:
        """Stage 1: 화가 캐릭터 설정 (핸들러에 위임)"""
        return self.stage1_handler.handle(state)

    def stage2(self, state: ConversationState) -> ConversationState:
        """Stage 2: AI 예술 입장 유도 (핸들러에 위임)"""
        return self.stage2_handler.handle(state)

    def stage3(self, state: ConversationState) -> ConversationState:
        """Stage 3: 윤리 주제 탐구 대화 (핸들러에 위임)"""
        return self.stage3_handler.handle(state)

    # ========= 라우팅 함수들 =========

    def determine_stage(self, state: ConversationState) -> str:
        """
        Entry point: 현재 state를 보고 어느 stage로 갈지 결정
        ✨ 각 invoke()마다 한 번만 호출됨

        Returns:
            "stage1", "stage2", "stage3", "end"
        """
        # 🔍 DEBUG: Full state dump for routing decision
        logger.info(f"🔍 [ROUTER] determine_stage ENTRY")
        logger.info(f"🔍 [ROUTER] stage={state.get('stage')}, previous_stage={state.get('previous_stage')}")
        logger.info(f"🔍 [ROUTER] artist_character_set={state.get('artist_character_set')}, stage1_attempts={state.get('stage1_attempts')}")
        logger.info(f"🔍 [ROUTER] stage2_question_asked={state.get('stage2_question_asked')}, stage2_complete={state.get('stage2_complete')}, stage2_completed={state.get('stage2_completed')}")
        logger.info(f"🔍 [ROUTER] current_question_index={state.get('current_question_index')}, should_end={state.get('should_end')}")
        logger.info(f"🔍 [ROUTER] message_count={state.get('message_count')}, messages_len={len(state.get('messages', []))}")

        # 1. 대화 종료 여부 확인
        if state.get("should_end", False):
            logger.info("🔍 Routing to: end")
            return "end"

        # 2. 현재 stage에 따라 다음 stage 결정
        current_stage = state.get("stage", "stage1")

        # Stage1에서 캐릭터 설정 완료 또는 3회 시도 후 → Stage2
        if current_stage == "stage1":
            if state.get("artist_character_set", False) or state.get("stage1_attempts", 0) >= 3:
                # Stage1 완료 → Stage2로 전환 (Stage1Handler에서 이미 Stage2 질문을 했음)
                logger.info(f"✅ [ROUTER] Stage 1 COMPLETE - artist_character_set={state.get('artist_character_set')}, stage1_attempts={state.get('stage1_attempts')} → routing to: stage2")
                return "stage2"
            logger.info(f"🔍 [ROUTER] Stage 1 NOT complete - routing to: stage1")
            return "stage1"

        # Stage2에서 AI 예술 입장 표현 → Stage3
        elif current_stage == "stage2":
            # ✨ stage2_complete 플래그 확인 (Stage2Handler에서 설정)
            if state.get("stage2_complete", False):
                logger.info(f"✅ [ROUTER] Stage 2 COMPLETE - stage2_complete={state.get('stage2_complete')}, stage2_completed={state.get('stage2_completed')} → routing to: stage3")
                return "stage3"
            # 아직 완료 안 됨 → Stage2 유지
            logger.info(f"🔍 [ROUTER] Stage 2 NOT complete - stage2_complete={state.get('stage2_complete')} → routing to: stage2")
            return "stage2"

        # Stage3에서 5개 주제 완료 → end
        elif current_stage == "stage3":
            if state.get("should_end", False):
                logger.info(f"🔍 [ROUTER] Stage 3 COMPLETE - should_end=True → routing to: end")
                return "end"
            logger.info(f"🔍 [ROUTER] Stage 3 NOT complete - current_question_index={state.get('current_question_index')} → routing to: stage3")
            return "stage3"

        logger.info(f"🔍 [ROUTER] Fallback - returning current_stage={current_stage}")
        return current_stage

    # ========= 외부 인터페이스 =========

    def process(self, state_dict: Dict[str, Any], user_input: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        사용자 입력 처리 (LangChain Service에서 호출)

        Args:
            state_dict: 현재 상태 딕셔너리
            user_input: 사용자 메시지
            thread_id: 세션 ID (연속대화 관리용)

        Returns:
            업데이트된 상태 딕셔너리
        """
        # 🔍 DEBUG: Process entry state
        logger.info(f"🔍 [PROCESS] ENTRY - user_input='{user_input[:50]}...', thread_id={thread_id}")
        logger.info(f"🔍 [PROCESS] ENTRY - stage={state_dict.get('stage')}, message_count={state_dict.get('message_count')}, messages_len={len(state_dict.get('messages', []))}")

        # 사용자 메시지 추가
        if "messages" not in state_dict:
            state_dict["messages"] = []
        state_dict["messages"].append({"role": "user", "content": user_input})

        # ✨ LangGraph 실행 with thread_id (연속대화 자동 관리)
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state_dict, config=config)

        # 🔍 DEBUG: Process exit state
        logger.info(f"🔍 [PROCESS] EXIT - stage={result.get('stage')}, message_count={result.get('message_count')}, messages_len={len(result.get('messages', []))}")
        logger.info(f"🔍 [PROCESS] EXIT - stage2_complete={result.get('stage2_complete')}, stage2_completed={result.get('stage2_completed')}")
        logger.info(f"🔍 [PROCESS] EXIT - response='{result.get('last_response', '')[:80]}...'")

        return result

    def get_initial_message(self) -> str:
        """게임 시작 메시지"""
        return "선생님, 요새 잘 지내세요? 지난번 진로 조언 해주신 덕에 저도 열심히 연습 중입니다! 요즘 작품활동은 좀 어떠세요?"
