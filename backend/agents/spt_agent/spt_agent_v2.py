"""
SPT Agent V2
새로운 4계층 아키텍처 기반 SPT 에이전트

아키텍처:
User → [① DST] → [② SPT Planner] → [③ SPT Controller] → [④ Persona Agent] → User

핵심 변화:
- SPT는 "질문 생성"이 아니라 "질문 후보 제안"만 함
- 질문을 실제로 던질지 말지는 Controller가 결정
- DST가 "지금 질문해도 되는가?"를 판단
"""
from typing import List, Dict, Any, Optional
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .dst import DialogueStateTracker, DialogueState, UserResponseStatus, SPTFrame
from .spt_controller import SPTController, ControllerDecision, ResponseStrategy
from .spt_planner import SPTPlanner, QuestionCandidate
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


class SPTAgentV2:
    """
    SPT Agent V2: 4계층 아키텍처

    흐름:
    1. DST: 사용자 응답 분석 → 질문 허용 여부 판단
    2. Planner: 질문 후보 생성 (허용된 경우에만)
    3. Controller: 최종 응답 전략 결정
    4. Output: Persona Agent에게 전달할 지시사항 생성

    이 에이전트는 직접 응답을 생성하지 않음.
    Persona Agent에게 "무엇을 해야 하는지" 지시만 함.
    """

    def __init__(self, api_key: str, planner_model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API key
            planner_model: SPT Planner에서 사용할 모델
        """
        self.api_key = api_key

        # 3개 모듈 초기화
        self.dst = DialogueStateTracker()
        self.planner = SPTPlanner(api_key=api_key, model=planner_model)
        self.controller = SPTController(dst=self.dst)

        logger.info("✅ SPT Agent V2 initialized with DST + Planner + Controller")

    async def process(
        self,
        session_id: str,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        topic_context: str = "",
        question_keywords: List[str] = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지 처리 및 Persona Agent 지시사항 생성

        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            conversation_history: 전체 대화 기록
            topic_context: 주제 컨텍스트 (예: "AI 예술 전시")
            question_keywords: 현재 질문의 핵심 키워드

        Returns:
            {
                "instruction": str,  # Persona Agent에게 전달할 지시사항
                "strategy": str,     # 응답 전략
                "allow_question": bool,  # 새 질문 허용 여부
                "suggested_question": str | None,  # 제안된 질문
                "state": DialogueState,  # 현재 대화 상태
            }
        """
        logger.info(f"🧠 [SPT V2] Processing message for session {session_id}")

        # 대화 기록에서 마지막 assistant 메시지 추출 (이전 질문)
        last_agent_question = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                last_agent_question = msg.get("content", "")
                break

        # Step 1: DST - 사용자 응답 분석
        state = self.dst.analyze_user_response(
            session_id=session_id,
            user_message=user_message,
            question_keywords=question_keywords,
            last_agent_question=last_agent_question
        )
        logger.info(f"🔍 [SPT V2] DST analysis: status={state.response_status}, answered={state.user_answered}")

        # Step 2: 질문 허용 여부 확인
        can_ask, reason = self.dst.can_ask_new_question(session_id)
        logger.info(f"🔍 [SPT V2] Can ask new question: {can_ask}, reason={reason}")

        # Step 3: Planner - 질문 후보 생성 (허용된 경우에만)
        question_candidates = []
        if can_ask:
            question_candidates = await self.planner.generate_candidates(
                state=state,
                user_message=user_message,
                conversation_history=conversation_history,
                topic_context=topic_context
            )
            logger.info(f"📝 [SPT V2] Planner generated {len(question_candidates)} candidates")

        # Step 4: Controller - 최종 결정
        decision = self.controller.decide(
            session_id=session_id,
            user_message=user_message,
            question_candidates=[c.question for c in question_candidates],
            question_keywords=question_keywords
        )
        logger.info(f"🎯 [SPT V2] Controller decision: {decision.strategy.value}")

        # Step 5: Persona Agent 지시사항 생성
        instruction = self.controller.get_instruction_for_persona(decision, state)

        # 질문 후보의 키워드도 함께 전달 (다음 턴의 off_topic 감지용)
        suggested_keywords = None
        if question_candidates and decision.allow_question:
            suggested_keywords = question_candidates[0].keywords

        return {
            "instruction": instruction,
            "strategy": decision.strategy.value,
            "allow_question": decision.allow_question,
            "suggested_question": decision.suggested_content,
            "suggested_keywords": suggested_keywords,
            "state": state,
            "response_status": state.response_status.value if state.response_status else None,
        }

    def update_after_agent_response(
        self,
        session_id: str,
        agent_response: str,
        asked_question: bool = False
    ) -> None:
        """
        에이전트 응답 후 상태 업데이트

        Args:
            session_id: 세션 ID
            agent_response: 에이전트가 생성한 응답
            asked_question: 응답에 질문이 포함되었는지 여부
        """
        self.dst.update_after_agent_turn(
            session_id=session_id,
            agent_response=agent_response,
            asked_question=asked_question
        )
        logger.info(f"📝 [SPT V2] Updated state after agent response (asked_question={asked_question})")

    def clear_session(self, session_id: str) -> None:
        """세션 초기화"""
        self.dst.clear_state(session_id)
        logger.info(f"🗑️ [SPT V2] Cleared session: {session_id}")

    def get_state(self, session_id: str) -> DialogueState:
        """현재 세션 상태 조회"""
        return self.dst.get_or_create_state(session_id)

    def advance_frame(self, session_id: str, new_frame: SPTFrame) -> None:
        """프레임 전환"""
        self.dst.advance_frame(session_id, new_frame)


# ============================================================================
# 기존 SPTAgent와의 호환을 위한 래퍼
# ============================================================================

class SPTAgentV2Wrapper:
    """
    기존 langchain_service.py와 호환되는 래퍼

    기존 인터페이스:
    - chat(messages, temperature, max_tokens, session_id) -> str

    새 인터페이스:
    - process(session_id, user_message, history, ...) -> Dict
    """

    def __init__(self, api_key: str, planner_model: str = "gpt-4o-mini"):
        self.spt_v2 = SPTAgentV2(api_key=api_key, planner_model=planner_model)
        self.api_key = api_key

        # 응답 생성용 LLM (Persona Agent가 없을 때 직접 사용)
        self.response_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7,
            max_tokens=300
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        session_id: str = "default",
        topic_context: str = "",
        question_keywords: List[str] = None
    ) -> str:
        """
        기존 chat 인터페이스 호환

        Returns:
            응답 텍스트 (Persona Agent가 없을 때는 직접 생성)
        """
        # 마지막 사용자 메시지 추출
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return "무엇이 궁금하세요?"

        # SPT V2 처리
        result = await self.spt_v2.process(
            session_id=session_id,
            user_message=user_message,
            conversation_history=messages,
            topic_context=topic_context,
            question_keywords=question_keywords
        )

        # 지시사항 기반으로 응답 생성 (Persona Agent 역할 대행)
        instruction = result["instruction"]

        system_prompt = f"""
당신은 윤리 상담 대화 에이전트입니다.
다음 지시사항에 따라 응답하세요:

{instruction}

규칙:
- 한국어 존댓말로 자연스럽게 답하세요
- 2~3문장으로 간결하게 답하세요
- 사용자의 말에 먼저 공감하세요
"""

        try:
            response = await self.response_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"사용자: {user_message}")
            ])

            response_text = clean_gpt_response(response.content)

            # 질문이 포함되었는지 체크
            has_question = "?" in response_text or "요?" in response_text

            # 상태 업데이트
            self.spt_v2.update_after_agent_response(
                session_id=session_id,
                agent_response=response_text,
                asked_question=has_question
            )

            return response_text

        except Exception as e:
            logger.error(f"❌ [SPT V2 Wrapper] Error generating response: {e}")
            return "죄송해요, 다시 한번 말씀해주시겠어요?"

    def get_initial_message(self) -> str:
        """첫 메시지 반환"""
        return "안녕하세요! 어떻게 도와드릴까요?"

    def clear_session(self, session_id: str) -> bool:
        """세션 초기화"""
        self.spt_v2.clear_session(session_id)
        return True
