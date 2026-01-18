"""
Stage 1: 화가 캐릭터 설정 단계
플레이어가 자신의 화가 캐릭터(작품 스타일, 작업 방향 등)를 구체화
"""
from typing import Dict, Any
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ✨ Stage 2 전환용 메서드 import
from .stage2_handler import Stage2Handler

logger = logging.getLogger(__name__)


class Stage1Handler:
    """Stage 1: 화가 캐릭터 설정 핸들러"""

    # 고정 시작 질문 (conversation_agent.get_initial_message()와 동일해야 함!)
    FIXED_GREETING = "선생님, 요새 잘 지내세요? 지난번 진로 조언 해주신 덕에 저도 열심히 연습 중입니다! 요즘 작품활동은 좀 어떠세요?"
    CASUAL_GREETING_RESPONSES = [
        "잘 지내신다니 다행이에요. 요즘에는 어떤 작업을 붙잡고 계신지 궁금해요.",
        "그렇게 지내신다니 반갑네요! 요즘 손대고 있는 작품이 있다면 살짝 들려주실 수 있을까요?",
    ]

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = "", analyzer: ChatOpenAI = None):
        self.llm = llm
        self.prompts = prompts
        self.persona_prompt = persona_prompt
        # 의도 분석용 LLM (temperature=0)
        self.analyzer = analyzer if analyzer else llm

    def _check_artist_character(self, messages: list) -> bool:
        """
        LLM을 사용하여 화가 캐릭터 설명 여부를 판단
        ✨ 전체 대화 컨텍스트를 보고 판단 (이전 대화 + 현재 메시지)

        Args:
            messages: 전체 대화 기록 (최근 대화 포함)

        Returns:
            화가 캐릭터 설명 포함 여부
        """
        # 최근 3턴의 대화만 가져오기 (너무 길면 토큰 낭비)
        recent_messages = messages[-6:] if len(messages) > 6 else messages

        # 대화 히스토리 포맷팅
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages
        ])

        # 프롬프트 파일에서 로드
        character_check_template = self.prompts.get(
            "stage1_character_check",
            "사용자가 화가/예술가 정체성을 설명했는지 판단하세요. yes 또는 no만 출력하세요."
        )

        # {conversation_history} 치환 (user_message 대신)
        character_check_prompt = character_check_template.replace("{user_message}", conversation_history)

        try:
            result = self.analyzer.invoke(character_check_prompt)
            decision = result.content.strip().lower()
            verdict = "yes" in decision
            logger.info(f"🔍 [Stage1] character_check verdict={verdict}, raw='{decision}'")
            return verdict
        except Exception:
            # LLM 호출 실패 시 보수적으로 False 반환
            return False

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1 처리: 화가 캐릭터 구체화 유도

        Args:
            state: 현재 대화 상태

        Returns:
            업데이트된 대화 상태
        """
        messages = state.get("messages", [])
        user_message = messages[-1]["content"] if messages and messages[-1].get("role") == "user" else ""

        logger.info(f"=" * 50)
        logger.info(f"🔍 [Stage1Handler.handle] ENTRY")
        logger.info(f"🔍 [Stage1] messages_count={len(messages)}")
        logger.info(f"🔍 [Stage1] user_message='{user_message}'")

        if not messages:
            # 🎯 게임 시작: 고정 인사 메시지
            response = self.FIXED_GREETING
            artist_character_set = False
        else:
            # ✅ 마지막 메시지가 user 메시지인지 확인
            last_message = messages[-1]
            if last_message.get("role") != "user":
                # 마지막 메시지가 user가 아니면 첫 메시지 처리
                response = self.FIXED_GREETING
                artist_character_set = False
            else:
                # 😊 인사/근황만 있는 경우는 먼저 자연스럽게 흡수
                if self._is_casual_greeting(last_message.get("content", "")):
                    response = self._get_casual_greeting_response()
                    artist_character_set = False
                    logger.info("😊 [Stage1] Casual greeting detected → lightweight acknowledgement before re-asking")
                else:
                    # ✅ LLM 기반: 전체 대화 컨텍스트를 보고 작품활동 설명 여부 확인
                    has_artist_content = self._check_artist_character(messages)
                    current_attempts = state.get("stage1_attempts", 0)

                    if has_artist_content or current_attempts >= 1:
                        # ✅ 작품활동 설명했거나 2번째 시도 → Stage 2로 전환
                        if has_artist_content:
                            logger.info(f"🔍 [Stage1] has_artist_content=True → generating Stage2 question")
                        else:
                            logger.info(f"🔍 [Stage1] stage1_attempts={current_attempts} >= 1 → forcing Stage2 transition")
                        response = self._generate_acknowledgment_and_transition(messages)
                        artist_character_set = True
                    else:
                        # ❌ 첫 번째 시도에서 무관한 답변 → 한 번만 재질문
                        logger.info(f"🔍 [Stage1] has_artist_content=False, attempts={current_attempts} → rephrase question")
                        response = self._generate_rephrase_question(messages)
                        artist_character_set = False

        # ✨ Stage 2로 전환 시 stage를 "stage2"로 설정
        if artist_character_set:
            state["stage"] = "stage2"
            state["previous_stage"] = "stage1"
            state["stage2_question_asked"] = True  # Stage1에서 이미 Stage2 질문 포함
            state["stage2_complete"] = False
            state["stage2_completed"] = False
            logger.info(f"🔍 [Stage1] TRANSITION → stage2")
            logger.info(f"🔍 [Stage1] stage2_question_asked=True")
            logger.info(f"🔍 [Stage1] response='{response}'")
        else:
            state["stage"] = "stage1"
            state["previous_stage"] = "stage1"
            logger.info(f"🔍 [Stage1] STAY in stage1")
            logger.info(f"🔍 [Stage1] response='{response}'")


        state["last_response"] = response
        state["artist_character_set"] = artist_character_set
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        # 카운터 증가 (무한 루프 방지용)
        if not artist_character_set:
            state["stage1_attempts"] = state.get("stage1_attempts", 0) + 1
        else:
            state["stage1_attempts"] = 0

        logger.info(f"🔍 [Stage1] EXIT → {state.get('stage')}")
        logger.info(f"=" * 50)
        return state

    def _generate_acknowledgment_and_transition(self, messages: list) -> str:
        """
        작품 설명에 대한 짧은 공감 + Stage 2 고정 질문
        ✨ Stage2Handler의 static 메서드 호출 (책임 분리)

        Args:
            messages: 전체 대화 기록 (컨텍스트 전달)

        Returns:
            짧은 공감 + Stage 2의 고정 질문
        """
        # ✨ Stage 2의 로직은 Stage2Handler가 담당 (전체 대화 컨텍스트 전달)
        return Stage2Handler.generate_acknowledgment_and_transition(self.llm, messages, self.prompts)

    @staticmethod
    def _is_casual_greeting(user_message: str) -> bool:
        if not user_message:
            return False
        normalized = user_message.strip().lower().replace(" ", "")
        greeting_set = {"웅잘지내", "응잘지내", "잘지내", "괜찮아", "그냥그래"}
        return normalized in greeting_set

    def _get_casual_greeting_response(self) -> str:
        import random
        return random.choice(self.CASUAL_GREETING_RESPONSES)

    def _generate_rephrase_question(self, messages: list) -> str:
        """
        작품활동 관련 재질문 (LLM 기반 자연스러운 대화)

        Args:
            messages: 전체 대화 기록

        Returns:
            자연스러운 재질문 문구
        """
        # 마지막 user 메시지 찾기
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"] if user_messages else ""

        # 최근 대화 컨텍스트 (최대 4턴)
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        # 프롬프트 파일에서 로드
        rephrase_template = self.prompts.get(
            "stage1_rephrase",
            "사용자가 말한 '{user_message}'를 인정하면서 작품활동에 대해 질문하세요."
        )

        # {user_message}와 {context} 치환 - .replace() 사용 (중괄호 리터럴 보존)
        rephrase_prompt = rephrase_template.replace("{user_message}", user_message).replace("{context}", context)

        try:
            # PersonaLLM이 페르소나를 자동 주입하므로 rephrase_prompt만 전달
            result = self.llm.invoke(rephrase_prompt)
            generated_response = result.content.strip().strip('"')

            # 빈 응답이거나 '...'만 있는 경우 fallback
            if not generated_response or generated_response in ['...', '..', '.']:
                logger.warning(f"⚠️ [Stage1] LLM returned empty/ellipsis, using fallback")
                return "선생님은 주로 어떤 작품을 그리시는 편이에요?"

            logger.info(f"✅ [Stage1] LLM generated question: '{generated_response[:50]}...'")
            return generated_response
        except Exception as e:
            # LLM 실패 시 fallback
            logger.error(f"❌ [Stage1] LLM error: {e}, using fallback")
            return "선생님은 주로 어떤 작품을 그리시는 편이에요?"
