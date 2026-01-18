"""
Stage 1: 플레이어 캐릭터 구체화 단계
플레이어가 자신의 캐릭터(요즘 어떻게 살고 있는지)를 구체화
"""
from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI

# ✨ Stage 2 전환용 메서드 import
from .stage2_handler import Stage2Handler

logger = logging.getLogger(__name__)


class Stage1Handler:
    """Stage 1: 플레이어 캐릭터 구체화 핸들러"""

    # 고정 시작 질문 (conversation_agent.get_initial_message()와 동일해야 함!)
    FIXED_GREETING = "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"
    # 후속 질문 (한 번만 사용) - empathy_line 뒤에 붙으므로 "그렇구나" 제외
    FOLLOW_UP_QUESTION = "요즘 밥은 먹고 있어?"


    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = "", analyzer: ChatOpenAI = None):
        self.llm = llm
        self.prompts = prompts
        self.persona_prompt = persona_prompt
        # 의도 분석용 LLM (temperature=0)
        self.analyzer = analyzer if analyzer else llm

    def _check_artist_character(self, messages: list) -> bool:
        """
        LLM을 사용하여 플레이어가 자신의 캐릭터/요즘 생활에 대해 구체적으로 설명했는지 판단
        ✨ 전체 대화 컨텍스트를 보고 판단 (이전 대화 + 현재 메시지)

        Args:
            messages: 전체 대화 기록 (최근 대화 포함)

        Returns:
            플레이어 캐릭터/요즘 생활에 대한 구체적 설명 포함 여부
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
            "사용자가 자신의 캐릭터나 요즘 생활에 대해 구체적으로 설명했는지 판단하세요. yes 또는 no만 출력하세요."
        )

        # {conversation_history} 치환 (user_message 대신)
        character_check_prompt = character_check_template.replace("{user_message}", conversation_history)

        try:
            logger.info(f"🔍 [Stage1 Character Check] Conversation history: '{conversation_history}'")
            logger.info(f"🔍 [Stage1 Character Check] Prompt length: {len(character_check_prompt)} chars")

            result = self.analyzer.invoke(character_check_prompt)
            decision = result.content.strip().lower()

            logger.info(f"🔍 [Stage1 Character Check] LLM raw response: '{decision}'")

            is_character_set = "yes" in decision
            if is_character_set:
                logger.info(f"🔍 [Stage1 Character Check] ✅ RESULT: YES (will move to Stage 2)")
            else:
                logger.info(f"🔍 [Stage1 Character Check] ❌ RESULT: NO (will ask follow-up question)")

            return is_character_set
        except Exception as e:
            # LLM 호출 실패 시 보수적으로 False 반환
            logger.error(f"❌ Character check error: {e}")
            return False

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1 처리: 플레이어 캐릭터 구체화 유도

        Args:
            state: 현재 대화 상태

        Returns:
            업데이트된 대화 상태
        """
        messages = state.get("messages", [])

        # 세션 플래그 추출
        greeting_sent = state.get("stage1_greeting_sent", False)
        if not greeting_sent:
            assistant_msgs = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
            greeting_sent = any(self.FIXED_GREETING in msg for msg in assistant_msgs)
            if greeting_sent:
                state["stage1_greeting_sent"] = True
        follow_up_asked = state.get("stage1_follow_up_asked", False)

        # ✨ 대화 히스토리에서 인사/후속질문 여부 확인 (세션 플래그가 리셋되어도 중복 방지)
        assistant_msgs = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        if not greeting_sent:
            # 대화 히스토리에 이미 인사가 있으면 greeting_sent=True로 설정
            greeting_sent = any(self.FIXED_GREETING in msg for msg in assistant_msgs)
            if greeting_sent:
                state["stage1_greeting_sent"] = True
        if not follow_up_asked:
            follow_up_asked = any("밥" in msg and "먹고" in msg for msg in assistant_msgs)
            if follow_up_asked:
                state["stage1_follow_up_asked"] = True
                logger.info("🔍 [Stage1] Detected follow-up question in history - setting stage1_follow_up_asked=True")

        should_increment_attempts = False

        if not greeting_sent:
            # 🎯 첫 진입: 반드시 고정 인사 출력
            response = self.FIXED_GREETING
            artist_character_set = False
            state["stage1_greeting_sent"] = True
            logger.info("🔔 [Stage1] Sending initial greeting")
        elif not messages:
            # 이례 상황: 인사 후에도 메시지가 없다면 다시 고정 질문
            response = self.FIXED_GREETING
            artist_character_set = False
        else:
            # ✅ 마지막 메시지가 user 메시지인지 확인
            last_message = messages[-1]
            if last_message.get("role") != "user":
                logger.warning(f"⚠️ [Stage1] Last message is not user: {last_message.get('role')}")
                # 고정 질문이 아직 나오지 않았다면 먼저 고정 질문
                assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                previous_questions = [msg.get("content", "") for msg in assistant_messages]
                if not any(self.FIXED_GREETING in q for q in previous_questions):
                    response = self.FIXED_GREETING
                else:
                    response = "그렇구나. 요즘 어떻게 지내고 있어?"
                artist_character_set = False
            else:
                # 사용자 답변 확인
                user_message = last_message["content"]
                logger.info(f"🔍 [Stage1] User message: '{user_message}'")

                # LLM 기반: 전체 대화 컨텍스트를 보고 캐릭터/생활 설명 여부 확인
                has_artist_content = self._check_artist_character(messages)
                logger.info(f"🔍 [Stage1] Character check result: {has_artist_content}")

                if follow_up_asked:
                    logger.info("✅ [Stage1] Follow-up answer detected → transitioning to Stage 2")
                    response = self._generate_acknowledgment_and_transition(messages)
                    artist_character_set = True
                else:
                    current_attempt = state.get("stage1_attempts", 0)

                    if has_artist_content or current_attempt >= 1:
                        # ✅ 캐릭터 설명했거나, 이미 후속 질문(밥 먹고 있어?)을 한 번 했으면 → Stage 2로
                        logger.info(f"✅ [Stage1] Transitioning to Stage 2 (has_content={has_artist_content}, attempt={current_attempt})")
                        response = self._generate_acknowledgment_and_transition(messages)
                        artist_character_set = True
                    else:
                        # ❌ 첫 번째 무관한 답변 → 후속 질문 (한 번만)
                        logger.info(f"⚠️ [Stage1] User did not provide character details - asking follow-up question")
                        response = self._generate_rephrase_question(messages)
                        artist_character_set = False
                        state["stage1_follow_up_asked"] = True
                        should_increment_attempts = True

        # ✨ Stage 2로 전환 시 stage를 "stage2"로 설정
        if artist_character_set:
            state["stage"] = "stage2"
            state["previous_stage"] = "stage1"  # ✨ Stage 1에서 왔다는 것을 표시
            state["stage2_question_asked"] = True  # ✨ 이미 질문했다는 플래그
            state["stage1_follow_up_asked"] = False
            logger.info(f"🔍 [Stage1] ✅ Transitioning to Stage 2")
        else:
            state["stage"] = "stage1"
            state["previous_stage"] = "stage1"
            logger.info(f"🔍 [Stage1] ⚠️ Staying in Stage 1")


        state["last_response"] = response
        state["artist_character_set"] = artist_character_set
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        # 카운터 증가 (무한 루프 방지용)
        if not artist_character_set:
            if should_increment_attempts:
                state["stage1_attempts"] = state.get("stage1_attempts", 0) + 1
        else:
            # 캐릭터 설정 완료 시 카운터 리셋
            state["stage1_attempts"] = 0

        logger.info(f"🔍 [Stage1] FINAL STATE - stage={state['stage']}, response: {response[:100]}...")

        return state

    def _generate_acknowledgment_and_transition(self, messages: list) -> str:
        """
        캐릭터 설명에 대한 짧은 공감 + Stage 2 고정 질문
        ✨ Stage2Handler의 static 메서드 호출 (책임 분리)

        Args:
            messages: 전체 대화 기록 (컨텍스트 전달)

        Returns:
            짧은 공감 + Stage 2의 고정 질문
        """
        # ✨ Stage 2의 로직은 Stage2Handler가 담당 (전체 대화 컨텍스트 전달)
        return Stage2Handler.generate_acknowledgment_and_transition(self.llm, messages, self.prompts)

    def _generate_rephrase_question(self, messages: list) -> str:
        """
        Stage 1 추가 질문 (고정 문구)
        플레이어가 캐릭터를 구체화하지 않았을 때 스크립트에 맞춰 재질문
        """
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"].strip() if user_messages else ""

        # 최근 대화 컨텍스트 (최대 4턴)
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        empathy_template = self.prompts.get(
            "stage1_rephrase",
            "최근 대화:\n{context}\n사용자 마지막 발화: {user_message}\n"
            "사용자의 말을 자연스럽게 인정하고 걱정하는 반말 한 문장만 출력하세요."
        )
        empathy_prompt = empathy_template.replace("{context}", context).replace("{user_message}", user_message)

        try:
            result = self.llm.invoke(empathy_prompt)
            empathy_line = result.content.strip().strip('"')
            if not empathy_line:
                empathy_line = "그랬구나."
        except Exception as e:
            logger.error(f"❌ [Stage1] Empathy generation error: {e}")
            empathy_line = "그랬구나."

        logger.info(f"⚠️ [Stage1] Empathy line: '{empathy_line}'")
        return f"{empathy_line} {self.FOLLOW_UP_QUESTION}"
