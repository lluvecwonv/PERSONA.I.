"""Stage 1: Player character establishment - elicit details about player's daily life."""
from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI

from .stage2_handler import Stage2Handler

logger = logging.getLogger(__name__)


class Stage1Handler:
    # Must match conversation_agent.get_initial_message()
    FIXED_GREETING = "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"
    # One-time follow-up (appended after empathy line)
    FOLLOW_UP_QUESTION = "요즘 밥은 먹고 있어?"

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = "", analyzer: ChatOpenAI = None):
        self.llm = llm
        self.prompts = prompts
        self.persona_prompt = persona_prompt
        self.analyzer = analyzer if analyzer else llm

    def _check_artist_character(self, messages: list) -> bool:
        """Use LLM to check if user gave concrete details about their daily life."""
        recent_messages = messages[-6:] if len(messages) > 6 else messages

        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages
        ])

        character_check_template = self.prompts.get(
            "stage1_character_check",
            "사용자가 자신의 캐릭터나 요즘 생활에 대해 구체적으로 설명했는지 판단하세요. yes 또는 no만 출력하세요."
        )

        character_check_prompt = character_check_template.replace("{user_message}", conversation_history)

        try:
            logger.info(f"[Stage1 Character Check] Conversation history: '{conversation_history}'")
            logger.info(f"[Stage1 Character Check] Prompt length: {len(character_check_prompt)} chars")

            result = self.analyzer.invoke(character_check_prompt)
            decision = result.content.strip().lower()

            logger.info(f"[Stage1 Character Check] LLM raw response: '{decision}'")

            is_character_set = "yes" in decision
            logger.info(f"[Stage1 Character Check] Result: {'YES' if is_character_set else 'NO'}")

            return is_character_set
        except Exception as e:
            logger.error(f"Character check error: {e}")
            return False

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])

        greeting_sent = state.get("stage1_greeting_sent", False)
        if not greeting_sent:
            assistant_msgs = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
            greeting_sent = any(self.FIXED_GREETING in msg for msg in assistant_msgs)
            if greeting_sent:
                state["stage1_greeting_sent"] = True
        follow_up_asked = state.get("stage1_follow_up_asked", False)

        # Recover flags from history in case session flags were reset
        assistant_msgs = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        if not greeting_sent:
            greeting_sent = any(self.FIXED_GREETING in msg for msg in assistant_msgs)
            if greeting_sent:
                state["stage1_greeting_sent"] = True
        if not follow_up_asked:
            follow_up_asked = any("밥" in msg and "먹고" in msg for msg in assistant_msgs)
            if follow_up_asked:
                state["stage1_follow_up_asked"] = True
                logger.info("[Stage1] Detected follow-up question in history")

        should_increment_attempts = False

        if not greeting_sent:
            response = self.FIXED_GREETING
            artist_character_set = False
            state["stage1_greeting_sent"] = True
            logger.info("[Stage1] Sending initial greeting")
        elif not messages:
            response = self.FIXED_GREETING
            artist_character_set = False
        else:
            last_message = messages[-1]
            if last_message.get("role") != "user":
                logger.warning(f"[Stage1] Last message is not user: {last_message.get('role')}")
                assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                previous_questions = [msg.get("content", "") for msg in assistant_messages]
                if not any(self.FIXED_GREETING in q for q in previous_questions):
                    response = self.FIXED_GREETING
                else:
                    response = "그렇구나. 요즘 어떻게 지내고 있어?"
                artist_character_set = False
            else:
                user_message = last_message["content"]
                logger.info(f"[Stage1] User message: '{user_message}'")

                has_artist_content = self._check_artist_character(messages)
                logger.info(f"[Stage1] Character check result: {has_artist_content}")

                if follow_up_asked:
                    logger.info("[Stage1] Follow-up answer detected -> transitioning to Stage 2")
                    response = self._generate_acknowledgment_and_transition(messages)
                    artist_character_set = True
                else:
                    current_attempt = state.get("stage1_attempts", 0)

                    if has_artist_content or current_attempt >= 1:
                        logger.info(f"[Stage1] Transitioning to Stage 2 (has_content={has_artist_content}, attempt={current_attempt})")
                        response = self._generate_acknowledgment_and_transition(messages)
                        artist_character_set = True
                    else:
                        logger.info("[Stage1] User did not provide character details - asking follow-up")
                        response = self._generate_rephrase_question(messages)
                        artist_character_set = False
                        state["stage1_follow_up_asked"] = True
                        should_increment_attempts = True

        if artist_character_set:
            state["stage"] = "stage2"
            state["previous_stage"] = "stage1"
            state["stage2_question_asked"] = True
            state["stage1_follow_up_asked"] = False
            logger.info("[Stage1] Transitioning to Stage 2")
        else:
            state["stage"] = "stage1"
            state["previous_stage"] = "stage1"
            logger.info("[Stage1] Staying in Stage 1")


        state["last_response"] = response
        state["artist_character_set"] = artist_character_set
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        if not artist_character_set:
            if should_increment_attempts:
                state["stage1_attempts"] = state.get("stage1_attempts", 0) + 1
        else:
            state["stage1_attempts"] = 0

        logger.info(f"[Stage1] FINAL STATE - stage={state['stage']}, response: {response[:100]}...")

        return state

    def _generate_acknowledgment_and_transition(self, messages: list) -> str:
        """Generate short empathy + Stage 2 fixed question (delegated to Stage2Handler)."""
        return Stage2Handler.generate_acknowledgment_and_transition(self.llm, messages, self.prompts)

    def _generate_rephrase_question(self, messages: list) -> str:
        """Generate empathy line + fixed follow-up question when user didn't elaborate."""
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"].strip() if user_messages else ""

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
            logger.error(f"[Stage1] Empathy generation error: {e}")
            empathy_line = "그랬구나."

        logger.info(f"[Stage1] Empathy line: '{empathy_line}'")
        return f"{empathy_line} {self.FOLLOW_UP_QUESTION}"
