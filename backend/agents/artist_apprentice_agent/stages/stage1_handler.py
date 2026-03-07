"""Stage 1: Artist character setup - player describes their art style and direction."""
from typing import Dict, Any
import logging
import random
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .stage2_handler import Stage2Handler

logger = logging.getLogger(__name__)


class Stage1Handler:
    # Must match conversation_agent.get_initial_message()
    FIXED_GREETING = "선생님, 요새 잘 지내세요? 지난번 진로 조언 해주신 덕에 저도 열심히 연습 중입니다! 요즘 작품활동은 좀 어떠세요?"
    CASUAL_GREETING_RESPONSES = [
        "잘 지내신다니 다행이에요. 요즘에는 어떤 작업을 붙잡고 계신지 궁금해요.",
        "그렇게 지내신다니 반갑네요! 요즘 손대고 있는 작품이 있다면 살짝 들려주실 수 있을까요?",
    ]

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = "", analyzer: ChatOpenAI = None):
        self.llm = llm
        self.prompts = prompts
        self.persona_prompt = persona_prompt
        self.analyzer = analyzer if analyzer else llm

    def _check_artist_character(self, messages: list) -> bool:
        """Use LLM to judge whether the user described their artist identity."""
        recent_messages = messages[-6:] if len(messages) > 6 else messages

        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages
        ])

        character_check_template = self.prompts.get(
            "stage1_character_check",
            "사용자가 화가/예술가 정체성을 설명했는지 판단하세요. yes 또는 no만 출력하세요."
        )

        character_check_prompt = character_check_template.replace("{user_message}", conversation_history)

        try:
            result = self.analyzer.invoke(character_check_prompt)
            decision = result.content.strip().lower()
            verdict = "yes" in decision
            logger.info(f"[Stage1] character_check verdict={verdict}, raw='{decision}'")
            return verdict
        except Exception:
            return False

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        user_message = messages[-1]["content"] if messages and messages[-1].get("role") == "user" else ""

        logger.info(f"[Stage1Handler.handle] ENTRY, messages_count={len(messages)}, user_message='{user_message}'")

        if not messages:
            response = self.FIXED_GREETING
            artist_character_set = False
        else:
            last_message = messages[-1]
            if last_message.get("role") != "user":
                response = self.FIXED_GREETING
                artist_character_set = False
            else:
                if self._is_casual_greeting(last_message.get("content", "")):
                    response = self._get_casual_greeting_response()
                    artist_character_set = False
                    logger.info("[Stage1] Casual greeting detected")
                else:
                    has_artist_content = self._check_artist_character(messages)
                    current_attempts = state.get("stage1_attempts", 0)

                    # Transition to stage2 if artist content found or on 2nd+ attempt
                    if has_artist_content or current_attempts >= 1:
                        logger.info(f"[Stage1] has_artist_content={has_artist_content}, attempts={current_attempts} -> Stage2")
                        response = self._generate_acknowledgment_and_transition(messages)
                        artist_character_set = True
                    else:
                        logger.info(f"[Stage1] No artist content, attempts={current_attempts} -> rephrase")
                        response = self._generate_rephrase_question(messages)
                        artist_character_set = False

        if artist_character_set:
            state["stage"] = "stage2"
            state["previous_stage"] = "stage1"
            state["stage2_question_asked"] = True
            state["stage2_complete"] = False
            state["stage2_completed"] = False
            logger.info(f"[Stage1] TRANSITION -> stage2")
        else:
            state["stage"] = "stage1"
            state["previous_stage"] = "stage1"
            logger.info(f"[Stage1] STAY in stage1")

        state["last_response"] = response
        state["artist_character_set"] = artist_character_set
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        if not artist_character_set:
            state["stage1_attempts"] = state.get("stage1_attempts", 0) + 1
        else:
            state["stage1_attempts"] = 0

        logger.info(f"[Stage1] EXIT -> {state.get('stage')}")
        return state

    def _generate_acknowledgment_and_transition(self, messages: list) -> str:
        return Stage2Handler.generate_acknowledgment_and_transition(self.llm, messages, self.prompts)

    @staticmethod
    def _is_casual_greeting(user_message: str) -> bool:
        if not user_message:
            return False
        normalized = user_message.strip().lower().replace(" ", "")
        greeting_set = {"웅잘지내", "응잘지내", "잘지내", "괜찮아", "그냥그래"}
        return normalized in greeting_set

    def _get_casual_greeting_response(self) -> str:
        return random.choice(self.CASUAL_GREETING_RESPONSES)

    def _generate_rephrase_question(self, messages: list) -> str:
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"] if user_messages else ""

        recent_messages = messages[-4:] if len(messages) > 4 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        rephrase_template = self.prompts.get(
            "stage1_rephrase",
            "사용자가 말한 '{user_message}'를 인정하면서 작품활동에 대해 질문하세요."
        )

        # Use .replace() to preserve literal braces in the template
        rephrase_prompt = rephrase_template.replace("{user_message}", user_message).replace("{context}", context)

        try:
            result = self.llm.invoke(rephrase_prompt)
            generated_response = result.content.strip().strip('"')

            if not generated_response or generated_response in ['...', '..', '.']:
                logger.warning(f"[Stage1] LLM returned empty/ellipsis, using fallback")
                return "선생님은 주로 어떤 작품을 그리시는 편이에요?"

            logger.info(f"[Stage1] LLM generated question: '{generated_response[:50]}...'")
            return generated_response
        except Exception as e:
            logger.error(f"[Stage1] LLM error: {e}, using fallback")
            return "선생님은 주로 어떤 작품을 그리시는 편이에요?"
