"""Stage 2: Ask the user's stance on AI art."""
from typing import Dict, Any
import logging
from pathlib import Path
import sys
from langchain_openai import ChatOpenAI

from .stage2 import (
    AcknowledgmentGenerator,
    FirstQuestionGenerator,
    RephraseQuestionGenerator,
    IntentDetector,
    ExplanationGenerator,
    TransitionGenerator,
)

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import format_prompt


class Stage2Handler:
    STAGE2_QUESTION = "그런데.. 기사 보셨어요? 이번에 우리나라에서 제일 큰 기업이 '그림 그리는 AI'를 만들었다는데, 이 AI의 그림이 국립현대예술관에 전시가 된대요. 어떻게 생각하시는지 여쭤보려고 왔어요."

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = "", analyzer: ChatOpenAI = None):
        self.llm = llm
        self.prompts = prompts
        self.persona_prompt = persona_prompt
        self.analyzer = analyzer if analyzer else llm

        self.first_question_generator = FirstQuestionGenerator(llm, prompts)
        self.rephrase_question_generator = RephraseQuestionGenerator(llm, prompts)
        self.intent_detector = IntentDetector(self.analyzer, prompts)
        self.explanation_generator = ExplanationGenerator(llm, prompts)
        self.transition_generator = TransitionGenerator(llm, prompts)

    def _needs_explanation(self, user_message: str, messages: list) -> bool:
        """Check if user is asking for reasons/explanation via LLM."""
        if not user_message:
            return False

        recent_messages = messages[-12:] if messages and len(messages) > 12 else messages or []
        conversation_context = "\n".join([
            f"{msg.get('role')}: {msg.get('content')}"
            for msg in recent_messages
        ])

        template = self.prompts.get(
            "stage2_detect_explanation",
            (
                "대화 맥락:\n{conversation_context}\n\n"
                "선생님의 마지막 발화: {user_message}\n"
                "위 발화가 '왜?','어떻게?' 등 이유를 따져 묻거나 추가 설명을 요구하면 explain, "
                "그렇지 않으면 continue 라고만 답하세요."
            )
        )

        prompt = format_prompt(
            template,
            user_message=user_message,
            conversation_context=conversation_context
        )

        try:
            result = self.analyzer.invoke(prompt)
            decision = result.content.strip().lower()
            return "explain" in decision
        except Exception as e:
            logger.error(f"Stage2 explanation detection error: {e}")
            return False

    @staticmethod
    def generate_acknowledgment_and_transition(llm: ChatOpenAI, messages: list, prompts: Dict[str, str]) -> str:
        """Called from Stage1 to generate empathy + Stage2 fixed question."""
        return AcknowledgmentGenerator.generate(llm, messages, prompts)


    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: ask about AI art, then transition to Stage 3 once user responds."""
        previous_stage = state.get("previous_stage", "")
        messages = state.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        stage2_question_asked = state.get("stage2_question_asked", False)

        logger.info(f"[Stage2Handler.handle] ENTRY previous_stage={previous_stage}, stage2_question_asked={stage2_question_asked}")
        logger.info(f"[Stage2] stage2_complete={state.get('stage2_complete')}, stage2_completed={state.get('stage2_completed')}")
        logger.info(f"[Stage2] messages_count={len(messages)}, user_message='{user_message}'")

        # Already completed: skip to Stage3 first question
        if state.get("stage2_completed", False):
            logger.info(f"[Stage2] stage2_completed=True -> skip to Stage3")
            state["stage"] = "stage3"
            state["previous_stage"] = "stage2"
            state["current_asking_topic"] = "이익"

            response = self.transition_generator.STAGE3_FIRST_QUESTION

            state["stage2_complete"] = True
            state["stage2_question_asked"] = True
            state["last_response"] = response
            state["messages"].append({"role": "assistant", "content": response})
            state["message_count"] = state.get("message_count", 0) + 1

            return state

        # Question was already asked (from Stage1 transition) -> empathize + move to Stage3
        if stage2_question_asked:
            logger.info(f"[Stage2] stage2_question_asked=True -> transition to Stage3")

            response = self.transition_generator.generate(user_message)
            logger.info(f"[Stage2] response='{response}'")

            state["current_asking_topic"] = "이익"
            state["stage2_completed"] = True
            state["stage2_complete"] = True
            state["stage"] = "stage3"
            state["previous_stage"] = "stage2"

            state["last_response"] = response
            state["messages"].append({"role": "assistant", "content": response})
            state["message_count"] = state.get("message_count", 0) + 1

            logger.info(f"[Stage2] EXIT -> stage3")
            return state

        # Rare path: question not yet asked -> output fixed question
        logger.info(f"[Stage2] stage2_question_asked=False -> output fixed question")

        response = AcknowledgmentGenerator.generate(self.llm, messages, self.prompts)
        logger.info(f"[Stage2] response='{response}'")

        state["stage2_question_asked"] = True
        state["previous_stage"] = "stage1"

        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        logger.info(f"[Stage2] EXIT -> staying in stage2")
        return state
