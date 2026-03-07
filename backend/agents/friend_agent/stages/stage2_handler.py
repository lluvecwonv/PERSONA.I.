"""Stage 2: Ask user's stance on AI recreation of deceased people."""
from typing import Dict, Any
import logging
import random
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

    def __init__(self, llm: ChatOpenAI, analyzer: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = ""):
        self.llm = llm
        self.analyzer = analyzer
        self.prompts = prompts
        self.persona_prompt = persona_prompt

        self.first_question_generator = FirstQuestionGenerator(llm, prompts)
        self.rephrase_question_generator = RephraseQuestionGenerator(llm, prompts)
        self.intent_detector = IntentDetector(analyzer, prompts)
        self.explanation_generator = ExplanationGenerator(llm, prompts)
        self.transition_generator = TransitionGenerator(llm, prompts)

    def _needs_explanation(self, user_message: str, messages: list) -> bool:
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
                "사용자 마지막 발화: {user_message}\n"
                "위 발화가 '왜?','어떻게?' 등 이유를 따져 묻거나, "
                "추가 설명을 요구하면 explain, 그렇지 않으면 continue 라고만 답하세요."
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

    def _handle_ask_opinion(self, user_message: str, messages: list) -> str:
        """Respond with "I don't know" and redirect question back to user."""
        responses = [
            "음, 나는 솔직히 잘 모르겠어. 오히려 네 생각이 궁금해. 너는 어떻게 생각해?",
            "글쎄... 나도 어떻게 생각해야 할지 모르겠더라. 너는 어때?",
            "나? 나는 아직 잘 모르겠어. 그래서 네 의견이 궁금한 건데. 어떻게 생각해?",
            "솔직히 나도 정리가 안 됐어. 네가 어떻게 생각하는지 먼저 듣고 싶어."
        ]
        response = random.choice(responses)
        logger.info(f"[Stage2] Responding to ask_opinion with: {response}")
        return response

    @staticmethod
    def generate_acknowledgment_and_transition(llm: ChatOpenAI, messages: list, prompts: Dict[str, str]) -> str:
        """Stage 1->2 transition: empathy + Stage 2 fixed question."""
        return AcknowledgmentGenerator.generate(llm, messages, prompts)


    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        previous_stage = state.get("previous_stage", "")
        messages = state.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        stage2_question_asked = state.get("stage2_question_asked", False)

        # Skip if already completed (prevent duplicate questions on re-entry)
        if state.get("stage2_completed", False):
            logger.info("Stage 2 already completed - skipping to Stage 3")
            state["stage"] = "stage3"
            state["previous_stage"] = "stage2"
            state["current_question_index"] = 0
            state["dont_know_count"] = 0

            response = self.transition_generator.STAGE3_FIRST_QUESTION

            state["stage2_complete"] = True
            state["stage2_question_asked"] = True
            state["last_response"] = response
            state["messages"].append({"role": "assistant", "content": response})
            state["message_count"] = state.get("message_count", 0) + 1

            return state

        if previous_stage != "stage2" and not stage2_question_asked:
            response = self.first_question_generator.generate(messages)
            stage2_complete = False
            logger.info("Stage 2 first visit - asking initial question")
        else:
            intent = self.intent_detector.detect(user_message, messages)
            logger.info(f"Stage 2 revisit - detected intent: {intent}")

            needs_explanation = self._needs_explanation(user_message, messages)

            if intent == "ask_opinion":
                response = self._handle_ask_opinion(user_message, messages)
                stage2_complete = False
                logger.info("User asked for agent's opinion")
            elif intent == "opinion":
                response = self.transition_generator.generate(user_message)
                stage2_complete = True
                state["current_question_index"] = 0
                state["dont_know_count"] = 0
                state["stage2_completed"] = True
                logger.info("User gave opinion - transitioning to Stage 3")
            elif intent == "dont_know" or needs_explanation:
                response = self.explanation_generator.generate(user_message, messages)
                stage2_complete = False
                logger.info("User needs explanation - providing context")
            else:
                response = self.rephrase_question_generator.generate(messages)
                stage2_complete = False
                logger.info("Unclear response - rephrasing question")

        if stage2_complete:
            state["stage"] = "stage3"
        else:
            state["stage"] = "stage2"


        state["previous_stage"] = "stage2"
        state["stage2_complete"] = stage2_complete
        state["stage2_question_asked"] = True
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        return state
