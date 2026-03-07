"""Stage 3: Ethics topic exploration dialogue."""
from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI

from .stage3 import (
    IntentDetector,
    OpinionGenerator,
    ClarificationGenerator,
    ResponseGenerator,
)

logger = logging.getLogger(__name__)


class Stage3Handler:
    def __init__(
        self,
        llm: ChatOpenAI,
        analyzer: ChatOpenAI,
        prompts: Dict[str, str],
        ethics_topics: Dict[str, Any],
        persona_prompt: str = ""
    ):
        self.llm = llm
        self.analyzer = analyzer
        self.prompts = prompts
        self.ethics_topics = ethics_topics
        self.persona_prompt = persona_prompt

        self.intent_detector = IntentDetector(analyzer, prompts)
        self.opinion_generator = OpinionGenerator(llm, prompts, ethics_topics, persona_prompt)
        self.clarification_generator = ClarificationGenerator(llm, prompts, ethics_topics, persona_prompt)
        self.response_generator = ResponseGenerator(llm, prompts, ethics_topics, persona_prompt)

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect user intent, handle accordingly, advance through 5 ethics questions."""
        messages = state.get("messages", [])
        current_question_index = state.get("current_question_index", 0)
        variation_index = state.get("variation_index", 0)
        need_reason_count = state.get("need_reason_count", 0)
        unsure_count = state.get("unsure_count", 0)

        logger.info(f"Stage 3 ENTRY - q_idx: {current_question_index}, var_idx: {variation_index}, need_reason: {need_reason_count}, unsure: {unsure_count}")

        recent_messages = messages[-8:] if len(messages) > 8 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        user_message = messages[-1]["content"] if messages else ""
        logger.info(f"User message: {user_message[:50]}...")

        questions = self.ethics_topics.get("questions", [])
        current_question_text = ""
        if 0 <= current_question_index < len(questions):
            variations = questions[current_question_index].get("variations", [])
            current_question_text = variations[0] if variations else ""

        intent = self.intent_detector.detect(user_message, current_question=current_question_text, context=context)
        is_asking_clarification = (intent == "clarification")
        logger.info(f"[Stage3] Intent: {intent}")

        next_question_index = current_question_index
        is_sufficient = False

        if intent == "ask_opinion":
            response = self._handle_ask_opinion(user_message, current_question_index, context)
            logger.info(f"[Stage3] ask_opinion - staying on Q{current_question_index}")
        elif intent == "ask_why_unsure":
            unsure_count = state.get("unsure_count", 0)
            if unsure_count >= 1:
                # Already asked once -> move on
                logger.info(f"[Stage3] Already asked why unsure -> next question")
                state["unsure_count"] = 0
                state["variation_index"] = 0
                is_sufficient = current_question_index >= 4
                response, next_question_index = self.response_generator.generate_with_topic(
                    is_sufficient=is_sufficient,
                    context=context,
                    user_message=user_message,
                    answered_count=current_question_index + 1
                )
            else:
                response = self._handle_ask_why_unsure(user_message, current_question_index, context)
                state["unsure_count"] = unsure_count + 1
                logger.info(f"[Stage3] User unsure - asking why (staying on Q{current_question_index})")
        elif intent == "ask_concept":
            response = self._handle_concept_explanation(user_message, current_question_index, context)
            logger.info(f"[Stage3] Concept explanation - staying on Q{current_question_index}")
        elif intent == "need_reason":
            # Max 1 reason request per question
            if need_reason_count >= 1:
                logger.info(f"[Stage3] Already asked for reason -> treating as answer")
                state["need_reason_count"] = 0
                state["variation_index"] = 0
                is_sufficient = current_question_index >= 4
                response, next_question_index = self.response_generator.generate_with_topic(
                    is_sufficient=is_sufficient,
                    context=context,
                    user_message=user_message,
                    answered_count=current_question_index + 1
                )
            else:
                response = self._handle_need_reason(user_message, current_question_index, context, current_question_text)
                state["need_reason_count"] = need_reason_count + 1
                logger.info(f"[Stage3] Asking for reason (1st time) - staying on Q{current_question_index}")
        elif is_asking_clarification:
            variation_index += 1
            logger.info(f"[Stage3] Clarification - variation #{variation_index} for Q{current_question_index}")

            MAX_VARIATIONS = 2
            if variation_index >= MAX_VARIATIONS:
                logger.info(f"[Stage3] Max variations reached -> next question")
                state["variation_index"] = 0
                is_sufficient = current_question_index >= 4
                response, next_question_index = self.response_generator.generate_with_topic(
                    is_sufficient=is_sufficient,
                    context=context,
                    user_message=user_message,
                    answered_count=current_question_index + 1
                )
            else:
                response = self.clarification_generator.generate(user_message, current_question_index, context, variation_index)
                state["variation_index"] = variation_index
        elif intent == "ask_explanation":
            response = self._handle_ask_explanation(user_message, current_question_index, context)
            logger.info(f"[Stage3] Explanation request - staying on Q{current_question_index}")
        else:
            # Normal answer -> next question
            logger.info(f"[Stage3] Answer to Q{current_question_index + 1}/5")
            state["variation_index"] = 0
            state["need_reason_count"] = 0
            state["unsure_count"] = 0

            is_sufficient = current_question_index >= 4
            response, next_question_index = self.response_generator.generate_with_topic(
                is_sufficient=is_sufficient,
                context=context,
                user_message=user_message,
                answered_count=current_question_index + 1
            )
            logger.info(f"[Stage3] Q{current_question_index} -> Q{next_question_index}")

        state["stage"] = "stage3"
        state["previous_stage"] = "stage3"
        state["current_question_index"] = next_question_index
        state["should_end"] = is_sufficient
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        logger.info(f"[Stage3] FINAL - q_idx={next_question_index}, should_end={is_sufficient}")
        logger.info(f"[Stage3] Response: {response[:100]}...")

        return state

    def _handle_concept_explanation(self, user_message: str, question_index: int, context: str) -> str:
        from utils import format_prompt

        questions = self.ethics_topics.get("questions", [])
        if question_index >= len(questions):
            logger.error(f"Invalid question_index: {question_index}")
            return "죄송해요, 질문을 찾을 수 없어요."

        current_question_data = questions[question_index]
        original_question = current_question_data.get("question", "")
        if not original_question:
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"

        concept_prompt = format_prompt(
            self.prompts.get("stage3_concept_explanation", ""),
            user_message=user_message,
            original_question=original_question
        )

        try:
            result = self.llm.invoke(concept_prompt)
            response = result.content.strip()
            logger.info(f"[Stage3] Generated concept explanation for: {user_message[:30]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating concept explanation: {e}")
            return f"죄송해요, 설명하는 데 문제가 있었어요. 다시 질문해주시겠어요? {original_question}"

    def _handle_need_reason(self, user_message: str, question_index: int, context: str, current_question: str) -> str:
        """Generate follow-up when user states opinion without reasoning."""
        from utils import format_prompt

        prompt = format_prompt(
            self.prompts.get("stage3_request_reason", ""),
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            current_question=current_question or "이 부분은 왜 그렇게 느끼세요?"
        )

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip()
        except Exception as e:
            logger.error(f"Error generating reason follow-up: {e}")
            return "말씀 들으니 궁금해졌어요. 왜 그렇게 생각하세요?"

    def _handle_ask_opinion(self, user_message: str, question_index: int, context: str) -> str:
        from utils import format_prompt

        questions = self.ethics_topics.get("questions", [])
        if 0 <= question_index < len(questions):
            current_question_data = questions[question_index]
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"
        else:
            original_question = "이 부분에 대해 어떻게 생각하세요?"

        prompt_template = self.prompts.get("stage3_ask_opinion", "")

        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip().strip('"')
        except Exception as e:
            logger.error(f"Error generating ask_opinion response: {e}")
            return f"저도 아직 잘 모르겠어요, 그래서 선생님께 여쭤보는 거예요. {original_question}"

    def _handle_ask_why_unsure(self, user_message: str, question_index: int, context: str) -> str:
        from utils import format_prompt

        questions = self.ethics_topics.get("questions", [])
        if 0 <= question_index < len(questions):
            current_question_data = questions[question_index]
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"
        else:
            original_question = "이 부분에 대해 어떻게 생각하세요?"

        prompt_template = self.prompts.get("stage3_ask_why_unsure", "")
        if not prompt_template:
            return f"아직 잘 모르시는군요. 어떤 부분이 헷갈리세요?"

        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        try:
            result = self.llm.invoke(prompt)
            response = result.content.strip().strip('"')
            logger.info(f"[Stage3] Generated ask_why_unsure response for: {user_message[:30]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating ask_why_unsure response: {e}")
            return f"아직 잘 모르시는군요. 어떤 부분이 헷갈리세요?"

    def _handle_ask_explanation(self, user_message: str, question_index: int, context: str) -> str:
        from utils import format_prompt

        questions = self.ethics_topics.get("questions", [])
        if 0 <= question_index < len(questions):
            current_question_data = questions[question_index]
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"
        else:
            original_question = "이 부분에 대해 어떻게 생각하세요?"

        prompt_template = self.prompts.get("stage3_explain_reasoning", "")

        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip().strip('"')
        except Exception as e:
            logger.error(f"Error generating ask_explanation response: {e}")
            return f"그런 질문을 드린 건 여러 관점이 있어서예요. 선생님은 어떻게 생각하세요?"
