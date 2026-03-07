"""Re-asks the current question using a variation when the user needs clarification."""
import logging
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class ClarificationGenerator:
    def __init__(self, llm: ChatOpenAI, prompts: dict, ethics_topics: dict, persona_prompt: str):
        self.llm = llm
        self.prompts = prompts
        self.ethics_topics = ethics_topics
        self.persona_prompt = persona_prompt

    def generate(self, user_message: str, question_index: int, context: str, variation_index: int = 1) -> str:
        """Pick the next variation of the question and wrap it with empathy."""
        if 0 <= question_index < len(self.ethics_topics.get("questions", [])):
            question_data = self.ethics_topics["questions"][question_index]
            variations = question_data.get("variations", [])

            if variations:
                actual_index = min(variation_index, len(variations) - 1)
                original_question = variations[actual_index]
                logger.info(f"Using Q{question_index} variation[{actual_index}]: {original_question[:50]}...")

                clarification_template = self.prompts.get(
                    "stage3_clarification",
                    "선생님이 모르겠다고 하셨습니다: {user_message}\\n질문을 다시 물어보세요: {original_question}"
                )

                clarification_prompt = format_prompt(
                    clarification_template,
                    user_message=user_message,
                    context=context,
                    original_question=original_question
                )

                full_prompt = self.persona_prompt + "\\n\\n" + clarification_prompt

                try:
                    result = self.llm.invoke(full_prompt)
                    return result.content.strip().strip('"')
                except Exception as e:
                    logger.error(f"Clarification generation error: {e}")
                    return f"그럴 수 있죠. {original_question}"

        return "이 부분에 대해 어떻게 생각하세요?"
