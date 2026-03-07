"""Generates Stage 2 -> Stage 3 transition: react to user opinion + first ethics question."""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class TransitionGenerator:
    STAGE3_FIRST_QUESTION = "AI의 그림을 국립현대예술관에 전시한다라… 누구에게 어떤 이익이 있을까요?"

    def __init__(self, llm: ChatOpenAI, prompts: dict):
        self.llm = llm
        self.prompts = prompts

    def generate(self, user_message: str) -> str:
        transition_template = self.prompts.get(
            "stage2_to_stage3_transition",
            "사용자 의견: {user_message}\n다음 질문을 하세요: " + self.STAGE3_FIRST_QUESTION
        )

        transition_prompt = format_prompt(
            transition_template,
            user_message=user_message
        )

        try:
            result = self.llm.invoke(transition_prompt)
            return result.content.strip().strip('"')
        except Exception as e:
            logger.error(f"Transition generation error: {e}")
            # Fallback: simple reaction + question
            return f"네, 말씀 잘 들었어요! {self.STAGE3_FIRST_QUESTION}"
