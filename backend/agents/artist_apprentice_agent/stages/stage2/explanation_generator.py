"""Provides explanation when user says they don't know, then re-asks."""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    def __init__(self, llm: ChatOpenAI, prompts: dict):
        self.llm = llm
        self.prompts = prompts

    def generate(self, user_message: str) -> str:
        explanation_template = self.prompts.get(
            "stage2_explanation",
            "사용자가 모르겠다고 했습니다: {user_message}\n설명을 제공하세요."
        )

        explanation_prompt = format_prompt(
            explanation_template,
            user_message=user_message
        )

        try:
            result = self.llm.invoke(explanation_prompt)
            return result.content.strip().strip('"')
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return "아, 그럴 수 있죠! AI가 그린 그림이 박물관에 전시된다는데, 어떻게 생각하시나요?"
