"""
Explanation generator for Stage 2
사용자가 모르겠다고 할 때 설명 제공
"""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Stage 2 설명 생성 클래스"""

    def __init__(self, llm: ChatOpenAI, prompts: dict):
        """
        Args:
            llm: 설명 생성용 LLM
            prompts: 프롬프트 딕셔너리
        """
        self.llm = llm
        self.prompts = prompts

    def generate(self, user_message: str) -> str:
        """
        사용자가 모르겠다고 할 때 설명 제공 후 다시 질문

        Args:
            user_message: 사용자 메시지

        Returns:
            설명 + 다시 질문
        """
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
