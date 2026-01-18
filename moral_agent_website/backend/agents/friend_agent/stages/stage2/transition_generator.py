"""
Transition generator for Stage 2 → Stage 3
사용자가 의견을 낸 후 Stage 3 첫 질문으로 전환
"""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class TransitionGenerator:
    """Stage 2 → Stage 3 전환 생성 클래스"""

    # Stage 3의 첫 번째 질문 (이익 주제 - questions[0])
    STAGE3_FIRST_QUESTION = "이 기술이 요즘 홍보를 많이 하던데, 사람들에게 도움을 줄 수 있지 않을까?"

    def __init__(self, llm: ChatOpenAI, prompts: dict):
        """
        Args:
            llm: 전환 응답 생성용 LLM
            prompts: 프롬프트 딕셔너리
        """
        self.llm = llm
        self.prompts = prompts

    def generate(self, user_message: str) -> str:
        """
        사용자 의견에 반응 + Stage 3 첫 질문 생성

        Args:
            user_message: 사용자 의견

        Returns:
            자연스러운 반응 + 첫 질문
        """
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
            # Fallback: 간단한 반응 + 질문
            return f"네, 말씀 잘 들었어요! {self.STAGE3_FIRST_QUESTION}"
