"""Generate transition response from Stage 2 to Stage 3."""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class TransitionGenerator:
    STAGE3_FIRST_QUESTION = "이 기술이 요즘 홍보를 많이 하던데, 사람들에게 도움을 줄 수 있지 않을까?"

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
            return f"네, 말씀 잘 들었어요! {self.STAGE3_FIRST_QUESTION}"
