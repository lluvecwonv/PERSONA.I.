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

    def generate(self, user_message: str, messages: list = None) -> str:
        """
        사용자가 모르겠다고 할 때 설명 제공 후 다시 질문

        Args:
            user_message: 사용자 메시지
            messages: 전체 대화 히스토리 (맥락 유지용)

        Returns:
            설명 + 다시 질문
        """
        # ✨ 대화 히스토리 맥락 생성
        conversation_context = ""
        if messages:
            recent_messages = messages[-8:] if len(messages) > 8 else messages
            conversation_context = "\n".join([
                f"{msg.get('role')}: {msg.get('content')}"
                for msg in recent_messages
            ])

        # ✨ 이전 assistant 응답 추출 (반복 방지용)
        previous_responses = []
        if messages:
            previous_responses = [
                msg.get("content", "")[:100]
                for msg in messages
                if msg.get("role") == "assistant"
            ][-3:]  # 최근 3개만

        explanation_template = self.prompts.get(
            "stage2_explanation",
            "사용자가 모르겠다고 했습니다: {user_message}\n설명을 제공하세요."
        )

        # ✨ 대화 맥락과 반복 방지 지시 추가
        no_repeat_instruction = ""
        if previous_responses:
            no_repeat_instruction = f"\n\n⚠️ 반복 금지! 다음 문장들은 이미 말했으니 절대 반복하지 마세요:\n" + "\n".join([f"- {r}" for r in previous_responses])

        explanation_prompt = format_prompt(
            explanation_template,
            user_message=user_message,
            conversation_context=conversation_context
        ) + no_repeat_instruction

        try:
            result = self.llm.invoke(explanation_prompt)
            return result.content.strip().strip('"')
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return "응, 그런 서비스가 실제로 있어. 죽은 사람을 그대로 다시 만날 수 있게 해준대. 너는 이런 거 어떻게 생각해?"
