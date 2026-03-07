"""Provide explanation when user says they don't understand in Stage 2."""
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

    def generate(self, user_message: str, messages: list = None) -> str:
        conversation_context = ""
        if messages:
            recent_messages = messages[-8:] if len(messages) > 8 else messages
            conversation_context = "\n".join([
                f"{msg.get('role')}: {msg.get('content')}"
                for msg in recent_messages
            ])

        # Extract recent assistant responses to avoid repetition
        previous_responses = []
        if messages:
            previous_responses = [
                msg.get("content", "")[:100]
                for msg in messages
                if msg.get("role") == "assistant"
            ][-3:]

        explanation_template = self.prompts.get(
            "stage2_explanation",
            "사용자가 모르겠다고 했습니다: {user_message}\n설명을 제공하세요."
        )

        no_repeat_instruction = ""
        if previous_responses:
            no_repeat_instruction = f"\n\n반복 금지! 다음 문장들은 이미 말했으니 절대 반복하지 마세요:\n" + "\n".join([f"- {r}" for r in previous_responses])

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
