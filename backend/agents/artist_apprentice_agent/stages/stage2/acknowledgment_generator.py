"""Generates empathy response + fixed Stage 2 question for Stage1->2 transition."""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class AcknowledgmentGenerator:
    STAGE2_QUESTION = "그런데.. 기사 보셨어요? 이번에 우리나라에서 제일 큰 기업이 '그림 그리는 AI'를 만들었다는데, 이 AI의 그림이 국립현대예술관에 전시가 된대요. 어떻게 생각하시는지 여쭤보려고 왔어요."

    @staticmethod
    def generate(llm: ChatOpenAI, messages: list, prompts: dict) -> str:
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"] if user_messages else ""

        recent_messages = messages[-6:] if len(messages) > 6 else messages
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages
        ])

        acknowledgment_template = prompts.get(
            "stage2_acknowledgment",
            "선생님의 작품활동에 대해 짧게 공감하세요. 선생님의 말씀: {user_message}"
        )

        acknowledgment_prompt = acknowledgment_template.replace("{user_message}", user_message)
        acknowledgment_prompt = acknowledgment_prompt.replace("{conversation_history}", conversation_history)

        try:
            result = llm.invoke(acknowledgment_prompt)
            acknowledgment = result.content.strip().strip('"')
            return f"{acknowledgment} {AcknowledgmentGenerator.STAGE2_QUESTION}"
        except Exception:
            return AcknowledgmentGenerator.STAGE2_QUESTION
