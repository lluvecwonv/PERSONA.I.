"""Generates empathy + fixed Stage 2 question on first visit."""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class FirstQuestionGenerator:
    STAGE2_QUESTION = "그런데.. 기사 보셨어요? 이번에 우리나라에서 제일 큰 기업이 '그림 그리는 AI'를 만들었다는데, 이 AI의 그림이 국립현대예술관에 전시가 된대요. 어떻게 생각하시는지 여쭤보려고 왔어요."

    def __init__(self, llm: ChatOpenAI, prompts: dict):
        self.llm = llm
        self.prompts = prompts

    def generate(self, messages: list) -> str:
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        last_user_message = user_messages[-1].get("content", "") if user_messages else ""

        if not last_user_message:
            return self.STAGE2_QUESTION

        acknowledgment_template = self.prompts.get(
            "stage2_first_question",
            "선생님의 작품활동에 대해 짧게 공감하세요. 선생님의 말씀: {last_user_message}"
        )

        acknowledgment_prompt = format_prompt(
            acknowledgment_template,
            last_user_message=last_user_message
        )

        try:
            result = self.llm.invoke(acknowledgment_prompt)
            acknowledgment = result.content.strip().strip('"')
            return f"{acknowledgment} {self.STAGE2_QUESTION}"
        except Exception:
            return self.STAGE2_QUESTION
