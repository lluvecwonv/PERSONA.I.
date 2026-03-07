"""Rephrase the Stage 2 question when user gives an irrelevant answer."""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class RephraseQuestionGenerator:
    # Fixed Stage 2 question (exact wording, never change)
    STAGE2_QUESTION = "내일 그… 아내분 기일이잖아. 그냥 네가 걱정돼서 와봤어. 그, 왜 AI로 죽은 사람을 다시 재현한다는 기술 구매했다는 사람들이 꽤 있잖아. 넌 어떻게 생각해?"

    def __init__(self, llm: ChatOpenAI, prompts: dict):
        self.llm = llm
        self.prompts = prompts

    def generate(self, messages: list) -> str:
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        user_message = messages[-1]["content"] if messages else ""

        rephrase_template = self.prompts.get(
            "stage2_rephrase",
            "AI 예술에 대한 의견을 다시 물어보세요. 컨텍스트: {context}, 사용자 메시지: {user_message}"
        )

        rephrase_prompt = format_prompt(
            rephrase_template,
            context=context,
            user_message=user_message,
            original_question=self.STAGE2_QUESTION
        )

        response = self.llm.invoke(rephrase_prompt)
        return response.content.strip().strip('"')
