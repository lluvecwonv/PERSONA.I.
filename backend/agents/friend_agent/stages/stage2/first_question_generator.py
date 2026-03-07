"""Generate empathy + fixed question on first Stage 2 visit."""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class FirstQuestionGenerator:
    # Fixed Stage 2 question (exact wording, never change)
    STAGE2_QUESTION = "내일 그… 아내분 기일이잖아. 그냥 네가 걱정돼서 와봤어. 그, 왜 AI로 죽은 사람을 다시 재현한다는 기술 구매했다는 사람들이 꽤 있잖아. 넌 어떻게 생각해?"

    def __init__(self, llm: ChatOpenAI, prompts: dict):
        self.llm = llm
        self.prompts = prompts

    def generate(self, messages: list) -> str:
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        last_user_message = user_messages[-1].get("content", "") if user_messages else ""

        if not last_user_message:
            return self.STAGE2_QUESTION

        recent_messages = messages[-8:] if len(messages) > 8 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        acknowledgment_template = self.prompts.get(
            "stage2_first_question",
            "선생님의 작품활동에 대해 짧게 공감하세요. 선생님의 말씀: {last_user_message}"
        )

        acknowledgment_prompt = format_prompt(
            acknowledgment_template,
            context=context,
            last_user_message=last_user_message
        )

        try:
            result = self.llm.invoke(acknowledgment_prompt)
            acknowledgment = result.content.strip().strip('"')

            # Skip appending if LLM already included the question
            if self.STAGE2_QUESTION in acknowledgment or "넌 어떻게 생각해?" in acknowledgment or "아내분 기일" in acknowledgment:
                return acknowledgment

            return f"{acknowledgment} {self.STAGE2_QUESTION}"
        except Exception:
            return self.STAGE2_QUESTION
