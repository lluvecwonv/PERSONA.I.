"""Generates a bridge sentence + re-asks the fixed Stage 2 question on irrelevant responses."""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class RephraseQuestionGenerator:
    STAGE2_QUESTION = "그런데.. 기사 보셨어요? 이번에 우리나라에서 제일 큰 기업이 '그림 그리는 AI'를 만들었다는데, 이 AI의 그림이 국립현대예술관에 전시가 된대요. 어떻게 생각하시는지 여쭤보려고 왔어요."

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

        try:
            result = self.llm.invoke(rephrase_prompt)
            bridge = result.content.strip().strip('"')
        except Exception:
            bridge = ""

        bridge = bridge.strip()
        if bridge:
            return f"{bridge} {self.STAGE2_QUESTION}"
        return self.STAGE2_QUESTION
