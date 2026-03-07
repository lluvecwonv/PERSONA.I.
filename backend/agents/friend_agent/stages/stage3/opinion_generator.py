"""Generate in-persona opinion responses."""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class OpinionGenerator:

    def __init__(self, llm: ChatOpenAI, prompts: dict, ethics_topics: dict, persona_prompt: str):
        self.llm = llm
        self.prompts = prompts
        self.ethics_topics = ethics_topics
        self.persona_prompt = persona_prompt

    def generate(self, user_message: str, question_index: int, context: str) -> str:
        topic_name = ""
        topic_description = ""

        opinion_prompt = format_prompt(
            self.prompts.get("ai_opinion", ""),
            topic_name=topic_name,
            topic_description=topic_description,
            context=context,
            user_message=user_message
        )

        full_prompt = self.persona_prompt + "\n\n" + opinion_prompt

        result = self.llm.invoke(full_prompt)
        return result.content.strip().strip('"')
