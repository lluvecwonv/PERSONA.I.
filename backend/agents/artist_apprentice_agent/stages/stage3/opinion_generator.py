"""Generate persona responses when user asks for AI's opinion."""
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

    def generate(self, user_message: str, current_topic: str, context: str) -> str:
        topic_info = self.ethics_topics.get(current_topic, {})
        topic_name = topic_info.get("name", "윤리적 측면")
        topic_description = topic_info.get("description", "")

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
