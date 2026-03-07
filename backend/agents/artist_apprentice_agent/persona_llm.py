"""LLM wrapper that auto-injects persona as a system message."""
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from typing import List, Union

LANGUAGE_GUARD = (
    "언어 규칙:\n"
    "- 모든 최종 응답은 100% 자연스러운 한국어 문장으로만 작성하세요.\n"
    "- 영어 단어, 로마자 표기, 번역 괄호, 영어 예시 문장을 출력에 포함하지 마세요.\n"
    "- 번역이나 설명이 필요해도 한국어 표현만 사용하세요."
)


class PersonaLLM:
    """ChatOpenAI-compatible wrapper that prepends persona to every call."""

    def __init__(self, base_llm: ChatOpenAI, persona_prompt: str):
        self.base_llm = base_llm
        base_persona = persona_prompt.strip()
        if base_persona:
            self.persona_prompt = f"{base_persona}\n\n{LANGUAGE_GUARD}"
        else:
            self.persona_prompt = LANGUAGE_GUARD

    def invoke(self, messages: Union[str, List[BaseMessage]]):
        if isinstance(messages, str):
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.persona_prompt),
                ("human", messages)
            ])
            return self.base_llm.invoke(prompt.format_messages())

        # For message lists, merge any existing system messages into the persona
        processed_messages = [SystemMessage(content=self.persona_prompt)]
        for msg in messages:
            if msg.type == "system":
                processed_messages[0].content += "\n\n" + msg.content
            else:
                processed_messages.append(msg)

        return self.base_llm.invoke(processed_messages)
