"""
페르소나가 적용된 LLM Wrapper
"""
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from typing import List, Union

LANGUAGE_GUARD = (
    "언어 규칙:\n"
    "- 최종 응답은 반드시 자연스러운 한국어 문장으로만 작성하세요.\n"
    "- 영어 단어, 로마자 표기, 번역 괄호, 영어 예시 문장을 출력에 포함하지 마세요.\n"
    "- 설명이나 강조가 필요해도 한국어 표현만 사용하세요."
)


class PersonaLLM:
    """
    페르소나가 미리 적용된 LLM Wrapper
    ChatOpenAI와 동일한 인터페이스를 제공하되, 자동으로 페르소나를 시스템 메시지로 주입
    """

    def __init__(self, base_llm: ChatOpenAI, persona_prompt: str):
        """
        Args:
            base_llm: 기본 LLM (ChatOpenAI)
            persona_prompt: 페르소나 프롬프트 (persona.txt 내용)
        """
        self.base_llm = base_llm
        base_persona = persona_prompt.strip()
        if base_persona:
            self.persona_prompt = f"{base_persona}\n\n{LANGUAGE_GUARD}"
        else:
            self.persona_prompt = LANGUAGE_GUARD

    def invoke(self, messages: Union[str, List[BaseMessage]]):
        """
        페르소나가 적용된 LLM 호출 (ChatOpenAI 호환)

        Args:
            messages: 문자열 또는 메시지 리스트 (ChatPromptTemplate.format_messages() 결과)

        Returns:
            LLM 응답
        """
        # 문자열인 경우 직접 처리
        if isinstance(messages, str):
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.persona_prompt),
                ("human", messages)
            ])
            return self.base_llm.invoke(prompt.format_messages())

        # 메시지 리스트인 경우 (ChatPromptTemplate.format_messages() 결과)
        # 페르소나를 첫 번째 시스템 메시지로 추가
        processed_messages = [SystemMessage(content=self.persona_prompt)]

        # 기존 메시지들을 추가 (system 메시지가 있으면 내용에 추가)
        for msg in messages:
            if msg.type == "system":
                # 기존 system 메시지가 있으면 페르소나 뒤에 추가
                processed_messages[0].content += "\n\n" + msg.content
            else:
                processed_messages.append(msg)

        # LLM 호출
        return self.base_llm.invoke(processed_messages)
