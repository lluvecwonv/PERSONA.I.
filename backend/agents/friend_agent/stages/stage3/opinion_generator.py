"""
AI 의견 답변 생성 모듈
"""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class OpinionGenerator:
    """AI 의견 답변 생성 클래스"""

    def __init__(self, llm: ChatOpenAI, prompts: dict, ethics_topics: list, persona_prompt: str):
        """
        Args:
            llm: 응답 생성용 LLM
            prompts: 프롬프트 딕셔너리
            ethics_topics: 질문 배열 (단순 리스트)
            persona_prompt: 페르소나 프롬프트
        """
        self.llm = llm
        self.prompts = prompts
        self.ethics_topics = ethics_topics
        self.persona_prompt = persona_prompt

    def generate(self, user_message: str, question_index: int, context: str) -> str:
        """
        사용자가 AI의 의견을 물어봤을 때 페르소나로 답변 생성

        Args:
            user_message: 사용자 메시지
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            페르소나 답변
        """
        # 주제 정보는 더 이상 사용하지 않음 (단순화)
        topic_name = ""
        topic_description = ""

        # 페르소나 + AI 의견 프롬프트
        opinion_prompt = format_prompt(
            self.prompts.get("ai_opinion", ""),
            topic_name=topic_name,
            topic_description=topic_description,
            context=context,
            user_message=user_message
        )

        # 페르소나 결합
        full_prompt = self.persona_prompt + "\n\n" + opinion_prompt

        result = self.llm.invoke(full_prompt)
        return result.content.strip().strip('"')
