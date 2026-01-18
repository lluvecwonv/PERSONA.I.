"""
설명 제공 모듈
"""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class ClarificationGenerator:
    """설명 제공 클래스"""

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

    def generate(self, user_message: str, question_index: int, context: str, dont_know_count: int = 0) -> str:
        """
        사용자가 질문/이해 요청했을 때 공감 + 질문을 다시 물어봄

        Args:
            user_message: 사용자 메시지
            question_index: 현재 질문 인덱스 (0~4)
            context: 대화 컨텍스트
            dont_know_count: 모르겠다 카운트 (0: 첫 질문, 1: 재질문1, 2: 재질문2)

        Returns:
            공감 + 재질문
        """
        # 질문 정보 가져오기
        if 0 <= question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[question_index]
            variations = question_data.get("variations", [])

            # ✨ dont_know_count에 따라 순서대로 variations 배열에서 선택
            # dont_know_count == 0 → variations[0] (첫 번째 질문)
            # dont_know_count == 1 → variations[1] (두 번째 질문)
            # dont_know_count == 2 → variations[2] (세 번째 질문)
            if variations:
                variation_index = min(dont_know_count, len(variations) - 1)
                original_question = variations[variation_index]
                logger.info(f"✅ Using Q{question_index} variation[{variation_index}] for dont_know_count={dont_know_count}: {original_question[:50]}...")

                # ✨ 프롬프트를 사용하여 공감 + 재질문 생성
                clarification_template = self.prompts.get(
                    "stage3_clarification",
                    "친구가 모르겠다고 했습니다: {user_message}\n질문을 다시 물어보세요: {original_question}"
                )

                clarification_prompt = format_prompt(
                    clarification_template,
                    user_message=user_message,
                    original_question=original_question,
                    context=context
                )

                full_prompt = self.persona_prompt + "\n\n" + clarification_prompt

                try:
                    result = self.llm.invoke(full_prompt)
                    return result.content.strip().strip('"')
                except Exception as e:
                    logger.error(f"Clarification generation error: {e}")
                    # 폴백: 간단한 멘트 + 질문
                    return f"그럴 수 있지. {original_question}"

        # 폴백: 기본 메시지
        return "이 부분에 대해 어떻게 생각하세요?"