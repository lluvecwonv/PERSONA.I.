"""
설명 제공 모듈
"""
import logging
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class ClarificationGenerator:
    """설명 제공 클래스"""

    def __init__(self, llm: ChatOpenAI, prompts: dict, ethics_topics: dict, persona_prompt: str):
        """
        Args:
            llm: 응답 생성용 LLM
            prompts: 프롬프트 딕셔너리
            ethics_topics: 윤리 주제 딕셔너리
            persona_prompt: 페르소나 프롬프트
        """
        self.llm = llm
        self.prompts = prompts
        self.ethics_topics = ethics_topics
        self.persona_prompt = persona_prompt

    def generate(self, user_message: str, question_index: int, context: str, variation_index: int = 1) -> str:
        """
        사용자가 질문/이해 요청했을 때 공감 + 질문을 다시 물어봄

        Args:
            user_message: 사용자 메시지
            question_index: 현재 질문 인덱스 (0~4)
            context: 대화 컨텍스트
            variation_index: 변형 인덱스 (1: variations[1], 2: variations[2])

        Returns:
            공감 + 재질문
        """
        # 질문 데이터 가져오기
        if 0 <= question_index < len(self.ethics_topics.get("questions", [])):
            question_data = self.ethics_topics["questions"][question_index]
            variations = question_data.get("variations", [])

            # ✨ variation_index에 따라 순서대로 variations 배열에서 선택
            if variations:
                actual_index = min(variation_index, len(variations) - 1)
                original_question = variations[actual_index]
                logger.info(f"✅ Using Q{question_index} variation[{actual_index}] for variation_index={variation_index}: {original_question[:50]}...")

                # ✨ 프롬프트를 사용하여 공감 + 재질문 생성
                clarification_template = self.prompts.get(
                    "stage3_clarification",
                    "친구가 모르겠다고 했어: {user_message}\\n질문을 다시 물어봐: {original_question}"
                )

                clarification_prompt = format_prompt(
                    clarification_template,
                    user_message=user_message,
                    context=context,
                    original_question=original_question
                )

                full_prompt = self.persona_prompt + "\\n\\n" + clarification_prompt

                try:
                    result = self.llm.invoke(full_prompt)
                    return result.content.strip().strip('"')
                except Exception as e:
                    logger.error(f"Clarification generation error: {e}")
                    # 폴백: 간단한 멘트 + 질문
                    return f"그럴 수 있지. {original_question}"

        # 폴백: 기본 메시지
        return "이 부분에 대해 어떻게 생각해?"
