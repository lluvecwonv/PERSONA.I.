"""
Stage 2 첫 방문 시 질문 생성 모듈
"""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class FirstQuestionGenerator:
    """Stage 2 첫 방문 시 이전 대화 기반 공감 + 고정 질문 생성"""

    # ✨ Stage 2의 고정된 질문 (정확한 문구, 절대 변경 금지!)
    STAGE2_QUESTION = "그런데.. 기사 보셨어요? 이번에 우리나라에서 제일 큰 기업이 '그림 그리는 AI'를 만들었다는데, 이 AI의 그림이 국립현대예술관에 전시가 된대요. 어떻게 생각하시는지 여쭤보려고 왔어요."

    def __init__(self, llm: ChatOpenAI, prompts: dict):
        """
        Args:
            llm: 응답 생성용 LLM
            prompts: 프롬프트 딕셔너리
        """
        self.llm = llm
        self.prompts = prompts

    def generate(self, messages: list) -> str:
        """
        첫 방문 시 이전 대화 히스토리 기반 짧은 공감 + Stage 2 고정 질문

        Args:
            messages: 전체 대화 기록

        Returns:
            공감 + 고정 질문
        """
        # 사용자의 마지막 메시지 (작품활동 내용)
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        last_user_message = user_messages[-1].get("content", "") if user_messages else ""

        if not last_user_message:
            # 히스토리 없으면 고정 질문만 반환
            return self.STAGE2_QUESTION

        # 프롬프트 템플릿 로드
        acknowledgment_template = self.prompts.get(
            "stage2_first_question",
            "선생님의 작품활동에 대해 짧게 공감하세요. 선생님의 말씀: {last_user_message}"
        )

        # 프롬프트 포맷팅
        acknowledgment_prompt = format_prompt(
            acknowledgment_template,
            last_user_message=last_user_message
        )

        try:
            result = self.llm.invoke(acknowledgment_prompt)
            acknowledgment = result.content.strip().strip('"')

            # 공감 + 고정 질문 결합
            return f"{acknowledgment} {self.STAGE2_QUESTION}"
        except Exception:
            # LLM 호출 실패 시 고정 질문만 반환
            return self.STAGE2_QUESTION
