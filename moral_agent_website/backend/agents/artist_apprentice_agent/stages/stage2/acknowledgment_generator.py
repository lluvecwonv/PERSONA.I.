"""
Stage 1 → Stage 2 전환용 공감 응답 생성 모듈
"""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class AcknowledgmentGenerator:
    """Stage 1에서 작품활동 감지 시 공감 + Stage 2 질문 생성"""

    # ✨ Stage 2의 고정된 질문 (정확한 문구, 절대 변경 금지!)
    STAGE2_QUESTION = "그런데.. 기사 보셨어요? 이번에 우리나라에서 제일 큰 기업이 '그림 그리는 AI'를 만들었다는데, 이 AI의 그림이 국립현대예술관에 전시가 된대요. 어떻게 생각하시는지 여쭤보려고 왔어요."

    @staticmethod
    def generate(llm: ChatOpenAI, messages: list, prompts: dict) -> str:
        """
        Stage 1 → Stage 2 전환 시: 이전 대화 기반 짧은 공감 + Stage 2 고정 질문

        Args:
            llm: LLM 인스턴스
            messages: 전체 대화 기록 (컨텍스트)
            prompts: 프롬프트 딕셔너리

        Returns:
            짧은 공감 + Stage 2 고정 질문
        """
        # 마지막 user 메시지 추출
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"] if user_messages else ""

        # 최근 3턴의 대화 컨텍스트 (최대 6개 메시지)
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages
        ])

        # 프롬프트 템플릿 로드
        acknowledgment_template = prompts.get(
            "stage2_acknowledgment",
            "선생님의 작품활동에 대해 짧게 공감하세요. 선생님의 말씀: {user_message}"
        )

        # 프롬프트 포맷팅 - conversation_history 추가
        acknowledgment_prompt = acknowledgment_template.replace("{user_message}", user_message)
        acknowledgment_prompt = acknowledgment_prompt.replace("{conversation_history}", conversation_history)

        try:
            result = llm.invoke(acknowledgment_prompt)
            acknowledgment = result.content.strip().strip('"')
            # 공감 + 고정 질문 결합
            return f"{acknowledgment} {AcknowledgmentGenerator.STAGE2_QUESTION}"
        except Exception:
            # LLM 호출 실패 시 고정 질문만 반환
            return AcknowledgmentGenerator.STAGE2_QUESTION
