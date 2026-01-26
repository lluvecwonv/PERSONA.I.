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
    STAGE2_QUESTION = "내일 그… 아내분 기일이잖아. 그냥 네가 걱정돼서 와봤어. 그, 왜 AI로 죽은 사람을 다시 재현한다는 기술 구매했다는 사람들이 꽤 있잖아. 넌 어떻게 생각해?"

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
            "친구의 말에 대해 짧게 공감하세요. 친구의 말: {user_message}"
        )

        # 프롬프트 포맷팅 - conversation_history 추가
        acknowledgment_prompt = acknowledgment_template.replace("{user_message}", user_message)
        acknowledgment_prompt = acknowledgment_prompt.replace("{conversation_history}", conversation_history)

        try:
            result = llm.invoke(acknowledgment_prompt)
            acknowledgment = result.content.strip().strip('"')

            # ✅ 강력한 중복 제거: LLM이 Stage 2 질문을 여러 번 생성했는지 확인
            stage2_q = AcknowledgmentGenerator.STAGE2_QUESTION

            # 부분 문자열 매칭으로 유사 질문도 감지
            partial_matches = ["넌 어떻게 생각해?", "아내분 기일", "AI로 죽은 사람"]
            has_partial_match = any(p in acknowledgment for p in partial_matches)

            # Stage 2 질문이 몇 번 포함되었는지 확인
            count = acknowledgment.count(stage2_q)

            if count > 0:
                # Stage 2 질문이 포함되어 있으면, 모든 중복을 제거하고 1번만 남김
                parts = acknowledgment.split(stage2_q)
                # 첫 부분 (공감)만 추출
                clean_acknowledgment = parts[0].strip()

                # 공감 부분이 있으면 공감 + 질문 1번, 없으면 질문만
                if clean_acknowledgment:
                    return f"{clean_acknowledgment} {stage2_q}"
                else:
                    return stage2_q

            # 부분 매칭된 유사 질문이 있으면 그대로 반환 (추가 질문 안 함)
            if has_partial_match:
                return acknowledgment

            # Stage 2 질문이 없으면 추가
            return f"{acknowledgment} {stage2_q}"
        except Exception:
            # LLM 호출 실패 시 고정 질문만 반환
            return AcknowledgmentGenerator.STAGE2_QUESTION
