"""
Stage 2 재방문 시 재질문 생성 모듈
"""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class RephraseQuestionGenerator:
    """무관한 응답 시 짧게 공감하고 동일한 고정 질문으로 다시 물어보기"""

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
        무관한 응답 시 짧은 공감/연결 문장을 생성하고,
        고정된 Stage2 질문을 정확히 그대로 이어붙인다.
        """
        # 최근 대화 컨텍스트
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        # 사용자의 최근 메시지
        user_message = messages[-1]["content"] if messages else ""

        # 프롬프트 템플릿 로드
        rephrase_template = self.prompts.get(
            "stage2_rephrase",
            "AI 예술에 대한 의견을 다시 물어보세요. 컨텍스트: {context}, 사용자 메시지: {user_message}"
        )

        # 프롬프트 포맷팅
        rephrase_prompt = format_prompt(
            rephrase_template,
            context=context,
            user_message=user_message,
            original_question=self.STAGE2_QUESTION
        )

        try:
            result = self.llm.invoke(rephrase_prompt)
            bridge = result.content.strip().strip('"')
        except Exception:
            bridge = ""

        bridge = bridge.strip()
        if bridge:
            return f"{bridge} {self.STAGE2_QUESTION}"
        return self.STAGE2_QUESTION
