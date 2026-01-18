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
    """무관한 응답 시 다른 표현으로 AI 예술 입장 재질문"""

    # ✨ Stage 2의 고정된 질문 (정확한 문구, 절대 변경 금지!)
    STAGE2_QUESTION = "내일 그… 아내분 기일이잖아. 그냥 네가 걱정돼서 와봤어. 그, 왜 AI로 죽은 사람을 다시 재현한다는 기술 구매했다는 사람들이 꽤 있잖아. 넌 어떻게 생각해?"

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
        무관한 응답 시 다른 표현으로 AI 예술 입장 재질문

        Args:
            messages: 대화 기록

        Returns:
            재질문 응답
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

        response = self.llm.invoke(rephrase_prompt)
        return response.content.strip().strip('"')
