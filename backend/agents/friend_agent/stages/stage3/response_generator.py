"""
응답 생성 모듈
"""
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """응답 생성 클래스"""

    TOTAL_QUESTIONS = 5  # 총 질문 개수
    FINAL_CLOSING_LINE = (
        "친구: 하, 좀 더 고민해봐야겠다. 너도 잘 생각해서 결정해. 난 간다. "
        "오늘 장모님이랑 아들이랑 만나기로 했다며? 얘기 잘해봐."
    )

    def __init__(
        self,
        llm: ChatOpenAI,
        prompts: dict,
        ethics_topics: list,
        persona_prompt: str
    ):
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

    def generate_with_index(
        self,
        is_sufficient: bool,
        context: str,
        user_message: str,
        current_question_index: int
    ) -> tuple:
        """
        응답 생성 및 다음 질문 인덱스 반환

        Args:
            is_sufficient: 모든 질문을 다뤘는지 여부
            context: 대화 컨텍스트
            user_message: 사용자 메시지
            current_question_index: 현재 질문 인덱스

        Returns:
            (생성된 응답, 다음 질문 인덱스) - 종료 시 인덱스는 -1
        """
        if is_sufficient:
            # 모든 질문 완료 → 사용자 답변에 반응 + 마무리 응답 생성
            return self.generate_closing(context, user_message), -1
        else:
            # 더 탐구 필요 → 다음 질문으로 유도
            response, next_index = self.generate_guiding(user_message, current_question_index, context)
            return response, next_index

    def generate_closing(self, context: str, user_message: str = "") -> str:
        """
        마무리 응답 생성 - 고정된 스크립트 출력
        """
        logger.info("🔚 Returning fixed final closing line for friend agent")
        return self.FINAL_CLOSING_LINE

    def generate_guiding(
        self,
        user_message: str,
        current_question_index: int,
        context: str
    ) -> tuple:
        """
        다음 질문으로 유도하는 응답 생성 및 다음 질문 인덱스 반환

        Args:
            user_message: 사용자 메시지
            current_question_index: 현재 완료한 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            (유도 응답, 다음 질문 인덱스)
        """
        # 다음 질문 인덱스 계산 (현재 완료 + 1)
        next_question_index = current_question_index + 1
        logger.info(f"🔍 ResponseGenerator - current_question_index: {current_question_index}, next_question_index: {next_question_index}")

        # 아직 물어볼 질문이 남아있는지 확인
        if next_question_index < len(self.ethics_topics):
            # 다음 질문 선택
            next_question_data = self.ethics_topics[next_question_index]
            next_question = next_question_data.get("variations", [""])[0]

            logger.info(f"🔍 Next question #{next_question_index}: {next_question[:50]}...")

            # 페르소나 + Stage 3 유도 프롬프트 통합
            stage_instruction = format_prompt(
                self.prompts.get("stage3_guide_back", "다음 질문: {next_topic_question}"),
                persona_prompt=self.persona_prompt,
                context=context,
                user_message=user_message,
                next_topic_name="",  # 더 이상 사용 안 함
                next_topic_description="",  # 더 이상 사용 안 함
                next_topic_question=next_question
            )
            system_prompt = stage_instruction

            logger.info(f"✅ Guiding to question #{next_question_index + 1}/4")

            # ✅ 사용자 메시지를 직접 인용하지 않고, 대화 흐름만 전달 (echo 방지!)
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                (
                    "human",
                    "위 대화 맥락을 보고, 친구에게 짧게 공감한 뒤 다음 질문으로 자연스럽게 이어가는 대화를 생성해주세요. "
                    "⚠️ 친구가 방금 쓴 단어나 문장을 3글자 이상 그대로 복사하지 말고, 같은 의미라도 다른 표현으로 바꿔 말하세요."
                )
            ])

            result = self.llm.invoke(response_prompt.format_messages())
            return result.content.strip().strip('"'), next_question_index
        else:
            # 모든 질문을 다뤘음 (이 경우는 발생하지 않아야 함)
            logger.warning("All questions covered but still in guiding mode")
            return "모든 질문을 다뤘습니다.", -1
