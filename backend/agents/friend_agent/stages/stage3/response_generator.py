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

    FINAL_CLOSING_LINE = (
        "친구: 하, 좀 더 고민해봐야겠다. 너도 잘 생각해서 결정해. 난 간다. "
        "오늘 장모님이랑 아들이랑 만나기로 했다며? 얘기 잘해봐."
    )

    def __init__(
        self,
        llm: ChatOpenAI,
        prompts: dict,
        ethics_topics: dict,
        persona_prompt: str
    ):
        """
        Args:
            llm: 응답 생성용 LLM
            prompts: 프롬프트 딕셔너리
            ethics_topics: 윤리 주제 딕셔너리 (평면 구조)
            persona_prompt: 페르소나 프롬프트
        """
        self.llm = llm
        self.prompts = prompts
        self.ethics_topics = ethics_topics
        self.persona_prompt = persona_prompt
        self.total_questions = len(ethics_topics.get("questions", []))

    def generate_with_topic(
        self,
        is_sufficient: bool,
        context: str,
        user_message: str,
        answered_count: int
    ) -> tuple:
        """
        응답 생성 및 다음 질문 인덱스 반환

        Args:
            is_sufficient: 충분한 질문을 다뤘는지 여부
            context: 대화 컨텍스트
            user_message: 사용자 메시지
            answered_count: 답변한 질문 개수

        Returns:
            (생성된 응답, 다음 질문 인덱스)
        """
        if is_sufficient:
            # 충분히 다룸 → 사용자 답변에 반응 + 마무리 응답 생성
            return self.generate_closing(context, user_message), -1
        else:
            # 더 탐구 필요 → 다음 질문으로 유도
            response, next_index = self.generate_guiding(user_message, answered_count, context)
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
        answered_count: int,
        context: str
    ) -> tuple:
        """
        다음 질문으로 유도하는 응답 생성 및 다음 질문 인덱스 반환

        Args:
            user_message: 사용자 메시지
            answered_count: 답변한 질문 개수
            context: 대화 컨텍스트

        Returns:
            (유도 응답, 다음 질문 인덱스)
        """
        # 다음 질문 인덱스 = 답변한 개수
        next_question_index = answered_count
        logger.info(f"🔍 ResponseGenerator - answered_count: {answered_count}, next_question_index: {next_question_index}")

        # 아직 물어볼 질문이 남아있는지 확인
        if next_question_index < self.total_questions:
            # 다음 질문 선택
            next_question_data = self.ethics_topics["questions"][next_question_index]
            variations = next_question_data.get("variations", [])

            if not variations:
                logger.warning(f"No variations found for question index: {next_question_index}")
                return "질문이 없습니다.", -1

            # 첫 번째 변형 사용 (variations[0])
            selected_question = variations[0]
            logger.info(f"🔍 Next question index: {next_question_index}, selected variation [0]: {selected_question[:50]}...")

            # 페르소나 + Stage 3 유도 프롬프트 통합
            stage_instruction = format_prompt(
                self.prompts.get("stage3_guide_back", "다음 질문: {next_topic_question}"),
                persona_prompt=self.persona_prompt,
                context=context,
                user_message=user_message,
                next_topic_name="",
                next_topic_description="",
                next_topic_question=selected_question
            )
            system_prompt = stage_instruction

            logger.info(f"✅ Guiding to next question #{next_question_index + 1}/{self.total_questions}")

            # ✅ 사용자 메시지를 human role로 전달하여 대화 흐름 유지
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                (
                    "human",
                    f"친구가 방금 이렇게 말했어: \"{user_message}\"\n\n"
                    "이제 다음 질문으로 자연스럽게 이어가는 대화를 생성해줘. "
                    "⚠️ 친구가 방금 사용한 단어나 문장을 3글자 이상 그대로 복사하지 말고, 같은 의미라도 다른 표현으로 바꿔서 공감해."
                )
            ])

            result = self.llm.invoke(response_prompt.format_messages())
            response = result.content.strip().strip('"')

            # ✅ 프롬프트 포맷 레이블 제거 (혹시 LLM이 포함시킨 경우)
            response = self._clean_response(response)

            return response, next_question_index
        else:
            # 모든 질문을 다뤘음 (이 경우는 발생하지 않아야 함)
            logger.warning("All questions covered but still in guiding mode")
            return "모든 질문을 다뤘습니다.", -1

    def _clean_response(self, response: str) -> str:
        """
        응답에서 프롬프트 포맷 레이블 제거

        Args:
            response: 원본 응답

        Returns:
            정제된 응답
        """
        # 제거할 레이블 패턴들
        labels_to_remove = [
            "친구 말씀:",
            "다음 질문:",
            "친구의 답변:",
            "→",
            "✅",
            "❌"
        ]

        cleaned = response
        for label in labels_to_remove:
            cleaned = cleaned.replace(label, "")

        # 여러 줄바꿈을 하나로
        import re
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # 앞뒤 공백 제거
        cleaned = cleaned.strip()

        if cleaned != response:
            logger.warning(f"⚠️ Cleaned response: removed labels from '{response[:50]}...'")

        return cleaned
