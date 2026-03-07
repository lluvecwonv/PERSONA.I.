"""Generate responses and guide conversation to next ethics question."""
import logging
import re
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class ResponseGenerator:

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
        if is_sufficient:
            return self.generate_closing(context, user_message), -1
        else:
            response, next_index = self.generate_guiding(user_message, answered_count, context)
            return response, next_index

    def generate_closing(self, context: str, user_message: str = "") -> str:
        logger.info("Returning fixed final closing line")
        return self.FINAL_CLOSING_LINE

    def generate_guiding(
        self,
        user_message: str,
        answered_count: int,
        context: str
    ) -> tuple:
        next_question_index = answered_count
        logger.info(f"ResponseGenerator - answered: {answered_count}, next_index: {next_question_index}")

        if next_question_index < self.total_questions:
            next_question_data = self.ethics_topics["questions"][next_question_index]
            variations = next_question_data.get("variations", [])

            if not variations:
                logger.warning(f"No variations for question index: {next_question_index}")
                return "질문이 없습니다.", -1

            selected_question = variations[0]
            logger.info(f"Next question [{next_question_index}]: {selected_question[:50]}...")

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

            logger.info(f"Guiding to question #{next_question_index + 1}/{self.total_questions}")

            response_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                (
                    "human",
                    f"친구가 방금 이렇게 말했어: \"{user_message}\"\n\n"
                    "이제 다음 질문으로 자연스럽게 이어가는 대화를 생성해줘. "
                    "친구가 방금 사용한 단어나 문장을 3글자 이상 그대로 복사하지 말고, 같은 의미라도 다른 표현으로 바꿔서 공감해."
                )
            ])

            result = self.llm.invoke(response_prompt.format_messages())
            response = result.content.strip().strip('"')

            response = self._clean_response(response)

            return response, next_question_index
        else:
            logger.warning("All questions covered but still in guiding mode")
            return "모든 질문을 다뤘습니다.", -1

    def _clean_response(self, response: str) -> str:
        """Remove prompt format labels that LLM may have included."""
        labels_to_remove = [
            "친구 말씀:",
            "다음 질문:",
            "친구의 답변:",
        ]

        cleaned = response
        for label in labels_to_remove:
            cleaned = cleaned.replace(label, "")

        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        if cleaned != response:
            logger.warning(f"Cleaned response: removed labels from '{response[:50]}...'")

        return cleaned
