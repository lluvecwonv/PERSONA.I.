"""
Stage 3: 윤리 주제 탐구 대화
"""
from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import format_prompt

from .stage3 import (
    IntentDetector,
    OpinionGenerator,
    ClarificationGenerator,
    ResponseGenerator,
)

logger = logging.getLogger(__name__)


class Stage3Handler:
    """Stage 3: 윤리 주제 탐구 대화 핸들러"""

    def __init__(
        self,
        llm: ChatOpenAI,
        analyzer: ChatOpenAI,
        prompts: Dict[str, str],
        ethics_topics: list,  # 단순 배열로 변경
        persona_prompt: str = ""
    ):
        self.llm = llm
        self.analyzer = analyzer
        self.prompts = prompts
        self.ethics_topics = ethics_topics
        self.persona_prompt = persona_prompt

        # 모듈 초기화
        self.intent_detector = IntentDetector(analyzer, prompts)
        self.opinion_generator = OpinionGenerator(llm, prompts, ethics_topics, persona_prompt)
        self.clarification_generator = ClarificationGenerator(llm, prompts, ethics_topics, persona_prompt)
        self.response_generator = ResponseGenerator(llm, prompts, ethics_topics, persona_prompt)

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3 처리: 윤리 질문 탐구 대화

        로직:
        1. 질문의 답을 하기 → 다음 질문으로
        2. 질문 해석해달라고/모르겠다 → 재질문 제공 (variation 사용)
        3. 질문과 무관한 응답 → 게임 설정에 맞게 답하고 질문 다시
        4. "너는 어떻게 생각해" → "잘 모르겠다" 표현하고 질문 다시
        5. 5개 질문 다 응답하면 대화 끝

        Args:
            state: 현재 대화 상태

        Returns:
            업데이트된 대화 상태
        """
        messages = state.get("messages", [])
        current_question_index = state.get("current_question_index", 0)  # 현재 질문 인덱스 (0~4)
        dont_know_count = state.get("dont_know_count", 0)  # 현재 질문의 모르겠다 횟수

        logger.info(f"🔍 Stage 3 ENTRY - current_question_index: {current_question_index}, dont_know_count: {dont_know_count}")

        # ✨ 바로 전 턴의 대화만 컨텍스트로 사용 (이전 턴 언급 방지!)
        # 최근 2개 메시지만: 직전 assistant 발화 + 현재 user 발화
        recent_messages = messages[-2:] if len(messages) >= 2 else messages

        context_parts = []
        for msg in recent_messages:
            if msg['role'] == 'user':
                context_parts.append(f"user: {msg['content']}")
            elif msg['role'] == 'assistant':
                # Assistant 메시지는 간단히 요약만 (질문 텍스트 제외)
                context_parts.append(f"assistant: [이전 질문]")

        context = "\n".join(context_parts)

        # 사용자의 최근 답변
        user_message = messages[-1]["content"] if messages else ""
        logger.info(f"🔍 User message: {user_message[:50]}...")

        # ✨ 다음 질문 가져오기
        next_topic_question = ""
        if 0 <= current_question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[current_question_index]
            next_topic_question = question_data.get("variations", [""])[0]

        # ✨ 사용자 의도 감지
        intent = self.intent_detector.detect(
            user_message,
            current_question_index,
            dont_know_count,
            context=context,
            next_topic_question=next_topic_question
        )
        logger.info(f"🔍 Intent detected: {intent}")

        # 응답 생성 및 다음 질문 인덱스 설정
        next_question_index = current_question_index  # 기본값: 현재 질문 유지
        is_sufficient = False  # 기본값
        response = ""

        if intent == "ask_concept":
            # 개념 설명 생성
            response = self._handle_concept_explanation(user_message, current_question_index, context)
            next_question_index = current_question_index  # 같은 질문 계속
            is_sufficient = False
            logger.info(f"✅ User asked for concept explanation - staying on Q{current_question_index}")

        elif intent == "ask_why_unsure":
            # ✨ "글세", "모르겠어" 첫 번째 → "왜 모르겠어?" 질문
            response = self._handle_ask_why_unsure(user_message, current_question_index, context)
            next_question_index = current_question_index  # 같은 질문 계속
            is_sufficient = False
            state["dont_know_count"] = 1  # 모르겠다 카운터 증가
            logger.info(f"⚠️ User is unsure - asking why (staying on Q{current_question_index})")

        elif intent in ("clarification", "dont_know_second"):
            # 모르겠다 카운터 증가
            dont_know_count += 1
            MAX_DONT_KNOW = 2

            # 두 번째 "모르겠다"면 즉시 상한에 도달시켜 다음 단계로 넘어갈 수 있도록 처리
            if intent == "dont_know_second":
                dont_know_count = max(dont_know_count, MAX_DONT_KNOW + 1)

            # ✨ "모르겠다"를 최대 2번까지만 받고, 그 다음에는 다음 질문으로 넘어감
            if dont_know_count > MAX_DONT_KNOW:
                # 2번 이상 모르겠다고 했으면 다음 질문으로 넘어감
                logger.info(f"⚠️ Max dont_know ({MAX_DONT_KNOW}) reached - moving to next question")
                state["dont_know_count"] = 0  # 리셋

                # 5개 질문 완료 여부 체크
                is_sufficient = current_question_index >= 4

                # 다음 질문으로 이동
                response, next_question_index = self.response_generator.generate_with_index(
                    is_sufficient=is_sufficient,
                    context=context,
                    user_message=user_message,
                    current_question_index=current_question_index
                )
                logger.info(f"✅ Skipping to next question: Q{current_question_index} → Q{next_question_index}")
            else:
                # variation 질문 생성
                response = self.clarification_generator.generate(
                    user_message,
                    current_question_index,
                    context,
                    dont_know_count
                )
                next_question_index = current_question_index  # 같은 질문 계속
                is_sufficient = False
                state["dont_know_count"] = dont_know_count
                logger.info(
                    f"⚠️ User needs clarification (intent={intent}, count: {dont_know_count}) - staying on Q{current_question_index}"
                )

        elif intent == "need_reason":
            response = self._handle_need_reason(user_message, current_question_index, context)
            next_question_index = current_question_index
            is_sufficient = False
            logger.info(f"⚠️ User gave opinion without reason - asking follow-up on Q{current_question_index}")

        elif intent == "unrelated":
            # 무관한 응답 → 게임 설정에 맞게 답하고 질문 다시
            response = self._handle_unrelated(user_message, current_question_index, context)
            next_question_index = current_question_index  # 같은 질문 계속
            is_sufficient = False
            logger.info(f"⚠️ User gave unrelated response - staying on Q{current_question_index}")

        elif intent == "ask_opinion":
            # 의견을 물어봄 → "잘 모르겠다" 표현하고 질문 다시
            response = self._handle_ask_opinion(user_message, current_question_index, context)
            next_question_index = current_question_index  # 같은 질문 계속
            is_sufficient = False
            logger.info(f"⚠️ User asked for opinion - staying on Q{current_question_index}")

        elif intent == "ask_explanation":
            # ✨ 에이전트가 한 말에 대해 "왜?"라고 설명 요청
            response = self._handle_ask_explanation(user_message, current_question_index, context)
            next_question_index = current_question_index  # 같은 질문 계속
            is_sufficient = False
            logger.info(f"💡 User asked for explanation - staying on Q{current_question_index}")

        else:
            # answer: 일반 대답 → 다음 질문으로
            logger.info(f"✅ User answered question #{current_question_index}")

            # 모르겠다 카운터 리셋 (답변을 했으므로)
            state["dont_know_count"] = 0

            # 5가지 질문을 모두 했는지 체크
            is_sufficient = current_question_index >= 4  # 0~4 = 5개
            logger.info(f"🔍 Question progress: {current_question_index + 1}/5 questions asked")

            # 다음 질문 또는 마무리
            response, next_question_index = self.response_generator.generate_with_index(
                is_sufficient=is_sufficient,
                context=context,
                user_message=user_message,
                current_question_index=current_question_index
            )


        state["stage"] = "stage3"
        state["previous_stage"] = "stage3"
        state["current_question_index"] = next_question_index
        state["should_end"] = is_sufficient
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        return state

    def _handle_unrelated(self, user_message: str, question_index: int, context: str) -> str:
        """
        무관한 응답 처리 → 게임 설정에 맞게 답하고 질문 다시

        Args:
            user_message: 사용자 메시지
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            게임 설정에 맞는 답변 + 질문 재시도
        """
        if 0 <= question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[question_index]
            original_question = question_data.get("variations", [""])[0]
        else:
            original_question = "이 부분에 대해 어떻게 생각하세요?"

        # ✨ 프롬프트 파일에서 로드
        prompt_template = self.prompts.get("stage3_unrelated", "")

        # 프롬프트 포맷팅
        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        result = self.llm.invoke(prompt)
        return result.content.strip().strip('"')

    def _handle_need_reason(self, user_message: str, question_index: int, context: str) -> str:
        """
        근거 없이 입장만 말했을 때 후속 질문으로 이유를 요청
        """
        if 0 <= question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[question_index]
            original_question = question_data.get("variations", [""])[0]
        else:
            original_question = "이 부분에 대해 어떻게 생각해?"

        prompt_template = self.prompts.get("stage3_request_reason", "")
        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        result = self.llm.invoke(prompt)
        return result.content.strip().strip('"')

    def _handle_ask_why_unsure(self, user_message: str, question_index: int, context: str) -> str:
        """
        "글세", "모르겠어" 처리 → "왜 모르겠어?" 질문

        Args:
            user_message: 사용자 메시지 (예: "글세...", "모르겠어")
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            "왜 모르겠어?" 질문
        """
        if 0 <= question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[question_index]
            original_question = question_data.get("variations", [""])[0]
        else:
            original_question = "이 부분에 대해 어떻게 생각해?"

        prompt_template = self.prompts.get("stage3_ask_why_unsure", "")
        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        result = self.llm.invoke(prompt)
        return result.content.strip().strip('"')

    def _handle_ask_opinion(self, user_message: str, question_index: int, context: str) -> str:
        """
        의견 물어봄 처리 → "잘 모르겠다" 표현하고 질문 다시

        Args:
            user_message: 사용자 메시지
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            "잘 모르겠다" 표현 + 질문 재시도
        """
        if 0 <= question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[question_index]
            original_question = question_data.get("variations", [""])[0]
        else:
            original_question = "이 부분에 대해 어떻게 생각하세요?"

        # ✨ 프롬프트 파일에서 로드
        prompt_template = self.prompts.get("stage3_ask_opinion", "")

        # 프롬프트 포맷팅
        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        result = self.llm.invoke(prompt)
        return result.content.strip().strip('"')

    def _handle_concept_explanation(self, user_message: str, question_index: int, context: str) -> str:
        """
        개념 설명 요청 처리 → 개념 설명 + 원래 질문 반복

        Args:
            user_message: 사용자 메시지 (예: "자율성이 뭐야?")
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            개념 설명 + 원래 질문
        """
        if 0 <= question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[question_index]
            original_question = question_data.get("variations", [""])[0]
        else:
            original_question = "이 부분에 대해 어떻게 생각하세요?"

        # ✨ 프롬프트 파일에서 로드
        prompt_template = self.prompts.get("stage3_concept_explanation", "")

        # 프롬프트 포맷팅
        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        result = self.llm.invoke(prompt)
        return result.content.strip().strip('"')

    def _handle_ask_explanation(self, user_message: str, question_index: int, context: str) -> str:
        """
        에이전트가 한 말에 대해 "왜?"라고 설명 요청 처리

        Args:
            user_message: 사용자 메시지 (예: "왜 못물어?", "왜 그래?")
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            설명 + 질문
        """
        if 0 <= question_index < len(self.ethics_topics):
            question_data = self.ethics_topics[question_index]
            original_question = question_data.get("variations", [""])[0]
        else:
            original_question = "이 부분에 대해 어떻게 생각해?"

        # ✨ 프롬프트 파일에서 로드
        prompt_template = self.prompts.get("stage3_explain_reasoning", "")

        # 프롬프트 포맷팅
        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        result = self.llm.invoke(prompt)
        return result.content.strip().strip('"')
