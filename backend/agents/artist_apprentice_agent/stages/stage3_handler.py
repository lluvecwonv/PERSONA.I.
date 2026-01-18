"""
Stage 3: 윤리 주제 탐구 대화
"""
from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI

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
        ethics_topics: Dict[str, Any],
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
        Stage 3 처리: 윤리 주제 탐구 대화

        로직:
        1. 질문에 답을 함 → 다음 질문으로
        2. 질문 이해 못함/모르겠다 → 재질문 제공 (variation 사용)
        3. 5개 질문 다 응답하면 대화 끝

        Args:
            state: 현재 대화 상태

        Returns:
            업데이트된 대화 상태
        """
        messages = state.get("messages", [])
        current_question_index = state.get("current_question_index", 0)  # 현재 질문 인덱스 (0~4)
        variation_index = state.get("variation_index", 0)  # 현재 질문의 변형 인덱스 (0: variations[0], 1: variations[1], 2: variations[2])
        need_reason_count = state.get("need_reason_count", 0)  # 이유 요청 횟수
        unsure_count = state.get("unsure_count", 0)  # "왜 모르겠어?" 질문 횟수

        logger.info(f"🔍 Stage 3 ENTRY - current_question_index: {current_question_index}, variation_index: {variation_index}, need_reason_count: {need_reason_count}, unsure_count: {unsure_count}")

        # 최근 대화 컨텍스트
        recent_messages = messages[-8:] if len(messages) > 8 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        # 사용자의 최근 답변
        user_message = messages[-1]["content"] if messages else ""
        logger.info(f"🔍 User message: {user_message[:50]}...")

        questions = self.ethics_topics.get("questions", [])
        current_question_text = ""
        if 0 <= current_question_index < len(questions):
            variations = questions[current_question_index].get("variations", [])
            current_question_text = variations[0] if variations else ""

        # ✨ 사용자 의도 감지 (이전 대화 컨텍스트 전달)
        logger.info(f"🔍 [Stage3] Detecting intent for user message: '{user_message}'")
        intent = self.intent_detector.detect(user_message, current_question=current_question_text, context=context)
        is_asking_clarification = (intent == "clarification")
        logger.info(f"🔍 [Stage3] Intent detected: {intent}, is_asking_clarification: {is_asking_clarification}")

        # 응답 생성 및 다음 질문 인덱스 설정
        next_question_index = current_question_index  # 기본값: 현재 질문 유지
        is_sufficient = False  # 기본값

        if intent == "ask_opinion":
            # 사용자가 에이전트에게 의견 질문 → "잘 모르겠다" 표현하고 질문 다시
            response = self._handle_ask_opinion(user_message, current_question_index, context)
            next_question_index = current_question_index  # 같은 질문 유지
            is_sufficient = False
            logger.info(f"💬 [Stage3] User asked for agent's opinion - staying on Q{current_question_index}")
        elif intent == "ask_why_unsure":
            # ✨ "글세", "모르겠어", "어?", "?" → "왜 모르겠어?" 질문
            # ⚠️ 최대 1번만! 이미 물어봤으면 다음 질문으로 넘어감
            unsure_count = state.get("unsure_count", 0)
            if unsure_count >= 1:
                # 이미 한 번 "왜 모르겠어?" 물어봤으면 다음 질문으로 넘어감
                logger.info(f"⚠️ [Stage3] Already asked why unsure once - moving to next question")
                state["unsure_count"] = 0  # 리셋
                state["variation_index"] = 0
                is_sufficient = current_question_index >= 4
                response, next_question_index = self.response_generator.generate_with_topic(
                    is_sufficient=is_sufficient,
                    context=context,
                    user_message=user_message,
                    answered_count=current_question_index + 1
                )
            else:
                # 첫 번째 "왜 모르겠어?" 질문
                response = self._handle_ask_why_unsure(user_message, current_question_index, context)
                next_question_index = current_question_index  # 같은 질문 계속
                is_sufficient = False
                state["unsure_count"] = unsure_count + 1  # 카운터 증가
                logger.info(f"⚠️ [Stage3] User is unsure - asking why (staying on Q{current_question_index})")
        elif intent == "ask_concept":
            # 사용자가 개념 질문 → 개념 설명 + 원래 질문 반복
            response = self._handle_concept_explanation(user_message, current_question_index, context)
            next_question_index = current_question_index  # 같은 질문 유지
            is_sufficient = False
            logger.info(f"✅ [Stage3] User asked for concept explanation - staying on Q{current_question_index}")
        elif intent == "need_reason":
            # ⚠️ 이유 요청은 최대 1번만! 이미 물어봤으면 다음 질문으로 넘어감
            if need_reason_count >= 1:
                # 이미 한 번 이유를 물어봤으면 다음 질문으로 넘어감
                logger.info(f"⚠️ [Stage3] Already asked for reason once - treating as answer and moving on")
                state["need_reason_count"] = 0  # 리셋
                state["variation_index"] = 0
                is_sufficient = current_question_index >= 4
                response, next_question_index = self.response_generator.generate_with_topic(
                    is_sufficient=is_sufficient,
                    context=context,
                    user_message=user_message,
                    answered_count=current_question_index + 1
                )
            else:
                # 첫 번째 이유 요청
                response = self._handle_need_reason(user_message, current_question_index, context, current_question_text)
                next_question_index = current_question_index
                is_sufficient = False
                state["need_reason_count"] = need_reason_count + 1  # 카운터 증가
                logger.info(f"⚠️ [Stage3] Asking for reason (1st time) - staying on Q{current_question_index}")
        elif intent == "off_topic_answer":
            # ✨ 사용자가 질문에 답하지 않고 다른 얘기를 함 → 질문 다시 연결
            logger.info(f"⚠️ [Stage3] User gave off-topic answer - reconnecting to previous question")
            response = self._handle_off_topic_answer(user_message, current_question_index, context, current_question_text)
            next_question_index = current_question_index  # 같은 질문 유지
            is_sufficient = False
        elif is_asking_clarification:
            # 사용자가 질문 이해 못함 → variations 배열에서 다음 변형 선택
            # 변형 인덱스 증가
            variation_index += 1
            logger.info(f"⚠️ [Stage3] User needs clarification - using variation #{variation_index} for Q{current_question_index}")

            # ✨ variation을 최대 2번까지만 보여주고, 그 다음에는 다음 질문으로 넘어감
            MAX_VARIATIONS = 2
            if variation_index >= MAX_VARIATIONS:
                # 2번 이상 모르겠다고 했으면 다음 질문으로 넘어감
                logger.info(f"⚠️ [Stage3] Max variations ({MAX_VARIATIONS}) reached - moving to next question")
                state["variation_index"] = 0  # 리셋

                # 5개 질문 완료 여부 체크
                is_sufficient = current_question_index >= 4

                # 다음 질문으로 이동
                response, next_question_index = self.response_generator.generate_with_topic(
                    is_sufficient=is_sufficient,
                    context=context,
                    user_message=user_message,
                    answered_count=current_question_index + 1
                )
                logger.info(f"✅ [Stage3] Skipping to next question: Q{current_question_index} → Q{next_question_index}")
            else:
                # variation 질문 생성
                response = self.clarification_generator.generate(user_message, current_question_index, context, variation_index)
                next_question_index = current_question_index  # 같은 질문 계속
                is_sufficient = False  # 아직 끝나지 않음
                state["variation_index"] = variation_index
                logger.info(f"⚠️ [Stage3] Keeping same question: current_question_index={current_question_index}, next_question_index={next_question_index}")
        else:
            # 일반 대답 → 다음 질문으로
            logger.info(f"✅ [Stage3] User answered question #{current_question_index} (Q{current_question_index + 1}/5)")

            # 변형 인덱스 및 카운터 리셋 (답변을 했으므로)
            state["variation_index"] = 0
            state["need_reason_count"] = 0
            state["unsure_count"] = 0

            # 5가지 질문을 모두 했는지 체크
            is_sufficient = current_question_index >= 4  # 0~4 = 5개
            logger.info(f"🔍 [Stage3] Question progress: {current_question_index + 1}/5 questions asked, is_sufficient={is_sufficient}")

            # 다음 질문 또는 마무리
            logger.info(f"🔍 [Stage3] Calling response_generator with answered_count={current_question_index + 1}")
            response, next_question_index = self.response_generator.generate_with_topic(
                is_sufficient=is_sufficient,
                context=context,
                user_message=user_message,
                answered_count=current_question_index + 1
            )
            logger.info(f"✅ [Stage3] Moving to next question: current_question_index={current_question_index} → next_question_index={next_question_index}")


        state["stage"] = "stage3"
        state["previous_stage"] = "stage3"
        state["current_question_index"] = next_question_index
        state["should_end"] = is_sufficient
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        logger.info(f"🔍 [Stage3] FINAL STATE - current_question_index={next_question_index}, should_end={is_sufficient}, variation_index={state.get('variation_index', 0)}")
        logger.info(f"🔍 [Stage3] Response: {response[:100]}...")

        return state

    def _handle_concept_explanation(self, user_message: str, question_index: int, context: str) -> str:
        """
        개념 설명 요청 처리 → 개념 설명 + 원래 질문 반복

        Args:
            user_message: 사용자 메시지 (개념 질문)
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            개념 설명 + 원래 질문을 포함한 응답
        """
        from utils import format_prompt

        # 현재 질문 가져오기
        questions = self.ethics_topics.get("questions", [])
        if question_index >= len(questions):
            logger.error(f"❌ Invalid question_index: {question_index}")
            return "죄송해요, 질문을 찾을 수 없어요."

        current_question_data = questions[question_index]
        original_question = current_question_data.get("question", "")
        if not original_question:
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"

        # 개념 설명 프롬프트
        concept_prompt = format_prompt(
            self.prompts.get("stage3_concept_explanation", ""),
            user_message=user_message,
            original_question=original_question
        )

        try:
            result = self.llm.invoke(concept_prompt)
            response = result.content.strip()
            logger.info(f"✅ [Stage3] Generated concept explanation for: {user_message[:30]}...")
            return response
        except Exception as e:
            logger.error(f"❌ Error generating concept explanation: {e}")
            return f"죄송해요, 설명하는 데 문제가 있었어요. 다시 질문해주시겠어요? {original_question}"

    def _handle_need_reason(self, user_message: str, question_index: int, context: str, current_question: str) -> str:
        """
        사용자가 이유 없이 입장만 말했을 때 후속 질문을 생성
        """
        from utils import format_prompt

        prompt = format_prompt(
            self.prompts.get("stage3_request_reason", ""),
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            current_question=current_question or "이 부분은 왜 그렇게 느끼세요?"
        )

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip()
        except Exception as e:
            logger.error(f"❌ Error generating reason follow-up: {e}")
            return "말씀 들으니 궁금해졌어요. 왜 그렇게 생각하세요?"

    def _handle_ask_opinion(self, user_message: str, question_index: int, context: str) -> str:
        """
        의견 요청 처리 → "잘 모르겠다" 표현하고 질문 다시

        Args:
            user_message: 사용자 메시지 (예: "너는 어떻게 생각해?")
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            "잘 모르겠다" 표현 + 질문 재시도
        """
        from utils import format_prompt

        # 현재 질문 가져오기
        questions = self.ethics_topics.get("questions", [])
        if 0 <= question_index < len(questions):
            current_question_data = questions[question_index]
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"
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

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip().strip('"')
        except Exception as e:
            logger.error(f"❌ Error generating ask_opinion response: {e}")
            return f"저도 아직 잘 모르겠어요, 그래서 선생님께 여쭤보는 거예요. {original_question}"

    def _handle_ask_why_unsure(self, user_message: str, question_index: int, context: str) -> str:
        """
        "글세", "모르겠어", "어?", "?" 처리 → "왜 모르겠어?" 질문

        Args:
            user_message: 사용자 메시지 (예: "글세...", "모르겠어요", "어?", "?")
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트

        Returns:
            "왜 모르겠어?" 질문
        """
        from utils import format_prompt

        # 현재 질문 가져오기
        questions = self.ethics_topics.get("questions", [])
        if 0 <= question_index < len(questions):
            current_question_data = questions[question_index]
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"
        else:
            original_question = "이 부분에 대해 어떻게 생각하세요?"

        prompt_template = self.prompts.get("stage3_ask_why_unsure", "")
        if not prompt_template:
            # 폴백: 기본 응답
            return f"아직 잘 모르시는군요. 어떤 부분이 헷갈리세요?"

        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        try:
            result = self.llm.invoke(prompt)
            response = result.content.strip().strip('"')
            logger.info(f"✅ [Stage3] Generated ask_why_unsure response for: {user_message[:30]}...")
            return response
        except Exception as e:
            logger.error(f"❌ Error generating ask_why_unsure response: {e}")
            return f"아직 잘 모르시는군요. 어떤 부분이 헷갈리세요?"

    def _handle_off_topic_answer(self, user_message: str, question_index: int, context: str, current_question: str) -> str:
        """
        사용자가 질문에 답하지 않고 다른 얘기를 할 때 → 사용자 발언 인정 + 질문 다시 연결

        Args:
            user_message: 사용자 메시지 (질문과 무관한 내용)
            question_index: 현재 질문 인덱스
            context: 대화 컨텍스트
            current_question: 현재 질문 텍스트

        Returns:
            사용자 발언 인정 + 질문 다시 연결
        """
        from utils import format_prompt

        # 현재 질문 가져오기
        questions = self.ethics_topics.get("questions", [])
        if 0 <= question_index < len(questions):
            current_question_data = questions[question_index]
            variations = current_question_data.get("variations", [])
            original_question = variations[0] if variations else "이 부분에 대해 어떻게 생각하세요?"
        else:
            original_question = current_question or "이 부분에 대해 어떻게 생각하세요?"

        # ✨ 프롬프트 파일에서 로드
        prompt_template = self.prompts.get("stage3_off_topic", "")

        if not prompt_template:
            # 폴백: 기본 응답
            return f"말씀하신 부분 이해해요. 그런데 제가 아까 여쭤본 건요, {original_question}"

        # 프롬프트 포맷팅
        prompt = format_prompt(
            prompt_template,
            persona_prompt=self.persona_prompt,
            context=context,
            user_message=user_message,
            original_question=original_question
        )

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip().strip('"')
        except Exception as e:
            logger.error(f"❌ Error generating off_topic response: {e}")
            return f"말씀하신 부분 이해해요. 그런데 제가 아까 여쭤본 건요, {original_question}"
