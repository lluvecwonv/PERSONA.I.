"""
Stage 2: 죽은 사람을 AI로 재현하는 기술에 대한 입장 질문 단계
"""
from typing import Dict, Any
import logging
from pathlib import Path
import sys
from langchain_openai import ChatOpenAI

from .stage2 import (
    AcknowledgmentGenerator,
    FirstQuestionGenerator,
    RephraseQuestionGenerator,
    IntentDetector,
    ExplanationGenerator,
    TransitionGenerator,
)

logger = logging.getLogger(__name__)

# utils import
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import format_prompt


class Stage2Handler:
    """Stage 2: 죽은 사람을 AI로 재현하는 기술에 대한 입장 질문 핸들러"""

    def __init__(self, llm: ChatOpenAI, analyzer: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = ""):
        self.llm = llm
        self.analyzer = analyzer
        self.prompts = prompts
        self.persona_prompt = persona_prompt

        # ✨ 모듈화된 생성기 초기화
        self.first_question_generator = FirstQuestionGenerator(llm, prompts)
        self.rephrase_question_generator = RephraseQuestionGenerator(llm, prompts)
        # 의도 판단은 페르소나/온도 영향을 받지 않도록 analyzer 사용
        self.intent_detector = IntentDetector(analyzer, prompts)
        self.explanation_generator = ExplanationGenerator(llm, prompts)
        self.transition_generator = TransitionGenerator(llm, prompts)

    def _needs_explanation(self, user_message: str, messages: list) -> bool:
        """사용자가 이유/설명을 요구하는지 LLM으로 판단"""
        if not user_message:
            return False

        recent_messages = messages[-12:] if messages and len(messages) > 12 else messages or []
        conversation_context = "\n".join([
            f"{msg.get('role')}: {msg.get('content')}"
            for msg in recent_messages
        ])

        template = self.prompts.get(
            "stage2_detect_explanation",
            (
                "대화 맥락:\n{conversation_context}\n\n"
                "사용자 마지막 발화: {user_message}\n"
                "위 발화가 '왜?','어떻게?' 등 이유를 따져 묻거나, "
                "추가 설명을 요구하면 explain, 그렇지 않으면 continue 라고만 답하세요."
            )
        )

        prompt = format_prompt(
            template,
            user_message=user_message,
            conversation_context=conversation_context
        )

        try:
            result = self.analyzer.invoke(prompt)
            decision = result.content.strip().lower()
            return "explain" in decision
        except Exception as e:
            logger.error(f"Stage2 explanation detection error: {e}")
            return False

    def _handle_ask_opinion(self, user_message: str, messages: list) -> str:
        """
        에이전트에게 의견을 물었을 때 처리 → "잘 모르겠다"고 답변

        Args:
            user_message: 사용자 메시지
            messages: 전체 대화 기록

        Returns:
            "잘 모르겠다" + 되묻기 응답
        """
        # ✨ 다양한 "잘 모르겠다" 응답 중 하나 선택
        import random
        responses = [
            "음, 나는 솔직히 잘 모르겠어. 오히려 네 생각이 궁금해. 너는 어떻게 생각해?",
            "글쎄... 나도 어떻게 생각해야 할지 모르겠더라. 너는 어때?",
            "나? 나는 아직 잘 모르겠어. 그래서 네 의견이 궁금한 건데. 어떻게 생각해?",
            "솔직히 나도 정리가 안 됐어. 네가 어떻게 생각하는지 먼저 듣고 싶어."
        ]
        response = random.choice(responses)
        logger.info(f"✅ [Stage2] Responding to ask_opinion with: {response}")
        return response

    @staticmethod
    def generate_acknowledgment_and_transition(llm: ChatOpenAI, messages: list, prompts: Dict[str, str]) -> str:
        """
        Stage 1 → Stage 2 전환 시: 이전 대화 기반 짧은 공감 + Stage 2 고정 질문
        ✨ Stage 1에서 작품활동 감지 시 호출되는 static 메서드

        Args:
            llm: LLM 인스턴스
            messages: 전체 대화 기록 (컨텍스트)
            prompts: 프롬프트 딕셔너리

        Returns:
            짧은 공감 + Stage 2 고정 질문
        """
        return AcknowledgmentGenerator.generate(llm, messages, prompts)


    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2 처리: 죽은 사람을 AI로 재현하는 기술에 대한 입장 질문

        의도: 죽은 사람을 AI로 재현하는 기술에 대한 입장 물어보기 (구체적 이유 불필요)

        첫 방문: 이전 대화 히스토리 기반 공감 + 고정된 질문
        재방문:
          - 명확한 의견 → Stage 3로 전환
          - 모르겠다고 함 → 설명 제공 후 다시 질문
          - 무관한 답변 → 다른 표현으로 재질문

        Args:
            state: 현재 대화 상태

        Returns:
            업데이트된 대화 상태
        """
        previous_stage = state.get("previous_stage", "")
        messages = state.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        stage2_question_asked = state.get("stage2_question_asked", False)

        # ⏭ Stage2가 이미 완료된 경우: 중복 질문 방지 및 Stage3 첫 질문으로 스킵
        if state.get("stage2_completed", False):
            logger.info("⏭ Stage 2 already completed - skipping duplicate question")
            # Stage 3 유지 및 첫 질문 설정
            state["stage"] = "stage3"
            state["previous_stage"] = "stage2"
            state["current_question_index"] = 0
            state["dont_know_count"] = 0

            # Stage3 첫 질문만 간단히 제시 (추가 전환 멘트 없이)
            response = self.transition_generator.STAGE3_FIRST_QUESTION

            # 일관성 유지용 플래그들 보강
            state["stage2_complete"] = True
            state["stage2_question_asked"] = True
            state["last_response"] = response
            state["messages"].append({"role": "assistant", "content": response})
            state["message_count"] = state.get("message_count", 0) + 1

            return state

        # 첫 방문 여부 체크
        if previous_stage != "stage2" and not stage2_question_asked:
            # 첫 방문: 죽은 사람을 AI로 재현하는 기술에 대한 입장 질문
            response = self.first_question_generator.generate(messages)
            stage2_complete = False
            logger.info("Stage 2 first visit - asking initial question")
        else:
            # ✨ 재방문: LLM이 사용자 의도 파악 (이전 대화 컨텍스트 포함)
            intent = self.intent_detector.detect(user_message, messages)
            logger.info(f"Stage 2 revisit - detected intent: {intent}")

            needs_explanation = self._needs_explanation(user_message, messages)

            if intent == "ask_opinion":
                # ✨ 에이전트에게 의견을 물음 → "잘 모르겠다"고 답변
                response = self._handle_ask_opinion(user_message, messages)
                stage2_complete = False
                logger.info("⚠️ User asked for agent's opinion - responding with 'I don't know'")
            elif intent == "opinion":
                # 충분한 의견 → Stage 3로 전환
                response = self.transition_generator.generate(user_message)
                stage2_complete = True
                # ✨ Stage 3의 첫 번째 질문 설정 (인덱스 0)
                state["current_question_index"] = 0
                state["dont_know_count"] = 0
                # 보수적 안전장치: Stage2 완료 플래그를 별도로 기록 (중복 방문 방지)
                state["stage2_completed"] = True
                logger.info("✅ User gave opinion - transitioning to Stage 3 with first question")
            elif intent == "dont_know" or needs_explanation:
                # 모르겠거나 이유를 물으면 상황을 부연 설명 후 같은 질문 반복
                # ✨ messages 전달하여 대화 맥락 유지 + 반복 방지
                response = self.explanation_generator.generate(user_message, messages)
                stage2_complete = False
                logger.info("ℹ️ User needs explanation - providing context before reasking")
            else:
                # 부족한 답변/무관한 답변 → 다른 표현으로 재질문
                response = self.rephrase_question_generator.generate(messages)
                stage2_complete = False
                logger.info("⚠️ Unclear response - rephrasing question")

        # ✨ Stage 2 완료 시 Stage 3로 전환
        if stage2_complete:
            state["stage"] = "stage3"
        else:
            state["stage"] = "stage2"


        state["previous_stage"] = "stage2"
        state["stage2_complete"] = stage2_complete
        state["stage2_question_asked"] = True  # ✨ 질문을 물어봤음을 기록
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        return state
