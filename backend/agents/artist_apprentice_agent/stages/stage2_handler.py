"""
Stage 2: AI 예술 입장 질문 단계
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
    """Stage 2: AI 예술 입장 질문 핸들러"""

    # ✨ Stage 2의 고정된 질문 (정확한 문구, 절대 변경 금지!)
    STAGE2_QUESTION = "그런데.. 기사 보셨어요? 이번에 우리나라에서 제일 큰 기업이 '그림 그리는 AI'를 만들었다는데, 이 AI의 그림이 국립현대예술관에 전시가 된대요. 어떻게 생각하시는지 여쭤보려고 왔어요."

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str], persona_prompt: str = "", analyzer: ChatOpenAI = None):
        self.llm = llm
        self.prompts = prompts
        self.persona_prompt = persona_prompt
        self.analyzer = analyzer if analyzer else llm

        # ✨ 모듈화된 생성기 초기화
        self.first_question_generator = FirstQuestionGenerator(llm, prompts)
        self.rephrase_question_generator = RephraseQuestionGenerator(llm, prompts)
        self.intent_detector = IntentDetector(self.analyzer, prompts)
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
                "선생님의 마지막 발화: {user_message}\n"
                "위 발화가 '왜?','어떻게?' 등 이유를 따져 묻거나 추가 설명을 요구하면 explain, "
                "그렇지 않으면 continue 라고만 답하세요."
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
        Stage 2 처리: AI 예술에 대한 입장 질문

        의도: AI 작품을 예술로 인정할지 입장 물어보기 (구체적 이유 불필요)

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

        # 🔍 DEBUG: Stage 2 Entry State
        logger.info(f"=" * 50)
        logger.info(f"🔍 [Stage2Handler.handle] ENTRY")
        logger.info(f"🔍 [Stage2] previous_stage={previous_stage}")
        logger.info(f"🔍 [Stage2] stage2_question_asked={stage2_question_asked}")
        logger.info(f"🔍 [Stage2] stage2_complete={state.get('stage2_complete')}")
        logger.info(f"🔍 [Stage2] stage2_completed={state.get('stage2_completed')}")
        logger.info(f"🔍 [Stage2] messages_count={len(messages)}")
        logger.info(f"🔍 [Stage2] user_message='{user_message}'")

        # ⏭ Stage2가 이미 완료된 경우: 중복 질문 방지 및 Stage3 첫 질문으로 스킵
        if state.get("stage2_completed", False):
            logger.info(f"🔍 [Stage2] BRANCH: stage2_completed=True → skip to Stage3")
            # Stage 3 유지 및 첫 주제 설정
            state["stage"] = "stage3"
            state["previous_stage"] = "stage2"
            state["current_asking_topic"] = "이익"

            # Stage3 첫 질문만 간단히 제시 (추가 전환 멘트 없이)
            response = self.transition_generator.STAGE3_FIRST_QUESTION

            # 일관성 유지용 플래그들 보강
            state["stage2_complete"] = True
            state["stage2_question_asked"] = True
            state["last_response"] = response
            state["messages"].append({"role": "assistant", "content": response})
            state["message_count"] = state.get("message_count", 0) + 1

            return state

        # ✨ Stage2 질문이 이미 나갔으면 (Stage1에서 전환 시 출력됨)
        # → 유저 응답에 짧은 공감 + 바로 Stage3로 전환 (의견 감지 없이!)
        if stage2_question_asked:
            logger.info(f"🔍 [Stage2] BRANCH: stage2_question_asked=True → transition to Stage3")

            # ✅ LLM이 사용자 답변에 맞게 공감 + Stage3 첫 질문 생성
            response = self.transition_generator.generate(user_message)
            logger.info(f"🔍 [Stage2] response='{response}'")

            # Stage2 완료, Stage3로 전환
            state["current_asking_topic"] = "이익"
            state["stage2_completed"] = True
            state["stage2_complete"] = True
            state["stage"] = "stage3"
            state["previous_stage"] = "stage2"

            state["last_response"] = response
            state["messages"].append({"role": "assistant", "content": response})
            state["message_count"] = state.get("message_count", 0) + 1

            logger.info(f"🔍 [Stage2] EXIT → stage3")
            logger.info(f"=" * 50)
            return state

        # ✨ Stage2 질문이 아직 안 나갔으면 (일반적으로 여기 오지 않음, Stage1에서 전환 시 출력됨)
        logger.info(f"🔍 [Stage2] BRANCH: stage2_question_asked=False → output fixed question")

        # 공감 생성 + Stage2 고정 질문 (AcknowledgmentGenerator.generate는 static method)
        response = AcknowledgmentGenerator.generate(self.llm, messages, self.prompts)
        logger.info(f"🔍 [Stage2] response='{response}'")

        # Stage2 질문 완료 (아직 Stage3로 전환하지 않음)
        state["stage2_question_asked"] = True
        state["previous_stage"] = "stage1"

        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["message_count"] = state.get("message_count", 0) + 1

        logger.info(f"🔍 [Stage2] EXIT → staying in stage2")
        logger.info(f"=" * 50)
        return state
