"""
SPT Controller / Gate
SPT Planner의 질문 후보를 받아서 실제로 질문을 던질지 결정

핵심 역할:
- DST 상태 확인
- 질문 후보 필터링
- 최종 응답 전략 결정
"""
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .dst import DialogueStateTracker, DialogueState, UserResponseStatus, SPTFrame

logger = logging.getLogger(__name__)


class ResponseStrategy(Enum):
    """응답 전략"""
    ASK_NEW_QUESTION = "ask_new_question"          # 새 질문 던지기
    REPHRASE_QUESTION = "rephrase_question"        # 질문 다시 표현
    CLARIFY_QUESTION = "clarify_question"          # 질문 설명
    ANSWER_USER_QUESTION = "answer_user_question"  # 사용자 질문에 답
    RECONNECT_TO_TOPIC = "reconnect_to_topic"      # 주제로 다시 연결
    ASK_REASONING = "ask_reasoning"                # 이유 물어보기
    ACKNOWLEDGE_AND_PROCEED = "acknowledge_proceed"  # 인정하고 넘어가기
    EMPATHIZE_AND_GUIDE = "empathize_guide"        # 공감하고 안내


@dataclass
class ControllerDecision:
    """Controller의 결정"""
    strategy: ResponseStrategy
    allow_question: bool
    reason: str
    suggested_content: Optional[str] = None
    fallback_question: Optional[str] = None


class SPTController:
    """
    SPT Controller: 질문 게이트키퍼

    "SPT Planner가 제안한 질문을 지금 던져도 되는가?"

    판단 흐름:
    1. DST에서 현재 상태 확인
    2. can_ask_new_question() 결과 확인
    3. 상태에 따른 응답 전략 결정
    4. 질문 후보 필터링 (반복 방지)
    """

    def __init__(self, dst: DialogueStateTracker):
        self.dst = dst

    def decide(
        self,
        session_id: str,
        user_message: str,
        question_candidates: List[str],
        question_keywords: List[str] = None
    ) -> ControllerDecision:
        """
        질문 후보에 대한 최종 결정

        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            question_candidates: SPT Planner가 제안한 질문 후보들
            question_keywords: 현재 질문의 핵심 키워드

        Returns:
            ControllerDecision: 응답 전략과 허용 여부
        """
        # Step 1: 현재 상태 가져오기 (이미 SPT V2에서 분석됨)
        state = self.dst.get_or_create_state(session_id)

        # Step 2: 질문 허용 여부 확인
        can_ask, reason = self.dst.can_ask_new_question(session_id)

        logger.info(f"🎯 [Controller] can_ask={can_ask}, reason={reason}, status={state.response_status}")

        # Step 3: 상태별 전략 결정
        if can_ask:
            # 질문 가능 → 후보 중에서 선택
            filtered_candidates = self._filter_candidates(state, question_candidates)

            if filtered_candidates:
                return ControllerDecision(
                    strategy=ResponseStrategy.ASK_NEW_QUESTION,
                    allow_question=True,
                    reason=reason,
                    suggested_content=filtered_candidates[0],
                    fallback_question=filtered_candidates[1] if len(filtered_candidates) > 1 else None
                )
            else:
                # 후보가 없거나 모두 필터됨
                return ControllerDecision(
                    strategy=ResponseStrategy.ACKNOWLEDGE_AND_PROCEED,
                    allow_question=False,
                    reason="no_valid_candidates"
                )

        # Step 4: 질문 불가 상황별 처리
        return self._decide_alternative_strategy(state, reason, question_candidates)

    def _decide_alternative_strategy(
        self,
        state: DialogueState,
        reason: str,
        question_candidates: List[str]
    ) -> ControllerDecision:
        """
        질문을 던지지 않아야 할 때의 대안 전략 결정
        """
        status = state.response_status

        # Case 1: 사용자가 이해 못함
        if status == UserResponseStatus.NEEDS_CLARIFICATION:
            return ControllerDecision(
                strategy=ResponseStrategy.CLARIFY_QUESTION,
                allow_question=False,
                reason=reason,
                suggested_content=f"이전 질문을 다르게 설명해야 함: {state.last_agent_question[:50]}..."
            )

        # Case 2: 사용자가 역질문
        if status == UserResponseStatus.ASKING_BACK:
            return ControllerDecision(
                strategy=ResponseStrategy.ANSWER_USER_QUESTION,
                allow_question=False,
                reason=reason,
                suggested_content="먼저 사용자 질문에 답하고, 이후 원래 질문으로 돌아가기"
            )

        # Case 3: 사용자가 모르겠다
        if status == UserResponseStatus.DONT_KNOW:
            return ControllerDecision(
                strategy=ResponseStrategy.EMPATHIZE_AND_GUIDE,
                allow_question=False,
                reason=reason,
                suggested_content="공감 표현 + 구체적 예시 제공 + 다시 질문 (variation)"
            )

        # Case 4: Off-topic 응답
        if status == UserResponseStatus.OFF_TOPIC:
            return ControllerDecision(
                strategy=ResponseStrategy.RECONNECT_TO_TOPIC,
                allow_question=False,
                reason=reason,
                suggested_content=f"사용자 발언 인정 + 원래 질문 다시 연결: {state.last_agent_question[:50]}...",
                fallback_question=state.last_agent_question
            )

        # Case 5: 단답 (이유 없음)
        if status == UserResponseStatus.SHORT_RESPONSE:
            return ControllerDecision(
                strategy=ResponseStrategy.ASK_REASONING,
                allow_question=False,
                reason=reason,
                suggested_content="왜 그렇게 생각하는지 물어보기"
            )

        # Default: 진행
        return ControllerDecision(
            strategy=ResponseStrategy.ACKNOWLEDGE_AND_PROCEED,
            allow_question=True,
            reason="default"
        )

    def _filter_candidates(
        self,
        state: DialogueState,
        candidates: List[str]
    ) -> List[str]:
        """
        질문 후보 필터링 (반복 방지)

        Args:
            state: 현재 대화 상태
            candidates: 질문 후보 목록

        Returns:
            필터링된 후보 목록
        """
        if not candidates:
            return []

        asked = set(state.asked_questions)
        filtered = []

        for candidate in candidates:
            # 이미 물어본 질문과 유사한지 체크 (간단한 중복 방지)
            is_duplicate = any(
                self._is_similar(candidate, asked_q)
                for asked_q in asked
            )

            if not is_duplicate:
                filtered.append(candidate)

        logger.info(f"🔍 [Controller] Filtered {len(candidates)} → {len(filtered)} candidates")
        return filtered

    @staticmethod
    def _is_similar(q1: str, q2: str, threshold: float = 0.7) -> bool:
        """
        두 질문이 유사한지 간단히 체크 (단어 중복 비율)
        """
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1 & words2
        union = words1 | words2

        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold

    def get_instruction_for_persona(
        self,
        decision: ControllerDecision,
        state: DialogueState
    ) -> str:
        """
        Persona Agent에게 전달할 지시사항 생성

        Args:
            decision: Controller 결정
            state: 현재 대화 상태

        Returns:
            Persona Agent 지시사항
        """
        strategy = decision.strategy

        instructions = {
            ResponseStrategy.ASK_NEW_QUESTION: (
                f"📌 [질문 허용]\n"
                f"다음 질문을 자연스럽게 표현하세요: {decision.suggested_content}\n"
                f"먼저 사용자 발언에 공감한 후 질문으로 연결하세요."
            ),
            ResponseStrategy.CLARIFY_QUESTION: (
                f"📌 [설명 필요]\n"
                f"사용자가 이해하지 못했습니다.\n"
                f"이전 질문을 쉽게 다시 설명하세요: {state.last_agent_question}\n"
                f"새로운 질문을 던지지 마세요."
            ),
            ResponseStrategy.ANSWER_USER_QUESTION: (
                f"📌 [사용자 질문 답변]\n"
                f"사용자가 질문했습니다. 먼저 간단히 답변하세요.\n"
                f"답변 후 원래 주제로 자연스럽게 돌아가세요.\n"
                f"새로운 질문을 던지지 마세요."
            ),
            ResponseStrategy.RECONNECT_TO_TOPIC: (
                f"📌 [주제 재연결]\n"
                f"사용자가 다른 얘기를 했습니다.\n"
                f"1. 먼저 사용자 발언을 짧게 인정하세요.\n"
                f"2. 원래 질문과 연결하세요: {state.last_agent_question}\n"
                f"완전히 새로운 질문을 던지지 마세요."
            ),
            ResponseStrategy.ASK_REASONING: (
                f"📌 [이유 요청]\n"
                f"사용자가 단답만 했습니다.\n"
                f"\"왜 그렇게 생각해?\" 같은 후속 질문을 하세요.\n"
                f"새로운 주제로 넘어가지 마세요."
            ),
            ResponseStrategy.EMPATHIZE_AND_GUIDE: (
                f"📌 [공감 + 안내]\n"
                f"사용자가 모르겠다고 했습니다.\n"
                f"1. 공감을 표현하세요 (\"어려운 질문이지\")\n"
                f"2. 구체적 예시를 들어주세요\n"
                f"3. 같은 질문을 다른 방식으로 다시 물어보세요"
            ),
            ResponseStrategy.ACKNOWLEDGE_AND_PROCEED: (
                f"📌 [진행]\n"
                f"사용자 발언에 자연스럽게 반응하고 대화를 이어가세요."
            ),
        }

        return instructions.get(strategy, "자연스럽게 대화를 이어가세요.")
