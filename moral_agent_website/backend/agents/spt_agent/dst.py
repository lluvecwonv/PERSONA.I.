"""
DST (Dialogue State Tracker)
대화 상태를 추적하고 "지금 질문해도 되는가?"를 판단

핵심 역할:
- 사용자가 이전 질문에 답했는지 추적
- 현재 SPT 단계(frame) 추적
- 질문 허용 여부 결정
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class SPTFrame(Enum):
    """SPT 대화 단계 (프레임)"""
    RAPPORT = "rapport"           # 라포 형성 단계
    EXPLORE = "explore"           # 탐색 단계 - 사용자 입장 파악
    CHALLENGE = "challenge"       # 도전 단계 - 새로운 관점 제시
    REFLECT = "reflect"           # 성찰 단계 - 자기 성찰 유도
    CLOSE = "close"               # 마무리 단계


class UserResponseStatus(Enum):
    """사용자 응답 상태"""
    ANSWERED = "answered"              # 질문에 답함
    OFF_TOPIC = "off_topic"            # 질문 무시하고 다른 얘기
    NEEDS_CLARIFICATION = "clarification"  # 질문을 이해 못함
    DONT_KNOW = "dont_know"            # 모르겠다
    ASKING_BACK = "asking_back"        # 역질문
    SHORT_RESPONSE = "short"           # 단답 (이유 없음)


@dataclass
class DialogueState:
    """대화 상태 데이터"""
    # 현재 프레임
    current_frame: SPTFrame = SPTFrame.RAPPORT

    # 마지막으로 에이전트가 던진 질문
    last_agent_question: str = ""

    # 사용자가 답했는지 여부
    user_answered: bool = False

    # 응답 상태 상세
    response_status: Optional[UserResponseStatus] = None

    # SPT 단계 (0~4)
    spt_stage: int = 0

    # 프레임 전환 허용 여부
    allow_frame_shift: bool = False

    # 모르겠다 횟수 (같은 질문에 대해)
    dont_know_count: int = 0

    # 이전 질문 목록 (반복 방지)
    asked_questions: List[str] = field(default_factory=list)

    # 사용자가 표현한 관점들
    user_perspectives: List[str] = field(default_factory=list)


class DialogueStateTracker:
    """
    DST: 대화 상태 추적기

    핵심 질문: "지금 이 대화에서 질문을 해도 되는가?"

    판단 기준:
    1. 사용자가 이전 질문에 답했는가?
    2. 사용자가 off-topic 응답을 했는가?
    3. 현재 프레임에서 질문이 적절한가?
    """

    # 질문 무시 패턴 (off_topic 감지용)
    OFF_TOPIC_INDICATORS = [
        # 질문과 무관하게 장점/이익만 얘기
        r'(돈.*벌|이익|수익|편리|편해|도움될|좋겠|신기)',
        # 기술 발전 논의로 빠짐
        r'(기술.*(발전|좋|대단)|혁신|진보|대단해)',
        # 경제적 이익 얘기
        r'(벌겠|많이\s*벌|돈\s*많)',
    ]

    # 모르겠다 패턴
    DONT_KNOW_PATTERNS = [
        r'^(글쎄|몰라|모르겠|잘\s*모르겠|모르겠어)',
        r'(모르겠|확실하지\s*않|잘\s*모르)',
    ]

    # 이해 못함 패턴
    CLARIFICATION_PATTERNS = [
        r'(무슨\s*(말|소리|뜻)|뭔\s*(말|소리|뜻))',
        r'(이해\s*(가|를|못)|뭐라고\?|뭐라는)',
    ]

    # 역질문 패턴
    ASKING_BACK_PATTERNS = [
        r'(너는|네\s*생각|어떻게\s*생각)',
        r'\?$',
    ]

    def __init__(self):
        self.states: Dict[str, DialogueState] = {}

    def get_or_create_state(self, session_id: str) -> DialogueState:
        """세션별 상태 가져오기 (없으면 생성)"""
        if session_id not in self.states:
            self.states[session_id] = DialogueState()
            logger.info(f"✅ [DST] Created new state for session: {session_id}")
        return self.states[session_id]

    def clear_state(self, session_id: str) -> None:
        """세션 상태 초기화"""
        if session_id in self.states:
            del self.states[session_id]
            logger.info(f"🗑️ [DST] Cleared state for session: {session_id}")

    def update_after_agent_turn(
        self,
        session_id: str,
        agent_response: str,
        asked_question: bool = False
    ) -> DialogueState:
        """
        에이전트 턴 후 상태 업데이트

        Args:
            session_id: 세션 ID
            agent_response: 에이전트 응답
            asked_question: 에이전트가 질문을 던졌는지 여부
        """
        state = self.get_or_create_state(session_id)

        if asked_question:
            # 질문을 던졌으면 사용자 응답 대기 상태로
            state.last_agent_question = agent_response
            state.user_answered = False
            state.response_status = None
            state.asked_questions.append(agent_response)
            logger.info(f"📝 [DST] Agent asked question, waiting for user answer")
        else:
            # 질문 없이 응답만 했으면 상태 유지
            logger.info(f"📝 [DST] Agent responded without question")

        return state

    def analyze_user_response(
        self,
        session_id: str,
        user_message: str,
        question_keywords: List[str] = None,
        last_agent_question: str = None
    ) -> DialogueState:
        """
        사용자 응답 분석 및 상태 업데이트

        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            question_keywords: 현재 질문의 핵심 키워드 (예: ["우려", "걱정", "불안"])
            last_agent_question: 마지막 에이전트 질문 (외부에서 전달)

        Returns:
            업데이트된 대화 상태
        """
        state = self.get_or_create_state(session_id)
        text = user_message.strip().lower()

        # 외부에서 마지막 질문이 전달되면 상태에 반영
        if last_agent_question and not state.last_agent_question:
            state.last_agent_question = last_agent_question
            logger.info(f"📝 [DST] Set last_agent_question from external: {last_agent_question[:50]}...")

        # 1. 이해 못함 체크 (최우선)
        for pattern in self.CLARIFICATION_PATTERNS:
            if re.search(pattern, text):
                state.response_status = UserResponseStatus.NEEDS_CLARIFICATION
                state.user_answered = False
                logger.info(f"🔍 [DST] User needs clarification")
                return state

        # 2. 역질문 체크
        for pattern in self.ASKING_BACK_PATTERNS:
            if re.search(pattern, text):
                state.response_status = UserResponseStatus.ASKING_BACK
                state.user_answered = False
                logger.info(f"🔍 [DST] User asking back")
                return state

        # 3. 모르겠다 체크
        for pattern in self.DONT_KNOW_PATTERNS:
            if re.search(pattern, text):
                state.response_status = UserResponseStatus.DONT_KNOW
                state.user_answered = False
                state.dont_know_count += 1
                logger.info(f"🔍 [DST] User doesn't know (count: {state.dont_know_count})")
                return state

        # 4. Off-topic 체크 (질문 키워드와 비교)
        if question_keywords:
            has_keyword = any(kw in text for kw in question_keywords)

            # 키워드 없이 다른 주제만 얘기
            if not has_keyword:
                for pattern in self.OFF_TOPIC_INDICATORS:
                    if re.search(pattern, text):
                        state.response_status = UserResponseStatus.OFF_TOPIC
                        state.user_answered = False
                        logger.info(f"🔍 [DST] User gave off-topic response")
                        return state

        # 5. 단답 체크
        if len(text) <= 5 and text not in ['응', '아니', '네', '예', '맞아']:
            state.response_status = UserResponseStatus.SHORT_RESPONSE
            state.user_answered = True  # 일단 답은 했음
            logger.info(f"🔍 [DST] User gave short response")
            return state

        # 6. 정상 답변
        state.response_status = UserResponseStatus.ANSWERED
        state.user_answered = True
        state.dont_know_count = 0  # 리셋
        logger.info(f"🔍 [DST] User answered the question")
        return state

    def can_ask_new_question(self, session_id: str) -> tuple[bool, str]:
        """
        새로운 질문을 던져도 되는지 판단

        Returns:
            (허용 여부, 이유)
        """
        state = self.get_or_create_state(session_id)

        # Case 1: 첫 대화 또는 질문 없었음
        if not state.last_agent_question:
            return True, "no_previous_question"

        # Case 2: 사용자가 답했음
        if state.user_answered:
            return True, "user_answered"

        # Case 3: 사용자가 이해 못함 → 질문 대신 설명 필요
        if state.response_status == UserResponseStatus.NEEDS_CLARIFICATION:
            return False, "needs_clarification"

        # Case 4: 사용자가 역질문 → 먼저 대답해야 함
        if state.response_status == UserResponseStatus.ASKING_BACK:
            return False, "user_asked_question"

        # Case 5: 사용자가 모르겠다고 함
        if state.response_status == UserResponseStatus.DONT_KNOW:
            # 2번 이상이면 다음 질문으로 넘어감
            if state.dont_know_count >= 2:
                return True, "max_dont_know_reached"
            return False, "user_uncertain"

        # Case 6: Off-topic 응답 → 이전 질문 다시 연결
        if state.response_status == UserResponseStatus.OFF_TOPIC:
            return False, "off_topic_response"

        # Case 7: 단답 → 이유 물어봐야 함
        if state.response_status == UserResponseStatus.SHORT_RESPONSE:
            return False, "need_reasoning"

        return True, "default_allow"

    def get_suggested_action(self, session_id: str) -> str:
        """
        현재 상태에서 권장 행동 반환

        Returns:
            권장 행동 문자열
        """
        can_ask, reason = self.can_ask_new_question(session_id)
        state = self.get_or_create_state(session_id)

        if can_ask:
            return "proceed_new_question"

        action_map = {
            "needs_clarification": "clarify_question",
            "user_asked_question": "answer_user_question",
            "user_uncertain": "provide_example_or_rephrase",
            "off_topic_response": "reconnect_to_question",
            "need_reasoning": "ask_for_reasoning",
        }

        return action_map.get(reason, "proceed_new_question")

    def advance_frame(self, session_id: str, new_frame: SPTFrame) -> DialogueState:
        """프레임 전환"""
        state = self.get_or_create_state(session_id)
        old_frame = state.current_frame
        state.current_frame = new_frame
        state.allow_frame_shift = False
        logger.info(f"🔄 [DST] Frame shift: {old_frame.value} → {new_frame.value}")
        return state

    def record_user_perspective(self, session_id: str, perspective: str) -> None:
        """사용자가 표현한 관점 기록"""
        state = self.get_or_create_state(session_id)
        state.user_perspectives.append(perspective)
        logger.info(f"📝 [DST] Recorded user perspective: {perspective[:50]}...")
