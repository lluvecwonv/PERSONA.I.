"""
Routing logic for conversation flow
✨ Stage 1 → Stage 2: LLM 기반 플래그 확인 (Stage1Handler에서 판단)
✨ Stage 2 → Stage 3: LLM 의도 분류 (자연스러움)
✨ Stage 3: 플래그 기반 (빠름)
"""
from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ConversationRouter:
    """대화 흐름 라우팅 핸들러 - 3단계 구조"""

    def __init__(self, analyzer: ChatOpenAI):
        """
        라우터 초기화

        Args:
            analyzer: 의도 분류용 LLM (temperature=0)
        """
        self.analyzer = analyzer

    def route_from_stage1(self, state: Dict[str, Any]) -> str:
        """
        Stage 1에서 Stage 2로 전환
        ✨ 플래그 기반: Stage1Handler에서 LLM으로 이미 판단한 결과 확인
        ✨ 카운터 기반 강제 전환: 3회 이상 반복되면 자동으로 Stage2로 넘어감 (무한 루프 방지)

        Stage1Handler가 LLM을 사용하여 "작품활동에 대한 구체적 설명을 했는가?"를
        판단하고 artist_character_set 플래그를 설정합니다.

        Args:
            state: 현재 대화 상태

        Returns:
            다음 스테이지 ("stage1" 또는 "stage2")
        """
        # Stage1Handler에서 설정한 플래그 확인 (LLM 중복 호출 방지)
        artist_character_set = state.get("artist_character_set", False)
        stage1_attempts = state.get("stage1_attempts", 0)

        # 1. 플래그가 True이면 Stage2로 전환
        if artist_character_set:
            logger.info("Stage 1→2 전환: 화가 캐릭터 설정 완료")
            return "stage2"

        # 2. 카운터 기반 강제 전환 (무한 루프 방지)
        if stage1_attempts >= 3:
            logger.warning(f"Stage 1→2 강제 전환: {stage1_attempts}회 시도 후 자동 전환")
            return "stage2"

        return "stage1"

    def route_from_stage2(self, state: Dict[str, Any]) -> str:
        """
        Stage 2에서 Stage 3로 전환
        ✨ 플래그 기반: Stage2Handler에서 판단한 stage2_complete 플래그 사용

        Args:
            state: 현재 대화 상태

        Returns:
            다음 스테이지 ("stage2" 또는 "stage3")
        """
        # Stage2Handler에서 설정한 플래그 확인
        stage2_complete = state.get("stage2_complete", False)

        if stage2_complete:
            logger.info("Stage 2→3 전환: 사용자가 명확한 의견을 냄")
            return "stage3"
        else:
            logger.info("Stage 2 유지: 사용자 의견 대기 중")
            return "stage2"

    def route_from_stage3(self, state: Dict[str, Any]) -> str:
        """
        Stage 3에서 계속 또는 종료 결정
        ✨ 최적화: stage3_handler에서 이미 판단한 플래그만 확인 (LLM 호출 X)

        Args:
            state: 현재 대화 상태

        Returns:
            다음 스테이지 ("stage3" 또는 "end")
        """
        # stage3_handler에서 이미 판단해서 저장한 플래그 사용
        # → LLM 중복 호출 제거, 훨씬 빠름!
        should_end = state.get("should_end", False)

        if should_end:
            return "end"

        return "stage3"
