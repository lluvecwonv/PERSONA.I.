"""
사용자 의도 감지 모듈
"""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)

_QUESTION_CHARS = ("?", "？")
_REASON_KEYWORDS_STRICT = (
    "왜냐하면",
    "왜냐면",
    "왜냐하",
    "그덕분에",
    "덕분에",
    "덕분이라",
    "덕분이야",
    "덕분이지",
)
_REASON_KEYWORDS_ALLOW_TRAILING = (
    "기때문에",
    "것때문에",
    "거때문에",
    "걸때문에",
    "땜에",
    "때문이라",
    "때문이라서",
    "때문이야",
    "때문이지",
    "때문인걸",
    "때문인거야",
    "때문일걸",
    "때문일거야",
    "때문탓에",
    "탓에",
    "거라서",
    "거니까",
    "거거든",
    "거거든요",
    # ✨ 추가: ~니까, ~라서, ~해서 패턴
    "니까",
    "으니까",
    "이니까",
    "라서",
    "이라서",
    "해서",
    "아서",
    "어서",
    "잖아",
    "거든",
)
_REASON_KEYWORDS_NEED_FOLLOW = (
    "그래서",
    "그러니까",
    "그렇다보니",
    "그렇다보니까",
    "그렇기때문에",
)
_EN_REASON_KEYWORDS = (
    "because",
    "since",
    "so that",
    "that's why",
    "due to",
)
_QUESTION_PREFIXES_NEAR_REASON = (
    "무슨",
    "뭐",
    "어떤",
    "누가",
    "누구",
    "어디",
)

# ✨ dont_know 패턴 (답을 모르겠음) - "왜 모르겠어?" 질문을 위해
_DONT_KNOW_PATTERNS = (
    "모르겠어",
    "모르겟어",  # 오타 변형
    "모르겠어요",
    "잘 모르겠어",
    "잘모르겠어",
    "잘 모르겠어요",
    "잘모르겠어요",
    "글쎄",
    "글세",  # 오타 변형
    "글쎄요",
    "몰라",
    "모르겠는데",
    "모르겟는데",  # 오타 변형
    "생각이 안 나",
    "생각안나",
    "어?",  # 짧은 혼란 표현
    "?",  # 단독 물음표
)


class IntentDetector:
    """사용자 의도 감지 클래스"""

    def __init__(self, analyzer: ChatOpenAI, prompts: dict):
        """
        Args:
            analyzer: 의도 분석용 LLM (temperature=0)
            prompts: 프롬프트 딕셔너리
        """
        self.analyzer = analyzer
        self.prompts = prompts

    def detect(self, user_message: str, current_question: str = "", context: str = "") -> str:
        """
        사용자가 질문을 이해했는지 판단 (100% LLM 기반)

        Args:
            user_message: 사용자 메시지
            current_question: 현재 질문
            context: 최근 대화 히스토리

        Returns:
            의도:
            - "answer": 질문에 대한 답변 (이유 포함)
            - "need_reason": 의견만 있고 이유 없음 → "왜 그렇게 생각하세요?" 질문
            - "ask_concept": 개념 설명 요청 (예: "자율성이 뭐야?")
            - "clarification": 질문 이해 못함
            - "ask_why_unsure": 모르겠다고 함 → "왜 모르겠어?" 질문
            - "ask_opinion": 에이전트에게 의견 질문
        """
        # ✨ 모든 의도 분류는 LLM이 담당 (휴리스틱 제거)
        intent_prompt = format_prompt(
            self.prompts.get("detect_intent", ""),
            user_message=user_message,
            current_question=current_question,
            context=context
        )

        try:
            logger.info(f"🔍 [Intent Detection] User message: '{user_message}'")
            logger.info(f"🔍 [Intent Detection] Prompt length: {len(intent_prompt)} chars")

            result = self.analyzer.invoke(intent_prompt)
            intent = result.content.strip().lower()

            logger.info(f"🔍 [Intent Detection] LLM raw response: '{intent}'")

            # 의도 분류
            if "ask_opinion" in intent:
                logger.info(f"🔍 [Intent Detection] 💬 RESULT: ask_opinion (user asks for agent's opinion)")
                return "ask_opinion"
            elif "ask_concept" in intent:
                logger.info(f"🔍 [Intent Detection] 📚 RESULT: ask_concept (will explain concept)")
                return "ask_concept"
            elif "need_reason" in intent:
                if self._contains_reason_statement(user_message):
                    logger.info(
                        "🔍 [Intent Detection] need_reason flagged but reasoning markers detected → override to answer"
                    )
                    return "answer"
                logger.info("🔍 [Intent Detection] 🔁 RESULT: need_reason (ask for reasoning)")
                return "need_reason"
            elif "dont_know" in intent or "dont know" in intent:
                # ✨ "모르겠어" 같은 불확실한 응답 → "왜 모르겠어?" 질문
                logger.info(f"🔍 [Intent Detection] ⚠️ RESULT: ask_why_unsure (user is unsure)")
                return "ask_why_unsure"
            elif "clarification" in intent:
                logger.info(f"🔍 [Intent Detection] ❌ RESULT: clarification (will repeat question)")
                return "clarification"
            else:
                # 모든 일반 답변은 "answer"로 처리
                logger.info(f"🔍 [Intent Detection] ✅ RESULT: answer (will move to next question)")
                return "answer"
        except Exception as e:
            logger.error(f"❌ Intent detection error: {e}")
            return "answer"  # 기본값: 다음 질문으로

    @staticmethod
    def _contains_reason_statement(text: str) -> bool:
        """
        간단한 휴리스틱: 사용자가 이미 이유를 설명했으면 need_reason 분기를 막는다.
        """
        if not text:
            return False

        if any(ch in text for ch in _QUESTION_CHARS):
            return False

        lowered = text.lower()
        normalized = lowered.replace(" ", "").replace("\n", "").replace("\t", "")

        for keyword in _REASON_KEYWORDS_STRICT:
            if keyword in normalized:
                return True

        for keyword in _REASON_KEYWORDS_ALLOW_TRAILING:
            if keyword in normalized:
                return True

        for keyword in _REASON_KEYWORDS_NEED_FOLLOW:
            idx = normalized.find(keyword)
            if idx != -1 and idx + len(keyword) < len(normalized):
                return True

        for keyword in _EN_REASON_KEYWORDS:
            normalized_keyword = keyword.replace(" ", "")
            if keyword in lowered or normalized_keyword in normalized:
                return True

        if IntentDetector._matches_plain_ttaemun_reason(normalized):
            return True

        return False

    @staticmethod
    def _matches_plain_ttaemun_reason(normalized: str) -> bool:
        """'~ 때문에' 패턴을 탐지하되 의문사 근처에서는 제외한다."""
        keyword = "때문에"
        start = 0
        while True:
            idx = normalized.find(keyword, start)
            if idx == -1:
                return False
            if not IntentDetector._has_nearby_question_prefix(normalized, idx):
                return True
            start = idx + len(keyword)

    @staticmethod
    def _has_nearby_question_prefix(normalized: str, idx: int) -> bool:
        for token in _QUESTION_PREFIXES_NEAR_REASON:
            token_idx = normalized.rfind(token, 0, idx)
            if token_idx == -1:
                continue
            distance = idx - token_idx
            if distance <= len(token) + 2:
                return True
        return False
