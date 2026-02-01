"""
사용자 의도 감지 모듈 (Stage 3용)
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
    "니까",
    "니까요",
    "이니까",
    "이니까요",
    "으니까",
    "으니까요",
    "라서",
    "라서요",
    "이라서",
    "이라서요",
    "해서",
    "해서요",
    # ✨ ~잖아 패턴 추가 (이유/근거 표현)
    "잖아",
    "잖아요",
    "잔아",  # 오타 변형
    "수있잖아",
    "수있으니까",
    "수있어서",
    "가능하잖아",
    "되잖아",
    "되니까",
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

# ✨ clarification 패턴 (질문을 이해 못함)
_CLARIFICATION_PATTERNS = (
    "무슨 말이야",
    "무슨말이야",
    "뭔 말이야",
    "뭔말이야",
    "뭔 소리야",
    "뭔소리야",
    "무슨 소리야",
    "무슨소리야",
    "무슨 뜻이야",
    "무슨뜻이야",
    "뭔 뜻이야",
    "뭔뜻이야",
    "이해가 안 돼",
    "이해가안돼",
    "이해 못 했어",
    "이해못했어",
    "뭐라고?",
    "뭐라고",
    "뭐라는 거야",
    "뭐라는거야",
    "무슨 얘기야",
    "무슨얘기야",
    "뭔 얘기야",
    "뭔얘기야",
    "그게 뭐야",
    "그게뭐야",
    "뭐야",
)

# ✨ dont_know 패턴 (답을 모르겠음)
_DONT_KNOW_PATTERNS = (
    "모르겠어",
    "모르겟어",  # 오타 변형
    "잘 모르겠어",
    "잘모르겠어",
    "잘모르겟어",  # 오타 변형
    "글쎄",
    "글세",  # 오타 변형
    "몰라",
    "모르겠는데",
    "모르겟는데",  # 오타 변형
    "생각이 안 나",
    "생각안나",
)

# ✨ 질문 되묻기 패턴 (에이전트 질문을 그대로 되묻는 표현 = 불확실)
_ECHO_QUESTION_PATTERNS = (
    "괜찮을까",
    "될까",
    "할까",
    "있을까",
    "없을까",
    "맞을까",
    "좋을까",
    "나쁠까",
    "그럴까",
    "아닐까",
)


class IntentDetector:
    """사용자 의도 감지 클래스 (Stage 3용) - 100% LLM 기반"""

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
            - "need_reason": 의견만 있고 이유 없음 → "왜 그렇게 생각해?" 질문
            - "ask_concept": 개념 설명 요청 (예: "자율성이 뭐야?")
            - "clarification": 질문 이해 못함
            - "ask_why_unsure": 모르겠다고 함 → "왜 모르겠어?" 질문
            - "ask_opinion": 에이전트에게 의견 질문
            - "ask_explanation": 에이전트 말에 대해 "왜?"라고 질문
        """
        # ✨ 휴리스틱 프리 체크 - LLM 호출 전에 명확한 패턴 먼저 처리
        normalized = user_message.replace(" ", "").lower()

        # ✨ ask_opinion 패턴 체크 (에이전트에게 의견을 물음) - 다른 패턴보다 우선!
        ask_opinion_patterns = ("너는", "넌", "네생각", "네 생각", "너생각", "너 생각", "니생각", "넌어떻게", "너는어떻게", "넌 어떻게", "너는 어떻게")
        for pattern in ask_opinion_patterns:
            if pattern.replace(" ", "") in normalized:
                logger.info(f"✅ Heuristic: ask_opinion pattern detected: '{pattern}' in '{user_message}'")
                return "ask_opinion"

        # clarification 패턴 체크 (질문을 이해 못함)
        for pattern in _CLARIFICATION_PATTERNS:
            if pattern.replace(" ", "") in normalized:
                logger.info(f"✅ Heuristic: clarification pattern detected: '{pattern}' in '{user_message}'")
                return "clarification"

        # dont_know 패턴 체크 (답을 모르겠음) → "왜 모르겠어?" 질문
        for pattern in _DONT_KNOW_PATTERNS:
            if pattern.replace(" ", "") in normalized:
                logger.info(f"✅ Heuristic: dont_know pattern detected: '{pattern}' in '{user_message}'")
                return "ask_why_unsure"

        # ✨ 질문 되묻기 패턴 체크 (짧은 "~까?" 질문 = 불확실 표현)
        # 예: "괜찮을까?", "될까?", "그럴까?" 등
        if user_message.strip().endswith("?") and len(user_message.strip()) <= 10:
            for pattern in _ECHO_QUESTION_PATTERNS:
                if pattern in normalized:
                    logger.info(f"✅ Heuristic: echo question pattern detected: '{pattern}' in '{user_message}'")
                    return "ask_why_unsure"

        # ✨ "~문제?", "~걱정?" 같은 짧은 우려 표현은 answer로 처리
        concern_keywords = ("문제", "걱정", "우려", "불안", "위험", "피해", "손해")
        if user_message.strip().endswith("?") and len(user_message.strip()) < 15:
            for keyword in concern_keywords:
                if keyword in normalized:
                    logger.info(f"✅ Heuristic: short concern answer detected: '{keyword}' in '{user_message}'")
                    return "answer"

        # ✨ 짧은 동의/반대 표현은 need_reason으로 처리 (이유를 물어봐야 함)
        short_agreement_patterns = (
            "그럴수도", "그럴 수도", "그런것같", "그런 것 같", "그럴것같", "그럴 것 같",
            "맞아", "맞는것같", "맞는 것 같", "그래", "응", "어", "그렇지",
            "아닐것같", "아닐 것 같", "아닌것같", "아닌 것 같",
        )
        if len(user_message.strip()) <= 15:
            for pattern in short_agreement_patterns:
                if pattern.replace(" ", "") in normalized:
                    logger.info(f"✅ Heuristic: short agreement/disagreement detected: '{pattern}' in '{user_message}' → need_reason")
                    return "need_reason"

        # ✨ 휴리스틱으로 처리 안 되면 LLM 호출
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
            elif "ask_explanation" in intent:
                logger.info(f"🔍 [Intent Detection] 💡 RESULT: ask_explanation (user asks why)")
                return "ask_explanation"
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
        간단한 휴리스틱으로 사용자가 이미 이유/근거를 제시했는지 확인한다.
        LLM이 need_reason을 반환했더라도 이유 연결어가 있으면 answer로 간주한다.
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
            # 질문형 “무슨/뭐/어떤/누가/누구/어디 + (한 글자) + 때문에” 패턴만 제외
            distance = idx - token_idx
            if distance <= len(token) + 2:
                return True
        return False
