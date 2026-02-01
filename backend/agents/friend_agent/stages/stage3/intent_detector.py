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

    def detect(
        self,
        user_message: str,
        question_index: int = 0,
        dont_know_count: int = 0,
        context: str = "",
        next_topic_question: str = ""
    ) -> str:
        """
        사용자의 의도를 판단 (휴리스틱 + LLM 기반)

        Args:
            user_message: 사용자 메시지
            question_index: 현재 질문 인덱스
            dont_know_count: 모르겠다 표현 횟수
            context: 최근 대화 히스토리
            next_topic_question: 다음에 물어볼 질문

        Returns:
            의도:
            - "answer": 질문에 대한 답변
            - "ask_concept": 개념 설명 요청 (예: "자율성이 뭐야?")
            - "clarification": 질문 이해 못함 / 모르겠다
            - "dont_know_second": 동일 질문에서 두 번째 모름/망설임
            - "unrelated": 무관한 응답
            - "ask_opinion": 의견을 물어봄
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
                # dont_know_count에 따라 ask_why_unsure 또는 dont_know_second 반환
                if dont_know_count >= 1:
                    return "dont_know_second"
                else:
                    # ✨ "글세", "모르겠어" 첫 번째 → "왜 모르겠어?" 질문
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
            current_topic="",  # 더 이상 사용하지 않음
            dont_know_count=dont_know_count,
            context=context,
            next_topic_question=next_topic_question
        )

        try:
            result = self.analyzer.invoke(intent_prompt)
            intent = result.content.strip().lower()
            logger.info(f"LLM intent detection result: '{intent}' for message: {user_message[:50]}...")

            final_intent = "answer"
            if "ask_concept" in intent:
                logger.info("Detected intent: ask_concept")
                final_intent = "ask_concept"
            elif "need_reason" in intent:
                if self._contains_reason_statement(user_message):
                    logger.info(
                        "Detected intent: need_reason but found explicit reasoning markers → override to answer"
                    )
                    final_intent = "answer"
                else:
                    logger.info("Detected intent: need_reason (follow-up required)")
                    final_intent = "need_reason"
            elif "clarification" in intent:
                logger.info("Detected intent: clarification")
                final_intent = "clarification"
            elif "dont_know_second" in intent:
                logger.info("Detected intent: dont_know_second")
                final_intent = "dont_know_second"
            elif "unrelated" in intent:
                logger.info("Detected intent: unrelated")
                final_intent = "unrelated"
            elif "ask_opinion" in intent:
                logger.info("Detected intent: ask_opinion")
                final_intent = "ask_opinion"
            else:
                # 일부 모델은 "answer" 이외의 텍스트를 반환할 수 있으므로 안전하게 처리
                logger.info(f"Detected intent: answer (default for '{intent}')")
                final_intent = "answer"

            # ✨ LLM이 dont_know_second를 반환해도, dont_know_count=0이면 ask_why_unsure로 변환
            if final_intent == "dont_know_second" and dont_know_count == 0:
                logger.info("LLM returned dont_know_second but count=0 → switching to ask_why_unsure")
                return "ask_why_unsure"

            # ✨ 짧은 답변이면서 이유가 없으면 need_reason으로 전환
            if final_intent == "answer":
                if self._is_positive_response(user_message):
                    # "웅 도움 될 것 같아" 같은 긍정 답변 → need_reason으로
                    logger.info("Detected positive response without reason → switching to need_reason")
                    return "need_reason"
                elif self._needs_reason_follow_up(user_message):
                    logger.info("Detected short answer without explicit reasoning → switching to need_reason")
                    return "need_reason"

            return final_intent
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
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
    def _needs_reason_follow_up(text: str) -> bool:
        """
        짧은 단정형 답변이면서 이유 표현이 없으면 추가 근거를 물어보도록 표시
        ✨ 너무 짧은 경우(10자 이하)만 need_reason으로 처리
        """
        if not text:
            return False

        stripped = text.strip()
        if not stripped:
            return False

        # ✨ 이유가 포함되어 있으면 False (더 이상 물어보지 않음)
        if IntentDetector._contains_reason_statement(stripped):
            return False

        # ✨ 10자 이하의 매우 짧은 답변만 need_reason으로 처리
        # (예: "응", "그래", "맞아" 등)
        if len(stripped) > 10:
            return False

        return True

    @staticmethod
    def _is_positive_response(text: str) -> bool:
        """
        긍정적 답변인지 확인 (예: "웅 도움 될 것 같아", "그럴 것 같아")
        이런 답변은 이유를 물어봐야 함
        """
        if not text:
            return False

        normalized = text.replace(" ", "").lower()

        # 긍정 답변 패턴
        positive_patterns = (
            "도움될것같",
            "도움이될것같",
            "도움줄것같",
            "도움을줄것같",
            "좋을것같",
            "괜찮을것같",
            "그럴것같",
            "그런것같",
            "맞는것같",
            "맞을것같",
            "될것같",
            "할것같",
            "있을것같",
            "도움되겠",
            "도움이되겠",
            "도움줄거같",
            "도움될거같",
            "좋을거같",
            "괜찮을거같",
            "그럴거같",
            "될거같",
        )

        # 이유가 이미 포함되어 있으면 False
        if IntentDetector._contains_reason_statement(text):
            return False

        for pattern in positive_patterns:
            if pattern in normalized:
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
