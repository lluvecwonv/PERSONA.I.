"""Stage 3 intent detection: classify user response to ethics questions."""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)

_QUESTION_CHARS = ("?", "？")
_REASON_KEYWORDS_STRICT = (
    "왜냐하면", "왜냐면", "왜냐하",
    "그덕분에", "덕분에", "덕분이라", "덕분이야", "덕분이지",
)
_REASON_KEYWORDS_ALLOW_TRAILING = (
    "기때문에", "것때문에", "거때문에", "걸때문에", "땜에",
    "때문이라", "때문이라서", "때문이야", "때문이지",
    "때문인걸", "때문인거야", "때문일걸", "때문일거야",
    "때문탓에", "탓에", "거라서", "거니까", "거거든", "거거든요",
    "니까", "으니까", "이니까", "라서", "이라서",
    "해서", "아서", "어서", "잖아", "거든",
)
_REASON_KEYWORDS_NEED_FOLLOW = (
    "그래서", "그러니까", "그렇다보니", "그렇다보니까", "그렇기때문에",
)
_EN_REASON_KEYWORDS = (
    "because", "since", "so that", "that's why", "due to",
)
_QUESTION_PREFIXES_NEAR_REASON = (
    "무슨", "뭐", "어떤", "누가", "누구", "어디",
)

# Typo variants included intentionally
_DONT_KNOW_PATTERNS = (
    "모르겠어", "모르겟어", "모르겠어요",
    "잘 모르겠어", "잘모르겠어", "잘 모르겠어요", "잘모르겠어요",
    "글쎄", "글세", "글쎄요", "몰라",
    "모르겠는데", "모르겟는데",
    "생각이 안 나", "생각안나",
    "어?", "?",
)


class IntentDetector:
    def __init__(self, analyzer: ChatOpenAI, prompts: dict):
        self.analyzer = analyzer
        self.prompts = prompts

    def detect(self, user_message: str, current_question: str = "", context: str = "") -> str:
        """LLM-based intent classification. Returns one of: answer, need_reason,
        ask_concept, clarification, ask_why_unsure, ask_opinion, ask_explanation."""
        intent_prompt = format_prompt(
            self.prompts.get("detect_intent", ""),
            user_message=user_message,
            current_question=current_question,
            context=context
        )

        try:
            logger.info(f"[Intent Detection] User: '{user_message}', prompt_len: {len(intent_prompt)}")

            result = self.analyzer.invoke(intent_prompt)
            intent = result.content.strip().lower()

            logger.info(f"[Intent Detection] LLM response: '{intent}'")

            if "ask_opinion" in intent:
                logger.info(f"[Intent Detection] -> ask_opinion")
                return "ask_opinion"
            elif "ask_concept" in intent:
                logger.info(f"[Intent Detection] -> ask_concept")
                return "ask_concept"
            elif "need_reason" in intent:
                # Override if user already provided reasoning markers
                if self._contains_reason_statement(user_message):
                    logger.info("[Intent Detection] need_reason overridden -> answer (reasoning markers found)")
                    return "answer"
                logger.info("[Intent Detection] -> need_reason")
                return "need_reason"
            elif "dont_know" in intent or "dont know" in intent:
                logger.info(f"[Intent Detection] -> ask_why_unsure")
                return "ask_why_unsure"
            elif "clarification" in intent:
                logger.info(f"[Intent Detection] -> clarification")
                return "clarification"
            elif "ask_explanation" in intent:
                logger.info(f"[Intent Detection] -> ask_explanation")
                return "ask_explanation"
            else:
                logger.info(f"[Intent Detection] -> answer (default)")
                return "answer"
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return "answer"

    @staticmethod
    def _contains_reason_statement(text: str) -> bool:
        """Heuristic check: returns True if user already provided reasoning."""
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
        """Detect '~ 때문에' pattern but exclude when near interrogative prefixes."""
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
