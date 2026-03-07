"""Intent detection for Stage 2 - classify user stance on AI art."""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class IntentDetector:

    OPINION_KEYWORDS = (
        "원하", "원해", "싶어", "하고싶", "만나고", "만날거", "만나고싶",
        "찬성", "반대", "괜찮", "필요", "해야", "하고싶다", "사용하",
        "쓰고싶", "믿어", "믿고싶", "좋아", "싫어", "할래", "할거야"
    )
    YES_RESPONSES = ("응", "웅", "그래", "맞아", "좋아", "당연")
    DONT_KNOW_KEYWORDS = ("모르겠", "모르겠어", "몰라", "글쎄", "잘모르", "잘 모르")
    ASK_OPINION_KEYWORDS = ("너는", "넌", "네생각", "네 생각", "너생각", "너 생각", "니생각", "넌어떻게", "너는어떻게", "넌 어떻게", "너는 어떻게")

    def __init__(self, analyzer: ChatOpenAI, prompts: dict):
        self.analyzer = analyzer
        self.prompts = prompts

    def detect(self, user_message: str, messages: list = None) -> str:
        """Returns: "opinion", "unclear", "dont_know", or "ask_opinion"."""
        if not user_message:
            return "unclear"

        normalized = user_message.strip().lower().replace(" ", "")

        if any(keyword in normalized for keyword in self.ASK_OPINION_KEYWORDS):
            logger.info(f"[Stage2Intent] Heuristic ask_opinion: {user_message}")
            return "ask_opinion"

        if any(keyword in normalized for keyword in self.DONT_KNOW_KEYWORDS):
            logger.info(f"[Stage2Intent] Heuristic dont_know: {user_message}")
            return "dont_know"

        if normalized in self.YES_RESPONSES:
            logger.info(f"[Stage2Intent] Heuristic opinion (yes-response): {user_message}")
            return "opinion"

        if any(keyword in normalized for keyword in self.OPINION_KEYWORDS):
            logger.info(f"[Stage2Intent] Heuristic opinion keyword: {user_message}")
            return "opinion"

        conversation_context = ""
        if messages:
            recent_messages = messages[-8:] if len(messages) > 8 else messages
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_messages
            ])

        intent_prompt = format_prompt(
            self.prompts.get("stage2_detect_intent", ""),
            user_message=user_message,
            conversation_context=conversation_context
        )

        try:
            result = self.analyzer.invoke(intent_prompt)
            decision = result.content.strip().lower()

            if "opinion" in decision:
                logger.info(f"Detected opinion: {user_message}")
                return "opinion"
            elif "dont_know" in decision or "dont know" in decision:
                logger.info(f"Detected dont_know: {user_message}")
                return "dont_know"
            else:
                logger.info(f"Detected unclear: {user_message}")
                return "unclear"
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return "unclear"
