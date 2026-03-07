"""Stage 2 intent detection: did the user express an opinion on AI art?"""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class IntentDetector:
    def __init__(self, analyzer: ChatOpenAI, prompts: dict):
        self.analyzer = analyzer
        self.prompts = prompts

    def detect(self, user_message: str, messages: list = None) -> str:
        """Returns "opinion", "unclear", or "dont_know"."""
        conversation_context = ""
        if messages:
            recent_messages = messages[-4:] if len(messages) > 4 else messages
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
            logger.info(f"[Stage2 IntentDetector] User: '{user_message}', context_len: {len(conversation_context)}")

            result = self.analyzer.invoke(intent_prompt)
            decision = result.content.strip().lower()

            logger.info(f"[Stage2 IntentDetector] LLM response: '{decision}'")

            if "opinion" in decision:
                logger.info(f"[Stage2 IntentDetector] OPINION -> Stage 3")
                return "opinion"
            elif "dont_know" in decision or "dont know" in decision:
                logger.info(f"[Stage2 IntentDetector] DONT_KNOW -> explanation needed")
                return "dont_know"
            else:
                logger.info(f"[Stage2 IntentDetector] UNCLEAR -> rephrase")
                return "unclear"
        except Exception as e:
            logger.error(f"[Stage2 IntentDetector] Error: {e}")
            return "unclear"
