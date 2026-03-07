"""Flag-based routing between conversation stages (no LLM calls here)."""
from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ConversationRouter:

    def __init__(self, analyzer: ChatOpenAI):
        self.analyzer = analyzer

    def route_from_stage1(self, state: Dict[str, Any]) -> str:
        """Check flags set by Stage1Handler; force-advance after 1 retry."""
        artist_character_set = state.get("artist_character_set", False)
        stage1_attempts = state.get("stage1_attempts", 0)

        if artist_character_set:
            logger.info("Stage 1->2: character setup complete")
            return "stage2"

        # Forced transition after threshold (actual gate is in determine_stage >= 2)
        if stage1_attempts >= 1:
            logger.warning(f"Stage 1->2 forced: {stage1_attempts} attempts")
            return "stage2"

        return "stage1"

    def route_from_stage2(self, state: Dict[str, Any]) -> str:
        stage2_complete = state.get("stage2_complete", False)

        if stage2_complete:
            logger.info("Stage 2->3: user gave clear opinion")
            return "stage3"
        else:
            logger.info("Stage 2: awaiting user opinion")
            return "stage2"

    def route_from_stage3(self, state: Dict[str, Any]) -> str:
        """Uses pre-computed should_end flag (no LLM call)."""
        if state.get("should_end", False):
            return "end"
        return "stage3"
