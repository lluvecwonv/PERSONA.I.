"""
SPT Agent V2 (Simplified)
Simplified SPT agent - DST removed

Architecture:
User → [SPT Planner (LLM)] → Instruction → [Persona Agent]
"""
from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


def _load_prompt(filename: str) -> str:
    """Load prompt file from prompts folder"""
    prompt_path = Path(__file__).parent / "prompts" / filename
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load prompt {filename}: {e}")
        return ""


class SPTAgentV2:
    """
    SPT Agent V2: Simplified Architecture

    Flow:
    1. Analyze conversation_history + user_message
    2. LLM generates SPT strategy-based instruction
    3. Pass instruction to Persona Agent
    """

    def __init__(self, api_key: str, planner_model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API key
            planner_model: Model for SPT Planner
        """
        self.api_key = api_key
        self.llm = ChatOpenAI(
            model=planner_model,
            api_key=api_key,
            temperature=0.7,
            max_tokens=400
        )

        self.system_prompt = _load_prompt("spt_planner_system.txt")
        self.user_prompt_template = _load_prompt("spt_user_prompt.txt")

        logger.info("✅ SPT Agent V2 initialized (simplified, no DST)")

    def _format_history(self, history: List[Dict[str, str]], max_turns: int = 6) -> str:
        """Format conversation history"""
        if not history:
            return "(No conversation history)"

        recent = history[-max_turns:]
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Agent"
            content = msg.get("content", "")[:150]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    async def process(
        self,
        session_id: str,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        topic_context: str = ""
    ) -> Dict[str, Any]:
        """
        Process user message and generate Persona Agent instructions

        Args:
            session_id: Session ID
            user_message: User message
            conversation_history: Full conversation history
            topic_context: Topic context (e.g., "AI Art", "AI Resurrection")
        Returns:
            {
                "instruction": str,  # Instructions for Persona Agent
            }
        """
        logger.info(f"🧠 [SPT V2] Processing message for session {session_id}")

        history_text = self._format_history(conversation_history)

        user_prompt = self.user_prompt_template.format(
            topic_context=topic_context if topic_context else "Moral dilemma",
            history_text=history_text,
            user_message=user_message
        )

        try:
            result = await self.llm.ainvoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ])

            instruction = clean_gpt_response(result.content)
            logger.info(f"📝 [SPT V2] Generated instruction: {instruction[:100]}...")

            return {"instruction": instruction}

        except Exception as e:
            logger.error(f"❌ [SPT V2] Error: {e}")
            return {"instruction": "Empathize with user's statement and ask why they think that way."}

    def clear_session(self, session_id: str) -> None:
        """Clear session (No-op, DST removed)"""
        pass
