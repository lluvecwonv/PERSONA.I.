"""
Colleague2 Agent - utilitarian stance, pro-AI art.
Three-Phase Architecture: Basic Reflection -> SPT Planner (conditional) -> Response
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import re
import sys
import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None

sys.path.append(str(Path(__file__).parent.parent))
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


def _load_reflection_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "reflection_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load reflection prompt: {e}")
        return ""


def _load_spt_planner_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "spt_planner_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load SPT planner prompt: {e}")
        return ""


def _load_response_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "response_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load response prompt: {e}")
        return ""


class Colleague2Agent:

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
    ):
        self.api_key = api_key
        self.model = model
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")

        self.session_store: Dict[str, ChatMessageHistory] = {}

        self.reflection_prompt = _load_reflection_prompt()
        if self.reflection_prompt:
            logger.info("Reflection prompt loaded successfully")

        self.spt_planner_prompt = _load_spt_planner_prompt()
        if self.spt_planner_prompt:
            logger.info("SPT Planner prompt loaded successfully")

        self.response_prompt = _load_response_prompt()
        if self.response_prompt:
            logger.info("Response prompt loaded successfully")

        if self.google_api_key:
            self.response_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.7,
                max_output_tokens=2000
            )
            logger.info("[Colleague2] Using Gemini for Phase 1 & 2, GPT-4o for SPT Planner")
        else:
            self.response_llm = None
            logger.warning("[Colleague2] GOOGLE_API_KEY not found, will use OpenAI fallback")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            logger.info(f"[Colleague2] Created new session: {session_id}")
        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"[Colleague2] Cleared session: {session_id}")
            return True
        return False

    def _format_history(self, messages: List[Dict[str, str]], limit: int = 14) -> str:
        if not messages:
            return "(No previous conversation)"

        recent = messages[-limit:] if len(messages) > limit else messages
        lines = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if content:
                speaker = "User" if role == "user" else "Agent"
                lines.append(f"{speaker}: {content}")

        return "\n".join(lines) if lines else "(No previous conversation)"

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, with fallback for truncated output."""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {e}")
            result = {}
            sq_match = re.search(r'"strategic_question"\s*:\s*"([^"]+)"', text)
            if sq_match:
                result["strategic_question"] = sq_match.group(1)

            st_match = re.search(r'"stakeholders"\s*:\s*\[([^\]]+)\]', text)
            if st_match:
                stakeholders = re.findall(r'"([^"]+)"', st_match.group(1))
                result["stakeholders"] = stakeholders

            bs_match = re.search(r'"blind_spot"\s*:\s*"([^"]+)"', text)
            if bs_match:
                result["blind_spot"] = bs_match.group(1)

            spt_match = re.search(r'"spt_required"\s*:\s*(true|false)', text, re.IGNORECASE)
            if spt_match:
                result["spt_required"] = spt_match.group(1).lower() == "true"

            ca_match = re.search(r'"context_analysis"\s*:\s*"([^"]+)"', text)
            if ca_match:
                result["context_analysis"] = ca_match.group(1)

            ut_match = re.search(r'"user_utterance_type"\s*:\s*"([^"]+)"', text)
            if ut_match:
                result["user_utterance_type"] = ut_match.group(1)

            if result:
                logger.info(f"Recovered partial JSON: {result}")
            else:
                logger.error(f"Could not recover any data from: {text[:200]}")

            return result
        except Exception as e:
            logger.error(f"Unexpected parsing error: {e}")
            return {}

    async def _perform_basic_reflection(
        self,
        messages: List[Dict[str, str]],
        last_user_msg: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Phase 1: Determine whether SPT planning is needed."""
        logger.info(f"[PHASE1_START] session_id={session_id}, Performing basic reflection...")

        history_text = self._format_history(messages)

        reflection_prompt = self.reflection_prompt.format(
            last_user_msg=last_user_msg,
            history=history_text
        )

        try:
            reflection_llm = ChatOpenAI(
                model="gpt-4o",
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=2000
            )

            result = await reflection_llm.ainvoke([
                HumanMessage(content=reflection_prompt)
            ])

            reflection_text = result.content.strip()
            logger.info(f"[PHASE1_OUTPUT] {reflection_text}")

            parsed = self._parse_json_response(reflection_text)

            spt_required = parsed.get("spt_required", False)
            context_analysis = parsed.get("context_analysis", "")
            user_utterance_type = parsed.get("user_utterance_type", "Question")

            logger.info(f"[SPT_REQUIRED] {spt_required}")
            logger.info(f"[PHASE1_COMPLETE] Basic reflection done")

            return {
                "spt_required": spt_required,
                "context_analysis": context_analysis,
                "user_utterance_type": user_utterance_type
            }

        except Exception as e:
            logger.error(f"[PHASE1_ERROR] {e}")
            return {
                "spt_required": False,
                "context_analysis": "Reflection failed",
                "user_utterance_type": "Question"
            }

    async def _perform_spt_planning(
        self,
        messages: List[Dict[str, str]],
        last_user_msg: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Phase 1.5: Run SPT planner (only called when spt_required=True)."""
        logger.info(f"[SPT_PLANNER_START] session_id={session_id}, Performing SPT planning...")

        history_text = self._format_history(messages)

        spt_prompt = self.spt_planner_prompt.format(
            last_user_msg=last_user_msg,
            history=history_text
        )

        try:
            # GPT-4o to avoid truncation
            spt_llm = ChatOpenAI(
                model="gpt-4o",
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=500
            )

            result = await spt_llm.ainvoke([
                HumanMessage(content=spt_prompt)
            ])

            spt_text = result.content.strip()
            logger.info(f"[SPT_PLANNER_OUTPUT] {spt_text}")

            parsed = self._parse_json_response(spt_text)

            strategic_question = parsed.get("strategic_question", "")
            stakeholders = parsed.get("stakeholders", [])
            blind_spot = parsed.get("blind_spot", "")

            logger.info(f"[STRATEGIC_QUESTION] {strategic_question}")
            logger.info(f"[SPT_PLANNER_COMPLETE] SPT planning done")

            formatted_section = ""
            if strategic_question:
                formatted_section = f"""=== SPT ANALYSIS (MUST USE) ===
Stakeholders: {', '.join(stakeholders)}
Blind spot user is missing: {blind_spot}
Strategic question to ask: {strategic_question}
=== END SPT ===

CRITICAL: You MUST include the strategic question in your response to guide the user's thinking."""

            return {
                "stakeholders": stakeholders,
                "empathy_analysis": parsed.get("empathy_analysis", ""),
                "stance_alignment": parsed.get("stance_alignment", ""),
                "blind_spot": blind_spot,
                "strategic_question": strategic_question,
                "formatted_section": formatted_section
            }

        except Exception as e:
            logger.error(f"[SPT_PLANNER_ERROR] {e}")
            return {
                "stakeholders": [],
                "empathy_analysis": "",
                "stance_alignment": "",
                "blind_spot": "",
                "strategic_question": "",
                "formatted_section": ""
            }

    def _create_llm(self, max_tokens: int, streaming: bool, temperature: float = 0.7) -> ChatOpenAI:
        kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "streaming": streaming,
        }
        # gpt-5 only supports temperature=1 and uses max_completion_tokens
        if "gpt-5" in self.model:
            kwargs["temperature"] = 1.0
            if max_tokens:
                kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    def _build_response_prompt(
        self,
        reflection: Dict[str, Any],
        spt_result: Optional[Dict[str, Any]],
        messages: List[Dict[str, str]]
    ) -> str:

        spt_section = spt_result.get("formatted_section", "") if spt_result else ""
        conversation_history = self._format_history(messages, limit=6)

        return self.response_prompt.format(
            user_utterance_type=reflection.get('user_utterance_type', 'Question'),
            context_analysis=reflection.get('context_analysis', ''),
            spt_section=spt_section,
            conversation_history=conversation_history
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        session_id: str = "default"
    ) -> str:
        """Three-phase conversation: Reflection -> SPT Planning (conditional) -> Response."""
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # Phase 1: Basic Reflection
            reflection = await self._perform_basic_reflection(messages, last_user_msg, session_id)

            # Phase 1.5: SPT Planning (conditional)
            spt_result = None
            if reflection.get("spt_required", False):
                spt_result = await self._perform_spt_planning(messages, last_user_msg, session_id)

            # Phase 2: Response Generation
            logger.info(f"[PHASE2_START] session_id={session_id}, Generating response...")

            llm = self._create_llm(max_tokens=500, streaming=False, temperature=temperature)
            logger.info(f"[PHASE2] Using GPT-4o for response")

            combined_prompt = self._build_response_prompt(reflection, spt_result, messages)

            lc_messages = [SystemMessage(content=combined_prompt)]
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if not content:
                    continue
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                else:
                    lc_messages.append(AIMessage(content=content))

            result = await llm.ainvoke(lc_messages)

            raw_content = result.content.strip()
            cleaned_content = clean_gpt_response(raw_content)

            logger.info(f"[PHASE2_COMPLETE] session_id={session_id}, response: '{cleaned_content}'")

            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_content)

            return cleaned_content

        except Exception as e:
            logger.error(f"[Colleague2] Error: {e}", exc_info=True)
            return "선생님, 다시 한 번만 말씀해주시겠어요?"

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        session_id: str = "default"
    ):
        """Three-phase streaming conversation."""
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # Phase 1: Basic Reflection
            reflection = await self._perform_basic_reflection(messages, last_user_msg, session_id)

            # Phase 1.5: SPT Planning (conditional)
            spt_result = None
            if reflection.get("spt_required", False):
                spt_result = await self._perform_spt_planning(messages, last_user_msg, session_id)

            # Phase 2: Response Generation (Streaming)
            logger.info(f"[PHASE2_START] session_id={session_id}, Generating streamed response...")

            llm = self._create_llm(max_tokens, streaming=True, temperature=temperature)

            combined_prompt = self._build_response_prompt(reflection, spt_result, messages)

            lc_messages = [SystemMessage(content=combined_prompt)]
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if not content:
                    continue
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                else:
                    lc_messages.append(AIMessage(content=content))

            full_response = ""
            async for chunk in llm.astream(lc_messages):
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text:
                    full_response += chunk_text
                    yield chunk_text

            cleaned_response = clean_gpt_response(full_response)
            logger.info(f"[PHASE2_COMPLETE] session_id={session_id}, streamed: '{cleaned_response}'")

            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_response)

        except Exception as e:
            logger.error(f"[Colleague2] Streaming error: {e}", exc_info=True)
            for char in "선생님, 다시 한 번만 말씀해주시겠어요?":
                yield char

    def _extract_chunk_text(self, chunk) -> str:
        if hasattr(chunk, "content"):
            if isinstance(chunk.content, list):
                return "".join(part["text"] for part in chunk.content if isinstance(part, dict) and part.get("text"))
            else:
                return str(chunk.content)
        else:
            return str(chunk)

    @staticmethod
    def _extract_last_user_message(messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"].strip()
        return ""
