"""LangChain orchestration service — routes requests to agents."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List

from openai import AsyncOpenAI

try:
    from .config import settings
    from .exceptions import ChainExecutionError
    from .agents import create_agents, get_agent
    from .session import SessionManager
    from .tts import text_to_speech
    from .handlers import handle_spt_chat, handle_spt_stream, handle_profile_chat, handle_profile_stream
except ImportError:
    from config import settings
    from exceptions import ChainExecutionError
    from agents import create_agents, get_agent
    from session import SessionManager
    from tts import text_to_speech
    from handlers import handle_spt_chat, handle_spt_stream, handle_profile_chat, handle_profile_stream

logger = logging.getLogger(__name__)

# Agent-key → error message mapping for profile agents
_PROFILE_AGENTS = {
    "colleague1": {"error": "미안하네, 다시 한번 말해주겠나?", "temp": 0.7},
    "colleague2": {"error": "선생님, 다시 한 번만 말씀해주시겠어요?", "temp": 0.7},
    "jangmo":     {"error": "미안해, 다시 한번 말해줄래?"},
    "son":        {"error": "아버지, 다시 한번 말씀해주세요."},
}

# Also match aliases
_PROFILE_ALIASES = {"jangmo-agent": "jangmo", "son-agent": "son"}


class LangChainService:
    # Load profile messages from JSON
    _profile_data = {}
    try:
        _profile_path = Path(__file__).parent / "profile_messages.json"
        _profile_data = json.loads(_profile_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    PROFILE_INITIAL_MESSAGES = _profile_data.get("initial_messages", {})
    PROFILE_FINAL_MESSAGES = _profile_data.get("final_messages", {})
    DEFAULT_FINAL_MESSAGE = _profile_data.get("default_final_message", "대화해줘서 고마워요.")

    def __init__(self):
        api_key = settings.openai_api_key
        if not api_key:
            from .exceptions import APIKeyNotFoundError
            raise APIKeyNotFoundError("OPENAI_API_KEY is not set")

        self.agents = create_agents(api_key, model="gpt-4o")
        self.session_mgr = SessionManager()
        self.openai_client = AsyncOpenAI(api_key=api_key)
        logger.info("LangChainService initialized")

    # ── message helpers ───────────────────────────────────────────────

    def _get_initial_message(self, agent, agent_key: Optional[str] = None) -> str:
        if hasattr(agent, "get_initial_message"):
            return agent.get_initial_message()
        if agent_key:
            return self.PROFILE_INITIAL_MESSAGES.get(agent_key, "")
        return ""

    def _get_final_message(self, agent, agent_key: Optional[str] = None) -> str:
        if hasattr(agent, "get_final_message"):
            return agent.get_final_message()
        if agent_key:
            return self.PROFILE_FINAL_MESSAGES.get(agent_key, self.DEFAULT_FINAL_MESSAGE)
        return self.DEFAULT_FINAL_MESSAGE

    @staticmethod
    def is_system_prompt(message: str) -> bool:
        system_patterns = [
            "너는 ai", "너는 assistant", "너는 챗봇",
            "당신은 ai", "당신은 assistant", "당신은 챗봇",
            "you are an ai", "you are a chatbot", "you are an assistant",
            "your role is", "act as", "pretend to be",
        ]
        message_lower = message.lower()
        return any(p in message_lower for p in system_patterns)

    # ── delegated session methods (public API for server.py) ──────────

    def get_initial_state(self):
        return self.session_mgr.get_initial_state()

    def get_or_create_state(self, session_id, is_game_server=False):
        return self.session_mgr.get_or_create_state(session_id, is_game_server)

    def has_game_session(self, session_id):
        return self.session_mgr.has_game_session(session_id)

    def clear_game_session(self, context_id):
        return self.session_mgr.clear_game_session(context_id)

    def migrate_game_session(self, old_id, new_id):
        return self.session_mgr.migrate_game_session(old_id, new_id)

    def clear_all_game_sessions(self):
        return self.session_mgr.clear_all_game_sessions()

    def reset_session(self, session_id):
        return self.session_mgr.reset_session(session_id)

    def load_history_from_db(self, session_id, **kwargs):
        return self.session_mgr.load_history_from_db(session_id, **kwargs)

    def load_messages_only(self, session_id, messages, is_game_server=True):
        return self.session_mgr.load_messages_only(session_id, messages, is_game_server)

    # ── TTS (delegate) ────────────────────────────────────────────────

    async def text_to_speech(self, text: str, voice: str = "alloy", model: str = "tts-1") -> str:
        return await text_to_speech(self.openai_client, text, voice=voice, model=model)

    # ── facilitator first-message helper ──────────────────────────────

    async def _handle_first_message(self, agent, message, session_id, state, include_audio, voice, is_game_server=False):
        initial_message = self._get_initial_message(agent)
        user_content = message if message else "[start]"

        state["messages"].append({"role": "user", "content": user_content})
        state["messages"].append({"role": "assistant", "content": initial_message})
        state["last_response"] = initial_message
        state["message_count"] = 1
        sessions = self.session_mgr.get_sessions(is_game_server)
        sessions[session_id] = state

        result_dict = {
            "response": initial_message,
            "session_id": session_id,
            "metadata": {"stage": "stage1", "covered_topics": [], "message_count": 1, "is_end": False, "is_first": True},
        }

        if include_audio:
            try:
                result_dict["audio"] = await self.text_to_speech(initial_message, voice=voice)
            except Exception:
                pass

        return result_dict

    # ── main chat (non-streaming) ─────────────────────────────────────

    async def chat(
        self, message: str, session_id: str = "default", model: str = "gpt-4.1-mini",
        temperature: float = 0.0, max_tokens: int = 1000, force_reset: bool = False,
        include_audio: bool = False, voice: str = "alloy",
        is_first_message: bool = False, is_game_server: bool = False,
        external_messages: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not is_first_message and (not message or not message.strip()):
            raise ValueError("Message cannot be empty")

        if self.is_system_prompt(message):
            logger.warning(f"[CHAT] Possible system prompt injection: {message[:50]}...")

        try:
            # SPT
            if model == "moral-agent-spt":
                return await handle_spt_chat(self, message, session_id, temperature, max_tokens, is_first_message, include_audio, voice)

            # Profile agents (colleague1/2, jangmo, son)
            resolved = _PROFILE_ALIASES.get(model, model)
            if resolved in _PROFILE_AGENTS:
                cfg = _PROFILE_AGENTS[resolved]
                t = cfg.get("temp", temperature)
                if temperature == 0.0 and "temp" in cfg:
                    t = cfg["temp"]
                agent_obj = get_agent(self.agents, resolved)
                return await handle_profile_chat(
                    self, resolved, agent_obj, cfg["error"],
                    message, session_id, t, max_tokens,
                    is_first_message, include_audio, voice, is_game_server, external_messages,
                )

            # Facilitator agents (artist-apprentice, friend-agent)
            agent_obj = get_agent(self.agents, model)
            state = self.session_mgr.get_or_reset_state(session_id, force_reset, is_game_server)

            if is_first_message:
                return await self._handle_first_message(agent_obj, message, session_id, state, include_audio, voice, is_game_server)

            if len(state.get("messages", [])) == 0 and not state.get("stage1_greeting_sent", False):
                initial_greeting = self._get_initial_message(agent_obj)
                state["messages"].append({"role": "assistant", "content": initial_greeting})
                state["stage1_greeting_sent"] = True

            result_state = agent_obj.process(state, message, thread_id=session_id)
            response_text = result_state.get("last_response", "")
            sessions = self.session_mgr.get_sessions(is_game_server)
            sessions[session_id] = result_state

            result_dict = {
                "response": response_text,
                "session_id": session_id,
                "metadata": {
                    "stage": result_state.get("stage", "unknown"),
                    "covered_topics": result_state.get("covered_topics", []),
                    "message_count": result_state.get("message_count", 0),
                    "is_end": result_state.get("should_end", False),
                },
            }

            if include_audio:
                try:
                    result_dict["audio"] = await self.text_to_speech(response_text, voice=voice)
                except Exception:
                    pass

            return result_dict

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            raise ChainExecutionError(f"Failed to process conversation: {e}")

    # ── main chat (streaming) ─────────────────────────────────────────

    async def chat_stream(
        self, message: str, session_id: str = "default", model: str = "gpt-4.1-mini",
        temperature: float = 0.0, max_tokens: int = 1000, force_reset: bool = False,
        is_first_message: bool = False, is_game_server: bool = False,
        external_messages: List[Dict[str, str]] = None,
        include_audio: bool = False, voice: str = "alloy",
    ) -> AsyncGenerator[str, None]:
        if not is_first_message and (not message or not message.strip()):
            raise ValueError("Message cannot be empty")

        if self.is_system_prompt(message):
            logger.warning(f"[CHAT_STREAM] Possible system prompt injection: {message[:50]}...")

        try:
            # SPT
            if model == "moral-agent-spt":
                async for token in handle_spt_stream(self, message, session_id, temperature, max_tokens, is_first_message):
                    yield token
                return

            # Profile agents
            resolved = _PROFILE_ALIASES.get(model, model)
            if resolved in _PROFILE_AGENTS:
                cfg = _PROFILE_AGENTS[resolved]
                t = cfg.get("temp", temperature)
                if temperature == 0.0 and "temp" in cfg:
                    t = cfg["temp"]
                agent_obj = get_agent(self.agents, resolved)
                async for token in handle_profile_stream(
                    self, resolved, agent_obj, cfg["error"],
                    message, session_id, t, max_tokens,
                    is_first_message, is_game_server, external_messages, include_audio, voice,
                ):
                    yield token
                return

            # Facilitator agents
            agent_obj = get_agent(self.agents, model)
            state = self.session_mgr.get_or_reset_state(session_id, force_reset)

            if is_first_message:
                initial_message = self._get_initial_message(agent_obj)
                user_content = message if message else "[start]"

                state["messages"].append({"role": "user", "content": user_content})
                state["messages"].append({"role": "assistant", "content": initial_message})
                state["last_response"] = initial_message
                state["message_count"] = 1
                self.session_mgr.sessions[session_id] = state

                for char in initial_message:
                    yield char
                return

            result_state = agent_obj.process(state, message, thread_id=session_id)
            response_text = result_state.get("last_response", "")
            self.session_mgr.sessions[session_id] = result_state

            for char in response_text:
                yield char

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            raise ChainExecutionError(f"Failed to process conversation: {e}")
