import sys
from pathlib import Path
import re
import copy

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from typing import Optional, Dict, Any, AsyncGenerator, List
import logging
import base64
import hashlib
from functools import lru_cache

try:
    from .config import settings
    from .exceptions import APIKeyNotFoundError, ChainExecutionError
except ImportError:
    from config import settings
    from exceptions import APIKeyNotFoundError, ChainExecutionError
from agents.artist_apprentice_agent.conversation_agent import ConversationAgent as ArtistApprenticeAgent
from agents.friend_agent.conversation_agent import ConversationAgent as FriendAgent
from agents.spt_agent import SPTAgent
from agents.colleague1_agent import Colleague1Agent
from agents.colleague2_agent import Colleague2Agent

try:
    from utils.persona_validator import validate_and_fix_persona
    HAS_PERSONA_VALIDATOR = True
except ImportError:
    HAS_PERSONA_VALIDATOR = False
    validate_and_fix_persona = None

try:
    from agents.jangmo_agent import JangmoAgent
    HAS_JANGMO = True
except ImportError:
    HAS_JANGMO = False
    JangmoAgent = None

try:
    from agents.son_agent import SonAgent
    HAS_SON = True
except ImportError:
    HAS_SON = False
    SonAgent = None
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LangChainService:
    PROFILE_INITIAL_MESSAGES = {
        "jangmo": "아무리 생각해도 이 기술은 너무 비윤리적이지 않니?",
        "son": "아버지, 저는요.. 이 기술이 가져올 행복이 더 크다고 생각해요. 동의하지 않으세요?",
        "colleague1": "AI가 그린 그림을 어떻게 국립 예술관에 전시를 할 수가 있지? 그걸 예술로 공식적으로 인정한다는 건 말이 안 되네. 나는 무조건 전시 반대에 투표할걸세.",
        "colleague2": "선생님, 저는 AI 예술도 사람들에게 감동을 주면 충분히 가치 있다고 생각해요.",
    }
    PROFILE_FINAL_MESSAGES = {
        "jangmo": "...네 뜻 잘 알겠다. 모쪼록 우리 둘의 의견을 잘 고려해서 결정해주렴.",
        "son": "...그렇군요. 아버지 의견 잘 들었어요. 잘 생각하셔서 결정해주세요.",
        "colleague1": "...그래, 자네 생각 잘 들었네. 투표 때 신중하게 결정하게나.",
        "colleague2": "선생님, 좋은 말씀 감사합니다. 투표 때 신중하게 결정해주시길 바랍니다.",
    }
    DEFAULT_FINAL_MESSAGE = "대화해줘서 고마워요. 좋은 결정 내리시길 바랍니다."

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.profile_sessions: Dict[tuple, Dict[str, Any]] = {}
        self.sessions_game: Dict[str, Dict[str, Any]] = {}
        self.profile_sessions_game: Dict[tuple, Dict[str, Any]] = {}

        api_key = settings.openai_api_key
        if not api_key:
            raise APIKeyNotFoundError("OPENAI_API_KEY is not set")

        self.artist_apprentice_agent = ArtistApprenticeAgent(api_key=api_key, model="gpt-4o")
        self.friend_agent = FriendAgent(api_key=api_key, model="gpt-4o")
        self.spt_agent = SPTAgent(api_key=api_key)
        self.colleague1_agent = Colleague1Agent(api_key=api_key, model="gpt-4o")
        self.colleague2_agent = Colleague2Agent(api_key=api_key, model="gpt-4o")
        self.jangmo_agent = JangmoAgent(api_key=api_key, model="gpt-4o") if HAS_JANGMO else None
        self.son_agent = SonAgent(api_key=api_key, model="gpt-4o") if HAS_SON else None

        self.agent = self.artist_apprentice_agent
        self.openai_client = AsyncOpenAI(api_key=api_key)
        logger.info("LangChainService initialized")

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

    def _get_sessions(self, is_game_server: bool = False) -> Dict[str, Dict[str, Any]]:
        return self.sessions_game if is_game_server else self.sessions

    def _get_profile_sessions(self, is_game_server: bool = False) -> Dict[tuple, Dict[str, Any]]:
        return self.profile_sessions_game if is_game_server else self.profile_sessions

    def get_initial_state(self) -> Dict[str, Any]:
        return {
            "stage": "stage1",
            "previous_stage": "",
            "messages": [],
            "covered_topics": [],
            "current_asking_topic": "",
            "message_count": 0,
            "last_response": "",
            "artist_character_set": False,
            "should_end": False,
            "stage1_attempts": 0,
            "stage1_greeting_sent": False,
            "stage1_follow_up_asked": False,
            "stage2_question_asked": False,
            "stage2_complete": False,
            "stage2_completed": False,
            "current_question_index": 0,
            "variation_index": 0,
            "dont_know_count": 0
        }

    def get_or_create_state(self, session_id: str, is_game_server: bool = False) -> Dict[str, Any]:
        sessions = self._get_sessions(is_game_server)
        if session_id not in sessions:
            sessions[session_id] = self.get_initial_state()
        return sessions[session_id]

    def has_game_session(self, session_id: str) -> bool:
        return session_id in self.sessions_game

    def is_system_prompt(self, message: str) -> bool:
        """Detect system prompt injection attempts."""
        system_patterns = [
            "너는 ai", "너는 assistant", "너는 챗봇",
            "당신은 ai", "당신은 assistant", "당신은 챗봇",
            "you are an ai", "you are a chatbot", "you are an assistant",
            "your role is", "act as", "pretend to be"
        ]
        message_lower = message.lower()
        return any(pattern in message_lower for pattern in system_patterns)

    async def _normalize_user_message_llm(self, message: str) -> str:
        """Correct typos in user input via LLM while preserving meaning and tone."""
        if not message or not message.strip():
            return message

        system_prompt = (
            "너는 한국어 사용자 입력의 오타만 교정하는 도우미다.\n"
            "- 의미/의도/말투/존댓말/반말을 절대 바꾸지 말 것\n"
            "- 문장 구조를 크게 바꾸지 말 것\n"
            "- 고유명사와 신조어는 그대로 둘 것\n"
            "- 교정 결과만 한 줄로 출력할 것\n"
        )
        user_prompt = f"입력: {message}\n교정:"

        try:
            result = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            corrected = result.choices[0].message.content.strip()
            return corrected or message
        except Exception as exc:
            logger.warning(f"[LLM_NORMALIZE] Failed: {exc}")
            return message

    def _get_agent(self, model: str):
        if model == "friend-agent":
            return self.friend_agent
        elif model == "artist-apprentice":
            return self.artist_apprentice_agent
        elif model == "colleague1":
            return self.colleague1_agent
        elif model == "colleague2":
            return self.colleague2_agent
        elif model in ["jangmo", "jangmo-agent"] and self.jangmo_agent:
            return self.jangmo_agent
        elif model in ["son", "son-agent"] and self.son_agent:
            return self.son_agent
        else:
            return self.artist_apprentice_agent

    def _get_or_reset_state(self, session_id: str, force_reset: bool, is_game_server: bool = False) -> Dict[str, Any]:
        sessions = self._get_sessions(is_game_server)
        if force_reset:
            state = self.get_initial_state()
            sessions[session_id] = state
            return state

        state = self.get_or_create_state(session_id, is_game_server)
        if state.get("should_end", False) or state.get("stage") == "end":
            state = self.get_initial_state()
            sessions[session_id] = state
        return state

    async def _handle_first_message(self, agent, message: str, session_id: str, state: Dict[str, Any], include_audio: bool, voice: str, is_game_server: bool = False) -> Dict[str, Any]:
        initial_message = self._get_initial_message(agent)
        user_content = message if message else "[start]"

        state["messages"].append({"role": "user", "content": user_content})
        state["messages"].append({"role": "assistant", "content": initial_message})
        state["last_response"] = initial_message
        state["message_count"] = 1
        sessions = self._get_sessions(is_game_server)
        sessions[session_id] = state

        result_dict = {
            "response": initial_message,
            "session_id": session_id,
            "metadata": {"stage": "stage1", "covered_topics": [], "message_count": 1, "is_end": False, "is_first": True}
        }

        if include_audio:
            try:
                result_dict["audio"] = await self.text_to_speech(initial_message, voice=voice)
            except Exception:
                pass

        return result_dict

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        force_reset: bool = False,
        include_audio: bool = False,
        voice: str = "alloy",
        is_first_message: bool = False,
        is_game_server: bool = False,
        external_messages: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        if not is_first_message and (not message or not message.strip()):
            raise ValueError("Message cannot be empty")

        if self.is_system_prompt(message):
            logger.warning(f"[CHAT] Possible system prompt injection: {message[:50]}...")

        corrected_message = await self._normalize_user_message_llm(message)

        try:
            if model == "moral-agent-spt":
                return await self._handle_spt_chat(corrected_message, session_id, temperature, max_tokens, is_first_message, include_audio, voice)

            if model == "colleague1":
                colleague_temp = 0.7 if temperature == 0.0 else temperature
                return await self._handle_profile_chat("colleague1", self.colleague1_agent, "미안하네, 다시 한번 말해주겠나?", corrected_message, session_id, colleague_temp, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model == "colleague2":
                colleague_temp = 0.7 if temperature == 0.0 else temperature
                return await self._handle_profile_chat("colleague2", self.colleague2_agent, "선생님, 다시 한 번만 말씀해주시겠어요?", corrected_message, session_id, colleague_temp, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model in ["jangmo", "jangmo-agent"] and self.jangmo_agent:
                return await self._handle_profile_chat("jangmo", self.jangmo_agent, "미안해, 다시 한번 말해줄래?", corrected_message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model in ["son", "son-agent"] and self.son_agent:
                return await self._handle_profile_chat("son", self.son_agent, "아버지, 다시 한번 말씀해주세요.", corrected_message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            agent = self._get_agent(model)
            state = self._get_or_reset_state(session_id, force_reset, is_game_server)

            if is_first_message:
                return await self._handle_first_message(agent, corrected_message, session_id, state, include_audio, voice, is_game_server)

            # Auto-inject greeting if session is empty but not first message
            if len(state.get("messages", [])) == 0 and not state.get("stage1_greeting_sent", False):
                initial_greeting = self._get_initial_message(agent)
                state["messages"].append({"role": "assistant", "content": initial_greeting})
                state["stage1_greeting_sent"] = True

            result_state = agent.process(state, corrected_message, thread_id=session_id)
            response_text = result_state.get("last_response", "")
            sessions = self._get_sessions(is_game_server)
            sessions[session_id] = result_state

            result_dict = {
                "response": response_text,
                "session_id": session_id,
                "metadata": {
                    "stage": result_state.get("stage", "unknown"),
                    "covered_topics": result_state.get("covered_topics", []),
                    "message_count": result_state.get("message_count", 0),
                    "is_end": result_state.get("should_end", False)
                }
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

    async def chat_stream(
        self,
        message: str,
        session_id: str = "default",
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        force_reset: bool = False,
        is_first_message: bool = False,
        is_game_server: bool = False,
        external_messages: List[Dict[str, str]] = None,
        include_audio: bool = False,
        voice: str = "alloy"
    ) -> AsyncGenerator[str, None]:
        if not is_first_message and (not message or not message.strip()):
            raise ValueError("Message cannot be empty")

        if self.is_system_prompt(message):
            logger.warning(f"[CHAT_STREAM] Possible system prompt injection: {message[:50]}...")

        corrected_message = await self._normalize_user_message_llm(message)

        try:
            if model == "moral-agent-spt":
                async for token in self._handle_spt_stream(corrected_message, session_id, temperature, max_tokens, is_first_message):
                    yield token
                return

            if model == "colleague1":
                colleague_temp = 0.7 if temperature == 0.0 else temperature
                async for token in self._handle_profile_stream("colleague1", self.colleague1_agent, "미안하네, 다시 한번 말해주겠나?", corrected_message, session_id, colleague_temp, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            if model == "colleague2":
                colleague_temp = 0.7 if temperature == 0.0 else temperature
                async for token in self._handle_profile_stream("colleague2", self.colleague2_agent, "선생님, 다시 한 번만 말씀해주시겠어요?", corrected_message, session_id, colleague_temp, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            if model in ["jangmo", "jangmo-agent"] and self.jangmo_agent:
                async for token in self._handle_profile_stream("jangmo", self.jangmo_agent, "미안해, 다시 한번 말해줄래?", corrected_message, session_id, temperature, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            if model in ["son", "son-agent"] and self.son_agent:
                async for token in self._handle_profile_stream("son", self.son_agent, "아버지, 다시 한번 말씀해주세요.", corrected_message, session_id, temperature, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            agent = self._get_agent(model)
            state = self._get_or_reset_state(session_id, force_reset)

            if is_first_message:
                initial_message = self._get_initial_message(agent)
                user_content = corrected_message if corrected_message else "[start]"

                state["messages"].append({"role": "user", "content": user_content})
                state["messages"].append({"role": "assistant", "content": initial_message})
                state["last_response"] = initial_message
                state["message_count"] = 1
                self.sessions[session_id] = state

                for char in initial_message:
                    yield char
                return

            result_state = agent.process(state, corrected_message, thread_id=session_id)
            response_text = result_state.get("last_response", "")
            self.sessions[session_id] = result_state

            for char in response_text:
                yield char

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            raise ChainExecutionError(f"Failed to process conversation: {e}")

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        return self.get_initial_state()

    def clear_game_session(self, context_id: str) -> bool:
        """Reset game session to initial state."""
        self.sessions_game[context_id] = self.get_initial_state()
        keys_to_delete = [key for key in self.profile_sessions_game if key[1] == context_id]
        for key in keys_to_delete:
            del self.profile_sessions_game[key]
        return True

    def migrate_game_session(self, old_context_id: str, new_context_id: str) -> bool:
        """Migrate session from old context to new context for conversation continuity."""
        if old_context_id in self.sessions_game:
            old_state = self.sessions_game[old_context_id]
            self.sessions_game[new_context_id] = copy.deepcopy(old_state)
            del self.sessions_game[old_context_id]

        keys_to_delete = [key for key in self.profile_sessions_game if key[1] == old_context_id]
        for old_key in keys_to_delete:
            del self.profile_sessions_game[old_key]
        return True

    def clear_all_game_sessions(self) -> bool:
        """Clear all game sessions to prevent memory leaks."""
        self.sessions_game.clear()
        self.profile_sessions_game.clear()
        return True

    async def _handle_spt_chat(self, message: str, session_id: str, temperature: float, max_tokens: int, is_first_message: bool, include_audio: bool, voice: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {"messages": []}

        session_data = self.sessions[session_id]

        if is_first_message:
            initial_message = self.spt_agent.get_initial_message()
            session_data["messages"].append({"role": "assistant", "content": initial_message})
            result_dict = {
                "response": initial_message,
                "session_id": session_id,
                "metadata": {"stage": "spt", "message_count": 1, "is_end": False, "is_first": True}
            }
            if include_audio:
                try:
                    result_dict["audio"] = await self.text_to_speech(initial_message, voice=voice)
                except Exception:
                    pass
            return result_dict

        session_data["messages"].append({"role": "user", "content": message})

        try:
            response_text = await self.spt_agent.chat(messages=session_data["messages"], temperature=temperature, max_tokens=max_tokens)
            session_data["messages"].append({"role": "assistant", "content": response_text})

            result_dict = {
                "response": response_text,
                "session_id": session_id,
                "metadata": {"stage": "spt", "message_count": len(session_data["messages"]), "is_end": False}
            }
            if include_audio:
                try:
                    result_dict["audio"] = await self.text_to_speech(response_text, voice=voice)
                except Exception:
                    pass
            return result_dict

        except Exception as e:
            logger.error(f"SPT error: {e}")
            raise ChainExecutionError(f"SPT failed: {str(e)}")

    def _get_profile_session(self, agent_key: str, session_id: str, is_game_server: bool = False) -> Dict[str, Any]:
        profile_sessions = self._get_profile_sessions(is_game_server)
        key = (agent_key, session_id)

        if key not in profile_sessions:
            other_sessions = self._get_profile_sessions(not is_game_server)
            if key in other_sessions:
                profile_sessions[key] = other_sessions[key]
            else:
                profile_sessions[key] = {"messages": [], "turn_count": 1}

        return profile_sessions[key]

    def load_history_from_db(self, session_id: str, messages: List[Dict[str, str]] = None, history: List[Dict[str, str]] = None, turn_count: int = None, agent_key: str = None, is_game_server: bool = True):
        """Load conversation history from DB into session for resuming."""
        if messages is not None:
            formatted_messages = messages
            msg_count = turn_count if turn_count else len(messages)
        elif history is not None:
            formatted_messages = []
            for row in history:
                formatted_messages.append({"role": "user", "content": row["user_message"]})
                formatted_messages.append({"role": "assistant", "content": row["ai_message"]})
            msg_count = len(history)
        else:
            logger.error("[LOAD_HISTORY] No messages or history provided")
            return

        if agent_key:
            profile_sessions = self._get_profile_sessions(is_game_server)
            key = (agent_key, session_id)
            profile_sessions[key] = {"messages": formatted_messages, "turn_count": msg_count}
        else:
            sessions = self._get_sessions(is_game_server)
            state = self.get_initial_state()
            state["messages"] = formatted_messages
            state["message_count"] = msg_count

            stage, _ = self._infer_stage_from_messages(formatted_messages)
            state["stage"] = stage
            state["previous_stage"] = stage

            if stage == "stage2":
                state["artist_character_set"] = True
                state["stage2_question_asked"] = True
            elif stage == "stage3":
                state["artist_character_set"] = True
                state["stage2_question_asked"] = True
                state["stage2_complete"] = True
                state["stage2_completed"] = True

            sessions[session_id] = state

    def load_messages_only(self, session_id: str, messages: List[Dict[str, str]], is_game_server: bool = True):
        """Load messages for LLM context only (stage stays at stage1)."""
        sessions = self._get_sessions(is_game_server)
        state = self.get_initial_state()
        state["messages"] = messages
        state["message_count"] = len(messages)
        sessions[session_id] = state

    def _infer_stage_from_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Infer current stage from conversation content using keyword matching."""
        if not messages:
            return "stage1", "no_messages"

        assistant_messages = " ".join([
            msg["content"] for msg in messages
            if msg.get("role") == "assistant"
        ])

        # Check stage3 first (later stage takes priority)
        stage3_keywords = ["어떤 이익", "어떤 피해", "대안", "이익이 있을까", "피해가 있을까"]
        for keyword in stage3_keywords:
            if keyword in assistant_messages:
                return "stage3", f"keyword:{keyword}"

        stage2_keywords = ["그림 그리는 AI", "국립현대예술관", "AI의 그림", "전시가 된대요"]
        for keyword in stage2_keywords:
            if keyword in assistant_messages:
                return "stage2", f"keyword:{keyword}"

        return "stage1", "default"

    def _trim_profile_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= 15:
            return messages
        first_message = messages[0]
        return [first_message] + messages[-14:]

    async def _handle_profile_chat(self, agent_key: str, agent, error_message: str, message: str, session_id: str, temperature: float, max_tokens: int, is_first_message: bool, include_audio: bool, voice: str, is_game_server: bool = False, external_messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        session_data = self._get_profile_session(agent_key, session_id, is_game_server)

        # Initialize from external messages if session is empty
        if external_messages and len(session_data["messages"]) == 0:
            filtered_messages = [
                msg for msg in external_messages
                if msg.get("role") in ["user", "assistant"]
                and not any(pattern in msg.get("content", "").lower() for pattern in [
                    "### task:", "suggest 3-5", "generate a concise", "generate 1-3"
                ])
            ]
            if filtered_messages:
                if filtered_messages[-1].get("role") == "user":
                    session_data["messages"] = filtered_messages[:-1]
                else:
                    session_data["messages"] = filtered_messages
                user_count = len([m for m in session_data["messages"] if m.get("role") == "user"])
                session_data["turn_count"] = user_count + 1

        if is_first_message:
            initial_message = self._get_initial_message(agent, agent_key)
            session_data["messages"].append({"role": "assistant", "content": initial_message})

            if not message:
                session_data["turn_count"] = 0
                result_dict = {
                    "response": initial_message,
                    "session_id": session_id,
                    "metadata": {"stage": agent_key, "message_count": len(session_data["messages"]), "turn_count": session_data["turn_count"], "is_end": False, "is_first": True}
                }
                if include_audio:
                    try:
                        result_dict["audio"] = await self.text_to_speech(initial_message, voice=voice)
                    except Exception:
                        pass
                return result_dict

        session_data["messages"].append({"role": "user", "content": message})
        if is_first_message:
            session_data["turn_count"] = 1
        else:
            session_data["turn_count"] += 1
        session_data["messages"] = self._trim_profile_history(session_data["messages"])

        # End conversation at turn 8 for persona agents
        if session_data["turn_count"] >= 8 and agent_key not in ["friend", "artist_apprentice"]:
            final_message = self._get_final_message(agent, agent_key)
            session_data["messages"].append({"role": "assistant", "content": final_message})
            result_dict = {
                "response": final_message,
                "session_id": session_id,
                "metadata": {"stage": agent_key, "message_count": len(session_data["messages"]), "turn_count": session_data["turn_count"], "is_end": True}
            }
            if include_audio:
                try:
                    result_dict["audio"] = await self.text_to_speech(final_message, voice=voice)
                except Exception:
                    pass
            return result_dict

        try:
            response_text = await agent.chat(
                messages=session_data["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=session_id
            )

            # Persona validation post-processing
            if HAS_PERSONA_VALIDATOR and agent_key not in ["friend", "artist_apprentice"]:
                response_text, _ = await validate_and_fix_persona(
                    response_text, agent_key, settings.openai_api_key
                )

            # Fallback for empty responses
            if not response_text or len(response_text.strip()) < 5 or response_text.strip() in ["...", "..."]:
                response_text = error_message

            session_data["messages"].append({"role": "assistant", "content": response_text})

            # Facilitator agents have no turn limit
            if agent_key in ["friend", "artist_apprentice"]:
                is_end = False
            else:
                is_end = session_data["turn_count"] >= 8

            result_dict = {
                "response": response_text,
                "session_id": session_id,
                "metadata": {"stage": agent_key, "message_count": len(session_data["messages"]), "turn_count": session_data["turn_count"], "is_end": is_end}
            }
            if include_audio:
                try:
                    result_dict["audio"] = await self.text_to_speech(response_text, voice=voice)
                except Exception:
                    pass
            return result_dict

        except Exception as e:
            logger.error(f"{agent_key} error: {e}")
            raise ChainExecutionError(f"{agent_key} failed: {str(e)}")

    async def _handle_profile_stream(self, agent_key: str, agent, error_message: str, message: str, session_id: str, temperature: float, max_tokens: int, is_first_message: bool, is_game_server: bool = False, external_messages: List[Dict[str, str]] = None, include_audio: bool = False, voice: str = "alloy") -> AsyncGenerator[str, None]:
        session_data = self._get_profile_session(agent_key, session_id, is_game_server)

        # Initialize from external messages if session is empty
        if external_messages and len(session_data["messages"]) == 0:
            filtered_messages = [
                msg for msg in external_messages
                if msg.get("role") in ["user", "assistant"]
                and not any(pattern in msg.get("content", "").lower() for pattern in [
                    "### task:", "suggest 3-5", "generate a concise", "generate 1-3"
                ])
            ]
            if filtered_messages:
                if filtered_messages[-1].get("role") == "user":
                    session_data["messages"] = filtered_messages[:-1]
                else:
                    session_data["messages"] = filtered_messages
                user_count = len([m for m in session_data["messages"] if m.get("role") == "user"])
                session_data["turn_count"] = user_count + 1

        if is_first_message:
            initial_message = self._get_initial_message(agent, agent_key)
            session_data["messages"].append({"role": "assistant", "content": initial_message})

            if not message:
                session_data["turn_count"] = 0
                for char in initial_message:
                    yield char
                if include_audio:
                    try:
                        audio_data = await self.text_to_speech(initial_message, voice=voice)
                        yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                    except Exception as e:
                        logger.warning(f"TTS failed: {e}")
                return

        session_data["messages"].append({"role": "user", "content": message})
        if is_first_message:
            session_data["turn_count"] = 1
        else:
            session_data["turn_count"] += 1
        session_data["messages"] = self._trim_profile_history(session_data["messages"])

        # End conversation at turn 8 for persona agents
        if session_data["turn_count"] >= 8 and agent_key not in ["friend", "artist_apprentice"]:
            final_message = self._get_final_message(agent, agent_key)
            session_data["messages"].append({"role": "assistant", "content": final_message})
            for char in final_message:
                yield char
            if include_audio:
                try:
                    audio_data = await self.text_to_speech(final_message, voice=voice)
                    yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")
            return

        if not max_tokens or max_tokens > 300:
            max_tokens = 300

        try:
            full_response = await agent.chat(
                messages=session_data["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=session_id
            )

            response = full_response

            # Persona validation post-processing
            if HAS_PERSONA_VALIDATOR and agent_key not in ["friend", "artist_apprentice"]:
                response, _ = await validate_and_fix_persona(
                    response, agent_key, settings.openai_api_key
                )

            # Fallback for empty responses
            if not response or len(response.strip()) < 5 or response.strip() in ["...", "..."]:
                response = error_message

            for char in response:
                yield char

            if include_audio:
                try:
                    audio_data = await self.text_to_speech(response, voice=voice)
                    yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")

            session_data["messages"].append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"{agent_key} stream error: {e}")
            for char in error_message:
                yield char

    async def _handle_spt_stream(self, message: str, session_id: str, temperature: float, max_tokens: int, is_first_message: bool) -> AsyncGenerator[str, None]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {"messages": []}

        session_data = self.sessions[session_id]

        if is_first_message:
            initial_message = self.spt_agent.get_initial_message()
            session_data["messages"].append({"role": "assistant", "content": initial_message})
            for char in initial_message:
                yield char
            return

        session_data["messages"].append({"role": "user", "content": message})

        try:
            response_text = await self.spt_agent.chat(messages=session_data["messages"], temperature=temperature, max_tokens=max_tokens)
            session_data["messages"].append({"role": "assistant", "content": response_text})
            for char in response_text:
                yield char
        except Exception as e:
            logger.error(f"SPT stream error: {e}")
            error_msg = "죄송해요, 다시 한번 말씀해주시겠어요?"
            for char in error_msg:
                yield char

    async def text_to_speech(self, text: str, voice: str = "alloy", model: str = "tts-1") -> str:
        """Convert text to speech via OpenAI TTS API. Returns base64-encoded MP3."""
        if not text or not text.strip():
            raise ValueError("Empty text cannot be converted to speech")

        try:
            response = await self.openai_client.audio.speech.create(model=model, voice=voice, input=text)
            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return audio_base64
        except Exception as e:
            logger.error(f"[TTS] Error: {type(e).__name__}: {e}")
            raise

    @staticmethod
    def _limit_sentences(text: str, max_sentences: int = 3) -> str:
        """Ensure colleague replies stay within the required sentence count."""
        if not text:
            return text
        condensed = " ".join(text.strip().split())
        if not condensed:
            return condensed
        sentences = re.split(r'(?<=[.!?])\s+', condensed)
        if len(sentences) <= max_sentences:
            return condensed
        trimmed = " ".join(sentences[:max_sentences]).strip()
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."
        return trimmed
