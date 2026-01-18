import sys
from pathlib import Path
import re
import copy

# 상위 디렉터리와 현재 디렉터리를 Python 경로에 추가
# - parent_dir: agents 모듈을 찾기 위해
# - current_dir: utils 모듈을 찾기 위해
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
    # 패키지 모듈로 실행할 때
    from .config import settings
    from .exceptions import APIKeyNotFoundError, ChainExecutionError
except ImportError:
    # 직접 실행할 때
    from config import settings
    from exceptions import APIKeyNotFoundError, ChainExecutionError
from agents.artist_apprentice_agent.conversation_agent import ConversationAgent as ArtistApprenticeAgent
from agents.friend_agent.conversation_agent import ConversationAgent as FriendAgent
from agents.spt_agent import SPTAgent
from agents.colleague1_agent import Colleague1Agent
from agents.colleague2_agent import Colleague2Agent

# 페르소나 검증 import (경로에 따라 다름)
try:
    from utils.persona_validator import validate_and_fix_persona
    HAS_PERSONA_VALIDATOR = True
except ImportError:
    HAS_PERSONA_VALIDATOR = False
    validate_and_fix_persona = None

# Optional imports (배포 전까지 없을 수 있음)
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
    def __init__(self):
        # Open WebUI 전용 세션
        self.sessions_openwebui: Dict[str, Dict[str, Any]] = {}
        self.profile_sessions_openwebui: Dict[tuple, Dict[str, Any]] = {}

        # 게임 서버 전용 세션
        self.sessions_game: Dict[str, Dict[str, Any]] = {}
        self.profile_sessions_game: Dict[tuple, Dict[str, Any]] = {}

        # 하위 호환성을 위한 기본 세션 (Open WebUI용)
        self.sessions = self.sessions_openwebui
        self.profile_sessions = self.profile_sessions_openwebui

        api_key = settings.openai_api_key
        if not api_key:
            raise APIKeyNotFoundError("OPENAI_API_KEY is not set")

        self.artist_apprentice_agent = ArtistApprenticeAgent(api_key=api_key, model="gpt-4o")
        self.friend_agent = FriendAgent(api_key=api_key, model="gpt-4o")
        self.spt_agent = SPTAgent(api_key=api_key)  # fine-tuned 모델: ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs
        self.colleague1_agent = Colleague1Agent(api_key=api_key, model="gpt-4o")
        self.colleague2_agent = Colleague2Agent(api_key=api_key, model="gpt-4o")
        self.jangmo_agent = JangmoAgent(api_key=api_key, model="gpt-4o") if HAS_JANGMO else None
        self.son_agent = SonAgent(api_key=api_key, model="gpt-4o") if HAS_SON else None

        self.agent = self.artist_apprentice_agent
        self.openai_client = AsyncOpenAI(api_key=api_key)

        logger.info("LangChainService initialized (Open WebUI + Game Server sessions separated)")

    def _get_sessions(self, is_game_server: bool = False) -> Dict[str, Dict[str, Any]]:
        """세션 저장소 선택 (Open WebUI vs 게임 서버)"""
        return self.sessions_game if is_game_server else self.sessions_openwebui

    def _get_profile_sessions(self, is_game_server: bool = False) -> Dict[tuple, Dict[str, Any]]:
        """프로필 세션 저장소 선택 (Open WebUI vs 게임 서버)"""
        return self.profile_sessions_game if is_game_server else self.profile_sessions_openwebui

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
        """게임 서버용 메모리 세션이 존재하는지 확인"""
        return session_id in self.sessions_game

    def is_system_prompt(self, message: str) -> bool:
        """
        시스템 프롬프트 주입 공격 감지
        주의: 일반 대화에서 자주 사용되는 표현(예: "너는 어떻게 생각해?")은 제외
        """
        # 시스템 프롬프트 패턴 (문장 시작부에서 역할 지정하는 경우만)
        system_patterns = [
            "너는 ai", "너는 assistant", "너는 챗봇",
            "당신은 ai", "당신은 assistant", "당신은 챗봇",
            "you are an ai", "you are a chatbot", "you are an assistant",
            "your role is", "act as", "pretend to be"
        ]
        message_lower = message.lower()
        return any(pattern in message_lower for pattern in system_patterns)

    async def _normalize_user_message_llm(self, message: str) -> str:
        """LLM 기반 오타/철자 교정 (의미/말투 보존)"""
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
            logger.warning(f"⚠️ [LLM_NORMALIZE] Failed to normalize message: {exc}")
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
        initial_message = agent.get_initial_message()
        user_content = message if message else "[시작]"

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
        external_messages: List[Dict[str, str]] = None  # ✨ 외부에서 전달받은 메시지 히스토리
    ) -> Dict[str, Any]:
        if not is_first_message and (not message or not message.strip()):
            raise ValueError("Message cannot be empty")

        # ⚠️ 시스템 프롬프트 감지 시 로깅만 하고 정상 처리 (사용자 응답은 항상 제공)
        if self.is_system_prompt(message):
            logger.warning(f"⚠️ [CHAT] Possible system prompt injection detected: {message[:50]}...")

        corrected_message = await self._normalize_user_message_llm(message)

        try:
            if model == "moral-agent-spt":
                return await self._handle_spt_chat(corrected_message, session_id, temperature, max_tokens, is_first_message, include_audio, voice)

            if model == "colleague1":
                # gpt-5-mini는 temperature=0 지원 안 함 → 0.7 사용
                colleague_temp = 0.7 if temperature == 0.0 else temperature
                return await self._handle_profile_chat("colleague1", self.colleague1_agent, "미안하네, 다시 한번 말해주겠나?", corrected_message, session_id, colleague_temp, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model == "colleague2":
                # gpt-5-mini는 temperature=0 지원 안 함 → 0.7 사용
                colleague_temp = 0.7 if temperature == 0.0 else temperature
                return await self._handle_profile_chat("colleague2", self.colleague2_agent, "선생님, 다시 한 번만 말씀해주시겠어요?", corrected_message, session_id, colleague_temp, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model in ["jangmo", "jangmo-agent"] and self.jangmo_agent:
                return await self._handle_profile_chat("jangmo", self.jangmo_agent, "미안해, 다시 한번 말해줄래?", corrected_message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model in ["son", "son-agent"] and self.son_agent:
                return await self._handle_profile_chat("son", self.son_agent, "아버지, 다시 한번 말씀해주세요.", corrected_message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            agent = self._get_agent(model)
            state = self._get_or_reset_state(session_id, force_reset, is_game_server)

            # 🔍 DEBUG: State before processing
            logger.info(f"🔍 [CHAT] session_id={session_id[:8]}..., model={model}, is_first_message={is_first_message}")
            logger.info(f"🔍 [CHAT] State BEFORE process: stage={state.get('stage')}, message_count={state.get('message_count')}, messages_len={len(state.get('messages', []))}")
            logger.info(f"🔍 [CHAT] State BEFORE process: stage2_complete={state.get('stage2_complete')}, stage2_completed={state.get('stage2_completed')}")

            if is_first_message:
                return await self._handle_first_message(agent, corrected_message, session_id, state, include_audio, voice, is_game_server)

            # ✨ 세션이 비어있는데 is_first_message=False로 들어온 경우 (NestJS가 /start 없이 바로 /chat 호출)
            # → 인사를 먼저 세션에 추가한 후 사용자 메시지 처리
            if len(state.get("messages", [])) == 0 and not state.get("stage1_greeting_sent", False):
                initial_greeting = agent.get_initial_message()
                state["messages"].append({"role": "assistant", "content": initial_greeting})
                state["stage1_greeting_sent"] = True
                logger.info(f"🔔 [CHAT] Auto-injected initial greeting for empty session")

            # ✨ thread_id 전달 (연속대화 자동 관리)
            result_state = agent.process(state, corrected_message, thread_id=session_id)
            response_text = result_state.get("last_response", "")
            sessions = self._get_sessions(is_game_server)
            sessions[session_id] = result_state

            # 🔍 DEBUG: State after processing
            logger.info(f"🔍 [CHAT] State AFTER process: stage={result_state.get('stage')}, message_count={result_state.get('message_count')}, messages_len={len(result_state.get('messages', []))}")
            logger.info(f"🔍 [CHAT] State AFTER process: stage2_complete={result_state.get('stage2_complete')}, stage2_completed={result_state.get('stage2_completed')}")

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
        external_messages: List[Dict[str, str]] = None,  # ✨ 외부에서 전달받은 메시지 히스토리
        include_audio: bool = False,  # ✨ TTS 지원
        voice: str = "alloy"  # ✨ TTS 음성
    ) -> AsyncGenerator[str, None]:
        if not is_first_message and (not message or not message.strip()):
            raise ValueError("Message cannot be empty")

        # ⚠️ 시스템 프롬프트 감지 시 로깅만 하고 정상 처리 (사용자 응답은 항상 제공)
        if self.is_system_prompt(message):
            logger.warning(f"⚠️ [CHAT_STREAM] Possible system prompt injection detected: {message[:50]}...")

        corrected_message = await self._normalize_user_message_llm(message)

        try:
            if model == "moral-agent-spt":
                async for token in self._handle_spt_stream(corrected_message, session_id, temperature, max_tokens, is_first_message):
                    yield token
                return

            if model == "colleague1":
                # gpt-5-mini는 temperature=0 지원 안 함 → 0.7 사용
                colleague_temp = 0.7 if temperature == 0.0 else temperature
                async for token in self._handle_profile_stream("colleague1", self.colleague1_agent, "미안하네, 다시 한번 말해주겠나?", corrected_message, session_id, colleague_temp, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            if model == "colleague2":
                # gpt-5-mini는 temperature=0 지원 안 함 → 0.7 사용
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
                initial_message = agent.get_initial_message()
                user_content = corrected_message if corrected_message else "[시작]"

                state["messages"].append({"role": "user", "content": user_content})
                state["messages"].append({"role": "assistant", "content": initial_message})
                state["last_response"] = initial_message
                state["message_count"] = 1
                self.sessions[session_id] = state

                for char in initial_message:
                    yield char
                return

            # ✨ thread_id 전달 (연속대화 자동 관리)
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
        """
        게임 서버 세션 초기화 (새 게임 시작 시 사용)
        - context_id에 해당하는 세션을 완전한 초기 상태로 리셋
        """
        # 일반 세션: 삭제가 아니라 초기 상태로 설정
        self.sessions_game[context_id] = self.get_initial_state()
        logger.info(f"Reset game session to initial state: {context_id}")

        # 프로필 세션 클리어 (colleague1, colleague2 등)
        keys_to_delete = [key for key in self.profile_sessions_game if key[1] == context_id]
        for key in keys_to_delete:
            del self.profile_sessions_game[key]
            logger.info(f"Cleared profile session: {key}")

        return True

    def migrate_game_session(self, old_context_id: str, new_context_id: str) -> bool:
        """
        게임 세션을 이전 context에서 새 context로 이전 (대화 연속성 유지)
        - 클라이언트가 context_id를 바꿔도 대화가 이어지도록 함
        """
        # 이전 세션이 있으면 새 context로 복사 (깊은 복사로 상태 완전 복제)
        if old_context_id in self.sessions_game:
            old_state = self.sessions_game[old_context_id]
            self.sessions_game[new_context_id] = copy.deepcopy(old_state)
            # 이전 세션 삭제
            del self.sessions_game[old_context_id]
            logger.info(f"🔄 Migrated game session: {old_context_id[:8]}... → {new_context_id[:8]}... (messages: {len(old_state.get('messages', []))})")
        else:
            logger.info(f"🔄 No session to migrate from {old_context_id[:8]}...")

        # 프로필 세션은 이전하지 않고 삭제 (새 context에서는 새로운 대화 시작)
        keys_to_delete = [key for key in self.profile_sessions_game if key[1] == old_context_id]
        for old_key in keys_to_delete:
            agent_key = old_key[0]
            del self.profile_sessions_game[old_key]
            logger.info(f"🔄 Cleared profile session for new context: {agent_key}")

        return True

    def clear_all_game_sessions(self) -> bool:
        """
        모든 게임 서버 세션 초기화 (context_id 변경 시 사용)
        - 메모리 누수 방지 + 이전 세션 상태가 남아있는 문제 해결
        """
        session_count = len(self.sessions_game)
        profile_count = len(self.profile_sessions_game)

        self.sessions_game.clear()
        self.profile_sessions_game.clear()

        logger.info(f"🧹 Cleared ALL game sessions: {session_count} sessions, {profile_count} profile sessions")
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
            # refine 제거됨
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

    def _format_history_for_refinement(self, history: List[Dict[str, str]], limit: int = 6) -> str:
        if not history:
            return ""

        relevant = history[-limit:]
        lines = []
        for msg in relevant:
            role = msg.get("role", "assistant")
            speaker = "사용자" if role == "user" else "에이전트"
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    async def _refine_spt_response(self, history: List[Dict[str, str]], draft_response: str) -> str:
        if not draft_response or not draft_response.strip():
            return draft_response

        history_text = self._format_history_for_refinement(history)
        prompt_body = (
            "다음은 사용자와 윤리 상담 에이전트의 최근 대화 일부입니다.\n"
            f"{history_text or '기록 없음'}\n\n"
            "위 흐름을 참고하여 에이전트가 방금 작성한 초안 응답을 자연스럽고 사람다운 한국어로 2~3문장에 맞춰 다듬어 주세요. "
            "초안의 핵심 의미는 유지하되 존댓말을 쓰고, 새로운 주장이나 사실을 추가하지 마세요.\n\n"
            f"초안 응답:\n{draft_response.strip()}\n"
        )

        try:
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 대화 흐름을 자연스럽게 손보는 한국어 작가입니다. "
                                   "원문과 의미를 같게 유지하고 존댓말로 2~3문장만 출력하세요."
                    },
                    {"role": "user", "content": prompt_body}
                ],
                temperature=0.4,
                max_tokens=300
            )
            refined = completion.choices[0].message.content.strip()
            return refined or draft_response
        except Exception as e:
            logger.warning(f"SPT refinement failed: {e}")
            return draft_response

    async def _refine_colleague_response(self, agent_key: str, history: List[Dict[str, str]], draft_response: str) -> str:
        """Colleague 에이전트 응답을 다듬기 - 사용자 발언 인용 제거"""
        if not draft_response or not draft_response.strip():
            return draft_response

        # 너무 짧거나 의미없는 응답은 refine 스킵
        stripped = draft_response.strip()
        if len(stripped) < 10 or stripped in ["...", "…", ".", "..", "..."]:
            logger.warning(f"🔧 [REFINE] Skipping refine - draft too short or meaningless: '{stripped}'")
            return draft_response

        history_text = self._format_history_for_refinement(history)

        if agent_key == "colleague1":
            system_content = (
                "당신은 50대 여성 화가로, AI 예술 전시를 책임과 규정 중심으로 반대합니다. "
                "반말을 사용하며 '자네'는 가끔만 씁니다.\n\n"
                "⚠️ 사용자 질문에 답변 의무 (최우선!):\n"
                "- 사용자가 '그게 뭐야?', '무슨 뜻이야?', '왜?' 같이 명확화를 요청하면 반드시 답변하세요\n"
                "- 이전 AI 발언에 대한 질문은 off-topic이 아닙니다! 반드시 설명하세요\n"
                "- 질문에 답하지 않고 '지금 중요한 건...'으로 돌리면 안 됩니다\n\n"
                "⚠️ 주제 유지 (off-topic 판단 기준):\n"
                "- off-topic = 날씨, 음식, 취미, 개인적 질문 등 AI 예술과 완전히 무관한 주제\n"
                "- AI 예술/전시/투표에 대한 질문, 명확화 요청, 의견 질문 = on-topic (답변 필수!)\n"
                "- 완전히 무관한 주제일 때만 '지금 중요한 건 AI 전시 투표야'라고 되돌리세요\n\n"
                "⚠️ 절대 금지 사항:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요 (예: '자네가 ~라고 했는데' 금지)\n"
                "- 사용자 발언을 그대로 반복하거나 요약하지 마세요\n"
                "- '방금', '아까' 같은 표현으로 사용자 말을 언급하지 마세요\n"
                "- '의무론', '의무론적', '공리주의', 'utilitarian' 단어는 절대 쓰지 마세요\n\n"
                "⚠️ 문장 구조:\n"
                "- 항상 3문장(재진술 → 근거 → 개방형 질문)으로 답하세요\n"
                "- 마지막 문장은 YES/NO로 답할 수 없는 질문이며, 방금 사용자가 드러낸 망설임·감정·근거 부족을 더 묻게 만드세요\n\n"
                "올바른 응답 방식:\n"
                "1. AI 예술에 대한 자신의 입장과 근거를 먼저 명확히 밝히세요\n"
                "2. 필요시 AI 예술 관련 짧은 후속 질문을 던지세요\n"
                "3. 자연스러운 대화처럼 말하세요"
            )
        elif agent_key == "jangmo":
            system_content = (
                "당신은 노인 여성 장모로, AI로 죽은 딸을 복원하는 일을 책임과 예의 관점에서 반대합니다. "
                "사위에게 반말을 사용합니다.\n\n"
                "⚠️ 사용자 질문에 답변 의무 (최우선!):\n"
                "- 사용자가 '그게 뭐야?', '무슨 뜻이야?', '왜?' 같이 명확화를 요청하면 반드시 답변하세요\n"
                "- 이전 AI 발언에 대한 질문은 off-topic이 아닙니다! 반드시 설명하세요\n\n"
                "⚠️ 반복 금지:\n"
                "- 이전에 말한 내용을 절대 반복하지 마세요\n"
                "- 매번 새로운 관점, 새로운 논거, 새로운 질문을 제시하세요\n"
                "- '그리움', '기억', '소중히' 같은 표현을 반복하지 마세요\n\n"
                "⚠️ 주제 유지 (off-topic 판단 기준):\n"
                "- off-topic = 날씨, 음식, 취미 등 AI 복원과 완전히 무관한 주제\n"
                "- AI 복원/딸 복원/가족 결정에 대한 질문 = on-topic (답변 필수!)\n\n"
                "⚠️ 절대 금지 사항:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요\n"
                "- '의무론', '공리주의' 단어 금지\n\n"
                "올바른 응답 방식:\n"
                "1. 사용자의 마지막 발언에 직접 반응하세요\n"
                "2. 새로운 각도에서 AI 복원 반대 논거를 제시하세요\n"
                "3. 2문장 이내로 답하세요"
            )
        elif agent_key == "son":
            system_content = (
                "당신은 20대 초반 남성 청년으로, AI로 죽은 어머니를 복원하는 일을 책임 중심 관점에서 찬성합니다. "
                "⚠️ 아버지에게 반드시 존댓말(-요, -세요, -습니다)을 사용합니다. 반말 절대 금지!\n\n"
                "⚠️ 사용자 질문에 답변 의무 (최우선!):\n"
                "- 사용자가 '그게 뭐야?', '무슨 뜻이야?', '왜?' 같이 명확화를 요청하면 반드시 답변하세요\n"
                "- 이전 AI 발언에 대한 질문은 off-topic이 아닙니다! 반드시 설명하세요\n\n"
                "⚠️ 존댓말 필수:\n"
                "- 모든 문장은 '-요', '-세요', '-습니다'로 끝나야 합니다\n"
                "- 반말 금지: '~해', '~야', '~거야', '~지', '~네' 등\n"
                "- 예시: '그렇게 생각해' (X) → '그렇게 생각해요' (O)\n"
                "- 예시: '결정이야' (X) → '결정이에요' (O)\n\n"
                "⚠️ 주제 유지 (off-topic 판단 기준):\n"
                "- off-topic = 날씨, 음식, 취미 등 AI 복원과 완전히 무관한 주제\n"
                "- AI 복원/어머니 복원/가족 결정에 대한 질문 = on-topic (답변 필수!)\n"
                "- 완전히 무관한 주제일 때만 '지금 중요한 건 엄마 복원 결정이에요'라고 되돌리세요\n\n"
                "⚠️ 절대 금지 사항:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요\n"
                "- '의무론', '공리주의' 단어 금지\n\n"
                "올바른 응답 방식:\n"
                "1. AI 복원에 대한 자신의 입장과 근거를 먼저 명확히 밝히세요\n"
                "2. 필요시 AI 복원 관련 짧은 후속 질문을 던지세요\n"
                "3. 자연스러운 대화처럼 말하되 반드시 존댓말을 사용하세요"
            )
        else:  # colleague2
            system_content = (
                "당신은 30대 남성 화가로, AI 예술 전시를 책임과 규정 일관성 관점에서 찬성합니다. "
                "항상 존댓말을 쓰며 상대를 '선생님'이라고 부르세요.\n\n"
                "⚠️ 언어/길이 규칙:\n"
                "- '의무론', '의무론적', '공리주의', 'utilitarian' 단어는 절대 쓰지 마세요\n"
                "- 답변은 항상 3문장 이하, 문장당 15단어 내외로 유지하세요\n\n"
                "⚠️ 사용자 질문에 답변 의무 (최우선!):\n"
                "- 사용자가 '그게 뭐야?', '무슨 뜻이야?', '왜?' 같이 명확화를 요청하면 반드시 답변하세요\n"
                "- 이전 AI 발언에 대한 질문은 off-topic이 아닙니다! 반드시 설명하세요\n"
                "- 질문에 답하지 않고 '지금 중요한 건...'으로 돌리면 안 됩니다\n\n"
                "⚠️ 주제 유지 (off-topic 판단 기준):\n"
                "- off-topic = 날씨, 음식, 취미, 개인적 질문 등 AI 예술과 완전히 무관한 주제\n"
                "- AI 예술/전시/투표에 대한 질문, 명확화 요청, 의견 질문 = on-topic (답변 필수!)\n"
                "- 완전히 무관한 주제일 때만 '지금 중요한 건 AI 전시 투표입니다'라고 되돌리세요\n\n"
                "⚠️ 절대 금지:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요\n"
                "- 사용자 발언을 그대로 반복하거나 요약하지 마세요\n"
                "- '방금', '아까' 같은 표현으로 사용자 말을 언급하지 마세요\n\n"
                "올바른 응답 방식:\n"
                "1. AI 예술에 대한 자신의 의무 중심 입장과 근거를 명확히 밝히세요\n"
                "2. 필요시 협회 책임 또는 제자·관객 영향을 짚으며 짧은 후속 질문을 던지세요\n"
                "3. 자연스럽고 단정한 대화처럼 말하세요"
            )

        prompt_body = (
            f"최근 대화:\n{history_text or '기록 없음'}\n\n"
            f"에이전트 초안 응답:\n{draft_response.strip()}\n\n"
            "⚠️ 중요: 초안의 핵심 내용과 어조를 그대로 유지하세요!\n"
            "- 사용자 발언을 따옴표로 직접 인용한 부분만 수정하세요 (예: '당신이 ~라고 했는데')\n"
            "- 그 외 내용은 절대 변경하지 마세요\n"
            "- 새로운 문장을 추가하거나 내용을 확장하지 마세요\n"
            "- 인용 부분이 없다면 초안을 그대로 출력하세요"
        )

        try:
            logger.info(f"🔧 [REFINE] Starting refine for {agent_key}, draft_length={len(draft_response)}")
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 속도 개선: gpt-4.1-mini → gpt-4o-mini
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt_body}
                ],
                temperature=0.4,
                max_completion_tokens=300
            )
            refined = completion.choices[0].message.content.strip()
            refined = self._limit_sentences(refined)
            logger.info(f"🔧 [REFINE] Completed for {agent_key}, refined_length={len(refined)}")
            return refined or self._limit_sentences(draft_response)
        except Exception as e:
            logger.warning(f"🔧 [REFINE] Failed for {agent_key}: {e}")
            return self._limit_sentences(draft_response)

    def _get_profile_session(self, agent_key: str, session_id: str, is_game_server: bool = False) -> Dict[str, Any]:
        profile_sessions = self._get_profile_sessions(is_game_server)
        key = (agent_key, session_id)

        # 🔍 DEBUG: 세션 조회 상태 로깅
        store_name = "game" if is_game_server else "openwebui"
        all_keys = list(profile_sessions.keys())
        logger.info(f"🔍 [GET_PROFILE_SESSION] Looking for key={key} in {store_name} store")
        logger.info(f"🔍 [GET_PROFILE_SESSION] Available keys: {all_keys[:5]}... (total: {len(all_keys)})")

        if key not in profile_sessions:
            # 다른 스토어에서도 찾아보기 (is_game_server 불일치 방지)
            other_sessions = self._get_profile_sessions(not is_game_server)
            other_store_name = "openwebui" if is_game_server else "game"
            if key in other_sessions:
                logger.warning(f"⚠️ [GET_PROFILE_SESSION] Found session in {other_store_name} store instead! Copying to {store_name}")
                profile_sessions[key] = other_sessions[key]
            else:
                logger.info(f"🆕 [GET_PROFILE_SESSION] Creating new session for key={key}")
                profile_sessions[key] = {"messages": [], "turn_count": 0}
        else:
            logger.info(f"✅ [GET_PROFILE_SESSION] Found existing session, messages={len(profile_sessions[key].get('messages', []))}, turn_count={profile_sessions[key].get('turn_count', 0)}")

        return profile_sessions[key]

    def load_history_from_db(self, session_id: str, messages: List[Dict[str, str]] = None, history: List[Dict[str, str]] = None, turn_count: int = None, agent_key: str = None, is_game_server: bool = True):
        """
        DB에서 가져온 히스토리를 세션에 로드 (이어하기)

        두 가지 형식 지원:
        1. messages: [{"role": "user/assistant", "content": "..."}, ...] - server.py에서 직접 호출
        2. history: [{"user_message": "...", "ai_message": "..."}, ...] - 기존 형식

        - agent_key: colleague1, colleague2 등 (프로필 세션용)
        - 이어하기 시: DB 메시지 복원 + Stage는 메시지 기반으로 추론
        """
        # messages 형식이 직접 전달된 경우
        if messages is not None:
            formatted_messages = messages
            msg_count = turn_count if turn_count else len(messages)
        elif history is not None:
            # 기존 history 형식 변환
            formatted_messages = []
            for row in history:
                formatted_messages.append({"role": "user", "content": row["user_message"]})
                formatted_messages.append({"role": "assistant", "content": row["ai_message"]})
            msg_count = len(history)
        else:
            logger.error("🔍 [LOAD_HISTORY] No messages or history provided")
            return

        logger.info(f"🔍 [LOAD_HISTORY] ENTRY - session_id={session_id[:8]}..., message_count={len(formatted_messages)}, agent_key={agent_key}")

        if agent_key:
            # 프로필 세션 (colleague1, colleague2)
            profile_sessions = self._get_profile_sessions(is_game_server)
            key = (agent_key, session_id)
            profile_sessions[key] = {"messages": formatted_messages, "turn_count": msg_count}
            logger.info(f"🔍 [LOAD_HISTORY] Loaded {msg_count} turns from DB into profile session: {key}")
        else:
            # 일반 세션: 메시지 복원 + Stage 추론
            sessions = self._get_sessions(is_game_server)
            state = self.get_initial_state()

            # 메시지와 카운트 복원
            state["messages"] = formatted_messages
            state["message_count"] = msg_count

            # Stage 추론: 대화 내용 기반으로 현재 Stage 판단
            stage, stage_info = self._infer_stage_from_messages(formatted_messages)
            state["stage"] = stage
            state["previous_stage"] = stage

            # Stage별 플래그 설정
            if stage == "stage2":
                state["artist_character_set"] = True
                state["stage2_question_asked"] = True
            elif stage == "stage3":
                state["artist_character_set"] = True
                state["stage2_question_asked"] = True
                state["stage2_complete"] = True
                state["stage2_completed"] = True

            sessions[session_id] = state
            logger.info(f"✅ [LOAD_HISTORY] Session restored - messages={len(formatted_messages)}, stage={stage}, info={stage_info}")

    def load_messages_only(self, session_id: str, messages: List[Dict[str, str]], is_game_server: bool = True):
        """
        메시지만 로드 (Stage는 stage1 유지)
        - 새 게임은 항상 Stage1부터 시작
        - DB 메시지는 LLM 컨텍스트용으로만 사용
        """
        sessions = self._get_sessions(is_game_server)
        state = self.get_initial_state()  # stage1부터 시작
        state["messages"] = messages
        state["message_count"] = len(messages)
        sessions[session_id] = state
        logger.info(f"✅ [LOAD_MESSAGES] Loaded {len(messages)} messages for context: {session_id[:8]}... (Stage=stage1)")

    def _infer_stage_from_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """
        대화 내용을 기반으로 현재 Stage 추론

        키워드 기반 판단:
        - Stage2 질문: "그림 그리는 AI", "국립현대예술관"
        - Stage3 질문: "이익", "피해", "대안"

        Returns:
            (stage, info): 추론된 stage와 판단 근거
        """
        if not messages:
            return "stage1", "no_messages"

        # 모든 assistant 메시지를 합쳐서 분석
        assistant_messages = " ".join([
            msg["content"] for msg in messages
            if msg.get("role") == "assistant"
        ])

        # Stage3 키워드 (Stage2보다 먼저 체크 - Stage3가 더 뒤이므로)
        stage3_keywords = ["어떤 이익", "어떤 피해", "대안", "이익이 있을까", "피해가 있을까"]
        for keyword in stage3_keywords:
            if keyword in assistant_messages:
                return "stage3", f"keyword:{keyword}"

        # Stage2 키워드
        stage2_keywords = ["그림 그리는 AI", "국립현대예술관", "AI의 그림", "전시가 된대요"]
        for keyword in stage2_keywords:
            if keyword in assistant_messages:
                return "stage2", f"keyword:{keyword}"

        # 기본값: Stage1
        return "stage1", "default"

    def _trim_profile_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= 15:
            return messages
        first_message = messages[0]
        return [first_message] + messages[-14:]

    async def _handle_profile_chat(self, agent_key: str, agent, error_message: str, message: str, session_id: str, temperature: float, max_tokens: int, is_first_message: bool, include_audio: bool, voice: str, is_game_server: bool = False, external_messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        session_data = self._get_profile_session(agent_key, session_id, is_game_server)

        # ✨ 외부 메시지가 있고 세션이 비어있으면 외부 메시지로 초기화
        if external_messages and len(session_data["messages"]) == 0:
            # 시스템 메시지 필터링 (Open WebUI의 시스템 프롬프트 제외)
            filtered_messages = [
                msg for msg in external_messages
                if msg.get("role") in ["user", "assistant"]
                and not any(pattern in msg.get("content", "").lower() for pattern in [
                    "### task:", "suggest 3-5", "generate a concise", "generate 1-3"
                ])
            ]
            if filtered_messages:
                # 마지막 user 메시지는 제외 (아래에서 다시 추가됨)
                if filtered_messages[-1].get("role") == "user":
                    session_data["messages"] = filtered_messages[:-1]
                else:
                    session_data["messages"] = filtered_messages
                session_data["turn_count"] = len([m for m in session_data["messages"] if m.get("role") == "assistant"])
                logger.info(f"✅ [PROFILE_CHAT] Initialized from external_messages: {len(session_data['messages'])} messages, turn_count={session_data['turn_count']}")

        # 🔍 DEBUG: 세션 상태 로깅
        logger.info(f"🔍 [PROFILE_CHAT] agent_key={agent_key}, session_id={session_id[:8]}..., is_first_message={is_first_message}")
        logger.info(f"🔍 [PROFILE_CHAT] session_data messages count: {len(session_data['messages'])}, turn_count: {session_data.get('turn_count', 0)}")
        if session_data['messages']:
            logger.info(f"🔍 [PROFILE_CHAT] first message: {session_data['messages'][0] if session_data['messages'] else 'NONE'}")

        if is_first_message:
            if message:
                session_data["messages"].append({"role": "user", "content": message})

            initial_message = agent.get_initial_message()
            session_data["messages"].append({"role": "assistant", "content": initial_message})
            session_data["turn_count"] = 1
            result_dict = {
                "response": initial_message,
                "session_id": session_id,
                "metadata": {"stage": agent_key, "message_count": len(session_data["messages"]), "turn_count": 1, "is_end": False, "is_first": True}
            }
            if include_audio:
                try:
                    result_dict["audio"] = await self.text_to_speech(initial_message, voice=voice)
                except Exception:
                    pass
            return result_dict

        session_data["messages"].append({"role": "user", "content": message})
        session_data["turn_count"] += 1
        session_data["messages"] = self._trim_profile_history(session_data["messages"])

        # ✨ 10턴이면 마무리 응답 반환 (대화 종료)
        if session_data["turn_count"] >= 10 and agent_key not in ["friend", "artist_apprentice"]:
            logger.info(f"🔚 [PROFILE_CHAT] Turn 10 reached - returning final message for {agent_key}")
            final_message = agent.get_final_message() if hasattr(agent, 'get_final_message') else "대화해줘서 고마워요. 좋은 결정 내리시길 바랍니다."
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

        # 🔍 DEBUG: agent.chat() 호출 전 메시지 상태
        logger.info(f"🔍 [PROFILE_CHAT] Before agent.chat() - messages count: {len(session_data['messages'])}")
        for idx, msg in enumerate(session_data["messages"]):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:50]
            logger.info(f"🔍 [PROFILE_CHAT] msg[{idx}]: role={role}, content='{content}...'")

        try:
            # ✨ Step 0: 이전 AI 응답들에서 핵심 주장 추출 (반복 방지)
            previous_ai_claims = self._extract_previous_claims(session_data["messages"])

            # ✨ Step 1: SPT 에이전트로 도덕적 추론 생성 (이전 주장 포함하여 반복 방지)
            logger.info(f"🧠 [PROFILE_CHAT] Step 1: Calling SPT agent for moral reasoning...")

            # SPT 메시지에 반복 방지 지시 추가
            spt_messages = session_data["messages"].copy()
            if previous_ai_claims:
                anti_repeat_instruction = (
                    f"\n\n⚠️ 아래 논점은 이미 사용됨 (절대 반복 금지!):\n"
                    + "\n".join([f"- {claim[:80]}" for claim in previous_ai_claims[-3:]])
                    + "\n→ 완전히 새로운 관점/논점으로 질문하세요!"
                )
                # 마지막 user 메시지에 지시 추가
                if spt_messages and spt_messages[-1].get("role") == "user":
                    spt_messages[-1] = {
                        "role": "user",
                        "content": spt_messages[-1]["content"] + anti_repeat_instruction
                    }

            spt_draft = await self.spt_agent.chat(
                messages=spt_messages,
                temperature=0.7,
                max_tokens=300
            )
            logger.info(f"🧠 [PROFILE_CHAT] SPT draft: '{spt_draft[:100] if spt_draft else 'EMPTY'}...'")
            claims_warning = ""
            if previous_ai_claims:
                claims_list = "\n".join([f"- {claim}" for claim in previous_ai_claims[-3:]])  # 최근 3개만
                claims_warning = f"\n\n⚠️ 이미 말한 주장 (절대 반복 금지!):\n{claims_list}\n→ 위 내용과 다른 새로운 논점으로 답하세요!"

            # ✨ Step 3: 페르소나 에이전트로 정제 (SPT 초안을 강력한 지시로 전달)
            logger.info(f"🎭 [PROFILE_CHAT] Step 2: Calling persona agent ({agent_key}) for refinement...")
            refinement_messages = session_data["messages"][:-1].copy()  # 마지막 user 메시지 제외

            # SPT draft를 "지시"로 승격 (참고 → 필수 표현)
            spt_instruction = (
                f"📌 필수 지시사항:\n"
                f"1. 다음 관점을 당신의 말투로 반드시 표현하세요: {spt_draft}\n"
                f"2. 사용자 발화에 직접 반응하는 질문을 던지세요.\n"
                f"3. 같은 주장을 반복하지 마세요.\n"
                f"4. 반드시 한국어(반말 톤)로 답하세요.{claims_warning}"
            )

            refinement_messages.append({
                "role": "user",
                "content": f"{message}\n\n{spt_instruction}"
            })
            response_text = await agent.chat(
                messages=refinement_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=session_id
            )

            logger.info(f"🎭 [PROFILE_CHAT] Persona response: '{response_text[:100] if response_text else 'EMPTY'}' ({len(response_text) if response_text else 0} chars)")

            # refine 제거됨

            # ✨ 페르소나 검증 및 수정 (friend, artist_apprentice 제외)
            if HAS_PERSONA_VALIDATOR and agent_key not in ["friend", "artist_apprentice"]:
                response_text, was_fixed = await validate_and_fix_persona(
                    response_text, agent_key, settings.openai_api_key
                )
                if was_fixed:
                    logger.info(f"🔧 [PROFILE_CHAT] Persona fixed for {agent_key}")

            # 빈 응답이나 의미없는 응답 처리 → gpt-4o-mini로 재시도
            if not response_text or len(response_text.strip()) < 5 or response_text.strip() in ["...", "…"]:
                logger.warning(f"⚠️ [PROFILE_CHAT] Empty response detected, retrying with gpt-4o-mini...")
                response_text = await self._fallback_with_gpt4o_mini(agent_key, session_data["messages"], message)
                if not response_text or len(response_text.strip()) < 5:
                    response_text = error_message  # 최종 fallback

            session_data["messages"].append({"role": "assistant", "content": response_text})

            # friend, artist_apprentice는 Stage 기반이라 턴 제한 없음
            if agent_key in ["friend", "artist_apprentice"]:
                is_end = False
            else:
                is_end = session_data["turn_count"] >= 10

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

        # ✨ 외부 메시지가 있고 세션이 비어있으면 외부 메시지로 초기화
        if external_messages and len(session_data["messages"]) == 0:
            # 시스템 메시지 필터링 (Open WebUI의 시스템 프롬프트 제외)
            filtered_messages = [
                msg for msg in external_messages
                if msg.get("role") in ["user", "assistant"]
                and not any(pattern in msg.get("content", "").lower() for pattern in [
                    "### task:", "suggest 3-5", "generate a concise", "generate 1-3"
                ])
            ]
            if filtered_messages:
                # 마지막 user 메시지는 제외 (아래에서 다시 추가됨)
                if filtered_messages[-1].get("role") == "user":
                    session_data["messages"] = filtered_messages[:-1]
                else:
                    session_data["messages"] = filtered_messages
                session_data["turn_count"] = len([m for m in session_data["messages"] if m.get("role") == "assistant"])
                logger.info(f"✅ [PROFILE_STREAM] Initialized from external_messages: {len(session_data['messages'])} messages, turn_count={session_data['turn_count']}")

        if is_first_message:
            if message:
                session_data["messages"].append({"role": "user", "content": message})

            initial_message = agent.get_initial_message()
            session_data["messages"].append({"role": "assistant", "content": initial_message})
            session_data["turn_count"] = 1
            for char in initial_message:
                yield char
            # ✨ TTS 지원: 텍스트 스트리밍 후 오디오 데이터 전송
            if include_audio:
                try:
                    audio_data = await self.text_to_speech(initial_message, voice=voice)
                    yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                except Exception as e:
                    logger.warning(f"TTS failed for stream: {e}")
            return

        session_data["messages"].append({"role": "user", "content": message})
        session_data["turn_count"] += 1
        session_data["messages"] = self._trim_profile_history(session_data["messages"])

        if not max_tokens or max_tokens > 300:
            max_tokens = 300

        try:
            # ✨ Step 0: 이전 AI 응답들에서 핵심 주장 추출 (반복 방지)
            previous_ai_claims = self._extract_previous_claims(session_data["messages"])

            # ✨ Step 1: SPT 에이전트로 도덕적 추론 생성 (이전 주장 포함하여 반복 방지)
            logger.info(f"🧠 [PROFILE_STREAM] Step 1: Calling SPT agent for moral reasoning...")

            # SPT 메시지에 반복 방지 지시 추가
            spt_messages = session_data["messages"].copy()
            if previous_ai_claims:
                anti_repeat_instruction = (
                    f"\n\n⚠️ 아래 논점은 이미 사용됨 (절대 반복 금지!):\n"
                    + "\n".join([f"- {claim[:80]}" for claim in previous_ai_claims[-3:]])
                    + "\n→ 완전히 새로운 관점/논점으로 질문하세요!"
                )
                # 마지막 user 메시지에 지시 추가
                if spt_messages and spt_messages[-1].get("role") == "user":
                    spt_messages[-1] = {
                        "role": "user",
                        "content": spt_messages[-1]["content"] + anti_repeat_instruction
                    }

            spt_draft = await self.spt_agent.chat(
                messages=spt_messages,
                temperature=0.7,
                max_tokens=300
            )
            logger.info(f"🧠 [PROFILE_STREAM] SPT draft: '{spt_draft[:100] if spt_draft else 'EMPTY'}...'")
            claims_warning = ""
            if previous_ai_claims:
                claims_list = "\n".join([f"- {claim}" for claim in previous_ai_claims[-3:]])  # 최근 3개만
                claims_warning = f"\n\n⚠️ 이미 말한 주장 (절대 반복 금지!):\n{claims_list}\n→ 위 내용과 다른 새로운 논점으로 답하세요!"

            # ✨ Step 3: 페르소나 에이전트로 정제 (SPT 초안을 강력한 지시로 전달)
            logger.info(f"🎭 [PROFILE_STREAM] Step 2: Calling persona agent ({agent_key}) for refinement...")
            refinement_messages = session_data["messages"][:-1].copy()  # 마지막 user 메시지 제외

            # SPT draft를 "지시"로 승격 (참고 → 필수 표현)
            spt_instruction = (
                f"📌 필수 지시사항:\n"
                f"1. 다음 관점을 당신의 말투로 반드시 표현하세요: {spt_draft}\n"
                f"2. 사용자 발화에 직접 반응하는 질문을 던지세요.\n"
                f"3. 같은 주장을 반복하지 마세요.{claims_warning}"
            )

            refinement_messages.append({
                "role": "user",
                "content": f"{message}\n\n{spt_instruction}"
            })
            full_response = await agent.chat(
                messages=refinement_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=session_id
            )
            logger.info(f"🎭 [PROFILE_STREAM] Persona response: '{full_response[:100] if full_response else 'EMPTY'}...'")

            # refine 제거됨
            refined_response = full_response

            # ✨ 페르소나 검증 및 수정 (friend, artist_apprentice 제외)
            if HAS_PERSONA_VALIDATOR and agent_key not in ["friend", "artist_apprentice"]:
                refined_response, was_fixed = await validate_and_fix_persona(
                    refined_response, agent_key, settings.openai_api_key
                )
                if was_fixed:
                    logger.info(f"🔧 [PROFILE_STREAM] Persona fixed for {agent_key}")

            # 빈 응답이나 의미없는 응답 처리 → gpt-4o-mini로 재시도
            if not refined_response or len(refined_response.strip()) < 5 or refined_response.strip() in ["...", "…"]:
                logger.warning(f"⚠️ [PROFILE_STREAM] Empty response detected, retrying with gpt-4o-mini...")
                refined_response = await self._fallback_with_gpt4o_mini(agent_key, session_data["messages"], message)
                if not refined_response or len(refined_response.strip()) < 5:
                    refined_response = error_message  # 최종 fallback

            # refine된 응답을 스트림으로 전송
            for char in refined_response:
                yield char

            # ✨ TTS 지원: 텍스트 스트리밍 후 오디오 데이터 전송
            if include_audio:
                try:
                    audio_data = await self.text_to_speech(refined_response, voice=voice)
                    yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                except Exception as e:
                    logger.warning(f"TTS failed for stream: {e}")

            session_data["messages"].append({"role": "assistant", "content": refined_response})
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
            # refine 제거됨
            session_data["messages"].append({"role": "assistant", "content": response_text})
            for char in response_text:
                yield char
        except Exception as e:
            logger.error(f"SPT stream error: {e}")
            error_message = "죄송해요, 다시 한번 말씀해주시겠어요?"
            for char in error_message:
                yield char

    async def text_to_speech(self, text: str, voice: str = "alloy", model: str = "tts-1") -> str:
        """
        텍스트를 음성으로 변환 (OpenAI TTS API)

        Args:
            text: 변환할 텍스트
            voice: 음성 종류 (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS 모델 (tts-1, tts-1-hd)

        Returns:
            base64 인코딩된 MP3 오디오 데이터
        """
        logger.info(f"🎙️ [TTS] Starting: text_length={len(text)}, voice={voice}, model={model}")
        logger.debug(f"🎙️ [TTS] Text preview: '{text[:100]}...'")

        if not text or not text.strip():
            logger.warning("🎙️ [TTS] Empty text received, skipping TTS")
            raise ValueError("Empty text cannot be converted to speech")

        try:
            logger.info(f"🎙️ [TTS] Calling OpenAI API...")
            response = await self.openai_client.audio.speech.create(model=model, voice=voice, input=text)

            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            logger.info(f"🎙️ [TTS] ✅ Success: audio_size={len(audio_bytes)} bytes, base64_length={len(audio_base64)}")
            return audio_base64
        except Exception as e:
            logger.error(f"🎙️ [TTS] ❌ Error: {type(e).__name__}: {e}")
            logger.error(f"🎙️ [TTS] ❌ Text was: '{text[:200]}...'")
            raise

    async def _fallback_with_gpt4o_mini(self, agent_key: str, history: List[Dict[str, str]], user_message: str) -> str:
        """빈 응답 시 gpt-4o-mini로 fallback 응답 생성"""
        history_text = self._format_history_for_refinement(history, limit=4)

        if agent_key == "colleague1":
            system_content = (
                "당신은 50대 여성 화가로, AI 예술 전시를 반대합니다. "
                "반말을 사용하며 의무·책임 관점에서 말합니다. 2문장으로 답하세요."
            )
        elif agent_key == "colleague2":
            system_content = (
                "당신은 30대 남성 화가로, AI 예술 전시를 찬성합니다. "
                "존댓말을 쓰며 '선생님'이라고 부릅니다. 2문장으로 답하세요."
            )
        elif agent_key == "jangmo":
            system_content = (
                "당신은 노인 여성 장모로, AI로 딸을 복원하는 것을 반대합니다. "
                "사위에게 반말을 사용합니다. 2문장으로 답하세요."
            )
        elif agent_key == "son":
            system_content = (
                "당신은 20대 남성으로, AI로 어머니를 복원하는 것을 찬성합니다. "
                "⚠️ 아버지에게 반드시 존댓말(-요, -세요, -습니다)을 사용합니다. 반말 절대 금지! 2문장으로 답하세요."
            )
        else:
            system_content = "2문장으로 자연스럽게 답하세요."

        try:
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"대화 기록:\n{history_text}\n\n사용자: {user_message}\n\n위 대화에 자연스럽게 응답하세요."}
                ],
                temperature=0.7,
                max_tokens=200
            )
            response = completion.choices[0].message.content.strip()
            logger.info(f"🔄 [FALLBACK] gpt-4o-mini response: '{response[:50]}...'")
            return response
        except Exception as e:
            logger.error(f"🔄 [FALLBACK] gpt-4o-mini failed: {e}")
            return ""

    @staticmethod
    def _extract_previous_claims(messages: List[Dict[str, str]]) -> List[str]:
        """이전 AI 응답들에서 전체 응답을 추출 (반복 방지용)"""
        claims = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "").strip()
                if content and len(content) > 10:
                    # 전체 응답을 추출 (150자로 제한)
                    claims.append(content[:150] + ("..." if len(content) > 150 else ""))
        return claims

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
