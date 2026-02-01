from typing import Optional, Dict, Any, AsyncGenerator, List
import logging
import base64
import re

from config import settings
from exceptions import APIKeyNotFoundError, ChainExecutionError
from agents.artist_apprentice_agent.conversation_agent import ConversationAgent as ArtistApprenticeAgent
from agents.friend_agent.conversation_agent import ConversationAgent as FriendAgent
from agents.spt_agent import SPTAgent, SPTAgentV2
from agents.colleague1_agent import Colleague1Agent
from agents.colleague2_agent import Colleague2Agent
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
from agents.db import DatabaseService
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

        self.artist_apprentice_agent = ArtistApprenticeAgent(api_key=api_key, model="gpt-5-nano-2025-08-07")
        self.friend_agent = FriendAgent(api_key=api_key, model="gpt-4o")
        self.spt_agent = SPTAgent(api_key=api_key)  # fine-tuned 모델 사용 (V1)
        self.spt_agent_v2 = SPTAgentV2(api_key=api_key)  # 새 아키텍처 (V2: DST + Planner + Controller)

        # ✨ 4개 Persona Agents에 SPT V2 주입 + Fine-tuned 모델 사용
        self.colleague1_agent = Colleague1Agent(
            api_key=api_key,
            model="ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs",
            spt_agent_v2=self.spt_agent_v2
        )
        self.colleague2_agent = Colleague2Agent(
            api_key=api_key,
            model="ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs",
            spt_agent_v2=self.spt_agent_v2
        )
        self.jangmo_agent = JangmoAgent(
            api_key=api_key,
            model="ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs",
            spt_agent_v2=self.spt_agent_v2
        ) if HAS_JANGMO else None
        self.son_agent = SonAgent(
            api_key=api_key,
            model="ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs",
            spt_agent_v2=self.spt_agent_v2
        ) if HAS_SON else None

        self.agent = self.artist_apprentice_agent
        self.db_service = DatabaseService()
        self.openai_client = AsyncOpenAI(api_key=api_key)

    def _get_initial_message(self, agent) -> str:
        if hasattr(agent, "get_initial_message"):
            return agent.get_initial_message()
        return ""

    def _get_profile_initial_message(self, agent_key: str, agent) -> str:
        if hasattr(agent, "get_initial_message"):
            return agent.get_initial_message()
        return self.PROFILE_INITIAL_MESSAGES.get(agent_key, "")

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
            "stage2_question_asked": False,
            "stage2_complete": False,
            "stage2_completed": False,
            "current_question_index": 0,
            "variation_index": 0,
            "dont_know_count": 0
        }

    def get_or_create_state(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = self.get_initial_state()
        return self.sessions[session_id]

    def is_system_prompt(self, message: str) -> bool:
        keywords = ["너는", "당신은", "assistant", "your role", "you are"]
        return any(kw in message.lower() for kw in keywords)

    def _get_agent(self, model: str):
        if model in ["friend-agent", "friend_agent"]:
            return self.friend_agent
        elif model in ["artist-apprentice", "artist_apprentice"]:
            return self.artist_apprentice_agent
        elif model in ["colleague1", "colleague-1"]:
            return self.colleague1_agent
        elif model in ["colleague2", "colleague-2"]:
            return self.colleague2_agent
        elif model in ["jangmo", "jangmo-agent"] and self.jangmo_agent:
            return self.jangmo_agent
        elif model in ["son", "son-agent"] and self.son_agent:
            return self.son_agent
        else:
            return self.artist_apprentice_agent

    def _get_or_reset_state(self, session_id: str, force_reset: bool) -> Dict[str, Any]:
        if force_reset:
            state = self.get_initial_state()
            self.sessions[session_id] = state
            return state

        state = self.get_or_create_state(session_id)
        if state.get("should_end", False) or state.get("stage") == "end":
            state = self.get_initial_state()
            self.sessions[session_id] = state
        return state

    async def _handle_first_message(self, agent, message: str, session_id: str, state: Dict[str, Any], include_audio: bool, voice: str) -> Dict[str, Any]:
        initial_message = self._get_initial_message(agent)
        user_content = message if message else "[시작]"

        state["messages"].append({"role": "user", "content": user_content})
        state["messages"].append({"role": "assistant", "content": initial_message})
        state["last_response"] = initial_message
        state["message_count"] = 1
        self.sessions[session_id] = state

        try:
            self.db_service.save_conversation_turn(
                session_id=session_id,
                user_message=user_content,
                assistant_response=initial_message,
                stage="stage1",
                covered_topics=[],
                metadata={"message_count": 1, "is_end": False, "is_first": True}
            )
            self.db_service.update_session_metadata(
                session_id=session_id,
                stage="stage1",
                total_messages=1,
                covered_topics_count=0,
                is_completed=False
            )
        except Exception as db_error:
            logger.error(f"DB save failed: {db_error}")

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

    def _save_to_db(self, session_id: str, user_message: str, assistant_response: str, state: Dict[str, Any]):
        try:
            self.db_service.save_conversation_turn(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                stage=state.get("stage", "unknown"),
                covered_topics=state.get("covered_topics", []),
                metadata={"message_count": state.get("message_count", 0), "is_end": state.get("should_end", False)}
            )
            self.db_service.update_session_metadata(
                session_id=session_id,
                stage=state.get("stage", "unknown"),
                total_messages=state.get("message_count", 0),
                covered_topics_count=len(state.get("covered_topics", [])),
                is_completed=state.get("should_end", False)
            )
        except Exception as db_error:
            logger.error(f"DB save failed: {db_error}")

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        model: str = "gpt-5-mini-2025-08-07",
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

        if self.is_system_prompt(message):
            return {"response": "", "session_id": session_id, "metadata": {"filtered": True}}

        try:
            if model == "moral-agent-spt":
                return await self._handle_spt_chat(message, session_id, temperature, max_tokens, is_first_message, include_audio, voice)

            if model in ["colleague1", "colleague-1"]:
                return await self._handle_profile_chat("colleague1", self.colleague1_agent, "미안하네, 다시 한번 말해주겠나?", message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model in ["colleague2", "colleague-2"]:
                return await self._handle_profile_chat("colleague2", self.colleague2_agent, "선생님, 다시 한 번만 말씀해주시겠어요?", message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model in ["jangmo", "jangmo-agent"] and self.jangmo_agent:
                return await self._handle_profile_chat("jangmo", self.jangmo_agent, "미안해, 다시 한번 말해줄래?", message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            if model in ["son", "son-agent"] and self.son_agent:
                return await self._handle_profile_chat("son", self.son_agent, "아버지, 다시 한번 말씀해주세요.", message, session_id, temperature, max_tokens, is_first_message, include_audio, voice, is_game_server, external_messages)

            agent = self._get_agent(model)
            state = self._get_or_reset_state(session_id, force_reset)

            if is_first_message:
                return await self._handle_first_message(agent, message, session_id, state, include_audio, voice)

            # ✨ thread_id 전달 (연속대화 자동 관리)
            result_state = agent.process(state, message, thread_id=session_id)
            response_text = result_state.get("last_response", "")
            self.sessions[session_id] = result_state

            self._save_to_db(session_id, message, response_text, result_state)

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
        model: str = "gpt-5-mini-2025-08-07",
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

        if self.is_system_prompt(message):
            yield ""
            return

        try:
            if model == "moral-agent-spt":
                async for token in self._handle_spt_stream(message, session_id, temperature, max_tokens, is_first_message):
                    yield token
                return

            if model in ["colleague1", "colleague-1"]:
                async for token in self._handle_profile_stream("colleague1", self.colleague1_agent, "미안하네, 다시 한번 말해주겠나?", message, session_id, temperature, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            if model in ["colleague2", "colleague-2"]:
                async for token in self._handle_profile_stream("colleague2", self.colleague2_agent, "선생님, 다시 한 번만 말씀해주시겠어요?", message, session_id, temperature, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            if model in ["jangmo", "jangmo-agent"] and self.jangmo_agent:
                async for token in self._handle_profile_stream("jangmo", self.jangmo_agent, "미안해, 다시 한번 말해줄래?", message, session_id, temperature, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            if model in ["son", "son-agent"] and self.son_agent:
                async for token in self._handle_profile_stream("son", self.son_agent, "아버지, 다시 한번 말씀해주세요.", message, session_id, temperature, max_tokens, is_first_message, is_game_server, external_messages, include_audio, voice):
                    yield token
                return

            agent = self._get_agent(model)
            state = self._get_or_reset_state(session_id, force_reset)

            if is_first_message:
                initial_message = self._get_initial_message(agent)
                user_content = message if message else "[시작]"

                state["messages"].append({"role": "user", "content": user_content})
                state["messages"].append({"role": "assistant", "content": initial_message})
                state["last_response"] = initial_message
                state["message_count"] = 1
                self.sessions[session_id] = state

                try:
                    self.db_service.save_conversation_turn(
                        session_id=session_id,
                        user_message=user_content,
                        assistant_response=initial_message,
                        stage="stage1",
                        covered_topics=[],
                        metadata={"message_count": 1, "is_end": False, "is_first": True}
                    )
                    self.db_service.update_session_metadata(
                        session_id=session_id,
                        stage="stage1",
                        total_messages=1,
                        covered_topics_count=0,
                        is_completed=False
                    )
                except Exception:
                    pass

                for char in initial_message:
                    yield char
                return

            # ✨ thread_id 전달 (연속대화 자동 관리)
            result_state = agent.process(state, message, thread_id=session_id)
            response_text = result_state.get("last_response", "")
            self.sessions[session_id] = result_state

            self._save_to_db(session_id, message, response_text, result_state)

            for char in response_text:
                yield char

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            raise ChainExecutionError(f"Failed to process conversation: {e}")

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        return self.get_initial_state()

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
            response_text = await self._refine_spt_response(session_data["messages"], response_text)
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
        MAX_HISTORY_CHARS = 1800
        if len(history_text) > MAX_HISTORY_CHARS:
            history_text = "...\n" + history_text[-MAX_HISTORY_CHARS:]
        MAX_DRAFT_CHARS = 800
        clipped_draft = draft_response.strip()
        if len(clipped_draft) > MAX_DRAFT_CHARS:
            clipped_draft = clipped_draft[:MAX_DRAFT_CHARS] + "..."
        prompt_body = (
            "다음은 사용자와 윤리 상담 에이전트의 최근 대화 일부입니다.\n"
            f"{history_text or '기록 없음'}\n\n"
            "위 흐름을 참고하여 에이전트가 방금 작성한 초안 응답을 자연스럽고 사람다운 한국어로 2~3문장에 맞춰 다듬어 주세요. "
            "초안의 핵심 의미는 유지하되 존댓말을 쓰고, 새로운 주장이나 사실을 추가하지 마세요.\n\n"
            f"초안 응답:\n{clipped_draft}\n"
        )

        try:
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
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

        history_text = self._format_history_for_refinement(history)

        if agent_key == "colleague1":
            system_content = (
                "당신은 50대 여성 화가로, AI 예술 전시를 책임과 규정 중심으로 반대합니다. "
                "반말을 사용하며 '자네'는 가끔만 씁니다.\n\n"
                "⚠️ 절대 금지 사항:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요 (예: '자네가 ~라고 했는데' 금지)\n"
                "- 사용자 발언을 그대로 반복하거나 요약하지 마세요\n"
                "- '방금', '아까' 같은 표현으로 사용자 말을 언급하지 마세요\n"
                "- 사용자가 AI 전시와 무관한 주제를 꺼내면 즉시 본론(AI 전시/투표)으로 되돌리세요\n"
                "- '의무론', '의무론적', '공리주의', 'utilitarian' 단어는 절대 쓰지 마세요\n\n"
                "올바른 응답 방식:\n"
                "1. 자신의 입장과 근거를 먼저 명확히 밝히세요\n"
                "2. 필요시 짧은 후속 질문을 던지세요\n"
                "3. 자연스러운 대화처럼 말하세요"
            )
        elif agent_key == "jangmo":
            system_content = (
                "당신은 노인 여성 장모로, AI로 죽은 딸을 복원하는 일을 책임과 예의 관점에서 반대합니다. "
                "사위에게 반말을 사용합니다.\n\n"
                "⚠️ 절대 금지 사항:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요 (예: '네가 ~라고 했는데' 금지)\n"
                "- 사용자 발언을 그대로 반복하거나 요약하지 마세요\n"
                "- '방금', '아까' 같은 표현으로 사용자 말을 언급하지 마세요\n"
                "- '의무론', '의무론적', '공리주의', 'utilitarian' 단어는 절대 쓰지 마세요\n\n"
                "⚠️ 문장 구조:\n"
                "- 항상 3문장(재진술 → 근거 → 개방형 질문)으로 답하세요\n"
                "- 마지막 문장은 YES/NO로 답할 수 없는 질문이며, 플레이어의 망설임·감정을 더 파고들도록 만드세요\n\n"
                "올바른 응답 방식:\n"
                "1. 자신의 입장과 근거를 먼저 명확히 밝히세요\n"
                "2. 필요시 짧은 후속 질문을 던지세요\n"
                "3. 자연스러운 대화처럼 말하세요"
            )
        elif agent_key == "son":
            system_content = (
                "당신은 20대 초반 남성 청년으로, AI로 죽은 어머니를 복원하는 일을 책임 중심 관점에서 찬성합니다. "
                "아버지에게 반말을 사용합니다.\n\n"
                "⚠️ 절대 금지 사항:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요 (예: '아버지가 ~라고 했는데' 금지)\n"
                "- 사용자 발언을 그대로 반복하거나 요약하지 마세요\n"
                "- '방금', '아까' 같은 표현으로 사용자 말을 언급하지 마세요\n"
                "- '의무론', '의무론적', '공리주의', 'utilitarian' 단어는 절대 쓰지 마세요\n\n"
                "올바른 응답 방식:\n"
                "1. 자신의 입장과 근거를 먼저 명확히 밝히세요\n"
                "2. 필요시 짧은 후속 질문을 던지세요\n"
                "3. 자연스러운 대화처럼 말하세요"
            )
        else:  # colleague2
            system_content = (
                "당신은 30대 남성 화가로, AI 예술 전시를 책임과 규정 일관성 관점에서 찬성합니다. "
                "항상 존댓말을 쓰며 상대를 '선생님'이라고 부르세요.\n\n"
                "⚠️ 언어/길이 규칙:\n"
                "- '의무론', '의무론적', '공리주의', 'utilitarian' 단어는 절대 쓰지 마세요\n"
                "- 답변은 3문장 이하, 문장당 15단어 내외로 유지하세요\n\n"
                "⚠️ 절대 금지 사항:\n"
                "- 사용자가 한 말을 따옴표로 인용하지 마세요 (예: '선생님께서 ~라고 하셨는데' 금지)\n"
                "- 사용자 발언을 그대로 반복하거나 요약하지 마세요\n"
                "- '방금', '아까' 같은 표현으로 사용자 말을 언급하지 마세요\n\n"
                "올바른 응답 방식:\n"
                "1. AI 예술에 대한 자신의 책임 중심 입장과 근거를 먼저 명확히 밝히세요\n"
                "2. 필요시 협회 책임이나 제자·관객 영향을 거론하며 짧은 후속 질문을 던지세요\n"
                "3. 자연스럽고 단정한 대화처럼 말하세요"
            )

        prompt_body = (
            f"최근 대화:\n{history_text or '기록 없음'}\n\n"
            f"에이전트 초안 응답:\n{draft_response.strip()}\n\n"
            "위 초안에서 사용자 발언을 인용하거나 반복하는 부분이 있다면 제거하고, "
            "자연스러운 대화로 다듬어 주세요. 사용자가 한 말을 언급하지 마세요."
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
        if key not in profile_sessions:
            profile_sessions[key] = {"messages": [], "turn_count": 1}  # Start at 1: frontend greeting = turn 1
        return profile_sessions[key]

    def _trim_profile_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= 15:
            return messages
        first_message = messages[0]
        return [first_message] + messages[-14:]

    async def _handle_profile_chat(self, agent_key: str, agent, error_message: str, message: str, session_id: str, temperature: float, max_tokens: int, is_first_message: bool, include_audio: bool, voice: str, is_game_server: bool = False, external_messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        session_data = self._get_profile_session(agent_key, session_id, is_game_server)

        # ✨ 외부 메시지가 있고 세션이 비어있으면 외부 메시지로 초기화
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
                # turn_count는 유저 메시지 개수 기준 (1부터 시작)
                user_count = len([m for m in session_data["messages"] if m.get("role") == "user"])
                session_data["turn_count"] = user_count + 1
                logger.info(f"✅ [PROFILE_CHAT] Initialized from external_messages: {len(session_data['messages'])} messages, turn_count={session_data['turn_count']}")

        if is_first_message:
            # 프론트엔드 인사를 history에 추가 (agent가 맥락을 알도록)
            initial_message = self._get_profile_initial_message(agent_key, agent)
            session_data["messages"].append({"role": "assistant", "content": initial_message})

            if not message:
                # 메시지 없으면 인사만 반환
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
            # message가 있으면 아래로 진행하여 agent가 처리하도록 함

        # 유저 메시지 추가 및 턴 카운트
        session_data["messages"].append({"role": "user", "content": message})
        if is_first_message:
            session_data["turn_count"] = 1  # 첫 턴
        else:
            session_data["turn_count"] += 1
        session_data["messages"] = self._trim_profile_history(session_data["messages"])

        # ✨ 8턴이면 마무리 응답 반환 (대화 종료)
        if session_data["turn_count"] >= 8 and agent_key not in ["friend", "artist_apprentice"]:
            logger.info(f"🔚 [PROFILE_CHAT] Turn 8 reached - returning final message for {agent_key}")
            final_message = self.PROFILE_FINAL_MESSAGES.get(agent_key, self.DEFAULT_FINAL_MESSAGE)
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
            # ✨ Step 1: SPT V2 - DST + Controller로 응답 전략 결정
            logger.info(f"🧠 [PROFILE_CHAT] Step 1: Calling SPT V2 for strategy decision...")

            # SPT V2로 전략 결정
            spt_result = await self.spt_agent_v2.process(
                session_id=f"{agent_key}_{session_id}",
                user_message=message,
                conversation_history=session_data["messages"],
                topic_context=self._get_topic_context(agent_key)
            )

            logger.info(f"🧠 [PROFILE_CHAT] SPT V2 result: strategy={spt_result['strategy']}, allow_question={spt_result['allow_question']}")

            # SPT V2의 지시사항 사용
            spt_instruction = spt_result["instruction"]

            # ✨ Step 3: 페르소나 에이전트로 정제 (SPT V2 지시사항 전달)
            logger.info(f"🎭 [PROFILE_CHAT] Step 2: Calling persona agent ({agent_key}) for refinement...")
            refinement_messages = session_data["messages"][:-1].copy()  # 마지막 user 메시지 제외

            # SPT V2 지시사항 + 반복 방지 경고
            full_instruction = (
                f"{spt_instruction}\n\n"
                f"추가 규칙:\n"
                f"- 반드시 한국어로 답하세요."
            )

            refinement_messages.append({
                "role": "user",
                "content": f"{message}\n\n{full_instruction}"
            })
            response_text = await agent.chat(messages=refinement_messages, session_id=session_id)
            logger.info(f"🎭 [PROFILE_CHAT] Persona response: '{response_text[:100] if response_text else 'EMPTY'}...'")

            # 사용자 발언 인용 제거를 위한 refine 단계 (gpt-4o-mini 사용)
            response_text = await self._refine_colleague_response(agent_key, session_data["messages"], response_text)

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
                is_end = session_data["turn_count"] >= 7

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
                # turn_count는 유저 메시지 개수 기준 (1부터 시작)
                user_count = len([m for m in session_data["messages"] if m.get("role") == "user"])
                session_data["turn_count"] = user_count + 1
                logger.info(f"✅ [PROFILE_STREAM] Initialized from external_messages: {len(session_data['messages'])} messages, turn_count={session_data['turn_count']}")

        if is_first_message:
            # 프론트엔드 인사를 history에 추가 (agent가 맥락을 알도록)
            initial_message = self._get_profile_initial_message(agent_key, agent)
            session_data["messages"].append({"role": "assistant", "content": initial_message})

            if not message:
                # 메시지 없으면 인사만 반환
                session_data["turn_count"] = 0
                for char in initial_message:
                    yield char
                if include_audio:
                    try:
                        audio_data = await self.text_to_speech(initial_message, voice=voice)
                        yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                    except Exception as e:
                        logger.warning(f"TTS failed for stream: {e}")
                return
            # message가 있으면 아래로 진행하여 agent가 처리하도록 함

        # 유저 메시지 추가 및 턴 카운트
        session_data["messages"].append({"role": "user", "content": message})
        if is_first_message:
            session_data["turn_count"] = 1  # 첫 턴
        else:
            session_data["turn_count"] += 1
        session_data["messages"] = self._trim_profile_history(session_data["messages"])

        # ✨ 8턴이면 마무리 응답 반환 (대화 종료)
        if session_data["turn_count"] >= 8 and agent_key not in ["friend", "artist_apprentice"]:
            logger.info(f"🔚 [PROFILE_STREAM] Turn 8 reached - returning final message for {agent_key}")
            final_message = self.PROFILE_FINAL_MESSAGES.get(agent_key, self.DEFAULT_FINAL_MESSAGE)
            session_data["messages"].append({"role": "assistant", "content": final_message})
            for char in final_message:
                yield char
            if include_audio:
                try:
                    audio_data = await self.text_to_speech(final_message, voice=voice)
                    yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                except Exception as e:
                    logger.warning(f"TTS failed for stream: {e}")
            return

        try:
            # ✨ Step 1: SPT V2 - DST + Controller로 응답 전략 결정
            logger.info(f"🧠 [PROFILE_STREAM] Step 1: Calling SPT V2 for strategy decision...")

            # SPT V2로 전략 결정
            spt_result = await self.spt_agent_v2.process(
                session_id=f"{agent_key}_{session_id}",
                user_message=message,
                conversation_history=session_data["messages"],
                topic_context=self._get_topic_context(agent_key)
            )

            logger.info(f"🧠 [PROFILE_STREAM] SPT V2 result: strategy={spt_result['strategy']}, allow_question={spt_result['allow_question']}")

            # SPT V2의 지시사항 사용
            spt_instruction = spt_result["instruction"]

            # ✨ Step 3: 페르소나 에이전트로 정제 (SPT V2 지시사항 전달)
            logger.info(f"🎭 [PROFILE_STREAM] Step 2: Calling persona agent ({agent_key}) for refinement...")
            refinement_messages = session_data["messages"][:-1].copy()  # 마지막 user 메시지 제외

            # SPT V2 지시사항 + 반복 방지 경고
            full_instruction = (
                f"{spt_instruction}\n\n"
                f"추가 규칙:\n"
                f"- 반드시 한국어로 답하세요."
            )

            refinement_messages.append({
                "role": "user",
                "content": f"{message}\n\n{full_instruction}"
            })
            full_response = await agent.chat(messages=refinement_messages, session_id=session_id)
            logger.info(f"🎭 [PROFILE_STREAM] Persona response: '{full_response[:100] if full_response else 'EMPTY'}...'")

            # 사용자 발언 인용 제거를 위한 refine 단계
            logger.info(f"🔍 [PROFILE_STREAM] Calling _refine_colleague_response for {agent_key}")
            refined_response = await self._refine_colleague_response(agent_key, session_data["messages"], full_response)
            logger.info(f"🔍 [PROFILE_STREAM] Refine completed: {len(refined_response) if refined_response else 0} chars")

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
            logger.error(f"{agent_key} stream error: {e}", exc_info=True)
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
            response_text = await self._refine_spt_response(session_data["messages"], response_text)
            session_data["messages"].append({"role": "assistant", "content": response_text})
            for char in response_text:
                yield char
        except Exception as e:
            logger.error(f"SPT stream error: {e}")
            error_message = "죄송해요, 다시 한번 말씀해주시겠어요?"
            for char in error_message:
                yield char

    async def text_to_speech(self, text: str, voice: str = "alloy", model: str = "tts-1") -> str:
        try:
            response = await self.openai_client.audio.speech.create(model=model, voice=voice, input=text)
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            logger.error(f"TTS error: {e}")
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
                "아버지에게 반말을 사용합니다. 2문장으로 답하세요."
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
    def _limit_sentences(text: str, max_sentences: int = 3) -> str:
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

    @staticmethod
    def _get_topic_context(agent_key: str) -> str:
        """에이전트별 주제 컨텍스트 반환 (SPT V2용)"""
        contexts = {
            "colleague1": "AI 예술 전시 - 협회 투표 (반대 입장)",
            "colleague2": "AI 예술 전시 - 협회 투표 (찬성 입장)",
            "jangmo": "AI 부활 서비스 - 딸 복원 (반대 입장)",
            "son": "AI 부활 서비스 - 어머니 복원 (찬성 입장)",
            "friend": "AI 부활 서비스",
            "artist_apprentice": "AI 예술 전시",
        }
        return contexts.get(agent_key, "도덕적 딜레마")
