"""Session and state management for game conversations."""
import copy
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.profile_sessions: Dict[tuple, Dict[str, Any]] = {}
        self.sessions_game: Dict[str, Dict[str, Any]] = {}
        self.profile_sessions_game: Dict[tuple, Dict[str, Any]] = {}

    @staticmethod
    def get_initial_state() -> Dict[str, Any]:
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
            "need_reason_count": 0,
            "unsure_count": 0,
        }

    def get_sessions(self, is_game_server: bool = False) -> Dict[str, Dict[str, Any]]:
        return self.sessions_game if is_game_server else self.sessions

    def get_profile_sessions(self, is_game_server: bool = False) -> Dict[tuple, Dict[str, Any]]:
        return self.profile_sessions_game if is_game_server else self.profile_sessions

    def get_or_create_state(self, session_id: str, is_game_server: bool = False) -> Dict[str, Any]:
        sessions = self.get_sessions(is_game_server)
        if session_id not in sessions:
            sessions[session_id] = self.get_initial_state()
        return sessions[session_id]

    def has_game_session(self, session_id: str) -> bool:
        return session_id in self.sessions_game

    def get_or_reset_state(self, session_id: str, force_reset: bool, is_game_server: bool = False) -> Dict[str, Any]:
        sessions = self.get_sessions(is_game_server)
        if force_reset:
            state = self.get_initial_state()
            sessions[session_id] = state
            return state

        state = self.get_or_create_state(session_id, is_game_server)
        if state.get("should_end", False) or state.get("stage") == "end":
            state = self.get_initial_state()
            sessions[session_id] = state
        return state

    def get_profile_session(self, agent_key: str, session_id: str, is_game_server: bool = False) -> Dict[str, Any]:
        profile_sessions = self.get_profile_sessions(is_game_server)
        key = (agent_key, session_id)

        if key not in profile_sessions:
            other_sessions = self.get_profile_sessions(not is_game_server)
            if key in other_sessions:
                profile_sessions[key] = other_sessions[key]
            else:
                profile_sessions[key] = {"messages": [], "turn_count": 1}

        return profile_sessions[key]

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        return self.get_initial_state()

    def clear_game_session(self, context_id: str) -> bool:
        self.sessions_game[context_id] = self.get_initial_state()
        keys_to_delete = [key for key in self.profile_sessions_game if key[1] == context_id]
        for key in keys_to_delete:
            del self.profile_sessions_game[key]
        return True

    def migrate_game_session(self, old_context_id: str, new_context_id: str) -> bool:
        if old_context_id in self.sessions_game:
            old_state = self.sessions_game[old_context_id]
            self.sessions_game[new_context_id] = copy.deepcopy(old_state)
            del self.sessions_game[old_context_id]

        keys_to_delete = [key for key in self.profile_sessions_game if key[1] == old_context_id]
        for old_key in keys_to_delete:
            del self.profile_sessions_game[old_key]
        return True

    def clear_all_game_sessions(self) -> bool:
        self.sessions_game.clear()
        self.profile_sessions_game.clear()
        return True

    @staticmethod
    def trim_profile_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= 15:
            return messages
        first_message = messages[0]
        return [first_message] + messages[-14:]

    def load_history_from_db(
        self,
        session_id: str,
        messages: Optional[List[Dict[str, str]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        turn_count: Optional[int] = None,
        agent_key: Optional[str] = None,
        is_game_server: bool = True,
    ):
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
            profile_sessions = self.get_profile_sessions(is_game_server)
            key = (agent_key, session_id)
            profile_sessions[key] = {"messages": formatted_messages, "turn_count": msg_count}
        else:
            sessions = self.get_sessions(is_game_server)
            state = self.get_initial_state()
            state["messages"] = formatted_messages
            state["message_count"] = msg_count

            stage, _ = self.infer_stage_from_messages(formatted_messages)
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
        sessions = self.get_sessions(is_game_server)
        state = self.get_initial_state()
        state["messages"] = messages
        state["message_count"] = len(messages)
        sessions[session_id] = state

    @staticmethod
    def infer_stage_from_messages(messages: List[Dict[str, str]]) -> tuple:
        if not messages:
            return "stage1", "no_messages"

        assistant_messages = " ".join(
            msg["content"] for msg in messages if msg.get("role") == "assistant"
        )

        stage3_keywords = ["어떤 이익", "어떤 피해", "대안", "이익이 있을까", "피해가 있을까"]
        for keyword in stage3_keywords:
            if keyword in assistant_messages:
                return "stage3", f"keyword:{keyword}"

        stage2_keywords = ["그림 그리는 AI", "국립현대예술관", "AI의 그림", "전시가 된대요"]
        for keyword in stage2_keywords:
            if keyword in assistant_messages:
                return "stage2", f"keyword:{keyword}"

        return "stage1", "default"
