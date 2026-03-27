"""Stage 1: Character / situation setup."""
from __future__ import annotations

import random
from typing import Any, Dict


def handle_stage1(agent, state: Dict[str, Any], reflection: dict) -> None:
    """Route stage1 based on reflection intent."""
    intent = reflection.get("intent", "")
    messages = state.get("messages", [])
    user_msg = _last_user_msg(messages)

    if intent == "casual_greeting":
        response = random.choice(
            agent.config.get("casual_greeting_responses", ["그렇군요."])
        )
        state["stage1_attempts"] = state.get("stage1_attempts", 0) + 1
        _commit(state, response, stage="stage1")
        return

    # situation_described or attempts exhausted → transition to stage2
    attempts = state.get("stage1_attempts", 0)
    if intent == "situation_described" or attempts >= 1:
        ack = agent.call_response_llm(
            intent="situation_described",
            context_analysis=reflection.get("context_analysis", ""),
            spt_section="",
            conversation_history=_recent_context(messages, 4),
            user_message=user_msg,
            current_question="",
            next_question="",
            forbidden_questions="",
        )
        stage2_q = agent.config.get("stage2_question", "")
        response = f"{ack} {stage2_q}" if stage2_q not in ack else ack

        state["artist_character_set"] = True
        state["stage2_question_asked"] = True
        state["stage2_complete"] = False
        state["stage2_completed"] = False
        _commit(state, response, stage="stage2", previous="stage1")
    else:
        # rephrase: ask about their situation again
        response = agent.call_response_llm(
            intent="casual_greeting",
            context_analysis=reflection.get("context_analysis", ""),
            spt_section="",
            conversation_history=_recent_context(messages, 4),
            user_message=user_msg,
            current_question="",
            next_question="",
            forbidden_questions="",
        )
        if not response or response in ("...", "..", "."):
            response = agent.config.get("rephrase_fallback", "요즘 어떻게 지내세요?")
        state["stage1_attempts"] = attempts + 1
        _commit(state, response, stage="stage1")


def _last_user_msg(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"]
    return ""


def _recent_context(messages: list, n: int) -> str:
    recent = messages[-n:] if len(messages) > n else messages
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def _commit(state: dict, response: str, stage: str, previous: str | None = None) -> None:
    state["last_response"] = response
    state["stage"] = stage
    if previous is not None:
        state["previous_stage"] = previous
    state["messages"].append({"role": "assistant", "content": response})
    state["message_count"] = state.get("message_count", 0) + 1
