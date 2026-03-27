"""Stage 2: AI stance question + transition to stage3."""
from __future__ import annotations

from typing import Any, Dict


def handle_stage2(agent, state: Dict[str, Any], reflection: dict) -> None:
    """User responded to AI topic → empathize + transition to stage3."""
    messages = state.get("messages", [])
    user_msg = _last_user_msg(messages)

    if state.get("stage2_question_asked"):
        # Generate empathy for user's AI stance
        ack = agent.call_response_llm(
            intent="opinion_given",
            context_analysis=reflection.get("context_analysis", ""),
            spt_section="",
            conversation_history=_recent_context(messages, 6),
            user_message=user_msg,
            current_question="",
            next_question="",
            forbidden_questions="",
        )
        first_q = agent.config.get("stage3_first_question", "")
        response = f"{ack} {first_q}" if first_q not in ack else ack

        state["stage2_completed"] = True
        state["stage2_complete"] = True
        state["current_asking_topic"] = "이익"
        _commit(state, response, stage="stage3", previous="stage2")
    else:
        # Rare: question not yet asked
        stage2_q = agent.config.get("stage2_question", "")
        state["stage2_question_asked"] = True
        _commit(state, stage2_q, stage="stage2", previous="stage1")


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
