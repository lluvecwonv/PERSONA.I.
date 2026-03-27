"""Stage 3: Ethics question loop (5 Floridi values)."""
from __future__ import annotations

import json
from typing import Any, Dict


def handle_stage3(agent, state: Dict[str, Any], reflection: dict, spt_result: dict | None) -> None:
    """Route stage3 based on reflection intent. Generates response via agent LLM."""
    messages = state.get("messages", [])
    user_msg = _last_user_msg(messages)
    intent = reflection.get("intent", "answer")
    q_idx = state.get("current_question_index", 0)

    questions = agent.ethics_topics.get("questions", [])
    current_q = ""
    if 0 <= q_idx < len(questions):
        variations = questions[q_idx].get("variations", [])
        current_q = variations[0] if variations else ""

    context = _recent_context(messages, 8)
    spt_section = _format_spt(spt_result) if spt_result else ""
    forbidden = json.dumps(reflection.get("forbidden_questions", []), ensure_ascii=False)

    need_reason_count = state.get("need_reason_count", 0)
    unsure_count = state.get("unsure_count", 0)
    next_q_idx = q_idx
    is_done = False

    if intent == "ask_opinion":
        response = _generate(agent, "ask_opinion", reflection, spt_section, context,
                             user_msg, current_q, "", forbidden)

    elif intent == "dont_know":
        if unsure_count >= 1:
            state["unsure_count"] = 0
            is_done, response, next_q_idx = _advance(agent, q_idx, reflection, spt_section,
                                                      context, user_msg, forbidden)
        else:
            response = _generate(agent, "dont_know", reflection, spt_section, context,
                                 user_msg, current_q, "", forbidden)
            state["unsure_count"] = unsure_count + 1

    elif intent == "ask_concept":
        response = _generate(agent, "ask_concept", reflection, spt_section, context,
                             user_msg, current_q, "", forbidden)

    elif intent == "need_reason":
        if need_reason_count >= 1:
            state["need_reason_count"] = 0
            is_done, response, next_q_idx = _advance(agent, q_idx, reflection, spt_section,
                                                      context, user_msg, forbidden)
        else:
            response = _generate(agent, "need_reason", reflection, spt_section, context,
                                 user_msg, current_q, "", forbidden)
            state["need_reason_count"] = need_reason_count + 1

    elif intent == "clarification":
        variation_idx = state.get("variation_index", 0) + 1
        if variation_idx >= 2:
            state["variation_index"] = 0
            is_done, response, next_q_idx = _advance(agent, q_idx, reflection, spt_section,
                                                      context, user_msg, forbidden)
        else:
            response = _generate(agent, "clarification", reflection, spt_section, context,
                                 user_msg, current_q, "", forbidden)
            state["variation_index"] = variation_idx

    elif intent == "ask_explanation":
        response = _generate(agent, "ask_explanation", reflection, spt_section, context,
                             user_msg, current_q, "", forbidden)

    else:  # "answer" → advance to next question
        state["variation_index"] = 0
        state["need_reason_count"] = 0
        state["unsure_count"] = 0
        is_done, response, next_q_idx = _advance(agent, q_idx, reflection, spt_section,
                                                  context, user_msg, forbidden)

    state["current_question_index"] = next_q_idx
    state["should_end"] = is_done
    _commit(state, response, stage="stage3", previous="stage3")


# ── helpers ──────────────────────────────────────────────────────────────────

def _generate(agent, intent, reflection, spt_section, context,
              user_msg, current_q, next_q, forbidden):
    return agent.call_response_llm(
        intent=intent,
        context_analysis=reflection.get("context_analysis", ""),
        spt_section=spt_section,
        conversation_history=context,
        user_message=user_msg,
        current_question=current_q,
        next_question=next_q,
        forbidden_questions=forbidden,
    )


def _advance(agent, current_idx, reflection, spt_section, context, user_msg, forbidden):
    """Move to next ethics question or close. Returns (is_done, response, next_idx)."""
    questions = agent.ethics_topics.get("questions", [])
    is_last = current_idx >= len(questions) - 1

    if is_last:
        response = agent.call_response_llm(
            intent="close",
            context_analysis=reflection.get("context_analysis", ""),
            spt_section=spt_section,
            conversation_history=context,
            user_message=user_msg,
            current_question="",
            next_question="",
            forbidden_questions=forbidden,
        )
        return True, response, -1

    next_idx = current_idx + 1
    next_q = questions[next_idx]["variations"][0]
    response = agent.call_response_llm(
        intent="answer",
        context_analysis=reflection.get("context_analysis", ""),
        spt_section=spt_section,
        conversation_history=context,
        user_message=user_msg,
        current_question="",
        next_question=next_q,
        forbidden_questions=forbidden,
    )
    return False, response, next_idx


def _format_spt(spt: dict) -> str:
    if not spt:
        return ""
    lines = ["=== SPT 분석 ==="]
    if spt.get("stakeholders"):
        lines.append(f"이해관계자: {', '.join(spt['stakeholders'])}")
    if spt.get("empathy_analysis"):
        lines.append(f"공감 분석: {spt['empathy_analysis']}")
    if spt.get("blind_spot"):
        lines.append(f"간과된 관점: {spt['blind_spot']}")
    if spt.get("strategic_question"):
        lines.append(f"전략적 질문: {spt['strategic_question']}")
    lines.append("=== END SPT ===")
    return "\n".join(lines)


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
