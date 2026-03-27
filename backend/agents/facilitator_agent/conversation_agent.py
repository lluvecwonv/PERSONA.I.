"""
FacilitatorAgent — unified 3-stage facilitator for ethics dialogue.

Replaces the duplicate artist_apprentice_agent and friend_agent with a single
class parameterised by persona_type ("artist_apprentice" | "friend").

Architecture (3-Phase, mirrors jangmo/son/colleague pattern):
  Phase 1   – Self-Reflection  (cheap model): stage/intent/SPT analysis
  Phase 1.5 – SPT Planner      (cheap model): conditional perspective analysis
  Phase 2   – Response          (main model):  generate persona-appropriate reply
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .utils import LANGUAGE_GUARD, load_text, load_json, fmt, clean_response
from .stages import handle_stage1, handle_stage2, handle_stage3

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None

logger = logging.getLogger(__name__)

# ── persona context for reflection prompt ────────────────────────────────────

_PERSONA_CONTEXTS = {
    "artist_apprentice": (
        "Role: 화가 지망생 (20대 초반 남성)\n"
        "Relationship: 선생님(플레이어)의 제자, 존댓말 사용\n"
        "Topic: AI 그림의 국립현대예술관 전시 논란"
    ),
    "friend": (
        "Role: 50대 여성 친구\n"
        "Relationship: 오래된 친구, 반말 사용\n"
        "Topic: AI로 죽은 사람을 재현하는 기술"
    ),
}

_SITUATION_CONTEXTS = {
    "artist_apprentice": (
        "근미래, AI 그림이 보편화된 사회. 대기업이 만든 AI 화가의 그림이 "
        "국립현대예술관에 전시 예정. 플레이어는 화가협회 회원으로 투표권 보유."
    ),
    "friend": (
        "플레이어는 1년 전 아내를 잃은 50대 남성. "
        "AI로 고인을 재현하는 서비스가 등장한 상황."
    ),
}

_SPEECH_STYLE_CHECKS = {
    "artist_apprentice": (
        "CORRECT: Use formal speech (~yo, ~seyo, ~sumnida)\n"
        "WRONG: No banmal (~ya, ~eo, ~ji)\n"
        "Address player as: '선생님'"
    ),
    "friend": (
        "CORRECT: Use banmal (~ya, ~eo, ~ji)\n"
        "WRONG: No formal speech (~yo, ~seyo, ~sumnida)\n"
        "Address player by name or omit"
    ),
}


class FacilitatorAgent:
    """Single class for both artist-apprentice and friend facilitators."""

    def __init__(self, api_key: str, model: str = "gpt-4o", persona_type: str = "artist_apprentice"):
        self.persona_type = persona_type
        self.api_key = api_key
        self.model = model

        # ── load persona-specific assets ─────────────────────────────────
        persona_dir = Path(__file__).parent / "prompts" / persona_type
        self.persona_prompt = load_text(persona_dir / "persona.txt")
        self.config = load_json(persona_dir / "config.json")
        self.ethics_topics = load_json(persona_dir / "ethics_topics.json")

        # ── load shared prompts ──────────────────────────────────────────
        shared_dir = Path(__file__).parent / "prompts"
        self.reflection_template = load_text(shared_dir / "reflection_prompt.txt")
        self.response_template = load_text(shared_dir / "response_prompt.txt")
        self.spt_planner_template = load_text(shared_dir / "spt_planner_prompt.txt")

        # ── persona context strings ──────────────────────────────────────
        self.persona_context = _PERSONA_CONTEXTS.get(persona_type, "")
        self.situation_context = _SITUATION_CONTEXTS.get(persona_type, "")
        self.speech_style_check = _SPEECH_STYLE_CHECKS.get(persona_type, "")

        # ── LLMs ─────────────────────────────────────────────────────────
        google_key = os.getenv("GOOGLE_API_KEY", "")
        if HAS_GEMINI and google_key:
            self.analyzer = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0, google_api_key=google_key
            )
        else:
            self.analyzer = ChatOpenAI(
                model=model, temperature=0, max_tokens=500, api_key=api_key
            )

        llm_kwargs: dict = {"model": model, "api_key": api_key}
        if "gpt-5" in model:
            llm_kwargs["temperature"] = 1.0
            llm_kwargs["max_completion_tokens"] = 1000
        else:
            llm_kwargs["temperature"] = 0.7
            llm_kwargs["max_tokens"] = 500
        self.llm = ChatOpenAI(**llm_kwargs)

        logger.info(
            f"FacilitatorAgent({persona_type}) ready  model={model}  "
            f"questions={len(self.ethics_topics.get('questions', []))}"
        )

    # ── public API ───────────────────────────────────────────────────────

    def get_initial_message(self) -> str:
        return self.config.get("initial_greeting", "")

    def process(self, state: Dict[str, Any], user_input: str, thread_id: str = "default") -> Dict[str, Any]:
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append({"role": "user", "content": user_input})

        stage = state.get("stage", "stage1")
        messages = state.get("messages", [])

        # ── Phase 1: Self-Reflection ─────────────────────────────────
        reflection = self._run_reflection(stage, messages, user_input, state)

        # ── Phase 1.5: SPT Planner (conditional) ─────────────────────
        spt_result = None
        if reflection.get("spt_required") and stage == "stage3":
            spt_result = self._run_spt_planner(messages, user_input, state)

        # ── Phase 2: Route to stage handler ──────────────────────────
        if stage == "stage1":
            handle_stage1(self, state, reflection)
        elif stage == "stage2":
            handle_stage2(self, state, reflection)
        elif stage == "stage3":
            handle_stage3(self, state, reflection, spt_result)

        return state

    # ── Phase 1: Self-Reflection LLM ─────────────────────────────────

    def _run_reflection(self, stage: str, messages: list, user_msg: str, state: dict) -> dict:
        q_idx = state.get("current_question_index", 0)
        questions = self.ethics_topics.get("questions", [])
        current_q = ""
        if 0 <= q_idx < len(questions):
            variations = questions[q_idx].get("variations", [])
            current_q = variations[0] if variations else ""

        recent = messages[-10:] if len(messages) > 10 else messages
        history = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

        prompt = fmt(
            self.reflection_template,
            persona_context=self.persona_context,
            situation_context=self.situation_context,
            current_stage=stage,
            current_question=current_q,
            history=history,
            last_user_msg=user_msg,
        )

        try:
            result = self.analyzer.invoke(prompt)
            return self._parse_json(result.content)
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return {"stage": stage, "action": "stay", "intent": "answer", "spt_required": False}

    # ── Phase 1.5: SPT Planner LLM ──────────────────────────────────

    def _run_spt_planner(self, messages: list, user_msg: str, state: dict) -> dict | None:
        recent = messages[-8:] if len(messages) > 8 else messages
        history = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        topic = state.get("current_asking_topic", "")

        prompt = fmt(
            self.spt_planner_template,
            last_user_msg=user_msg,
            history=history,
            current_topic=topic,
        )

        try:
            result = self.analyzer.invoke(prompt)
            return self._parse_json(result.content)
        except Exception as e:
            logger.error(f"SPT planner error: {e}")
            return None

    # ── Phase 2: Response LLM ────────────────────────────────────────

    def call_response_llm(self, *, intent: str, context_analysis: str,
                          spt_section: str, conversation_history: str,
                          user_message: str, current_question: str,
                          next_question: str, forbidden_questions: str) -> str:
        prompt = fmt(
            self.response_template,
            persona_prompt=self.persona_prompt,
            intent=intent,
            context_analysis=context_analysis,
            spt_section=spt_section,
            conversation_history=conversation_history,
            user_message=user_message,
            current_question=current_question,
            next_question=next_question,
            forbidden_questions=forbidden_questions,
            speech_style_check=self.speech_style_check,
        )

        try:
            result = self.llm.invoke([
                SystemMessage(content=f"{self.persona_prompt}\n\n{LANGUAGE_GUARD}"),
                HumanMessage(content=prompt),
            ])
            resp = clean_response(result.content.strip().strip('"'))
            if resp and resp not in ("...", "..", "."):
                return resp
        except Exception as e:
            logger.error(f"Response LLM error: {e}")
        return self.config.get("error_message", "다시 한번 말씀해주시겠어요?")

    # ── JSON parsing helper ──────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {}
