"""Shared utilities for FacilitatorAgent."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

LANGUAGE_GUARD = (
    "언어 규칙:\n"
    "- 모든 최종 응답은 100% 자연스러운 한국어 문장으로만 작성하세요.\n"
    "- 영어 단어, 로마자 표기, 번역 괄호, 영어 예시 문장을 출력에 포함하지 마세요.\n"
    "- 번역이나 설명이 필요해도 한국어 표현만 사용하세요."
)

# ── file loaders ─────────────────────────────────────────────────────────────

def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return ""


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}


# ── safe formatting ──────────────────────────────────────────────────────────

def fmt(template: str, **kw) -> str:
    """str.replace-based formatting (keeps unmatched braces intact)."""
    result = template
    for k, v in kw.items():
        result = result.replace("{" + k + "}", str(v))
    return result


# ── response cleanup ─────────────────────────────────────────────────────────

def clean_response(text: str) -> str:
    for label in ("선생님 말씀:", "다음 질문:", "선생님의 답변:"):
        text = text.replace(label, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()
