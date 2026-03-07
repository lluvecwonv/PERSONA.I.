"""Jangmo Agent utility functions."""
import re
from typing import Match


def _replace_current_token(match: Match[str]) -> str:
    """Replace 'current'/'currently' while respecting the original casing."""
    token = match.group(0)
    lower = token.lower()
    replacement = "present" if lower == "current" else "now"

    if token.isupper():
        return replacement.upper()
    if token[0].isupper():
        return replacement.capitalize()
    return replacement


def clean_gpt_response(text: str) -> str:
    """Remove unwanted tokens from LLM response."""
    if not text:
        return "미안해요, 다시 한번 말해줄래요?"

    text = re.sub(r'current(?:ly)?', _replace_current_token, text, flags=re.IGNORECASE)
    text = text.replace("현재", "지금")
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if not text or len(text) < 3:
        return "미안해요, 다시 한번 말해줄래요?"

    return text
