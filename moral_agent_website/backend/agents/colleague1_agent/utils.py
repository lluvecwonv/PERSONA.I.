"""
SPT Agent 유틸리티 함수
"""
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
    """
    GPT 모델 응답에서 불필요한 단어 제거
    """
    if not text:
        return "미안하네, 다시 한번 말해주겠나?"

    # "current"/"currently" 제거 대신 자연스러운 치환
    text = re.sub(r'current(?:ly)?', _replace_current_token, text, flags=re.IGNORECASE)
    text = text.replace("현재", "지금")

    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)

    # 앞뒤 공백 제거
    text = text.strip()

    # 빈 응답이면 기본 메시지
    if not text or len(text) < 3:
        return "미안하네, 다시 한번 말해주겠나?"

    return text
