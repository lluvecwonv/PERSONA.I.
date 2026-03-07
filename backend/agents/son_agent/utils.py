"""Son Agent utility functions."""
import re


def clean_gpt_response(text: str) -> str:
    """Remove unwanted tokens from LLM response."""
    if not text:
        return text

    text = re.sub(r'\bcurrent\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'현재', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
