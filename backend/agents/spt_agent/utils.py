"""SPT Agent utility functions."""
import re


def clean_gpt_response(text: str) -> str:
    """Remove unwanted tokens from LLM response."""
    if not text or not text.strip():
        return "죄송해요, 다시 한번 말씀해주시겠어요?"

    text = re.sub(r'\bcurrent\b[\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'현재[\s]*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if not text or len(text) < 3:
        return "죄송해요, 다시 한번 말씀해주시겠어요?"

    return text
