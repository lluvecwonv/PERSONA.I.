"""
SPT Agent 유틸리티 함수
"""
import re


def clean_gpt_response(text: str) -> str:
    """
    GPT 모델 응답에서 불필요한 단어 제거

    Args:
        text: 원본 응답 텍스트

    Returns:
        정제된 응답 텍스트
    """
    # 빈 응답이면 기본 메시지 반환
    if not text or not text.strip():
        return "죄송해요, 다시 한번 말씀해주시겠어요?"

    # "current" 및 "현재" 단어 제거 (반복되는 경우 포함)
    text = re.sub(r'\bcurrent\b[\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'현재[\s]*', '', text)

    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)

    # 앞뒤 공백 제거
    text = text.strip()

    # 빈 응답이면 기본 메시지
    if not text or len(text) < 3:
        return "죄송해요, 다시 한번 말씀해주시겠어요?"

    return text
