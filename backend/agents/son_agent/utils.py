"""
Son Agent 유틸리티 함수
"""
import re


def clean_gpt_response(text: str) -> str:
    """
    GPT 응답에서 불필요한 패턴 제거
    - "current" 단어 제거
    - "현재" 단어 제거
    - 따옴표로 감싼 사용자 발언 인용 제거
    """
    if not text:
        return text

    # "current" 제거 (대소문자 무관)
    text = re.sub(r'\bcurrent\b', '', text, flags=re.IGNORECASE)

    # "현재" 제거
    text = re.sub(r'현재', '', text)

    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()

    return text
