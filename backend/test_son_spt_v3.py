#!/usr/bin/env python3
"""
Son Agent SPT 전략 테스트 V3 - ResponseType 분류 확인
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.son_agent import SonAgent

API_KEY = os.getenv("OPENAI_API_KEY")

# 다양한 ResponseType을 유발하는 테스트 발화
TEST_MESSAGES = [
    "글쎄다, 잘 모르겠어",  # UNCERTAIN
    "그래, 네 말이 맞는 것 같아",  # AGREE
    "아니, 난 그렇게 생각 안 해",  # DISAGREE
    "음, 일리가 있네",  # AGREE
    "응",  # SHORT
    "맞아",  # SHORT or AGREE
    "근데 비용은 얼마나 들어?",  # QUESTION
    "그건 그렇고...",  # OTHER
    "여전히 고민이야",  # UNCERTAIN
    "알겠어, 고마워",  # AGREE
]


async def test_classification():
    """ResponseType 분류만 테스트"""
    print("\n" + "="*70)
    print("🧑 SON AGENT - ResponseType 분류 테스트")
    print("="*70)

    agent = SonAgent(api_key=API_KEY)

    for i, msg in enumerate(TEST_MESSAGES, 1):
        response_type = agent.classify_response_type(msg)
        print(f"[{i}] '{msg}' → {response_type}")


if __name__ == "__main__":
    asyncio.run(test_classification())
