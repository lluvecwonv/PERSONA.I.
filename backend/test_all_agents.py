#!/usr/bin/env python3
"""
모든 에이전트 말투 테스트 스크립트
각 에이전트와 10턴씩 대화하여 말투가 올바른지 확인
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.son_agent import SonAgent
from agents.jangmo_agent import JangmoAgent
from agents.colleague1_agent import Colleague1Agent
from agents.colleague2_agent import Colleague2Agent

# API 키
API_KEY = os.getenv("OPENAI_API_KEY")

# 테스트용 사용자 발화 (각 에이전트에 맞게)
SON_USER_MESSAGES = [
    "글쎄다, 잘 모르겠어",
    "엄마를 다시 볼 수 있다는 건 좋지만...",
    "하지만 그게 진짜 엄마일까?",
    "기술이 완벽하지 않을 수도 있잖아",
    "다른 가족들은 어떻게 생각할까?",
    "비용도 많이 들텐데",
    "그래도 망설여지네",
    "네 말도 일리가 있어",
    "좀 더 생각해볼게",
    "알겠어, 고마워",
]

JANGMO_USER_MESSAGES = [
    "장모님, 저는 아내를 다시 보고 싶어요",
    "기술이 발전했으니까 가능하지 않을까요?",
    "아내가 원했을 수도 있지 않나요?",
    "그리움이 너무 큽니다",
    "아이들도 엄마를 그리워해요",
    "한 번만 생각해주세요",
    "저도 고민이 많습니다",
    "네, 이해해요",
    "감사합니다 장모님",
    "잘 생각해보겠습니다",
]

COLLEAGUE1_USER_MESSAGES = [
    "AI 그림도 예술 아닌가?",
    "요즘 AI 그림 퀄리티가 좋던데",
    "관객들은 AI 그림도 좋아하더라",
    "시대가 변하는 거 아닐까?",
    "전시회에 포함시켜도 될 것 같은데",
    "다른 화가들 생각은 어때?",
    "그래도 고민이 되네",
    "네 말도 일리가 있어",
    "좀 더 생각해볼게",
    "고마워, 좋은 의견이야",
]

COLLEAGUE2_USER_MESSAGES = [
    "AI 그림은 예술이 아니지 않나?",
    "인간의 창의성이 더 중요하지",
    "기계가 만든 건 진정한 예술이 아니야",
    "전통을 지켜야 하지 않을까?",
    "관객들이 속는 느낌이 들어",
    "선생님 생각은 어떠세요?",
    "그래도 확신이 안 서네요",
    "네, 이해가 됩니다",
    "감사합니다 선생님",
    "잘 생각해보겠습니다",
]


def check_banmal(response: str) -> bool:
    """반말 패턴 감지 (존댓말이 아닌 경우)"""
    # 존댓말 어미
    jondaenmal = ["요.", "요?", "요!", "습니다", "세요", "예요", "어요", "까요"]
    # 반말 어미
    banmal = ["해.", "야.", "지.", "네.", "어.", "아.", "거야", "해?", "지?", "야?"]

    has_jondaenmal = any(p in response for p in jondaenmal)
    has_banmal = any(p in response for p in banmal)

    # 반말이 있고 존댓말이 없으면 반말
    return has_banmal and not has_jondaenmal


def check_jondaenmal(response: str) -> bool:
    """존댓말 패턴 감지"""
    jondaenmal = ["요.", "요?", "요!", "습니다", "세요", "예요", "어요", "까요"]
    return any(p in response for p in jondaenmal)


async def test_son_agent():
    """Son Agent 테스트 - 존댓말 사용 확인"""
    print("\n" + "="*70)
    print("🧑 SON AGENT 테스트 (아들 → 아버지: 존댓말 필수)")
    print("="*70)

    agent = SonAgent(api_key=API_KEY)
    session_id = "test-son"
    messages = []

    for i, user_msg in enumerate(SON_USER_MESSAGES, 1):
        print(f"\n[턴 {i}] 아버지: {user_msg}")
        messages.append({"role": "user", "content": user_msg})

        try:
            response = await agent.chat(messages, session_id=session_id)
            print(f"[턴 {i}] 아들: {response}")
            messages.append({"role": "assistant", "content": response})

            # 반말 체크
            if check_banmal(response):
                print(f"   ⚠️ 반말 감지! (아들은 존댓말을 써야 함)")
            elif check_jondaenmal(response):
                print(f"   ✅ 존댓말 확인")
        except Exception as e:
            print(f"   ❌ 에러: {e}")

    print("\n" + "-"*70)


async def test_jangmo_agent():
    """Jangmo Agent 테스트 - 반말 사용 확인"""
    print("\n" + "="*70)
    print("👵 JANGMO AGENT 테스트 (장모 → 사위: 반말)")
    print("="*70)

    agent = JangmoAgent(api_key=API_KEY)
    session_id = "test-jangmo"
    messages = []

    for i, user_msg in enumerate(JANGMO_USER_MESSAGES, 1):
        print(f"\n[턴 {i}] 사위: {user_msg}")
        messages.append({"role": "user", "content": user_msg})

        try:
            response = await agent.chat(messages, session_id=session_id)
            print(f"[턴 {i}] 장모: {response}")
            messages.append({"role": "assistant", "content": response})

            # 존댓말 체크 (장모는 반말을 써야 함)
            if check_jondaenmal(response) and not check_banmal(response):
                print(f"   ⚠️ 존댓말 감지! (장모는 반말을 써야 함)")
            else:
                print(f"   ✅ 반말 확인")
        except Exception as e:
            print(f"   ❌ 에러: {e}")

    print("\n" + "-"*70)


async def test_colleague1_agent():
    """Colleague1 Agent 테스트 - 반말 + 자네 사용 확인"""
    print("\n" + "="*70)
    print("👩‍🎨 COLLEAGUE1 AGENT 테스트 (50대 여성 화가: 반말 + ~하나?)")
    print("="*70)

    agent = Colleague1Agent(api_key=API_KEY)
    session_id = "test-colleague1"
    messages = []

    for i, user_msg in enumerate(COLLEAGUE1_USER_MESSAGES, 1):
        print(f"\n[턴 {i}] 나: {user_msg}")
        messages.append({"role": "user", "content": user_msg})

        try:
            response = await agent.chat(messages, session_id=session_id)
            print(f"[턴 {i}] 동료1: {response}")
            messages.append({"role": "assistant", "content": response})

            # 존댓말 체크
            if check_jondaenmal(response) and not check_banmal(response):
                print(f"   ⚠️ 존댓말 감지! (동료1은 반말을 써야 함)")
            else:
                print(f"   ✅ 반말 확인")
        except Exception as e:
            print(f"   ❌ 에러: {e}")

    print("\n" + "-"*70)


async def test_colleague2_agent():
    """Colleague2 Agent 테스트 - 존댓말 + 선생님 사용 확인"""
    print("\n" + "="*70)
    print("👨‍🎨 COLLEAGUE2 AGENT 테스트 (30대 남성 화가: 존댓말 + 선생님)")
    print("="*70)

    agent = Colleague2Agent(api_key=API_KEY)
    session_id = "test-colleague2"
    messages = []

    for i, user_msg in enumerate(COLLEAGUE2_USER_MESSAGES, 1):
        print(f"\n[턴 {i}] 나: {user_msg}")
        messages.append({"role": "user", "content": user_msg})

        try:
            response = await agent.chat(messages, session_id=session_id)
            print(f"[턴 {i}] 동료2: {response}")
            messages.append({"role": "assistant", "content": response})

            # 반말 체크 (동료2는 존댓말을 써야 함)
            if check_banmal(response):
                print(f"   ⚠️ 반말 감지! (동료2는 존댓말을 써야 함)")
            elif check_jondaenmal(response):
                print(f"   ✅ 존댓말 확인")
        except Exception as e:
            print(f"   ❌ 에러: {e}")

    print("\n" + "-"*70)


async def main():
    print("\n" + "🚀"*35)
    print("모든 에이전트 말투 테스트 시작")
    print("🚀"*35)

    await test_son_agent()
    await test_jangmo_agent()
    await test_colleague1_agent()
    await test_colleague2_agent()

    print("\n" + "✅"*35)
    print("테스트 완료!")
    print("✅"*35 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
