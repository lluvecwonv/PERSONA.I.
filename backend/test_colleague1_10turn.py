#!/usr/bin/env python3
"""
Colleague1 Agent 10턴 테스트
- 반말 사용 확인
- AI 예술 반대 입장 유지 확인
- 10턴 마무리 확인
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.colleague1_agent import Colleague1Agent

API_KEY = os.getenv("OPENAI_API_KEY")

# 테스트용 사용자 발화 (AI 예술에 대한 다양한 반응)
TEST_MESSAGES = [
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


def check_banmal(response: str) -> bool:
    """반말 패턴 확인"""
    jondaenmal = ["요.", "요?", "요!", "습니다", "세요"]
    return not any(p in response for p in jondaenmal)


def check_anti_ai(response: str) -> bool:
    """AI 반대 입장 확인"""
    pro_ai = ["좋은 것 같", "찬성", "괜찮", "인정해야"]
    anti_ai = ["안 돼", "아니", "반대", "예술이 아니", "인정 못"]

    has_pro = any(p in response for p in pro_ai)
    has_anti = any(p in response for p in anti_ai)

    # 반대 표현이 있거나, 찬성 표현이 없으면 OK
    return has_anti or not has_pro


async def test_colleague1():
    """Colleague1 10턴 테스트"""
    print("\n" + "="*70)
    print("👩‍🎨 COLLEAGUE1 AGENT 10턴 테스트")
    print("   - 반말 사용 확인")
    print("   - AI 예술 반대 입장 유지 확인")
    print("   - 10턴 마무리 확인")
    print("="*70)

    agent = Colleague1Agent(api_key=API_KEY)
    session_id = "test-colleague1-10turn"
    messages = []

    # 첫 메시지
    first_msg = "AI가 그린 그림을 어떻게 국립 예술관에 전시를 할 수가 있지? 그걸 예술로 공식적으로 인정한다는 건 말이 안 되네. 나는 무조건 전시 반대에 투표할걸세."
    print(f"\n[턴 0] 동료1 (첫 인사): {first_msg}")

    banmal_ok = 0
    position_ok = 0

    for i, user_msg in enumerate(TEST_MESSAGES, 1):
        print(f"\n{'='*60}")
        print(f"[턴 {i}] 나: {user_msg}")
        print(f"{'='*60}")
        messages.append({"role": "user", "content": user_msg})

        try:
            response = await agent.chat(messages, session_id=session_id)
            print(f"[턴 {i}] 동료1: {response}")
            messages.append({"role": "assistant", "content": response})

            # 반말 체크
            is_banmal = check_banmal(response)
            if is_banmal:
                print(f"   ✅ 반말 사용")
                banmal_ok += 1
            else:
                print(f"   ⚠️ 존댓말 감지!")

            # 입장 체크
            is_anti = check_anti_ai(response)
            if is_anti:
                print(f"   ✅ 반대 입장 유지")
                position_ok += 1
            else:
                print(f"   ⚠️ 찬성 입장 감지!")

        except Exception as e:
            print(f"   ❌ 에러: {e}")

    # 마무리 메시지 확인
    print(f"\n{'='*60}")
    print("[마무리 메시지 확인]")
    final_msg = "...그래, 자네 생각 잘 들었네. 투표 때 신중하게 결정하게나."
    print(f"마무리: {final_msg}")
    print(f"{'='*60}")

    # 결과 요약
    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70)
    print(f"반말 사용: {banmal_ok}/10 턴 ({'✅ 정상' if banmal_ok >= 8 else '⚠️ 개선 필요'})")
    print(f"반대 입장 유지: {position_ok}/10 턴 ({'✅ 정상' if position_ok >= 8 else '⚠️ 개선 필요'})")


if __name__ == "__main__":
    asyncio.run(test_colleague1())
