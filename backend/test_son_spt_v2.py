#!/usr/bin/env python3
"""
Son Agent SPT 전략 테스트 V2 - ResponseType 기반
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
    # 1. UNCERTAIN - 불확실/망설임 → SPT 질문 사용
    ("글쎄다, 잘 모르겠어", "UNCERTAIN"),

    # 2. AGREE - 동의 → 공감만
    ("그래, 네 말이 맞는 것 같아", "AGREE"),

    # 3. DISAGREE - 반대 → SPT 질문 사용
    ("아니, 난 그렇게 생각 안 해", "DISAGREE"),

    # 4. AGREE - 동의 → 공감만
    ("음, 일리가 있네", "AGREE"),

    # 5. SHORT - 단답 → SPT 질문 사용
    ("응", "SHORT"),

    # 6. AGREE - 동의 → 공감만
    ("맞아", "AGREE"),

    # 7. QUESTION - 질문 → SPT 질문 사용 (역질문)
    ("근데 비용은 얼마나 들어?", "QUESTION"),

    # 8. OTHER - 기타 → 공감만
    ("그건 그렇고...", "OTHER"),

    # 9. UNCERTAIN - 불확실 → SPT 질문 사용
    ("여전히 고민이야", "UNCERTAIN"),

    # 10. AGREE - 동의 → 공감만
    ("알겠어, 고마워", "AGREE"),
]


async def test_son_spt_v2():
    """ResponseType 기반 SPT 테스트"""
    print("\n" + "="*70)
    print("🧑 SON AGENT SPT 전략 테스트 V2 - ResponseType 기반")
    print("="*70)

    agent = SonAgent(api_key=API_KEY)
    session_id = "test-son-spt-v2"
    messages = []

    spt_turns = []
    empathy_turns = []

    for i, (user_msg, expected_type) in enumerate(TEST_MESSAGES, 1):
        print(f"\n{'='*60}")
        print(f"[턴 {i}] 아버지: {user_msg}")
        print(f"        (예상 타입: {expected_type})")
        print(f"{'='*60}")
        messages.append({"role": "user", "content": user_msg})

        # SPT 상태 확인
        state_before = agent.get_spt_state(session_id)
        unused_before = len(state_before["unused_spt"])

        try:
            response = await agent.chat(messages, session_id=session_id)
            print(f"\n[턴 {i}] 아들: {response}")
            messages.append({"role": "assistant", "content": response})

            # SPT 소비 확인
            state_after = agent.get_spt_state(session_id)
            unused_after = len(state_after["unused_spt"])

            if unused_after < unused_before:
                print(f"✅ SPT 사용 (남은: {unused_after}개)")
                spt_turns.append((i, expected_type))
            else:
                print(f"💬 공감만 (SPT 미사용)")
                empathy_turns.append((i, expected_type))

        except Exception as e:
            print(f"❌ 에러: {e}")

    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70)

    print(f"\n✅ SPT 사용 턴 ({len(spt_turns)}개):")
    for turn, rtype in spt_turns:
        print(f"   턴 {turn}: {rtype}")

    print(f"\n💬 공감만 턴 ({len(empathy_turns)}개):")
    for turn, rtype in empathy_turns:
        print(f"   턴 {turn}: {rtype}")

    # 예상 결과 검증
    print("\n" + "-"*50)
    spt_types = {rtype for _, rtype in spt_turns}
    empathy_types = {rtype for _, rtype in empathy_turns}

    expected_spt = {"UNCERTAIN", "DISAGREE", "SHORT", "QUESTION"}
    expected_empathy = {"AGREE", "OTHER"}

    print(f"SPT 사용 타입: {spt_types}")
    print(f"공감만 타입: {empathy_types}")

    if spt_types <= expected_spt | empathy_types:
        print("✅ ResponseType 기반 SPT 배분 정상!")
    else:
        print("⚠️ 예상과 다른 배분")


if __name__ == "__main__":
    asyncio.run(test_son_spt_v2())
