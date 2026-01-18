#!/usr/bin/env python3
"""
Son Agent SPT 전략 + 공감 배분 테스트
"""
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.son_agent import SonAgent

# API 키
API_KEY = os.getenv("OPENAI_API_KEY")

# 테스트용 사용자 발화
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


async def test_son_spt():
    """Son Agent SPT 전략 테스트"""
    print("\n" + "="*70)
    print("🧑 SON AGENT SPT 전략 + 공감 테스트")
    print("="*70)

    agent = SonAgent(api_key=API_KEY)
    session_id = "test-son-spt"
    messages = []

    # SPT 상태 추적
    spt_used = []
    question_turns = []

    for i, user_msg in enumerate(SON_USER_MESSAGES, 1):
        print(f"\n{'='*50}")
        print(f"[턴 {i}] 아버지: {user_msg}")
        print(f"{'='*50}")
        messages.append({"role": "user", "content": user_msg})

        # SPT 상태 확인 (chat 호출 전)
        state = agent.get_spt_state(session_id)
        unused_count = len(state["unused_spt"])
        can_ask = agent.can_ask_question(session_id)

        print(f"📊 SPT 상태: 남은 전략={unused_count}개, 질문가능={can_ask}")

        try:
            response = await agent.chat(messages, session_id=session_id)
            print(f"\n[턴 {i}] 아들: {response}")
            messages.append({"role": "assistant", "content": response})

            # SPT 상태 확인 (chat 호출 후)
            state_after = agent.get_spt_state(session_id)
            unused_after = len(state_after["unused_spt"])

            # SPT가 소비되었는지 확인
            if unused_after < unused_count:
                consumed = unused_count - unused_after
                print(f"✅ SPT 전략 {consumed}개 소비됨! (남은: {unused_after}개)")
                spt_used.append(i)
                question_turns.append(i)
            else:
                print(f"📝 공감만 (SPT 소비 없음)")

            # 질문 포함 여부 확인
            has_question = "?" in response or "요?" in response or "까요" in response
            if has_question:
                print(f"❓ 질문 포함됨")

        except Exception as e:
            print(f"❌ 에러: {e}")

    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70)

    final_state = agent.get_spt_state(session_id)
    print(f"\n총 턴 수: 10")
    print(f"SPT 사용 턴: {spt_used} ({len(spt_used)}개)")
    print(f"공감만 턴: {[i for i in range(1, 11) if i not in spt_used]} ({10 - len(spt_used)}개)")
    print(f"남은 SPT 전략: {len(final_state['unused_spt'])}개")
    print(f"사용된 SPT 전략: {final_state['used_spt']}")

    # 기대값 확인
    print("\n" + "-"*50)
    if len(spt_used) == 5:
        print("✅ SPT 5개 사용 - 정상!")
    else:
        print(f"⚠️ SPT {len(spt_used)}개 사용 (기대: 5개)")

    if 10 - len(spt_used) == 5:
        print("✅ 공감만 5개 - 정상!")
    else:
        print(f"⚠️ 공감만 {10 - len(spt_used)}개 (기대: 5개)")


if __name__ == "__main__":
    asyncio.run(test_son_spt())
