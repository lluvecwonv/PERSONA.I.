#!/usr/bin/env python3
"""
SPT V2 아키텍처 테스트
- DST: 대화 상태 추적
- Controller: 응답 전략 결정
- Planner: 질문 후보 생성
"""
import os
import sys
import asyncio
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv()

from agents.spt_agent import (
    SPTAgentV2,
    DialogueStateTracker,
    SPTController,
    UserResponseStatus,
)


def test_dst_basic():
    """DST 기본 기능 테스트"""
    print("\n" + "="*60)
    print("🧪 DST (Dialogue State Tracker) 테스트")
    print("="*60)

    dst = DialogueStateTracker()
    session_id = "test_session_1"

    # 테스트 케이스: (사용자 메시지, 예상 상태)
    test_cases = [
        # 정상 답변
        ("개인정보 유출이 걱정돼", UserResponseStatus.ANSWERED),
        ("피해볼 사람 있을 것 같아", UserResponseStatus.ANSWERED),

        # 이해 못함
        ("무슨 말이야?", UserResponseStatus.NEEDS_CLARIFICATION),
        ("뭔 소리야", UserResponseStatus.NEEDS_CLARIFICATION),

        # 모르겠다
        ("글쎄 모르겠어", UserResponseStatus.DONT_KNOW),
        ("잘 모르겠는데", UserResponseStatus.DONT_KNOW),

        # 역질문
        ("너는 어떻게 생각해?", UserResponseStatus.ASKING_BACK),
    ]

    passed = 0
    failed = 0

    for user_msg, expected_status in test_cases:
        # 각 테스트마다 새 세션
        test_session = f"test_{user_msg[:10]}"

        state = dst.analyze_user_response(
            session_id=test_session,
            user_message=user_msg
        )

        is_pass = state.response_status == expected_status
        status_icon = "✅" if is_pass else "❌"

        print(f"\n{status_icon} 메시지: \"{user_msg}\"")
        print(f"   예상: {expected_status.value if expected_status else 'None'}")
        print(f"   결과: {state.response_status.value if state.response_status else 'None'}")

        if is_pass:
            passed += 1
        else:
            failed += 1

    print(f"\n📈 DST 테스트 결과: {passed}/{passed+failed} 통과")
    return passed, failed


def test_controller_decisions():
    """Controller 응답 전략 결정 테스트"""
    print("\n" + "="*60)
    print("🧪 SPT Controller 테스트")
    print("="*60)

    dst = DialogueStateTracker()
    controller = SPTController(dst=dst)

    # 테스트용 이전 질문
    prev_question = "혹시 구매한다면 우려되는 점이 있어?"

    # 테스트 시나리오
    scenarios = [
        {
            "name": "정상 답변 후 새 질문 허용",
            "setup": lambda s: dst.analyze_user_response(s, "개인정보가 걱정돼", prev_question),
            "user_msg": "개인정보가 걱정돼",
            "candidates": ["다른 사람 입장에서는 어떨까?"],
            "expected_allow": True,
        },
        {
            "name": "이해 못함 → 설명 필요",
            "setup": lambda s: dst.analyze_user_response(s, "무슨 말이야?", prev_question),
            "user_msg": "무슨 말이야?",
            "candidates": [],
            "expected_allow": False,
        },
        {
            "name": "역질문 → 먼저 답해야 함",
            "setup": lambda s: dst.analyze_user_response(s, "너는 어떻게 생각해?", prev_question),
            "user_msg": "너는 어떻게 생각해?",
            "candidates": [],
            "expected_allow": False,
        },
    ]

    passed = 0
    failed = 0

    for i, scenario in enumerate(scenarios):
        session_id = f"controller_test_{i}"

        # 상태 설정
        scenario["setup"](session_id)

        # Controller 결정
        decision = controller.decide(
            session_id=session_id,
            user_message=scenario["user_msg"],
            question_candidates=scenario["candidates"]
        )

        is_pass = decision.allow_question == scenario["expected_allow"]
        status_icon = "✅" if is_pass else "❌"

        print(f"\n{status_icon} {scenario['name']}")
        print(f"   메시지: \"{scenario['user_msg']}\"")
        print(f"   전략: {decision.strategy.value}")
        print(f"   질문 허용: {decision.allow_question} (예상: {scenario['expected_allow']})")

        if is_pass:
            passed += 1
        else:
            failed += 1

    print(f"\n📈 Controller 테스트 결과: {passed}/{passed+failed} 통과")
    return passed, failed


async def test_spt_v2_integration():
    """SPT V2 통합 테스트"""
    print("\n" + "="*60)
    print("🧪 SPT V2 통합 테스트")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return 0, 1

    spt_v2 = SPTAgentV2(api_key=api_key)

    # 테스트 시나리오
    scenarios = [
        {
            "name": "정상 답변",
            "user_msg": "개인정보 유출이 걱정돼",
            "history": [
                {"role": "assistant", "content": "혹시 구매한다면 우려되는 점이 있어?"},
            ],
            "expected_allow": True,
        },
        {
            "name": "이해 못함",
            "user_msg": "무슨 말이야?",
            "history": [
                {"role": "assistant", "content": "이 서비스가 자율성을 침해한다고 생각해?"},
            ],
            "expected_allow": False,
        },
    ]

    passed = 0
    failed = 0

    for i, scenario in enumerate(scenarios):
        session_id = f"integration_test_{i}"

        print(f"\n🔍 테스트: {scenario['name']}")
        print(f"   사용자: \"{scenario['user_msg']}\"")

        try:
            result = await spt_v2.process(
                session_id=session_id,
                user_message=scenario["user_msg"],
                conversation_history=scenario["history"],
                topic_context="AI 부활 서비스"
            )

            is_pass = result["allow_question"] == scenario["expected_allow"]
            status_icon = "✅" if is_pass else "❌"

            print(f"   {status_icon} 전략: {result['strategy']}")
            print(f"   질문 허용: {result['allow_question']} (예상: {scenario['expected_allow']})")
            print(f"   지시사항: {result['instruction'][:80]}...")

            if is_pass:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"   ❌ 오류: {e}")
            failed += 1

    print(f"\n📈 SPT V2 통합 테스트 결과: {passed}/{passed+failed} 통과")
    return passed, failed


async def main():
    print("🚀 SPT V2 아키텍처 테스트 시작")
    print("="*60)

    # 1. DST 테스트
    dst_passed, dst_failed = test_dst_basic()

    # 2. Controller 테스트
    ctrl_passed, ctrl_failed = test_controller_decisions()

    # 3. 통합 테스트
    int_passed, int_failed = await test_spt_v2_integration()

    # 최종 결과
    total_passed = dst_passed + ctrl_passed + int_passed
    total_failed = dst_failed + ctrl_failed + int_failed

    print("\n" + "="*60)
    print("📊 최종 결과")
    print("="*60)
    print(f"DST: {dst_passed}/{dst_passed+dst_failed}")
    print(f"Controller: {ctrl_passed}/{ctrl_passed+ctrl_failed}")
    print(f"통합: {int_passed}/{int_passed+int_failed}")
    print(f"총합: {total_passed}/{total_passed+total_failed}")

    if total_failed == 0:
        print("\n🎉 모든 테스트 통과!")
    else:
        print(f"\n⚠️ {total_failed}개 테스트 실패")


if __name__ == "__main__":
    asyncio.run(main())
