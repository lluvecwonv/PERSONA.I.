#!/usr/bin/env python3
"""
SPT V2가 동료1, 동료2 에이전트에서 어떻게 작동하는지 테스트
실제 대화 시나리오 시뮬레이션
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

from agents.spt_agent import SPTAgentV2


async def test_colleague1_scenario():
    """동료1 (AI 전시 반대) 시나리오 테스트"""
    print("\n" + "="*70)
    print("🎨 동료1 에이전트 (AI 예술 전시 반대) 시나리오 테스트")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    spt_v2 = SPTAgentV2(api_key=api_key)

    # 대화 시나리오
    conversation = [
        {"role": "assistant", "content": "AI 전시에 대해 어떻게 생각하나? 혹시 우려되는 점이 있어?"},
    ]

    test_cases = [
        {
            "name": "정상 답변",
            "user_msg": "기존 화가들이 피해볼 것 같아",
        },
        {
            "name": "Off-topic (이익만 얘기)",
            "user_msg": "예술관이 돈 많이 벌겠네",
        },
        {
            "name": "Off-topic (기술 칭찬)",
            "user_msg": "기술이 대단해졌네",
        },
        {
            "name": "이해 못함",
            "user_msg": "무슨 말이야? 뭘 물어보는 거야?",
        },
        {
            "name": "역질문",
            "user_msg": "선배는 어떻게 생각해?",
        },
    ]

    for i, tc in enumerate(test_cases):
        session_id = f"colleague1_test_{i}"
        print(f"\n{'─'*60}")
        print(f"📋 테스트: {tc['name']}")
        print(f"👤 사용자: \"{tc['user_msg']}\"")

        result = await spt_v2.process(
            session_id=session_id,
            user_message=tc["user_msg"],
            conversation_history=conversation,
            topic_context="AI 예술 전시 - 협회 투표 (반대 입장)"
        )

        print(f"\n🎯 전략: {result['strategy']}")
        print(f"❓ 질문 허용: {result['allow_question']}")
        print(f"📝 지시사항:")
        print(f"   {result['instruction'][:200]}...")

        if result.get("suggested_question"):
            print(f"💡 제안 질문: {result['suggested_question'][:100]}...")


async def test_colleague2_scenario():
    """동료2 (AI 전시 찬성) 시나리오 테스트"""
    print("\n" + "="*70)
    print("🖼️ 동료2 에이전트 (AI 예술 전시 찬성) 시나리오 테스트")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    spt_v2 = SPTAgentV2(api_key=api_key)

    # 대화 시나리오
    conversation = [
        {"role": "assistant", "content": "선생님, AI 작품이 공정하다고 생각하세요?"},
    ]

    test_cases = [
        {
            "name": "정상 답변 (공정)",
            "user_msg": "공정하다고 생각해, 규칙 안에서 만들었으니까",
        },
        {
            "name": "정상 답변 (불공정)",
            "user_msg": "불공정해, 다른 작가 그림 베꼈잖아",
        },
        {
            "name": "Off-topic (돈 얘기)",
            "user_msg": "예술관이 이익 보겠지 뭐",
        },
        {
            "name": "Off-topic (기술 얘기)",
            "user_msg": "AI 기술이 정말 발전했네요",
        },
        {
            "name": "모르겠다",
            "user_msg": "글쎄요, 잘 모르겠어요",
        },
    ]

    for i, tc in enumerate(test_cases):
        session_id = f"colleague2_test_{i}"
        print(f"\n{'─'*60}")
        print(f"📋 테스트: {tc['name']}")
        print(f"👤 사용자: \"{tc['user_msg']}\"")

        result = await spt_v2.process(
            session_id=session_id,
            user_message=tc["user_msg"],
            conversation_history=conversation,
            topic_context="AI 예술 전시 - 협회 투표 (찬성 입장)"
        )

        print(f"\n🎯 전략: {result['strategy']}")
        print(f"❓ 질문 허용: {result['allow_question']}")
        print(f"📝 지시사항:")
        print(f"   {result['instruction'][:200]}...")

        if result.get("suggested_question"):
            print(f"💡 제안 질문: {result['suggested_question'][:100]}...")


async def test_multi_turn_conversation():
    """멀티턴 대화 테스트 - 상태 추적 확인"""
    print("\n" + "="*70)
    print("🔄 멀티턴 대화 테스트 (상태 추적)")
    print("="*70)

    api_key = os.getenv("OPENAI_API_KEY")
    spt_v2 = SPTAgentV2(api_key=api_key)
    session_id = "multi_turn_test"

    # 대화 시뮬레이션
    turns = [
        {
            "agent": "AI 전시에 대해 어떻게 생각하나? 우려되는 점이 있어?",
            "user": "돈 벌겠지 뭐",
        },
        {
            "agent": "그렇긴 하지. 근데 혹시 걱정되는 부분은 없어?",  # 재연결
            "user": "기존 화가들이 힘들어질 것 같아",
        },
        {
            "agent": "그렇구나. 그 화가들 입장에서는 어떨 것 같아?",
            "user": "무슨 말이야?",
        },
    ]

    conversation = []

    for i, turn in enumerate(turns):
        print(f"\n{'─'*60}")
        print(f"🔄 턴 {i+1}")
        print(f"🤖 에이전트: \"{turn['agent']}\"")
        print(f"👤 사용자: \"{turn['user']}\"")

        conversation.append({"role": "assistant", "content": turn["agent"]})

        result = await spt_v2.process(
            session_id=session_id,
            user_message=turn["user"],
            conversation_history=conversation,
            topic_context="AI 예술 전시"
        )

        print(f"\n🎯 전략: {result['strategy']}")
        print(f"❓ 질문 허용: {result['allow_question']}")
        print(f"📊 응답 상태: {result.get('response_status', 'N/A')}")

        conversation.append({"role": "user", "content": turn["user"]})


async def main():
    print("🚀 SPT V2 동료 에이전트 테스트 시작")

    await test_colleague1_scenario()
    await test_colleague2_scenario()
    await test_multi_turn_conversation()

    print("\n" + "="*70)
    print("✅ 모든 테스트 완료!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
