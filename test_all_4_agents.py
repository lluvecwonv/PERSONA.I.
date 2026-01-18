"""
4개 에이전트 모두 Fine-tuned 모델로 간단 대화 테스트
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드 (올바른 경로 지정)
load_dotenv(dotenv_path=Path(__file__).parent / "moral_agent_website" / ".env")

# Backend path 추가
backend_path = Path(__file__).parent / "moral_agent_website" / "backend"
sys.path.insert(0, str(backend_path))

from agents.jangmo_agent.conversation_agent import JangmoAgent
from agents.son_agent.conversation_agent import SonAgent
from agents.colleague1_agent.conversation_agent import Colleague1Agent
from agents.colleague2_agent.conversation_agent import Colleague2Agent

async def test_agent(agent_class, agent_name, user_message, session_id):
    """단일 에이전트 테스트"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"❌ OPENAI_API_KEY not set")
        return False

    try:
        print(f"\n{'=' * 70}")
        print(f"{agent_name} 테스트")
        print(f"{'=' * 70}")

        # 에이전트 생성
        agent = agent_class(api_key=api_key)
        print(f"✅ 모델: {agent.model}")

        # 간단한 대화 테스트
        messages = [{"role": "user", "content": user_message}]
        print(f"\n사용자: {user_message}")

        response = await agent.chat(messages, session_id=session_id)
        print(f"{agent_name}: {response}")

        print(f"\n✅ {agent_name} 테스트 성공!")
        return True

    except Exception as e:
        print(f"\n❌ {agent_name} 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """4개 에이전트 모두 테스트"""
    print("\n" + "=" * 70)
    print("Fine-tuned 모델 - 4개 에이전트 통합 테스트")
    print("=" * 70)
    print(f"모델: ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs")
    print("=" * 70)

    results = []

    # 1. Jangmo (장모 - AI 복원 반대)
    results.append(await test_agent(
        JangmoAgent,
        "장모 (Jangmo)",
        "장모님, AI로 아내를 복원하는 것에 대해 어떻게 생각하세요?",
        "test_jangmo"
    ))

    # 2. Son (아들 - AI 복원 찬성)
    results.append(await test_agent(
        SonAgent,
        "아들 (Son)",
        "아버지, AI로 어머니를 복원하는 것에 대해 어떻게 생각하세요?",
        "test_son"
    ))

    # 3. Colleague1 (동료 화가 - AI 예술 반대)
    results.append(await test_agent(
        Colleague1Agent,
        "동료 화가 (Colleague1)",
        "동료님, AI가 그린 그림을 전시회에 내는 것 어떻게 생각하세요?",
        "test_colleague1"
    ))

    # 4. Colleague2 (후배 화가 - AI 예술 찬성)
    results.append(await test_agent(
        Colleague2Agent,
        "후배 화가 (Colleague2)",
        "선생님, AI가 그린 그림을 전시회에 내는 것 어떻게 생각하세요?",
        "test_colleague2"
    ))

    # 최종 결과
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    total = len(results)
    success = sum(results)

    print(f"총 {total}개 에이전트 중 {success}개 성공")

    if success == total:
        print("\n✅ 모든 에이전트가 Fine-tuned 모델로 정상 작동!")
    else:
        print(f"\n⚠️ {total - success}개 에이전트 실패")

    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
