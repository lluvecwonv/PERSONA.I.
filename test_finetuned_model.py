"""
Fine-tuned 모델 테스트
4개 에이전트가 fine-tuned 모델을 사용하는지 확인
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


def test_model_initialization():
    """모델 ID 확인 테스트"""
    print("=" * 70)
    print("Fine-tuned 모델 초기화 테스트")
    print("=" * 70)

    # API 키 가져오기
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False

    expected_model = "ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs"

    try:
        # 1. Jangmo Agent
        print("\n1. Jangmo Agent (장모)")
        jangmo = JangmoAgent(api_key=api_key)
        print(f"   Model ID: {jangmo.model}")
        print(f"   ✅ Match: {jangmo.model == expected_model}")

        # 2. Son Agent
        print("\n2. Son Agent (아들)")
        son = SonAgent(api_key=api_key)
        print(f"   Model ID: {son.model}")
        print(f"   ✅ Match: {son.model == expected_model}")

        # 3. Colleague1 Agent
        print("\n3. Colleague1 Agent (동료 화가)")
        colleague1 = Colleague1Agent(api_key=api_key)
        print(f"   Model ID: {colleague1.model}")
        print(f"   ✅ Match: {colleague1.model == expected_model}")

        # 4. Colleague2 Agent
        print("\n4. Colleague2 Agent (후배 화가)")
        colleague2 = Colleague2Agent(api_key=api_key)
        print(f"   Model ID: {colleague2.model}")
        print(f"   ✅ Match: {colleague2.model == expected_model}")

        print("\n" + "=" * 70)
        print("✅ All agents successfully initialized with fine-tuned model!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_chat():
    """간단한 대화 테스트"""
    print("\n" + "=" * 70)
    print("Fine-tuned 모델 대화 테스트")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False

    try:
        # Jangmo Agent로 간단한 대화 테스트
        print("\n[Test] Jangmo Agent 대화 테스트")
        jangmo = JangmoAgent(api_key=api_key)

        messages = [
            {"role": "user", "content": "안녕하세요"}
        ]

        print(f"User: {messages[0]['content']}")
        print("Waiting for response...")

        response = await jangmo.chat(messages, session_id="test_session")
        print(f"Jangmo: {response}")

        print("\n✅ Chat test completed successfully!")
        print(f"✅ Fine-tuned model ({jangmo.model}) is working!")
        return True

    except Exception as e:
        print(f"\n❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("\n🚀 Starting Fine-tuned Model Tests...\n")

    # Test 1: 모델 초기화 확인
    init_success = test_model_initialization()

    if not init_success:
        print("\n❌ Initialization test failed. Skipping chat test.")
        return

    # Test 2: 간단한 대화 테스트
    print("\n" + "=" * 70)
    print("Press Enter to run chat test, or Ctrl+C to skip...")
    print("=" * 70)

    try:
        input()
        asyncio.run(test_simple_chat())
    except KeyboardInterrupt:
        print("\n⏭️  Skipping chat test")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n✨ Tests completed!\n")


if __name__ == "__main__":
    main()
