#!/usr/bin/env python3
"""
에이전트와 대화 테스트 스크립트
"""
import requests
import json

# 서버 URL (고정 도메인!)
BASE_URL = "https://api.persona-ai-agent.com"

# 테스트 데이터
CONTEXT_ID = "test-context-123"
USER_ID = "test-user-456"
AGENT_ID = "agent-1"  # 화가 지망생


def test_first_chat():
    """첫 대화 테스트 (에이전트가 먼저 인사)"""
    print("\n" + "="*60)
    print("🎨 화가 지망생 에이전트 (agent-1) - 첫 대화")
    print("="*60)

    url = f"{BASE_URL}/api/game/context/{CONTEXT_ID}/chat/start"
    data = {
        "agent_id": AGENT_ID,
        "context_id": CONTEXT_ID,
        "user_id": USER_ID
    }

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        ai_message = result["data"]["response"]
        metadata = result["data"]["metadata"]

        print(f"\n[AI 첫 인사]")
        print(ai_message)
        print(f"\n[메타데이터] {metadata}")

        return True

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        return False


def test_chat(message):
    """일반 대화 테스트"""
    print(f"\n[사용자] {message}")

    url = f"{BASE_URL}/api/game/context/{CONTEXT_ID}/chat"
    data = {
        "agent_id": AGENT_ID,
        "message": message,
        "context_id": CONTEXT_ID,
        "user_id": USER_ID
    }

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        ai_message = result["data"]["response"]
        metadata = result["data"]["metadata"]

        print(f"\n[AI 응답]")
        print(ai_message)
        print(f"\n[메타데이터] {metadata}")

        return True

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 에이전트 대화 테스트 시작\n")

    # 1. 첫 대화 (에이전트가 먼저 인사)
    if not test_first_chat():
        print("\n테스트 실패!")
        exit(1)

    # 2. 사용자 응답
    test_chat("안녕하세요! AI 그림 생성 툴에 대해 어떻게 생각하시나요?")

    # 3. 추가 대화
    test_chat("AI가 만든 그림도 예술이라고 생각하세요?")

    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60 + "\n")
