"""
Fine-tuned 모델 실제 대화 테스트 + SPT V2 디버깅
Jangmo Agent와 5턴 대화
"""
import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드 (올바른 경로 지정)
load_dotenv(dotenv_path=Path(__file__).parent / "moral_agent_website" / ".env")

# Backend path 추가
backend_path = Path(__file__).parent / "moral_agent_website" / "backend"
sys.path.insert(0, str(backend_path))

# 로깅 설정 (DEBUG 레벨로 변경해서 모든 로그 확인)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

from agents.jangmo_agent.conversation_agent import JangmoAgent
from agents.spt_agent.spt_agent_v2 import SPTAgentV2

async def test_conversation():
    """여러 턴의 대화 테스트 + SPT V2 디버깅"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    print("=" * 80)
    print("🧪 Fine-tuned 모델 대화 테스트 + SPT V2 디버깅 (Jangmo Agent)")
    print("=" * 80)
    print(f"📦 모델: ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs")
    print("=" * 80)
    print()

    # SPT Agent V2 초기화
    spt_v2 = SPTAgentV2(api_key=api_key)
    print("✅ SPT Agent V2 초기화 완료\n")

    # Jangmo Agent 생성 (SPT V2 주입)
    jangmo = JangmoAgent(api_key=api_key, spt_agent_v2=spt_v2)
    print(f"✅ Jangmo Agent 초기화 완료")
    print(f"   - 모델: {jangmo.model}")
    print(f"   - SPT V2 통합: {'Yes' if jangmo.spt_agent_v2 else 'No'}\n")

    # 테스트 대화 시나리오
    user_messages = [
        "장모님, 저는 아내를 다시 보고 싶어요",
        "기술이 발전했으니까 가능하지 않을까요?",
        "아내가 원했을 수도 있지 않나요?",
        "그리움이 너무 큽니다",
        "한 번만 생각해주세요"
    ]

    messages = []

    for i, user_msg in enumerate(user_messages, 1):
        print("\n" + "=" * 80)
        print(f"🔄 [Turn {i}/5]")
        print("=" * 80)
        print(f"💬 사위: {user_msg}")
        print()

        # 메시지 추가
        messages.append({"role": "user", "content": user_msg})

        try:
            # ===== Step 1: Self-Reflection 생성 =====
            print("🧠 Step 1: Self-Reflection 생성 중...")
            reflection = await jangmo._generate_self_reflection(messages, temperature=0.3)
            print(f"\n📋 Self-Reflection 결과:")
            print("-" * 80)
            print(reflection)
            print("-" * 80)

            # ===== Step 2: SPT 필요성 판단 =====
            spt_needed = jangmo._parse_spt_necessity(reflection)
            print(f"\n🔍 Step 2: SPT 필요성 판단 → {'✅ YES (SPT 실행)' if spt_needed else '❌ NO (SPT 스킵)'}")

            # ===== Step 3: 조건부 SPT V2 호출 =====
            spt_instruction = None
            if spt_needed and jangmo.spt_agent_v2:
                print("\n🧠 Step 3: SPT Agent V2 실행 중...")
                last_user_msg = jangmo._extract_last_user_message(messages)
                stored_keywords = jangmo._get_stored_keywords("test_conversation")

                spt_result = await jangmo.spt_agent_v2.process(
                    session_id=f"{jangmo._get_character_name()}_test_conversation",
                    user_message=last_user_msg,
                    conversation_history=messages,
                    topic_context=jangmo._get_topic_context(),
                    question_keywords=stored_keywords
                )

                spt_instruction = spt_result["instruction"]

                print("\n📦 SPT V2 결과:")
                print("-" * 80)
                print(f"  전략: {spt_result['strategy']}")
                print(f"  질문 허용: {spt_result['allow_question']}")
                if spt_result.get('suggested_question'):
                    print(f"  제안된 질문: {spt_result['suggested_question']}")
                if spt_result.get('suggested_keywords'):
                    print(f"  키워드: {spt_result['suggested_keywords']}")
                print(f"\n  💡 Instruction:")
                print(f"  {spt_instruction}")
                print("-" * 80)

                # 키워드 저장
                if spt_result.get("suggested_keywords"):
                    jangmo._store_keywords("test_conversation", spt_result["suggested_keywords"])
            else:
                print("\n⏭️  Step 3: SPT 스킵됨 (필요하지 않음)")

            # ===== Step 4: 최종 응답 생성 =====
            print("\n💭 Step 4: 최종 응답 생성 중...")
            response = await jangmo.chat(messages, session_id="test_conversation")

            print(f"\n💬 장모: {response}")

            # AI 응답도 메시지에 추가
            messages.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"\n❌ Error at turn {i}: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n" + "=" * 80)
    print("✅ 대화 테스트 완료!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_conversation())
