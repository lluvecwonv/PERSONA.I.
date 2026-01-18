#!/usr/bin/env python3
"""
페르소나 검증 통합 테스트
langchain_service.py에 통합된 persona_validator가 정상 작동하는지 확인
"""
import os
import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv()

from utils.persona_validator import validate_and_fix_persona, PERSONA_RULES


async def test_persona_validation():
    """페르소나 검증 테스트"""
    print("\n" + "="*60)
    print("🧪 페르소나 검증 통합 테스트")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    # 테스트 케이스: (에이전트, 잘못된 응답, 예상 수정 여부)
    test_cases = [
        {
            "name": "son - 반말 사용 (잘못됨)",
            "agent": "son",
            "response": "아버지 뭐해? 밥 먹었냐?",
            "expect_fix": True,
        },
        {
            "name": "son - 존댓말 사용 (올바름)",
            "agent": "son",
            "response": "아버지, 저도 그렇게 생각해요. 엄마가 보고 싶으시죠?",
            "expect_fix": False,
        },
        {
            "name": "jangmo - 존댓말 사용 (잘못됨)",
            "agent": "jangmo",
            "response": "사위님, 그렇게 하시면 안 됩니다.",
            "expect_fix": True,
        },
        {
            "name": "jangmo - 반말 사용 (올바름)",
            "agent": "jangmo",
            "response": "자네, 그건 아니야. 우리 딸의 존엄성을 생각해봐.",
            "expect_fix": False,
        },
        {
            "name": "colleague1 - 존댓말 사용 (잘못됨, 반말이어야 함)",
            "agent": "colleague1",
            "response": "선배님, 저는 AI 전시가 걱정됩니다. 규칙을 어기는 거 아닌가요?",
            "expect_fix": True,
        },
        {
            "name": "colleague1 - 반말 사용 (올바름)",
            "agent": "colleague1",
            "response": "자네, 그건 아니야. 우리 협회 규정을 어기는 거 아닌가?",
            "expect_fix": False,
        },
        {
            "name": "colleague2 - 반말 사용 (잘못됨)",
            "agent": "colleague2",
            "response": "선생님 뭐해? 그건 아니지.",
            "expect_fix": True,
        },
        {
            "name": "colleague2 - 존댓말 사용 (올바름)",
            "agent": "colleague2",
            "response": "선생님, AI 전시가 더 많은 분들께 예술을 보여드릴 수 있어요.",
            "expect_fix": False,
        },
        {
            "name": "friend - 검증 제외",
            "agent": "friend",
            "response": "그냥 아무말이나 해도 됨",
            "expect_fix": False,
        },
        {
            "name": "artist_apprentice - 검증 제외",
            "agent": "artist_apprentice",
            "response": "아무말이나 해도 패스",
            "expect_fix": False,
        },
    ]

    passed = 0
    failed = 0

    for tc in test_cases:
        print(f"\n{'─'*50}")
        print(f"📋 테스트: {tc['name']}")
        print(f"   원본: \"{tc['response'][:50]}...\"")

        fixed_response, was_fixed = await validate_and_fix_persona(
            tc["response"], tc["agent"], api_key
        )

        if tc["expect_fix"] and was_fixed:
            print(f"   ✅ 예상대로 수정됨")
            print(f"   수정: \"{fixed_response[:50]}...\"")
            passed += 1
        elif not tc["expect_fix"] and not was_fixed:
            print(f"   ✅ 예상대로 수정 안 됨 (올바른 응답)")
            passed += 1
        else:
            print(f"   ❌ 예상과 다름! (expect_fix={tc['expect_fix']}, was_fixed={was_fixed})")
            failed += 1

    print("\n" + "="*60)
    print(f"📊 결과: {passed}/{len(test_cases)} 통과")
    if failed > 0:
        print(f"⚠️ {failed}개 실패")
    else:
        print("✅ 모든 테스트 통과!")
    print("="*60)


async def main():
    await test_persona_validation()


if __name__ == "__main__":
    asyncio.run(main())
