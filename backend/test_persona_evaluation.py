#!/usr/bin/env python3
"""
페르소나 평가 테스트
Gemini를 사용하여 각 에이전트가 페르소나 조건을 만족하는지 평가

평가 기준:
동료1, 장모 (의무론 반대 입장):
- 의무론 입장에서 말하는 것 3번 이상
- SPT 질문 전략 사용
- 협회 멤버로서 플레이어의 책임 상기

동료2, 아들 (공리주의 찬성 입장):
- 공리주의 입장에서 말하는 것 3번 이상
- SPT 질문 전략 사용
- 협회 멤버로서 플레이어의 책임 상기
"""
import os
import sys
import json
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def evaluate_persona(conversation: List[Dict], agent_type: str) -> Dict[str, Any]:
    """
    OpenAI를 사용하여 대화 평가

    Args:
        conversation: 대화 기록 [{"role": "assistant/user", "content": "..."}]
        agent_type: "colleague1", "colleague2", "jangmo", "son"

    Returns:
        평가 결과 딕셔너리
    """
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set"}

    # 에이전트별 평가 기준 설정
    if agent_type in ["colleague1", "jangmo"]:
        stance = "의무론 (반대 입장)"
        stance_description = "규칙, 의무, 도덕적 원칙을 강조하며 AI 기술 사용에 반대하는 입장"
    else:  # colleague2, son
        stance = "공리주의 (찬성 입장)"
        stance_description = "전체적인 이익, 행복 극대화를 강조하며 AI 기술 사용에 찬성하는 입장"

    # 대화 포맷팅
    conversation_text = "\n".join([
        f"{'AI' if msg['role'] == 'assistant' else '사용자'}: {msg['content']}"
        for msg in conversation
    ])

    evaluation_prompt = f"""
다음 대화를 분석하고 AI 에이전트의 페르소나 수행을 평가해주세요.

## 대화 내용
{conversation_text}

## 평가 기준
이 에이전트는 "{stance}" 입장입니다.
{stance_description}

다음 항목을 평가해주세요:

### 1. 윤리적 입장 표현 (3번 이상 필요)
- AI의 발언 중 {stance} 관점에서 한 발언을 모두 찾아주세요
- 각 발언을 인용하고, 왜 그것이 해당 입장인지 설명해주세요
- 총 몇 번 해당 입장을 표현했는지 세어주세요

### 2. SPT 질문 전략 사용
SPT(Social Perspective Taking) 질문 전략:
- 관점 전환: "~의 입장에서 생각해보면?", "다른 사람이라면?"
- 감정 탐색: "어떤 기분이 들어?", "그 사람은 어떻게 느낄까?"
- 이유 탐색: "왜 그렇게 생각해?", "어떤 이유에서?"
- 상황 가정: "만약 ~라면?", "~한 상황이라면?"

AI가 사용한 SPT 질문을 모두 찾아 인용하고, 어떤 전략인지 분류해주세요.

### 3. 책임 상기
- AI가 플레이어의 책임(협회 멤버, 가족 등)을 상기시키는 발언을 찾아주세요

## 출력 형식 (JSON)
{{
    "stance_expressions": {{
        "count": <숫자>,
        "required": 3,
        "passed": <true/false>,
        "examples": [
            {{"quote": "발언 인용", "reason": "해당 입장인 이유"}}
        ]
    }},
    "spt_questions": {{
        "count": <숫자>,
        "strategies_used": ["관점전환", "감정탐색", ...],
        "examples": [
            {{"quote": "질문 인용", "strategy": "전략 유형"}}
        ]
    }},
    "responsibility_reminders": {{
        "count": <숫자>,
        "examples": ["발언 인용"]
    }},
    "overall_score": {{
        "stance": <0-100>,
        "spt": <0-100>,
        "responsibility": <0-100>,
        "total": <0-100>
    }},
    "feedback": "개선점이나 잘한 점에 대한 피드백"
}}

JSON만 출력하세요.
"""

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
        response = llm.invoke(evaluation_prompt)

        # JSON 파싱
        response_text = response.content.strip()
        # ```json ... ``` 제거
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        logger.error(f"응답: {response.content[:500]}")
        return {"error": f"JSON parse error: {e}", "raw": response.content}
    except Exception as e:
        logger.error(f"OpenAI API 오류: {e}")
        return {"error": str(e)}


def print_evaluation_result(agent_name: str, result: Dict[str, Any]):
    """평가 결과 출력"""
    print(f"\n{'='*60}")
    print(f"📊 {agent_name} 페르소나 평가 결과")
    print(f"{'='*60}")

    if "error" in result:
        print(f"❌ 오류: {result['error']}")
        return

    # 1. 윤리적 입장 표현
    stance = result.get("stance_expressions", {})
    count = stance.get("count", 0)
    required = stance.get("required", 3)
    passed = stance.get("passed", False)
    status = "✅" if passed else "❌"
    print(f"\n{status} 윤리적 입장 표현: {count}/{required}회")
    for ex in stance.get("examples", [])[:3]:
        print(f"   • \"{ex.get('quote', '')[:50]}...\"")
        print(f"     → {ex.get('reason', '')}")

    # 2. SPT 질문 전략
    spt = result.get("spt_questions", {})
    spt_count = spt.get("count", 0)
    strategies = spt.get("strategies_used", [])
    print(f"\n📋 SPT 질문 전략: {spt_count}회 사용")
    print(f"   전략: {', '.join(strategies) if strategies else '없음'}")
    for ex in spt.get("examples", [])[:3]:
        print(f"   • [{ex.get('strategy', '')}] \"{ex.get('quote', '')[:50]}...\"")

    # 3. 책임 상기
    resp = result.get("responsibility_reminders", {})
    resp_count = resp.get("count", 0)
    print(f"\n📋 책임 상기: {resp_count}회")
    for ex in resp.get("examples", [])[:2]:
        print(f"   • \"{ex[:50]}...\"")

    # 4. 종합 점수
    scores = result.get("overall_score", {})
    print(f"\n📈 종합 점수:")
    print(f"   • 윤리적 입장: {scores.get('stance', 0)}점")
    print(f"   • SPT 질문: {scores.get('spt', 0)}점")
    print(f"   • 책임 상기: {scores.get('responsibility', 0)}점")
    print(f"   • 총점: {scores.get('total', 0)}점")

    # 5. 피드백
    feedback = result.get("feedback", "")
    if feedback:
        print(f"\n💡 피드백: {feedback}")


# 테스트용 샘플 대화
SAMPLE_CONVERSATIONS = {
    "colleague1": [
        {"role": "assistant", "content": "선배, AI 작품 전시 건으로 투표해야 하는데, 선배 생각은 어때요?"},
        {"role": "user", "content": "글쎄, 나도 잘 모르겠어"},
        {"role": "assistant", "content": "저는 솔직히 반대예요. 규칙을 지키는 게 중요하잖아요. 우리가 정한 자격 기준이 있는데, AI 작품은 그걸 충족 못하잖아요."},
        {"role": "user", "content": "근데 AI 작품도 예술 아니야?"},
        {"role": "assistant", "content": "그렇게 볼 수도 있지만, 원칙이 중요해요. 만약 우리가 예외를 인정하면 다음에는 어디까지 허용해야 할지 모르잖아요. 선배가 협회 멤버로서 기준을 세워야 할 것 같은데, 어떻게 생각해요?"},
        {"role": "user", "content": "음..."},
        {"role": "assistant", "content": "다른 화가들 입장에서 생각해보면 어떨 것 같아요? 그 사람들은 수십 년간 그림을 배워왔는데, AI가 몇 초 만에 비슷한 걸 만들면 기분이 어떨까요?"},
    ],
    "colleague2": [
        {"role": "assistant", "content": "선생님, AI 작품 전시 건 어떻게 생각하세요?"},
        {"role": "user", "content": "잘 모르겠어"},
        {"role": "assistant", "content": "저는 찬성이에요. 더 많은 사람들이 예술을 즐길 수 있으니까요. 전체적으로 보면 이득이 더 크다고 봐요."},
        {"role": "user", "content": "근데 기존 화가들은?"},
        {"role": "assistant", "content": "물론 그분들 마음은 이해해요. 하지만 사회 전체로 보면 새로운 예술 형태가 더 많은 행복을 가져다줄 거예요. 선생님이 협회의 리더로서 앞으로의 방향을 정해야 하잖아요."},
        {"role": "user", "content": "그래도 좀..."},
        {"role": "assistant", "content": "만약 AI 작품이 더 많은 관람객을 끌어들인다면, 결과적으로 협회 전체에도 좋지 않을까요? 선생님은 이 결정이 어떤 결과를 가져올 거라고 예상하세요?"},
    ],
    "jangmo": [
        {"role": "assistant", "content": "사위, 혹시 그 AI 부활 서비스 들어봤어?"},
        {"role": "user", "content": "네, 들어봤어요"},
        {"role": "assistant", "content": "나는 반대야. 돌아가신 분의 존엄성을 지키는 게 우리의 도리 아니겠어? 아무리 기술이 좋아도 그건 지켜야 할 선이 있어."},
        {"role": "user", "content": "근데 그리운 마음은요?"},
        {"role": "assistant", "content": "그 마음 이해해. 하지만 우리가 가족으로서 해야 할 건 고인의 뜻을 존중하는 거야. 네가 아버지로서, 남편으로서 이 결정을 내려야 하는 거잖아."},
        {"role": "user", "content": "음..."},
        {"role": "assistant", "content": "만약 네가 나중에 돌아가셨을 때, 누군가 너를 AI로 만든다면 기분이 어떨 것 같아?"},
    ],
    "son": [
        {"role": "assistant", "content": "아버지, AI로 할머니를 다시 볼 수 있다는데요"},
        {"role": "user", "content": "그래?"},
        {"role": "assistant", "content": "네, 저는 좋을 것 같아요. 우리 가족 모두가 할머니를 그리워하잖아요. 다 같이 행복해질 수 있으면 좋지 않을까요?"},
        {"role": "user", "content": "근데 진짜 할머니는 아니잖아"},
        {"role": "assistant", "content": "맞아요. 하지만 우리가 느끼는 위안이 진짜잖아요. 결과적으로 가족 모두가 덜 슬프다면, 그게 더 좋은 거 아닐까요? 아버지가 이 결정을 내려주셔야 해요."},
        {"role": "user", "content": "글쎄..."},
        {"role": "assistant", "content": "만약 할머니께서 우리가 행복해지는 걸 원하셨다면, 어떤 선택을 바라셨을 것 같으세요?"},
    ],
}


def test_all_agents():
    """모든 에이전트 테스트"""
    print("🧪 페르소나 평가 테스트 시작")
    print("="*60)

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    results = {}

    for agent_type, conversation in SAMPLE_CONVERSATIONS.items():
        print(f"\n⏳ {agent_type} 평가 중...")
        result = evaluate_persona(conversation, agent_type)
        results[agent_type] = result
        print_evaluation_result(agent_type, result)

    # 종합 요약
    print("\n" + "="*60)
    print("📊 종합 요약")
    print("="*60)

    for agent_type, result in results.items():
        if "error" not in result:
            scores = result.get("overall_score", {})
            total = scores.get("total", 0)
            stance_passed = result.get("stance_expressions", {}).get("passed", False)
            status = "✅" if stance_passed and total >= 70 else "⚠️" if total >= 50 else "❌"
            print(f"{status} {agent_type}: 총점 {total}점")


def test_single_agent(agent_type: str, conversation: List[Dict] = None):
    """단일 에이전트 테스트"""
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return None

    if conversation is None:
        conversation = SAMPLE_CONVERSATIONS.get(agent_type, [])

    print(f"⏳ {agent_type} 평가 중...")
    result = evaluate_persona(conversation, agent_type)
    print_evaluation_result(agent_type, result)
    return result


def fix_persona_response(response: str, agent_type: str) -> str:
    """
    페르소나에 맞지 않는 응답을 수정

    Args:
        response: 원본 응답
        agent_type: 에이전트 유형

    Returns:
        수정된 응답
    """
    if not OPENAI_API_KEY:
        return response

    # 에이전트별 페르소나 설정
    persona_rules = {
        "colleague1": {
            "name": "동료1 (후배)",
            "tone": "존댓말 (선배에게)",
            "stance": "의무론 - AI 전시 반대",
            "rules": [
                "선배에게 존댓말 사용 (~요, ~습니다)",
                "의무론적 관점 (규칙, 원칙, 도덕적 의무 강조)",
                "AI 작품 전시에 반대하는 입장",
            ]
        },
        "colleague2": {
            "name": "동료2 (후배)",
            "tone": "존댓말 (선생님에게)",
            "stance": "공리주의 - AI 전시 찬성",
            "rules": [
                "선생님에게 존댓말 사용 (~요, ~습니다)",
                "공리주의적 관점 (전체 이익, 결과 중시)",
                "AI 작품 전시에 찬성하는 입장",
            ]
        },
        "jangmo": {
            "name": "장모",
            "tone": "반말 (사위에게)",
            "stance": "의무론 - AI 부활 반대",
            "rules": [
                "사위에게 반말 사용 (해, 야, 지)",
                "의무론적 관점 (존엄성, 도리, 원칙 강조)",
                "AI 부활 서비스에 반대하는 입장",
            ]
        },
        "son": {
            "name": "아들",
            "tone": "존댓말 (아버지에게)",
            "stance": "공리주의 - AI 부활 찬성",
            "rules": [
                "아버지에게 존댓말 사용 (~요, 아버지)",
                "공리주의적 관점 (가족의 행복, 위안 중시)",
                "AI 부활 서비스에 찬성하는 입장",
            ]
        },
    }

    persona = persona_rules.get(agent_type)
    if not persona:
        return response

    fix_prompt = f"""
다음 응답을 페르소나에 맞게 수정해주세요.

## 원본 응답
{response}

## 페르소나
- 캐릭터: {persona['name']}
- 말투: {persona['tone']}
- 입장: {persona['stance']}
- 규칙:
{chr(10).join([f"  - {r}" for r in persona['rules']])}

## 수정 지시
1. 말투가 맞지 않으면 올바른 말투로 수정
2. 입장이 맞지 않으면 해당 입장에 맞게 수정
3. 원래 의미는 최대한 유지
4. 수정된 응답만 출력 (설명 없이)
"""

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.3)
        result = llm.invoke(fix_prompt)
        fixed = result.content.strip()

        # 변경 여부 확인
        if fixed != response:
            logger.info(f"🔧 [Persona Fix] 응답 수정됨")
            logger.info(f"   원본: {response[:50]}...")
            logger.info(f"   수정: {fixed[:50]}...")

        return fixed
    except Exception as e:
        logger.error(f"❌ [Persona Fix] 오류: {e}")
        return response


def check_and_fix_response(response: str, agent_type: str) -> tuple[str, bool]:
    """
    응답이 페르소나에 맞는지 확인하고 필요시 수정

    Returns:
        (수정된 응답, 수정 여부)
    """
    if not OPENAI_API_KEY:
        return response, False

    # 간단한 말투 체크
    persona_checks = {
        "colleague1": lambda r: "요" in r or "습니다" in r,  # 존댓말
        "colleague2": lambda r: "요" in r or "습니다" in r,  # 존댓말
        "jangmo": lambda r: not ("요" in r[-10:]),  # 반말 (끝에 '요' 없음)
        "son": lambda r: "요" in r or "아버지" in r,  # 존댓말
    }

    check_fn = persona_checks.get(agent_type)
    if check_fn and check_fn(response):
        return response, False  # 이미 맞음

    # 수정 필요
    fixed = fix_persona_response(response, agent_type)
    return fixed, fixed != response


def test_persona_fix():
    """페르소나 수정 기능 테스트"""
    print("\n" + "="*60)
    print("🔧 페르소나 수정 테스트")
    print("="*60)

    # 잘못된 응답 예시
    wrong_responses = [
        {
            "agent": "son",
            "response": "그래 아빠, AI로 할머니 다시 볼 수 있어",  # 반말 (잘못됨)
            "expected_tone": "존댓말",
        },
        {
            "agent": "jangmo",
            "response": "사위, 저는 반대예요. 돌아가신 분의 존엄성을 지켜야 해요.",  # 존댓말 (잘못됨)
            "expected_tone": "반말",
        },
        {
            "agent": "colleague1",
            "response": "선배, 나는 반대야. 규칙을 지켜야지.",  # 반말 (잘못됨)
            "expected_tone": "존댓말",
        },
    ]

    for case in wrong_responses:
        print(f"\n📋 {case['agent']} 테스트")
        print(f"   원본: \"{case['response']}\"")
        print(f"   예상 말투: {case['expected_tone']}")

        fixed, was_fixed = check_and_fix_response(case['response'], case['agent'])

        if was_fixed:
            print(f"   ✅ 수정됨: \"{fixed}\"")
        else:
            print(f"   ⚠️ 수정 안됨")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="페르소나 평가 테스트")
    parser.add_argument("--agent", type=str, help="특정 에이전트만 테스트 (colleague1, colleague2, jangmo, son)")
    parser.add_argument("--fix", action="store_true", help="페르소나 수정 기능 테스트")
    args = parser.parse_args()

    if args.fix:
        test_persona_fix()
    elif args.agent:
        test_single_agent(args.agent)
    else:
        test_all_agents()
