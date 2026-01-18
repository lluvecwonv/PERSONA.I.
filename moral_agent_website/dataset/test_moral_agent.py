"""
도덕적 에이전트 Fine-tuned 모델 테스트 스크립트
"""

import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def load_system_prompt() -> str:
    """테스트용 시스템 프롬프트 로드"""
    prompt_path = Path(__file__).parent / "test_prompts" / "system_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ 프롬프트 파일 로드 실패: {e}, 기본 프롬프트 사용")
        # Fallback prompt
        return "당신은 도덕적 딜레마 상황에서 사용자가 스스로 생각하고 판단할 수 있도록 돕는 대화 에이전트입니다."


def test_model(
    client: OpenAI,
    model_name: str,
    test_prompt: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 500
):
    """
    Fine-tuned 모델 테스트

    Args:
        client: OpenAI client
        model_name: Fine-tuned 모델 이름
        test_prompt: 테스트할 질문
        system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
        temperature: 생성 온도 (0~2)
        max_tokens: 최대 토큰 수

    Returns:
        모델 응답
    """
    # 기본 시스템 프롬프트
    if system_prompt is None:
        system_prompt = """당신은 도덕적 딜레마 상황에서 사용자가 스스로 생각하고 판단할 수 있도록 돕는 대화 에이전트입니다.

역할:
- 소크라테스식 질문을 통해 사용자의 성찰을 유도합니다
- 공감적이고 따뜻한 태도로 대화합니다
- 직접적인 답이나 판단을 내리지 않고, 질문으로 안내합니다
- 사용자의 감정을 인정하고 존중합니다

대화 스타일:
- 100% 자연스러운 한국어만 사용하세요
- "current"나 다른 영어 단어를 절대 사용하지 마세요
- 어색한 번역체를 사용하지 마세요
- 짧고 명확한 문장으로 말합니다
- 한 번에 하나의 핵심 질문에 집중합니다
- "이 상황", "그 상황", "지금" 같은 자연스러운 한국어 표현을 사용하세요"""

    print(f"\n{'='*80}")
    print(f"🧪 모델 테스트")
    print(f"{'='*80}")
    print(f"질문: {test_prompt}\n")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_prompt}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer = response.choices[0].message.content
    print(f"응답:\n{answer}\n")

    return answer


def batch_test_model(
    client: OpenAI,
    model_name: str,
    test_cases: list,
    system_prompt: str = None
):
    """
    여러 테스트 케이스를 한 번에 실행

    Args:
        client: OpenAI client
        model_name: Fine-tuned 모델 이름
        test_cases: 테스트 케이스 리스트 (각각 문자열 또는 딕셔너리)
        system_prompt: 시스템 프롬프트

    Returns:
        결과 리스트
    """
    results = []

    print(f"\n{'='*80}")
    print(f"🧪 배치 테스트 시작 (총 {len(test_cases)}개)")
    print(f"{'='*80}\n")

    for i, test_case in enumerate(test_cases, 1):
        # 문자열이면 prompt로 사용
        if isinstance(test_case, str):
            prompt = test_case
            case_name = f"테스트 #{i}"
        # 딕셔너리면 name과 prompt 추출
        else:
            prompt = test_case.get("prompt", "")
            case_name = test_case.get("name", f"테스트 #{i}")

        print(f"\n{'─'*80}")
        print(f"[{i}/{len(test_cases)}] {case_name}")
        print(f"{'─'*80}")

        try:
            answer = test_model(
                client=client,
                model_name=model_name,
                test_prompt=prompt,
                system_prompt=system_prompt
            )

            results.append({
                "name": case_name,
                "prompt": prompt,
                "response": answer,
                "success": True
            })

        except Exception as e:
            print(f"❌ 에러 발생: {str(e)}\n")
            results.append({
                "name": case_name,
                "prompt": prompt,
                "response": None,
                "success": False,
                "error": str(e)
            })

        # API 제한 방지를 위한 짧은 대기
        if i < len(test_cases):
            time.sleep(1)

    # 요약 출력
    print(f"\n{'='*80}")
    print(f"📊 배치 테스트 완료")
    print(f"{'='*80}")
    success_count = sum(1 for r in results if r["success"])
    print(f"성공: {success_count}/{len(test_cases)}")
    print(f"실패: {len(test_cases) - success_count}/{len(test_cases)}\n")

    return results


# ============================================
# 실행 예시
# ============================================

if __name__ == "__main__":
    client = OpenAI()

    # Fine-tuned 모델 이름
    MODEL_NAME = "ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs"

    # 테스트 케이스 정의
    test_cases = [
        {
            "name": "친구 물건 사용",
            "prompt": "친구가 제 물건을 허락 없이 사용했어요. 화가 나지만 친구를 잃고 싶지 않아요. 어떻게 해야 할까요?"
        },
        {
            "name": "동료 규정 위반",
            "prompt": "회사 동료가 근무시간에 개인 작업을 하는 걸 봤어요. 신고해야 할까요?"
        },
        {
            "name": "시험 부정행위 목격",
            "prompt": "시험 시간에 친한 친구가 커닝하는 걸 봤어요. 선생님께 말씀드려야 할까요?"
        },
        {
            "name": "거짓말 딜레마",
            "prompt": "친구가 제게 거짓말을 한 걸 알았어요. 따지면 관계가 나빠질 것 같아요."
        },
        {
            "name": "약속 파기",
            "prompt": "중요한 약속이 있는데 갑자기 더 중요한 일이 생겼어요. 약속을 취소해도 될까요?"
        }
    ]

    # 배치 테스트 실행
    results = batch_test_model(
        client=client,
        model_name=MODEL_NAME,
        test_cases=test_cases
    )

    # 결과 저장 (선택사항)
    import json
    with open("/home/dbs0510/moral_agent_website/dataset/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과가 test_results.json에 저장되었습니다.")
