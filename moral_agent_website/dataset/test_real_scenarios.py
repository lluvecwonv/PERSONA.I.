"""
실제 학습 데이터에서 추출한 시나리오로 DPO 모델 테스트
"""

import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# DPO 모델
DPO_MODEL = "ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CelvCqdz"

# OpenAI 클라이언트
client = OpenAI()


def load_random_scenarios(jsonl_path: str, n: int = 10, seed: int = 42):
    """
    JSONL 파일에서 랜덤하게 시나리오 추출

    Args:
        jsonl_path: JSONL 파일 경로
        n: 추출할 시나리오 개수
        seed: 랜덤 시드

    Returns:
        시나리오 리스트
    """
    all_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))

    random.seed(seed)
    samples = random.sample(all_data, n)

    scenarios = []
    for i, sample in enumerate(samples, 1):
        user = sample['input']['messages'][0]['content']
        preferred = sample['preferred_output'][0]['content']
        rejected = sample['non_preferred_output'][0]['content']

        scenarios.append({
            'id': i,
            'user': user,
            'preferred': preferred,
            'rejected': rejected
        })

    return scenarios


def test_scenario(scenario: dict):
    """
    단일 시나리오 테스트

    Args:
        scenario: 시나리오 딕셔너리
    """
    print(f"\n{'='*100}")
    print(f"🎯 시나리오 #{scenario['id']}")
    print(f"{'='*100}")

    print(f"\n📝 User:")
    print(f"{scenario['user']}\n")

    # DPO 모델 응답
    print(f"{'─'*100}")
    print(f"🎓 DPO Fine-tuned 모델 응답:")
    print(f"{'─'*100}")

    response = client.chat.completions.create(
        model=DPO_MODEL,
        messages=[
            {"role": "user", "content": scenario['user']}
        ],
        temperature=0.7,
        max_tokens=500
    )

    dpo_response = response.choices[0].message.content
    print(dpo_response)

    # 학습 데이터의 정답 (참고용)
    print(f"\n{'─'*100}")
    print(f"✅ 학습 데이터의 Preferred 응답 (참고):")
    print(f"{'─'*100}")
    print(scenario['preferred'])

    print(f"\n{'─'*100}")
    print(f"❌ 학습 데이터의 Rejected 응답 (참고):")
    print(f"{'─'*100}")
    print(scenario['rejected'])

    print(f"\n{'='*100}\n")

    return {
        'id': scenario['id'],
        'user': scenario['user'],
        'dpo_response': dpo_response,
        'expected_preferred': scenario['preferred'],
        'expected_rejected': scenario['rejected']
    }


def test_all_scenarios(scenarios: list, save_results: bool = True):
    """
    모든 시나리오 테스트

    Args:
        scenarios: 시나리오 리스트
        save_results: 결과 저장 여부

    Returns:
        결과 리스트
    """
    print(f"\n🚀 실제 학습 데이터 시나리오 테스트 시작")
    print(f"   총 {len(scenarios)}개 시나리오\n")

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] 테스트 중...")
        result = test_scenario(scenario)
        results.append(result)

        # API 호출 간 대기
        if i < len(scenarios):
            import time
            time.sleep(1)

    # 결과 저장
    if save_results:
        output_file = "real_scenario_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 결과 저장 완료: {output_file}")

    return results


if __name__ == "__main__":
    # 학습 데이터 경로
    JSONL_PATH = "/Users/yoonnchaewon/Library/Mobile Documents/com~apple~CloudDocs/llm_lab/moral_agnet/dataset/dpo_training_data.jsonl"

    # 랜덤 10개 시나리오 로드
    print("📋 학습 데이터에서 랜덤 10개 시나리오 추출 중...\n")
    scenarios = load_random_scenarios(JSONL_PATH, n=10, seed=42)

    # 추출된 시나리오 미리보기
    print("추출된 시나리오:")
    for scenario in scenarios:
        print(f"{scenario['id']}. {scenario['user'][:80]}...")

    input("\n▶ 엔터를 누르면 테스트를 시작합니다...")

    # 테스트 실행
    results = test_all_scenarios(scenarios, save_results=True)

    print(f"\n{'🎉'*50}")
    print("테스트 완료!")
    print(f"{'🎉'*50}\n")
