"""
Fine-tuned 모델 접근 가능 여부 확인
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경 변수 로드
load_dotenv(dotenv_path="/Users/yoonnchaewon/Desktop/moral_agent/moral_agent_website/.env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found")
    exit(1)

client = OpenAI(api_key=api_key)

print("=" * 70)
print("OpenAI 모델 접근 확인")
print("=" * 70)

target_model = "ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs"

print(f"\n[Target Model]")
print(f"{target_model}")

# 1. Fine-tuned 모델 목록 확인
print("\n" + "=" * 70)
print("1. Fine-tuned 모델 목록 조회")
print("=" * 70)

try:
    models = client.models.list()
    fine_tuned_models = [m for m in models.data if m.id.startswith("ft:")]

    if fine_tuned_models:
        print(f"\n✅ Fine-tuned 모델 {len(fine_tuned_models)}개 발견:\n")
        for model in fine_tuned_models:
            print(f"  - {model.id}")
            if model.id == target_model:
                print(f"    ✅ TARGET MODEL FOUND!")
    else:
        print("\n⚠️ Fine-tuned 모델이 없습니다.")

except Exception as e:
    print(f"\n❌ 모델 목록 조회 실패: {e}")

# 2. 특정 모델 정보 조회
print("\n" + "=" * 70)
print("2. Target 모델 직접 조회")
print("=" * 70)

try:
    model_info = client.models.retrieve(target_model)
    print(f"\n✅ 모델 정보:")
    print(f"  - ID: {model_info.id}")
    print(f"  - Created: {model_info.created}")
    print(f"  - Owned by: {model_info.owned_by}")
except Exception as e:
    print(f"\n❌ 모델 조회 실패: {e}")
    print("\n가능한 원인:")
    print("  1. 모델 ID가 잘못되었습니다")
    print("  2. API 키에 이 모델에 대한 접근 권한이 없습니다")
    print("  3. 모델이 삭제되었거나 만료되었습니다")

# 3. 간단한 API 호출 테스트
print("\n" + "=" * 70)
print("3. 실제 API 호출 테스트")
print("=" * 70)

try:
    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "user", "content": "안녕"}
        ],
        max_tokens=10
    )
    print(f"\n✅ API 호출 성공!")
    print(f"응답: {response.choices[0].message.content}")
except Exception as e:
    print(f"\n❌ API 호출 실패: {e}")

print("\n" + "=" * 70)
print("권장 사항")
print("=" * 70)
print("""
1. Fine-tuned 모델이 없으면:
   - OpenAI Platform에서 fine-tuning job 확인
   - 올바른 API 키 사용 여부 확인

2. 모델 ID가 다르면:
   - 위에서 출력된 fine-tuned 모델 ID로 변경

3. 접근 권한이 없으면:
   - 임시로 gpt-4o 또는 gpt-4o-mini 사용 권장
""")
