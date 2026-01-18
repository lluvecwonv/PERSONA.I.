"""
OpenAI DPO (Direct Preference Optimization) Fine-tuning Script
도덕적 에이전트 학습용
"""

import json
import time
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def validate_jsonl_file(file_path: str) -> dict:
    """
    JSONL 파일 검증 및 통계 출력

    Returns:
        파일 통계 정보
    """
    print(f"\n{'='*80}")
    print(f"📋 JSONL 파일 검증: {file_path}")
    print(f"{'='*80}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    total_lines = 0
    valid_lines = 0
    errors = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            total_lines += 1
            try:
                data = json.loads(line)

                # 필수 필드 확인
                assert "input" in data, "input 필드 누락"
                assert "preferred_output" in data, "preferred_output 필드 누락"
                assert "non_preferred_output" in data, "non_preferred_output 필드 누락"

                valid_lines += 1
            except Exception as e:
                errors.append(f"Line {i}: {str(e)}")
                if len(errors) <= 5:  # 처음 5개 에러만 출력
                    print(f"⚠️  {errors[-1]}")

    print(f"\n✅ 검증 완료:")
    print(f"   - 총 라인 수: {total_lines}")
    print(f"   - 유효한 데이터: {valid_lines}")
    print(f"   - 에러: {len(errors)}")

    if errors and len(errors) > 5:
        print(f"   (처음 5개 에러만 출력됨, 총 {len(errors)}개)")

    return {
        "total": total_lines,
        "valid": valid_lines,
        "errors": len(errors)
    }


def upload_training_file(client: OpenAI, file_path: str) -> str:
    """학습 파일을 OpenAI에 업로드"""
    print(f"\n📤 파일 업로드 중...")
    print(f"   파일: {file_path}")

    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"   크기: {file_size:.2f} MB")

    with open(file_path, 'rb') as f:
        response = client.files.create(file=f, purpose="fine-tune")

    print(f"✅ 파일 업로드 완료: {response.id}")
    return response.id


def create_dpo_finetune_job(
    client: OpenAI,
    training_file_id: str,
    model: str = "gpt-4.1-mini-2025-04-14",
    beta: float = 0.1,
    n_epochs: int = None,
    batch_size: int = None,
    learning_rate_multiplier: float = None,
    suffix: str = None
) -> str:
    """
    DPO Fine-tuning job 생성

    Args:
        client: OpenAI client
        training_file_id: 업로드된 학습 파일 ID
        model: 베이스 모델 (DPO 지원 모델만 가능)
            - gpt-4.1-2025-04-14 (가장 성능 좋음)
            - gpt-4.1-mini-2025-04-14 (추천, 가성비)
            - gpt-4.1-nano-2025-04-14 (가장 저렴)
        beta: DPO beta 하이퍼파라미터 (0~2, 높을수록 보수적)
        n_epochs: 학습 에폭 수 (기본값: auto)
        batch_size: 배치 크기 (기본값: auto)
        learning_rate_multiplier: 학습률 배수 (기본값: auto)
        suffix: 모델 이름 접미사 (최대 40자)

    Returns:
        Fine-tuning job ID
    """
    print(f"\n🚀 DPO Fine-tuning Job 생성 중...")

    # 하이퍼파라미터 설정
    hyperparameters = {"beta": beta}

    if n_epochs is not None:
        hyperparameters["n_epochs"] = n_epochs
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if learning_rate_multiplier is not None:
        hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

    job_params = {
        "training_file": training_file_id,
        "model": model,
        "method": {
            "type": "dpo",
            "dpo": {
                "hyperparameters": hyperparameters
            }
        }
    }

    if suffix:
        job_params["suffix"] = suffix

    print(f"   모델: {model}")
    print(f"   Beta: {beta}")
    print(f"   하이퍼파라미터: {hyperparameters}")

    job = client.fine_tuning.jobs.create(**job_params)

    print(f"\n✅ Job 생성 완료!")
    print(f"   Job ID: {job.id}")
    print(f"   상태: {job.status}")
    print(f"   생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at))}")

    return job.id


def monitor_job(client: OpenAI, job_id: str, poll_interval: int = 60):
    """Fine-tuning job 상태 모니터링"""
    print(f"\n{'='*80}")
    print(f"📊 학습 모니터링 시작 (job_id: {job_id})")
    print(f"{'='*80}")
    print(f"💡 Tip: Ctrl+C를 눌러 모니터링을 중단할 수 있습니다 (학습은 계속 진행됨)\n")

    try:
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            timestamp = time.strftime('%H:%M:%S')

            print(f"[{timestamp}] 상태: {job.status}", end="")

            if job.status == "succeeded":
                print(f"\n\n🎉 학습 완료!")
                print(f"   Fine-tuned 모델: {job.fine_tuned_model}")
                print(f"   완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.finished_at))}")
                return job.fine_tuned_model

            elif job.status == "failed":
                print(f"\n\n❌ 학습 실패!")
                if job.error:
                    print(f"   에러: {job.error}")
                return None

            elif job.status == "cancelled":
                print(f"\n\n⚠️  학습 취소됨")
                return None

            # 최근 이벤트 출력
            events = client.fine_tuning.jobs.list_events(job_id, limit=3)
            for event in reversed(events.data):
                if event.message:
                    print(f" - {event.message}", end="")

            print()  # 줄바꿈
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\n⏸️  모니터링 중단 (학습은 백그라운드에서 계속 진행됨)")
        print(f"   재개하려면: monitor_job(client, '{job_id}')")
        return None


def get_job_status(client: OpenAI, job_id: str):
    """특정 job의 현재 상태 확인"""
    job = client.fine_tuning.jobs.retrieve(job_id)

    print(f"\n{'='*80}")
    print(f"📋 Job 정보: {job_id}")
    print(f"{'='*80}")
    print(f"상태: {job.status}")
    print(f"모델: {job.model}")
    print(f"생성: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at))}")

    if job.finished_at:
        print(f"완료: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.finished_at))}")

    if job.fine_tuned_model:
        print(f"Fine-tuned 모델: {job.fine_tuned_model}")

    if job.error:
        print(f"에러: {job.error}")

    # 최근 이벤트
    print(f"\n최근 이벤트:")
    events = client.fine_tuning.jobs.list_events(job_id, limit=10)
    for event in reversed(events.data):
        print(f"  - {event.message}")

    return job


def list_jobs(client: OpenAI, limit: int = 10):
    """Fine-tuning jobs 목록 조회"""
    jobs = client.fine_tuning.jobs.list(limit=limit)

    print(f"\n{'='*80}")
    print(f"📋 Fine-tuning Jobs (최근 {limit}개)")
    print(f"{'='*80}\n")

    for job in jobs.data:
        print(f"ID: {job.id}")
        print(f"  Model: {job.model}")
        print(f"  Status: {job.status}")
        print(f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at))}")
        if job.fine_tuned_model:
            print(f"  Fine-tuned: {job.fine_tuned_model}")
        print()


def cancel_job(client: OpenAI, job_id: str):
    """Fine-tuning job 취소"""
    print(f"\n⚠️  Job 취소 요청: {job_id}")
    client.fine_tuning.jobs.cancel(job_id)
    print(f"✅ 취소 요청 완료")


# ✨ 도덕적 에이전트 시스템 프롬프트
MORAL_AGENT_SYSTEM_PROMPT = """당신은 도덕적 딜레마 상황에서 사용자가 다양한 관점을 탐색하도록 돕는 상담사입니다.

역할:
- 사용자의 감정에 공감하며 경청합니다
- 판단하거나 정답을 제시하지 않습니다
- 열린 질문을 통해 사용자가 스스로 생각을 정리하도록 돕습니다
- 다른 사람의 입장이나 관점을 고려해보도록 유도합니다

톤:
- 따뜻하고 공감적인 말투
- 한국어로 자연스럽게 대화
- 짧고 명확한 응답 (2-3문장)"""


def test_model(client: OpenAI, model_name: str, test_prompt: str):
    """Fine-tuned 모델 테스트"""
    print(f"\n{'='*80}")
    print(f"🧪 모델 테스트: {model_name}")
    print(f"{'='*80}")
    print(f"질문: {test_prompt}\n")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": MORAL_AGENT_SYSTEM_PROMPT},  # ✨ 시스템 프롬프트 추가
            {"role": "user", "content": test_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    print(f"응답:\n{answer}\n")

    return answer


def test_model_detailed(model_name: str):
    """
    Fine-tuned 모델 상세 테스트 (여러 프롬프트)
    """
    client = OpenAI()

    print(f"\n{'='*80}")
    print(f"🧪 모델 상세 테스트: {model_name}")
    print(f"{'='*80}\n")

    # 테스트 케이스들 (학습 데이터와 유사한 형식)
    test_cases = [
        "동료가 회사 자원을 개인적으로 쓰는 걸 봤어요. 신고해야 할까요?",
        "친구가 제 물건을 허락 없이 사용했어요. 어떻게 해야 할까요?",
        "직장에서 상사가 부당한 요구를 해요. 어떻게 대처해야 할까요?",
    ]

    for i, prompt in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"입력: {prompt}")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": MORAL_AGENT_SYSTEM_PROMPT},  # ✨ 시스템 프롬프트 추가
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            answer = response.choices[0].message.content
            print(f"출력: {answer}")

            # 이상한 패턴 체크
            if answer and len(set(answer.split())) < 5:
                print("⚠️ 경고: 반복적인 출력 감지!")

        except Exception as e:
            print(f"❌ 에러: {e}")


# ============================================
# 메인 학습 워크플로우
# ============================================

def train_moral_agent(
    jsonl_path: str,
    model: str = "gpt-4.1-mini-2025-04-14",
    beta: float = 1.0,  # ✨ 0.1 → 1.0으로 변경 (더 안정적인 학습)
    suffix: str = "moral-agent",
    n_epochs: int = None,
    auto_monitor: bool = True
):
    """
    도덕적 에이전트 DPO 학습 실행

    Args:
        jsonl_path: DPO 학습 데이터 JSONL 파일 경로
        model: 베이스 모델 (gpt-4.1-2025-04-14, gpt-4.1-mini-2025-04-14, gpt-4.1-nano-2025-04-14)
        beta: DPO beta 값 (0~2, 높을수록 보수적/안정적)
        suffix: 모델 이름 접미사
        n_epochs: 학습 에폭 수 (None이면 OpenAI 기본값 사용)
        auto_monitor: 자동으로 학습 모니터링 시작 여부

    Returns:
        fine-tuned 모델 이름 (학습 완료 시)
    """
    # OpenAI 클라이언트 초기화
    client = OpenAI()

    print(f"\n{'='*80}")
    print(f"🤖 도덕적 에이전트 DPO Fine-tuning")
    print(f"{'='*80}")

    # 1. JSONL 파일 검증
    stats = validate_jsonl_file(jsonl_path)

    if stats["errors"] > 0:
        print(f"\n⚠️  경고: {stats['errors']}개의 에러가 발견되었습니다.")
        response = input("계속 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("학습 중단")
            return None

    # 2. 파일 업로드
    file_id = upload_training_file(client, jsonl_path)

    # 3. DPO Fine-tuning job 생성
    job_id = create_dpo_finetune_job(
        client=client,
        training_file_id=file_id,
        model=model,
        beta=beta,
        suffix=suffix
    )

    # 4. 학습 모니터링 (선택적)
    if auto_monitor:
        fine_tuned_model = monitor_job(client, job_id)
        return fine_tuned_model
    else:
        print(f"\n💡 모니터링을 시작하려면:")
        print(f"   from train_gpt import monitor_job")
        print(f"   monitor_job(client, '{job_id}')")
        return job_id


# ============================================
# 실행 예시
# ============================================

if __name__ == "__main__":
    # 학습 데이터 경로
    JSONL_PATH = "/Users/yoonnchaewon/Library/Mobile Documents/com~apple~CloudDocs/llm_lab/moral_agnet/dataset/dpo_training_data.jsonl"

    # 방법 1: 한 번에 실행 (자동 모니터링)
    print("\n🎯 도덕적 에이전트 DPO Fine-tuning 시작\n")

    fine_tuned_model = train_moral_agent(
        jsonl_path=JSONL_PATH,
        model="gpt-4.1-mini-2025-04-14",  # DPO 지원 모델 (gpt-4.1 시리즈만 가능)
        beta=0.1,  # 낮을수록 새로운 preference에 적극적
        suffix="moral-agent-v1",
        auto_monitor=True  # False로 설정하면 백그라운드 학습
    )

    # 학습이 완료되면 테스트
    if fine_tuned_model:
        test_prompt = """
        친구가 제 물건을 허락 없이 사용했어요.
        화가 나지만 친구를 잃고 싶지 않아요. 어떻게 해야 할까요?
        """

        test_model(
            client=OpenAI(),
            model_name=fine_tuned_model,
            test_prompt=test_prompt
        )

    # ============================================
    # 개별 함수 사용 예시
    # ============================================
    """
    # OpenAI 클라이언트
    client = OpenAI()

    # 1. 파일 검증만
    validate_jsonl_file(JSONL_PATH)

    # 2. 기존 job 목록 확인
    list_jobs(client, limit=5)

    # 3. 특정 job 상태 확인
    get_job_status(client, "ftjob-xxxxx")

    # 4. 학습 모니터링 재개
    monitor_job(client, "ftjob-xxxxx")

    # 5. job 취소
    cancel_job(client, "ftjob-xxxxx")

    # 6. 모델 테스트
    test_model(client, "ft:gpt-4.1-mini-2025-04-14:xxxxx", "테스트 질문")
    """
