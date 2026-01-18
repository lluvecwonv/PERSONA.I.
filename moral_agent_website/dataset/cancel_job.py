"""
학습 Job 취소 스크립트
"""

from train_gpt import cancel_job, get_job_status
from openai import OpenAI
import sys

if __name__ == "__main__":
    # Job ID를 명령줄 인자로 받거나, 파일에서 읽기
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        # latest_job_id.txt에서 읽기
        from pathlib import Path
        jsonl_path = "/Users/yoonnchaewon/Library/Mobile Documents/com~apple~CloudDocs/llm_lab/moral_agnet/dataset/dpo_training_data.jsonl"
        job_info_path = Path(jsonl_path).parent / "latest_job_id.txt"
        
        if not job_info_path.exists():
            print("❌ Job ID 파일을 찾을 수 없습니다.")
            print("💡 사용법: python3 cancel_job.py [job_id]")
            sys.exit(1)
        
        with open(job_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("Job ID:"):
                    job_id = line.split(":", 1)[1].strip()
                    break
            else:
                print("❌ Job ID를 찾을 수 없습니다.")
                sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"⚠️  학습 Job 취소")
    print(f"{'='*80}")
    
    # 현재 상태 확인
    client = OpenAI()
    job = get_job_status(client, job_id)
    
    if job.status in ["succeeded", "failed", "cancelled"]:
        print(f"\n⚠️  Job이 이미 {job.status} 상태입니다.")
        sys.exit(0)
    
    # 확인
    response = input(f"\n정말로 Job '{job_id}'를 취소하시겠습니까? (yes/no): ")
    if response.lower() != 'yes':
        print("취소되었습니다.")
        sys.exit(0)
    
    # Job 취소
    cancel_job(client, job_id)
    
    # 상태 확인
    print("\n" + "="*80)
    print("최종 상태 확인:")
    get_job_status(client, job_id)

