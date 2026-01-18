"""
병합된 clean 데이터를 DPO 형식의 JSONL로 변환
"""

import json
import pandas as pd
from pathlib import Path


def convert_excel_to_dpo_jsonl(excel_path: str, output_path: str = None):
    """
    Excel 파일을 DPO 형식의 JSONL로 변환

    Args:
        excel_path: 입력 Excel 파일 경로
        output_path: 출력 JSONL 파일 경로

    Returns:
        변환된 데이터 개수
    """
    # Excel 파일 읽기
    print(f"📖 파일 읽는 중: {excel_path}")
    df = pd.read_excel(excel_path)

    print(f"   - 총 행 수: {len(df)}")
    print(f"   - 컬럼: {list(df.columns)}\n")

    # 필요한 컬럼 확인
    required_columns = ['user', 'agent_preferred', 'agent_rejected']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"❌ 필수 컬럼 누락: {missing_columns}")
        return 0

    # 결측치 제거
    df_clean = df.dropna(subset=required_columns)
    print(f"✅ 유효한 데이터: {len(df_clean)}행 (결측치 제거 후)")

    # 출력 파일 경로 설정
    if output_path is None:
        output_path = Path(excel_path).parent / "dpo_training_data.jsonl"

    # DPO 형식으로 변환
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df_clean.iterrows():
            # 빈 문자열이나 공백만 있는 경우 스킵
            if not str(row['user']).strip() or \
               not str(row['agent_preferred']).strip() or \
               not str(row['agent_rejected']).strip():
                continue

            dpo_example = {
                "input": {
                    "messages": [
                        {"role": "user", "content": str(row['user']).strip()}
                    ],
                    "tools": [],
                    "parallel_tool_calls": True
                },
                "preferred_output": [
                    {"role": "assistant", "content": str(row['agent_preferred']).strip()}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": str(row['agent_rejected']).strip()}
                ]
            }

            f.write(json.dumps(dpo_example, ensure_ascii=False) + '\n')
            count += 1

    print(f"\n✨ 변환 완료!")
    print(f"   - 변환된 데이터: {count}개")
    print(f"   - 저장 위치: {output_path}")

    return count


def preview_jsonl(jsonl_path: str, n: int = 3):
    """JSONL 파일 미리보기"""
    print(f"\n{'='*80}")
    print(f"📄 JSONL 파일 미리보기: {jsonl_path}")
    print(f"{'='*80}\n")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break

            data = json.loads(line)
            print(f"--- 예시 {i+1} ---")
            print(f"🙋 User: {data['input']['messages'][0]['content'][:100]}...")
            print(f"✅ Preferred: {data['preferred_output'][0]['content'][:100]}...")
            print(f"❌ Rejected: {data['non_preferred_output'][0]['content'][:100]}...")
            print()


if __name__ == "__main__":
    excel_path = "/Users/yoonnchaewon/Library/Mobile Documents/com~apple~CloudDocs/llm_lab/moral_agnet/dataset/merged_clean_data.xlsx"
    output_path = "/Users/yoonnchaewon/Library/Mobile Documents/com~apple~CloudDocs/llm_lab/moral_agnet/dataset/dpo_training_data.jsonl"

    # 변환 실행
    count = convert_excel_to_dpo_jsonl(excel_path, output_path)

    # 미리보기
    if count > 0:
        preview_jsonl(output_path, n=3)
