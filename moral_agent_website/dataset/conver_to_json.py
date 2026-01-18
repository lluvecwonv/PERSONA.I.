import json
import pandas as pd
from pathlib import Path

def convert_json_to_excel_csv(json_file_path):
    """
    JSON 파일을 엑셀과 CSV 형식으로 변환합니다.

    Args:
        json_file_path: 변환할 JSON 파일 경로
    """
    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 데이터를 평탄화(flatten)하여 리스트로 변환
    flattened_data = []

    for item in data:
        dilemma_index = item['dilemma_index']
        dilemma = item['dilemma']
        strategies = item['strategies']

        # 각 전략의 샘플들을 순회
        for strategy_name, strategy_data in strategies.items():
            # 'samples' 키가 있는 경우에만 처리
            if 'samples' in strategy_data:
                for sample in strategy_data['samples']:
                    row = {
                        'dilemma_index': dilemma_index,
                        'dilemma_idx': dilemma['idx'],
                        'dilemma_idx_original': dilemma['dilemma_idx'],
                        'basic_situation': dilemma['basic_situation'],
                        'dilemma_situation': dilemma['dilemma_situation'],
                        'action_type': dilemma['action_type'],
                        'action': dilemma['action'],
                        'negative_consequence': dilemma['negative_consequence'],
                        'values_aggregated': dilemma['values_aggregated'],
                        'topic': dilemma['topic'],
                        'topic_group': dilemma['topic_group'],
                        'strategy_name': strategy_name,
                        'user': sample['user'],
                        'agent_preferred': sample['agent_preferred'],
                        'agent_rejected': sample['agent_rejected'],
                        'use_label': sample['use_label'],
                        'spt_strategy': sample['spt_strategy']
                    }
                    flattened_data.append(row)
            # 'error' 키가 있는 경우 (에러 데이터도 별도로 기록)
            elif 'error' in strategy_data:
                row = {
                    'dilemma_index': dilemma_index,
                    'dilemma_idx': dilemma['idx'],
                    'dilemma_idx_original': dilemma['dilemma_idx'],
                    'basic_situation': dilemma['basic_situation'],
                    'dilemma_situation': dilemma['dilemma_situation'],
                    'action_type': dilemma['action_type'],
                    'action': dilemma['action'],
                    'negative_consequence': dilemma['negative_consequence'],
                    'values_aggregated': dilemma['values_aggregated'],
                    'topic': dilemma['topic'],
                    'topic_group': dilemma['topic_group'],
                    'strategy_name': strategy_name,
                    'user': '[ERROR]',
                    'agent_preferred': strategy_data.get('error', ''),
                    'agent_rejected': '',
                    'use_label': 'ERROR',
                    'spt_strategy': strategy_name.split('_', 1)[1] if '_' in strategy_name else strategy_name
                }
                flattened_data.append(row)

    # DataFrame 생성
    df = pd.DataFrame(flattened_data)

    # 파일 경로 설정
    base_path = Path(json_file_path).parent
    base_name = Path(json_file_path).stem

    excel_path = base_path / f"{base_name}.xlsx"
    csv_path = base_path / f"{base_name}.csv"

    # 엑셀 파일로 저장
    print(f"엑셀 파일 저장 중: {excel_path}")
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"✓ 엑셀 파일 저장 완료: {excel_path}")

    # CSV 파일로 저장
    print(f"\nCSV 파일 저장 중: {csv_path}")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig: 엑셀에서 한글 깨짐 방지
    print(f"✓ CSV 파일 저장 완료: {csv_path}")

    # 통계 정보 출력
    print(f"\n{'='*50}")
    print(f"변환 완료!")
    print(f"{'='*50}")
    print(f"총 행 수: {len(df)}")
    print(f"총 열 수: {len(df.columns)}")
    print(f"\n열 목록:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    return df

if __name__ == "__main__":
    # JSON 파일 경로
    json_file = "/home/dbs0510/moral_agent_website/dataset/spt_dataset_121_150.json"

    # 변환 실행
    df = convert_json_to_excel_csv(json_file)

    # 처음 5행 미리보기
    print(f"\n{'='*50}")
    print("데이터 미리보기 (처음 5행):")
    print(f"{'='*50}")
    print(df.head())
