"""
Excel 파일에서 'clean' 시트들을 하나로 합치는 스크립트
"""

import pandas as pd
from pathlib import Path


def merge_clean_sheets(excel_path: str, output_path: str = None):
    """
    Excel 파일에서 'clean'이라는 이름이 포함된 시트들을 찾아서 하나로 합침

    Args:
        excel_path: Excel 파일 경로
        output_path: 결과 저장 경로 (None이면 merged_clean_data.xlsx로 저장)

    Returns:
        합쳐진 DataFrame
    """
    # Excel 파일 읽기
    excel_file = pd.ExcelFile(excel_path)

    print(f"📁 파일: {excel_path}")
    print(f"📊 전체 시트: {excel_file.sheet_names}\n")

    # 'clean'이 포함된 시트 찾기
    clean_sheets = [sheet for sheet in excel_file.sheet_names if 'clean' in sheet.lower()]

    if not clean_sheets:
        print("❌ 'clean'이 포함된 시트를 찾을 수 없습니다.")
        return None

    print(f"✅ 'clean' 시트 발견: {clean_sheets}\n")

    # 각 시트 읽어서 합치기
    all_data = []

    for sheet_name in clean_sheets:
        print(f"📖 읽는 중: {sheet_name}")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # 시트 이름을 컬럼으로 추가 (어느 시트에서 왔는지 추적)
        df['source_sheet'] = sheet_name

        all_data.append(df)
        print(f"   - 행 수: {len(df)}, 열 수: {len(df.columns)}")

    # 데이터 합치기
    merged_df = pd.concat(all_data, ignore_index=True)

    print(f"\n✨ 병합 완료!")
    print(f"   - 총 행 수: {len(merged_df)}")
    print(f"   - 총 열 수: {len(merged_df.columns)}")
    print(f"   - 컬럼: {list(merged_df.columns)}")

    # 파일 저장
    if output_path is None:
        output_path = Path(excel_path).parent / "merged_clean_data.xlsx"

    merged_df.to_excel(output_path, index=False)
    print(f"\n💾 저장 완료: {output_path}")

    return merged_df


if __name__ == "__main__":
    excel_path = "/Users/yoonnchaewon/Library/Mobile Documents/com~apple~CloudDocs/llm_lab/moral_agnet/dataset/Dataset 검토 (1114).xlsx"

    # clean 시트들 병합
    df = merge_clean_sheets(excel_path)

    # 데이터 미리보기
    if df is not None:
        print("\n" + "="*80)
        print("📊 데이터 미리보기 (처음 5행):")
        print("="*80)
        print(df.head())

        print("\n" + "="*80)
        print("📊 데이터 정보:")
        print("="*80)
        print(df.info())
