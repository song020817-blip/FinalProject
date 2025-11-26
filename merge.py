import pandas as pd
import chardet

# ✅ 파일 목록 및 연도 매핑
files = {
    "아파트전월세_실거래가_2021.csv": 2021,
    "아파트전월세_실거래가_2022.csv": 2022,
    "아파트전월세_실거래가_2023.csv": 2023,
    "아파트전월세_실거래가_2024.csv": 2024,
    "아파트전월세_실거래가_2025.csv": 2025
}

merged = pd.DataFrame()

for file, year in files.items():
    print(f"\n📂 {file} 불러오는 중...")

    # 🔹 인코딩 감지
    with open(file, 'rb') as f:
        enc = chardet.detect(f.read(10000))['encoding']
    print(f"  → 감지된 인코딩: {enc}")

    # 🔹 CSV 읽기 (기본 쉼표 구분)
    try:
        df = pd.read_csv(file, encoding=enc)
    except Exception as e:
        print(f"❌ {file} 읽기 실패: {e}")
        continue

    # 🔹 연도 구분 열 추가
    df['데이터연도'] = year

    merged = pd.concat([merged, df], ignore_index=True)

# 🔹 최종 저장
output_file = "아파트전월세_실거래가_2021_2025_병합.csv"
merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 병합 완료! 총 {len(merged):,}건 저장됨")
print("출력 파일:", output_file)
