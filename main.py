# ============================================================
# 📦 1. 기본 환경 설정 및 라이브러리 불러오기 (PyCharm 버전)
# ============================================================

import pandas as pd
import numpy as np
import requests
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgbm
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')  # 윈도우 기본 한글 폰트


# ============================================================
# 📂 2. 로컬 파일 경로 설정 (중요!)
# ============================================================

BASE_PATH = r"C:\Users\james\PycharmProjects\finalproject\\"

FILE_NAME = BASE_PATH + "merged_apartment_with_coords.csv"
UNITS_FILE_NAME = BASE_PATH + "complex_units.csv"
RAW_KB_FILE_NAME = "weekly_apartment_jeonse_index_20251114.xlsx"
RAW_EXCEL_PATH = BASE_PATH + RAW_KB_FILE_NAME
OUTPUT_CSV_PATH = BASE_PATH + "proxy_data.csv"


# ============================================================
# 🔑 3. 카카오 API 키
# ============================================================

KAKAO_API_KEY = "c6943568281ead90d30d6c07d618eb7d"

def get_coords(address):
    url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    response = requests.get(url, headers=headers).json()
    if not response.get('documents'):
        return None, None
    lon = response['documents'][0]['x']
    lat = response['documents'][0]['y']
    return float(lon), float(lat)


# ============================================================
# 📘 4. 데이터 로드 (로컬 CSV)
# ============================================================

print("📂 merged_apartment_with_coords.csv 불러오는 중...")
df = pd.read_csv(FILE_NAME)
print("  → 로드 완료")

try:
    df_units = pd.read_csv(UNITS_FILE_NAME)
    df = pd.merge(df, df_units, on=['시군구', '단지명'], how='left')
except:
    print("⚠ complex_units.csv 파일이 없어 총세대수=0으로 처리됨")
    df['총세대수'] = 0

print(f"데이터 크기: {df.shape}")


# ============================================================
# 📦 5. proxy_data 생성 (엑셀 → CSV)
# ============================================================

print("📦 proxy_data.csv 생성 시작...")

df_raw = pd.read_excel(RAW_EXCEL_PATH, header=0)

if '지역명' in df_raw.columns:
    df_raw = df_raw.rename(columns={'지역명': '지역'})

df_seoul = df_raw[df_raw['지역'] == '서울']

df_seoul_tall = pd.melt(
    df_seoul,
    id_vars=['지역'],
    var_name='주차',
    value_name='주간변동률'
)

df_seoul_tall['주차'] = pd.to_datetime(df_seoul_tall['주차'])
df_seoul_tall['주간변동률'] = pd.to_numeric(df_seoul_tall['주간변동률'], errors='ignore')

df_proxy = df_seoul_tall[['주차', '주간변동률']].sort_values('주차')

df_proxy.to_csv(OUTPUT_CSV_PATH, index=False)
print("  → proxy_data.csv 생성 완료")


# ============================================================
# 🧹 6. 본격적 데이터 전처리
# ============================================================

def get_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


STATION_COORDS = {
    '건대입구역': (37.540458, 127.069320), '강변역': (37.535102, 127.094761),
    '구의역': (37.537190, 127.086164), '군자역': (37.557200, 127.079546),
    '아차산역': (37.551944, 127.089722), '광나루역': (37.545291, 127.103485),
    '자양역': (37.531667, 127.066667), '어린이대공원역': (37.547778, 127.074444),
    '중곡역': (37.565833, 127.084167), '성수역': (37.544583, 127.055972),
    '뚝섬역': (37.547222, 127.047306), '한양대역': (37.555806, 127.043667),
    '왕십리역': (37.561194, 127.037444), '상왕십리역': (37.564260, 127.029230),
    '용마산역': (37.573611, 127.086667), '사가정역': (37.580833, 127.088333),
    '면목역': (37.588611, 127.087500),
}

KONKUK_UNIV = (37.5408, 127.0794)

def get_nearest_station_info(lat, lon):
    best_dist = np.inf
    best_name = None
    for name, (slat, slon) in STATION_COORDS.items():
        d = get_haversine_distance(lat, lon, slat, slon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_dist, best_name


df_jeonse = df[df['전월세구분'] == '전세'].copy()

if '계약구분' in df_jeonse.columns:
    df_jeonse = df_jeonse[df_jeonse['계약구분'] == '신규']


target = '보증금(만원)'
features = [
    '전용면적(㎡)', '층', '건축년도', '시군구', '계약년월',
    '단지명', '총세대수', '위도', '경도'
]

df_model = df_jeonse[features + [target]].dropna()

# 날짜 생성
s = df_model['계약년월'].astype(str)
dt_series = pd.to_datetime(s, format='%Y%m')

days_in_month = dt_series.dt.days_in_month
np.random.seed(42)
random_days = [np.random.randint(1, d + 1) for d in days_in_month]

df_model['계약일_dt'] = dt_series + pd.to_timedelta(np.array(random_days)-1, unit='D')

# 연식 계산
df_model['건축년도_int'] = pd.to_numeric(df_model['건축년도'])
df_model['아파트연식'] = (df_model['계약년월']//100) - df_model['건축년도_int']

# 브랜드 아파트 여부
brand_list = ['자이','래미안','푸르지오','힐스테이트','아이파크','e편한세상','더샵','롯데캐슬','SKVIEW','위브','아이원']
brand_pattern = '|'.join(brand_list)
df_model['is_brand'] = df_model['단지명'].str.contains(brand_pattern).astype(int)

# 기준금리 맵
rate_map = {
    202101: 0.50, 202102: 0.50, 202103: 0.50, 202104: 0.50, 202105: 0.50, 202106: 0.50,
    202107: 0.50, 202108: 0.75, 202109: 0.75, 202110: 0.75, 202111: 1.00, 202112: 1.00,
    202201: 1.25, 202202: 1.25, 202203: 1.25, 202204: 1.50, 202205: 1.75, 202206: 1.75,
    202207: 2.25, 202208: 2.50, 202209: 2.50, 202210: 3.00, 202211: 3.25, 202212: 3.25,
    202301: 3.50, 202302: 3.50, 202303: 3.50, 202304: 3.50, 202305: 3.50, 202306: 3.50,
    202307: 3.50, 202308: 3.50, 202309: 3.50, 202310: 3.50, 202311: 3.50, 202312: 3.50,
    202401: 3.50, 202402: 3.50, 202403: 3.50, 202404: 3.50, 202405: 3.50, 202406: 3.50,
    202407: 3.50, 202408: 3.50, 202409: 3.50, 202410: 3.25, 202411: 3.00, 202412: 3.00,
    202501: 3.00, 202502: 2.75, 202503: 2.75, 202504: 2.75, 202505: 2.50, 202506: 2.50,
    202507: 2.50, 202508: 2.50, 202509: 2.50, 202510: 2.50, 202511: 2.50, 202512: 2.50
}
# (여기 학습 코드에서는 실제 rate_map 그대로 넣어야 함)

df_model['기준금리'] = df_model['계약년월'].map(rate_map)

df_model['총세대수'] = df_model['총세대수'].fillna(0)

# 역 거리 계산
df_model['역까지거리(km)'] = df_model.apply(
    lambda r: get_nearest_station_info(r['위도'], r['경도'])[0], axis=1
)
df_model['학교까지거리(km)'] = df_model.apply(
    lambda r: get_haversine_distance(KONKUK_UNIV[0], KONKUK_UNIV[1], r['위도'], r['경도']), axis=1
)

# 프록시 데이터 병합
df_proxy = pd.read_csv(OUTPUT_CSV_PATH)
df_proxy['주차'] = pd.to_datetime(df_proxy['주차'])
df_proxy = df_proxy.sort_values('주차')

df_model = pd.merge_asof(
    df_model.sort_values('계약일_dt'),
    df_proxy,
    left_on='계약일_dt',
    right_on='주차',
    direction='backward'
)

df_model['주간변동률'] = df_model['주간변동률'].fillna(0)

final_features = [
    '전용면적(㎡)', '층', '시군구', '계약년월', '아파트연식',
    'is_brand', '기준금리', '총세대수', '위도', '경도',
    '역까지거리(km)', '학교까지거리(km)', '주간변동률'
]

df_model = df_model[final_features + [target]]

df_model = pd.get_dummies(df_model, columns=['시군구'], drop_first=True)

X = df_model.drop(target, axis=1)
y = df_model[target]


# ============================================================
# 🧪 7. 데이터 분할
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)


# ============================================================
# 🚀 8. 모델 학습 (XGBoost + LightGBM)
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=2000, learning_rate=0.02, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    n_jobs=-1, early_stopping_rounds=50
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

lgbm_model = LGBMRegressor(
    n_estimators=2000, learning_rate=0.02, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)],
    eval_metric='rmse'
)


# ============================================================
# 🏠 9. 예측 (1개 샘플)
# ============================================================

new_house = pd.DataFrame(columns=X_train.columns)
new_house.loc[0] = 0

"""# 예측 입력
input_area = 27
input_floor = 6
input_year_built = 2005
input_contract_date_str = "2025-01-20"
input_sigungu = "서울특별시 광진구 화양동"
input_total_units = 312
input_is_brand = 0
input_address = "서울특별시 광진구 광나루로 410"

contract_dt = pd.to_datetime(input_contract_date_str)
contract_ym = int(contract_dt.strftime('%Y%m'))

new_house['전용면적(㎡)'] = input_area
new_house['층'] = input_floor
new_house['계약년월'] = contract_ym
new_house['아파트연식'] = (contract_ym//100) - input_year_built
new_house['기준금리'] = 3.5
new_house['총세대수'] = input_total_units
new_house['is_brand'] = input_is_brand

# 프록시 적용
latest_proxy = df_proxy[df_proxy['주차'] <= contract_dt].iloc[-1]
new_house['주간변동률'] = latest_proxy['주간변동률']

# 좌표 계산
lon, lat = get_coords(input_address)
new_house['경도'] = lon
new_house['위도'] = lat

dist, stn = get_nearest_station_info(lat, lon)
new_house['역까지거리(km)'] = dist

school_dist = get_haversine_distance(KONKUK_UNIV[0], KONKUK_UNIV[1], lat, lon)
new_house['학교까지거리(km)'] = school_dist

# 시군구 더미
dummy_name = '시군구_' + input_sigungu
if dummy_name in new_house.columns:
    new_house[dummy_name] = 1

pred_xgb = xgb_model.predict(new_house[X_train.columns])[0]
pred_lgbm = lgbm_model.predict(new_house[X_train.columns])[0]

print("\n===== 예측 결과 =====")
print(f"XGBoost: {pred_xgb:,.0f} 만원")
print(f"LightGBM: {pred_lgbm:,.0f} 만원")"""

import pickle

# 모델 저장
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("lgbm_model.pkl", "wb") as f:
    pickle.dump(lgbm_model, f)

pd.DataFrame({"columns": X_train.columns}).to_csv("feature_columns.csv", index=False)
print("feature_columns.csv 저장 완료")