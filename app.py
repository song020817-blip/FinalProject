from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import requests
import datetime as dt

# =====================================
# FastAPI 초기화
# =====================================
app = FastAPI()

# =====================================
# 🔓 CORS 설정 (React / Netlify 연결용)
# =====================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 개발 단계: 모두 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 0. 테스트용 엔드포인트
# =====================================
@app.get("/api/hello")
def hello():
    return {"msg": "Hello from FastAPI!"}


# =====================================
# 1. 공통 유틸 함수들 (main.py에서 가져옴)
# =====================================

# 🔑 카카오 API 키 (지금은 하드코딩, 나중에 환경변수로 빼도 됨)
KAKAO_API_KEY = "c6943568281ead90d30d6c07d618eb7d"


def get_coords(address: str):
    """주소 -> (경도, 위도) 변환 (카카오 API)"""
    url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        if not data.get("documents"):
            return None, None
        lon = float(data["documents"][0]["x"])
        lat = float(data["documents"][0]["y"])
        return lon, lat
    except Exception:
        return None, None


def get_haversine_distance(lat1, lon1, lat2, lon2):
    """두 좌표 사이 거리(km)"""
    R = 6371
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


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
    """아파트 좌표 기준 가장 가까운 역까지 거리(km)"""
    best_dist = np.inf
    best_name = None
    for name, (slat, slon) in STATION_COORDS.items():
        d = get_haversine_distance(lat, lon, slat, slon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_dist, best_name


# 🔁 proxy_data (주간변동률) 로드
# main.py에서 만든 proxy_data.csv를 그대로 사용
df_proxy = pd.read_csv("proxy_data.csv")
df_proxy["주차"] = pd.to_datetime(df_proxy["주차"])
df_proxy = df_proxy.sort_values("주차")


def get_proxy_value_from_ym(contract_ym: int) -> float:
    """
    계약년월(예: 202501) 기준으로
    해당 월의 1일 날짜를 잡고, 그 이전 주차 중 가장 최근 '주간변동률' 사용
    """
    year = contract_ym // 100
    month = contract_ym % 100
    try:
        contract_dt = dt.datetime(year, month, 1)
    except ValueError:
        # 잘못된 년/월이면 0 처리
        return 0.0

    df_tmp = df_proxy[df_proxy["주차"] <= contract_dt]
    if df_tmp.empty:
        return 0.0
    return float(df_tmp.iloc[-1]["주간변동률"])


# =====================================
# 2. 모델 로드
# =====================================
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("lgbm_model.pkl", "rb") as f:
    lgbm_model = pickle.load(f)

feature_columns = pd.read_csv("feature_columns.csv")["columns"].tolist()


# =====================================
# 3. 입력 데이터 정의
# =====================================

# 🔹 기존 버전: 프론트에서 모든 값을 계산해서 보내는 버전
class HouseInputFull(BaseModel):
    area: float
    floor: int
    year_built: int
    contract_ym: int
    sigungu: str
    total_units: int
    is_brand: int
    lat: float
    lon: float
    station_dist: float
    univ_dist: float
    proxy_value: float


# 🔹 간단 버전: 주소만 보내면 백엔드가 다 계산해주는 버전
class HouseInputSimple(BaseModel):
    area: float
    floor: int
    year_built: int
    contract_ym: int
    sigungu: str          # "서울특별시 광진구" 형식
    total_units: int
    is_brand: int
    address: str          # "서울특별시 광진구 광나루로 410" 이런 식


# =====================================
# 4-1. 기존 예측 API (/predict) - 그대로 유지
# =====================================
@app.post("/predict")
def predict_price_full(data: HouseInputFull):

    raw_dict = {
        "전용면적(㎡)": data.area,
        "층": data.floor,
        "계약년월": data.contract_ym,
        "아파트연식": (data.contract_ym // 100) - data.year_built,
        "기준금리": 3.5,
        "총세대수": data.total_units,
        "is_brand": data.is_brand,
        "위도": data.lat,
        "경도": data.lon,
        "역까지거리(km)": data.station_dist,
        "학교까지거리(km)": data.univ_dist,
        "주간변동률": data.proxy_value,
    }

    # 시군구 더미
    for col in feature_columns:
        if col.startswith("시군구_"):
            raw_dict[col] = 1 if col == f"시군구_{data.sigungu}" else 0

    new_house = pd.DataFrame(
        [[raw_dict[col] for col in feature_columns]],
        columns=feature_columns,
    )

    xgb_pred = xgb_model.predict(new_house)[0]
    lgbm_pred = lgbm_model.predict(new_house)[0]

    return {
        "xgb_pred": float(xgb_pred),
        "lgbm_pred": float(lgbm_pred),
    }


# =====================================
# 4-2. 간단 예측 API (/predict_simple)
#    → 주소만 넣으면 위도/경도/거리/proxy 자동 계산
# =====================================
@app.post("/predict_simple")
def predict_price_simple(data: HouseInputSimple):

    # 1) 주소 → 좌표
    lon, lat = get_coords(data.address)
    if lon is None or lat is None:
        return {"error": "주소로 좌표를 찾을 수 없습니다.", "detail": data.address}

    # 2) 역까지 거리 / 가장 가까운 역
    station_dist, station_name = get_nearest_station_info(lat, lon)

    # 3) 건국대까지 거리
    univ_dist = get_haversine_distance(
        KONKUK_UNIV[0], KONKUK_UNIV[1], lat, lon
    )

    # 4) 계약년월 → proxy_value
    proxy_value = get_proxy_value_from_ym(data.contract_ym)

    # 5) raw_dict 구성
    raw_dict = {
        "전용면적(㎡)": data.area,
        "층": data.floor,
        "계약년월": data.contract_ym,
        "아파트연식": (data.contract_ym // 100) - data.year_built,
        "기준금리": 3.5,
        "총세대수": data.total_units,
        "is_brand": data.is_brand,
        "위도": lat,
        "경도": lon,
        "역까지거리(km)": station_dist,
        "학교까지거리(km)": univ_dist,
        "주간변동률": proxy_value,
    }

    # 시군구 더미
    for col in feature_columns:
        if col.startswith("시군구_"):
            raw_dict[col] = 1 if col == f"시군구_{data.sigungu}" else 0

    new_house = pd.DataFrame(
        [[raw_dict[col] for col in feature_columns]],
        columns=feature_columns,
    )

    xgb_pred = xgb_model.predict(new_house)[0]
    lgbm_pred = lgbm_model.predict(new_house)[0]

    return {
        "xgb_pred": float(xgb_pred),
        "lgbm_pred": float(lgbm_pred),
        "lat": lat,
        "lon": lon,
        "nearest_station": station_name,
        "station_dist": station_dist,
        "univ_dist": univ_dist,
        "proxy_value": proxy_value,
    }


# =====================================
# 서버 실행 (로컬 디버깅용)
# =====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
