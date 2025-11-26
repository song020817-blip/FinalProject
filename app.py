from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle

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
# 1. 모델 로드
# =====================================
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("lgbm_model.pkl", "rb") as f:
    lgbm_model = pickle.load(f)

# feature columns 로드
feature_columns = pd.read_csv("feature_columns.csv")["columns"].tolist()


# =====================================
# 2. 입력 데이터 정의
# =====================================
class HouseInput(BaseModel):
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


# =====================================
# 3. 예측 API
# =====================================
@app.post("/predict")
def predict_price(data: HouseInput):

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

    # 🔥 지역 더미 변수 생성
    for col in feature_columns:
        if col.startswith("시군구_"):
            raw_dict[col] = 1 if col == f"시군구_{data.sigungu}" else 0

    # DataFrame 생성 (feature 순서 완전히 맞춤)
    new_house = pd.DataFrame(
        [[raw_dict[col] for col in feature_columns]],
        columns=feature_columns,
    )

    # 모델 예측
    xgb_pred = xgb_model.predict(new_house)[0]
    lgbm_pred = lgbm_model.predict(new_house)[0]

    return {
        "xgb_pred": float(xgb_pred),
        "lgbm_pred": float(lgbm_pred),
    }


# =====================================
# 서버 실행
# =====================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)