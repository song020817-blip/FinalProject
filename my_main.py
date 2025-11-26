import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# 1️⃣ 데이터 불러오기
# ==========================================
file_path = "merged_apartment_2021_2025.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

print("✅ 데이터 로드 완료!")

# ==========================================
# 2️⃣ 컬럼명 표준화
# ==========================================
df = df.rename(columns={
    '시군구': '주소',
    '전용면적(㎡)': '전용면적',
    '보증금(만원)': '보증금',
    '월세금(만원)': '월세금'
})

# ==========================================
# 3️⃣ 전처리
# ==========================================
df = df.dropna(subset=['주소', '전용면적', '층', '보증금'])

for col in ['전용면적', '층', '건축년도', '보증금', '월세금']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['주소', '전용면적', '층', '건축년도', '보증금'])

X = df[['주소', '전용면적', '층', '건축년도']]
y = df['보증금']

# ==========================================
# 4️⃣ 데이터 분리
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 5️⃣ 전처리 (카테고리형/수치형 분리)
# ==========================================
categorical_features = ['주소']
numerical_features = ['전용면적', '층', '건축년도']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# ==========================================
# 6️⃣ XGBoost 모델
# ==========================================
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

xgb_pipeline.fit(X_train, y_train)
xgb_pred = xgb_pipeline.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# ==========================================
# 7️⃣ LightGBM 모델
# ==========================================
lgbm_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lgbm_model)
])

lgbm_pipeline.fit(X_train, y_train)
lgbm_pred = lgbm_pipeline.predict(X_test)

lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
lgbm_r2 = r2_score(y_test, lgbm_pred)

# ==========================================
# 8️⃣ 결과 비교 출력
# ==========================================
print("\n✅ 모델 성능 비교 결과")
print("───────────────────────────────")
print(f"📦 XGBoost → MAE: {xgb_mae:,.0f} | R²: {xgb_r2:.3f}")
print(f"⚡ LightGBM → MAE: {lgbm_mae:,.0f} | R²: {lgbm_r2:.3f}")
print("───────────────────────────────")

# ==========================================
# 9️⃣ 샘플 예측
# ==========================================
sample = pd.DataFrame([{
    '주소': '서울특별시 강남구 압구정동',
    '전용면적': 84.0,
    '층': 10,
    '건축년도': 2015
}])

xgb_price = xgb_pipeline.predict(sample)[0]
lgbm_price = lgbm_pipeline.predict(sample)[0]

print("\n🏠 예측 테스트 (입력값:", dict(sample.iloc[0]), ")")
print(f"💰 XGBoost 예측 보증금: {xgb_price:,.0f} 만원")
print(f"💰 LightGBM 예측 보증금: {lgbm_price:,.0f} 만원")