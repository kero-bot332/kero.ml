"""
Rebuilds the traffic-volume pipeline from clean_traffic.csv and exports it as
SEPARATE artifact files (matching the target repo layout) instead of one
joblib bundle:

    scaler.pkl            RobustScaler fitted on the numeric feature columns
    power_transformer.pkl PowerTransformer (Yeo-Johnson) fitted on the target
    features.pkl          final ordered list of model input columns
    transform_cols.pkl    subset of `features` that scaler.pkl scales
    target_info.pkl       target name + raw-scale stats needed by the app
                           (min/max/quantiles for congestion thresholds, delay)
    traffic_volume_model.pkl   the fitted regressor (joblib)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)

OUT_DIR = "/home/claude/build/out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the already-cleaned data (duplicates / 0K temp / rain outlier removed
#    upstream, exactly as in the notebook's Section 4).
# ---------------------------------------------------------------------------
df = pd.read_csv("/home/claude/build/clean_traffic.csv")
df["date_time"] = pd.to_datetime(df["date_time"])
datetime_col = "date_time"
target = "traffic_volume"

# ---------------------------------------------------------------------------
# 2. Feature engineering (notebook Section 5)
# ---------------------------------------------------------------------------
df_feat = df.copy()
df_feat["hour"] = df_feat[datetime_col].dt.hour
df_feat["day_of_week"] = df_feat[datetime_col].dt.dayofweek
df_feat["month"] = df_feat[datetime_col].dt.month
df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
df_feat["is_holiday"] = (df_feat["holiday"].astype(str) != "None") & df_feat["holiday"].notna()
df_feat["is_holiday"] = df_feat["is_holiday"].astype(int)
df_feat["temp_c"] = df_feat["temp"] - 273.15


def time_of_day_bucket(h):
    if 5 <= h < 11:
        return "Morning"
    elif 11 <= h < 17:
        return "Afternoon"
    elif 17 <= h < 21:
        return "Evening"
    else:
        return "Night"


df_feat["time_of_day"] = df_feat["hour"].apply(time_of_day_bucket)

# Cyclical encoding
df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)


def is_rush_hour(h):
    return int((7 <= h <= 9) or (16 <= h <= 18))


df_feat["is_rush_hour"] = df_feat["hour"].apply(is_rush_hour)
df_feat["weather_severity"] = (
    df_feat["rain_1h"] * 2.0 + df_feat["snow_1h"] * 3.0 + df_feat["clouds_all"] / 100.0
)

# ---------------------------------------------------------------------------
# 3. Encoding categoricals (notebook Section 6) — capture category lists
# ---------------------------------------------------------------------------
weather_categories = sorted(df_feat["weather_main"].dropna().unique().tolist())
time_of_day_categories = sorted(df_feat["time_of_day"].unique().tolist())

model_cols = [
    "temp_c", "rain_1h", "snow_1h", "clouds_all",
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_rush_hour", "weather_severity",
    "weather_main", "time_of_day", target,
]
df_model = df_feat[model_cols].copy()
df_encoded = pd.get_dummies(
    df_model, columns=["weather_main", "time_of_day"], drop_first=True
)

# ---------------------------------------------------------------------------
# 4. PowerTransformer on the target (Yeo-Johnson: traffic_volume has a couple
#    of zero values, so Box-Cox is not usable). This corrects the target's
#    skew before training, matching the reference project's separate
#    power_transformer.pkl artifact.
# ---------------------------------------------------------------------------
power_transformer = PowerTransformer(method="yeo-johnson")
y_raw = df_encoded[target].values.reshape(-1, 1)
y_trans = power_transformer.fit_transform(y_raw).ravel()

X = df_encoded.drop(columns=[target])
feature_columns = X.columns.tolist()

X_train, X_test, y_train_t, y_test_t, y_train_raw, y_test_raw = train_test_split(
    X, y_trans, df_encoded[target].values, test_size=0.2, random_state=SEED
)

# ---------------------------------------------------------------------------
# 5. Scaling numeric (non-binary, non-cyclical) features (notebook Section 8)
# ---------------------------------------------------------------------------
transform_cols = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "weather_severity"]

scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[transform_cols] = scaler.fit_transform(X_train[transform_cols])
X_test_scaled[transform_cols] = scaler.transform(X_test[transform_cols])

# ---------------------------------------------------------------------------
# 6. Train candidate models on the transformed target, compare on the
#    RAW (inverse-transformed) scale, keep the best.
# ---------------------------------------------------------------------------
def inv(y):
    return power_transformer.inverse_transform(np.array(y).reshape(-1, 1)).ravel()


candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_split=4,
        min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_depth=8, learning_rate=0.08, max_iter=400, random_state=SEED,
    ),
}

results = []
fitted = {}
for name, mdl in candidates.items():
    mdl.fit(X_train_scaled, y_train_t)
    pred_t = mdl.predict(X_test_scaled)
    pred_raw = np.clip(inv(pred_t), 0, None)
    mae = mean_absolute_error(y_test_raw, pred_raw)
    rmse = mean_squared_error(y_test_raw, pred_raw) ** 0.5
    r2 = r2_score(y_test_raw, pred_raw)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})
    fitted[name] = mdl

df_results = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
print(df_results.to_string(index=False))

best_model_name = df_results.iloc[0]["Model"]
final_model = fitted[best_model_name]
print("\nBest model:", best_model_name)

# ---------------------------------------------------------------------------
# 7. Save every artifact SEPARATELY
# ---------------------------------------------------------------------------
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
joblib.dump(power_transformer, os.path.join(OUT_DIR, "power_transformer.pkl"))
joblib.dump(feature_columns, os.path.join(OUT_DIR, "features.pkl"))
joblib.dump(transform_cols, os.path.join(OUT_DIR, "transform_cols.pkl"))

target_info = {
    "target_name": target,
    "min": float(df_encoded[target].min()),
    "max": float(df_encoded[target].max()),
    "q33": float(df_encoded[target].quantile(0.33)),
    "q66": float(df_encoded[target].quantile(0.66)),
    "mean": float(df_encoded[target].mean()),
    "weather_categories": weather_categories,
    "time_of_day_categories": time_of_day_categories,
    "best_model_name": best_model_name,
    "model_results": df_results.to_dict(orient="records"),
}
joblib.dump(target_info, os.path.join(OUT_DIR, "target_info.pkl"))
joblib.dump(final_model, os.path.join(OUT_DIR, "traffic_volume_model.pkl"))

print("\nSaved artifacts to", OUT_DIR)
print(os.listdir(OUT_DIR))
