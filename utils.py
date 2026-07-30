"""
utils.py
--------
Shared preprocessing + prediction helpers for the AI Smart Traffic Advisor.

Mirrors the notebook's Sections 5-8 (Feature Engineering, Encoding, Scaling)
and Section 15 (Final Model Export) exactly, so a prediction made through
`predict_traffic()` goes through the same transformations the model was
trained on. Kept separate from app.py so the preprocessing logic has a single
source of truth and can be unit-tested / reused outside Streamlit.

Loads 6 artifacts (all produced by train_pipeline.py from the notebook):
    traffic_volume_model.pkl   the fitted regressor
    scaler.pkl                 RobustScaler fitted on transform_cols
    power_transformer.pkl      Yeo-Johnson PowerTransformer on target
    features.pkl               ordered list of model input columns
    transform_cols.pkl         subset of features that scaler.pkl scales
    target_info.pkl            target stats + category lists
"""
import os
import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ARTIFACT_DIR, "traffic_volume_model.pkl")


def load_artifacts(artifact_dir: str = ARTIFACT_DIR):
    """Load every saved artifact and return them as a dict."""
    model = joblib.load(os.path.join(artifact_dir, "traffic_volume_model.pkl"))
    scaler = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))
    power_transformer = joblib.load(os.path.join(artifact_dir, "power_transformer.pkl"))
    features = joblib.load(os.path.join(artifact_dir, "features.pkl"))
    transform_cols = joblib.load(os.path.join(artifact_dir, "transform_cols.pkl"))
    target_info = joblib.load(os.path.join(artifact_dir, "target_info.pkl"))
    return {
        "model": model,
        "scaler": scaler,
        "power_transformer": power_transformer,
        "features": features,
        "transform_cols": transform_cols,
        "target_info": target_info,
    }


# ---------------------------------------------------------------------------
# Feature engineering helpers (must match train_pipeline.py exactly)
# ---------------------------------------------------------------------------
def is_rush_hour(h: int) -> int:
    return int((7 <= h <= 9) or (16 <= h <= 18))


def time_of_day_bucket(h: int) -> str:
    if 5 <= h < 11:
        return "Morning"
    elif 11 <= h < 17:
        return "Afternoon"
    elif 17 <= h < 21:
        return "Evening"
    return "Night"


def build_feature_row(hour, day_of_week, month, temp_c, rain_1h, snow_1h,
                       clouds_all, weather_main, is_holiday, features, transform_cols, scaler):
    """Turn human-readable trip conditions into the exact one-row DataFrame
    the model expects (same columns/order as `features`, already scaled)."""
    is_weekend = int(day_of_week >= 5)
    weather_severity = rain_1h * 2.0 + snow_1h * 3.0 + clouds_all / 100.0
    bucket = time_of_day_bucket(hour)

    row = {
        "temp_c": temp_c,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday": int(is_holiday),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "is_rush_hour": is_rush_hour(hour),
        "weather_severity": weather_severity,
    }
    row_df = pd.DataFrame([row])

    # Recreate the one-hot columns exactly as they appear in `features`
    # (drop_first=True was used when training, so the reference category is
    # implicitly encoded as all-zeros).
    for col in features:
        if col.startswith("weather_main_"):
            row_df[col] = int(col == f"weather_main_{weather_main}")
        elif col.startswith("time_of_day_"):
            row_df[col] = int(col == f"time_of_day_{bucket}")

    row_df[transform_cols] = scaler.transform(row_df[transform_cols])
    row_df = row_df.reindex(columns=features, fill_value=0)
    return row_df


def predict_traffic(hour, day_of_week, month, temp_c, rain_1h, snow_1h,
                     clouds_all, weather_main, is_holiday, artifacts) -> float:
    """Predict traffic volume (raw scale) for one set of trip conditions."""
    row_df = build_feature_row(
        hour, day_of_week, month, temp_c, rain_1h, snow_1h, clouds_all,
        weather_main, is_holiday,
        artifacts["features"], artifacts["transform_cols"], artifacts["scaler"],
    )
    pred_transformed = artifacts["model"].predict(row_df)[0]
    pred_raw = artifacts["power_transformer"].inverse_transform(
        np.array([[pred_transformed]])
    ).ravel()[0]
    return float(max(0, pred_raw))


# ---------------------------------------------------------------------------
# Congestion helpers (thresholds come from target_info.pkl, computed once on
# the training data so the app doesn't need the raw CSV just for this)
# ---------------------------------------------------------------------------
def congestion_level(volume: float, target_info: dict) -> str:
    if volume < target_info["q33"]:
        return "Low"
    if volume < target_info["q66"]:
        return "Medium"
    return "High"


def recommendation_for(level: str) -> str:
    return {
        "Low": "Good time to travel.",
        "Medium": "Traffic is moderate. Consider travelling later.",
        "High": "Avoid this road now.",
    }[level]


def delay_minutes(volume: float, target_info: dict) -> int:
    return int(round(np.interp(
        volume, [target_info["min"], target_info["max"]], [0, 30]
    )))


def congestion_pct(volume: float, target_info: dict) -> float:
    return float(np.clip(volume / target_info["max"] * 100, 0, 100))
