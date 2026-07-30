# 🚦 AI Smart Traffic Advisor

End-to-end machine learning project that predicts hourly traffic volume on
Metro Interstate I-94 from weather and calendar conditions, wrapped in a
3-page Streamlit dashboard (Executive Summary / Deep-Dive EDA / AI Traffic
Prediction).

## Project structure

```
├── Final_Project.ipynb       Full pipeline: EDA, cleaning, feature
│                              engineering, model comparison, tuning
├── train_pipeline.py         Standalone script version of the notebook's
│                              modeling steps — regenerates every .pkl below
├── app.py                    Streamlit dashboard (entry point)
├── utils.py                  Shared preprocessing / prediction helpers,
│                              used by app.py (single source of truth so a
│                              prediction always matches how the model was
│                              trained)
├── clean_traffic.csv         Cleaned data (duplicates, 0K-temp sensor
│                              errors, and a rain_1h outlier removed) — same
│                              data the model was trained on
├── traffic_volume_model.pkl  Fitted regressor (HistGradientBoosting, chosen
│                              on R²; Random Forest and Linear Regression are
│                              also compared in the notebook)
├── scaler.pkl                RobustScaler fit on the numeric feature columns
├── power_transformer.pkl     Yeo-Johnson PowerTransformer fit on the target
│                              (traffic_volume is right-skewed; the model is
│                              trained on the transformed target and every
│                              prediction is inverse-transformed back)
├── features.pkl               Ordered list of the model's input columns
├── transform_cols.pkl        Subset of `features` that scaler.pkl scales
├── target_info.pkl           Target stats (min/max/quantiles for the
│                              Low/Medium/High congestion thresholds) +
│                              weather/time-of-day category lists + the
│                              model comparison table shown on the dashboard
└── requirements.txt
```

## How the pieces fit together

`Final_Project.ipynb` documents the full data-science process end to end.
`train_pipeline.py` is the same modeling logic distilled into one script so
the six `.pkl` artifacts can be regenerated without re-running the whole
notebook. `utils.py` re-implements the exact same feature engineering /
scaling / inverse-transform steps so that a live prediction in `app.py` goes
through the identical transformations the model was trained on.

## Running it

```bash
pip install -r requirements.txt

# (optional) regenerate all .pkl files from clean_traffic.csv
python train_pipeline.py

streamlit run app.py
```

## Model

Three candidates are trained on the engineered/encoded/scaled features with
the target power-transformed to correct its skew, then compared on the
**raw** (inverse-transformed) scale:

| Model | R² |
|---|---|
| HistGradientBoosting | **0.948** |
| Random Forest | 0.942 |
| Linear Regression (baseline) | 0.803 |

The best model (HistGradientBoosting here) is the one exported to
`traffic_volume_model.pkl`.
