"""
GaneshShah_AQIPredictor.py
==========================
AI AQI Predictor - Complete Streamlit Application
Author : Ganesh Shah
Developed using IBM Bob
Description: Predicts Air Quality Index (AQI) for Indian monitoring stations
             using a scikit-learn ensemble model with Streamlit UI.
"""

# =============================================================================
# SECTION 1 : IMPORTS AND CONSTANTS
# =============================================================================
from __future__ import annotations

import io
import pathlib
import warnings
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "enriched_aqi_dataset.csv"

# ── Random seed ──────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── AQI Breakpoints (US-EPA style used by the dataset) ───────────────────────
AQI_BREAKPOINTS: list[tuple[int, int, str]] = [
    (0,   50,  "Good"),
    (51,  100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]

AQI_COLORS: dict[str, str] = {
    "Good":                              "#00E400",
    "Moderate":                          "#FFFF00",
    "Unhealthy for Sensitive Groups":    "#FF7E00",
    "Unhealthy":                         "#FF0000",
    "Very Unhealthy":                    "#8F3F97",
    "Hazardous":                         "#7E0023",
}

AQI_TEXT_COLORS: dict[str, str] = {
    "Good":                              "#000000",
    "Moderate":                          "#000000",
    "Unhealthy for Sensitive Groups":    "#FFFFFF",
    "Unhealthy":                         "#FFFFFF",
    "Very Unhealthy":                    "#FFFFFF",
    "Hazardous":                         "#FFFFFF",
}

HEALTH_ADVISORY: dict[str, dict[str, str]] = {
    "Good": {
        "summary": "Air quality is satisfactory and poses little or no risk.",
        "sensitive": "Everyone can enjoy normal outdoor activities.",
        "actions": "No special precautions needed. Great day for outdoor activities!",
    },
    "Moderate": {
        "summary": "Air quality is acceptable; minor concern for very sensitive people.",
        "sensitive": "Unusually sensitive individuals should consider limiting prolonged outdoor activity.",
        "actions": "General public can be active outdoors. Sensitive individuals should watch for symptoms.",
    },
    "Unhealthy for Sensitive Groups": {
        "summary": "Members of sensitive groups may experience health effects.",
        "sensitive": "Heart/lung disease patients, elderly and children should reduce prolonged outdoor activity.",
        "actions": "Wear a mask if outdoors for extended periods. Keep windows closed if possible.",
    },
    "Unhealthy": {
        "summary": "Everyone may experience health effects; sensitive groups may experience more serious effects.",
        "sensitive": "Avoid prolonged outdoor exertion. Children, elderly and those with heart/lung disease should stay indoors.",
        "actions": "Wear N95 mask outdoors. Keep indoor air filtered. Reduce physical exertion.",
    },
    "Very Unhealthy": {
        "summary": "Health alert: everyone may experience more serious health effects.",
        "sensitive": "Everyone should avoid all outdoor physical activity.",
        "actions": "Stay indoors with windows closed. Use air purifiers. Wear N95/P100 masks if going outside is necessary.",
    },
    "Hazardous": {
        "summary": "Health warning of emergency conditions. The entire population is likely to be affected.",
        "sensitive": "Everyone should avoid ALL outdoor activity. Seek medical attention if experiencing symptoms.",
        "actions": "Stay indoors with filtered air. Seal gaps in doors/windows. Seek immediate medical care for respiratory symptoms.",
    },
}

# ── Feature column lists ──────────────────────────────────────────────────────
CAT_FEATURES: list[str] = ["pollutant_id", "state", "Season"]
NUM_FEATURES: list[str] = [
    "pollutant_min", "pollutant_max", "pollutant_avg", "pollutant_range",
    "pollutant_missing",
    "Temperature_C", "Humidity_%", "Wind_Speed_kmh",
    "Month_sin", "Month_cos", "Hour_sin", "Hour_cos",
    "Day_of_Week", "latitude", "longitude",
]
ALL_FEATURES: list[str] = CAT_FEATURES + NUM_FEATURES

BUCKET_ORDER: list[str] = [
    "Good", "Moderate", "Unhealthy for Sensitive Groups",
    "Unhealthy", "Very Unhealthy", "Hazardous",
]


# =============================================================================
# SECTION 2 : CSS THEME
# =============================================================================
def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* ── Global ─────────────────────────────────────────────────── */
        [data-testid="stAppViewContainer"] { background: #f5f7fa; }
        [data-testid="stSidebar"] { background: #1a1f36; color: #e0e4f0; }
        [data-testid="stSidebar"] * { color: #e0e4f0 !important; }
        [data-testid="stSidebar"] .stMarkdown h3 { color: #7dd3fc !important; }

        /* ── Hero header ─────────────────────────────────────────────── */
        .hero-header {
            background: linear-gradient(135deg, #1a1f36 0%, #0f4c81 50%, #1a6fa8 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .hero-header h1 { font-size: 2.2rem; margin:0; font-weight: 700; }
        .hero-header p  { font-size: 1rem;   margin: 0.3rem 0 0; opacity: 0.85; }

        /* ── Metric cards ────────────────────────────────────────────── */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
            border-top: 4px solid #0f4c81;
        }
        .metric-card .metric-value {
            font-size: 2rem; font-weight: 700; color: #1a1f36;
        }
        .metric-card .metric-label {
            font-size: 0.82rem; color: #64748b; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.04em;
        }

        /* ── AQI badge ───────────────────────────────────────────────── */
        .aqi-badge {
            display: inline-block;
            padding: 0.4rem 1.1rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: 0.02em;
        }

        /* ── Section divider ─────────────────────────────────────────── */
        .section-title {
            font-size: 1.15rem; font-weight: 600; color: #1a1f36;
            border-left: 4px solid #0f4c81;
            padding-left: 0.7rem; margin: 1.2rem 0 0.6rem;
        }

        /* ── Prediction result card ──────────────────────────────────── */
        .pred-card {
            background: white;
            border-radius: 16px;
            padding: 1.8rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.10);
            margin-top: 1rem;
        }

        /* ── Advisory card ───────────────────────────────────────────── */
        .advisory-card {
            background: #f8fafc;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-top: 0.8rem;
            border-left: 5px solid #0f4c81;
        }

        /* ── Tab styling ─────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: white;
            border-radius: 10px;
            padding: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 500; border-radius: 7px; padding: 6px 16px;
        }
        .stTabs [aria-selected="true"] {
            background: #0f4c81 !important; color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 3 : DATA LOADING AND VALIDATION
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file: Any = None) -> pd.DataFrame:
    """Load and validate the AQI dataset.

    Parameters
    ----------
    uploaded_file : file-like, optional
        If the CSV is not found on disk, accept a Streamlit uploaded file.

    Returns
    -------
    pd.DataFrame
        Validated and minimally processed DataFrame.
    """
    required_cols = {
        "country", "state", "city", "station", "last_update",
        "latitude", "longitude", "pollutant_id",
        "pollutant_min", "pollutant_max", "pollutant_avg",
        "AQI", "AQI_Bucket",
        "Temperature_C", "Humidity_%", "Wind_Speed_kmh",
        "Year", "Month", "Day", "Hour", "Day_of_Week", "Season",
    }

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()  # caller checks for empty

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    return df


# =============================================================================
# SECTION 4 : FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features needed by the model.

    Adds:
    - pollutant_range  : pollutant_max - pollutant_min
    - pollutant_missing: 1 if pollutant readings are NaN, else 0
    - Month_sin/cos    : cyclic encoding of Month
    - Hour_sin/cos     : cyclic encoding of Hour

    Does NOT modify AQI_Bucket (kept for stratified split only).
    Does NOT drop any columns (the pipeline handles that).
    """
    df = df.copy()

    # Pollutant missing indicator (BEFORE any imputation)
    df["pollutant_missing"] = (
        df["pollutant_avg"].isna().astype(int)
    )

    # Pollutant range (NaN if readings missing → imputed downstream)
    df["pollutant_range"] = df["pollutant_max"] - df["pollutant_min"]

    # Cyclic month encoding (1-12 → sin/cos)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Cyclic hour encoding (0-23 → sin/cos)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    return df


def aqi_to_bucket(aqi: float) -> str:
    """Convert a numeric AQI value to the corresponding bucket label."""
    for lo, hi, label in AQI_BREAKPOINTS:
        if lo <= aqi <= hi:
            return label
    return "Hazardous"  # >500 capped


# =============================================================================
# SECTION 5 : MODEL TRAINING
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_model(
    df_raw: pd.DataFrame,
) -> dict[str, Any]:
    """Engineer features, split, compare models and return all artefacts.

    Returns
    -------
    dict with keys:
        best_pipeline, best_name, comparison_df,
        X_test, y_test, y_pred, residuals,
        r2, mae, rmse, bucket_acc, within1_acc,
        perm_imp_df, pred_lo, pred_hi,
        train_time_best
    """
    import time

    df = engineer_features(df_raw)

    X = df[ALL_FEATURES].copy()
    y = df["AQI"].values

    # Stratified split by AQI_Bucket
    strat_col = df["AQI_Bucket"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE,
        stratify=strat_col,
    )

    # ── Pipeline builder ────────────────────────────────────────────────────
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUM_FEATURES),
        ("cat", cat_transformer, CAT_FEATURES),
    ])

    candidates = {
        "Ridge (baseline)":             Ridge(alpha=10.0),
        "Random Forest":                RandomForestRegressor(
                                            n_estimators=200, max_depth=None,
                                            min_samples_leaf=2, n_jobs=-1,
                                            random_state=RANDOM_STATE),
        "Extra Trees":                  ExtraTreesRegressor(
                                            n_estimators=200, max_depth=None,
                                            min_samples_leaf=2, n_jobs=-1,
                                            random_state=RANDOM_STATE),
        "Gradient Boosting":            GradientBoostingRegressor(
                                            n_estimators=300, learning_rate=0.08,
                                            max_depth=5, subsample=0.8,
                                            random_state=RANDOM_STATE),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows: list[dict] = []
    cv_scores: dict[str, float] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([("pre", preprocessor), ("reg", estimator)])
        t0 = time.time()
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        cv_time = time.time() - t0
        cv_rmse = -scores.mean()
        cv_scores[name] = cv_rmse
        comparison_rows.append({
            "Model":    name,
            "CV RMSE":  round(cv_rmse, 3),
            "CV Time (s)": round(cv_time, 1),
        })

    best_name = min(cv_scores, key=cv_scores.__getitem__)
    best_estimator = candidates[best_name]
    best_pipeline = Pipeline([("pre", preprocessor), ("reg", best_estimator)])

    t0 = time.time()
    best_pipeline.fit(X_train, y_train)
    train_time_best = round(time.time() - t0, 1)

    y_pred: np.ndarray = np.clip(best_pipeline.predict(X_test), 0, 499).astype(float)

    r2   = round(float(r2_score(y_test, y_pred)), 4)
    mae  = round(float(mean_absolute_error(y_test, y_pred)), 3)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)

    # Bucket accuracy
    pred_buckets = np.array([aqi_to_bucket(v) for v in y_pred])
    true_buckets = np.array([aqi_to_bucket(v) for v in y_test])
    bucket_acc   = round((pred_buckets == true_buckets).mean(), 4)

    # Within-one-category accuracy
    bucket_idx = {b: i for i, b in enumerate(BUCKET_ORDER)}
    pred_idx   = np.array([bucket_idx[b] for b in pred_buckets])
    true_idx   = np.array([bucket_idx[b] for b in true_buckets])
    within1_acc = round((np.abs(pred_idx - true_idx) <= 1).mean(), 4)

    # Residuals and prediction interval (10th-90th percentile)
    residuals = y_pred - y_test
    pred_lo = float(np.percentile(residuals, 10))
    pred_hi = float(np.percentile(residuals, 90))

    # Fill comparison table with test metrics
    for row in comparison_rows:
        if row["Model"] == best_name:
            row["Test R²"]   = r2
            row["Test MAE"]  = mae
            row["Test RMSE"] = rmse
            row["Train Time (s)"] = train_time_best
            row["Best ✓"] = "✓"
        else:
            row["Test R²"]   = "-"
            row["Test MAE"]  = "-"
            row["Test RMSE"] = "-"
            row["Train Time (s)"] = "-"
            row["Best ✓"] = ""

    comparison_df = pd.DataFrame(comparison_rows)

    # Permutation importance (on 500-row sample for speed)
    sample_size = min(500, len(X_test))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test), sample_size, replace=False)
    perm_result = permutation_importance(
        best_pipeline, X_test.iloc[idx], y_test[idx],
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    perm_imp_df = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance": np.asarray(perm_result.importances_mean),  # type: ignore[union-attr]
        "std":        np.asarray(perm_result.importances_std),   # type: ignore[union-attr]
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "best_pipeline":  best_pipeline,
        "best_name":      best_name,
        "comparison_df":  comparison_df,
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "y_pred":         y_pred,
        "residuals":      residuals,
        "r2":             r2,
        "mae":            mae,
        "rmse":           rmse,
        "bucket_acc":     bucket_acc,
        "within1_acc":    within1_acc,
        "pred_lo":        pred_lo,
        "pred_hi":        pred_hi,
        "perm_imp_df":    perm_imp_df,
        "train_time_best": train_time_best,
    }


# =============================================================================
# SECTION 6 : EVALUATION HELPERS
# =============================================================================
def get_confusion_matrix_df(y_true_aqi: np.ndarray, y_pred_aqi: np.ndarray) -> pd.DataFrame:
    """Return a labelled confusion matrix DataFrame."""
    true_b = [aqi_to_bucket(v) for v in y_true_aqi]
    pred_b = [aqi_to_bucket(v) for v in y_pred_aqi]
    cm = confusion_matrix(true_b, pred_b, labels=BUCKET_ORDER)
    return pd.DataFrame(cm, index=list(BUCKET_ORDER), columns=list(BUCKET_ORDER))


def get_classification_report_df(
    y_true_aqi: np.ndarray, y_pred_aqi: np.ndarray
) -> pd.DataFrame:
    """Return precision / recall / F1 per bucket as a DataFrame."""
    true_b = [aqi_to_bucket(v) for v in y_true_aqi]
    pred_b = [aqi_to_bucket(v) for v in y_pred_aqi]
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_b, pred_b, labels=BUCKET_ORDER, zero_division=0  # type: ignore[call-overload]
    )
    return pd.DataFrame({
        "Category":  BUCKET_ORDER,
        "Precision": np.round(prec, 3),
        "Recall":    np.round(rec,  3),
        "F1-Score":  np.round(f1,   3),
        "Support":   sup,
    })


# =============================================================================
# SECTION 7 : PREDICTION HELPERS AND HEALTH ADVISORIES
# =============================================================================
def build_input_row(
    *,
    pollutant_id: str,
    state: str,
    season: str,
    pollutant_min: float | None,
    pollutant_max: float | None,
    pollutant_avg: float | None,
    pollutant_missing: int,
    temperature: float,
    humidity: float,
    wind_speed: float,
    month: int,
    hour: int,
    day_of_week: int,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Build a single-row DataFrame ready for pipeline.predict()."""
    pm = pollutant_missing
    p_min = None if pm else pollutant_min
    p_max = None if pm else pollutant_max
    p_avg = None if pm else pollutant_avg
    p_range = (None if pm else (p_max - p_min) if (p_max is not None and p_min is not None) else None)

    row = {
        "pollutant_id":      pollutant_id,
        "state":             state,
        "Season":            season,
        "pollutant_min":     p_min,
        "pollutant_max":     p_max,
        "pollutant_avg":     p_avg,
        "pollutant_range":   p_range,
        "pollutant_missing": pm,
        "Temperature_C":     temperature,
        "Humidity_%":        humidity,
        "Wind_Speed_kmh":    wind_speed,
        "Month_sin":         np.sin(2 * np.pi * month / 12),
        "Month_cos":         np.cos(2 * np.pi * month / 12),
        "Hour_sin":          np.sin(2 * np.pi * hour / 24),
        "Hour_cos":          np.cos(2 * np.pi * hour / 24),
        "Day_of_Week":       day_of_week,
        "latitude":          latitude,
        "longitude":         longitude,
    }
    return pd.DataFrame([row])[ALL_FEATURES]


def predict_aqi(pipeline: Any, input_df: pd.DataFrame, pred_lo: float, pred_hi: float) -> dict:
    """Run prediction and return a result dict."""
    raw = float(pipeline.predict(input_df)[0])
    aqi = int(np.clip(round(raw), 0, 499))
    bucket = aqi_to_bucket(aqi)
    likely_lo = max(0, int(aqi + pred_lo))
    likely_hi = min(499, int(aqi + pred_hi))
    advisory  = HEALTH_ADVISORY[bucket]
    color     = AQI_COLORS[bucket]
    txt_color = AQI_TEXT_COLORS[bucket]
    return {
        "aqi": aqi, "bucket": bucket,
        "likely_lo": likely_lo, "likely_hi": likely_hi,
        "advisory": advisory, "color": color, "txt_color": txt_color,
    }


def month_to_season(month: int) -> str:
    """Map a month number (1-12) to a season name."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"


# =============================================================================
# SECTION 8 : UI TABS
# =============================================================================

# ── Tab 1: Predict ────────────────────────────────────────────────────────────
def tab_predict(df: pd.DataFrame, model_artefacts: dict) -> None:
    """Prediction tab with cascading location inputs and prediction output."""
    pipeline  = model_artefacts["best_pipeline"]
    pred_lo   = model_artefacts["pred_lo"]
    pred_hi   = model_artefacts["pred_hi"]
    perm_imp  = model_artefacts["perm_imp_df"]

    st.markdown('<div class="section-title">Location</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        states = sorted(df["state"].unique().tolist())
        sel_state = st.selectbox("State", states, key="pred_state")
    with col2:
        cities = sorted(df[df["state"] == sel_state]["city"].unique().tolist())
        sel_city = st.selectbox("City", cities, key="pred_city")
    with col3:
        stations = sorted(df[(df["state"] == sel_state) & (df["city"] == sel_city)]["station"].unique().tolist())
        stations_opts = ["(use city median)"] + list(stations)
        sel_station = st.selectbox("Station (optional)", stations_opts, key="pred_station")

    # Latitude / longitude auto-fill
    if sel_station != "(use city median)":
        loc_mask = (df["station"] == sel_station)
    else:
        loc_mask = (df["state"] == sel_state) & (df["city"] == sel_city)
    lat_default = float(df[loc_mask]["latitude"].median())  # type: ignore[arg-type]
    lon_default = float(df[loc_mask]["longitude"].median())  # type: ignore[arg-type]

    col4, col5 = st.columns(2)
    with col4:
        lat = st.number_input("Latitude",  value=lat_default, format="%.4f", key="pred_lat")
    with col5:
        lon = st.number_input("Longitude", value=lon_default, format="%.4f", key="pred_lon")

    st.markdown('<div class="section-title">Pollutant Reading</div>', unsafe_allow_html=True)
    col6, col7 = st.columns([2, 1])
    with col6:
        poll_id = st.selectbox("Pollutant Type", sorted(df["pollutant_id"].unique()), key="pred_poll")
    with col7:
        poll_missing = st.checkbox("Pollutant reading not reported", key="pred_miss",
                                   help="Check this if no pollutant measurement is available. "
                                        "In this dataset, ALL 357 missing readings correspond to Hazardous AQI rows.")

    if not poll_missing:
        p_avg_med  = float(df["pollutant_avg"].median())
        p_min_med  = float(df["pollutant_min"].median())
        p_max_med  = float(df["pollutant_max"].median())
        p_min_glob = float(df["pollutant_min"].min())
        p_max_glob = float(df["pollutant_max"].max())

        c8, c9, c10 = st.columns(3)
        with c8:
            p_min = st.number_input("Pollutant Min",  min_value=p_min_glob, max_value=p_max_glob,
                                    value=p_min_med, format="%.1f", key="pred_pmin")
        with c10:
            p_max = st.number_input("Pollutant Max",  min_value=p_min_glob, max_value=p_max_glob,
                                    value=p_max_med, format="%.1f", key="pred_pmax")
        with c9:
            p_avg = st.number_input("Pollutant Avg",  min_value=p_min_glob, max_value=p_max_glob,
                                    value=p_avg_med, format="%.1f", key="pred_pavg")

        if not (p_min <= p_avg <= p_max):
            st.error("⚠️ Validation error: Min ≤ Avg ≤ Max must hold. Please fix the pollutant values.")
            return
    else:
        p_min = p_max = p_avg = None

    st.markdown('<div class="section-title">Weather Conditions</div>', unsafe_allow_html=True)
    c11, c12, c13 = st.columns(3)
    with c11:
        temperature = st.slider("Temperature (°C)", 15.0, 40.0, 27.0, 0.5, key="pred_temp")
    with c12:
        humidity    = st.slider("Humidity (%)",      30.0, 95.0, 62.0, 1.0, key="pred_hum")
    with c13:
        wind_speed  = st.slider("Wind Speed (km/h)", 0.0,  30.0, 15.0, 0.5, key="pred_wind")

    st.markdown('<div class="section-title">Date & Time</div>', unsafe_allow_html=True)
    c14, c15 = st.columns(2)
    with c14:
        sel_date = st.date_input("Date", value=date.today(), key="pred_date")
    with c15:
        sel_hour = st.slider("Hour (0-23)", 0, 23, 12, key="pred_hour")

    month      = sel_date.month  # type: ignore[union-attr]
    day_of_week = sel_date.weekday()   # type: ignore[union-attr]  # 0=Monday
    season     = month_to_season(month)

    if st.button("🔍 Predict AQI", type="primary", key="predict_btn"):
        input_df = build_input_row(
            pollutant_id=str(poll_id), state=str(sel_state), season=season,
            pollutant_min=p_min, pollutant_max=p_max, pollutant_avg=p_avg,
            pollutant_missing=int(poll_missing),
            temperature=temperature, humidity=humidity, wind_speed=wind_speed,
            month=month, hour=sel_hour, day_of_week=day_of_week,
            latitude=lat, longitude=lon,
        )
        try:
            result = predict_aqi(pipeline, input_df, pred_lo, pred_hi)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        # ── Output ──────────────────────────────────────────────────────────
        st.markdown("---")
        aqi_val    = result["aqi"]
        bucket     = result["bucket"]
        color      = result["color"]
        txt_color  = result["txt_color"]
        lo, hi     = result["likely_lo"], result["likely_hi"]
        advisory   = result["advisory"]
        top_feats  = perm_imp.head(5)["feature"].tolist()

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            title={"text": "AQI", "font": {"size": 20}},
            gauge={
                "axis":  {"range": [0, 500], "tickwidth": 1},
                "bar":   {"color": color},
                "steps": [
                    {"range": [0,   50],  "color": "#00E400"},
                    {"range": [51,  100], "color": "#FFFF00"},
                    {"range": [101, 150], "color": "#FF7E00"},
                    {"range": [151, 200], "color": "#FF0000"},
                    {"range": [201, 300], "color": "#8F3F97"},
                    {"range": [301, 500], "color": "#7E0023"},
                ],
            },
            number={"font": {"size": 48, "color": color}},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20))

        col_g, col_r = st.columns([1, 1])
        with col_g:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_r:
            st.markdown(
                f'<div class="pred-card">'
                f'<div style="font-size:1rem; color:#64748b; margin-bottom:0.3rem;">Predicted AQI</div>'
                f'<div style="font-size:3.5rem; font-weight:800; color:{color};">{aqi_val}</div>'
                f'<span class="aqi-badge" style="background:{color}; color:{txt_color};">{bucket}</span><br><br>'
                f'<div style="font-size:0.9rem; color:#475569;">Likely range: <b>{lo} – {hi}</b> AQI points '
                f'<span style="color:#94a3b8;">(10th–90th pct of hold-out residuals)</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="advisory-card">'
            f'<b>🌡️ Summary:</b> {advisory["summary"]}<br>'
            f'<b>⚠️ For Sensitive Groups:</b> {advisory["sensitive"]}<br>'
            f'<b>✅ Recommended Actions:</b> {advisory["actions"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**📊 Top model drivers (permutation importance):** "
            + ", ".join(f"`{f}`" for f in top_feats)
        )

        # Session history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "State": sel_state, "City": sel_city,
            "Pollutant": poll_id,
            "Missing": "Yes" if poll_missing else "No",
            "AQI":    aqi_val, "Category": bucket,
            "Likely Lo": lo,   "Likely Hi": hi,
            "Date": str(sel_date), "Hour": sel_hour,
        })

    if st.session_state.get("history"):
        st.markdown("---")
        st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv = hist_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download History CSV", csv, "prediction_history.csv", "text/csv")


# ── Tab 2: Data Explorer ──────────────────────────────────────────────────────
def tab_explorer(df: pd.DataFrame) -> None:
    """Data Explorer tab with KPI cards, filters, and interactive charts."""
    # KPI cards
    most_polluted_state = df.groupby("state")["AQI"].mean().idxmax()
    kpis = [
        ("Rows",         f"{len(df):,}"),
        ("States",       str(df["state"].nunique())),
        ("Cities",       str(df["city"].nunique())),
        ("Mean AQI",     f"{df['AQI'].mean():.1f}"),
        ("Most Polluted State", most_polluted_state),
    ]
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Filters
    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_states = st.multiselect("State", sorted(df["state"].unique()),
                                    key="expl_state")
    with fc2:
        sel_polls  = st.multiselect("Pollutant", sorted(df["pollutant_id"].unique()),
                                    key="expl_poll")
    with fc3:
        sel_seasons = st.multiselect("Season", sorted(df["Season"].unique()),
                                     key="expl_season")

    filt = df.copy()
    if sel_states:
        filt = filt[filt["state"].isin(sel_states)]
    if sel_polls:
        filt = filt[filt["pollutant_id"].isin(sel_polls)]
    if sel_seasons:
        filt = filt[filt["Season"].isin(sel_seasons)]

    if filt.empty:
        st.warning("No data matches the current filters.")
        return

    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(filt, x="AQI", nbins=50, title="AQI Distribution",
                                color_discrete_sequence=["#0f4c81"])
        fig_hist.update_layout(bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        bkt_counts = filt["AQI_Bucket"].value_counts().reset_index()
        bkt_counts.columns = ["Category", "Count"]
        fig_donut = px.pie(bkt_counts, names="Category", values="Count",
                           title="AQI Category Distribution",
                           color="Category",
                           color_discrete_map=AQI_COLORS,
                           hole=0.45)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Mean AQI by state – top/bottom 10
    state_aqi = filt.groupby("state")["AQI"].mean().reset_index()
    state_aqi.columns = ["State", "Mean AQI"]
    state_aqi = state_aqi.sort_values("Mean AQI", ascending=False)
    top10   = state_aqi.head(10)
    bottom10 = state_aqi.tail(10).sort_values("Mean AQI")
    c3, c4 = st.columns(2)
    with c3:
        fig_top = px.bar(top10, x="Mean AQI", y="State", orientation="h",
                         title="Top 10 Most Polluted States",
                         color_discrete_sequence=["#FF0000"])
        st.plotly_chart(fig_top, use_container_width=True)
    with c4:
        fig_bot = px.bar(bottom10, x="Mean AQI", y="State", orientation="h",
                         title="Top 10 Cleanest States",
                         color_discrete_sequence=["#00E400"])
        st.plotly_chart(fig_bot, use_container_width=True)

    # AQI by season and by pollutant
    c5, c6 = st.columns(2)
    with c5:
        season_aqi = filt.groupby("Season")["AQI"].mean().reset_index()
        fig_seas = px.bar(season_aqi, x="Season", y="AQI",
                          title="Mean AQI by Season",
                          color_discrete_sequence=["#0f4c81"])
        st.plotly_chart(fig_seas, use_container_width=True)
    with c6:
        poll_aqi = filt.groupby("pollutant_id")["AQI"].mean().sort_values(ascending=False).reset_index()
        fig_poll = px.bar(poll_aqi, x="pollutant_id", y="AQI",
                          title="Mean AQI by Pollutant",
                          color_discrete_sequence=["#7c5cd8"])
        st.plotly_chart(fig_poll, use_container_width=True)

    # Monthly trend
    month_aqi = filt.groupby("Month")["AQI"].mean().reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_aqi["Month_Name"] = month_aqi["Month"].map(month_names)
    fig_month = px.line(month_aqi, x="Month_Name", y="AQI",
                        title="Mean AQI by Month", markers=True,
                        color_discrete_sequence=["#0f4c81"])
    st.plotly_chart(fig_month, use_container_width=True)

    # Correlation heatmap
    num_cols_corr = ["AQI", "pollutant_min", "pollutant_max", "pollutant_avg",
                     "Temperature_C", "Humidity_%", "Wind_Speed_kmh"]
    corr_matrix = filt[num_cols_corr].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f",
                         title="Correlation Heatmap",
                         color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Station map
    st.markdown('<div class="section-title">India Station Map (Mean AQI)</div>',
                unsafe_allow_html=True)
    map_data = (filt.groupby(["station", "state", "city", "latitude", "longitude"])
                ["AQI"].mean().reset_index())
    map_data["AQI"] = map_data["AQI"].round(1)
    try:
        fig_map = px.scatter_map(
            map_data, lat="latitude", lon="longitude",
            color="AQI", size="AQI",
            hover_name="station",
            hover_data={"state": True, "city": True, "AQI": True,
                        "latitude": False, "longitude": False},
            color_continuous_scale=[
                [0.0,  "#00E400"], [0.1,  "#FFFF00"],
                [0.3,  "#FF7E00"], [0.4,  "#FF0000"],
                [0.6,  "#8F3F97"], [1.0,  "#7E0023"],
            ],
            range_color=[0, 350],
            zoom=4, height=500,
            map_style="open-street-map",
            title="Air Quality Monitoring Stations",
        )
    except (AttributeError, TypeError):
        fig_map = px.scatter_mapbox(
            map_data, lat="latitude", lon="longitude",
            color="AQI", size="AQI",
            hover_name="station",
            hover_data={"state": True, "city": True, "AQI": True,
                        "latitude": False, "longitude": False},
            color_continuous_scale=[
                [0.0,  "#00E400"], [0.1,  "#FFFF00"],
                [0.3,  "#FF7E00"], [0.4,  "#FF0000"],
                [0.6,  "#8F3F97"], [1.0,  "#7E0023"],
            ],
            range_color=[0, 350],
            zoom=4, height=500,
            mapbox_style="open-street-map",
            title="Air Quality Monitoring Stations",
        )
    st.plotly_chart(fig_map, use_container_width=True)

    # Raw data preview
    st.markdown('<div class="section-title">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(200), use_container_width=True)
    csv = filt.to_csv(index=False).encode()
    st.download_button("⬇️ Download Filtered CSV", csv, "filtered_aqi_data.csv", "text/csv")


# ── Tab 3: Model Performance ──────────────────────────────────────────────────
def tab_performance(model_artefacts: dict) -> None:
    """Model Performance tab with full evaluation suite."""
    ma       = model_artefacts
    y_test   = ma["y_test"]
    y_pred   = ma["y_pred"]
    comp_df  = ma["comparison_df"]
    perm_imp = ma["perm_imp_df"]

    # Model comparison table
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.dataframe(comp_df, use_container_width=True)

    # Headline metrics
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, (label, value) in zip(
        [mc1, mc2, mc3, mc4, mc5],
        [("R²",             f"{ma['r2']:.4f}"),
         ("MAE",            f"{ma['mae']:.2f}"),
         ("RMSE",           f"{ma['rmse']:.2f}"),
         ("Bucket Acc.",    f"{ma['bucket_acc']*100:.1f}%"),
         ("Within-1 Acc.",  f"{ma['within1_acc']*100:.1f}%")],
    ):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Actual vs Predicted scatter
    st.markdown('<div class="section-title">Actual vs Predicted AQI</div>', unsafe_allow_html=True)
    fig_avp = px.scatter(x=y_test, y=y_pred, opacity=0.5,
                         labels={"x": "Actual AQI", "y": "Predicted AQI"},
                         title="Actual vs Predicted AQI")
    fig_avp.add_shape(type="line", x0=0, y0=0, x1=500, y1=500,
                      line=dict(color="red", dash="dash"))
    fig_avp.update_layout(height=400)
    st.plotly_chart(fig_avp, use_container_width=True)

    # Residuals
    c1, c2 = st.columns(2)
    with c1:
        fig_res = px.histogram(ma["residuals"], nbins=60,
                               title="Residuals Distribution (Predicted − Actual)",
                               labels={"value": "Residual"},
                               color_discrete_sequence=["#7c5cd8"])
        fig_res.update_layout(showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)
    with c2:
        # Feature importance
        fig_fi = px.bar(
            perm_imp.head(12),
            x="importance", y="feature", orientation="h",
            error_x="std",
            title="Permutation Feature Importance (Top 12)",
            color_discrete_sequence=["#0f4c81"],
        )
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrix
    st.markdown('<div class="section-title">Confusion Matrix (Predicted Category)</div>',
                unsafe_allow_html=True)
    cm_df = get_confusion_matrix_df(y_test, y_pred)
    fig_cm = px.imshow(cm_df, text_auto=True,
                       color_continuous_scale="Blues",
                       title="Confusion Matrix",
                       labels={"x": "Predicted", "y": "Actual"})
    fig_cm.update_layout(height=420)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    st.markdown('<div class="section-title">Per-Category Classification Report</div>',
                unsafe_allow_html=True)
    cr_df = get_classification_report_df(y_test, y_pred)
    st.dataframe(cr_df, use_container_width=True)

    # Plain-English interpretation
    st.markdown('<div class="section-title">Interpretation & Key Findings</div>',
                unsafe_allow_html=True)
    top_feat_name = perm_imp.iloc[0]["feature"]
    st.info(
        f"**Best model:** {ma['best_name']}  |  "
        f"**Test R²:** {ma['r2']}  |  **MAE:** {ma['mae']} AQI pts  |  **RMSE:** {ma['rmse']} AQI pts\n\n"
        f"**Dominant feature:** `{top_feat_name}` (permutation importance). "
        f"`pollutant_avg` has a Pearson correlation of ~0.92 with AQI.\n\n"
        f"**Weather is near-noise:** Temperature, Humidity, and Wind Speed each have |r| < 0.02 with AQI "
        f"and near-zero permutation importance. They are kept as inputs for completeness but should not be "
        f"interpreted as AQI drivers.\n\n"
        f"**Missing pollutant flag:** All 357 rows with missing pollutant readings are Hazardous (AQI 301–499). "
        f"The `pollutant_missing` binary feature captures this signal. "
        f"When you tick 'Pollutant reading not reported' in the Predict tab, the model correctly pushes "
        f"predictions toward the Hazardous range.\n\n"
        f"**No time-based split:** The dataset spans 2025–2027 with 3,523 rows sharing a single "
        f"2026-07-01 23:00 timestamp, indicating augmented/synthetic enrichment. "
        f"A stratified random split is used instead of chronological ordering."
    )


# ── Tab 4: Batch Prediction ───────────────────────────────────────────────────
def tab_batch(df: pd.DataFrame, model_artefacts: dict) -> None:
    """Batch Prediction tab: upload CSV, predict, download results."""
    pipeline = model_artefacts["best_pipeline"]
    pred_lo  = model_artefacts["pred_lo"]
    pred_hi  = model_artefacts["pred_hi"]

    required_input_cols = [
        "pollutant_id", "state", "Season",
        "pollutant_min", "pollutant_max", "pollutant_avg",
        "pollutant_missing",
        "Temperature_C", "Humidity_%", "Wind_Speed_kmh",
        "Month", "Hour", "Day_of_Week",
        "latitude", "longitude",
    ]

    # Template download
    template_df = pd.DataFrame(columns=required_input_cols)
    sample_row = {
        "pollutant_id": "PM2.5", "state": "Delhi", "Season": "Winter",
        "pollutant_min": 80.0,   "pollutant_max": 200.0, "pollutant_avg": 140.0,
        "pollutant_missing": 0,
        "Temperature_C": 22.0,   "Humidity_%": 55.0, "Wind_Speed_kmh": 8.0,
        "Month": 12, "Hour": 9,  "Day_of_Week": 0,
        "latitude": 28.6448,     "longitude": 77.2167,
    }
    template_df = pd.DataFrame([sample_row])
    template_csv = template_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV Template", template_csv,
                       "batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_upload")
    if uploaded is None:
        st.info("Upload a CSV file with the required input columns to run batch predictions.")
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return

    if batch_df.empty:
        st.error("Uploaded file is empty.")
        return

    missing_cols = set(required_input_cols) - set(batch_df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # Validation
    errors: list[str] = []
    valid_polls   = set(df["pollutant_id"].unique())
    valid_states  = set(df["state"].unique())
    valid_seasons = {"Winter", "Spring", "Summer", "Autumn"}

    for i, row in batch_df.iterrows():
        row_num = int(i) + 1  # type: ignore[arg-type]
        if row["pollutant_id"] not in valid_polls:
            errors.append(f"Row {row_num}: unknown pollutant_id '{row['pollutant_id']}'")
        if row["state"] not in valid_states:
            errors.append(f"Row {row_num}: unknown state '{row['state']}'")
        if row["Season"] not in valid_seasons:
            errors.append(f"Row {row_num}: invalid Season '{row['Season']}'")
        if not bool(row["pollutant_missing"]):
            try:
                if not (float(row["pollutant_min"]) <= float(row["pollutant_avg"]) <= float(row["pollutant_max"])):
                    errors.append(f"Row {row_num}: pollutant_min ≤ avg ≤ max violated")
            except (ValueError, TypeError):
                errors.append(f"Row {row_num}: non-numeric pollutant value")

    if errors:
        for err in errors[:10]:
            st.error(err)
        if len(errors) > 10:
            st.error(f"... and {len(errors)-10} more errors.")
        return

    # Feature engineering and prediction
    batch_df = batch_df.copy()
    batch_df["pollutant_range"] = batch_df["pollutant_max"] - batch_df["pollutant_min"]
    batch_df.loc[batch_df["pollutant_missing"] == 1, ["pollutant_min", "pollutant_max", "pollutant_avg", "pollutant_range"]] = np.nan
    batch_df["Month_sin"] = np.sin(2 * np.pi * batch_df["Month"] / 12)
    batch_df["Month_cos"] = np.cos(2 * np.pi * batch_df["Month"] / 12)
    batch_df["Hour_sin"]  = np.sin(2 * np.pi * batch_df["Hour"]  / 24)
    batch_df["Hour_cos"]  = np.cos(2 * np.pi * batch_df["Hour"]  / 24)

    try:
        preds = pipeline.predict(batch_df[ALL_FEATURES])
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        return

    preds = np.clip(preds, 0, 499)
    batch_df["Predicted_AQI"]      = preds.round().astype(int)
    batch_df["AQI_Category"]       = [aqi_to_bucket(v) for v in batch_df["Predicted_AQI"]]
    batch_df["Likely_Low"]         = (preds + pred_lo).clip(0, 499).round().astype(int)
    batch_df["Likely_High"]        = (preds + pred_hi).clip(0, 499).round().astype(int)

    st.success(f"✅ Predicted {len(batch_df)} rows.")
    st.dataframe(batch_df[["pollutant_id", "state", "Season",
                            "Predicted_AQI", "AQI_Category",
                            "Likely_Low", "Likely_High"]], use_container_width=True)
    result_csv = batch_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Results CSV", result_csv,
                       "batch_predictions.csv", "text/csv")


# ── Tab 5: About ───────────────────────────────────────────────────────────────
def tab_about() -> None:
    """About tab: project info, methodology, limitations and disclaimer."""
    st.markdown("""
## 🌍 About this Project

**AI AQI Predictor** is an educational machine-learning application that predicts the Air Quality Index
(AQI) at Indian monitoring stations using a scikit-learn ensemble regression model.

---
### 🔬 Methodology

| Step | Details |
|------|---------|
| Dataset | 10,521 rows × 22 columns; 30 Indian states, 268 cities, 503 stations; 7 pollutants |
| Target | AQI (integer 0–499); US-EPA-style breakpoints |
| Split | 80/20 stratified by AQI_Bucket; random_state=42 |
| Validation | 5-fold CV on training set; best model by CV RMSE |
| Models | Ridge (baseline), Random Forest, Extra Trees, Gradient Boosting |
| Encoding | Cyclic sin/cos for Month and Hour; OneHot for pollutant_id, state, Season |
| Imputation | Median imputation inside the pipeline for pollutant_min/max/avg |

---
### ⚠️ Data Traps & How They Were Handled

| Trap | Explanation | Handling |
|------|-------------|---------|
| **Leakage** | `AQI_Bucket` is derived from AQI | Never used as a feature; category derived from predicted AQI |
| **Missing ≠ Random** | All 357 missing pollutant rows are Hazardous | `pollutant_missing` binary flag added; median imputation inside pipeline |
| **Weather is near-noise** | Temperature, Humidity, Wind Speed have |r| < 0.02 with AQI | Kept as inputs; honestly flagged in Model Performance tab |
| **Timestamps** | Dataset spans 2025–2027 with 3,523 rows sharing 2026-07-01; appears enriched/augmented | No time-based split; Year/Day/last_update dropped |
| **High cardinality** | Station (503), City (268) are too granular | Dropped from model; State + lat/lon used instead |

---
### 🔲 Pollutant "Not Reported" Checkbox
When you check **"Pollutant reading not reported"** in the Predict tab:
- The three pollutant reading fields (min, avg, max) are treated as `NaN`.
- The `pollutant_missing` feature is set to **1**.
- The model imputes median values for the raw fields but the missing flag remains 1, pushing predictions
  toward the Hazardous range — consistent with the dataset pattern where all 357 missing readings
  are Hazardous.

---
### ⚠️ Limitations
- Dataset appears enriched/augmented with future dates (2026–2027); not a live operational feed.
- No time-series forecasting; each row is an independent snapshot.
- Weather variables (temperature, humidity, wind) have negligible model importance.
- City and station granularity are dropped from the model to avoid high-cardinality issues.
- Prediction intervals are empirical (hold-out residual percentiles), not formal Bayesian intervals.

---
### 👤 Author
**Ganesh Shah** — Developed using **IBM Bob**

---
### ⚖️ Disclaimer
> This application is an **educational tool only** and is **not** an official AQI source,
> regulatory product, or medical device. Do not use these predictions to make health or
> safety decisions. Always consult official government AQI sources and medical professionals.
    """)


# =============================================================================
# SECTION 9 : MAIN FUNCTION
# =============================================================================
def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="AI AQI Predictor",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-header">'
        '<h1>🌍 AI AQI Predictor</h1>'
        '<p>India Air Quality Index Prediction · Ganesh Shah · Developed using IBM Bob</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Data loading ─────────────────────────────────────────────────────────
    uploaded_file = None
    if not DATA_PATH.exists():
        st.sidebar.warning("⚠️ Dataset not found at expected path.")
        uploaded_file = st.sidebar.file_uploader(
            "Upload enriched_aqi_dataset.csv", type=["csv"]
        )
        if uploaded_file is None:
            st.error(
                f"Dataset not found: `{DATA_PATH}`\n\n"
                "Please place `enriched_aqi_dataset.csv` next to the script, "
                "or upload it via the sidebar."
            )
            st.stop()

    try:
        df = load_data(uploaded_file)
    except ValueError as exc:
        st.error(f"Dataset validation failed: {exc}")
        st.stop()

    if df.empty:
        st.error("Dataset could not be loaded or is empty.")
        st.stop()

    # ── Model training (cached) ──────────────────────────────────────────────
    # Show a visible loading placeholder so the page isn't blank while training
    _loading = st.empty()
    _loading.info(
        "⚙️ **Training models on first load — please wait ~30 seconds…**\n\n"
        "Comparing Ridge · Random Forest · Extra Trees · Gradient Boosting "
        "with 5-fold cross-validation. This runs once per session and is cached."
    )
    with st.spinner("Training models… (first run only)"):
        try:
            model_artefacts = train_model(df)
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()
    _loading.empty()   # Clear the loading message once done

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📊 Dataset Summary")
    st.sidebar.markdown(f"**Rows:** {len(df):,}")
    st.sidebar.markdown(f"**States:** {df['state'].nunique()}")
    st.sidebar.markdown(f"**Cities:** {df['city'].nunique()}")
    st.sidebar.markdown(f"**Stations:** {df['station'].nunique()}")
    st.sidebar.markdown(f"**AQI range:** {df['AQI'].min()} – {df['AQI'].max()}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model")
    st.sidebar.markdown(f"**Best Model:** {model_artefacts['best_name']}")
    st.sidebar.markdown(f"**Test R²:** {model_artefacts['r2']}")
    st.sidebar.markdown(f"**Test MAE:** {model_artefacts['mae']} AQI pts")
    st.sidebar.markdown(f"**Test RMSE:** {model_artefacts['rmse']} AQI pts")
    st.sidebar.markdown(f"**Bucket Accuracy:** {model_artefacts['bucket_acc']*100:.1f}%")
    st.sidebar.markdown(f"**Within-1 Acc.:** {model_artefacts['within1_acc']*100:.1f}%")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🔍 Predict",
        "📊 Data Explorer",
        "📈 Model Performance",
        "⚡ Batch Prediction",
        "ℹ️ About",
    ])

    with tabs[0]:
        tab_predict(df, model_artefacts)
    with tabs[1]:
        tab_explorer(df)
    with tabs[2]:
        tab_performance(model_artefacts)
    with tabs[3]:
        tab_batch(df, model_artefacts)
    with tabs[4]:
        tab_about()


if __name__ == "__main__":
    main()
