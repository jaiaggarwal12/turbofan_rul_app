import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

from src.model import AttentionLayer
from src.config import FEATURE_COLS, SEQ_LEN, MODEL_SAVE_PATH


# -----------------------------
# Streamlit basic config
# -----------------------------
st.set_page_config(
    page_title="Turbofan Engine RUL Predictor",
    layout="wide",
    page_icon="🛩️",
)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False
    )


@st.cache_resource
def load_scaler():
    scaler_path = Path("data/processed/scaler.pkl")
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


model = load_model()
scaler = load_scaler()


# -----------------------------
# Helper functions
# -----------------------------
def prepare_engine_window(df_engine: pd.DataFrame) -> np.ndarray:
    df_engine = df_engine.sort_values("cycle").copy()

    if scaler is not None:
        features = scaler.transform(df_engine[FEATURE_COLS])
    else:
        features = df_engine[FEATURE_COLS].values

    if len(features) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(features), features.shape[1]))
        window = np.vstack([pad, features])
    else:
        window = features[-SEQ_LEN:]

    return window[np.newaxis, ...]


def classify_health(rul: float) -> str:
    if rul > 100:
        return "🟢 Healthy"
    elif rul > 50:
        return "🟡 Moderate Wear"
    elif rul > 20:
        return "🟠 Warning"
    else:
        return "🔴 Critical – Immediate Maintenance"


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Navigation",
    ["Predict RUL", "Project Details", "Samples / Download"],
    label_visibility="collapsed"
)


# -----------------------------
# Engine animation
# -----------------------------
engine_html = """
<div style="width:100%; height:320px; border-radius:14px; background:#02030a;
display:flex; align-items:center; justify-content:center;">
<svg width="260" height="260" viewBox="0 0 260 260">
<g style="transform-origin:130px 130px; animation:spin 3s linear infinite">
<circle cx="130" cy="130" r="80" fill="#1b2540"/>
</g>
</svg>
</div>
<style>
@keyframes spin { 100% { transform: rotate(360deg); } }
</style>
"""

if page in ["Predict RUL", "Project Details"]:
    html(engine_html, height=320)


# -----------------------------
# Predict RUL page
# -----------------------------
if page == "Predict RUL":
    st.header("Turbofan Engine Remaining Useful Life (RUL) Predictor")

    uploaded_file = st.file_uploader(
        "Upload engine sensor CSV",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "cycle" not in df.columns:
            df["cycle"] = np.arange(1, len(df) + 1)
        if "engine_id" not in df.columns:
            df["engine_id"] = 1

        engine_ids = sorted(df["engine_id"].unique())
        selected_engine = st.selectbox("Select Engine ID", engine_ids)
        df_engine = df[df["engine_id"] == selected_engine]

        st.dataframe(df_engine.head())

        if st.button("🔮 Predict RUL"):
            X = prepare_engine_window(df_engine)
            pred_rul = float(model.predict(X)[0, 0])

            st.subheader(f"Predicted RUL: **{pred_rul:.1f} cycles**")
            st.markdown(f"### Status: {classify_health(pred_rul)}")

            st.progress(min(pred_rul / 150, 1.0))


# -----------------------------
# Project Details page
# -----------------------------
if page == "Project Details":
    st.header("Project Details & Results")

    st.markdown("""
**Dataset:** NASA C-MAPSS FD001  
**Model:** CNN + BiLSTM + Attention  
**Test Performance:**  
- MAE ≈ 11.8  
- RMSE ≈ 15.7  
- R² ≈ 0.86
""")

    figs = {
        "Training Curve": "reports/figures/loss_curve.png",
        "Predicted vs True": "reports/figures/pred_vs_true.png",
        "Error Histogram": "reports/figures/error_histogram.png",
    }

    for title, path in figs.items():
        p = Path(path)
        if p.exists():
            st.subheader(title)
            st.image(str(p), width=1000)


# -----------------------------
# Samples / Download
# -----------------------------
if page == "Samples / Download":
    st.header("Downloads")

    model_path = Path(MODEL_SAVE_PATH)
    if model_path.exists():
        st.download_button(
            "Download trained model",
            data=model_path.read_bytes(),
            file_name="best_model.keras"
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🛩️ Turbofan RUL Prediction · CNN + BiLSTM + Attention · Streamlit")
