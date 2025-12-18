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
    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    return model


@st.cache_resource
def load_scaler():
    scaler_path = Path("data/processed/scaler.pkl")
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_fd001_test_raw():
    path = Path("data/raw/test_FD001.txt")
    if not path.exists():
        return None

    df = pd.read_csv(path, sep=" ", header=None)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    cols = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )
    df.columns = cols
    return df


model = load_model()
scaler = load_scaler()

# -----------------------------
# Helper functions
# -----------------------------
def prepare_engine_window(df_engine: pd.DataFrame) -> np.ndarray:
    df_engine = df_engine.sort_values("cycle").copy()

    features = df_engine[FEATURE_COLS].values
    if scaler is not None:
        features = scaler.transform(df_engine[FEATURE_COLS])

    if len(features) < SEQ_LEN:
        pad_rows = SEQ_LEN - len(features)
        pad = np.zeros((pad_rows, features.shape[1]))
        window = np.vstack([pad, features])
    else:
        window = features[-SEQ_LEN:]

    return window[np.newaxis, ...]


def classify_health(rul_value: float) -> str:
    if rul_value > 100:
        return "🟢 Healthy"
    elif rul_value > 50:
        return "🟡 Moderate Wear"
    elif rul_value > 20:
        return "🟠 Warning"
    else:
        return "🔴 Critical – Immediate Maintenance"


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "",
    ["Predict RUL", "Project Details", "Samples / Download"]
)

# -----------------------------
# Turbofan animation
# -----------------------------
engine_html = """
<div style="width:100%; height:320px; border-radius:14px; overflow:hidden; background:radial-gradient(circle at 20% 20%, #283046 0, #050814 45%, #02030a 100%); display:flex; align-items:center; justify-content:center;">
  <svg width="260" height="260" viewBox="0 0 260 260">
    <defs>
      <radialGradient id="fanGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#e0f4ff"/>
        <stop offset="40%" stop-color="#7aa6ff"/>
        <stop offset="100%" stop-color="#1b2540"/>
      </radialGradient>
    </defs>
    <circle cx="130" cy="130" r="120" fill="none" stroke="#6c7aa8" stroke-width="6"/>
    <g class="fan-blades">
      <circle cx="130" cy="130" r="80" fill="url(#fanGrad)" opacity="0.35"/>
      <polygon points="130,35 140,105 120,105" fill="#e3edf9"/>
      <polygon points="225,130 155,140 155,120" fill="#e3edf9"/>
      <polygon points="130,225 120,155 140,155" fill="#e3edf9"/>
      <polygon points="35,130 105,120 105,140" fill="#e3edf9"/>
    </g>
  </svg>
</div>
<style>
@keyframes spinFan {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.fan-blades {
  transform-origin: 130px 130px;
  animation: spinFan 2.8s linear infinite;
}
</style>
"""

if page in ["Predict RUL", "Project Details"]:
    html(engine_html, height=320)

# -----------------------------
# Page: Predict RUL
# -----------------------------
if page == "Predict RUL":
    st.header("Turbofan Engine Remaining Useful Life (RUL) Predictor")

    uploaded_file = st.file_uploader("Upload engine CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "cycle" not in df.columns:
            df["cycle"] = np.arange(1, len(df) + 1)
        if "engine_id" not in df.columns:
            df["engine_id"] = 1

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        engine_ids = sorted(df["engine_id"].unique())
        selected_engine = st.selectbox("Engine ID", engine_ids)
        df_engine = df[df["engine_id"] == selected_engine]

        st.dataframe(df_engine.head())

        if st.button("🔮 Predict RUL"):
            X = prepare_engine_window(df_engine)
            pred_rul = float(model.predict(X)[0, 0])
            health = classify_health(pred_rul)

            st.subheader("Prediction Results")
            st.markdown(f"### Predicted RUL: **{pred_rul:.1f} cycles**")
            st.markdown(f"### Health Status: {health}")

# -----------------------------
# Page: Project Details
# -----------------------------
if page == "Project Details":
    st.header("Project Details & Results")
    st.write(
        """
        **Dataset:** NASA C-MAPSS FD001  
        **Model:** CNN + BiLSTM + Attention  
        **Test Performance:**  
        - MAE ≈ 11.8 cycles  
        - RMSE ≈ 15.7  
        - R² ≈ 0.86  
        """
    )

# -----------------------------
# Page: Samples / Download
# -----------------------------
if page == "Samples / Download":
    st.header("📥 Download FD001 Engine CSV")

    df_test = load_fd001_test_raw()
    if df_test is None:
        st.error("FD001 test dataset not found in data/raw/")
        st.stop()

    engine_id = st.number_input(
        "Enter Engine ID (1–100)",
        min_value=1,
        max_value=100,
        value=1,
        step=1
    )

    engine_df = df_test[df_test["engine_id"] == engine_id]

    if engine_df.empty:
        st.warning(f"No data found for Engine {engine_id}")
    else:
        st.subheader(f"Engine {engine_id} Preview")
        st.dataframe(engine_df.head(10), use_container_width=True)
        st.info(f"Total cycles: {len(engine_df)}")

        csv_bytes = engine_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Engine CSV",
            data=csv_bytes,
            file_name=f"engine_{engine_id}_FD001.csv",
            mime="text/csv"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with 🛩️ Turbofan RUL · CNN + BiLSTM + Attention · Streamlit")
