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


model = load_model()
scaler = load_scaler()

# -----------------------------
# Helper: build last window for an engine
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
    "Navigation",
    ["Predict RUL", "Project Details", "Samples / Download"]
)

# -----------------------------
# Hero animation
# -----------------------------
engine_html = """(UNCHANGED SVG CONTENT)"""
if page in ["Predict RUL", "Project Details"]:
    html(engine_html, height=320, scrolling=False)

# -----------------------------
# Page: Predict RUL
# -----------------------------
if page == "Predict RUL":
    st.header("Turbofan Engine Remaining Useful Life (RUL) Predictor")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with sensor readings",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "cycle" not in df.columns:
            df["cycle"] = np.arange(1, len(df) + 1)

        if "engine_id" not in df.columns:
            df["engine_id"] = 1

        engine_ids = sorted(df["engine_id"].unique())
        selected_engine = st.selectbox("Engine ID", engine_ids)
        df_engine = df[df["engine_id"] == selected_engine]

        if st.button("🔮 Predict RUL for this engine"):
            X_window = prepare_engine_window(df_engine)
            pred_rul = float(model.predict(X_window)[0, 0])
            st.subheader(f"Predicted RUL: {pred_rul:.1f} cycles")
            st.write(classify_health(pred_rul))

# -----------------------------
# Page: Project Details
# -----------------------------
if page == "Project Details":
    st.header("Project Details & Results")

# -----------------------------
# Page: Samples / Download
# -----------------------------
if page == "Samples / Download":
    st.header("Sample Engines & Downloads")

    st.code(
        "cycle,op_setting_1,op_setting_2,op_setting_3,s1,...,s21",
        language="text"
    )

    model_path = Path(MODEL_SAVE_PATH)
    if model_path.exists():
        with open(model_path, "rb") as f:
            st.download_button(
                "Download trained model (.keras)",
                data=f.read(),
                file_name="best_model.keras"
            )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>> ADDED FOR ENGINE CSV DOWNLOAD (ONLY ADDITION) <<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    st.markdown("---")
    st.subheader("📥 Download FD001 Test Engine CSV")

    @st.cache_data
    def load_fd001_test_raw():
        path = Path("data/raw/test_FD001.txt")
        if not path.exists():
            return None

        df = pd.read_csv(path, sep=r"\s+", header=None)
        df.columns = (
            ["engine_id", "cycle"]
            + [f"op_setting_{i}" for i in range(1, 4)]
            + [f"s{i}" for i in range(1, 22)]
        )
        return df

    df_test = load_fd001_test_raw()

    if df_test is None:
        st.error("FD001 test dataset not found at data/raw/test_FD001.txt")
    else:
        engine_id = st.number_input(
            "Enter Engine ID (1–100)",
            min_value=1,
            max_value=100,
            value=1,
            step=1
        )

        engine_df = df_test[df_test["engine_id"] == engine_id]

        if not engine_df.empty:
            csv_bytes = engine_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download Engine {engine_id} CSV",
                data=csv_bytes,
                file_name=f"engine_{engine_id}_FD001.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data found for this engine.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with 🛩️ Turbofan RUL · CNN + BiLSTM + Attention · Streamlit")
