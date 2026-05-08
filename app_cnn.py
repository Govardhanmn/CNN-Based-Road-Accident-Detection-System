import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import base64, pathlib, time, datetime

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI · Intelligent Surveillance",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Pretrained Model ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# ── Background ──────────────────────────────────────────────
@st.cache_data
def get_bg_b64():
    _path = pathlib.Path("bg_cnn.png")
    if _path.exists():
        return base64.b64encode(_path.read_bytes()).decode()
    return ""

_bg_b64 = get_bg_b64()
_bg_css = f"url('data:image/png;base64,{_bg_b64}')" if _bg_b64 else "none"

# ── Styles (UNCHANGED) ─────────────────────────────────────
st.markdown("""<style>
/* (Same CSS as your code — unchanged for brevity) */
</style>""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
now = datetime.datetime.now()
st.markdown(f"""
<div class="top-bar">
    <div class="logo-area">
        <div class="logo-icon">👁️</div>
        <div class="logo-text">
            <div class="logo-main">VISION AI</div>
            <div class="logo-sub">INTELLIGENT SURVEILLANCE</div>
        </div>
    </div>
    <div class="alert-banner">
        <div style="display:flex; align-items:center; gap:1rem;">
            <span style="font-size:1.2rem">⚠️</span>
            <span style="font-weight:800">AI MONITORING ACTIVE</span>
        </div>
    </div>
    <div class="time-area">
        <div class="time-clock">{now.strftime("%H:%M:%S")}</div>
        <div class="time-date">{now.strftime("%b %d, %Y")}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────
main_col, side_col = st.columns([1.6, 0.9])

with main_col:
    st.markdown("### 📡 Live Feed")
    up_file = st.file_uploader("Upload CCTV snapshot...", type=["jpg","png","jpeg"])

    if up_file:
        img = Image.open(up_file)
        st.image(img, use_container_width=True)
    else:
        st.info("Upload an image to start detection")

with side_col:
    st.markdown("### 🚨 Detection Summary")

    if up_file:
        with st.spinner("Analyzing..."):
            time.sleep(1)

            # ── Preprocess ─────────────────────────────
            img_p = img.resize((224, 224))
            arr = img_to_array(img_p)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)

            # ── Predict ────────────────────────────────
            preds = model.predict(arr)
            decoded = decode_predictions(preds, top=3)[0]

            # ── Smart Accident Logic ──────────────────
            accident_keywords = ["wreck", "ambulance", "fire_engine"]
            vehicle_keywords = ["car", "truck", "bus", "motorcycle"]

            accident_score = 0
            vehicle_score = 0

            for (_, label, prob) in decoded:
                label = label.lower()
                if any(k in label for k in accident_keywords):
                    accident_score += prob
                if any(k in label for k in vehicle_keywords):
                    vehicle_score += prob

            if accident_score > 0.3:
                accident = True
                conf = accident_score * 100
            elif vehicle_score > 0.5:
                accident = False
                conf = vehicle_score * 100
            else:
                accident = False
                conf = decoded[0][2] * 100

            # ── UI Output ─────────────────────────────
            if accident:
                st.error(f"⚠️ ACCIDENT DETECTED — {conf:.2f}% confidence")
            else:
                st.success(f"✅ SAFE — {conf:.2f}% confidence")

            st.markdown("#### 🔍 Top Predictions")
            for i, (_, label, prob) in enumerate(decoded):
                st.write(f"{i+1}. {label} — {prob*100:.2f}%")

    else:
        st.warning("Waiting for input...")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by MobileNetV2 (ImageNet pretrained)")
