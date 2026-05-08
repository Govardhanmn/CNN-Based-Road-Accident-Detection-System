import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64, pathlib, time, datetime, os

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI · Intelligent Surveillance",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Model SAFELY ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "best_cnn_model.h5"   # ✅ USE .h5

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# ── Background Image ──────────────────────────────────────────
@st.cache_data
def get_bg_b64():
    path = pathlib.Path("bg_cnn.png")
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""

_bg = get_bg_b64()
_bg_css = f"url('data:image/png;base64,{_bg}')" if _bg else "none"

# ── Minimal Clean UI (keeping your premium feel but stable) ───
st.markdown(f"""
<style>
body, .stApp {{
    background-color: #060b16;
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}}

.stApp {{
    background-image: {_bg_css};
    background-size: cover;
    background-position: center;
}}

.block-container {{
    padding: 2rem;
}}

.card {{
    background: rgba(13, 20, 36, 0.7);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}}

.title {{
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}}

.success {{
    color: #10b981;
    font-weight: bold;
}}

.danger {{
    color: #ef4444;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
now = datetime.datetime.now()

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center">
    <div>
        <h2>👁️ Vision AI</h2>
        <div style="color:#64748b">Intelligent Surveillance</div>
    </div>
    <div style="text-align:right">
        <div>{now.strftime("%H:%M:%S")}</div>
        <div style="color:#64748b">{now.strftime("%b %d, %Y")}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────
col1, col2 = st.columns([2,1])

# ── LEFT: Image Upload ───────────────────────────────────────
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Live Feed / Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CCTV image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image to start detection")

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Prediction ────────────────────────────────────────
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Detection Result</div>', unsafe_allow_html=True)

    if uploaded_file:

        if model is None:
            st.error("🚨 Model not loaded. Fix model file.")
        else:
            with st.spinner("Analyzing image..."):
                time.sleep(1)

                # Preprocess
                img = image.resize((128, 128))
                arr = img_to_array(img)
                arr = np.expand_dims(arr, axis=0) / 255.0

                try:
                    prob = model.predict(arr, verbose=0)[0][0]

                    # ⚠️ IMPORTANT: adjust if needed
                    accident = prob < 0.5  

                    conf = (1 - prob) * 100 if accident else prob * 100

                    # ── OUTPUT ──
                    if accident:
                        st.markdown(f"<div class='danger'>⚠️ ACCIDENT DETECTED</div>", unsafe_allow_html=True)
                        st.write(f"Confidence: **{conf:.2f}%**")
                    else:
                        st.markdown(f"<div class='success'>✅ NO ACCIDENT</div>", unsafe_allow_html=True)
                        st.write(f"Confidence: **{conf:.2f}%**")

                    # Debug (remove later)
                    st.caption(f"Raw model output: {prob:.4f}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    else:
        st.warning("Upload an image to see prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<hr>
<center style="color:#64748b">Vision AI · Accident Detection System</center>
""", unsafe_allow_html=True)
