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
)

# ── MODEL LOADING (FULL DEBUG VERSION) ─────────────────────────
@st.cache_resource
def load_model():
    model_path = "mobilenetv2_transfer_model.h5"

    # Show files in directory (DEBUG)
    st.write("📂 Files in current directory:", os.listdir())

    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file NOT FOUND: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False   # 🔥 fixes version issues
        )
        st.success("✅ Model loaded successfully")
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None


model = load_model()

# ── Background Image ──────────────────────────────────────────
@st.cache_data
def get_bg():
    path = pathlib.Path("bg_cnn.png")
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""

bg = get_bg()
bg_css = f"url('data:image/png;base64,{bg}')" if bg else "none"

# ── Simple UI Styling ─────────────────────────────────────────
st.markdown(f"""
<style>
body, .stApp {{
    background-color: #060b16;
    color: white;
}}
.stApp {{
    background-image: {bg_css};
    background-size: cover;
}}
.card {{
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
}}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
now = datetime.datetime.now()

st.markdown(f"""
<div style="display:flex; justify-content:space-between">
    <div>
        <h2>👁️ Vision AI</h2>
        <small>Accident Detection System</small>
    </div>
    <div>
        <b>{now.strftime("%H:%M:%S")}</b><br>
        <small>{now.strftime("%b %d, %Y")}</small>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────
col1, col2 = st.columns([2,1])

# ── LEFT: Upload ─────────────────────────────────────────────
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CCTV Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image to begin")

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Prediction ─────────────────────────────────────────
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Detection Result")

    if uploaded_file:

        if model is None:
            st.error("🚨 Model NOT loaded. Fix model file first.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(1)

                try:
                    # Preprocess
                    img = image.resize((128, 128))
                    arr = img_to_array(img)
                    arr = np.expand_dims(arr, axis=0) / 255.0

                    # Predict
                    prob = model.predict(arr, verbose=0)[0][0]

                    # ⚠️ Adjust if needed
                    accident = prob < 0.5  

                    confidence = (1 - prob) * 100 if accident else prob * 100

                    # ── Output ──
                    if accident:
                        st.error(f"⚠️ ACCIDENT DETECTED")
                        st.write(f"Confidence: **{confidence:.2f}%**")
                    else:
                        st.success("✅ NO ACCIDENT")
                        st.write(f"Confidence: **{confidence:.2f}%**")

                    # Debug info
                    st.caption(f"Raw Output: {prob:.4f}")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    else:
        st.warning("Upload image to see result")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Vision AI · CNN-based Accident Detection")
