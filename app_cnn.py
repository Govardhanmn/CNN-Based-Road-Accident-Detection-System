import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from PIL import Image
import time

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI · Detection System",
    page_icon="👁️",
    layout="wide"
)

# ── FIX for BatchNormalization (IMPORTANT) ───────────────────
class FixedBatchNorm(BatchNormalization):
    @classmethod
    def from_config(cls, config):
        if isinstance(config.get("axis"), list):
            config["axis"] = config["axis"][0]  # FIX: [3] → 3
        return super().from_config(config)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_fixed():
    try:
        model = load_model(
            "mobilenetv2_transfer_model.h5",
            custom_objects={"BatchNormalization": FixedBatchNorm},
            compile=False
        )
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model_fixed()

# ── UI Header ───────────────────────────────────────────────
st.title("🚨 VisionAI Detection System")
st.markdown("Upload an image to analyze using MobileNetV2 Transfer Learning model")

# ── Show Error if Model Failed ──────────────────────────────
if error:
    st.error(f"❌ Model loading failed: {error}")

# ── Image Upload ────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ── Preprocessing ───────────────────────────────────────
    img = image.resize((128, 128))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # ── Prediction ──────────────────────────────────────────
    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            time.sleep(1)

            if model is not None:
                prob = model.predict(arr, verbose=0)[0][0]

                st.subheader("🔍 Result")

                if prob > 0.5:
                    st.error(f"⚠️ Positive Detection ({prob*100:.2f}%)")
                else:
                    st.success(f"✅ Negative Detection ({(1-prob)*100:.2f}%)")

            else:
                st.error("❌ Model not loaded properly. Check error above.")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by MobileNetV2 Transfer Learning")
