import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from PIL import Image
import h5py
import json
import time

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI · Detection System",
    page_icon="👁️",
    layout="wide"
)

# ── Robust Model Loader (FIXED) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_fixed():
    try:
        with h5py.File("mobilenetv2_transfer_model.h5", "r") as f:
            model_config = f.attrs.get("model_config")

            if model_config is None:
                return None, "No model config found in H5 file"

            model_config = json.loads(model_config)

            # 🔥 Fix BatchNormalization axis issue everywhere
            def fix_bn(config):
                if isinstance(config, dict):
                    if config.get("class_name") == "BatchNormalization":
                        axis = config["config"].get("axis")
                        if isinstance(axis, list):
                            config["config"]["axis"] = axis[0]

                    for v in config.values():
                        fix_bn(v)

                elif isinstance(config, list):
                    for item in config:
                        fix_bn(item)

            fix_bn(model_config)

            # Rebuild model
            model = model_from_json(json.dumps(model_config))

            # Load weights
            model.load_weights("mobilenetv2_transfer_model.h5")

        return model, None

    except Exception as e:
        return None, str(e)

model, error = load_model_fixed()

# ── UI Header ───────────────────────────────────────────────
st.title("🚨 VisionAI Detection System")
st.markdown("Upload an image to analyze using MobileNetV2 Transfer Learning model")

# ── Show Model Load Error ───────────────────────────────────
if error:
    st.error(f"❌ Model loading failed: {error}")

# ── Upload Image ────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ── Preprocess ──────────────────────────────────────────
    img = image.resize((128, 128))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # ── Predict ─────────────────────────────────────────────
    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            time.sleep(1)

            if model is not None:
                try:
                    prob = model.predict(arr, verbose=0)[0][0]

                    st.subheader("🔍 Result")

                    if prob > 0.5:
                        st.error(f"⚠️ Positive Detection ({prob*100:.2f}%)")
                    else:
                        st.success(f"✅ Negative Detection ({(1-prob)*100:.2f}%)")

                except Exception as pred_error:
                    st.error(f"❌ Prediction failed: {pred_error}")

            else:
                st.error("❌ Model not loaded properly. Check error above.")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by MobileNetV2 Transfer Learning")
