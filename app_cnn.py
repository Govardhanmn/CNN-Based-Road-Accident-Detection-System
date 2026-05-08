import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI · Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# ── Load Pretrained Model ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

model = load_model()

# ── UI Header ───────────────────────────────────────────────
st.title("🧠 VisionAI Image Classifier")
st.markdown("Upload an image and classify it using pretrained MobileNetV2 (ImageNet)")

# ── Upload Image ────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ── Preprocess ──────────────────────────────────────────
    img = image.resize((224, 224))   # MobileNetV2 expects 224x224
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    # ── Predict ─────────────────────────────────────────────
    if st.button("Analyze Image"):
        with st.spinner("Analyzing with AI..."):
            time.sleep(1)

            preds = model.predict(arr)
            decoded = decode_predictions(preds, top=3)[0]

            st.subheader("🔍 Top Predictions")

            for i, (imagenetID, label, prob) in enumerate(decoded):
                st.write(f"**{i+1}. {label}** — {prob*100:.2f}%")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by MobileNetV2 pretrained on ImageNet")
