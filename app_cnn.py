import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mobilenetv2_transfer_modelh5')
    return model

model = load_model()

st.title("🚨 Accident Detection System")
st.write("Upload an image of a road scene to detect potential accidents.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    
    # Assuming sigmoid output: 0 = Accident, 1 = No Accident (adjust based on your generator mapping)
    if prediction[0][0] > 0.5:
        st.error(f"Prediction: **No Accident** ({100-prediction[0][0]*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: **Accident Detected** ({100-prediction[0][0]*100:.2f}% confidence)")
        st.warning("Immediate attention required!")
