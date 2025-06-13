import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import requests
from streamlit_lottie import st_lottie

# ------------------------------
# Load Lottie animation safely
# ------------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None

# ------------------------------
# Configuration and resources
# ------------------------------
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
logo_url = logo_url = "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg"
lottie_digit = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tll0j4bb.json")

# Load external CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("📍 Navigate", ["Digit Classifier", "About"])

# ------------------------------
# App Header
# ------------------------------
st.image(logo_url, width=100)
st.title("🧠 MNIST Digit Classifier")
st.markdown("Upload a **28x28 grayscale** handwritten digit image (white on black) to predict.")

# ------------------------------
# Digit Classifier Page
# ------------------------------
if page == "Digit Classifier":
    if lottie_digit:
        st_lottie(lottie_digit, height=200)
    else:
        st.info("⚠️ Animation failed to load. Please check your internet connection or use a different Lottie URL.")

    uploaded_file = st.file_uploader("📤 Upload a digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Load and process image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            st.image(image, caption="🖼️ Uploaded Digit", width=150)

            # Preprocess
            image = ImageOps.invert(image.resize((28, 28)))
            img_array = np.array(image) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Load model and predict
            model = tf.keras.models.load_model("mnist_cnn_model.h5")
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)

            st.success(f"✅ Predicted Digit: **{predicted_digit}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# ------------------------------
# About Page
# ------------------------------
elif page == "About":
    st.subheader("📘 About This App")
    st.write("""
        This app uses a trained **Convolutional Neural Network (CNN)** built with TensorFlow to classify 
        handwritten digits from the MNIST dataset.

        Just upload a digit image (28x28 pixels, grayscale) and let the model tell you what it thinks the digit is!

        Built with ❤️ by FALCON GROUP - PLP ACADEMY using **Streamlit**, **TensorFlow**, and **Lottie animations** for interactive UI.
    """)
