import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load model
model = keras.models.load_model('best_cnn_model.h5')

# Class names
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# App title
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess
    img = image.resize((32, 32))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"Prediction : **{predicted_class}**")
    st.info(f"Confidence : **{confidence:.2f}%**")