import json
import h5py
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Image Classifier", page_icon="🔬")
st.title("Skin Cancer Image Classifier")
st.write("Upload an image to view the model prediction.")

MODEL_PATH = "Skin_Cancer_Model.h5"

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with h5py.File(MODEL_PATH, "r") as f:
        raw = f.attrs.get("class_names")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        names = json.loads(raw) if raw is not None else None
    if not names:
        names = [f"Class {i}" for i in range(int(model.output_shape[-1]))]
    return model, names

try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error("Model load nahi hua. H5 file ko app.py ke saath same folder mein rakhein.")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict", type="primary"):
        try:
            shape = model.input_shape
            height = int(shape[1]) if shape[1] else 224
            width = int(shape[2]) if shape[2] else 224
            image = image.resize((width, height))
            array = np.asarray(image, dtype=np.float32)
            array = np.expand_dims(array, axis=0)  # pixels remain 0–255, matching training
            scores = model.predict(array, verbose=0)[0]
            index = int(np.argmax(scores))
            label = class_names[index] if index < len(class_names) else f"Class {index}"
            st.success(f"Predicted class: {label}")
            st.metric("Model confidence", f"{float(scores[index]) * 100:.2f}%")
            st.subheader("Class probabilities")
            st.bar_chart({
                class_names[i] if i < len(class_names) else f"Class {i}": float(scores[i])
                for i in range(len(scores))
            })
        except Exception as e:
            st.error("Prediction error:")
            st.exception(e)

st.warning("Educational project only. This is not a medical diagnosis. Consult a qualified dermatologist for medical concerns.")
