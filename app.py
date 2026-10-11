import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")

# Order must match the classes used in training
CLASSES = {
    "Apple___Apple_scab": ("Apple", "Apple scab", False),
    "Apple___healthy": ("Apple", "Healthy", True),
    "Potato___Early_blight": ("Potato", "Early blight", False),
    "Potato___healthy": ("Potato", "Healthy", True),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Tomato mosaic virus", False),
}
names = list(CLASSES)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")


model = load_model()

st.title("🌿 Plant Disease Classifier")
st.caption("Upload a close-up photo of an apple, potato, or tomato leaf.")

file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if file:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    arr = np.array(img.resize((180, 180)), dtype="float32")[None]
    probs = tf.nn.softmax(model.predict(arr, verbose=0)[0]).numpy()

    best = int(np.argmax(probs))
    crop, label, healthy = CLASSES[names[best]]

    left, right = st.columns(2)
    left.image(img, use_container_width=True)

    with right:
        (st.success if healthy else st.error)(f"**{crop}: {label}**")
        st.metric("Confidence", f"{probs[best] * 100:.1f}%")
        if probs[best] < 0.7:
            st.warning("Low confidence. Try a clearer photo of a single leaf.")

    st.divider()
    for i in np.argsort(probs)[::-1]:
        c, l, _ = CLASSES[names[i]]
        st.progress(float(probs[i]), text=f"{c}: {l} ({probs[i] * 100:.1f}%)")

st.caption("Only knows 5 classes and was trained on clean PlantVillage photos, so results on other plants or messy backgrounds may be wrong.")