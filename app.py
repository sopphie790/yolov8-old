import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# UI
# =========================
st.title("🧠 AI Object Detection (Cloud Stable Version)")
st.write("Upload an image for real-time object detection using YOLOv8.")

# =========================
# IMAGE DETECTION ONLY
# =========================
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    img = np.array(image)

    results = model(img, conf=0.5)

    annotated = results[0].plot()

    st.image(annotated, caption="Detected Image", use_container_width=True)

    # Object summary
    st.subheader("Detected Objects:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        st.write(f"- {name}")
