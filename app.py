import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# =========================
# Load YOLOv8 Model (cached)
# =========================
@st.cache_resource
def load_model():
    return YOLO("./yolov8n.pt")

model = load_model()

# =========================
# UI
# =========================
st.title("🧠 AI Object Detection (Stable Version)")
st.write("Upload an image or video for object detection using YOLOv8.")
st.info("Stable version: No webcam, no crash during deployment.")

# =========================
# Allowed Classes
# =========================
allowed_classes = [0, 25, 13, 39, 41, 63, 67]

# =========================
# IMAGE UPLOAD
# =========================
st.subheader("📸 Image Detection")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    img = np.array(image)

    results = model(img, conf=0.5)

    annotated = results[0].plot()

    # COUNT OBJECTS
    counts = {}
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls in allowed_classes:
                name = model.names[cls]
                counts[name] = counts.get(name, 0) + 1

    # DISPLAY COUNTS
    for obj, count in counts.items():
        cv2.putText(
            annotated,
            f"{obj.upper()}: {count}",
            (15, 30 + list(counts.keys()).index(obj)*30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ALERT
    if "person" in counts:
        cv2.putText(
            annotated,
            "ALERT: Person Detected!",
            (15, 200),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 255),
            2
        )

    st.image(annotated, channels="BGR")

# =========================
# VIDEO UPLOAD (OPTIONAL)
# =========================
st.subheader("🎥 Video Detection (Optional)")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)
        annotated = results[0].plot()

        stframe.image(annotated, channels="BGR")

    cap.release()
