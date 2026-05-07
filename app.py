import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🚨",
    layout="wide"
)

# =========================
# LIGHT PINK MODERN UI/UX (UNCHANGED)
# =========================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #ffe4ec, #fff0f5);
    color: #333;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffb6c1, #ff69b4);
    color: white;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] img {
    border-radius: 50%;
    border: 3px solid white;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.6);
}

.profile-name {
    text-align:center;
    font-size:22px;
    font-weight:bold;
    margin-bottom:0;
}

.profile-course {
    text-align:center;
    font-size:13px;
    margin-top:0;
    opacity:0.9;
}

div[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
}

.stButton > button {
    background: linear-gradient(90deg, #ff69b4, #ff85c1);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    padding: 0.6rem 1rem;
}

h1, h2, h3 {
    color: #d63384;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SESSION STATE
# =========================
if "detections" not in st.session_state:
    st.session_state.detections = []

if "timeline" not in st.session_state:
    st.session_state.timeline = []

if "mode" not in st.session_state:
    st.session_state.mode = "camera"

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.title("🚨 DASHBOARD")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("profile.png", width=130)

    st.markdown("""
    <p class="profile-name">Liza S. Jaime</p>
    <p class="profile-course">BSCS - 3A</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # TRANSPARENT BUTTONS ONLY
    # =========================
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] button {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6rem;
        backdrop-filter: blur(10px);
        transition: 0.3s ease;
    }

    div[data-testid="stSidebar"] button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        transform: scale(1.03);
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("📡 Live Camera"):
        st.session_state.mode = "camera"

    if st.button("🖼 Upload Image"):
        st.session_state.mode = "upload"

    CONF = st.slider("🎯 Confidence", 0.3, 0.8, 0.5)

# =========================
# TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.caption("AI-powered real-time object detection system with YOLOv8")
# =========================
# FRAME PREPROCESSING
# =========================
def prepare_frame(image):

    img = Image.open(image).convert("RGB")
    frame = np.array(img)

    # FORCE CLEAN FORMAT (RGB → BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):
    
    frame_rgb = np.array(frame)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(frame_bgr, conf=CONF, verbose=False)

    annotated_frame = results[0].plot()

    detected = []

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for c in boxes.cls:
            detected.append(model.names[int(c)])

    if len(detected) > 0:
        unique_detected = list(set(detected))

        st.session_state.detections.append({
            "frame": len(st.session_state.detections),
            "objects": unique_detected
        })

        st.session_state.timeline.append({
            "frame": len(st.session_state.timeline),
            "count": len(unique_detected)
        })

        st.toast(f"🚨 Detected: {', '.join(unique_detected)}")

    # 🔥 FIX COLOR BEFORE DISPLAY
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    return annotated_frame, detected

# =========================
# LIVE CAMERA
# =========================
if st.session_state.mode == "camera":

    st.subheader("📸 Camera Detection")

    camera = st.camera_input("Open Camera")

    if camera is not None:

        image = Image.open(camera).convert("RGB")
        frame = np.array(image)

        result, detected = detect(frame)

        col1, col2 = st.columns(2)

        with col1:
            st.image(frame, caption="Original Image")

        with col2:
            st.image(result, caption="AI Detection")

        st.success(f"Detected Objects: {', '.join(set(detected))}")

# =========================
# UPLOAD IMAGE
# =========================
elif st.session_state.mode == "upload":

    st.subheader("🖼 Upload Detection")

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file is not None:

        img = Image.open(file).convert("RGB")
        frame = np.array(img)

        result, detected = detect(frame)

        col1, col2 = st.columns(2)

        with col1:
            st.image(frame, caption="Original Image")

        with col2:
            st.image(result, caption="AI Detection")

        st.success(f"Detected Objects: {', '.join(set(detected))}")

# =========================
# ANALYTICS
# =========================
st.markdown("### 📊 Object Distribution (Pie Chart)")

if st.session_state.detections:

    all_objects = []
    for d in st.session_state.detections:
        all_objects.extend(d["objects"])

    counter = Counter(all_objects)

    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
    st.pyplot(fig)