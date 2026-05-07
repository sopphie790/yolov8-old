import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🚨",
    layout="wide"
)

# =========================
# LIGHT PINK MODERN UI/UX
# =========================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background: linear-gradient(135deg, #ffe4ec, #fff0f5);
    color: #333;
}

/* APP CONTAINER */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffb6c1, #ff69b4);
    color: white;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* PROFILE IMAGE */
[data-testid="stSidebar"] img {
    border-radius: 50%;
    border: 3px solid white;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.6);
}

/* PROFILE NAME */
.profile-name {
    text-align:center;
    font-size:22px;
    font-weight:bold;
    margin-bottom:0;
}

/* PROFILE COURSE */
.profile-course {
    text-align:center;
    font-size:13px;
    margin-top:0;
    opacity:0.9;
}

/* CARDS */
div[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #ff69b4, #ff85c1);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    padding: 0.6rem 1rem;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #ff85c1, #ff69b4);
}

/* RADIO STYLE */
div[data-baseweb="radio"] > div {
    background: rgba(255,255,255,0.7);
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* TITLE */
h1, h2, h3 {
    color: #d63384;
}

/* CAPTION */
.stCaption {
    color: #6c757d;
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
# SIDEBAR
# =========================
with st.sidebar:

    st.title("🚨 DASHBOARD")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("profile.png", width=130)

    st.markdown("""
    <p class="profile-name">Liza S. Jaime</p>
    <p class="profile-course">BSCS - A</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    mode = st.radio(
        "📌 Select Mode",
        ["📡 Live Camera", "🖼 Upload Image"]
    )

    CONF = st.slider("🎯 Confidence", 0.1, 1.0, 0.25)

# =========================
# MAIN TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.caption("AI-powered real-time object detection system with YOLOv8")

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):
    results = model.predict(frame, conf=CONF, verbose=False)
    annotated_frame = results[0].plot()

    detected = []
    boxes = results[0].boxes

    if boxes is not None:
        for c in boxes.cls:
            detected.append(model.names[int(c)])

    return annotated_frame, detected

# =========================
# LIVE CAMERA MODE
# =========================
if mode == "📡 Live Camera":

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
# IMAGE UPLOAD MODE
# =========================
elif mode == "🖼 Upload Image":

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