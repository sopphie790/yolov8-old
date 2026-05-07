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
# CUSTOM PINK GLITTER UI
# =========================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: white;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ff4da6, #ff1a75);
    position: relative;
    overflow: hidden;
}

/* STAR GLITTER */
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    width: 300%;
    height: 300%;
    top: -100%;
    left: -100%;

    background-image:
        radial-gradient(circle, rgba(255,255,255,0.9) 1px, transparent 1px),
        radial-gradient(circle, rgba(255,255,255,0.5) 1px, transparent 1px);

    background-size: 18px 18px, 28px 28px;

    animation: stars 12s linear infinite;

    opacity: 0.4;
}

/* FLOATING ANIMATION */
@keyframes stars {
    0% {
        transform: translate(0,0);
    }
    100% {
        transform: translate(150px,-150px);
    }
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* PROFILE IMAGE */
[data-testid="stSidebar"] img {
    border-radius: 50%;
    border: 4px solid white;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.5);
}

/* PROFILE NAME */
.profile-name {
    text-align:center;
    color:white;
    font-size:22px;
    font-weight:bold;
    margin-bottom:0;
}

/* PROFILE COURSE */
.profile-course {
    text-align:center;
    color:#ffd1e8;
    font-size:14px;
    margin-top:0;
}

/* RADIO BUTTON STYLE */
div[data-baseweb="radio"] > div {
    background: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(8px);
}

/* SLIDER */
.stSlider {
    padding-top: 10px;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: rgba(255,255,255,0.15);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
    font-weight: bold;
}

/* TITLE */
h1, h2, h3 {
    color: white;
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

    # CENTER IMAGE
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("profile.png", width=130)

    # PROFILE TEXT
    st.markdown("""
    <p class="profile-name">Liza S. Jaime</p>
    <p class="profile-course">BSCS - A</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # MODE
    mode = st.radio(
        "📌 Select Mode",
        ["📡 Live Camera", "🖼 Upload Image"],
        key="mode_radio"
    )

    # CONFIDENCE
    CONF = st.slider(
        "🎯 Confidence",
        0.1,
        1.0,
        0.25,
        key="confidence_slider"
    )

# =========================
# MAIN TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.caption("Point your camera at objects to identify them in real-time.")

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

            name = model.names[int(c)]
            detected.append(name)

    return annotated_frame, detected

# =========================
# LIVE CAMERA MODE
# =========================
if mode == "📡 Live Camera":

    st.subheader("📸 Camera Detection")

    camera = st.camera_input(
        "Open Camera",
        key="camera_input"
    )

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

    file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="upload_image"
    )

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