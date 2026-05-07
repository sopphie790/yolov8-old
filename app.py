import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import sqlite3

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

/* BACKGROUND */
.main {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: white;
}

/* SIDEBAR PINK */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ff4da6, #ff1a75);
    position: relative;
    overflow: hidden;
}

/* STAR GLITTER EFFECT */
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    width: 300%;
    height: 300%;
    top: -100%;
    left: -100%;

    background-image:
        radial-gradient(circle, white 1px, transparent 1px),
        radial-gradient(circle, rgba(255,255,255,0.6) 1px, transparent 1px);

    background-size: 18px 18px, 28px 28px;

    animation: floatStars 12s linear infinite;

    opacity: 0.4;
    pointer-events: none;
}

/* FLOATING STARS */
@keyframes floatStars {
    0% {transform: translate(0,0);}
    100% {transform: translate(150px,-150px);}
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}

/* PROFILE CARD */
.profile-card {
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 18px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(255,255,255,0.2);
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background: white;
    color: #ff1a75;
    font-weight: bold;
    border-radius: 12px;
}

/* TITLE */
h1 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# LOGIN CHECK (keep your system)
# =========================
if "login" not in st.session_state:
    st.session_state.login = True
    st.session_state.user = "admin"

# =========================
# SIDEBAR UI (UPGRADED)
# =========================
with st.sidebar:

    st.title("DASHBOARD")

    # =========================
    # CSS
    # =========================
    st.markdown("""
    <style>

    .profile-name {
        text-align:center;
        color:white;
        font-size:22px;
        font-weight:bold;
        margin-bottom:0;
    }

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

    label {
        color:white !important;
        font-weight:600 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # CENTER IMAGE
    # =========================
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("profile.png", width=130)

    # =========================
    # PROFILE TEXT
    # =========================
    st.markdown("""
    <p class="profile-name">Liza S. Jaime</p>
    <p class="profile-course">BSCS - A</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # MODE BUTTONS
    # =========================
    mode = st.radio(
        "📌 Select Mode",
        ["📡 Live Camera", "🖼 Upload Image"],
        key="mode_radio"
    )

    # =========================
    # CONFIDENCE
    # =========================
    CONF = st.slider(
        "🎯 Confidence",
        0.1,
        1.0,
        0.25,
        key="confidence_slider"
    )
    st.markdown("---")


# =========================
# TITLE MAIN
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.caption("Point your camera at objects to identify them in real-time")

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):

    results = model.predict(frame, conf=CONF, verbose=False)

    frame_out = results[0].plot()

    detected = []

    boxes = results[0].boxes

    if boxes is not None:

        for i, c in enumerate(boxes.cls):

            name = model.names[int(c)]
            detected.append(name)

    return frame_out, detected

# =========================
# LIVE CAMERA
# =========================
if mode == "📡 Live Camera":

    run = st.checkbox("Start Camera")

    if run:

        cap = cv2.VideoCapture(0)

        frame_box = st.empty()

        while run:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result, detected = detect(frame)

            frame_box.image(result)

        cap.release()

# =========================
# UPLOAD IMAGE
# =========================
elif mode == "🖼 Upload Image":

    file = st.file_uploader("Upload Image")

    if file:

        img = Image.open(file).convert("RGB")
        img = np.array(img)

        result, detected = detect(img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original")

        with col2:
            st.image(result, caption="AI Detection")