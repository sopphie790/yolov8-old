from pyexpat import model

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import time

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
st.markdown("""
<style>

/* =========================
PROFESSIONAL SIDEBAR BUTTON STYLE
========================= */

div[data-baseweb="radio"] > div {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 10px;
    backdrop-filter: blur(10px);
    transition: 0.3s ease-in-out;
}

div[data-baseweb="radio"] > div:hover {
    transform: scale(1.02);
    background: rgba(255, 255, 255, 0.25) !important;
}

div[role="radiogroup"] label[data-selected="true"] {
    background: rgba(255, 255, 255, 0.35) !important;
    border-radius: 16px;
}

div[data-baseweb="radio"] span {
    color: rgba(255, 255, 255, 0.75) !important;
    font-weight: 600;
}

div[role="radiogroup"] label[data-selected="true"] span {
    color: rgba(255, 255, 255, 1) !important;
    font-weight: 700;
}
/* =========================
   TRANSPARENT SIDEBAR BUTTONS
========================= */

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

/* HOVER EFFECT */
div[data-testid="stSidebar"] button:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    transform: scale(1.03);
    cursor: pointer;
}

/* CLICKED BUTTON LOOK (active feel) */
div[data-testid="stSidebar"] button:active {
    background: rgba(255, 255, 255, 0.35) !important;
}
            <style>

/* =========================
   ACTIVE SIDEBAR BUTTON STATE
========================= */

/* default button style */
div[data-testid="stSidebar"] button {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    color: white !important;
    font-weight: bold;
    border-radius: 12px;
    padding: 0.6rem;
    backdrop-filter: blur(10px);
    transition: 0.25s ease;
}

/* hover effect */
div[data-testid="stSidebar"] button:hover {
    background: rgba(255, 255, 255, 0.28) !important;
    transform: scale(1.03);
    cursor: pointer;
}

/* ACTIVE BUTTON LOOK (STREAMLIT SELECTED STATE) */
div[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(90deg, #ff4da6, #ff1a75) !important;
    color: white !important;
    border: 2px solid white !important;
    box-shadow: 0px 0px 15px rgba(255, 20, 147, 0.6);
}

/* TEXT TRANSPARENCY CONTROL */
div[data-testid="stSidebar"] button span {
    color: rgba(255,255,255,0.85) !important;
}

div[data-testid="stSidebar"] button[kind="primary"] span {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():

    # 🔥 BETTER ACCURACY
    return YOLO("yolov8m.pt")

# =========================
# ANALYTICS STORAGE
# =========================
if "detections" not in st.session_state:
    st.session_state.detections = []

if "timeline" not in st.session_state:
    st.session_state.timeline = []

# 🔥 ADVANCED ANALYTICS
if "unique_ids" not in st.session_state:
    st.session_state.unique_ids = set()

if "fps" not in st.session_state:
    st.session_state.fps = 0

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

    # 👇 DITO ILALAGAY
    st.markdown("### 📌 Select Mode")

    if "mode" not in st.session_state:
        st.session_state.mode = "📡 Live Camera"

    if st.button("📡 Live Camera", use_container_width=True):
        st.session_state.mode = "📡 Live Camera"

    if st.button("🖼 Upload Image", use_container_width=True):
        st.session_state.mode = "🖼 Upload Image"

    mode = st.session_state.mode

    CONF = st.slider("🎯 Confidence", 0.3, 0.8, 0.5)
    # 🔥 ADVANCED SETTINGS
    IOU = st.slider("📦 IOU Threshold", 0.1, 1.0, 0.5)

    MAX_DET = st.slider("🔍 Max Detection", 10, 500, 300)

    ENABLE_TRACKING = st.toggle(
    "🛰 Enable Tracking",
    value=True
    )
# =========================
# TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.caption("Point your camera at objects to identify them in real-time")
# 🔥 LIVE ANALYTICS
metric1, metric2, metric3 = st.columns(3)

# =========================
# DETECTION FUNCTION (FIXED)
# =========================
def detect(frame):
    
    start_time = time.time()

    # ✔ KEEP ORIGINAL CLEAN RGB
    frame_rgb = np.array(frame)

    # 🔥 BETTER SMALL OBJECT DETECTION
    frame_rgb = cv2.resize(frame_rgb, (1280, 720))

    # ✔ CONVERT ONLY FOR YOLO INPUT
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # =========================
    # 🔥 ADVANCED TRACKING
    # =========================
    if ENABLE_TRACKING:
        
        results = model.predict(
            source=frame_bgr,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            verbose=False
        )

    else:

        results = model.predict(
            source=frame_bgr,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            verbose=False
        )

    # ✔ YOLO OUTPUT
    annotated_frame = results[0].plot()

    # 🔥 FIX COLOR BACK TO RGB
    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    detected = []

    boxes = results[0].boxes

    # =========================
    # 🔥 SMART FILTERING
    # =========================
    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            conf = float(box.conf[0])

            # 🔥 REMOVE FAKE DETECTIONS
            if conf < 0.35:
                continue

            cls = int(box.cls[0])

            class_name = model.names[cls]

            detected.append(class_name)

            # 🔥 TRACKING IDS
            if ENABLE_TRACKING and box.id is not None:

                track_id = int(box.id[0])

                st.session_state.unique_ids.add(track_id)

    # =========================
    # 🔥 FPS ANALYTICS
    # =========================
    end_time = time.time()

    fps = 1 / (end_time - start_time)

    st.session_state.fps = round(fps, 2)

    # =========================
    # EXISTING ANALYTICS
    # =========================
    if len(detected) > 0:

        unique_detected = list(set(detected))

        st.session_state.detections.append({
            "frame": len(st.session_state.detections),
            "objects": unique_detected
        })

        st.session_state.timeline.append({
            "frame": len(st.session_state.timeline),
            "objects": unique_detected,
            "count": len(unique_detected)
        })

        st.toast(f"🚨 Detected: {', '.join(unique_detected)}")

    # =========================
    # 🔥 LIVE METRICS
    # =========================
    metric1.metric(
        "⚡ FPS",
        st.session_state.fps
    )

    metric2.metric(
        "🎯 Objects",
        len(detected)
    )

    metric3.metric(
        "🛰 Tracking IDs",
        len(st.session_state.unique_ids)
    )

    return annotated_frame, detected

# =========================
# LIVE CAMERA
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
# IMAGE UPLOAD
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

# =========================
# 📊 OBJECT DISTRIBUTION
# =========================

st.markdown("### 📊 Object Distribution (Pie Chart)")

if st.session_state.detections:

    all_objects = []

    for item in st.session_state.detections:
        all_objects.extend(item["objects"])

    counter = Counter(all_objects)

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.pie(
        counter.values(),
        labels=counter.keys(),
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

# =========================
# 🔥 HEATMAP
# =========================

st.markdown("### 🔥 Detection Heatmap")

if st.session_state.detections:

    heat_objects = []

    for item in st.session_state.detections:
        if "objects" in item:
            heat_objects.extend(item["objects"])

    heat_data = Counter(heat_objects)

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.imshow(
        [list(heat_data.values())],
        cmap="Reds",
        aspect="auto"
    )

    ax.set_yticks([])

    ax.set_xticks(range(len(heat_data)))

    ax.set_xticklabels(
        list(heat_data.keys()),
        rotation=45
    )

    st.pyplot(fig)

# =========================
# ⏱ TIMELINE
# =========================

st.markdown("### ⏱ Detection Timeline")

if st.session_state.timeline:

    df = pd.DataFrame(st.session_state.timeline)

    fig, ax = plt.subplots()

    ax.plot(df["frame"], df["count"], marker="o", linestyle="-")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Objects Detected")
    ax.set_title("Detection Over Time")

    st.pyplot(fig)