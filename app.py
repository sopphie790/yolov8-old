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
# YOUR UI/UX DESIGN (UNCHANGED)
# =========================
# KEEP YOUR FULL CSS HERE
# WALANG BINAGO SA UI/UX

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# =========================
# SESSION STATE
# =========================
if "detections" not in st.session_state:
    st.session_state.detections = []

if "timeline" not in st.session_state:
    st.session_state.timeline = []

if "fps" not in st.session_state:
    st.session_state.fps = 0

# =========================
# SETTINGS
# =========================
CONF = 0.5
IOU = 0.5
MAX_DET = 300

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.title("🚨 DASHBOARD")

    mode = st.selectbox(
        "Select Mode",
        ["📷 Live Camera", "🖼 Upload Image"]
    )

    CONF = st.slider(
        "🎯 Confidence",
        0.3,
        0.9,
        0.5
    )

    IOU = st.slider(
        "📦 IOU Threshold",
        0.1,
        1.0,
        0.5
    )

    MAX_DET = st.slider(
        "🔍 Max Detection",
        10,
        500,
        300
    )

    if st.button("🧹 Clear Analytics"):

        st.session_state.detections = []
        st.session_state.timeline = []

        st.rerun()

# =========================
# TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")

st.caption(
    "Professional Real-Time Object Detection Analytics"
)

metric1, metric2, metric3 = st.columns(3)

# =========================
# DETECTION ENGINE
# =========================
def detect_objects(frame):

    start = time.time()

    frame_rgb = np.array(frame)

    frame_resized = cv2.resize(
        frame_rgb,
        (1280, 720)
    )

    frame_bgr = cv2.cvtColor(
        frame_resized,
        cv2.COLOR_RGB2BGR
    )

    results = model.predict(
        source=frame_bgr,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        verbose=False
    )

    annotated = results[0].plot()

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    detected_objects = []

    boxes = results[0].boxes

    # =========================
    # REAL DETECTION COUNTS
    # =========================
    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < CONF:
                continue

            cls = int(box.cls[0])

            class_name = model.names[cls]

            # 🔥 IMPORTANT
            # KEEP DUPLICATES
            detected_objects.append(class_name)

    # =========================
    # FPS
    # =========================
    fps = 1 / (time.time() - start)

    st.session_state.fps = round(fps, 2)

    # =========================
    # SAVE ANALYTICS
    # =========================
    if len(detected_objects) > 0:

        st.session_state.detections.append({
            "frame": len(st.session_state.detections),
            "objects": detected_objects
        })

        st.session_state.timeline.append({
            "frame": len(st.session_state.timeline),
            "count": len(detected_objects)
        })

    return annotated, detected_objects

# =========================
# LIVE CAMERA
# =========================
if mode == "📷 Live Camera":

    st.subheader("📷 Camera Detection")

    camera = st.camera_input("Open Camera")

    if camera is not None:

        image = Image.open(camera).convert("RGB")

        frame = np.array(image)

        result, detected = detect_objects(frame)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                frame,
                caption="Original Image"
            )

        with col2:
            st.image(
                result,
                caption="AI Detection"
            )

        # =========================
        # DETECTED OBJECT DISPLAY
        # =========================
        if len(detected) > 0:

            detection_summary = Counter(detected)

            formatted = ", ".join([
                f"{obj} ({count})"
                for obj, count
                in detection_summary.items()
            ])

            st.success(
                f"Detected: {formatted}"
            )

        else:

            st.warning(
                "No objects detected"
            )

# =========================
# IMAGE UPLOAD
# =========================
elif mode == "🖼 Upload Image":

    st.subheader("🖼 Upload Detection")

    file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:

        # RESET FOR CLEAN ANALYTICS
        st.session_state.detections = []
        st.session_state.timeline = []

        img = Image.open(file).convert("RGB")

        frame = np.array(img)

        result, detected = detect_objects(frame)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                frame,
                caption="Original Image"
            )

        with col2:
            st.image(
                result,
                caption="AI Detection"
            )

        if len(detected) > 0:

            detection_summary = Counter(detected)

            formatted = ", ".join([
                f"{obj} ({count})"
                for obj, count
                in detection_summary.items()
            ])

            st.success(
                f"Detected: {formatted}"
            )

        else:

            st.warning(
                "No objects detected"
            )

# =========================
# GLOBAL ANALYTICS
# =========================
all_objects = []

for detection in st.session_state.detections:

    if "objects" in detection:

        all_objects.extend(
            detection["objects"]
        )

# 🔥 REAL COUNTS
counts = Counter(all_objects)

total_objects = sum(counts.values())

unique_objects = len(counts)

# =========================
# METRICS
# =========================
metric1.metric(
    "⚡ FPS",
    st.session_state.fps
)

metric2.metric(
    "🎯 Total Objects",
    total_objects
)

metric3.metric(
    "🧠 Unique Objects",
    unique_objects
)

# =========================
# PIE CHART
# =========================
st.markdown(
    "### 📊 Object Distribution"
)

if len(counts) > 0:

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.pie(
        counts.values(),
        labels=counts.keys(),
        autopct='%1.1f%%',
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)

# =========================
# HEATMAP
# =========================
st.markdown(
    "### 🔥 Detection Heatmap"
)

if len(counts) > 0:

    fig, ax = plt.subplots(
        figsize=(6, 2)
    )

    heat_values = list(
        counts.values()
    )

    heat_labels = list(
        counts.keys()
    )

    ax.imshow(
        [heat_values],
        cmap="Reds",
        aspect="auto"
    )

    ax.set_xticks(
        range(len(heat_labels))
    )

    ax.set_xticklabels(
        heat_labels,
        rotation=25
    )

    ax.set_yticks([])

    st.pyplot(fig)

# =========================
# TIMELINE
# =========================
st.markdown(
    "### ⏱ Detection Timeline"
)

if len(st.session_state.timeline) > 0:

    df = pd.DataFrame(
        st.session_state.timeline
    )

    fig, ax = plt.subplots(
        figsize=(6, 3)
    )

    ax.plot(
        df["frame"],
        df["count"],
        marker="o"
    )

    ax.set_xlabel("Frame")

    ax.set_ylabel("Objects")

    ax.set_title(
        "Detection Timeline"
    )

    st.pyplot(fig)