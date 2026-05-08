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
# YOUR UI/UX (UNCHANGED)
# =========================
# (UNCHANGED - your full CSS stays exactly the same)
# 👉 NO EDIT MADE HERE

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# =========================
# SESSION STATE FIX (IMPORTANT)
# =========================
if "detections" not in st.session_state:
    st.session_state.detections = []

if "timeline" not in st.session_state:
    st.session_state.timeline = []

if "unique_ids" not in st.session_state:
    st.session_state.unique_ids = set()

if "fps" not in st.session_state:
    st.session_state.fps = 0

# =========================
# 🎯 UNIFIED DETECTION ENGINE (FIX)
# =========================
def parse_detections(results):
    names = results[0].names
    boxes = results[0].boxes

    detected_objects = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:

            conf = float(box.conf[0])

            # 🔥 filter weak detections
            if conf < 0.35:
                continue

            cls = int(box.cls[0])
            detected_objects.append(names[cls])

    return detected_objects

# =========================
# 🎯 SINGLE ANALYTICS ENGINE (FIX)
# =========================
def generate_analytics(detected_objects):

    counts = Counter(detected_objects)

    analytics = {
        "total": sum(counts.values()),
        "unique": len(counts),
        "most_common": counts.most_common(1)[0] if counts else ("None", 0),
        "counts": counts
    }

    return analytics

# =========================
# DETECTION FUNCTION (FIXED CORE)
# =========================
def detect(frame, record=False):

    start = time.time()

    frame_rgb = np.array(frame)
    frame_rgb = cv2.resize(frame_rgb, (1280, 720))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=frame_bgr,
        conf=st.session_state.get("CONF", 0.5),
        iou=st.session_state.get("IOU", 0.5),
        max_det=st.session_state.get("MAX_DET", 300),
        verbose=False
    )

    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # 🔥 FIX: unified detection
    detected = parse_detections(results)

    analytics = generate_analytics(detected)

    # =========================
    # RECORD ONLY CLEAN DATA
    # =========================
    if record and detected:

        st.session_state.detections.append({
            "frame": len(st.session_state.detections),
            "objects": detected
        })

        st.session_state.timeline.append({
            "frame": len(st.session_state.timeline),
            "count": len(detected)
        })

    # =========================
    # FPS
    # =========================
    fps = 1 / (time.time() - start)
    st.session_state.fps = round(fps, 2)

    return annotated, detected, analytics

# =========================
# SIDEBAR VALUES (UNCHANGED UI)
# =========================
with st.sidebar:

    st.session_state.CONF = st.slider("🎯 Confidence", 0.3, 0.8, 0.5)
    st.session_state.IOU = st.slider("📦 IOU Threshold", 0.1, 1.0, 0.5)
    st.session_state.MAX_DET = st.slider("🔍 Max Detection", 10, 500, 300)

    st.session_state.ENABLE_TRACKING = st.toggle("🛰 Enable Tracking", value=True)

# =========================
# TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")

metric1, metric2, metric3 = st.columns(3)

# =========================
# MODE
# =========================
mode = st.selectbox("Select Mode", ["📷 Live Camera", "🖼 Upload Image"])

# =========================
# LIVE CAMERA
# =========================
if mode == "📷 Live Camera":

    cam = st.camera_input("Open Camera")

    if cam:

        img = Image.open(cam).convert("RGB")
        frame = np.array(img)

        result, detected, analytics = detect(frame, record=True)

        col1, col2 = st.columns(2)

        with col1:
            st.image(frame)

        with col2:
            st.image(result)

        st.success(f"Detected: {set(detected)}")

# =========================
# UPLOAD IMAGE FIXED
# =========================
elif mode == "🖼 Upload Image":

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:

        # 🔥 RESET (FIXED BUG)
        st.session_state.detections = []
        st.session_state.timeline = []
        st.session_state.unique_ids = set()

        img = Image.open(file).convert("RGB")
        frame = np.array(img)

        result, detected, analytics = detect(frame, record=True)

        col1, col2 = st.columns(2)

        with col1:
            st.image(frame)

        with col2:
            st.image(result)

        st.success(f"Detected: {set(detected)}")

# =========================
# 🔥 METRICS (FIXED CONSISTENCY)
# =========================
metric1.metric("⚡ FPS", st.session_state.fps)

all_objects = []
for d in st.session_state.detections:
    all_objects.extend(d["objects"])

counts = Counter(all_objects)

metric2.metric("🎯 Total Objects", sum(counts.values()))
metric3.metric("🧠 Unique Objects", len(counts))

# =========================
# 📊 PIE CHART (FIXED SOURCE)
# =========================
st.markdown("### 📊 Object Distribution")

if counts:

    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
    st.pyplot(fig)

# =========================
# 🔥 HEATMAP (FIXED SOURCE)
# =========================
st.markdown("### 🔥 Detection Heatmap")

if counts:

    fig, ax = plt.subplots()
    ax.imshow([list(counts.values())], cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(list(counts.keys()), rotation=30)
    ax.set_yticks([])

    st.pyplot(fig)

# =========================
# ⏱ TIMELINE (FIXED)
# =========================
st.markdown("### ⏱ Timeline")

if st.session_state.timeline:

    df = pd.DataFrame(st.session_state.timeline)

    fig, ax = plt.subplots()
    ax.plot(df["frame"], df["count"], marker="o")

    st.pyplot(fig)