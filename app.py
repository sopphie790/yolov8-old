import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🚨",
    layout="wide"
)

# =========================
# CUSTOM PINK GLITTER UI (UNCHANGED)
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

@keyframes stars {
    0% { transform: translate(0,0); }
    100% { transform: translate(150px,-150px); }
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

/* PROFILE TEXT */
.profile-name {
    text-align:center;
    color:white;
    font-size:22px;
    font-weight:bold;
}

.profile-course {
    text-align:center;
    color:#ffd1e8;
    font-size:14px;
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
# SESSION STORAGE (ANALYTICS)
# =========================
if "detections" not in st.session_state:
    st.session_state.detections = []

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame, CONF):

    results = model.predict(frame, conf=CONF, verbose=False)
    annotated_frame = results[0].plot()

    detected = []
    boxes = results[0].boxes

    if boxes is not None:
        for c in boxes.cls:
            detected.append(model.names[int(c)])

    # SAVE FOR ANALYTICS
    if detected:
        st.session_state.detections.extend(detected)

    return annotated_frame, detected

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
        ["📡 Live Camera", "🖼 Upload Image", "📊 Analytics"]
    )

    CONF = st.slider("🎯 Confidence", 0.1, 1.0, 0.25)

# =========================
# MAIN TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.caption("AI-powered YOLOv8 detection system")

# =========================
# =========================
# ANALYTICS PAGE
# =========================
# =========================
if mode == "📊 Analytics":

    st.subheader("📊 Detection Analytics Dashboard")

    data = st.session_state.detections

    if len(data) == 0:
        st.info("No detections yet. Start scanning objects.")
    else:

        counter = Counter(data)
        df = pd.DataFrame(counter.items(), columns=["Object", "Count"])
        df = df.sort_values(by="Count", ascending=False)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Detections", len(data))
        col2.metric("Unique Objects", len(counter))
        col3.metric("Top Object", df.iloc[0]["Object"])

        st.bar_chart(df.set_index("Object"))
        st.dataframe(df, use_container_width=True)

# =========================
# LIVE CAMERA
# =========================
elif mode == "📡 Live Camera":

    st.subheader("📸 Camera Detection")

    camera = st.camera_input("Open Camera")

    if camera:

        image = Image.open(camera).convert("RGB")
        frame = np.array(image)

        result, detected = detect(frame, CONF)

        col1, col2 = st.columns(2)

        with col1:
            st.image(frame, caption="Original")

        with col2:
            st.image(result, caption="Detected")

        if detected:
            st.success(f"Detected: {', '.join(set(detected))}")

# =========================
# IMAGE UPLOAD
# =========================
elif mode == "🖼 Upload Image":

    st.subheader("🖼 Upload Detection")

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file:

        img = Image.open(file).convert("RGB")
        frame = np.array(img)

        result, detected = detect(frame, CONF)

        col1, col2 = st.columns(2)

        with col1:
            st.image(frame)

        with col2:
            st.image(result)

        if detected:
            st.success(f"Detected: {', '.join(set(detected))}")