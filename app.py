import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# =========================
# Load YOLOv8 Model (cached)
# =========================
@st.cache_resource
def load_model():
    # Gagamit tayo ng yolov8n.pt para mabilis sa laptop mo
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# UI
# =========================
st.title("🎥 Live Object Detection & Tracking")
st.write("Real-time AI object detection using YOLOv8 and webcam.")
st.info("Tip: Itaas ang liwanag ng paligid para sa mas tumpak na detection.")

# =========================
# Video Processing Function
# =========================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
# TAMA NA IDs (YOLOv8 starts at 0):
    # 0: person, 25: umbrella, 13: bench, 39: bottle, 41: cup, 63: laptop, 67: cell phone
    allowed_classes = [0, 25, 13, 39, 41, 63, 67]
    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.50,      # Itinaas sa 0.50 para iwas sa maling hula
        iou=0.5,        # Para hindi mag-overlap ang boxes
        classes=allowed_classes, # Ipakita lang ang mga objects na ito
        verbose=False
    )

    annotated_frame = results[0].plot()

    # =========================
    # 🔢 OBJECT COUNTING
    # =========================
    counts = {}

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            counts[name] = counts.get(name, 0) + 1

    # Display counts on screen (Upper Left)
    y_offset = 35
    for obj, count in counts.items():
        text = f"{obj.upper()}: {count}"
        cv2.putText(
            annotated_frame,
            text,
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0), # Green color para sa counts
            2
        )
        y_offset += 30

    # =========================
    # 🚨 ALERT SYSTEM
    # =========================
    if "person" in counts:
        cv2.putText(
            annotated_frame,
            "ALERT: Person Detected!",
            (15, y_offset + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 255), # Red color para sa alert
            2
        )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# =========================
# Start Webcam Stream
# =========================
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)