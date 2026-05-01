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
    try:
        return YOLO("./yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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
    try:
        img = frame.to_ndarray(format="bgr24")

        # Allowed classes
        allowed_classes = [0, 25, 13, 39, 41, 63, 67]

        if model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Run YOLOv8 tracking
        results = model.track(
            img,
            persist=True,
            conf=0.50,
            iou=0.5,
            classes=allowed_classes,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # =========================
        # OBJECT COUNTING
        # =========================
        counts = {}

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                name = model.names.get(cls, str(cls))
                counts[name] = counts.get(name, 0) + 1

        # Display counts
        y_offset = 35
        for obj, count in counts.items():
            text = f"{obj.upper()}: {count}"
            cv2.putText(
                annotated_frame,
                text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30

        # =========================
        # ALERT SYSTEM
        # =========================
        if "person" in counts:
            cv2.putText(
                annotated_frame,
                "ALERT: Person Detected!",
                (15, y_offset + 20),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    except Exception as e:
        # fallback para hindi mag-crash
        img = frame.to_ndarray(format="bgr24")
        cv2.putText(
            img,
            "ERROR",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
