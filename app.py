import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import numpy as np

st.title("Live YOLO Webcam Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class YOLOCam(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.5)
        return results[0].plot()

webrtc_streamer(
    key="yolo",
    video_transformer_factory=YOLOCam,
    media_stream_constraints={"video": True, "audio": False},
)
