🎥 Live Object Detection & Tracking using YOLOv8
##📌 Project Overview
This project is a real-time AI-powered web application built using Streamlit and YOLOv8 (Ultralytics). It utilizes a webcam feed to detect, track, and label specific objects in real-time with high-precision bounding boxes.

The system demonstrates the practical application of computer vision and artificial intelligence in live environments by processing video frames instantly.

##🎯 Objectives
To understand real-time computer vision and image processing concepts.
To apply state-of-the-art AI object detection using the YOLOv8 architecture.
To build and deploy an interactive web application using the Streamlit framework.
To implement persistent object tracking across continuous video frames.

##⚙️ Technologies Used
Python: Core programming language.
Streamlit: Web interface framework.
YOLOv8 (Ultralytics): Object detection and tracking engine.
OpenCV / PyAV: Image and video frame processing.
streamlit-webrtc: Real-time webcam streaming for web browsers.
PyTorch: Deep learning backend.

##🚀 Features
🔍 Real-Time Object Detection
The model is optimized to detect specific classes:

Person (Class 0)
Umbrella (Class 25)
Bench (Class 13)
Bottle (Class 39)
Cup (Class 41)
Laptop (Class 63)
Cell Phone (Class 67)

##📦 Object Tracking
Assigns unique Tracking IDs to objects across frames to maintain identity during movement.

##🔢 Object Counting
Dynamically displays the total number of detected objects on the screen.

##🚨 Alert System
Triggers a visual "ALERT: Person Detected!" notification when a person enters the frame.

##💾 Frame Saving
Automatically captures and saves processed frames for documentation and analysis.

##▶️ How to Run the Project
1. Install Dependencies
pip install streamlit streamlit-webrtc ultralytics opencv-python av torch torchvision numpy pillow scipy matplotlib
2. Run the Application
py -m streamlit run app.py
3. Open in Browser
http://localhost:8501
 

## 📁 Project Structure

```
object-detection-app/
│
├── app.py              # Main application logic
├── requirements.txt    # Python library dependencies
├── packages.txt        # System-level dependencies for Cloud deployment
├── yolov8n.pt          # Pre-trained YOLOv8 model weights
├── README.md           # Project documentation
└── screenshots/        # Saved detection frames and samples
    ├── bench.png
    ├── cup.png
    └── umbrella.png

---

## 📊 Observation Report

* Detection works best in good lighting conditions
* Objects like person and cellphone are easily detected
* Performance may slow down with low-end devices or poor lighting

---

## 🧠 Reflection

### What objects were easily detected?

Common objects such as person, cellphone, and bottle were easily detected by the model.

### What factors affect detection accuracy?

* Lighting conditions
* Camera quality
* Distance of object
* Background noise/clutter

---

## 📸 Screenshots

Include at least 5 screenshots showing:
- Single object detection
- Multiple objects detection
- Person alert system
- Object counting display
- Tracking movement

---

## 🔗 Submission Links

* 🌐 Live App: (add Streamlit link here)
* 💻 GitHub Repository: https://github.com/sopphie790/yolov8-streamlit-app.git
* 📄 Documentation: (add Google Docs link here)

---

👨‍💻 Developer
LIZA S. JAIME
3rd Year Computer Science Student
DEBESMSCAT
---

## 📌 Note

This project is developed for educational purposes to demonstrate real-time AI object detection and tracking using computer vision techniques.
