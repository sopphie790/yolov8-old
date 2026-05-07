# 🚨 AI Surveillance System  
## Real-Time Object Detection and Tracking using YOLOv8 + Streamlit

---

## 📌 Project Overview

This project is a **real-time AI-powered object detection and tracking system** built using **Streamlit, YOLOv8, OpenCV, and Python**.

It enables users to:
- Detect objects in real time using a webcam 📷  
- Upload images for object detection 🖼  
- Track objects across frames 🛰  
- View analytics through interactive dashboards 📊  

The system transforms raw AI outputs into meaningful insights using visualization and analytics.

---

## 🎯 Features

### 🔥 Core Features
- Real-time object detection using YOLOv8
- Webcam live detection
- Image upload detection mode
- Bounding box visualization with labels

### 🛰 Tracking System
- Object tracking across frames
- Unique object ID counting
- Detection history storage

### 📊 Analytics Dashboard
- Pie chart (object distribution)
- Heatmap (object frequency)
- Timeline graph (detection over time)

### ⚡ Performance Monitoring
- FPS (Frames Per Second) tracking
- Object count per frame
- Tracking ID counter

### 🎛 Controls
- Confidence threshold slider
- IOU adjustment
- Max detection limit
- Tracking toggle
- Clear analytics button

---

## 🧠 Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| AI Model | YOLOv8 |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Image Handling | PIL |

---

## 📷 System Workflow
Input (Webcam / Image)
↓
Frame Processing (OpenCV)
↓
YOLOv8 Object Detection
↓
Tracking & Filtering
↓
Analytics Storage
↓
Visualization Dashboard


---

## ⚙️ System Requirements

### 📦 Python Dependencies (`requirements.txt`)

```txt
streamlit==1.36.0
streamlit-webrtc==0.47.1
opencv-python-headless
numpy
pillow
ultralytics
av
pandas
plotly

🖥️ System Dependencies (packages.txt)
ffmpeg
pkg-config
libavcodec-dev
libavformat-dev
libavdevice-dev
libavutil-dev
libswscale-dev
libswresample-dev

🚀 Installation & Run
1️⃣ Clone Repository
git clone https://github.com/your-username/ai-surveillance-system.git
cd ai-surveillance-system

2️⃣ Install Requirements
pip install -r requirements.txt

💻 Windows Command Prompt
Open Command Prompt
Go to your project folder:

git add .
git commit -m "Final clean version of app.py"
git push origin main

3️⃣ Run App
py -m streamlit run app.py
 
localhost
Local URL: http://localhost:8501

📊 Enhancements Implemented
📊 Analytics Dashboard (Pie, Heatmap, Timeline)
🛰 Object Tracking System
⚡ FPS Performance Monitoring
🎯 Smart Confidence Filtering
🖼 Dual Input System (Camera + Upload)
🎨 Professional UI/UX Design
👨‍🎓 Developer

Name: Liza S. Jaime
Course: BSCS - 3A
Institution: DEBESMSCAT

