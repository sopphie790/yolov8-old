# 🎥 Live Object Detection & Tracking using YOLOv8

## 📌 Project Overview

This project is a real-time AI-powered web application built using **Streamlit** and **YOLOv8 (Ultralytics)**. It uses a webcam to detect, track, and label objects in real time with bounding boxes.

The system demonstrates how computer vision and artificial intelligence work in live environments by processing video frames instantly.

---

## 🎯 Objectives

* To understand real-time computer vision concepts
* To apply AI object detection using YOLOv8
* To build an interactive web application using Streamlit
* To implement object tracking across video frames

---

## ⚙️ Technologies Used

* Python
* Streamlit
* YOLOv8 (Ultralytics)
* OpenCV
* streamlit-webrtc
* PyTorch

---

## 🚀 Features

### 🔍 Real-Time Object Detection

* Detects objects such as:

  * Person
  * Cell phone
  * Bottle
  * Chair

### 📦 Object Tracking

* Tracks objects across frames
* Maintains identity of moving objects

### 🔢 Object Counting

* Displays number of detected objects on screen

### 🚨 Alert System

* Shows warning when a person is detected

### 💾 Frame Saving

* Automatically saves detected frames as images

---

## ▶️ How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run the Application

```bash
streamlit run app.py
```

---

### 3. Open in Browser

```
http://localhost:8501
```

---

## 📁 Project Structure

```
object-detection-app/
│
├── app.py
├── requirements.txt
├── README.md
└── screenshots/ (optional)
```

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
* 💻 GitHub Repository: (add GitHub link here)
* 📄 Documentation: (add Google Docs link here)

---

## 👨‍💻 Developer

LIZA S. JAIME_BSCS-3A

---

## 📌 Note

This project is developed for educational purposes to demonstrate real-time AI object detection and tracking using computer vision techniques.
