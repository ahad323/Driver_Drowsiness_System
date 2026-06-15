# Driver Drowsiness Detection System

## Graduation Project II
Faculty of Engineering Technology  
El-Sewedy University of Technology

---

## Table of Contents

- Project Overview
- Features
- System Architecture
- Technologies Used
- Datasets
- Project Structure
- Installation
- Usage
- Results
- Screenshots
- Future Work
- Team Members
- Supervisor

---

## Project Overview

Driver drowsiness is one of the leading causes of road accidents worldwide.

This project presents an intelligent Driver Drowsiness Detection System based on Artificial Intelligence and Computer Vision techniques.

The system continuously monitors the driver's face in real time and detects fatigue indicators such as:

- Eye Closure
- Yawning
- Head Movement
- Facial Landmarks

When drowsiness is detected, the system immediately activates an alarm to alert the driver and help prevent accidents.

---

## Features

✅ Real-Time Face Detection

✅ Eye Closure Detection using EAR

✅ Yawning Detection using MAR

✅ Head Movement Analysis

✅ AI-Based Classification Models

✅ Audible Alarm System

✅ Arduino Integration

✅ Real-Time Monitoring

---

## System Architecture

Camera
↓
Face Detection
↓
MediaPipe Facial Landmarks
↓
EAR & MAR Calculation
↓
AI Models
↓
Drowsiness Decision
↓
Alarm System

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- MobileNetV2
- YOLOv8
- Arduino Uno
- NumPy
- Scikit-Learn
- Pygame

---

## Datasets

### NTHU-DDD Dataset

Used for driver fatigue detection under different lighting conditions and driver behaviors.

### YawDD Dataset

Used for yawning detection and mouth activity analysis.

### Driver Drowsiness Dataset (DDD)

Used for eye-state classification and fatigue detection.

---

## Project Structure

```text
drowsiness_project/
│
├── data/
├── models/
├── results/
├── utils/
│
├── train_eye.py
├── train_yawn.py
├── real_time.py
├── evaluate_visual.py
├── final_report.py
├── requirements.txt
└── alarm.mp3
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python real_time.py
```

---

## Results

### Performance Metrics

| Metric | Value |
|----------|----------|
| Overall Accuracy | 92.8% |
| Eye Model Accuracy | 95.68% |
| Yawn Model Accuracy | 92.72% |
| Inference Speed | 30 FPS |

---

## Screenshots

### System Workflow

![Workflow](images/workflow.png)

### Real-Time Detection

![Detection](images/detection.png)

### EAR Detection

![EAR](images/ear.png)

### MAR Detection

![MAR](images/mar.png)

---

## Future Work

- Night Vision Support
- NVIDIA Jetson Deployment
- Driver Distraction Detection
- Fleet Monitoring System
- Mobile Application Integration

---

## Team Members

- Abdelhakim Nabil Abdelhakim
- Ahmed Alham Mahmoud
- Ahd Malik Monair
- Magy Romani Ezzat
- Mariam Magdy Mohammed
- Abla Abdelmoneim

---

## Supervisor

Dr. Sahar Kamal

---

## Academic Year

2025 / 2026