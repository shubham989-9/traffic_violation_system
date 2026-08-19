# 🚦 AI-Powered Automated Traffic Violation Detection & E-Challan System

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-00FFFF.svg?style=for-the-badge&logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An intelligent, end-to-end Computer Vision system designed to automate traffic surveillance and law enforcement in real time. Leveraging state-of-the-art **YOLOv8/v11** object detection, multi-object tracking (**ByteTrack / DeepSORT**), and **OCR (ANPR)**, this system accurately identifies multiple traffic violations, extracts license plates, logs evidence, and serves an interactive web dashboard for monitoring.

---

## 🌟 Key Highlights & Core Violations

| Violation Module | Description | Detection Method |
| :--- | :--- | :--- |
| 🔴 **Red Light Violation (RLVD)** | Detects vehicles crossing the stop/zebra line during active red signals | ROI Line Crossing + Signal State Sync |
| 🪖 **Helmet Violation Detection** | Flags two-wheeler riders without helmets or safety headgear | Custom YOLOv8 Rider-Helmet Association |
| 🚫 **Wrong-Way Driving** | Identifies vehicles moving against the designated lane traffic flow | Vector Trajectory & Optical Flow |
| ⚡ **Overspeeding Detection** | Calculates instantaneous vehicle velocity across calibrated road zones | Distance-Over-Time Frame Calibration |
| 👥 **Triple Riding on Two-Wheelers** | Flags motorcycles carrying more than two individuals | Spatial Bounding Box Clustering |
| 📵 **Mobile Phone Use / Seatbelt** | Detects driver inattention and seatbelt non-compliance | Interior Cab Bounding Analysis |
| 🔢 **Automatic Number Plate Recognition (ANPR)** | Automatically crops plates and extracts characters | LP Detection + EasyOCR / PaddleOCR |

---

## 🏗️ System Architecture & Workflow

```mermaid
graph TD
    A[CCTV / Video / RTSP Stream] --> B[Frame Preprocessing & Enhancement]
    B --> C[YOLOv8 / YOLOv11 Multi-Class Detection]
    C --> D[Multi-Object Tracker ByteTrack / DeepSORT]
    D --> E{Violation Rule Engine}
    
    E -->|Red Light Jump| F[Trigger Event]
    E -->|No Helmet / Triple Ride| F
    E -->|Wrong Way / Overspeed| F
    
    F --> G[ANPR Pipeline: License Plate Localization & OCR]
    G --> H[Evidence Capture: Timestamp, Plate No, Cropped Image]
    H --> I[(Database Logging: SQLite / PostgreSQL)]
    I --> J[Streamlit Analytics Dashboard / E-Challan Generation]
