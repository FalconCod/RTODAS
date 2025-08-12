

# Real-Time Object Detection & Alert System

This project uses **YOLOv8**, **OpenCV**, and **Streamlit** to perform real-time object detection with a browser-based interface. It supports region-based alerts, automated snapshot capture, and event logging for objects entering specified areas of interest. The system is designed for quick demonstration purposes but can be adapted for production use with appropriate optimizations.

## Features

* **Real-time object detection** using YOLOv8 (pre-trained models; no additional training required).
* **Interactive web interface** built with Streamlit for live monitoring and configuration.
* **Restricted zone / ROI (Region of Interest) alerts** triggered when detected objects enter defined areas.
* **Automatic event logging** to CSV with timestamp, label, and detection confidence.
* **Snapshot capture** on alert, stored locally for review.
* Adjustable **confidence threshold** and **frame skipping** to balance accuracy and performance.
* **Target label filtering** to monitor only specific object classes.

## Technology Stack

* **Python**
* **OpenCV** — image and video processing
* **Streamlit** — web-based user interface
* **ultralytics** — YOLOv8 object detection framework
* **pandas** — structured data logging

## Quick Start

```bash
git clone <your-repo-url>
cd real-time-object-detection
./setup.sh
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

---
