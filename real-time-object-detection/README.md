# Real-Time Object Detection & Alert System

Built with YOLOv8, OpenCV and Streamlit to show real-time detection, region-based alerts, snapshotting, and event logging.

## 🚀 Highlights
- Real-time object detection using **YOLOv8 (pretrained)** — no training required.
- Streamlit UI for quick demo (web-based, local).
- ROI / restricted-zone alerts (trigger when detected object enters a defined area).
- Automatic logging to CSV and snapshot saving on alert.
- Configurable confidence threshold, frame skip for speed, and target label filtering.

## Tech stack
- Python, OpenCV, Streamlit
- ultralytics (YOLOv8)
- pandas for logging

## Quick start
```bash
git clone <your-repo-url>
cd real-time-object-detection
./setup.sh
source .venv/bin/activate
streamlit run app/streamlit_app.py
