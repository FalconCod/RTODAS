import streamlit as st
import cv2
import time
import numpy as np
import os
import pandas as pd
from detector import YoloDetector
from utils import draw_boxes, save_log_row, ensure_dir
import plotly.express as px
from datetime import datetime

st.cache_data.clear()
st.cache_resource.clear()
st.set_page_config(page_title="AI Surveillance Pro", layout="wide")

st.markdown("<h1 style='text-align: center;'>AI Surveillance Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Real-time object detection with analytics & alert logging</p>", unsafe_allow_html=True)

log_dir = "logs"
ensure_dir(log_dir)
snapshots_dir = os.path.join(log_dir, "snapshots")
ensure_dir(snapshots_dir)

with st.sidebar:
    st.markdown("### Controls")
    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    st.markdown("### Detection Settings")
    model_option = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"])
    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    frame_skip = st.slider("Process every Nth frame", 1, 20, 2)
    save_snapshots = st.checkbox("Save snapshots on detection", value=True)
    alert_hold_time = st.slider("Minimum seconds before logging", 1, 5, 2)

    st.markdown("### Session History")
    session_files = sorted([f for f in os.listdir(log_dir) if f.startswith("session_") and f.endswith(".csv")], reverse=True)
    selected_session = None
    if session_files:
        selected_session = st.selectbox("Select a session log", session_files)
        if st.button("Load Session Log"):
            df_prev = pd.read_csv(os.path.join(log_dir, selected_session))
            st.session_state.history_df = df_prev

if "detector" not in st.session_state:
    st.session_state.detector = YoloDetector(model_name=model_option, conf=confidence)
    st.session_state.started = False
    st.session_state.object_in_roi_since = {}
    st.session_state.stop = False
    st.session_state.current_session_log = None
    st.session_state.history_df = None

if start:
    st.session_state.started = True
    st.session_state.stop = False
    st.session_state.current_session_log = os.path.join(
        log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
if stop:
    st.session_state.stop = True

def process_frame(frame):
    detections = st.session_state.detector.detect(frame)
    now = time.time()
    for d in detections:
        label_id = f"{d['label']}_{d['conf']:.2f}"
        if label_id not in st.session_state.object_in_roi_since:
            st.session_state.object_in_roi_since[label_id] = now
        else:
            if now - st.session_state.object_in_roi_since[label_id] >= alert_hold_time:
                row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "label": d["label"],
                    "conf": d["conf"],
                    "x1": d["box"][0],
                    "y1": d["box"][1],
                    "x2": d["box"][2],
                    "y2": d["box"][3]
                }
                save_log_row(st.session_state.current_session_log, row)
                if save_snapshots:
                    fname = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{d['label']}.jpg"
                    cv2.imwrite(os.path.join(snapshots_dir, fname), frame)
    draw_boxes(frame, detections)
    return frame, detections

video_placeholder = st.empty()
stats_placeholder = st.empty()
log_placeholder = st.empty()

if st.session_state.started:
    cap = cv2.VideoCapture(0)
    frame_id = 0
    detection_counts = []

    while True:
        if st.session_state.stop:
            video_placeholder.empty()
            stats_placeholder.empty()
            log_placeholder.empty()
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip != 0:
            continue

        frame_proc, detections = process_frame(frame.copy())
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB", use_container_width=True, clamp=True, output_format="JPEG")

        counts = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        detection_counts.append(counts)

        df_counts = pd.DataFrame(detection_counts).fillna(0).astype(int)
        fig = px.line(df_counts, title="Detections Over Time")
        stats_placeholder.plotly_chart(fig, use_container_width=True, key=f"stats_chart_{frame_id}")

        if len(detection_counts) > 50:
            detection_counts.pop(0)

    cap.release()

if st.session_state.history_df is not None:
    st.subheader("Loaded Session Log")
    st.dataframe(st.session_state.history_df, use_container_width=True)
