# app/utils.py
import cv2
import pandas as pd
import os
import time
import numpy as np

LOG_COLUMNS = ["timestamp", "label", "conf", "x1","y1","x2","y2"]

def draw_boxes(frame, detections, color=(0,255,0), thickness=2, show_label=True):
    for d in detections:
        x1,y1,x2,y2 = map(int, d["box"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
        if show_label:
            text = f"{d['label']} {d['conf']:.2f}"
            cv2.putText(frame, text, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def bbox_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def in_roi(box, roi):
    """
    box: [x1,y1,x2,y2]
    roi: tuple (x_min, y_min, x_max, y_max) in pixel coords
    returns True if center of box inside ROI
    """
    cx, cy = bbox_center(box)
    x_min,y_min,x_max,y_max = roi
    return (cx >= x_min) and (cx <= x_max) and (cy >= y_min) and (cy <= y_max)

def save_log_row(log_path, row_dict):
    header = not os.path.exists(log_path)
    df = pd.DataFrame([row_dict])
    df.to_csv(log_path, mode='a', header=header, index=False)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# utils.py (append at end)

def draw_roi(frame, roi):
    """Draw ROI rectangle on frame"""
    x1, y1, x2, y2 = map(int, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label = "ROI"
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return frame
