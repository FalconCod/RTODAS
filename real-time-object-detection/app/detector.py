# app/detector.py
from ultralytics import YOLO
import numpy as np

# common COCO classes; YOLO returns class indices
COCO_NAMES = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
 "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
 "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
 "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
 "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
 "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
 "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

class YoloDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.35, iou=0.45, device=None):
        """
        model_name: path or model id. 'yolov8n.pt' downloads smallest model (fast)
        conf: confidence threshold
        iou: nms iou threshold
        device: None (auto) or 'cpu'/'0' etc.
        """
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame):
        """
        frame: BGR numpy array (OpenCV format)
        returns: list of detections: dicts {box: [x1,y1,x2,y2], conf, cls, label}
        """
        # ultralytics accepts BGR numpy arrays
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        # results is a list (batches). single frame => results[0]
        res = results[0]
        detections = []
        boxes = res.boxes  # ultralytics Boxes object
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten().tolist()  # [x1,y1,x2,y2]
            conf = float(box.conf.cpu().numpy().flatten()[0])
            cls_idx = int(box.cls.cpu().numpy().flatten()[0])
            label = COCO_NAMES[cls_idx] if cls_idx < len(COCO_NAMES) else str(cls_idx)
            detections.append({"box": xyxy, "conf": conf, "cls": cls_idx, "label": label})
        return detections
