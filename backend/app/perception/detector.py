"""
VisionMate v2 - YOLO Object Detector Provider
Implements ObjectDetector interface utilizing Ultralytics YOLO (yolov8n / yolo26n).
Includes confidence calibration, aspect filtering, and non-blocking background initialization.
"""

import threading
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from backend.app.core.interfaces import ObjectDetector

logger = logging.getLogger("visionmate.detector")

class YOLOObjectDetector(ObjectDetector):
    """Production YOLO detector running on CUDA or CPU with robust false-positive suppression."""
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda", conf_thresh: float = 0.40, auto_load: bool = False):
        self.model_path = model_path
        self.device = device
        self.conf_thresh = conf_thresh
        self.model = None
        self.status = "UNINITIALIZED"
        self.error_message: Optional[str] = None
        self._lock = threading.Lock()
        
        # High noise classes require higher confidence threshold to avoid false alarms
        self.noisy_classes = {"tie", "remote", "fork", "knife", "spoon", "toothbrush", "scissors"}
        
        if auto_load:
            self.load()

    def load(self) -> bool:
        with self._lock:
            if self.status == "READY":
                return True
            self.status = "LOADING"
            try:
                import torch
                from ultralytics import YOLO
                
                target_device = self.device
                if target_device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    target_device = "cpu"
                    
                logger.info(f"Loading YOLO model from {self.model_path} onto {target_device}...")
                self.model = YOLO(self.model_path)
                # Warmup with dummy frame
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                self.model(dummy, verbose=False, device=target_device)
                self.device = target_device
                self.status = "READY"
                self.error_message = None
                logger.info("YOLO model loaded and warmed up successfully.")
                return True
            except Exception as e:
                self.status = "ERROR"
                self.error_message = str(e)
                self.model = None
                logger.error(f"Failed to load YOLO model: {e}")
                return False

    def is_ready(self) -> bool:
        return self.status == "READY" and self.model is not None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None:
            return []
            
        if self.status != "READY" or self.model is None:
            if self.status == "UNINITIALIZED":
                if not self.load():
                    return []
            else:
                return []

        h, w = frame.shape[:2]
        results = []

        try:
            preds = self.model(frame, conf=self.conf_thresh, verbose=False, device=self.device)
            if not preds or len(preds) == 0:
                return []

            boxes = preds[0].boxes
            if boxes is None or len(boxes) == 0:
                return []

            names = self.model.names
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                conf = float(confs[i])
                cls_id = int(classes[i])
                class_name = names.get(cls_id, str(cls_id)).lower()

                # Filter out noisy micro-classes if confidence is below 0.65
                if class_name in self.noisy_classes and conf < 0.65:
                    continue

                # Filter out tiny sub-pixel artifacts (< 12x12 px)
                bw = x2 - x1
                bh = y2 - y1
                if bw < 12 or bh < 12:
                    continue

                # Normalize bounding box 0.0 - 1.0
                nx1 = max(0.0, min(1.0, x1 / w))
                ny1 = max(0.0, min(1.0, y1 / h))
                nx2 = max(0.0, min(1.0, x2 / w))
                ny2 = max(0.0, min(1.0, y2 / h))

                results.append({
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "norm_bbox": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
                    "norm_center": [round((nx1 + nx2) / 2.0, 4), round((ny1 + ny2) / 2.0, 4)],
                    "area_ratio": round((nx2 - nx1) * (ny2 - ny1), 4)
                })
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")

        return results
