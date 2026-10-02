"""
VisionMate v2 - Multi-Object Spatial Tracker with Temporal Confirmation
Associates detections across consecutive frames, maintains persistent track IDs,
computes smoothed spatial trajectories, motion trends, and filters out single-frame false positives.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque
from backend.app.core.interfaces import Tracker
from backend.app.schemas.world_state import MotionTrend, ImageDirection, RelativeDepthBand, PathZone
from backend.app.perception.relative_depth import RelativeDepthEstimator

class SpatialTracker(Tracker):
    """
    High-speed IoU and Centroid multi-object tracker with temporal stability confirmation.
    Transitions entities through RAW -> STABLE lifecycle.
    """
    def __init__(self, max_disappeared: int = 15, iou_threshold: float = 0.28, min_stable_frames: int = 3):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.min_stable_frames = min_stable_frames
        self.next_track_id = 1
        
        # Track storage: track_id -> dict
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _compute_iou(boxA: list, boxB: list) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0:
            return 0.0
        return interArea / denom

    @staticmethod
    def _resolve_direction(cx: float) -> ImageDirection:
        """Maps normalized center X into 7 natural directional bands."""
        if cx < 0.20:
            return ImageDirection.FAR_LEFT
        elif cx < 0.38:
            return ImageDirection.LEFT
        elif cx < 0.44:
            return ImageDirection.SLIGHT_LEFT
        elif cx <= 0.56:
            return ImageDirection.CENTER
        elif cx <= 0.62:
            return ImageDirection.SLIGHT_RIGHT
        elif cx <= 0.80:
            return ImageDirection.RIGHT
        else:
            return ImageDirection.FAR_RIGHT

    @staticmethod
    def _resolve_path_zone(cx: float) -> PathZone:
        """Determines if entity intersects user's forward walking corridor."""
        if 0.34 <= cx <= 0.66:
            return PathZone.INSIDE_PATH
        elif 0.20 <= cx < 0.34 or 0.66 < cx <= 0.80:
            return PathZone.NEAR_PATH
        else:
            return PathZone.OUTSIDE_PATH

    @staticmethod
    def _estimate_motion(history: deque) -> MotionTrend:
        """Computes true physical motion vector over recent temporal history."""
        if len(history) < 4:
            return MotionTrend.UNKNOWN

        first_entry = history[0]
        last_entry = history[-1]
        
        # Lateral movement
        dx = last_entry["cx"] - first_entry["cx"]
        # Area growth
        area_delta = last_entry["area"] - first_entry["area"]
        rel_growth = area_delta / max(1e-4, first_entry["area"])

        if dx < -0.14:
            return MotionTrend.CROSSING_LEFT
        elif dx > 0.14:
            return MotionTrend.CROSSING_RIGHT
        elif rel_growth > 0.30:
            return MotionTrend.APPROACHING
        elif rel_growth < -0.30:
            return MotionTrend.RECEDING
        else:
            return MotionTrend.STATIONARY

    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        now = time.time()
        
        # If no tracks exist, initialize all detections
        if len(self.tracks) == 0:
            for det in detections:
                self._create_track(det, now)
            return self._format_active_tracks()

        # If no detections in current frame, increment disappeared count
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["disappeared"] += 1
                if self.tracks[track_id]["disappeared"] > self.max_disappeared:
                    del self.tracks[track_id]
            return self._format_active_tracks()

        # Match existing tracks to detections using IoU and Class Match
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                if self.tracks[tid]["class_name"] == det["class_name"]:
                    iou_matrix[i, j] = self._compute_iou(self.tracks[tid]["bbox"], det["bbox"])
                else:
                    iou_matrix[i, j] = 0.0

        matched_tracks = set()
        matched_dets = set()

        while True:
            max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
            if max_val < self.iou_threshold:
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            tid = track_ids[i]
            
            self._update_track(tid, detections[j], now)
            matched_tracks.add(tid)
            matched_dets.add(j)
            
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        # Handle unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Handle unmatched new detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det, now)

        return self._format_active_tracks()

    def _create_track(self, det: Dict[str, Any], timestamp: float):
        tid = self.next_track_id
        self.next_track_id += 1

        norm_bbox = det["norm_bbox"]
        cx, cy = det["norm_center"]
        area = (norm_bbox[2] - norm_bbox[0]) * (norm_bbox[3] - norm_bbox[1])
        depth_band, depth_score = RelativeDepthEstimator.estimate_relative_depth(norm_bbox, det["class_name"])
        direction = self._resolve_direction(cx)
        path_zone = self._resolve_path_zone(cx)

        history = deque(maxlen=10)
        history.append({"t": timestamp, "cx": cx, "cy": cy, "area": area})

        self.tracks[tid] = {
            "tracking_id": tid,
            "class_name": det["class_name"],
            "bbox": det["bbox"],
            "norm_bbox": norm_bbox,
            "norm_center": [cx, cy],
            "confidence": det["confidence"],
            "disappeared": 0,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "history": history,
            "frame_count": 1,
            "is_stable": False,
            "stability_score": 0.25,
            "relative_depth": depth_band,
            "relative_depth_score": depth_score,
            "direction": direction,
            "path_zone": path_zone,
            "motion": MotionTrend.UNKNOWN,
            "hazard_score": 0.0,
            "hazard_severity": "none"
        }

    def _update_track(self, tid: int, det: Dict[str, Any], timestamp: float):
        t = self.tracks[tid]
        raw_norm_bbox = det["norm_bbox"]
        raw_cx, raw_cy = det["norm_center"]
        raw_area = (raw_norm_bbox[2] - raw_norm_bbox[0]) * (raw_norm_bbox[3] - raw_norm_bbox[1])

        # Exponential Moving Average (EMA) smoothing (alpha=0.6) to remove jitter
        alpha = 0.6
        smoothed_cx = round(alpha * raw_cx + (1 - alpha) * t["norm_center"][0], 4)
        smoothed_cy = round(alpha * raw_cy + (1 - alpha) * t["norm_center"][1], 4)
        
        # Increment frame count and evaluate temporal stability
        t["frame_count"] += 1
        duration = timestamp - t["first_seen"]
        
        # Entity becomes STABLE if seen for >= min_stable_frames or >= 0.25 sec
        if t["frame_count"] >= self.min_stable_frames or duration >= 0.25:
            t["is_stable"] = True
            t["stability_score"] = min(1.0, 0.4 + (t["frame_count"] * 0.15))
        else:
            t["is_stable"] = False
            t["stability_score"] = min(0.6, t["frame_count"] * 0.2)

        depth_band, depth_score = RelativeDepthEstimator.estimate_relative_depth(raw_norm_bbox, det["class_name"])
        direction = self._resolve_direction(smoothed_cx)
        path_zone = self._resolve_path_zone(smoothed_cx)

        t["history"].append({"t": timestamp, "cx": smoothed_cx, "cy": smoothed_cy, "area": raw_area})
        motion = self._estimate_motion(t["history"])

        t["bbox"] = det["bbox"]
        t["norm_bbox"] = raw_norm_bbox
        t["norm_center"] = [smoothed_cx, smoothed_cy]
        t["confidence"] = round(0.5 * det["confidence"] + 0.5 * t["confidence"], 3)
        t["disappeared"] = 0
        t["last_seen"] = timestamp
        t["relative_depth"] = depth_band
        t["relative_depth_score"] = depth_score
        t["direction"] = direction
        t["path_zone"] = path_zone
        t["motion"] = motion

    def _format_active_tracks(self) -> List[Dict[str, Any]]:
        active = []
        for tid, t in self.tracks.items():
            if t["disappeared"] == 0:
                active.append(t)
        return active
