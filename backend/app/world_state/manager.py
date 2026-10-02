"""
VisionMate v2 - Structured World State Manager
Serves as the single thread-safe source of truth across all system modes and engines.
Maintains 3-tier entity hierarchy: RAW -> STABLE -> EVENT.
"""

import time
import threading
from typing import List, Dict, Any, Optional
from backend.app.schemas.world_state import (
    WorldState,
    TrackedObject,
    SystemMode,
    ImageDirection,
    PathZone,
    HazardSeverity,
    RelativeDepthBand,
    MotionTrend
)

class WorldStateManager:
    """Thread-safe World State Manager managing current scene awareness and confirmed entities."""
    def __init__(self):
        self._state = WorldState()
        self._lock = threading.RLock()

    def update_from_tracks(self, tracks: List[Dict[str, Any]], fps: float = 0.0) -> WorldState:
        with self._lock:
            now = time.time()
            self._state.timestamp = now
            self._state.fps_telemetry = fps
            
            # Update tracked entities dictionary
            active_ids = set()
            for t in tracks:
                tid = t["tracking_id"]
                active_ids.add(tid)
                
                # Compute hazard severity and score
                severity, hazard_score = self._compute_entity_hazard(t)
                
                obj = TrackedObject(
                    tracking_id=tid,
                    class_name=t["class_name"],
                    confidence=t.get("confidence", 1.0),
                    bounding_box=t["bbox"],
                    normalized_bbox=t["norm_bbox"],
                    normalized_center=t["norm_center"],
                    direction=t["direction"],
                    path_zone=t.get("path_zone", PathZone.INSIDE_PATH),
                    relative_depth=t["relative_depth"],
                    relative_depth_score=t["relative_depth_score"],
                    motion_estimate=t["motion"],
                    is_stable=t.get("is_stable", True),
                    stability_score=t.get("stability_score", 1.0),
                    frame_count=t.get("frame_count", 1),
                    first_seen=t["first_seen"],
                    last_seen=t["last_seen"],
                    hazard_severity=severity,
                    hazard_score=hazard_score
                )
                self._state.tracked_objects[tid] = obj

            # Prune stale tracks older than 1.5 seconds
            stale_keys = [
                k for k, v in self._state.tracked_objects.items()
                if (now - v.last_seen) > 1.5
            ]
            for k in stale_keys:
                del self._state.tracked_objects[k]

            return self._state.model_copy(deep=True)

    @staticmethod
    def _compute_entity_hazard(t: Dict[str, Any]) -> tuple[HazardSeverity, float]:
        """
        Calculates true physical hazard severity:
        - Structural fall/step hazard (stairs/dropoff) -> EMERGENCY
        - Approaching vehicle/cyclist in path -> WARNING / EMERGENCY
        - Fast looming dynamic collision inside path -> WARNING
        - Ordinary static objects (chair, desk, bottle, laptop, person standing) -> NONE
        """
        cls = t["class_name"].lower()
        motion = t["motion"]
        path_zone = t.get("path_zone", PathZone.INSIDE_PATH)
        depth = t["relative_depth"]
        is_stable = t.get("is_stable", True)

        # 1. Critical structural drop-offs / stairs (always dangerous)
        if cls in ["stairs", "dropoff", "hole", "step"]:
            if depth in [RelativeDepthBand.VERY_NEAR, RelativeDepthBand.NEAR] and path_zone != PathZone.OUTSIDE_PATH:
                return HazardSeverity.EMERGENCY, 0.95
            return HazardSeverity.WARNING, 0.70

        # 2. Fast moving vehicles/bicycles
        if cls in ["car", "truck", "bus", "motorcycle", "bicycle"]:
            if is_stable and motion in [MotionTrend.APPROACHING, MotionTrend.CROSSING_LEFT, MotionTrend.CROSSING_RIGHT]:
                if path_zone in [PathZone.INSIDE_PATH, PathZone.NEAR_PATH]:
                    return HazardSeverity.EMERGENCY, 0.90
                return HazardSeverity.WARNING, 0.65
            return HazardSeverity.INFORMATION, 0.20

        # 3. Dynamic collisions (only if actively approaching INSIDE forward walking path)
        if is_stable and motion == MotionTrend.APPROACHING and path_zone == PathZone.INSIDE_PATH and depth == RelativeDepthBand.VERY_NEAR:
            return HazardSeverity.WARNING, 0.75

        # 4. Ordinary objects (table, chair, desk, bottle, laptop, normal standing person)
        # Seeing an object is NOT a hazard!
        return HazardSeverity.NONE, 0.0

    def set_mode(self, mode: SystemMode) -> None:
        with self._lock:
            self._state.current_mode = mode

    def set_find_target(self, target_query: Optional[str]) -> None:
        with self._lock:
            self._state.target_find_query = target_query
            if target_query:
                self._state.current_mode = SystemMode.FIND

    def get_snapshot(self) -> WorldState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def record_speech(self, text: str) -> None:
        with self._lock:
            self._state.last_spoken_narrative = text
            self._state.last_speech_timestamp = time.time()
            self._state.recent_events.append({
                "t": time.time(),
                "type": "speech_output",
                "text": text
            })
            if len(self._state.recent_events) > 50:
                self._state.recent_events.pop(0)

# Global world state manager singleton
world_state_mgr = WorldStateManager()
