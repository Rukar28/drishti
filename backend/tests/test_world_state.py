"""
Unit Tests for Structured World State Management
"""

import time
import pytest
from backend.app.world_state.manager import WorldStateManager
from backend.app.schemas.world_state import SystemMode, ImageDirection, RelativeDepthBand, MotionTrend

def test_world_state_update_and_pruning():
    mgr = WorldStateManager()
    
    # Ingest 2 mock tracks
    tracks = [
        {
            "tracking_id": 1,
            "class_name": "bottle",
            "bbox": [100, 100, 200, 300],
            "norm_bbox": [0.2, 0.2, 0.4, 0.6],
            "norm_center": [0.3, 0.4],
            "confidence": 0.88,
            "direction": ImageDirection.LEFT,
            "relative_depth": RelativeDepthBand.NEAR,
            "relative_depth_score": 0.6,
            "motion": MotionTrend.STATIONARY,
            "first_seen": time.time(),
            "last_seen": time.time()
        },
        {
            "tracking_id": 2,
            "class_name": "laptop",
            "bbox": [300, 200, 500, 400],
            "norm_bbox": [0.45, 0.35, 0.75, 0.7],
            "norm_center": [0.55, 0.5],
            "direction": ImageDirection.CENTER,
            "relative_depth": RelativeDepthBand.VERY_NEAR,
            "relative_depth_score": 0.8,
            "motion": MotionTrend.APPROACHING,
            "first_seen": time.time(),
            "last_seen": time.time()
        }
    ]

    state = mgr.update_from_tracks(tracks, fps=30.0)
    assert len(state.tracked_objects) == 2
    assert state.fps_telemetry == 30.0
    
    # Test class search
    bottles = state.get_objects_by_class("bottle")
    assert len(bottles) == 1
    assert bottles[0].tracking_id == 1
    assert bottles[0].direction == ImageDirection.LEFT

    # Test mode switching
    mgr.set_mode(SystemMode.FIND)
    assert mgr.get_snapshot().current_mode == SystemMode.FIND
