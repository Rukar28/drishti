"""
Unit Tests for Context Narration & Read Mode Text Extraction
"""

import pytest
import numpy as np
import cv2
from backend.app.context.engine import ContextEngine
from backend.app.schemas.world_state import WorldState, TrackedObject, ImageDirection, RelativeDepthBand, MotionTrend
from backend.app.reasoning.ocr import PPOCRv5Provider
from backend.app.modes.read import ReadModeHandler

def test_context_engine_spatial_grouping():
    state = WorldState()
    
    # Bottle and laptop ahead, person on left
    state.tracked_objects[1] = TrackedObject(
        tracking_id=1,
        class_name="laptop",
        confidence=0.90,
        bounding_box=[250, 200, 390, 320],
        normalized_bbox=[0.4, 0.4, 0.6, 0.65],
        normalized_center=[0.5, 0.52],
        direction=ImageDirection.CENTER,
        relative_depth=RelativeDepthBand.NEAR,
        is_stable=True
    )
    state.tracked_objects[2] = TrackedObject(
        tracking_id=2,
        class_name="bottle",
        confidence=0.85,
        bounding_box=[380, 220, 420, 310],
        normalized_bbox=[0.52, 0.45, 0.58, 0.62],
        normalized_center=[0.55, 0.53],
        direction=ImageDirection.CENTER,
        relative_depth=RelativeDepthBand.NEAR,
        is_stable=True
    )
    state.tracked_objects[3] = TrackedObject(
        tracking_id=3,
        class_name="person",
        confidence=0.88,
        bounding_box=[50, 100, 180, 400],
        normalized_bbox=[0.1, 0.2, 0.3, 0.8],
        normalized_center=[0.2, 0.5],
        direction=ImageDirection.LEFT,
        relative_depth=RelativeDepthBand.MID,
        is_stable=True
    )

    narrative, event_key = ContextEngine.generate_scene_narrative(state)
    assert narrative is not None
    assert "laptop" in narrative.lower()
    assert "bottle" in narrative.lower()
    assert "person" in narrative.lower()
    assert "ahead" in narrative.lower()
    assert "left" in narrative.lower()

def test_read_mode_ocr():
    ocr_provider = PPOCRv5Provider()
    read_handler = ReadModeHandler(ocr_provider)
    
    # Create test image with clear text
    test_img = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.putText(test_img, "VISIONMATE ASSISTANT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_img, "AI Lab Room 302", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    res = read_handler.trigger_read(test_img, full_reading=False)
    assert "text" in res
    assert res["has_text"] is True
    assert len(res["text"]) > 0
