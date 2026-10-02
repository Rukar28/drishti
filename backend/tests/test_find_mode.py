"""
Unit Tests for Find Mode Interactive Guidance
"""

import time
import pytest
from backend.app.modes.find import FindModeHandler
from backend.app.schemas.world_state import WorldState, TrackedObject, ImageDirection, RelativeDepthBand, MotionTrend

def test_find_mode_workflow():
    handler = FindModeHandler(update_interval_sec=0.1)
    
    # 1. Set target
    msg, session_id = handler.set_target("bottle")
    assert "bottle" in msg.lower()

    # 2. Scene with bottle on right
    state = WorldState()
    state.tracked_objects[1] = TrackedObject(
        tracking_id=1,
        class_name="bottle",
        confidence=0.89,
        bounding_box=[450, 200, 550, 400],
        normalized_bbox=[0.7, 0.4, 0.85, 0.8],
        normalized_center=[0.75, 0.6],
        direction=ImageDirection.RIGHT,
        relative_depth=RelativeDepthBand.MID,
        motion_estimate=MotionTrend.STATIONARY
    )

    cmd = handler.process_frame_state(state)
    assert cmd is not None
    assert "on your right" in cmd["text"].lower()

    # 3. User turns, bottle moves directly ahead
    time.sleep(0.15)
    state.tracked_objects[1].direction = ImageDirection.CENTER
    cmd2 = handler.process_frame_state(state)
    assert cmd2 is not None
    assert "directly ahead" in cmd2["text"].lower()

    # 4. Bottle lost
    time.sleep(0.15)
    del state.tracked_objects[1]
    # Simulate time passing
    handler.last_guidance_time = time.time() - 3.0
    cmd_lost = handler.process_frame_state(state)
    assert cmd_lost is not None
    assert "lost sight" in cmd_lost["text"].lower()
