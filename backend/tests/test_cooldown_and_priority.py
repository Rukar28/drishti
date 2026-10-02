"""
Unit Tests for Priority Arbitration & Cooldown Suppression
"""

import time
import pytest
from backend.app.priority.engine import PriorityEngine, PriorityLevel
from backend.app.schemas.world_state import WorldState, TrackedObject, ImageDirection, RelativeDepthBand, MotionTrend, HazardSeverity

def test_critical_hazard_detection():
    engine = PriorityEngine(hazard_cooldown=0.5)
    
    # State with descending stairs hazard
    state = WorldState()
    state.tracked_objects[1] = TrackedObject(
        tracking_id=1,
        class_name="stairs",
        confidence=0.92,
        bounding_box=[100, 200, 500, 450],
        normalized_bbox=[0.2, 0.4, 0.8, 0.9],
        normalized_center=[0.5, 0.65],
        direction=ImageDirection.CENTER,
        relative_depth=RelativeDepthBand.NEAR,
        motion_estimate=MotionTrend.STATIONARY,
        is_stable=True,
        hazard_severity=HazardSeverity.EMERGENCY
    )

    hazard = engine.evaluate_hazard(state)
    assert hazard is not None
    assert hazard["priority"] == PriorityLevel.CRITICAL_HAZARD
    assert hazard["interrupt"] is True
    assert "stairs" in hazard["text"].lower()

def test_awareness_speech_cooldown():
    engine = PriorityEngine(guidance_cooldown=0.2)
    narrative = "You have a desk with a laptop ahead."

    # First speech allowed
    assert engine.should_speak_awareness(narrative) is True
    
    # Immediate repeat blocked by cooldown
    assert engine.should_speak_awareness(narrative) is False

    # Wait for cooldown
    time.sleep(0.25)
    # Deduplication applies for identical string
    assert engine.should_speak_awareness(narrative) is False
    
    # New distinct narrative allowed after cooldown
    new_narrative = "A person is walking on your left."
    assert engine.should_speak_awareness(new_narrative) is True
