"""
P0 — Priority hierarchy, interruption, STOP, stale speech, duplicate suppression,
hazard cooldown and escalation, plus depth-aware hazard reasoning.
"""

import time
import pytest

from backend.app.priority.engine import PriorityEngine, PriorityLevel
from backend.app.speech.tts import WindowsSAPITTSProvider
from backend.app.services.pipeline import VisionMatePipeline
from backend.app.schemas.world_state import (
    WorldState, TrackedObject, ImageDirection, RelativeDepthBand, MotionTrend, HazardSeverity,
)


def test_priority_hierarchy_ordering():
    # Lower int == higher priority (spoken earlier / preempts).
    assert PriorityLevel.STOP < PriorityLevel.EMERGENCY
    assert PriorityLevel.EMERGENCY < PriorityLevel.OBSTACLE
    assert PriorityLevel.OBSTACLE < PriorityLevel.NAVIGATION
    assert PriorityLevel.NAVIGATION < PriorityLevel.USER_COMMAND
    assert PriorityLevel.USER_COMMAND < PriorityLevel.INTERACTION
    assert PriorityLevel.INTERACTION < PriorityLevel.BACKGROUND
    # backward-compatible aliases
    assert PriorityLevel.CRITICAL_HAZARD == PriorityLevel.EMERGENCY
    assert PriorityLevel.AWARENESS == PriorityLevel.INTERACTION


def _hazard_state(severity):
    state = WorldState()
    state.tracked_objects[1] = TrackedObject(
        tracking_id=1, class_name="stairs", confidence=0.92,
        bounding_box=[100, 200, 500, 450], normalized_bbox=[0.2, 0.4, 0.8, 0.9],
        normalized_center=[0.5, 0.65], direction=ImageDirection.CENTER,
        relative_depth=RelativeDepthBand.NEAR, motion_estimate=MotionTrend.STATIONARY,
        is_stable=True, hazard_severity=severity,
    )
    return state


def test_emergency_hazard_interrupts_and_not_repeated():
    engine = PriorityEngine(hazard_cooldown=3.0)
    state = _hazard_state(HazardSeverity.EMERGENCY)
    first = engine.evaluate_hazard(state)
    assert first is not None
    assert first["priority"] == PriorityLevel.CRITICAL_HAZARD
    assert first["interrupt"] is True
    # must not spam every frame
    assert engine.evaluate_hazard(state) is None


def test_hazard_escalation_warning_to_emergency():
    engine = PriorityEngine(hazard_cooldown=5.0)
    warning = _hazard_state(HazardSeverity.WARNING)
    first = engine.evaluate_hazard(warning)
    assert first is not None
    assert first["priority"] == PriorityLevel.NAVIGATION
    # escalate the same event
    escalation = _hazard_state(HazardSeverity.EMERGENCY)
    second = engine.evaluate_hazard(escalation)
    assert second is not None
    assert second["priority"] == PriorityLevel.CRITICAL_HAZARD
    assert second["interrupt"] is True


def test_guidance_duplicate_suppression_same_event_key():
    engine = PriorityEngine(guidance_cooldown=0.0)
    key = "guidance:ahead_bottle"
    text = "A bottle is ahead."
    assert engine.should_speak_guidance(key, text) is True
    # identical narrative for the same event must be suppressed
    assert engine.should_speak_guidance(key, text) is False


def test_stale_speech_is_dropped_after_stop():
    tts = WindowsSAPITTSProvider()
    old_gen = tts.speech_generation
    tts.speak("obsolete guidance", priority=PriorityLevel.NAVIGATION)
    tts.stop()
    assert tts.speech_generation > old_gen
    assert tts._queue.empty()
    # A late item tagged with the pre-STOP generation must be rejected outright.
    tts.speak("late stale item", priority=PriorityLevel.NAVIGATION, generation_id=old_gen)
    assert tts._queue.empty()


def test_stop_command_purges_and_invalidates_session():
    pipe = VisionMatePipeline(camera_source="mock")
    pipe.start_find("bottle")
    gen_before = pipe.voice_fsm.generation
    res = pipe.handle_stop_command()
    assert res["status"] == "STOPPED"
    assert pipe.world_state_mgr.get_snapshot().current_mode.value in ("guidance", "awareness")
    assert pipe.tts._queue.empty()
    assert pipe.voice_fsm.generation > gen_before


def test_depth_hazard_only_when_connected():
    engine = PriorityEngine(hazard_cooldown=3.0)
    # Mock reports not connected -> never a hazard
    assert engine.evaluate_depth_hazard({"connected": False, "center_mm": 100}) is None
    # Far away -> no hazard
    assert engine.evaluate_depth_hazard({"connected": True, "center_mm": 4000}) is None
    # Close -> hazard, then cooldown suppresses repeats
    near = {"connected": True, "center_mm": 400}
    first = engine.evaluate_depth_hazard(near)
    assert first is not None
    assert first["priority"] == PriorityLevel.EMERGENCY
    assert first["interrupt"] is True
    assert engine.evaluate_depth_hazard(near) is None
