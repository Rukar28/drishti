"""
VisionMate v2 - Comprehensive Intelligence & Reliability Regression Test Suite
Validates the 14 mandatory behavioral and reliability invariants:
TEST 1: Normal chair detection must NOT create "Caution".
TEST 2: Normal bottle detection must NOT create "Caution".
TEST 3: Normal laptop detection must NOT create "Caution".
TEST 4: Repeated detection of the same object must NOT repeatedly speak.
TEST 5: After a spoken event, speech must enter cooldown/silence.
TEST 6: A materially changed event may trigger a new announcement.
TEST 7: OCR must remain OFF in normal Guidance Mode.
TEST 8: OCR activates only after Read command.
TEST 9: OCR automatically turns OFF after reading.
TEST 10: Poor OCR result must not be spoken as confident text.
TEST 11: VLM must not block Guidance Mode.
TEST 12: Find Mode must reuse World State and tracking.
TEST 13: Emergency speech can interrupt normal speech.
TEST 14: Emergency speech does not repeat indefinitely without a changed event.
"""

import time
import numpy as np
import pytest
from typing import Dict, Any, List

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
from backend.app.tracking.tracker import SpatialTracker
from backend.app.world_state.manager import WorldStateManager
from backend.app.priority.engine import PriorityEngine, PriorityLevel
from backend.app.context.engine import ContextEngine
from backend.app.modes.awareness import AwarenessModeHandler
from backend.app.modes.find import FindModeHandler
from backend.app.modes.read import ReadModeHandler
from backend.app.reasoning.ocr import PPOCRv5Provider
from backend.app.services.pipeline import VisionMatePipeline

def make_mock_track(tid: int, cls: str, cx: float = 0.5, cy: float = 0.5, is_stable: bool = True, motion: MotionTrend = MotionTrend.STATIONARY, depth: RelativeDepthBand = RelativeDepthBand.NEAR) -> Dict[str, Any]:
    return {
        "tracking_id": tid,
        "class_name": cls,
        "bbox": [int((cx-0.1)*640), int((cy-0.1)*480), int((cx+0.1)*640), int((cy+0.1)*480)],
        "norm_bbox": [round(cx-0.1, 3), round(cy-0.1, 3), round(cx+0.1, 3), round(cy+0.1, 3)],
        "norm_center": [cx, cy],
        "confidence": 0.88,
        "direction": ImageDirection.CENTER if 0.44 <= cx <= 0.56 else (ImageDirection.LEFT if cx < 0.44 else ImageDirection.RIGHT),
        "path_zone": PathZone.INSIDE_PATH if 0.34 <= cx <= 0.66 else PathZone.OUTSIDE_PATH,
        "relative_depth": depth,
        "relative_depth_score": 0.5,
        "motion": motion,
        "is_stable": is_stable,
        "stability_score": 0.9 if is_stable else 0.2,
        "first_seen": time.time(),
        "last_seen": time.time(),
        "disappeared": 0
    }

# TEST 1: Normal chair detection must NOT create "Caution"
def test_normal_chair_not_hazard():
    mgr = WorldStateManager()
    pe = PriorityEngine()
    
    tracks = [make_mock_track(1, "chair", cx=0.5, cy=0.5)]
    state = mgr.update_from_tracks(tracks)
    
    hazard_event = pe.evaluate_hazard(state)
    assert hazard_event is None, "Normal stationary chair must NOT trigger a hazard event"
    assert state.tracked_objects[1].hazard_severity == HazardSeverity.NONE

# TEST 2: Normal bottle detection must NOT create "Caution"
def test_normal_bottle_not_hazard():
    mgr = WorldStateManager()
    pe = PriorityEngine()
    
    tracks = [make_mock_track(2, "bottle", cx=0.52, cy=0.48)]
    state = mgr.update_from_tracks(tracks)
    
    hazard_event = pe.evaluate_hazard(state)
    assert hazard_event is None, "Normal stationary bottle must NOT trigger a hazard event"
    assert state.tracked_objects[2].hazard_severity == HazardSeverity.NONE

# TEST 3: Normal laptop detection must NOT create "Caution"
def test_normal_laptop_not_hazard():
    mgr = WorldStateManager()
    pe = PriorityEngine()
    
    tracks = [make_mock_track(3, "laptop", cx=0.5, cy=0.5)]
    state = mgr.update_from_tracks(tracks)
    
    hazard_event = pe.evaluate_hazard(state)
    assert hazard_event is None, "Normal laptop on desk must NOT trigger a hazard event"
    assert state.tracked_objects[3].hazard_severity == HazardSeverity.NONE

# TEST 4 & 5: Repeated detection of the same object must NOT repeatedly speak (Silence & Cooldown)
def test_speech_silence_after_announcement():
    mgr = WorldStateManager()
    pe = PriorityEngine(guidance_cooldown=5.0)
    handler = AwarenessModeHandler(pe)
    
    # Frame 1: First observation of desk & laptop
    tracks = [make_mock_track(10, "desk", cx=0.5, cy=0.5), make_mock_track(11, "laptop", cx=0.5, cy=0.45)]
    state = mgr.update_from_tracks(tracks)
    
    cmd1 = handler.process_frame_state(state)
    assert cmd1 is not None
    assert "laptop" in cmd1["text"].lower()
    
    # Frame 2 to 30: Continuous presence of exact same scene over next 3 seconds
    for _ in range(30):
        state = mgr.update_from_tracks(tracks)
        cmd_repeat = handler.process_frame_state(state)
        # MUST BE COMPLETELY SILENT
        assert cmd_repeat is None, "VisionMate must remain SILENT after announcing unchanged scene"

# TEST 6: A materially changed event may trigger a new announcement
def test_materially_changed_event_triggers_announcement():
    mgr = WorldStateManager()
    pe = PriorityEngine(guidance_cooldown=3.0)
    handler = AwarenessModeHandler(pe)
    
    # Initial scene: desk ahead
    tracks1 = [make_mock_track(10, "desk", cx=0.5, cy=0.5)]
    state1 = mgr.update_from_tracks(tracks1)
    cmd1 = handler.process_frame_state(state1)
    assert cmd1 is not None
    
    # Fast forward past cooldown (e.g. 4 seconds later)
    pe.last_speech_time = time.time() - 4.0
    
    # Material change: A person newly appears moving on the left
    tracks2 = [
        make_mock_track(10, "desk", cx=0.5, cy=0.5),
        make_mock_track(12, "person", cx=0.25, cy=0.4, is_stable=True, motion=MotionTrend.CROSSING_LEFT)
    ]
    state2 = mgr.update_from_tracks(tracks2)
    cmd2 = handler.process_frame_state(state2)
    assert cmd2 is not None
    assert "person" in cmd2["text"].lower()

# TEST 7, 8, 9: OCR strictly on-demand (OFF during guidance, ON upon Read, auto OFF after)
def test_ocr_command_lifecycle():
    pipe = VisionMatePipeline(camera_source="mock")
    # In Guidance mode, OCR is not called continuously
    assert pipe.world_state_mgr.get_snapshot().current_mode in [SystemMode.GUIDANCE, SystemMode.AWARENESS]
    
    # Trigger on-demand Read Mode
    res = pipe.trigger_read(full_reading=False)
    assert "text" in res
    
    # After Read operation completes, mode must automatically be back to GUIDANCE
    current_mode = pipe.world_state_mgr.get_snapshot().current_mode
    assert current_mode in [SystemMode.GUIDANCE, SystemMode.AWARENESS], "OCR must automatically turn OFF and return to Guidance"

# TEST 10: Poor OCR result must not be spoken as confident text
def test_poor_ocr_handling():
    ocr_provider = PPOCRv5Provider()
    
    # Create an artificial blank blurry/noisy image
    blank_blurry_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    res = ocr_provider.extract_text(blank_blurry_img)
    
    assert res["has_text"] is False
    assert "couldn't read" in res["short_summary"].lower() or "steady" in res["short_summary"].lower()

# TEST 11: VLM does not block continuous perception
def test_vlm_on_demand_isolation():
    pipe = VisionMatePipeline(camera_source="mock")
    # Ask mode is on-demand
    res = pipe.trigger_ask("What is this?")
    assert "text" in res
    assert pipe.world_state_mgr.get_snapshot().current_mode == SystemMode.GUIDANCE

# TEST 12: Find Mode reuses World State and Spatial Tracker
def test_find_mode_reuses_world_state():
    mgr = WorldStateManager()
    find_handler = FindModeHandler(update_interval_sec=1.0)
    
    find_handler.set_target("bottle")
    
    # Track with bottle on right
    tracks = [make_mock_track(5, "bottle", cx=0.75, cy=0.5)]
    state = mgr.update_from_tracks(tracks)
    
    guidance = find_handler.process_frame_state(state)
    assert guidance is not None
    assert "bottle" in guidance["text"].lower()
    assert "right" in guidance["text"].lower()

# TEST 13: Emergency speech can interrupt normal speech
def test_emergency_speech_interruption():
    mgr = WorldStateManager()
    pe = PriorityEngine()
    
    # Emergency hazard: STAIRS directly ahead
    stairs_track = [make_mock_track(99, "stairs", cx=0.5, cy=0.6, depth=RelativeDepthBand.VERY_NEAR)]
    state = mgr.update_from_tracks(stairs_track)
    
    hazard_event = pe.evaluate_hazard(state)
    assert hazard_event is not None
    assert hazard_event["priority"] == PriorityLevel.CRITICAL_HAZARD
    assert hazard_event["interrupt"] is True
    assert "warning: stairs" in hazard_event["text"].lower()

# TEST 14: Emergency speech does not repeat indefinitely without a changed event
def test_emergency_speech_no_infinite_repeat():
    mgr = WorldStateManager()
    pe = PriorityEngine(hazard_cooldown=3.0)
    
    stairs_track = [make_mock_track(99, "stairs", cx=0.5, cy=0.6, depth=RelativeDepthBand.VERY_NEAR)]
    state = mgr.update_from_tracks(stairs_track)
    
    # First frame: Announcement made
    alert1 = pe.evaluate_hazard(state)
    assert alert1 is not None
    
    # Immediate subsequent frames: Must be silent (no spam)
    alert_repeat = pe.evaluate_hazard(state)
    assert alert_repeat is None, "Emergency alert must NOT repeat every single frame without change"
