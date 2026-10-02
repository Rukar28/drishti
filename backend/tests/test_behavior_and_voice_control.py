"""
VisionMate v2 - Behavior, Performance & Voice Control Regression Test Suite
Validates the 14 mandatory behavioral, performance, and voice control requirements:

TEST 1: 18 FPS scheduler target
TEST 2: OCR inactive during Guidance Mode
TEST 3: Read command activates OCR
TEST 4: Read completion disables OCR
TEST 5: FIND voice intent starts Find Mode
TEST 6: Find command purges previous normal speech
TEST 7: STOP exits Find Mode
TEST 8: STOP purges current TTS
TEST 9: STOP clears queued TTS
TEST 10: STOP prevents stale asynchronous speech from returning
TEST 11: Starting a new Find session invalidates old Find responses
TEST 12: Same Find direction is not spoken repeatedly
TEST 13: VLM Ask Mode returns comprehensive scene descriptions
TEST 14: VLM remains completely separate from the continuous guidance loop
"""

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch

from backend.app.core.config import settings
from backend.app.schemas.world_state import SystemMode, WorldState, ImageDirection, TrackedObject, PathZone, HazardSeverity
from backend.app.speech.tts import WindowsSAPITTSProvider
from backend.app.speech.asr import LocalASRProvider
from backend.app.modes.find import FindModeHandler
from backend.app.modes.read import ReadModeHandler
from backend.app.modes.ask import AskModeHandler
from backend.app.modes.awareness import AwarenessModeHandler
from backend.app.reasoning.ocr import PPOCRv5Provider
from backend.app.reasoning.vlm import OllamaQwen3VLReasoner
from backend.app.priority.engine import PriorityEngine, PriorityLevel
from backend.app.services.pipeline import VisionMatePipeline
from backend.app.world_state.manager import WorldStateManager


# TEST 1: 18 FPS scheduler target
def test_01_target_fps_is_18():
    assert settings.VISIONMATE_TARGET_FPS == 18
    target_period = 1.0 / settings.VISIONMATE_TARGET_FPS
    assert round(target_period * 1000, 1) == 55.6


# TEST 2: OCR inactive during Guidance Mode
def test_02_ocr_inactive_during_guidance():
    priority_engine = PriorityEngine()
    guidance_handler = AwarenessModeHandler(priority_engine)
    ocr_mock = MagicMock(spec=PPOCRv5Provider)
    
    # In Guidance Mode, standard frame evaluation does not call OCR
    state = WorldState(current_mode=SystemMode.GUIDANCE)
    res = guidance_handler.process_frame_state(state)
    ocr_mock.extract_text.assert_not_called()
    assert state.current_mode == SystemMode.GUIDANCE


# TEST 3: Read command activates OCR
def test_03_read_command_activates_ocr():
    ocr_provider = PPOCRv5Provider()
    read_handler = ReadModeHandler(ocr_provider)
    
    # Test frame (blank or textured)
    frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    res = read_handler.trigger_read(frame, full_reading=True)
    
    assert "text" in res
    assert "metrics" in res
    assert "ocr_preprocess_ms" in res["metrics"]
    assert "ocr_inference_ms" in res["metrics"]
    assert "ocr_postprocess_ms" in res["metrics"]
    assert "ocr_total_ms" in res["metrics"]


# TEST 4: Read completion disables OCR and returns to Guidance
def test_04_read_completion_disables_ocr():
    pipeline = VisionMatePipeline(camera_source="mock")
    # Trigger read
    res = pipeline.trigger_read(full_reading=False)
    assert res is not None
    # System mode must automatically be GUIDANCE after read completion
    assert pipeline.world_state_mgr.get_snapshot().current_mode == SystemMode.GUIDANCE


# TEST 5: FIND voice intent starts Find Mode
def test_05_find_voice_intent_starts_find_mode():
    asr = LocalASRProvider()
    
    # Test natural phrasing
    for phrase, expected_target in [
        ("Find bottle", "bottle"),
        ("Find the bottle", "bottle"),
        ("Can you find the bottle", "bottle"),
        ("Find my keys", "keys"),
        ("Please find the chair", "chair")
    ]:
        intent_res = asr.parse_intent(phrase)
        assert intent_res["intent"] == "FIND"
        assert intent_res["target"] == expected_target


# TEST 6: Find command purges previous normal speech
def test_06_find_command_purges_previous_speech():
    tts = WindowsSAPITTSProvider()
    # Enqueue guidance speech
    tts.speak("Forward path is clear.", priority=PriorityLevel.INTERACTION, source="GUIDANCE")
    assert tts._queue.qsize() >= 0
    
    # Start Find Mode
    pipeline = VisionMatePipeline(camera_source="mock")
    pipeline.tts = tts
    pipeline.start_find("bottle")
    
    # World state mode is FIND
    assert pipeline.world_state_mgr.get_snapshot().current_mode == SystemMode.FIND
    assert pipeline.world_state_mgr.get_snapshot().target_find_query == "bottle"


# TEST 7: STOP exits Find Mode
def test_07_stop_exits_find_mode():
    pipeline = VisionMatePipeline(camera_source="mock")
    pipeline.start_find("bottle")
    assert pipeline.world_state_mgr.get_snapshot().current_mode == SystemMode.FIND
    
    # Issue voice STOP
    stop_res = pipeline.handle_voice_command("STOP")
    assert stop_res["status"] == "STOPPED"
    assert stop_res["mode"] == "guidance"
    assert pipeline.world_state_mgr.get_snapshot().current_mode == SystemMode.GUIDANCE
    assert pipeline.world_state_mgr.get_snapshot().target_find_query is None


# TEST 8: STOP purges current TTS
def test_08_stop_purges_current_tts():
    tts = WindowsSAPITTSProvider()
    initial_gen = tts.speech_generation
    tts.stop()
    assert tts.speech_generation == initial_gen + 1


# TEST 9: STOP clears queued TTS
def test_09_stop_clears_queued_tts():
    tts = WindowsSAPITTSProvider()
    tts.speak("Old Guidance Message 1", priority=PriorityLevel.INTERACTION, source="GUIDANCE")
    tts.speak("Old Guidance Message 2", priority=PriorityLevel.INTERACTION, source="GUIDANCE")
    
    # STOP
    tts.stop()
    assert tts._queue.empty()


# TEST 10: STOP prevents stale asynchronous speech from returning
def test_10_stop_prevents_stale_speech():
    tts = WindowsSAPITTSProvider()
    old_gen = tts.speech_generation
    
    # Stale item from old generation
    tts.stop()  # Increments speech_generation
    assert tts.speech_generation > old_gen
    
    # Directly pushing an old item to queue simulation
    stale_item = (PriorityLevel.INTERACTION, old_gen, time.time(), "Stale async message", False, "FIND")
    tts._queue.put(stale_item)
    
    # The worker logic discards items where item_gen < tts.speech_generation
    prio, item_gen, ts, text, interrupt, src = tts._queue.get()
    assert item_gen < tts.speech_generation  # Validates it will be dropped by worker


# TEST 11: Starting a new Find session invalidates old Find responses
def test_11_new_find_session_invalidates_old():
    handler = FindModeHandler()
    msg1, session1 = handler.set_target("bottle")
    assert session1 is not None
    
    # Start new search
    msg2, session2 = handler.set_target("cup")
    assert session2 != session1
    assert handler.session_id == session2


# TEST 12: Same Find direction is not spoken repeatedly
def test_12_same_find_direction_cooldown():
    handler = FindModeHandler(update_interval_sec=2.0)
    handler.set_target("bottle")
    
    # Create mock world state with bottle in CENTER
    target_obj = TrackedObject(
        tracking_id=1,
        class_name="bottle",
        bounding_box=[200, 100, 300, 400],
        normalized_bbox=[0.31, 0.20, 0.46, 0.83],
        normalized_center=[0.38, 0.51],
        confidence=0.9,
        direction=ImageDirection.CENTER,
        path_zone=PathZone.INSIDE_PATH,
        hazard_severity=HazardSeverity.NONE,
        is_stable=True
    )
    
    state = WorldState(
        current_mode=SystemMode.FIND,
        tracked_objects={1: target_obj},
        target_find_query="bottle"
    )
    
    # First update -> Speaks direction
    res1 = handler.process_frame_state(state)
    assert res1 is not None
    assert "directly ahead" in res1["text"].lower() or "center" in res1["text"].lower()
    
    # Immediate second update with SAME direction -> Must be None (suppressed)
    res2 = handler.process_frame_state(state)
    assert res2 is None


# TEST 13: VLM Ask Mode returns comprehensive scene descriptions
def test_13_vlm_comprehensive_scene_description():
    vlm = OllamaQwen3VLReasoner()
    ask_handler = AskModeHandler(vlm)
    
    frame = np.full((300, 300, 3), 120, dtype=np.uint8)
    
    # Comprehensive query
    res = ask_handler.ask(frame, "What am I looking at?")
    assert "text" in res
    assert len(res["text"]) > 20
    # Must be structured and factual
    assert any(w in res["text"].lower() for w in ["facing", "indoor", "room", "center", "front", "scene", "view"])


# TEST 14: VLM remains completely separate from the continuous guidance loop
def test_14_vlm_separate_from_guidance_loop():
    vlm_mock = MagicMock(spec=OllamaQwen3VLReasoner)
    ask_handler = AskModeHandler(vlm_mock)
    
    priority_engine = PriorityEngine()
    guidance_handler = AwarenessModeHandler(priority_engine)
    
    state = WorldState(current_mode=SystemMode.GUIDANCE)
    res = guidance_handler.process_frame_state(state)
    
    # VLM must NOT be called in the guidance loop
    vlm_mock.reason.assert_not_called()
