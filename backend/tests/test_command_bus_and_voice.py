"""
Phase 2 command bus + Phase 3 voice intent / VAD tests.
"""

import numpy as np

from backend.app.speech.asr import LocalASRProvider
from backend.app.speech.vad import EnergyVAD
from backend.app.intelligence.command_bus import CommandBus
from backend.app.hardware.depth import create_depth_provider, MockDepthProvider
from backend.app.hardware.gps import create_gps_provider
from backend.app.services.pipeline import VisionMatePipeline
from backend.app.schemas.world_state import SystemMode


def test_must_pass_phrases_parse():
    asr = LocalASRProvider()
    cases = [
        ("Find my bottle", "FIND", "bottle"),
        ("Can you find my bottle?", "FIND", "bottle"),
        ("Where is my bottle?", "FIND", "bottle"),
        ("Stop", "STOP", None),
        ("Read this", "READ", None),
        ("What is in front of me?", "ASK", None),
        ("What's in front of me?", "ASK", None),
    ]
    for phrase, intent, target in cases:
        parsed = asr.parse_intent(phrase)
        assert parsed["intent"] == intent, phrase
        if target:
            assert parsed.get("target") == target, phrase


def test_energy_vad_segments_speech_not_fixed_window():
    vad = EnergyVAD(
        sample_rate=16000,
        frame_ms=30,
        start_rms=0.02,
        end_rms=0.01,
        min_speech_ms=90,
        silence_end_ms=90,
        max_utterance_ms=4000,
        pre_roll_ms=30,
    )
    silence = np.zeros(vad.frame_samples, dtype=np.float32)
    speech = np.ones(vad.frame_samples, dtype=np.float32) * 0.2

    got = None
    for _ in range(8):
        got = vad.accept_frame(silence)
        assert got is None
    for _ in range(6):
        got = vad.accept_frame(speech)
        assert got is None
    for _ in range(5):
        got = vad.accept_frame(silence)
        if got is not None:
            break
    assert got is not None
    duration_s = len(got) / 16000.0
    assert duration_s < 3.0
    assert duration_s > 0.05


def test_command_bus_find_stop_read_ask():
    pipe = VisionMatePipeline(camera_source="mock")
    find_res = pipe.handle_voice_command("Find my bottle")
    assert find_res["intent"] == "FIND"
    assert find_res["target"] == "bottle"
    assert find_res["status"] == "started"
    assert pipe.world_state_mgr.get_snapshot().current_mode == SystemMode.FIND

    stop_res = pipe.handle_voice_command("Stop")
    assert stop_res["intent"] == "STOP"
    assert stop_res["status"] == "STOPPED"
    assert pipe.world_state_mgr.get_snapshot().current_mode == SystemMode.GUIDANCE

    read_res = pipe.handle_voice_command("Read this")
    assert read_res["intent"] == "READ"
    assert read_res["status"] == "started"

    ask_res = pipe.handle_voice_command("What is in front of me?")
    assert ask_res["intent"] == "ASK"
    assert ask_res["status"] == "started"


def test_command_bus_does_not_replace_parser():
    calls = []

    def parser(text):
        calls.append(text)
        return {"intent": "STOP", "confidence": 1.0}

    bus = CommandBus(parser)
    bus.register("STOP", lambda p: {"status": "STOPPED"})
    res = bus.dispatch("stop now")
    assert calls == ["stop now"]
    assert res["status"] == "STOPPED"


def test_depth_and_gps_are_explicit_demo_mode():
    depth = create_depth_provider("mock")
    gps = create_gps_provider("mock")
    assert isinstance(depth, MockDepthProvider)
    assert depth.health()["status"] == "DEMO MODE"
    assert depth.get_distance("center")["distance_m"] is None
    assert gps.health()["status"] == "DEMO MODE"
    assert gps.get_location()["lat"] is None
