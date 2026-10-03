"""Deterministic tests for VisionMate ON-DEMAND COLOR."""

import numpy as np
import pytest

from backend.app.modes.color import ColorModeHandler


def solid_frame(bgr, h=480, w=640):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


@pytest.fixture()
def handler():
    return ColorModeHandler()


TRACKS = [
    {"class_name": "person", "bbox": (100, 100, 300, 460)},
    {"class_name": "bottle", "bbox": (400, 250, 440, 380)},
]


@pytest.mark.parametrize("bgr,expected", [
    ((60, 60, 220), "red"),
    ((30, 140, 250), "orange"),
    ((30, 220, 240), "yellow"),
    ((60, 200, 60), "green"),
    ((220, 120, 40), "blue"),
    ((245, 245, 245), "white"),
    ((10, 10, 10), "black"),
    ((120, 120, 120), "gray"),
])
def test_basic_colors(handler, bgr, expected):
    res = handler.analyze(solid_frame(bgr))
    assert res["success"] is True
    assert res["color"] == expected


def test_dark_and_light_modifiers(handler):
    assert handler.analyze(solid_frame((90, 40, 15)))["color"] == "dark blue"
    assert handler.analyze(solid_frame((250, 220, 170)))["color"] == "light blue"


def test_missing_frame(handler):
    res = handler.analyze(None)
    assert res["success"] is False
    assert "unavailable" in res["text"].lower()


def test_named_object_is_cropped(handler):
    frame = solid_frame((60, 200, 60))
    frame[250:380, 400:440] = (40, 40, 220)
    res = handler.analyze(frame, target="bottle", tracks=TRACKS)
    assert res["matched_object"] == "bottle"
    assert res["color"] == "red"
    assert res["text"] == "The bottle is red."


def test_shirt_maps_to_person_torso(handler):
    frame = solid_frame((120, 120, 120))
    y1, y2 = 100 + int(0.28 * 360), 100 + int(0.72 * 360)
    x1, x2 = 100 + int(0.18 * 200), 300 - int(0.18 * 200)
    frame[y1:y2, x1:x2] = (60, 180, 60)
    res = handler.analyze(frame, target="shirt", tracks=TRACKS)
    assert res["matched_object"] == "person"
    assert res["color"] == "green"
    assert res["text"] == "Your shirt is green."


def test_unmatched_target_falls_back_to_center(handler):
    res = handler.analyze(solid_frame((60, 200, 60)), target="wallet", tracks=TRACKS)
    assert res["matched_object"] is None
    assert "wallet" in res["text"]
    assert res["color"] == "green"


def test_degenerate_bbox_falls_back_to_center(handler):
    bad = [{"class_name": "bottle", "bbox": (0, 0, 2, 2)}]
    res = handler.analyze(solid_frame((220, 120, 40)), target="bottle", tracks=bad)
    assert res["matched_object"] is None
    assert res["color"] == "blue"


def test_target_normalization(handler):
    assert ColorModeHandler._normalize_target("My Shirt!") == "shirt"
    assert ColorModeHandler._normalize_target("do you see") is None
    assert ColorModeHandler._normalize_target("this") is None
    assert ColorModeHandler._normalize_target(None) is None


def test_asr_color_intent_with_target():
    pytest.importorskip("faster_whisper")
    pytest.importorskip("sounddevice")
    from backend.app.speech.asr import LocalASRProvider

    asr = LocalASRProvider()
    assert asr.parse_intent("what color is my shirt")["target"] == "shirt"
    p = asr.parse_intent("Whats the color of this mug?")
    assert (p["intent"], p["target"]) == ("COLOR", "mug")
    assert asr.parse_intent("what color")["target"] is None
    assert asr.parse_intent("what colour is it")["target"] is None


def test_command_bus_routes_color():
    from backend.app.intelligence.command_bus import CommandBus

    seen = {}
    bus = CommandBus(
        lambda t: {"intent": "COLOR", "target": "bottle", "confidence": 0.9}
    )
    bus.register(
        "COLOR",
        lambda p: (
            seen.update(p),
            {"status": "ok", "mode": "color", "message": "The bottle is red."},
        )[1],
    )
    res = bus.dispatch("what color is the bottle")
    assert res["intent"] == "COLOR" and res["status"] == "ok"
    assert seen["target"] == "bottle"
    assert res["message"] == "The bottle is red."


def test_pipeline_wiring_static():
    import inspect
    from backend.app.services.pipeline import VisionMatePipeline

    reg = inspect.getsource(VisionMatePipeline._register_command_handlers)
    assert "self._cmd_color" in reg
    assert hasattr(VisionMatePipeline, "trigger_color")
    assert "ColorModeHandler(" in inspect.getsource(VisionMatePipeline.__init__)
