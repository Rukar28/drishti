"""
P5 — On-demand currency and medicine recognition, plus Command Bus wiring.

Recognition only reports text that is actually read. Medicine output must never
be presented as medical advice.
"""

import pytest

from backend.app.reasoning.currency import CurrencyRecognizer
from backend.app.reasoning.medicine import MedicineRecognizer
from backend.app.speech.asr import LocalASRProvider
from backend.app.services.pipeline import VisionMatePipeline
from backend.app.priority.engine import PriorityLevel


class _FakeOCR:
    """Minimal OCRProvider stand-in returning deterministic text."""
    def __init__(self, text: str, has_text: bool = True):
        self._text = text
        self._has = has_text

    def extract_text(self, image):
        return {
            "full_text": self._text,
            "short_summary": self._text,
            "text_blocks": [],
            "has_text": self._has and bool(self._text),
        }


# ── Currency ─────────────────────────────────────────────────────
def test_currency_detects_inr_denomination():
    rec = CurrencyRecognizer(_FakeOCR("Reserve Bank of India 500 rupees note"), region="INR")
    res = rec.recognize(object())  # frame truthiness is checked only for None
    assert res["currency"] == "INR"
    assert res["denomination"] == 500
    assert res["has_result"] is True
    assert "500" in res["text"]


def test_currency_no_text_is_reported_honestly():
    rec = CurrencyRecognizer(_FakeOCR("", has_text=False))
    res = rec.recognize(object())
    assert res["has_result"] is False
    assert res["denomination"] is None
    assert "could not" in res["text"].lower()


def test_currency_none_frame_is_handled():
    rec = CurrencyRecognizer(_FakeOCR("anything"))
    res = rec.recognize(None)
    assert res["has_result"] is False
    assert res["recognized_text"] == ""


# ── Medicine ─────────────────────────────────────────────────────
def test_medicine_extracts_product_text_and_disclaims():
    rec = MedicineRecognizer(_FakeOCR("Paracetamol 500 mg tablets batch 2034 exp 2027"))
    res = rec.recognize(object())
    assert res["recognized_text"]
    assert res["is_medical_advice"] is False
    assert any("500 mg" in c for c in res["candidates"]) or res["candidates"]
    assert "not medical advice" in res["text"].lower()
    assert res["disclaimer"]


def test_medicine_no_text_is_handled():
    rec = MedicineRecognizer(_FakeOCR("", has_text=False))
    res = rec.recognize(object())
    assert res["has_result"] is False
    assert res["is_medical_advice"] is False


# ── Command Bus wiring ───────────────────────────────────────────
def test_intent_parser_maps_currency_and_medicine():
    asr = LocalASRProvider()
    assert asr.parse_intent("what currency is this?")["intent"] == "CURRENCY"
    assert asr.parse_intent("what medicine is this?")["intent"] == "MEDICINE"
    assert asr.parse_intent("what is this pill")["intent"] == "MEDICINE"


def test_pipeline_registers_real_handlers_not_stub():
    pipe = VisionMatePipeline(camera_source="mock")
    # These must no longer be the not-available stub.
    assert pipe.command_bus._handlers["CURRENCY"].__name__ == "_cmd_currency"
    assert pipe.command_bus._handlers["MEDICINE"].__name__ == "_cmd_medicine"
    assert pipe.command_bus._handlers["NAVIGATION"].__name__ == "_cmd_navigation"
    # Deferred features remain explicitly unavailable.
    assert pipe.command_bus._handlers["FACE"].__name__ == "_cmd_not_available"
    assert pipe.command_bus._handlers["SOS"].__name__ == "_cmd_not_available"
