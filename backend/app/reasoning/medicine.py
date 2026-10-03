"""
VisionMate — ON-DEMAND medicine / product label recognition (P5).

Never runs continuously. Triggered only by an explicit request such as
"what medicine is this?". Uses the existing OCR provider to read the visible
product label text and reports the recognized text/product name.

STRICT SAFETY CONTRACT:
  * Recognition is NOT medical advice. We only read and repeat visible text.
  * We never interpret, recommend, dose, or claim medical meaning.
  * Spoken output always separates "recognized text" from "medical interpretation"
    and includes an explicit disclaimer when a medicine-like request is made.
  * If nothing is readable we say so; we never invent a drug name.
"""

import re
from typing import Any, Dict, List, Optional

from backend.app.core.interfaces import OCRProvider
from backend.app.priority.engine import PriorityLevel
import logging

logger = logging.getLogger("visionmate.medicine")

_DISCLAIMER = (
    "This is only the text I can read on the label. It is not medical advice. "
    "Please confirm with a pharmacist or doctor."
)

# Common label tokens that hint this is a medicine/product label (recognition only).
_MEDICINE_HINTS = [
    "mg", "ml", "tablet", "tablets", "capsule", "capsules", "syrup", "drops",
    "ointment", "injection", "ip", "usp", "batch", "exp", "mfg", "dosage",
    "store below", "keep out of reach",
]


class MedicineRecognizer:
    """ON-DEMAND medicine/product label recognizer built on the shared OCR provider."""

    def __init__(self, ocr_provider: OCRProvider):
        self.ocr_provider = ocr_provider

    @staticmethod
    def _extract_candidates(text: str) -> List[str]:
        """Best-effort: pull probable product-name / strength tokens from OCR text."""
        candidates: List[str] = []

        # strength patterns e.g. "500 mg", "5 ml", "10mg"
        for m in re.findall(r"\b\d+\s?(?:mg|ml|mcg|g|iu)\b", text, flags=re.IGNORECASE):
            candidates.append(m.strip())

        # product-name-like capitalised words (>=3 letters), skipping boilerplate
        stop = {"the", "and", "for", "with", "tablet", "tablets", "capsule", "capsules", "syrup"}
        for word in re.findall(r"\b[A-Z][A-Za-z\-]{2,}\b", text):
            if word.lower() not in stop and word not in candidates:
                candidates.append(word)

        return candidates[:6]

    def recognize(self, frame) -> Dict[str, Any]:
        """Runs OCR on the provided frame and reports recognized product/label text."""
        if frame is None:
            return {
                "text": "Cannot read a medicine label. No camera frame available.",
                "has_result": False,
                "recognized_text": "",
                "candidates": [],
                "is_medical_advice": False,
                "disclaimer": _DISCLAIMER,
                "confidence": 0.0,
                "priority": PriorityLevel.INTERACTION,
            }

        ocr = self.ocr_provider.extract_text(frame)
        recognized = (ocr.get("full_text") or "").strip()
        has_text = bool(ocr.get("has_text")) and bool(recognized)

        if not has_text:
            return {
                "text": (
                    "I could not read any label text. Please hold the medicine closer and steady. "
                    + _DISCLAIMER
                ),
                "has_result": False,
                "recognized_text": "",
                "candidates": [],
                "is_medical_advice": False,
                "disclaimer": _DISCLAIMER,
                "confidence": 0.0,
                "priority": PriorityLevel.INTERACTION,
            }

        candidates = self._extract_candidates(recognized)
        low = recognized.lower()
        looks_like_medicine = any(h in low for h in _MEDICINE_HINTS)

        if candidates:
            product_part = f"Recognized product text: {', '.join(candidates)}."
        else:
            product_part = "I could read the label text but not a clear product name."

        recognized_part = f" Full label reads: {recognized}."
        spoken = product_part + recognized_part + " " + _DISCLAIMER

        logger.info("[MEDICINE] recognized=%r candidates=%s medicine_like=%s",
                    recognized, candidates, looks_like_medicine)

        return {
            "text": spoken,
            "has_result": looks_like_medicine or bool(candidates),
            "recognized_text": recognized,
            "candidates": candidates,
            "is_medical_advice": False,
            "disclaimer": _DISCLAIMER,
            "confidence": 0.6 if looks_like_medicine else 0.3,
            "priority": PriorityLevel.INTERACTION,
        }