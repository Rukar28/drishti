"""
VisionMate — ON-DEMAND currency recognition (P5).

Never runs continuously. Triggered only by an explicit user request such as
"what currency is this?". Uses the existing OCR provider to read visible text
(denomination numbers, currency words, issuing authority) and maps that to a
best-effort currency/denomination.

HONESTY CONTRACT:
  * We only report what is actually readable in the frame.
  * If the denomination cannot be determined we say so — we never invent a value.
  * No model/dependency is claimed to be present if it is not.
"""

import re
from typing import Any, Dict, Optional

from backend.app.core.interfaces import OCRProvider
from backend.app.core.config import settings
from backend.app.priority.engine import PriorityLevel
import logging

logger = logging.getLogger("visionmate.currency")


# Valid denominations per region (physical banknotes/coins that actually exist).
_DENOMINATIONS = {
    "INR": [2000, 500, 200, 100, 50, 20, 10, 5, 2, 1],
    "USD": [100, 50, 20, 10, 5, 2, 1],
    "EUR": [500, 200, 100, 50, 20, 10, 5],
}

_CURRENCY_WORDS = {
    "INR": ["rupee", "rupees", "rs", "inr", "reserve bank of india", "bharatiya"],
    "USD": ["dollar", "dollars", "usd", "united states", "federal reserve"],
    "EUR": ["euro", "euros", "eur", "european central bank"],
}

_WORD_NUMBERS = {
    "five": 5, "ten": 10, "twenty": 20, "fifty": 50, "hundred": 100,
    "two hundred": 200, "five hundred": 500, "one thousand": 1000,
    "thousand": 1000, "two thousand": 2000,
}


class CurrencyRecognizer:
    """ON-DEMAND currency recognizer built on the shared OCR provider."""

    def __init__(self, ocr_provider: OCRProvider, region: Optional[str] = None):
        self.ocr_provider = ocr_provider
        self.region = (region or settings.CURRENCY_DEFAULT_REGION or "INR").upper()

    @staticmethod
    def _detect_currency(text: str, default_region: str) -> str:
        low = text.lower()
        for code, words in _CURRENCY_WORDS.items():
            if any(w in low for w in words):
                return code
        # symbol heuristics
        if "\u20b9" in text:  # ₹
            return "INR"
        if "$" in text:
            return "USD"
        if "\u20ac" in text:  # €
            return "EUR"
        return default_region

    def _detect_denomination(self, text: str, currency: str) -> Optional[int]:
        valid = _DENOMINATIONS.get(currency, _DENOMINATIONS["INR"])
        low = text.lower()

        # numeric values, longest first
        numbers = sorted({int(n) for n in re.findall(r"\d+", text)}, reverse=True)
        for n in numbers:
            if n in valid:
                return n

        # word-form values
        for phrase, value in sorted(_WORD_NUMBERS.items(), key=lambda kv: -len(kv[0])):
            if phrase in low and value in valid:
                return value
        return None

    def recognize(self, frame) -> Dict[str, Any]:
        """Runs OCR on the provided frame and reports recognized currency text."""
        if frame is None:
            return {
                "text": "Cannot identify currency. No camera frame available.",
                "has_result": False,
                "recognized_text": "",
                "currency": None,
                "denomination": None,
                "confidence": 0.0,
                "priority": PriorityLevel.INTERACTION,
            }

        ocr = self.ocr_provider.extract_text(frame)
        recognized = (ocr.get("full_text") or "").strip()
        has_text = bool(ocr.get("has_text")) and bool(recognized)

        if ocr.get("status") == "UNAVAILABLE":
            return {"status": "UNAVAILABLE", "text": ocr.get("short_summary", "OCR is unavailable."),
                    "has_result": False, "recognized_text": "", "priority": PriorityLevel.INTERACTION}

        if not has_text:
            return {
                "text": "I could not read any currency text. Please hold the note steady and closer.",
                "has_result": False,
                "recognized_text": "",
                "currency": None,
                "denomination": None,
                "confidence": 0.0,
                "priority": PriorityLevel.INTERACTION,
            }

        currency = self._detect_currency(recognized, self.region)
        denomination = self._detect_denomination(recognized, currency)

        if denomination is not None:
            spoken = f"This looks like a {denomination} {currency} note."
            # Keep the raw recognized text available but do not fabricate.
            spoken += f" I can read: {recognized}."
            confidence = 0.75
        else:
            spoken = (
                f"I recognized text on the note but could not confidently determine the "
                f"denomination. I can read: {recognized}."
            )
            confidence = 0.4

        logger.info(
            "[CURRENCY] recognized=%r currency=%s denomination=%s", recognized, currency, denomination
        )
        return {
            "text": spoken,
            "has_result": denomination is not None,
            "recognized_text": recognized,
            "currency": currency,
            "denomination": denomination,
            "confidence": confidence,
            "priority": PriorityLevel.INTERACTION,
        }