"""
VisionMate v2 - Read Mode Handler
Manages OCR text extraction, reading order sorting, and spoken text summaries.
Triggered on-demand, never runs continuously.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from backend.app.core.interfaces import OCRProvider
from backend.app.priority.engine import PriorityLevel

logger = logging.getLogger("visionmate.read")

class ReadModeHandler:
    """Handles on-demand Read Mode extraction and spoken results."""
    def __init__(self, ocr_provider: OCRProvider):
        self.ocr_provider = ocr_provider
        self.latest_result: Optional[Dict[str, Any]] = None

    def trigger_read(self, frame: np.ndarray, full_reading: bool = False) -> Dict[str, Any]:
        """Runs OCR extraction on the provided image and prepares spoken output."""
        if frame is None:
            return {
                "text": "Cannot read text. No camera frame available.",
                "priority": PriorityLevel.INTERACTION,
                "has_text": False,
                "metrics": {
                    "ocr_capture_ms": 0.0,
                    "ocr_preprocess_ms": 0.0,
                    "ocr_inference_ms": 0.0,
                    "ocr_postprocess_ms": 0.0,
                    "ocr_total_ms": 0.0
                }
            }

        logger.info("Executing on-demand OCR in Read Mode...")
        ocr_res = self.ocr_provider.extract_text(frame)
        self.latest_result = ocr_res

        if not ocr_res["has_text"]:
            return {
                "text": ocr_res.get("short_summary") or "No text detected in view.",
                "priority": PriorityLevel.INTERACTION,
                "has_text": False,
                "metrics": ocr_res.get("metrics", {})
            }

        # Select spoken text format
        spoken_text = ocr_res["full_text"] if full_reading else ocr_res["short_summary"]

        return {
            "text": spoken_text,
            "full_text": ocr_res["full_text"],
            "short_summary": ocr_res["short_summary"],
            "priority": PriorityLevel.INTERACTION,
            "has_text": True,
            "metrics": ocr_res.get("metrics", {})
        }
