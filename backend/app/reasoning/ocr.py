"""
VisionMate v2 - PP-OCRv5 Optical Character Recognition Provider
Implements production PP-OCRv5 via PaddleOCR 3.x stack.
Optimized on-demand text extraction with image preprocessing, model reuse,
quality validation, and detailed execution timing metrics.
"""

import os
import cv2
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
import logging
from backend.app.core.interfaces import OCRProvider

# Disable PaddleX remote source connectivity checks for fast local initialization
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import threading

logger = logging.getLogger("visionmate.ocr")

class PPOCRv5Provider(OCRProvider):
    """
    Production PP-OCRv5 text extraction engine using PaddleOCR 3.x.
    Cached/reused singleton model with sub-millisecond preprocessing and granular timing metrics.
    """
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.ocr_engine = None
        self.model_name = "PP-OCRv5"
        self._lock = threading.Lock()
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes and caches the PP-OCRv5 engine once."""
        try:
            from paddleocr import PaddleOCR
            import paddle
            # Initialize PP-OCRv5 model with textline orientation detection
            self.ocr_engine = PaddleOCR(
                ocr_version="PP-OCRv5",
                device="gpu" if self.use_gpu and paddle.device.is_compiled_with_cuda() else "cpu",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                lang="en"
            )
            logger.info("PaddleOCR PP-OCRv5 engine initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR failed to initialize: {e}. OCR is unavailable.")
            self.ocr_engine = None

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fast contrast equalization (CLAHE) and Laplacian sharpness calculation."""
        h, w = image.shape[:2]
        
        # Scale image down if oversized for low latency OCR
        target_img = image
        if w > 960 or h > 720:
            scale = 960.0 / w
            target_img = cv2.resize(image, (960, int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape) == 3 else target_img
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr, sharpness

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        t_total_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("[OCR-DEBUG] 1. READ REQUEST RECEIVED")

        if image is None:
            logger.warning("[OCR-DEBUG] 2. Latest frame acquired: NO")
            logger.warning("[OCR-DEBUG] 5. Valid image content: NO")
            logger.info("=" * 60)
            return {
                "full_text": "",
                "short_summary": "Cannot read text. No camera frame available.",
                "text_blocks": [],
                "has_text": False,
                "model_name": self.model_name,
                "metrics": {
                    "ocr_capture_ms": 0.0,
                    "ocr_preprocess_ms": 0.0,
                    "ocr_inference_ms": 0.0,
                    "ocr_postprocess_ms": 0.0,
                    "ocr_total_ms": 0.0
                }
            }

        h, w = image.shape[:2]
        logger.info(f"[OCR-DEBUG] 2. Latest ESP32 frame successfully acquired: YES")
        logger.info(f"[OCR-DEBUG] 3. Frame resolution: {w}x{h}")
        logger.info(f"[OCR-DEBUG] 4. Frame dtype: {image.dtype}")
        is_valid = bool(image.size > 0 and np.max(image) > 0)
        logger.info(f"[OCR-DEBUG] 5. Frame contains valid image: {'YES' if is_valid else 'NO'} (pixel range: {int(np.min(image))}..{int(np.max(image))})")

        # 1. Preprocess
        t_prep_start = time.perf_counter()
        enhanced_img, sharpness = self._preprocess_image(image)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        logger.info(f"[OCR-DEBUG] 6. Sharpness / Laplacian variance VALUE: {sharpness:.2f}")
        logger.info("[OCR-DEBUG] 7. Sharpness blocking rule: DISABLED (non-blocking telemetry metric)")
        logger.info("[OCR-DEBUG] 8. Frame rejected by sharpness check: NO")
        logger.info(f"[OCR-DEBUG] 9. Preprocessing completed: YES ({t_prep_ms:.2f} ms)")

        text_lines = []
        text_blocks = []

        # 2. Inference via PP-OCRv5 / PaddleOCR 3.x
        t_inf_start = time.perf_counter()
        predict_called = False
        raw_pred_count = 0
        inference_error = False

        if self.ocr_engine is not None:
            predict_called = True
            logger.info("[OCR-DEBUG] 10. PaddleOCR predict() called: YES")
            try:
                with self._lock:
                    # PaddleOCR 3.x predict pipeline
                    predictions = list(self.ocr_engine.predict(enhanced_img))
                raw_pred_count = len(predictions) if predictions else 0
                if predictions:
                    for pred in predictions:
                        rec_texts = pred.get("rec_texts", pred.get("rec_text", []))
                        rec_scores = pred.get("rec_scores", pred.get("rec_score", []))
                        rec_boxes = pred.get("rec_boxes", pred.get("dt_polys", []))

                        for idx, text in enumerate(rec_texts):
                            conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.90
                            box = rec_boxes[idx] if idx < len(rec_boxes) else []
                            clean_t = text.strip()
                            if conf > 0.40 and len(clean_t) > 1:
                                text_lines.append(clean_t)
                                text_blocks.append({
                                    "text": clean_t,
                                    "bbox": box if isinstance(box, list) else [],
                                    "conf": round(conf, 3)
                                })
            except Exception as e:
                inference_error = True
                logger.error(f"[OCR-DEBUG] Error during PaddleOCR inference: {e}")
        else:
            logger.warning("[OCR-DEBUG] 10. PaddleOCR predict() called: NO (Engine is unavailable)")
        t_inf_ms = (time.perf_counter() - t_inf_start) * 1000.0

        logger.info(f"[OCR-DEBUG] 11. OCR detection result count: {raw_pred_count}")
        logger.info(f"[OCR-DEBUG] 12. Recognition result count: {len(text_lines)}")
        logger.info(f"[OCR-DEBUG] 13. Recognized text: {text_lines}")
        logger.info(f"[OCR-DEBUG] 14. OCR confidence: {[b.get('conf') for b in text_blocks]}")
        logger.info(f"[OCR-DEBUG] 15. OCR inference time: {t_inf_ms:.2f} ms")

        # 3. Postprocess & Cleanup
        t_post_start = time.perf_counter()
        if not text_lines:
            t_post_ms = (time.perf_counter() - t_post_start) * 1000.0
            t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
            msg = "I couldn't read the text because OCR is unavailable." if (self.ocr_engine is None or inference_error) else "I couldn't read the text clearly."
            logger.info(f"[OCR-DEBUG] 16. Final Read Mode result: has_text=False (0 text lines recognized)")
            logger.info(f"[OCR-DEBUG] 17. Exact TTS message: '{msg}'")
            logger.info("=" * 60)
            return {
                "full_text": "",
                "short_summary": msg,
                "status": "UNAVAILABLE" if self.ocr_engine is None or inference_error else "NO_TEXT",
                "text_blocks": [],
                "has_text": False,
                "model_name": self.model_name,
                "metrics": {
                    "ocr_capture_ms": 0.0,
                    "ocr_preprocess_ms": round(t_prep_ms, 2),
                    "ocr_inference_ms": round(t_inf_ms, 2),
                    "ocr_postprocess_ms": round(t_post_ms, 2),
                    "ocr_total_ms": round(t_total_ms, 2)
                }
            }

        full_text = " ".join(text_lines)

        # Build concise spoken summary
        if len(text_lines) == 1:
            short_summary = f"Text says: {text_lines[0]}"
        elif len(text_lines) <= 3:
            short_summary = f"Text reads: {', '.join(text_lines)}."
        else:
            short_summary = f"Document with heading: '{text_lines[0]}', containing {len(text_lines)-1} additional lines."

        t_post_ms = (time.perf_counter() - t_post_start) * 1000.0
        t_total_ms = (time.perf_counter() - t_total_start) * 1000.0

        logger.info(f"[OCR-DEBUG] 16. Final Read Mode result: has_text=True, full_text='{full_text}'")
        logger.info(f"[OCR-DEBUG] 17. Exact TTS message: '{short_summary}'")
        logger.info("=" * 60)

        return {
            "full_text": full_text,
            "short_summary": short_summary,
            "text_blocks": text_blocks,
            "has_text": True,
            "model_name": self.model_name,
            "metrics": {
                "ocr_capture_ms": 0.0,
                "ocr_preprocess_ms": round(t_prep_ms, 2),
                "ocr_inference_ms": round(t_inf_ms, 2),
                "ocr_postprocess_ms": round(t_post_ms, 2),
                "ocr_total_ms": round(t_total_ms, 2)
            }
        }
