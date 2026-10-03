"""
VisionMate v2 - Color Recognition (ON-DEMAND)

Answers spoken "what color ..." questions by analyzing the most recent
camera frame. Triggered strictly on demand; never runs in the 18 FPS loop.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.priority.engine import PriorityLevel

logger = logging.getLogger("visionmate.color")

TARGET_CLASS_MAP: Dict[str, Tuple[List[str], str]] = {
    "shirt": (["person"], "torso"),
    "tshirt": (["person"], "torso"),
    "t shirt": (["person"], "torso"),
    "clothes": (["person"], "torso"),
    "clothing": (["person"], "torso"),
    "jacket": (["person"], "torso"),
    "sweater": (["person"], "torso"),
    "dress": (["person"], "torso"),
    "pants": (["person"], "lower"),
    "trousers": (["person"], "lower"),
    "jeans": (["person"], "lower"),
    "shorts": (["person"], "lower"),
    "skirt": (["person"], "lower"),
    "hair": (["person"], "upper"),
    "hat": (["person"], "upper"),
    "cap": (["person"], "upper"),
    "head": (["person"], "upper"),
    "bag": (["backpack", "handbag"], "auto"),
    "backpack": (["backpack"], "auto"),
    "handbag": (["handbag"], "auto"),
    "purse": (["handbag"], "auto"),
    "phone": (["cell phone"], "auto"),
    "cellphone": (["cell phone"], "auto"),
    "cell phone": (["cell phone"], "auto"),
    "cup": (["cup"], "auto"),
    "mug": (["cup"], "auto"),
    "bottle": (["bottle"], "auto"),
    "book": (["book"], "auto"),
    "laptop": (["laptop"], "auto"),
    "chair": (["chair"], "auto"),
    "car": (["car"], "auto"),
    "banana": (["banana"], "auto"),
    "apple": (["apple"], "auto"),
}

GENERIC_TARGETS = {"scene", "view", "image", "picture", "screen", "background"}


def analyze_dominant_color(roi_bgr: np.ndarray) -> Tuple[str, float]:
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown", 0.0

    h, w = roi_bgr.shape[:2]
    if h < 2 or w < 2:
        return "unknown", 0.0

    scale = 160.0 / float(max(h, w))
    if scale < 1.0:
        roi_bgr = cv2.resize(
            roi_bgr,
            (max(2, int(w * scale)), max(2, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    total = H.size
    if total == 0:
        return "unknown", 0.0

    black = V < 60
    white = (S < 60) & (V >= 190)
    gray = (S < 60) & (V >= 60) & (V < 190)
    chroma = (S >= 60) & (V >= 60)

    pink = chroma & (H >= 160) & (S < 140) & (V > 160)
    red = chroma & ((H < 10) | (H >= 160)) & ~pink
    brown = chroma & (H >= 10) & (H < 33) & (V < 125)
    orange = chroma & (H >= 10) & (H < 22) & ~brown
    yellow = chroma & (H >= 22) & (H < 33) & ~brown
    green = chroma & (H >= 33) & (H < 78)
    cyan = chroma & (H >= 78) & (H < 100)
    blue = chroma & (H >= 100) & (H < 135)
    purple = chroma & (H >= 135) & (H < 160)

    named = {
        "black": black, "white": white, "gray": gray,
        "pink": pink, "red": red, "brown": brown, "orange": orange,
        "yellow": yellow, "green": green, "cyan": cyan,
        "blue": blue, "purple": purple,
    }

    best_name, best_count = "unknown", 0
    for name, mask in named.items():
        count = int(mask.sum())
        if count > best_count:
            best_name, best_count = name, count

    frac = best_count / float(total)
    if best_name == "unknown":
        return "unknown", frac
    if best_name in ("black", "white", "gray"):
        return best_name, frac

    mask = named[best_name]
    if mask.any():
        mean_v = float(V[mask].mean())
        mean_s = float(S[mask].mean())
        if mean_v < 95:
            return f"dark {best_name}", frac
        if mean_v > 195 and mean_s < 130 and best_name in (
            "blue", "green", "red", "purple", "brown",
        ):
            return f"light {best_name}", frac

    return best_name, frac


class ColorModeHandler:
    """ON-DEMAND color analysis of the current camera frame."""

    def __init__(self, min_roi_pixels: int = 400):
        self.min_roi_pixels = min_roi_pixels

    def analyze(
        self,
        frame: Optional[np.ndarray],
        target: Optional[str] = None,
        tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        norm_target = self._normalize_target(target)

        if frame is None or not hasattr(frame, "shape") or frame.ndim < 2:
            return self._result(
                "Cannot determine color. The camera image is unavailable.",
                success=False,
            )

        h, w = frame.shape[:2]
        roi, matched_class = self._select_roi(frame, norm_target, tracks or [])

        if roi is None or roi.size < max(self.min_roi_pixels, 16):
            if norm_target:
                return self._result(
                    f"I could not get a clear view of the {norm_target}. "
                    "Move closer and ask again.",
                    success=False,
                    target=norm_target,
                )
            return self._result(
                "I could not get a clear view to determine a color.",
                success=False,
            )

        color_name, frac = analyze_dominant_color(roi)
        text = self._compose_text(norm_target, matched_class, color_name, frac)

        logger.info(
            "[COLOR] target=%r matched=%s color=%s (%.0f%% of ROI)",
            norm_target, matched_class, color_name, frac * 100.0,
        )

        return self._result(
            text,
            success=(color_name != "unknown"),
            color=color_name,
            confidence=round(float(frac), 2),
            target=norm_target,
            matched_object=matched_class,
        )

    def _result(self, text, success, color=None, confidence=None,
                target=None, matched_object=None) -> Dict[str, Any]:
        return {
            "text": text,
            "priority": PriorityLevel.INTERACTION,
            "success": success,
            "color": color,
            "confidence": confidence,
            "target": target,
            "matched_object": matched_object,
        }

    def _select_roi(self, frame, target, tracks):
        h, w = frame.shape[:2]

        if target and target not in GENERIC_TARGETS:
            hit = self._match_track(target, tracks)
            if hit is not None:
                class_name, bbox, crop_hint = hit
                box = self._clamp_bbox(bbox, w, h)
                if box is not None:
                    roi = self._apply_crop_hint(frame, box, crop_hint)
                    if roi is not None and roi.size >= self.min_roi_pixels:
                        return roi, class_name

        cx0, cx1 = int(w * 0.30), int(w * 0.70)
        cy0, cy1 = int(h * 0.30), int(h * 0.70)
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            return frame, None
        return frame[cy0:cy1, cx0:cx1], None

    def _apply_crop_hint(self, frame, box, hint):
        x1, y1, x2, y2 = box
        ww, hh = x2 - x1, y2 - y1

        if hint == "torso":
            bx1, by1 = x1 + int(0.18 * ww), y1 + int(0.28 * hh)
            bx2, by2 = x2 - int(0.18 * ww), y1 + int(0.72 * hh)
        elif hint == "lower":
            bx1, by1, bx2, by2 = x1, y1 + int(0.62 * hh), x2, y2
        elif hint == "upper":
            bx1, by1, bx2, by2 = x1, y1, x2, y1 + int(0.32 * hh)
        else:
            bx1, by1, bx2, by2 = x1, y1, x2, y2

        if bx2 - bx1 < 4 or by2 - by1 < 4:
            bx1, by1, bx2, by2 = x1, y1, x2, y2

        roi = frame[by1:by2, bx1:bx2]
        if roi.size < self.min_roi_pixels:
            roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _match_track(self, target, tracks):
        if not tracks:
            return None

        candidates, crop_hint = TARGET_CLASS_MAP.get(target, ([target], "auto"))
        cand_norm = [self._norm(c) for c in candidates]

        for t in tracks:
            cn = self._norm(t.get("class_name", ""))
            if cn and cn in cand_norm:
                return t.get("class_name"), t.get("bbox"), crop_hint

        for t in tracks:
            cn = self._norm(t.get("class_name", ""))
            for c in cand_norm:
                if c and cn and (c in cn or cn in c):
                    return t.get("class_name"), t.get("bbox"), crop_hint

        return None

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).lower().strip())

    @staticmethod
    def _normalize_target(target) -> Optional[str]:
        if not target:
            return None

        t = re.sub(r"[^\w\s]", "", str(target).lower())
        tokens = re.sub(r"\s+", " ", t).strip().split()
        lead = {
            "the", "a", "an", "my", "this", "that", "it",
            "is", "of", "are", "do", "does", "you", "see",
        }
        trail = {"please", "now", "exactly", "there", "see", "you", "do"}

        while tokens and tokens[0] in lead:
            tokens.pop(0)
        while tokens and tokens[-1] in trail:
            tokens.pop()

        cleaned = " ".join(tokens).strip()
        if not cleaned or cleaned in ("color", "colour"):
            return None
        return cleaned

    @staticmethod
    def _clamp_bbox(bbox, w, h):
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        except (TypeError, ValueError):
            return None

        x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))

        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return x1, y1, x2, y2

    def _compose_text(self, target, matched_class, color_name, frac) -> str:
        if color_name == "unknown" or frac < 0.12:
            if target and matched_class:
                return f"I could not determine a clear color for the {target}."
            if target:
                return (
                    f"I could not find the {target} in view, and the colors "
                    "ahead are mixed with no clear dominant color."
                )
            return "The colors in front of you are mixed; no single dominant color."

        verb = "appears to be" if frac < 0.35 else "is"

        if target and matched_class and target != self._norm(matched_class):
            return f"Your {target} {verb} {color_name}."
        if matched_class:
            return f"The {matched_class} {verb} {color_name}."
        if target:
            return (
                f"I could not spot the {target} in view. The dominant color "
                f"directly ahead {verb} {color_name}."
            )
        return f"The dominant color directly in front of you {verb} {color_name}."
