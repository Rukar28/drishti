"""
VisionMate v2 - Spatial Relative Depth Estimator
Computes relative proximity gradients and proximity bands from 2D bounding boxes and ground plane heuristics.
Strictly adheres to spatial/relative proximity estimation rather than fabricated physical meter metrics.
"""

from typing import Tuple, Dict, Any
from backend.app.schemas.world_state import RelativeDepthBand

class RelativeDepthEstimator:
    """Estimates relative spatial proximity based on optical scale and image-space grounding."""

    @staticmethod
    def estimate_relative_depth(norm_bbox: list, class_name: str = "") -> Tuple[RelativeDepthBand, float]:
        """
        Calculates relative proximity score (0.0 to 1.0, where 1.0 is touching camera lens)
        and categorizes into RelativeDepthBand.
        """
        x1, y1, x2, y2 = norm_bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ground_y = y2  # Lower edge of bounding box in image space

        # Proximity score based on vertical extent and ground horizon proximity
        # Objects touching the bottom of the frame with large bounding box are VERY NEAR
        proximity_score = (h * 0.6) + (ground_y * 0.4)
        proximity_score = max(0.0, min(1.0, proximity_score))

        # Categorize into spatial proximity band
        if proximity_score > 0.72 or area > 0.35:
            band = RelativeDepthBand.VERY_NEAR
        elif proximity_score > 0.48 or area > 0.15:
            band = RelativeDepthBand.NEAR
        elif proximity_score > 0.25:
            band = RelativeDepthBand.MID
        else:
            band = RelativeDepthBand.FAR

        return band, round(proximity_score, 3)
