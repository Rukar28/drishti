"""
Unit Tests for Direction Resolution & Relative Depth Estimation
"""

import pytest
from backend.app.tracking.tracker import SpatialTracker
from backend.app.perception.relative_depth import RelativeDepthEstimator
from backend.app.schemas.world_state import ImageDirection, RelativeDepthBand

def test_direction_resolution():
    # Test Left
    assert SpatialTracker._resolve_direction(0.10) == ImageDirection.FAR_LEFT
    assert SpatialTracker._resolve_direction(0.30) == ImageDirection.LEFT
    assert SpatialTracker._resolve_direction(0.40) == ImageDirection.SLIGHT_LEFT
    
    # Test Center
    assert SpatialTracker._resolve_direction(0.50) == ImageDirection.CENTER
    assert SpatialTracker._resolve_direction(0.55) == ImageDirection.CENTER
    
    # Test Right
    assert SpatialTracker._resolve_direction(0.60) == ImageDirection.SLIGHT_RIGHT
    assert SpatialTracker._resolve_direction(0.75) == ImageDirection.RIGHT
    assert SpatialTracker._resolve_direction(0.95) == ImageDirection.FAR_RIGHT

def test_relative_depth_estimation():
    # Very Near (large bounding box close to bottom)
    band, score = RelativeDepthEstimator.estimate_relative_depth([0.1, 0.2, 0.9, 0.95], "chair")
    assert band in [RelativeDepthBand.VERY_NEAR, RelativeDepthBand.NEAR]
    assert score > 0.5

    # Far (small bounding box near top horizon)
    band_far, score_far = RelativeDepthEstimator.estimate_relative_depth([0.4, 0.1, 0.5, 0.2], "person")
    assert band_far == RelativeDepthBand.FAR
    assert score_far < 0.3
