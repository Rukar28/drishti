"""
P4 — Depth provider architecture.

The mock provider is DEMO MODE. The VL53L5CX provider is IMPLEMENTED but
HARDWARE-UNVERIFIED: without a sensor/endpoint it reports NOT CONNECTED and never
invents a distance. Zone math is verified against a synthetic grid.
"""

import pytest

from backend.app.hardware.depth import (
    MockDepthProvider, VL53L5CXDepthProvider, create_depth_provider,
)


def test_mock_depth_is_demo_mode_without_fake_distance():
    p = MockDepthProvider()
    assert p.health()["status"] == "DEMO MODE"
    assert p.health()["hardware_verified"] is False
    assert p.get_distance("center")["distance_m"] is None


def test_factory_defaults_to_mock():
    assert isinstance(create_depth_provider("mock"), MockDepthProvider)
    assert isinstance(create_depth_provider("vl53l5cx"), VL53L5CXDepthProvider)


def test_vl53l5cx_without_endpoint_reports_not_connected():
    p = VL53L5CXDepthProvider(endpoint_url="")
    zones = p.get_zones()
    assert zones["status"] == "NOT CONNECTED"
    assert zones["connected"] is False
    assert p.get_distance("center")["distance_m"] is None
    assert p.health()["hardware_verified"] is False


def test_vl53l5cx_zone_math_from_synthetic_grid(monkeypatch):
    # 8x8 grid: center columns close (400 mm), sides far (3000 mm)
    grid = []
    for _ in range(8):
        row = [3000] * 8
        row[3] = 400
        row[4] = 420
        grid.append(row)

    p = VL53L5CXDepthProvider(endpoint_url="http://example/depth")
    monkeypatch.setattr(p, "_fetch_grid", lambda: grid)

    zones = p.get_zones()
    assert zones["status"] == "ONLINE"
    assert zones["connected"] is True
    assert zones["center_mm"] in (400, 420)
    assert zones["left_mm"] == 3000
    assert zones["right_mm"] == 3000

    d = p.get_distance("center")
    assert d["distance_m"] == pytest.approx(0.4, abs=0.05)


def test_vl53l5cx_invalid_cells_are_ignored(monkeypatch):
    # 0 mm readings are below TOF_VALID_MIN_MM and must be discarded.
    grid = [[0] * 8 for _ in range(8)]
    grid[0][3] = 900
    p = VL53L5CXDepthProvider(endpoint_url="http://example/depth")
    monkeypatch.setattr(p, "_fetch_grid", lambda: grid)
    zones = p.get_zones()
    assert zones["center_mm"] == 900
    assert zones["left_mm"] is None
