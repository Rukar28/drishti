"""
Depth providers.

Phase 7 will add VL53L5CX. Until then the mock provider is explicit DEMO MODE
and never invents distances.
"""

from typing import Any, Dict
import logging

from backend.app.core.interfaces import DepthProvider

logger = logging.getLogger("visionmate.depth")


class MockDepthProvider(DepthProvider):
    """Laptop/dev stand-in. Reports DEMO MODE; no fake metric distances."""

    def get_depth(self) -> Dict[str, Any]:
        return {
            "status": "DEMO MODE",
            "connected": False,
            "grid": None,
            "source": "mock",
        }

    def get_zones(self) -> Dict[str, Any]:
        return {
            "left": None,
            "center": None,
            "right": None,
            "source": "mock",
            "status": "DEMO MODE",
        }

    def get_distance(self, direction: str) -> Dict[str, Any]:
        return {
            "distance_m": None,
            "direction": direction,
            "confidence": 0.0,
            "source": "DEMO MODE",
        }

    def health(self) -> Dict[str, Any]:
        return {
            "provider": "mock",
            "status": "DEMO MODE",
            "connected": False,
        }


class VL53L5CXDepthProvider(DepthProvider):
    """Integration path for hardware ToF. Not claimed complete without the sensor."""

    def __init__(self):
        logger.warning("[DEPTH] VL53L5CX selected but hardware is not attached; reporting NOT CONNECTED.")

    def get_depth(self) -> Dict[str, Any]:
        return {"status": "NOT CONNECTED", "connected": False, "grid": None, "source": "vl53l5cx"}

    def get_zones(self) -> Dict[str, Any]:
        return {"left": None, "center": None, "right": None, "status": "NOT CONNECTED", "source": "vl53l5cx"}

    def get_distance(self, direction: str) -> Dict[str, Any]:
        return {"distance_m": None, "direction": direction, "confidence": 0.0, "source": "NOT CONNECTED"}

    def health(self) -> Dict[str, Any]:
        return {"provider": "vl53l5cx", "status": "NOT CONNECTED", "connected": False}


def create_depth_provider(source: str = "mock") -> DepthProvider:
    s = (source or "mock").lower().strip()
    if s in ("vl53l5cx", "tof", "hardware"):
        return VL53L5CXDepthProvider()
    return MockDepthProvider()
