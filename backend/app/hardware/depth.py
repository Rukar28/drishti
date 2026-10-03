"""
Depth providers (P4).

- `MockDepthProvider`  : laptop/dev stand-in. Reports DEMO MODE; no fake distances.
- `VL53L5CXDepthProvider`: reads the 8x8 VL53L5CX multizone grid from a companion
  MCU/ESP32 JSON endpoint (the sensor is I2C-attached to the MCU, not the laptop).
  Without the sensor/endpoint it reports NOT CONNECTED and never invents distance.

HARDWARE STATUS: the VL53L5CX path is IMPLEMENTED but HARDWARE-UNVERIFIED in this
environment (no sensor attached). It must not be reported as physically verified.
"""

from typing import Any, Dict, List, Optional
import logging

from backend.app.core.config import settings
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
            "hardware_verified": False,
        }


class VL53L5CXDepthProvider(DepthProvider):
    """
    Real VL53L5CX integration path.

    The VL53L5CX is an I2C multizone ToF sensor; in this architecture it is wired
    to a companion microcontroller (e.g. the ESP32) which exposes the raw 8x8 grid
    as JSON. This provider consumes that grid and derives left/center/right zones.

    If `TOF_ENDPOINT_URL` is unset or unreachable it explicitly reports
    NOT CONNECTED — it never fabricates a distance.
    """

    def __init__(self, endpoint_url: Optional[str] = None, timeout_sec: Optional[float] = None):
        self.endpoint_url = (endpoint_url if endpoint_url is not None else settings.TOF_ENDPOINT_URL) or ""
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else settings.TOF_TIMEOUT_SEC)
        self._last_grid: Optional[List[List[int]]] = None
        self._last_error: Optional[str] = None
        if not self.endpoint_url:
            logger.warning(
                "[DEPTH] VL53L5CX selected but TOF_ENDPOINT_URL is not configured; "
                "reporting NOT CONNECTED (HARDWARE-UNVERIFIED)."
            )

    def _fetch_grid(self) -> Optional[List[List[int]]]:
        if not self.endpoint_url:
            return None
        try:
            import requests
            resp = requests.get(self.endpoint_url, timeout=self.timeout_sec)
            if resp.status_code != 200:
                self._last_error = f"HTTP {resp.status_code}"
                return None
            data = resp.json()
            grid = data.get("zones") or data.get("grid") or data.get("depth")
            if not grid or not isinstance(grid, list):
                self._last_error = "no grid in payload"
                return None
            self._last_grid = grid
            self._last_error = None
            return grid
        except Exception as exc:
            self._last_error = str(exc)
            logger.debug("[DEPTH] VL53L5CX endpoint fetch failed: %s", exc)
            return None

    @staticmethod
    def _zone_distance_mm(grid: List[List[int]], cols: range) -> Optional[int]:
        """Closest valid reading in a zone (minimum). Safety prioritizes the nearest surface."""
        vals: List[int] = []
        for row in grid:
            if not isinstance(row, list):
                continue
            for c in cols:
                if 0 <= c < len(row):
                    v = row[c]
                    if isinstance(v, (int, float)) and v >= settings.TOF_VALID_MIN_MM:
                        vals.append(int(v))
        if not vals:
            return None
        return min(vals)

    def get_depth(self) -> Dict[str, Any]:
        grid = self._fetch_grid()
        if grid is None:
            return {"status": "NOT CONNECTED", "connected": False, "grid": None, "source": "vl53l5cx"}
        return {"status": "ONLINE", "connected": True, "grid": grid, "source": "vl53l5cx"}

    def get_zones(self) -> Dict[str, Any]:
        grid = self._fetch_grid()
        if grid is None:
            return {"left": None, "center": None, "right": None,
                    "status": "NOT CONNECTED", "source": "vl53l5cx", "connected": False}
        ncols = max((len(r) for r in grid if isinstance(r, list)), default=8)
        third = max(1, ncols // 3)
        left = self._zone_distance_mm(grid, range(0, third))
        center = self._zone_distance_mm(grid, range(third, ncols - third))
        right = self._zone_distance_mm(grid, range(ncols - third, ncols))
        return {
            "left_mm": left,
            "center_mm": center,
            "right_mm": right,
            "left": left,
            "center": center,
            "right": right,
            "source": "vl53l5cx",
            "status": "ONLINE",
            "connected": True,
        }

    def get_distance(self, direction: str) -> Dict[str, Any]:
        zones = self.get_zones()
        key = f"{(direction or 'center').lower()}_mm"
        mm = zones.get(key, zones.get("center_mm"))
        return {
            "distance_m": round(mm / 1000.0, 2) if mm else None,
            "distance_mm": mm,
            "direction": direction,
            "confidence": 0.9 if mm else 0.0,
            "source": "vl53l5cx" if mm else "NOT CONNECTED",
        }

    def health(self) -> Dict[str, Any]:
        connected = self._last_grid is not None
        return {
            "provider": "vl53l5cx",
            "status": "ONLINE" if connected else ("CONFIGURED" if self.endpoint_url else "NOT CONNECTED"),
            "connected": connected,
            "hardware_verified": False,
            "endpoint_configured": bool(self.endpoint_url),
            "last_error": self._last_error,
        }


def create_depth_provider(source: str = "mock") -> DepthProvider:
    s = (source or "mock").lower().strip()
    if s in ("vl53l5cx", "tof", "hardware"):
        return VL53L5CXDepthProvider()
    return MockDepthProvider()
