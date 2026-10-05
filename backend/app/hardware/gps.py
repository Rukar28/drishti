"""
GPS providers.

Phase 10 will add real maps/GPS. Mock/browser sources are labelled explicitly.
"""

from typing import Any, Dict
import time
import logging

from backend.app.core.interfaces import GPSProvider

logger = logging.getLogger("visionmate.gps")


class MockGPSProvider(GPSProvider):
    def get_location(self) -> Dict[str, Any]:
        return {
            "lat": None,
            "lon": None,
            "accuracy": None,
            "timestamp": time.time(),
            "status": "DEMO MODE",
            "source": "mock",
        }

    def health(self) -> Dict[str, Any]:
        return {"provider": "mock", "status": "DEMO MODE", "connected": False}


class BrowserGPSProvider(GPSProvider):
    """Permission-based browser fixes expire after 30 seconds."""
    def __init__(self):
        self._location = None

    def update(self, lat, lon, accuracy):
        self._location = {"lat": lat, "lon": lon, "accuracy": accuracy,
                          "timestamp": time.time(), "status": "CONNECTED", "source": "browser"}
        return self.get_location()

    def get_location(self):
        if self._location and time.time() - self._location["timestamp"] <= 30:
            return dict(self._location)
        return {"lat": None, "lon": None, "accuracy": None,
                "timestamp": self._location["timestamp"] if self._location else None,
                "status": "STALE" if self._location else "NOT CONNECTED", "source": "browser"}

    def health(self):
        loc = self.get_location()
        return {"provider": "browser", "status": loc["status"], "connected": loc["lat"] is not None}


def create_gps_provider(source: str = "mock") -> GPSProvider:
    s = (source or "mock").lower().strip()
    if s == "browser":
        return BrowserGPSProvider()
    return MockGPSProvider()
