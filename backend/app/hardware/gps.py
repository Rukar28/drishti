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
    """Placeholder for browser-pushed coordinates. No invented location."""

    def get_location(self) -> Dict[str, Any]:
        return {
            "lat": None,
            "lon": None,
            "accuracy": None,
            "timestamp": time.time(),
            "status": "NOT CONNECTED",
            "source": "browser",
        }

    def health(self) -> Dict[str, Any]:
        return {"provider": "browser", "status": "NOT CONNECTED", "connected": False}


def create_gps_provider(source: str = "mock") -> GPSProvider:
    s = (source or "mock").lower().strip()
    if s == "browser":
        return BrowserGPSProvider()
    return MockGPSProvider()
