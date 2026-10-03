"""
VisionMate — Backend navigation architecture (P6).

Pipeline: Location → Destination → Route → Direction instruction → Voice guidance.

DESIGN RULES:
  * API keys are read ONLY from the environment (never hard-coded, never returned
    to any client/frontend).
  * If live routing credentials are unavailable, the mock provider supplies a
    fully deterministic, testable route (haversine distance + bearing). We never
    pretend that live turn-by-turn routing is active.
  * Unavailable GPS is handled gracefully (status explicitly reported).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import logging
import math
import time

from backend.app.core.config import settings

logger = logging.getLogger("visionmate.navigation")


# ── Geodesy helpers ───────────────────────────────────────────────
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


_COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def bearing_to_compass(bearing: float) -> str:
    idx = int((bearing + 22.5) % 360 // 45)
    return _COMPASS[idx]


class NavigationProvider(ABC):
    """Abstract routing provider."""

    name: str = "navigation"

    @abstractmethod
    def resolve_destination(self, name: str) -> Optional[Tuple[float, float]]:
        """Return (lat, lon) for a named destination, or None if unknown."""

    @abstractmethod
    def get_route(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Dict[str, Any]:
        """Return route info: distance_m, bearing_deg, instruction, status."""

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        ...


class MockNavigationProvider(NavigationProvider):
    """
    Deterministic offline routing for laptop dev / tests.

    Distances and bearings are computed locally from a configured destination
    registry; this is NOT live turn-by-turn routing and is reported as MOCK.
    """

    name = "mock"

    def __init__(self, registry: Optional[Dict[str, Tuple[float, float]]] = None):
        self.registry = registry if registry is not None else self._parse_registry(settings.NAVIGATION_DESTINATIONS)

    @staticmethod
    def _parse_registry(raw: str) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for entry in (raw or "").split(";"):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            name, coords = entry.split(":", 1)
            try:
                lat_s, lon_s = coords.split(",", 1)
                out[name.strip().lower()] = (float(lat_s), float(lon_s))
            except Exception:
                continue
        if not out:
            out = {
                "home": (12.9716, 77.5946),
                "office": (12.9352, 77.6245),
                "pharmacy": (12.9600, 77.6100),
            }
        return out

    def resolve_destination(self, name: str) -> Optional[Tuple[float, float]]:
        if not name:
            return None
        return self.registry.get(name.strip().lower())

    def get_route(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Dict[str, Any]:
        dist = haversine_m(origin[0], origin[1], destination[0], destination[1])
        brg = bearing_deg(origin[0], origin[1], destination[0], destination[1])
        return {
            "status": "MOCK",
            "distance_m": round(dist, 1),
            "bearing_deg": round(brg, 1),
            "compass": bearing_to_compass(brg),
            "instruction": f"Head {bearing_to_compass(brg)} for about {int(round(dist))} meters.",
            "provider": self.name,
        }

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "status": "MOCK",
            "connected": False,
            "destinations": sorted(self.registry.keys()),
            "note": "Offline haversine routing — not live turn-by-turn.",
        }
class GoogleDirectionsProvider(NavigationProvider):
    """
    Google Directions boundary. The API key is read ONLY from the environment.

    Without a configured key this provider reports STATUS NOT_CONFIGURED and never
    fabricates a route. The key is never logged, returned, or exposed to clients.
    """

    name = "google"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY") or settings.GOOGLE_MAPS_API_KEY
        self._geocode_cache: Dict[str, Tuple[float, float]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def resolve_destination(self, name: str) -> Optional[Tuple[float, float]]:
        return self._geocode_cache.get((name or "").strip().lower())

    def get_route(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Dict[str, Any]:
        return {
            "status": "NOT_CONFIGURED",
            "distance_m": None,
            "instruction": None,
            "provider": self.name,
            "note": (
                "Live directions unavailable; configure GOOGLE_MAPS_API_KEY on the backend."
                if not self.configured else
                "Live directions call is disabled in laptop/dev mode."
            ),
        }

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "status": "CONFIGURED" if self.configured else "NOT_CONFIGURED",
            "connected": False,
            "api_key_present": self.configured,  # boolean only — never the key itself
        }


def create_navigation_provider(name: Optional[str] = None) -> NavigationProvider:
    n = (name or settings.NAVIGATION_PROVIDER or "mock").lower().strip()
    if n == "google":
        return GoogleDirectionsProvider()
    return MockNavigationProvider()
class Navigator:
    """
    Session manager turning location + provider routing into calm voice guidance.

    States: IDLE → ROUTING → GUIDING → ARRIVED (or CANCELLED / UNAVAILABLE).
    """

    def __init__(self, provider: Optional[NavigationProvider] = None):
        self.provider = provider or create_navigation_provider()
        self.destination_name: Optional[str] = None
        self.destination_coords: Optional[Tuple[float, float]] = None
        self.last_instruction: Optional[str] = None
        self.last_update: float = 0.0
        self.state: str = "IDLE"

    def stop(self) -> None:
        self.destination_name = None
        self.destination_coords = None
        self.last_instruction = None
        self.state = "CANCELLED"

    def start(self, destination_name: str, origin: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        name = (destination_name or "").strip().lower()
        self.stop()
        if not name:
            return {"status": "NO_DESTINATION", "text": "Which place would you like to navigate to?"}

        if origin is None or origin[0] is None:
            self.state = "UNAVAILABLE"
            return {
                "status": "GPS_UNAVAILABLE",
                "text": "I cannot start navigation because location is unavailable.",
                "navigation": self.health(),
            }

        coords = self.provider.resolve_destination(name)
        if coords is None:
            return {
                "status": "UNKNOWN_DESTINATION",
                "text": f"I don't have a saved location for {name}.",
                "navigation": self.health(),
            }

        self.destination_name = name
        self.destination_coords = coords
        self.state = "ROUTING"
        route = self.provider.get_route(origin, coords)
        self.state = "GUIDING"
        self.last_instruction = route.get("instruction")
        self.last_update = time.time()
        return {
            "status": "STARTED",
            "text": f"Starting navigation to {name}. {self.last_instruction or ''}".strip(),
            "route": route,
            "navigation": self.health(),
        }

    def update(self, origin: Optional[Tuple[float, float]], min_interval_sec: float = 8.0) -> Optional[Dict[str, Any]]:
        if self.state != "GUIDING" or self.destination_coords is None:
            return None
        if origin is None or origin[0] is None:
            return None

        now = time.time()
        dist = haversine_m(origin[0], origin[1], self.destination_coords[0], self.destination_coords[1])

        if dist <= settings.NAVIGATION_ARRIVAL_RADIUS_M:
            self.state = "ARRIVED"
            self.last_update = now
            return {"status": "ARRIVED", "distance_m": round(dist, 1),
                    "text": f"You have arrived at {self.destination_name}."}

        if (now - self.last_update) < min_interval_sec:
            return None

        route = self.provider.get_route(origin, self.destination_coords)
        instruction = route.get("instruction")
        self.last_instruction = instruction
        self.last_update = now
        return {
            "status": "GUIDING",
            "distance_m": round(dist, 1),
            "instruction": instruction,
            "text": f"{instruction} {int(round(dist))} meters to {self.destination_name}.",
        }

    def where_am_i(self, origin: Optional[Tuple[float, float]]) -> str:
        if origin is None or origin[0] is None:
            return "Current location is unavailable."
        return f"Your current coordinates are latitude {origin[0]:.5f}, longitude {origin[1]:.5f}."

    def health(self) -> Dict[str, Any]:
        h = self.provider.health()
        h["active_destination"] = self.destination_name
        h["state"] = self.state
        return h