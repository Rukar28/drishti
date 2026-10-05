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
    """Google Geocoding + Routes API. Credentials and upstream errors stay private."""
    name = "google"

    def __init__(self, api_key=None):
        import os
        self.api_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or settings.GOOGLE_MAPS_API_KEY) if api_key is None else api_key
        self._geocode_cache = {}
        self.last_error = None
        self._connected = False

    @property
    def configured(self):
        return bool(self.api_key)

    def resolve_destination(self, name):
        import httpx
        if not self.configured:
            self.last_error = "Google Maps is not configured."
            return None
        key = name.strip().lower()
        if key in self._geocode_cache:
            return self._geocode_cache[key]
        try:
            response = httpx.get("https://maps.googleapis.com/maps/api/geocode/json",
                                 params={"address": name, "key": self.api_key}, timeout=12)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "OK" or not data.get("results"):
                self.last_error = "Google geocoding: " + str(data.get("status", "NO_RESULTS"))
                return None
            point = data["results"][0]["geometry"]["location"]
            coords = (point["lat"], point["lng"])
            self._geocode_cache[key] = coords
            self.last_error = None
            return coords
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            self.last_error = "Google geocoding request failed. Check backend configuration and network."
            return None

    def get_route(self, origin, destination):
        import httpx
        self._connected = False
        failure = {"status": "NOT_CONFIGURED" if not self.configured else "UNAVAILABLE",
                   "distance_m": None, "instruction": None, "provider": self.name}
        if not self.configured:
            return {**failure, "note": "Google Maps is not configured."}
        def waypoint(point):
            return {"location": {"latLng": {"latitude": point[0], "longitude": point[1]}}}
        try:
            response = httpx.post("https://routes.googleapis.com/directions/v2:computeRoutes",
                headers={"X-Goog-Api-Key": self.api_key,
                         "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline,routes.legs.steps,routes.warnings"},
                json={"origin": waypoint(origin), "destination": waypoint(destination),
                      "travelMode": "WALK", "languageCode": "en-US"}, timeout=15)
            if response.status_code != 200:
                self.last_error = f"Google routing unavailable (HTTP {response.status_code}). Check API enablement, billing and key restrictions."
                return {**failure, "note": self.last_error}
            routes = response.json().get("routes", [])
            if not routes:
                return {**failure, "note": "No walking route found."}
            route = routes[0]
            steps = [{"instruction": step.get("navigationInstruction", {}).get("instructions", "Continue"),
                      "distance_m": step.get("distanceMeters"),
                      "duration_s": float(step["staticDuration"].rstrip("s")) if step.get("staticDuration") else None,
                      "maneuver": step.get("navigationInstruction", {}).get("maneuver"),
                      "end": step.get("endLocation", {}).get("latLng"),
                      "polyline": step.get("polyline", {}).get("encodedPolyline")}
                     for leg in route.get("legs", []) for step in leg.get("steps", [])]
            self.last_error = None
            self._connected = True
            return {"status": "OK", "provider": self.name, "distance_m": route.get("distanceMeters"),
                    "duration_s": float(route["duration"].rstrip("s")) if route.get("duration") else None,
                    "polyline": route.get("polyline", {}).get("encodedPolyline"), "steps": steps,
                    "warnings": route.get("warnings", []),
                    "instruction": steps[0]["instruction"] if steps else "Follow the walking route."}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            self.last_error = "Google routing request failed. Check backend configuration and network."
            return {**failure, "note": self.last_error}

    def health(self):
        return {"provider": self.name, "status": "CONFIGURED" if self.configured else "NOT_CONFIGURED",
                "connected": self._connected, "api_key_present": self.configured, "error": self.last_error}


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
        self.last_reroute: float = 0.0
        self.state: str = "IDLE"
        self.route = {}
        self.step_index = 0
        self.progress = None
        self.remaining_distance_m = None
        self.remaining_duration_s = None
        self.distance_to_maneuver_m = None
        self.maneuver = None
        self.location_available = False
        self._generation = getattr(self, "_generation", 0) + 1

    def stop(self) -> None:
        self.last_reroute = 0.0
        self.destination_name = None
        self.destination_coords = None
        self.last_instruction = None
        self.state = "CANCELLED"
        self.route = {}
        self.step_index = 0
        self.progress = None
        self.remaining_distance_m = None
        self.remaining_duration_s = None
        self.distance_to_maneuver_m = None
        self.maneuver = None
        self.location_available = False
        self._generation = getattr(self, "_generation", 0) + 1

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

        generation = self._generation
        coords = self.provider.resolve_destination(name)
        if generation != self._generation:
            return {"status": "CANCELLED", "text": "Navigation cancelled."}
        if coords is None:
            return {
                "status": "UNKNOWN_DESTINATION",
                "text": getattr(self.provider, "last_error", None) or f"I don't have a saved location for {name}.",
                "navigation": self.health(),
            }

        self.destination_name = name
        self.destination_coords = coords
        self.state = "ROUTING"
        route = self.provider.get_route(origin, coords)
        if generation != self._generation:
            return {"status": "CANCELLED", "text": "Navigation cancelled."}
        self.route = route
        if route.get("status") not in ("OK", "MOCK"):
            self.state = "UNAVAILABLE"
            return {"status": route.get("status", "UNAVAILABLE"), "text": route.get("note", "Routing unavailable."), "navigation": self.health()}
        self.state = "GUIDING"
        self.last_instruction = route.get("instruction")
        self.last_update = time.time()
        self.location_available = True
        self._update_progress(origin)
        return {
            "status": "STARTED",
            "text": f"Starting navigation to {name}. {self.last_instruction or ''}".strip(),
            "route": route,
            "navigation": self.health(),
        }

    def _update_progress(self, origin):
        """Project a measured GPS fix onto provider geometry; estimates are labelled."""
        points = decode_polyline(self.route.get("polyline") or "")
        if len(points) < 2:
            return
        _, along, total = project_on_route(origin, points)
        if total <= 0:
            return
        fraction = max(0.0, min(1.0, along / total))
        self.progress = round(fraction, 4)
        distance = self.route.get("distance_m")
        duration = self.route.get("duration_s")
        self.remaining_distance_m = round(distance * (1-fraction), 1) if distance is not None else None
        self.remaining_duration_s = round(duration * (1-fraction), 1) if duration is not None else None
        steps = self.route.get("steps", [])
        # Project Google step endpoints onto the same route geometry. This also
        # advances a missed maneuver when fixes arrive beyond its endpoint.
        while self.step_index < len(steps) - 1:
            end = steps[self.step_index].get("end") or {}
            if "latitude" not in end:
                break
            _, end_along, _ = project_on_route((end["latitude"], end["longitude"]), points)
            if along < end_along - 8:
                break
            self.step_index += 1
        if steps:
            step = steps[self.step_index]
            self.last_instruction = step.get("instruction")
            self.maneuver = step.get("maneuver")
            end = step.get("end") or {}
            if "latitude" in end:
                _, end_along, _ = project_on_route((end["latitude"], end["longitude"]), points)
                self.distance_to_maneuver_m = round(max(0, end_along-along), 1)

    def update(self, origin: Optional[Tuple[float, float]], min_interval_sec: float = 1.0) -> Optional[Dict[str, Any]]:
        if self.state not in ("GUIDING", "PAUSED_GPS", "OFF_ROUTE") or self.destination_coords is None:
            return None
        if origin is None or origin[0] is None or origin[1] is None:
            self.location_available = False
            if self.state == "PAUSED_GPS":
                return None
            self.state = "PAUSED_GPS"
            self.last_instruction = None
            return {"status": "GPS_UNAVAILABLE", "text": "Location lost. Navigation guidance is paused.",
                    "speak": True, "navigation": self.health()}

        now = time.time()
        resumed = self.state == "PAUSED_GPS"
        if not resumed and now - self.last_update < min_interval_sec:
            return None
        generation = self._generation
        previous_instruction = self.last_instruction
        previous_step = self.step_index
        self.state = "GUIDING"
        self.location_available = True
        self.last_update = now
        dist = haversine_m(*origin, *self.destination_coords)
        if dist <= settings.NAVIGATION_ARRIVAL_RADIUS_M:
            self.state = "ARRIVED"
            self.last_instruction = f"You have arrived at {self.destination_name}."
            self.progress = 1.0
            self.remaining_distance_m = self.remaining_duration_s = self.distance_to_maneuver_m = 0.0
            return {"status": "ARRIVED", "distance_m": round(dist, 1), "text": self.last_instruction,
                    "speak": True, "navigation": self.health()}

        if self.provider.name == "google":
            points = decode_polyline(self.route.get("polyline") or "")
            if len(points) >= 2 and distance_to_route(origin, points) > settings.NAVIGATION_OFF_ROUTE_RADIUS_M:
                self.state = "OFF_ROUTE"
                # No repeated paid routing requests while a noisy fix stays off route.
                if now - self.last_reroute < 15:
                    self.last_instruction = None
                    return {"status": "OFF_ROUTE", "text": "Off route. Guidance paused while waiting to reroute.",
                            "speak": previous_instruction is not None, "navigation": self.health()}
                self.last_reroute = now
                route = self.provider.get_route(origin, self.destination_coords)
                if generation != self._generation:
                    return None
                if route.get("status") != "OK":
                    self.state = "UNAVAILABLE"
                    self.last_instruction = None
                    return {"status": "UNAVAILABLE", "text": route.get("note"), "speak": True, "navigation": self.health()}
                self.route = route
                self.step_index = 0
                self.last_instruction = route.get("instruction")
                self.state = "GUIDING"
                self._update_progress(origin)
                return {"status": "REROUTED", "text": self.last_instruction, "speak": True, "navigation": self.health()}
            self._update_progress(origin)
            if self.last_instruction is None and self.route.get("steps"):
                self.last_instruction = self.route["steps"][self.step_index]["instruction"]
            return {"status": "GUIDING", "instruction": self.last_instruction, "text": self.last_instruction,
                    "speak": resumed or previous_step != self.step_index or previous_instruction != self.last_instruction,
                    "navigation": self.health()}

        route = self.provider.get_route(origin, self.destination_coords)
        if generation != self._generation:
            return None
        self.route = route
        self.last_instruction = route.get("instruction")
        return {"status": "GUIDING", "distance_m": round(dist, 1), "instruction": self.last_instruction,
                "text": self.last_instruction, "speak": resumed or previous_instruction != self.last_instruction,
                "navigation": self.health()}

    def where_am_i(self, origin: Optional[Tuple[float, float]]) -> str:
        if origin is None or origin[0] is None:
            return "Current location is unavailable."
        return f"Your current coordinates are latitude {origin[0]:.5f}, longitude {origin[1]:.5f}."

    def health(self) -> Dict[str, Any]:
        h = self.provider.health()
        h["active_destination"] = self.destination_name
        h["state"] = self.state
        h["route"] = self.route
        h["destination_coords"] = self.destination_coords
        h["instruction"] = self.last_instruction
        h["step_index"] = self.step_index
        h["last_update"] = self.last_update
        h["location_available"] = self.location_available
        h["progress"] = self.progress
        h["remaining_distance_m"] = self.remaining_distance_m
        h["remaining_duration_s"] = self.remaining_duration_s
        h["distance_to_maneuver_m"] = self.distance_to_maneuver_m
        h["maneuver"] = self.maneuver
        steps = self.route.get("steps", [])
        next_step = steps[self.step_index+1] if self.step_index+1 < len(steps) else {}
        h["next_maneuver"] = next_step.get("maneuver")
        h["next_instruction"] = next_step.get("instruction")
        h["progress_source"] = "GPS projection onto Google route" if self.progress is not None and self.provider.name == "google" else None
        return h

def decode_polyline(encoded):
    points, index, lat, lon = [], 0, 0, 0
    try:
        while index < len(encoded):
            deltas = []
            for _ in range(2):
                value = shift = 0
                while True:
                    byte = ord(encoded[index]) - 63
                    index += 1
                    value |= (byte & 31) << shift
                    shift += 5
                    if byte < 32:
                        break
                deltas.append(~(value >> 1) if value & 1 else value >> 1)
            lat += deltas[0]
            lon += deltas[1]
            points.append((lat / 1e5, lon / 1e5))
    except (IndexError, ValueError):
        return []
    return points


def project_on_route(origin, points):
    """Return lateral distance, distance along geometry, and total geometry length."""
    scale = math.cos(math.radians(origin[0]))
    xy = [((lon-origin[1])*111320*scale, (lat-origin[0])*111320) for lat,lon in points]
    best, along, accumulated = float("inf"), 0.0, 0.0
    for index, (a,b) in enumerate(zip(xy, xy[1:])):
        dx,dy = b[0]-a[0], b[1]-a[1]
        t = max(0, min(1, -(a[0]*dx+a[1]*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0
        distance = math.hypot(a[0]+t*dx, a[1]+t*dy)
        length = haversine_m(*points[index], *points[index+1])
        if distance < best:
            best, along = distance, accumulated+t*length
        accumulated += length
    return best, along, accumulated


def distance_to_route(origin, points):
    return project_on_route(origin, points)[0]
