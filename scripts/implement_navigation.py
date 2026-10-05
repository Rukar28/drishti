from pathlib import Path
p=Path('backend/app/navigation/navigator.py')
s=p.read_text(encoding='utf-8'); start=s.index('class GoogleDirectionsProvider'); end=s.index('\ndef create_navigation_provider',start)
s=s[:start]+'''class GoogleDirectionsProvider(NavigationProvider):
    """Google Geocoding + Routes API. Credentials and upstream errors stay private."""
    name = "google"

    def __init__(self, api_key=None):
        import os
        self.api_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or settings.GOOGLE_MAPS_API_KEY) if api_key is None else api_key
        self._geocode_cache = {}
        self.last_error = None

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
                      "end": step.get("endLocation", {}).get("latLng"),
                      "polyline": step.get("polyline", {}).get("encodedPolyline")}
                     for leg in route.get("legs", []) for step in leg.get("steps", [])]
            self.last_error = None
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
                "connected": False, "api_key_present": self.configured, "error": self.last_error}

''' + s[end:]
s=s.replace('self.state: str = "IDLE"','self.state: str = "IDLE"\n        self.route = {}\n        self.step_index = 0')
s=s.replace('self.state = "CANCELLED"','self.state = "CANCELLED"\n        self.route = {}\n        self.step_index = 0')
s=s.replace('"text": f"I don\'t have a saved location for {name}.",','"text": getattr(self.provider, "last_error", None) or f"I don\'t have a saved location for {name}.",')
s=s.replace('        self.state = "GUIDING"\n        self.last_instruction', '''        self.route = route
        if route.get("status") not in ("OK", "MOCK"):
            self.state = "UNAVAILABLE"
            return {"status": route.get("status", "UNAVAILABLE"), "text": route.get("note", "Routing unavailable."), "navigation": self.health()}
        self.state = "GUIDING"
        self.last_instruction''',1)
s=s.replace('        route = self.provider.get_route(origin, self.destination_coords)\n        instruction = route.get("instruction")', '''        if self.provider.name == "google":
            # Advance along actual Google steps; never invent a straight-line instruction.
            steps = self.route.get("steps", [])
            while self.step_index < len(steps) - 1:
                end = steps[self.step_index].get("end") or {}
                if "latitude" not in end or haversine_m(*origin, end["latitude"], end["longitude"]) > 20:
                    break
                self.step_index += 1
            self.last_update = now
            if steps:
                self.last_instruction = steps[self.step_index]["instruction"]
            # Re-route when GPS is more than 60m from the route geometry.
            points = decode_polyline(self.route.get("polyline") or "")
            if points and distance_to_route(origin, points) > 60:
                self.state = "OFF_ROUTE"
                route = self.provider.get_route(origin, self.destination_coords)
                if route.get("status") != "OK":
                    self.state = "UNAVAILABLE"
                    return {"status": "UNAVAILABLE", "text": route.get("note"), "navigation": self.health()}
                self.route = route
                self.step_index = 0
                self.last_instruction = route.get("instruction")
                self.state = "GUIDING"
                return {"status": "REROUTED", "text": self.last_instruction, "navigation": self.health()}
            return {"status": "GUIDING", "instruction": self.last_instruction, "text": self.last_instruction, "navigation": self.health()}
        route = self.provider.get_route(origin, self.destination_coords)
        self.route = route
        instruction = route.get("instruction")''')
s=s.replace('        h["state"] = self.state','''        h["state"] = self.state
        h["route"] = self.route
        h["destination_coords"] = self.destination_coords
        h["instruction"] = self.last_instruction
        h["step_index"] = self.step_index
        h["last_update"] = self.last_update''')
s += '''

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


def distance_to_route(origin, points):
    # Local tangent-plane point-to-segment distance, in meters.
    scale = math.cos(math.radians(origin[0]))
    xy = [((lon-origin[1])*111320*scale, (lat-origin[0])*111320) for lat,lon in points]
    best = float("inf")
    for a,b in zip(xy, xy[1:]):
        dx,dy = b[0]-a[0], b[1]-a[1]
        t = max(0, min(1, -(a[0]*dx+a[1]*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0
        best = min(best, math.hypot(a[0]+t*dx, a[1]+t*dy))
    return best
'''
p.write_text(s,encoding='utf-8')
p=Path('backend/app/hardware/gps.py');s=p.read_text();a=s.index('class BrowserGPSProvider');b=s.index('\ndef create_gps_provider',a)
s=s[:a]+'''class BrowserGPSProvider(GPSProvider):
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

''' + s[b:];p.write_text(s)
