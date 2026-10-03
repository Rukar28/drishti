"""
P6 — Navigation architecture: GPS → destination → route → voice guidance.

Live routing credentials are unavailable in this environment, so the mock
provider is exercised. We assert the system degrades gracefully and never
fabricates a route.
"""

import pytest

from backend.app.navigation.navigator import (
    Navigator, MockNavigationProvider, GoogleDirectionsProvider,
    create_navigation_provider, haversine_m, bearing_to_compass,
)


def test_haversine_and_compass_helpers():
    d = haversine_m(12.9716, 77.5946, 12.9352, 77.6245)
    assert 1000 < d < 8000
    assert bearing_to_compass(0) == "north"
    assert bearing_to_compass(90) == "east"


def test_mock_provider_reports_mock_not_live():
    p = MockNavigationProvider({"home": (12.9716, 77.5946)})
    assert p.health()["status"] == "MOCK"
    route = p.get_route((12.9716, 77.5946), (12.9352, 77.6245))
    assert route["status"] == "MOCK"
    assert route["instruction"]


def test_navigator_start_unknown_destination():
    nav = Navigator(MockNavigationProvider({"home": (12.9716, 77.5946)}))
    res = nav.start("nowhere", (12.9, 77.5))
    assert res["status"] == "UNKNOWN_DESTINATION"


def test_navigator_start_without_gps_is_graceful():
    nav = Navigator(MockNavigationProvider({"home": (12.9716, 77.5946)}))
    res = nav.start("home", (None, None))
    assert res["status"] == "GPS_UNAVAILABLE"


def test_navigator_start_and_guidance():
    nav = Navigator(MockNavigationProvider({
        "home": (12.9716, 77.5946), "office": (12.9352, 77.6245),
    }))
    res = nav.start("office", (12.9716, 77.5946))
    assert res["status"] == "STARTED"
    assert nav.state == "GUIDING"
    # Immediately after start we are within the update interval -> no repeat
    assert nav.update((12.9716, 77.5946)) is None
    # After forcing the interval to elapse, guidance is produced
    nav.last_update = 0.0
    upd = nav.update((12.9716, 77.5946))
    assert upd is not None
    assert upd["status"] == "GUIDING"
    assert upd["instruction"]


def test_navigator_arrival():
    nav = Navigator(MockNavigationProvider({"home": (12.9716, 77.5946)}))
    nav.start("home", (12.9716, 77.5946))  # already at destination
    nav.last_update = 0.0
    upd = nav.update((12.9716, 77.5946))
    assert upd is not None
    assert upd["status"] == "ARRIVED"


def test_navigator_stop_cancels():
    nav = Navigator(MockNavigationProvider({"home": (12.9716, 77.5946)}))
    nav.start("home", (12.90, 77.50))
    nav.stop()
    assert nav.state == "CANCELLED"
    assert nav.update((12.90, 77.50)) is None


def test_google_provider_not_configured_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    p = GoogleDirectionsProvider(api_key="")
    h = p.health()
    assert h["status"] == "NOT_CONFIGURED"
    assert h["api_key_present"] is False
    route = p.get_route((1.0, 1.0), (2.0, 2.0))
    assert route["status"] == "NOT_CONFIGURED"
    assert route["distance_m"] is None


def test_create_navigation_provider_defaults_to_mock():
    assert isinstance(create_navigation_provider("mock"), MockNavigationProvider)
    assert isinstance(create_navigation_provider("google"), GoogleDirectionsProvider)


def test_pipeline_navigation_wiring_graceful_without_gps():
    from backend.app.services.pipeline import VisionMatePipeline
    pipe = VisionMatePipeline(camera_source="mock")
    # Mock GPS returns lat=None -> navigation must fail gracefully, not crash.
    res = pipe.handle_voice_command("navigate to home")
    assert res["intent"] == "NAVIGATION"
    assert res["status"] == "gps_unavailable"
