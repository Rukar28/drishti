"""
VisionMate v2 - End-to-End Pipeline and API Integration Tests
"""

import time
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.pipeline import VisionMatePipeline
from backend.app.schemas.world_state import SystemMode

@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client

def test_fastapi_core_endpoints(test_client):
    # Test /health
    res = test_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "app" in data

    # Test /status
    res = test_client.get("/status")
    assert res.status_code == 200
    status_data = res.json()
    assert "mode" in status_data
    assert "active_object_count" in status_data

    # Test /detections
    res = test_client.get("/detections")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # Test /metrics
    res = test_client.get("/metrics")
    assert res.status_code == 200
    metrics = res.json()
    # Accept either the renamed field 'processed_fps' or legacy 'effective_fps'
    assert "processed_fps" in metrics or "effective_fps" in metrics

def test_prd_api_aliases(test_client):
    res = test_client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

    res = test_client.get("/api/world")
    assert res.status_code == 200
    world = res.json()
    assert "mode" in world
    assert "objects" in world
    assert "hazards" in world
    assert world.get("navigation") is None or isinstance(world.get("navigation"), dict)

    res = test_client.post("/api/voice/command", json={"transcript": "find my bottle"})
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "FIND"
    assert data["target"] == "bottle"
    assert data["status"] == "started"

    res = test_client.post("/api/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "STOPPED"


def test_mode_switching_api(test_client):
    # Switch to Guidance / Awareness
    res = test_client.post("/api/v1/mode", json={"mode": "guidance"})
    assert res.status_code == 200
    assert res.json()["current_mode"] in ["guidance", "awareness"]

    # Switch to Find Mode
    res = test_client.post("/api/v1/find", json={"target": "bottle"})
    assert res.status_code == 200
    assert "bottle" in res.json()["message"].lower()

    # Switch to Read Mode
    res = test_client.post("/api/v1/read", json={"full_reading": False})
    assert res.status_code == 200
    assert "result" in res.json()

    # Switch to Ask Mode
    res = test_client.post("/api/v1/ask", json={"query": "What is in front of me?"})
    assert res.status_code == 200
    assert "result" in res.json()

def test_pipeline_synthetic_camera_loop():
    # Instantiate standalone pipeline with SyntheticMockCamera
    pipe = VisionMatePipeline(camera_source="mock")
    started = pipe.start()
    assert started is True

    # Wait for actual CPU model warm-up instead of assuming a 1.5s startup.
    # Retain a hard deadline so a dead worker still fails the test.
    deadline = time.monotonic() + 10
    while pipe.frames_processed == 0 and time.monotonic() < deadline:
        time.sleep(0.05)

    assert pipe.frames_processed > 0
    annotated = pipe.get_annotated_frame()
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)

    # Test Find Target setting
    find_msg = pipe.start_find("laptop")
    assert "laptop" in find_msg.lower()

    pipe.stop()
    assert pipe._running is False
