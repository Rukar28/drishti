"""
VisionMate v2 - REST API Endpoints
Provides endpoints for:
- GET /health
- GET /status
- GET /scene
- GET /detections
- GET /metrics
- POST /api/v1/mode
- POST /api/v1/find
- POST /api/v1/read
- POST /api/v1/ask
- POST /api/v1/tts/stop
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import time
import subprocess

from backend.app.services.pipeline import pipeline
from backend.app.world_state.manager import world_state_mgr
from backend.app.core.config import settings

router = APIRouter()

class ModeRequest(BaseModel):
    mode: str

class FindRequest(BaseModel):
    target: str

class AskRequest(BaseModel):
    query: str

class ReadRequest(BaseModel):
    full_reading: bool = False

class VoiceCommandRequest(BaseModel):
    transcript: str

@router.get("/health")
def get_health() -> Dict[str, Any]:
    """Returns system health, GPU presence, and granular subsystem states."""
    import torch
    return {
        "status": "HEALTHY",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "camera_connected": pipeline.camera.is_connected(),
        "subsystems": pipeline.get_subsystems_health(),
        "timestamp": time.time()
    }

@router.get("/status")
def get_status() -> Dict[str, Any]:
    """Returns current operational mode, FPS, camera connection state, and telemetry."""
    state = world_state_mgr.get_snapshot()
    cam_health = pipeline.get_subsystems_health().get("camera", {})
    return {
        "mode": state.current_mode.value,
        "camera_source": cam_health.get("active_source", pipeline.camera_source_type),
        "camera_url": cam_health.get("active_url", settings.ESP32_CAPTURE_URL),
        "camera_configured": cam_health.get("configured_source", settings.CAMERA_SOURCE),
        "camera_connected": pipeline.camera.is_connected(),
        "connected": pipeline.camera.is_connected(),
        "camera_connection_state": cam_health.get("connection_state", "UNKNOWN"),
        "snapshot_fps": cam_health.get("snapshot_fps", cam_health.get("camera_received_fps", 0.0)),
        "camera_received_fps": cam_health.get("camera_received_fps", 0.0),
        "last_successful_capture_timestamp": cam_health.get("last_frame_timestamp", 0.0),
        "capture_latency_ms": cam_health.get("snapshot_request_latency_ms", 0.0),
        "camera_resolution": f"{cam_health.get('frame_width', 0)}x{cam_health.get('frame_height', 0)}",
        "dropped_frames": cam_health.get("dropped_frames", 0),
        "reconnect_count": cam_health.get("reconnect_count", 0),
        "frame_age_ms": cam_health.get("frame_age_ms", 0.0),
        "target_fps": settings.VISIONMATE_TARGET_FPS,
        "fps": pipeline.current_fps,
        "last_latency_ms": round(pipeline.last_latency_ms, 2),
        "active_object_count": len(state.get_active_objects()),
        "target_find_query": state.target_find_query,
        "last_spoken_narrative": state.last_spoken_narrative,
        "speech_generation": pipeline.tts.speech_generation
    }

@router.get("/scene")
def get_scene() -> Dict[str, Any]:
    """Returns complete World State snapshot."""
    return world_state_mgr.get_snapshot().model_dump()

@router.get("/detections")
def get_detections() -> List[Dict[str, Any]]:
    """Returns active tracked objects and directions."""
    state = world_state_mgr.get_snapshot()
    return [obj.model_dump() for obj in state.get_active_objects()]

@router.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    """Returns full performance telemetry including camera diagnostics."""
    import psutil
    proc = psutil.Process()
    cam_health = pipeline.get_subsystems_health().get("camera", {})
    return {
        "frames_processed": pipeline.frames_processed,
        "target_fps": settings.VISIONMATE_TARGET_FPS,
        "processed_fps": pipeline.current_fps,
        "effective_fps": pipeline.current_fps,
        "camera_received_fps": cam_health.get("camera_received_fps", 0.0),
        "snapshot_fps": cam_health.get("snapshot_fps", cam_health.get("camera_received_fps", 0.0)),
        "snapshot_request_latency_ms": cam_health.get("snapshot_request_latency_ms", 0.0),
        "avg_request_latency_ms": cam_health.get("avg_request_latency_ms", 0.0),
        "p95_request_latency_ms": cam_health.get("p95_request_latency_ms", 0.0),
        "jpeg_decode_latency_ms": cam_health.get("jpeg_decode_latency_ms", 0.0),
        "yolo_latency_ms": round(getattr(pipeline, "last_yolo_latency_ms", 0.0), 2),
        "total_perception_latency_ms": round(pipeline.last_latency_ms, 2),
        "last_inference_latency_ms": round(pipeline.last_latency_ms, 2),
        "dropped_frames": cam_health.get("dropped_frames", 0),
        "reconnect_count": cam_health.get("reconnect_count", 0),
        "decode_failures": cam_health.get("decode_failures", 0),
        "frame_age_ms": cam_health.get("frame_age_ms", 0.0),
        "last_frame_timestamp": cam_health.get("last_frame_timestamp", 0.0),
        "camera_source": cam_health.get("active_source", pipeline.camera_source_type),
        "camera_url": cam_health.get("active_url", settings.ESP32_CAPTURE_URL),
        "connected": pipeline.camera.is_connected(),
        "camera_connection_state": cam_health.get("connection_state", "UNKNOWN"),
        "active_url": cam_health.get("active_url", settings.ESP32_CAPTURE_URL),
        "process_ram_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
        "system_cpu_percent": psutil.cpu_percent(),
        "speech_engine_speaking": pipeline.tts.is_speaking(),
        "speech_generation": pipeline.tts.speech_generation
    }

# Interactive Action Endpoints
@router.post("/api/v1/voice_command")
def handle_voice_command(req: VoiceCommandRequest) -> Dict[str, Any]:
    """Parses and executes spoken voice commands (STOP, FIND, READ, ASK, GUIDANCE)."""
    return pipeline.handle_voice_command(req.transcript)

@router.post("/api/v1/mode")
def switch_mode(req: ModeRequest) -> Dict[str, Any]:
    res = pipeline.set_mode(req.mode)
    return {"status": "SUCCESS", "message": res, "current_mode": world_state_mgr.get_snapshot().current_mode.value}

@router.post("/api/v1/find")
def trigger_find(req: FindRequest) -> Dict[str, Any]:
    msg = pipeline.start_find(req.target)
    return {"status": "SUCCESS", "target": req.target, "message": msg}

@router.post("/api/v1/read")
def trigger_read(req: ReadRequest = ReadRequest()) -> Dict[str, Any]:
    res = pipeline.trigger_read(full_reading=req.full_reading)
    return {"status": "SUCCESS", "result": res}

@router.post("/api/v1/ask")
def trigger_ask(req: AskRequest) -> Dict[str, Any]:
    res = pipeline.trigger_ask(req.query)
    return {"status": "SUCCESS", "result": res}

@router.post("/api/v1/stop")
def stop_all() -> Dict[str, Any]:
    """Global STOP: stops TTS, cancels Find, purges queue, reverts to Guidance."""
    return pipeline.handle_stop_command()

@router.post("/api/v1/tts/stop")
def stop_speech() -> Dict[str, Any]:
    pipeline.tts.stop()
    return {"status": "SUCCESS", "message": "Speech interrupted."}

