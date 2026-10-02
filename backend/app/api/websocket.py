"""
VisionMate v2 - WebSocket Live Feed & Telemetry Streaming
Streams annotated video frames and real-time World State telemetry to the minimal competition dashboard.
"""

import asyncio
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
from backend.app.services.pipeline import pipeline
from backend.app.world_state.manager import world_state_mgr

logger = logging.getLogger("visionmate.ws")
ws_router = APIRouter()

@ws_router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Streams JPEG video frames and JSON telemetry over WebSocket."""
    await websocket.accept()
    logger.info("Dashboard WebSocket client connected.")
    try:
        while True:
            frame = pipeline.get_annotated_frame()
            if frame is not None:
                # Encode frame to JPEG
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    state = world_state_mgr.get_snapshot()
                    telemetry = {
                        "mode": state.current_mode.value,
                        "fps": pipeline.current_fps,
                        "latency_ms": round(pipeline.last_latency_ms, 1),
                        "objects": [obj.model_dump() for obj in state.get_active_objects()],
                        "last_speech": state.last_spoken_narrative
                    }
                    # Send binary JPEG frame + json metadata
                    await websocket.send_bytes(buffer.tobytes())
            
            # Send at ~20 FPS to conserve bandwidth
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")
