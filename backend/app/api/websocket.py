"""
VisionMate v2 - WebSocket Live Feed & Typed Event Streaming
"""

import asyncio
import json
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
from backend.app.services.pipeline import pipeline
from backend.app.world_state.manager import world_state_mgr
from backend.app.core.events import EventType, event_broker, make_envelope

logger = logging.getLogger("visionmate.ws")
ws_router = APIRouter()

@ws_router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Streams JPEG video frames over WebSocket (existing dashboard contract)."""
    await websocket.accept()
    logger.info("Dashboard WebSocket client connected.")
    try:
        while True:
            frame = pipeline.get_annotated_frame()
            if frame is not None:
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")


@ws_router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """JSON envelopes: {type, timestamp, data}."""
    await websocket.accept()
    q = event_broker.subscribe()
    logger.info("Events WebSocket client connected.")
    try:
        await websocket.send_text(json.dumps(make_envelope(EventType.SYSTEM_STATUS, {
            "voice_state": pipeline.voice_state,
            "mode": world_state_mgr.get_snapshot().current_mode.value,
        })))
        while True:
            try:
                envelope = q.get_nowait()
                await websocket.send_text(json.dumps(envelope))
            except Exception:
                view = pipeline.public_world_view()
                await websocket.send_text(json.dumps(make_envelope(EventType.WORLD_UPDATE, view)))
                await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        logger.info("Events WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket events error: {e}")
    finally:
        event_broker.unsubscribe(q)
