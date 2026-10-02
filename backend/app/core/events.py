"""
VisionMate v2 - Asynchronous Event Bus & Typed Events
Provides lightweight, non-blocking Pub/Sub event dispatching across subsystems.
"""

import asyncio
import queue
import threading
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger("visionmate.events")

class EventType:
    FRAME_RECEIVED = "camera.frame_received"
    DETECTIONS_UPDATED = "perception.detections"
    WORLD_STATE_UPDATED = "world_state.updated"
    HAZARD_TRIGGERED = "hazard.triggered"
    SPEECH_REQUESTED = "speech.requested"
    MODE_CHANGED = "system.mode_changed"
    FIND_TARGET_UPDATED = "find.target_updated"
    OCR_RESULT_READY = "ocr.result_ready"
    VLM_RESULT_READY = "vlm.result_ready"
    # Public WebSocket envelope types (PRD)
    WORLD_UPDATE = "WORLD_UPDATE"
    DETECTION = "DETECTION"
    HAZARD = "HAZARD"
    VOICE_STARTED = "VOICE_STARTED"
    VOICE_TRANSCRIPT = "VOICE_TRANSCRIPT"
    VOICE_COMMAND = "VOICE_COMMAND"
    VOICE_STATE = "VOICE_STATE"
    TTS_STARTED = "TTS_STARTED"
    TTS_FINISHED = "TTS_FINISHED"
    MODE_CHANGED_PUBLIC = "MODE_CHANGED"
    NAVIGATION_UPDATE = "NAVIGATION_UPDATE"
    SENSOR_UPDATE = "SENSOR_UPDATE"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    ERROR = "ERROR"


class Event:
    def __init__(self, event_type: str, data: Any = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()


class EventBus:
    """Thread-safe and async-compatible publish/subscribe event bus."""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Any]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def publish(self, event_type: str, data: Any = None) -> None:
        event = Event(event_type, data)
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error executing event handler for {event_type}: {e}")

# Global event bus singleton
event_bus = EventBus()


def make_envelope(event_type: str, data: Any = None) -> Dict[str, Any]:
    """PRD WebSocket envelope: {type, timestamp, data}."""
    return {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": data if data is not None else {},
    }


class EventBroker:
    """Thread-safe fan-out for WebSocket JSON events."""

    def __init__(self, maxsize: int = 128):
        self._maxsize = maxsize
        self._queues: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    def publish(self, event_type: str, data: Any = None) -> Dict[str, Any]:
        envelope = make_envelope(event_type, data)
        event_bus.publish(event_type, data)
        with self._lock:
            targets = list(self._queues)
        for q in targets:
            try:
                q.put_nowait(envelope)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(envelope)
                except Exception:
                    pass
        return envelope


event_broker = EventBroker()
