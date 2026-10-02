"""
VisionMate v2 - Asynchronous Event Bus & Typed Events
Provides lightweight, non-blocking Pub/Sub event dispatching across subsystems.
"""

import asyncio
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
