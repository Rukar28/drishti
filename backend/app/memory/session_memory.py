"""
VisionMate v2 - Session Memory Provider
Implements MemoryProvider for ephemeral session context and history logging.
"""

from typing import List, Dict, Any
import time
from backend.app.core.interfaces import MemoryProvider

class SessionMemory(MemoryProvider):
    """In-memory rolling session memory store."""
    def __init__(self, max_events: int = 100):
        self.max_events = max_events
        self.events: List[Dict[str, Any]] = []

    def store_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        })
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.events[-limit:]

    def clear(self) -> None:
        self.events.clear()

session_memory = SessionMemory()
