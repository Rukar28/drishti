"""
Single command routing layer for VisionMate.

All spoken/text commands go through this bus. Intent parsing stays in the
existing ASR parser — this module does not invent a second parser.
"""

from typing import Any, Callable, Dict
import logging

from backend.app.core.events import EventType, event_broker

logger = logging.getLogger("visionmate.intent")

Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


class CommandBus:
    """Routes parsed intents to registered feature handlers."""

    def __init__(self, parse_intent: Callable[[str], Dict[str, Any]]):
        self._parse_intent = parse_intent
        self._handlers: Dict[str, Handler] = {}

    def register(self, intent: str, handler: Handler) -> None:
        self._handlers[intent.upper()] = handler

    def parse(self, transcript: str) -> Dict[str, Any]:
        return self._parse_intent(transcript)

    def dispatch(self, transcript: str) -> Dict[str, Any]:
        parsed = self._parse_intent(transcript or "")
        intent = str(parsed.get("intent") or "UNKNOWN").upper()
        logger.info("[INTENT] %s from '%s' conf=%.2f", intent, transcript, float(parsed.get("confidence") or 0.0))

        handler = self._handlers.get(intent)
        if handler is None:
            result = {
                "intent": intent,
                "target": parsed.get("target"),
                "confidence": parsed.get("confidence", 0.0),
                "status": "unrecognized" if intent == "UNKNOWN" else "not_available",
                "raw": transcript,
                "message": (
                    "I did not understand that command."
                    if intent == "UNKNOWN"
                    else f"{intent} is not available in this phase."
                ),
            }
        else:
            handled = handler(parsed) or {}
            result = {
                "intent": intent,
                "target": parsed.get("target"),
                "confidence": parsed.get("confidence", 0.0),
                "raw": transcript,
                **handled,
            }
            if "status" not in result:
                result["status"] = "ok"

        event_broker.publish(EventType.VOICE_COMMAND, result)
        return result
