"""
VisionMate v2 - Priority Arbitration & Event Lifecycle Engine
Manages semantic event identity, state machine, deduplication, cooldowns, and silence hysteresis.
Separates object detection from hazard arbitration and eliminates repetitive chatter.
"""

import time
from typing import Optional, Dict, Any, Set, Tuple
from backend.app.schemas.world_state import (
    WorldState,
    TrackedObject,
    HazardSeverity,
    EventLifecycleState,
    PathZone,
    MotionTrend,
    ImageDirection
)

class PriorityLevel:
    CRITICAL_HAZARD = 1   # Purge and preempt all ongoing audio immediately
    NAVIGATION = 2        # High-priority directional updates
    INTERACTION = 3       # Calm guidance and conversational responses

class PriorityEngine:
    """
    Event-driven arbitration engine.
    Ensures: SEE -> UNDERSTAND -> DECIDE -> SPEAK -> SILENCE -> MONITOR FOR CHANGE.
    """
    def __init__(self, guidance_cooldown: float = 6.0, hazard_cooldown: float = 3.0, **kwargs):
        self.guidance_cooldown = kwargs.get("awareness_cooldown", guidance_cooldown)
        self.hazard_cooldown = hazard_cooldown
        self.last_speech_time = 0.0
        
        # Event tracking table: event_key -> Dict[str, Any]
        # {
        #   "state": EventLifecycleState,
        #   "first_seen": float,
        #   "announced_time": float,
        #   "last_text": str,
        #   "last_severity": HazardSeverity,
        #   "last_direction": str,
        #   "last_motion": str
        # }
        self._event_table: Dict[str, Dict[str, Any]] = {}

    def evaluate_hazard(self, world_state: WorldState) -> Optional[Dict[str, Any]]:
        """
        Scans for true hazards (EMERGENCY / WARNING).
        Does NOT trigger for ordinary static objects.
        Enforces post-announcement silence unless hazard escalates or changes.
        """
        now = time.time()
        stable_objects = world_state.get_stable_objects()

        for obj in stable_objects:
            severity = obj.hazard_severity
            if severity not in [HazardSeverity.EMERGENCY, HazardSeverity.WARNING]:
                continue

            cls = obj.class_name.lower()
            dir_val = obj.direction.value
            motion_val = obj.motion_estimate.value
            tid = obj.tracking_id
            
            # Semantic event identity
            event_key = f"hazard:{cls}:{tid if cls not in ['stairs', 'dropoff'] else 'structural'}"
            event_entry = self._event_table.get(event_key)

            # Construct appropriate warning phrasing
            if cls in ["stairs", "dropoff", "hole", "step"]:
                alert_text = f"Warning: {cls} {dir_val}."
            elif cls in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                if obj.motion_estimate in [MotionTrend.CROSSING_LEFT, MotionTrend.CROSSING_RIGHT]:
                    alert_text = f"Caution: {cls} moving across your path."
                else:
                    alert_text = f"Caution: {cls} approaching {dir_val}."
            elif obj.motion_estimate == MotionTrend.APPROACHING:
                alert_text = f"Caution: Moving object approaching {dir_val}."
            else:
                alert_text = f"Caution: Obstacle {dir_val}."

            # Determine whether this event should be announced
            should_announce = False
            is_escalation = False

            if event_entry is None:
                # Brand new confirmed hazard
                should_announce = True
            else:
                last_time = event_entry["announced_time"]
                last_sev = event_entry["last_severity"]
                last_dir = event_entry["last_direction"]
                
                # Check for severity escalation (WARNING -> EMERGENCY)
                if severity == HazardSeverity.EMERGENCY and last_sev != HazardSeverity.EMERGENCY:
                    should_announce = True
                    is_escalation = True
                # Check for material spatial transition
                elif last_dir != dir_val and (now - last_time) > self.hazard_cooldown:
                    should_announce = True
                # Re-announce only after long quiet period (e.g. 10s) if still active
                elif (now - last_time) > 10.0:
                    should_announce = True

            if should_announce:
                self._event_table[event_key] = {
                    "state": EventLifecycleState.ANNOUNCED,
                    "first_seen": obj.first_seen,
                    "announced_time": now,
                    "last_text": alert_text,
                    "last_severity": severity,
                    "last_direction": dir_val,
                    "last_motion": motion_val
                }
                self.last_speech_time = now

                return {
                    "text": alert_text,
                    "priority": PriorityLevel.CRITICAL_HAZARD if severity == HazardSeverity.EMERGENCY else PriorityLevel.NAVIGATION,
                    "interrupt": True if severity == HazardSeverity.EMERGENCY else False,
                    "event_key": event_key
                }

        return None

    def should_speak_guidance(self, event_key: str, narrative: str) -> bool:
        """
        Determines whether Guidance narration should be spoken.
        Enforces post-speech silence and event deduplication.
        """
        if not narrative or not narrative.strip():
            return False

        now = time.time()
        
        # Global minimum cooldown between any spoken guidance statements (silence period)
        if (now - self.last_speech_time) < self.guidance_cooldown:
            return False

        event_entry = self._event_table.get(event_key)
        if event_entry is not None:
            last_spoken_time = event_entry["announced_time"]
            last_text = event_entry["last_text"]
            
            # If identical guidance, require at least 20 seconds of silence before repeating
            if narrative == last_text:
                if (now - last_spoken_time) < 20.0:
                    return False
            else:
                # If wording changed slightly but within short window
                if (now - last_spoken_time) < self.guidance_cooldown:
                    return False

        # Record event announcement and begin silence period
        self._event_table[event_key] = {
            "state": EventLifecycleState.ANNOUNCED,
            "first_seen": now,
            "announced_time": now,
            "last_text": narrative,
            "last_severity": HazardSeverity.NONE,
            "last_direction": "",
            "last_motion": ""
        }
        self.last_speech_time = now
        return True

    def should_speak_awareness(self, narrative: str) -> bool:
        """Backwards compatibility alias for should_speak_guidance."""
        if not narrative:
            return False
        key = f"guidance:{narrative[:25]}"
        return self.should_speak_guidance(key, narrative)

    def reset_cooldowns(self):
        """Resets cooldown table (used for test setup or mode switch)."""
        self.last_speech_time = 0.0
        self._event_table.clear()
