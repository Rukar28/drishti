"""
VisionMate v2 - Priority Arbitration & Event Lifecycle Engine
Manages semantic event identity, state machine, deduplication, cooldowns, and silence hysteresis.
Separates object detection from hazard arbitration and eliminates repetitive chatter.
"""

import time
from typing import Optional, Dict, Any, Set, Tuple
from backend.app.core.config import settings
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
    """
    Speech/event priority hierarchy (lower int = spoken earlier / higher priority).

    Conceptual mapping (PRD):
      P0 critical emergency / immediate collision risk -> STOP / EMERGENCY
      P1 immediate dangerous obstacle                  -> OBSTACLE
      P2 navigation / directional guidance             -> NAVIGATION
      P3 active user command                           -> USER_COMMAND
      P4 awareness                                     -> AWARENESS / INTERACTION
      P5 background information                        -> BACKGROUND
    STOP is the absolute interruption command and outranks everything.
    """
    STOP = 0              # absolute interruption — purge all speech
    EMERGENCY = 1         # P0 (alias: CRITICAL_HAZARD)
    CRITICAL_HAZARD = 1   # P0 — kept for backward compatibility
    OBSTACLE = 2          # P1 — immediate dangerous obstacle
    NAVIGATION = 3        # P2 — directional guidance
    USER_COMMAND = 4      # P3 — active user command responses
    INTERACTION = 5       # P4 — awareness / conversational responses
    AWARENESS = 5         # P4 alias
    BACKGROUND = 6        # P5 — background information

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

    def evaluate_depth_hazard(self, depth_zones: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Physical proximity hazard from the ToF depth sensor (P4 integration).

        Only fires when real depth hardware is connected and the center zone is
        within the configured near threshold. The mock provider reports
        connected=False, so laptop dev/testing is unaffected.
        """
        if not depth_zones or not depth_zones.get("connected"):
            return None

        center_mm = depth_zones.get("center_mm")
        if center_mm is None:
            return None

        threshold = settings.TOF_CENTER_NEAR_THRESHOLD_MM
        if center_mm >= threshold:
            return None

        now = time.time()
        event_key = "hazard:depth:center"
        entry = self._event_table.get(event_key)

        severity = HazardSeverity.EMERGENCY if center_mm <= max(300, threshold // 2) else HazardSeverity.WARNING
        dist_text = f"{center_mm / 1000.0:.1f} meters"
        alert_text = (
            f"Warning: obstacle very close ahead, about {dist_text}."
            if severity == HazardSeverity.EMERGENCY
            else f"Caution: obstacle ahead, about {dist_text}."
        )

        should_announce = False
        if entry is None:
            should_announce = True
        else:
            last_sev = entry.get("last_severity")
            last_time = entry.get("announced_time", 0.0)
            if severity == HazardSeverity.EMERGENCY and last_sev != HazardSeverity.EMERGENCY:
                should_announce = True
            elif (now - last_time) > self.hazard_cooldown:
                should_announce = True

        if not should_announce:
            return None

        self._event_table[event_key] = {
            "state": EventLifecycleState.ANNOUNCED,
            "first_seen": entry["first_seen"] if entry else now,
            "announced_time": now,
            "last_text": alert_text,
            "last_severity": severity,
            "last_direction": "center",
            "last_motion": "",
        }
        self.last_speech_time = now

        return {
            "text": alert_text,
            "priority": PriorityLevel.EMERGENCY if severity == HazardSeverity.EMERGENCY else PriorityLevel.OBSTACLE,
            "interrupt": True if severity == HazardSeverity.EMERGENCY else False,
            "event_key": event_key,
            "source": "DEPTH",
        }

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
