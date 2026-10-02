"""
VisionMate v2 - Guidance / Awareness Mode Handler
Manages continuous background spatial perception, event confirmation,
and priority-governed narration. Remains completely silent after speaking until changes occur.
"""

from typing import Optional, Dict, Any, Tuple
import logging
from backend.app.schemas.world_state import WorldState, SystemMode
from backend.app.context.engine import ContextEngine
from backend.app.priority.engine import PriorityEngine, PriorityLevel

logger = logging.getLogger("visionmate.guidance")

class AwarenessModeHandler:
    """Handles Guidance Mode logic: continuous tracking and calm, selective narration."""
    def __init__(self, priority_engine: PriorityEngine):
        self.priority_engine = priority_engine
        self.last_spoken_signature: Optional[str] = None

    def process_frame_state(self, world_state: WorldState) -> Optional[Dict[str, Any]]:
        """
        Evaluates current World State for emergency hazards or meaningful guidance events.
        Enforces: SEE -> CONFIRM -> UNDERSTAND -> DECIDE -> SPEAK -> SILENCE -> MONITOR.
        """
        # 1. First evaluate genuine Hazards (stairs/dropoffs, fast vehicles, path collisions)
        hazard_event = self.priority_engine.evaluate_hazard(world_state)
        if hazard_event is not None:
            return hazard_event

        # 2. Extract stable scene narrative and semantic event identity
        narrative, event_key = ContextEngine.generate_scene_narrative(world_state)
        if not narrative or not event_key:
            return None

        # 3. Check event lifecycle and post-speech cooldown
        if self.priority_engine.should_speak_guidance(event_key, narrative):
            logger.info(f"[GUIDANCE EVENT]: '{narrative}' (key: {event_key})")
            return {
                "text": narrative,
                "priority": PriorityLevel.INTERACTION,
                "interrupt": False,
                "event_key": event_key
            }

        return None
