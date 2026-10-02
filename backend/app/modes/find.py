"""
VisionMate v2 - Find Mode Handler with Session Invalidation
Interactive target search and real-time directional guidance.
Supports session invalidation, directional change filtering, and quiet cooldowns.
"""

import time
import uuid
from typing import Optional, Dict, Any, Tuple
import logging
from backend.app.schemas.world_state import WorldState, ImageDirection, TrackedObject
from backend.app.priority.engine import PriorityLevel

logger = logging.getLogger("visionmate.find")

class FindModeHandler:
    """
    Manages Find Mode state, session IDs, and directional updates.
    Ensures that starting a new Find session or calling STOP cancels prior tasks permanently.
    """
    def __init__(self, update_interval_sec: float = 2.0):
        self.target_query: Optional[str] = None
        self.target_track_id: Optional[int] = None
        self.last_spoken_direction: Optional[ImageDirection] = None
        self.last_guidance_time: float = 0.0
        self.update_interval_sec = update_interval_sec
        self.lost_announced = False
        self.session_id: Optional[str] = None
        self.is_active = False

    def set_target(self, target_name: str) -> Tuple[str, str]:
        """
        Starts a new Find session, invalidating any previous session.
        Returns: (spoken_announcement, session_id)
        """
        self.session_id = uuid.uuid4().hex[:8]
        self.target_query = target_name.lower().strip()
        self.target_track_id = None
        self.last_spoken_direction = None
        self.last_guidance_time = time.time()
        self.lost_announced = False
        self.is_active = True
        
        logger.info(f"Find Mode active [Session {self.session_id}] for target: '{self.target_query}'")
        return f"Searching for {self.target_query}.", self.session_id

    def stop_find(self) -> None:
        """Terminates active Find session immediately."""
        self.is_active = False
        self.session_id = None
        self.target_query = None
        self.target_track_id = None
        self.last_spoken_direction = None
        self.lost_announced = False
        logger.info("Find Mode deactivated and session cleared.")

    def process_frame_state(self, world_state: WorldState) -> Optional[Dict[str, Any]]:
        """
        Processes current world state to locate target and produce calm directional updates.
        Only speaks when target is newly acquired, direction changes materially, or target is lost.
        """
        if not self.is_active or not self.target_query:
            return None

        now = time.time()
        current_session = self.session_id

        # Step 1: Find matched entity in World State
        target_obj: Optional[TrackedObject] = None

        # Check if previous track ID is still present
        if self.target_track_id is not None:
            if self.target_track_id in world_state.tracked_objects:
                cand = world_state.tracked_objects[self.target_track_id]
                if not cand.is_stale(now):
                    target_obj = cand

        # If not found by track ID, search by class name among stable objects first
        if target_obj is None:
            stable_matches = [
                obj for obj in world_state.get_stable_objects()
                if self.target_query in obj.class_name.lower() or obj.class_name.lower() in self.target_query
            ]
            if stable_matches:
                target_obj = max(stable_matches, key=lambda x: x.confidence)
                self.target_track_id = target_obj.tracking_id
            else:
                active_matches = world_state.get_objects_by_class(self.target_query)
                if active_matches:
                    target_obj = max(active_matches, key=lambda x: x.confidence)
                    self.target_track_id = target_obj.tracking_id

        # Step 2: Target is visible
        if target_obj is not None:
            current_dir = target_obj.direction
            cls_name = target_obj.class_name
            
            # Check if this is the first observation or direction changed
            is_first_acquire = (self.last_spoken_direction is None)
            direction_changed = (current_dir != self.last_spoken_direction)
            timer_expired = (now - self.last_guidance_time) > 8.0  # Periodic reminder only after 8s

            if is_first_acquire or direction_changed or timer_expired:
                # Discard if direction is unchanged and within cooldown
                if not is_first_acquire and not direction_changed and (now - self.last_guidance_time) < self.update_interval_sec:
                    return None

                self.last_spoken_direction = current_dir
                self.last_guidance_time = now
                self.lost_announced = False

                if current_dir == ImageDirection.CENTER:
                    msg = f"{cls_name.capitalize()} is directly ahead."
                elif current_dir in [ImageDirection.SLIGHT_LEFT, ImageDirection.SLIGHT_RIGHT]:
                    msg = f"{cls_name.capitalize()} is {current_dir.value}."
                else:
                    msg = f"{cls_name.capitalize()} is {current_dir.value}."

                return {
                    "text": msg,
                    "priority": PriorityLevel.NAVIGATION,
                    "interrupt": False,
                    "session_id": current_session,
                    "source": "FIND"
                }

        # Step 3: Target lost handling
        else:
            if self.target_track_id is not None and not self.lost_announced:
                if (now - self.last_guidance_time) > 2.0:
                    self.lost_announced = True
                    self.target_track_id = None
                    self.last_spoken_direction = None
                    self.last_guidance_time = now
                    return {
                        "text": f"I lost sight of the {self.target_query}.",
                        "priority": PriorityLevel.NAVIGATION,
                        "interrupt": False,
                        "session_id": current_session,
                        "source": "FIND"
                    }

        return None
