"""
VisionMate v2 - Spatial Context & Scene Narration Engine
Synthesizes intelligent, human-friendly contextual statements from confirmed World State entities.
Produces calm, situational summaries:
"You are facing a desk with a laptop and bottle ahead. A person is on your left."
"""

from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from backend.app.schemas.world_state import (
    WorldState,
    TrackedObject,
    ImageDirection,
    PathZone,
    RelativeDepthBand,
    MotionTrend
)

class ContextEngine:
    """Generates intelligent spatial scene descriptions from structured World State."""

    @staticmethod
    def generate_scene_narrative(world_state: WorldState) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates:
        1. Natural, calm spatial narrative.
        2. Semantic event key for lifecycle tracking and silence deduplication.
        """
        stable_objects = world_state.get_stable_objects()
        if not stable_objects:
            return None, None

        # Check for dynamic crossings first (high informative value)
        for obj in stable_objects:
            if obj.motion_estimate in [MotionTrend.CROSSING_LEFT, MotionTrend.CROSSING_RIGHT]:
                key = f"guidance:crossing:{obj.class_name}:{obj.direction.value}"
                text = f"{obj.class_name.capitalize()} crossing {obj.direction.value}."
                return text, key

        # Categorize stable entities by spatial orientation
        center_objs = []
        left_objs = []
        right_objs = []
        
        # Check for desk / table focal point
        has_desk = any(obj.class_name in ["desk", "dining table", "table"] for obj in stable_objects)
        desk_items = []

        for obj in stable_objects:
            cls = obj.class_name.lower()
            dir_val = obj.direction
            
            # If on a desk, group items together
            if has_desk and cls in ["laptop", "bottle", "cup", "mouse", "keyboard", "book", "cell phone"]:
                desk_items.append(cls)
                continue

            if cls in ["desk", "dining table", "table"]:
                continue

            if dir_val in [ImageDirection.CENTER, ImageDirection.SLIGHT_LEFT, ImageDirection.SLIGHT_RIGHT]:
                center_objs.append(cls)
            elif dir_val in [ImageDirection.LEFT, ImageDirection.FAR_LEFT]:
                left_objs.append(cls)
            elif dir_val in [ImageDirection.RIGHT, ImageDirection.FAR_RIGHT]:
                right_objs.append(cls)

        def format_items(items: List[str]) -> str:
            if not items:
                return ""
            counts = defaultdict(int)
            for item in items:
                counts[item] += 1
            parts = []
            for name, cnt in counts.items():
                if cnt == 1:
                    article = "an" if name[0] in "aeiou" else "a"
                    parts.append(f"{article} {name}")
                else:
                    parts.append(f"{cnt} {name}s")
            if len(parts) == 1:
                return parts[0]
            elif len(parts) == 2:
                return f"{parts[0]} and {parts[1]}"
            else:
                return ", ".join(parts[:-1]) + f", and {parts[-1]}"

        sentences = []
        event_signature_parts = []

        if has_desk:
            if desk_items:
                sentences.append(f"You are facing a desk with {format_items(desk_items)} ahead.")
                event_signature_parts.append(f"desk_with_{'_'.join(sorted(desk_items))}")
            else:
                sentences.append("You are facing a desk ahead.")
                event_signature_parts.append("desk_ahead")
        elif center_objs:
            sentences.append(f"{format_items(center_objs).capitalize()} is ahead.")
            event_signature_parts.append(f"ahead_{'_'.join(sorted(center_objs))}")

        if left_objs:
            sentences.append(f"{format_items(left_objs).capitalize()} is on your left.")
            event_signature_parts.append(f"left_{'_'.join(sorted(left_objs))}")

        if right_objs:
            sentences.append(f"{format_items(right_objs).capitalize()} is on your right.")
            event_signature_parts.append(f"right_{'_'.join(sorted(right_objs))}")

        if not sentences:
            return None, None

        narrative = " ".join(sentences)
        event_key = f"guidance:{':'.join(event_signature_parts)}"
        return narrative, event_key

    @staticmethod
    def detect_scene_change(current_state: WorldState, last_spoken_signature: Optional[str]) -> bool:
        """
        Determines whether the confirmed entity distribution has changed enough to warrant evaluation.
        """
        stable_objects = current_state.get_stable_objects()
        if not stable_objects and not last_spoken_signature:
            return False

        current_signature_items = sorted([
            f"{obj.class_name}_{obj.direction.value}"
            for obj in stable_objects
        ])
        current_sig = "|".join(current_signature_items)
        return current_sig != last_spoken_signature
