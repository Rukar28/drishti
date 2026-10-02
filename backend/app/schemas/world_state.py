"""
VisionMate v2 - Structured World State & Entity Schemas
Defines models for the centralized World State:
- TrackedObject (with 3-state hierarchy: RAW -> STABLE -> EVENT)
- PathZone (INSIDE_PATH, NEAR_PATH, OUTSIDE_PATH)
- HazardSeverity (NONE, INFORMATION, NAVIGATION, WARNING, EMERGENCY)
- EventLifecycleState (DETECTED, CONFIRMED, ANNOUNCED, COOLDOWN, MONITORED, CHANGED, CLEARED)
- SystemMode (GUIDANCE / AWARENESS, FIND, READ, ASK)
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import time

class SystemMode(str, Enum):
    GUIDANCE = "guidance"
    AWARENESS = "awareness"  # alias for backwards compatibility
    FIND = "find"
    READ = "read"
    ASK = "ask"


class PathZone(str, Enum):
    """Spatial partitioning of the wearer's forward walking corridor."""
    INSIDE_PATH = "inside_path"    # Center collision/gait corridor (cx: 0.35 - 0.65)
    NEAR_PATH = "near_path"        # Immediate flank (cx: 0.22 - 0.35 or 0.65 - 0.78)
    OUTSIDE_PATH = "outside_path"  # Peripheral / non-interfering (cx < 0.22 or cx > 0.78)


class HazardSeverity(str, Enum):
    """Discrete internal severity ratings for the Hazard Engine."""
    NONE = "none"
    INFORMATION = "information"  # Context note (e.g. static furniture outside path)
    NAVIGATION = "navigation"    # Directional notice (e.g. person approaching side)
    WARNING = "warning"          # Moving object on collision path
    EMERGENCY = "emergency"      # Immediate dropoff/stairs/fast looming collision


class EventLifecycleState(str, Enum):
    """Lifecycle states for semantic events to eliminate audio spam."""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    ANNOUNCED = "announced"
    COOLDOWN = "cooldown"
    MONITORED = "monitored"
    CHANGED = "changed"
    CLEARED = "cleared"


class RelativeDepthBand(str, Enum):
    VERY_NEAR = "very_near"  # Imminent spatial proximity
    NEAR = "near"            # Close foreground
    MID = "mid"              # Medium distance
    FAR = "far"              # Background / distant
    UNKNOWN = "unknown"


class MotionTrend(str, Enum):
    APPROACHING = "approaching"
    STATIONARY = "stationary"
    RECEDING = "receding"
    CROSSING_LEFT = "crossing_left"
    CROSSING_RIGHT = "crossing_right"
    UNKNOWN = "unknown"


class ImageDirection(str, Enum):
    FAR_LEFT = "far left"
    LEFT = "on your left"
    SLIGHT_LEFT = "slightly left"
    CENTER = "directly ahead"
    SLIGHT_RIGHT = "slightly right"
    RIGHT = "on your right"
    FAR_RIGHT = "far right"


class TrackedObject(BaseModel):
    """
    Unified entity schema in World State.
    Distinguishes RAW detection vs confirmed STABLE object.
    """
    tracking_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: List[int] = Field(..., description="[x1, y1, x2, y2] in pixel coords")
    normalized_bbox: List[float] = Field(..., description="[x1, y1, x2, y2] in 0.0-1.0 coords")
    normalized_center: List[float] = Field(..., description="[cx, cy] in 0.0-1.0 coords")
    direction: ImageDirection = ImageDirection.CENTER
    path_zone: PathZone = PathZone.INSIDE_PATH
    relative_depth: RelativeDepthBand = RelativeDepthBand.UNKNOWN
    relative_depth_score: float = Field(default=0.0, description="Estimated spatial relative proximity 0.0-1.0")
    motion_estimate: MotionTrend = MotionTrend.UNKNOWN
    
    # Stability & Temporal confirmation
    is_stable: bool = Field(default=False, description="True once confirmed over multiple consecutive frames")
    stability_score: float = Field(default=0.0, description="0.0 to 1.0 temporal confirmation metric")
    frame_count: int = Field(default=1, description="Number of consecutive frames observed")
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    
    # Hazard & Event state
    hazard_severity: HazardSeverity = HazardSeverity.NONE
    hazard_score: float = Field(default=0.0, description="Internal computed hazard rating 0.0-1.0")
    last_announced_time: float = 0.0
    announced_state: Optional[str] = None

    def is_stale(self, current_time: float, timeout_sec: float = 1.5) -> bool:
        return (current_time - self.last_seen) > timeout_sec


class WorldState(BaseModel):
    """Central shared source of truth for all VisionMate modules."""
    timestamp: float = Field(default_factory=time.time)
    current_mode: SystemMode = SystemMode.GUIDANCE
    tracked_objects: Dict[int, TrackedObject] = Field(default_factory=dict)
    active_hazard: Optional[str] = None
    target_find_query: Optional[str] = None
    target_find_status: Optional[str] = None
    latest_ocr_text: Optional[str] = None
    latest_vlm_narrative: Optional[str] = None
    last_spoken_narrative: Optional[str] = None
    last_speech_timestamp: float = 0.0
    camera_connected: bool = False
    fps_telemetry: float = 0.0
    recent_events: List[Dict[str, Any]] = Field(default_factory=list)

    def get_active_objects(self) -> List[TrackedObject]:
        """Returns non-stale tracked entities."""
        now = time.time()
        return [obj for obj in self.tracked_objects.values() if not obj.is_stale(now)]

    def get_stable_objects(self) -> List[TrackedObject]:
        """Returns confirmed stable entities only (filters out single-frame noise)."""
        return [obj for obj in self.get_active_objects() if obj.is_stable]

    def get_objects_by_class(self, class_query: str) -> List[TrackedObject]:
        """Case-insensitive class matching."""
        q = class_query.lower().strip()
        return [
            obj for obj in self.get_active_objects()
            if q in obj.class_name.lower() or obj.class_name.lower() in q
        ]
