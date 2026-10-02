"""
VisionMate v2 - Core Provider Interfaces
Defines strict Abstract Base Classes (ABCs) for all modular system components:
- CameraProvider
- ObjectDetector
- Tracker
- OCRProvider
- VisionReasoner
- TTSProvider
- ASRProvider
- MemoryProvider
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

class CameraProvider(ABC):
    """Abstract interface for video frame sources (ESP32-CAM, Webcam, Mock)."""
    @abstractmethod
    def start(self) -> bool:
        """Start the camera stream capture thread/connection."""
        pass

    @abstractmethod
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Fetch the most recent unconsumed frame (non-blocking)."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if camera feed is alive and healthy."""
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release camera resource and stop background workers."""
        pass


class ObjectDetector(ABC):
    """Abstract interface for high-speed object detection (YOLO26n / YOLOv8n)."""
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Executes fast object detection.
        Returns list of detections with format:
        [
            {
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2] (pixel coords),
                "norm_bbox": [x1, y1, x2, y2] (0.0 - 1.0)
            },
            ...
        ]
        """
        pass


class Tracker(ABC):
    """Abstract interface for multi-object tracking across frames."""
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Associates detections with persistent track IDs.
        Returns detections augmented with 'tracking_id', 'motion_trend', and 'center'.
        """
        pass


class OCRProvider(ABC):
    """Abstract interface for optical character recognition (PP-OCRv5 / PaddleOCR 3.x)."""
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extracts text bounding boxes and clean reading-order text.
        Returns:
        {
            "full_text": str,
            "short_summary": str,
            "text_blocks": List[Dict[str, Any]],
            "has_text": bool
        }
        """
        pass


class VisionReasoner(ABC):
    """Abstract interface for on-demand deep VLM reasoning (Qwen3-VL 4B via Ollama)."""
    @abstractmethod
    def reason(self, frame: np.ndarray, prompt: str) -> str:
        """Generates deep scene narrative or answers user queries from image."""
        pass


class TTSProvider(ABC):
    """Abstract interface for local text-to-speech engine."""
    @abstractmethod
    def speak(self, text: str, priority: int = 2, interrupt: bool = False) -> None:
        """Queues or immediately announces speech with priority level."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Interrupts ongoing speech immediately and clears speech queue."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Returns True if audio is currently playing."""
        pass


class ASRProvider(ABC):
    """Abstract interface for local streaming speech-to-text."""
    @abstractmethod
    def listen_chunk(self, pcm_audio: np.ndarray) -> Optional[str]:
        """Transcribes incoming audio stream chunk."""
        pass


class MemoryProvider(ABC):
    """Abstract interface for session context memory."""
    @abstractmethod
    def store_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Stores structured scene event."""
        pass

    @abstractmethod
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves most recent events."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears memory."""
        pass
