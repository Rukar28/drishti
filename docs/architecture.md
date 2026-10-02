# VisionMate v2 - System Architecture Document

**Version:** 2.1.0  
**Status:** Approved Architectural Specification  
**System Role:** Principal System Architect  
**Hardware Baseline:** Intel i7-13620H | NVIDIA RTX 5050 Laptop GPU (8GB VRAM) | 16GB RAM | Windows 11  
**Labeling Standard:** All performance and resource metrics are explicitly categorized as `[TARGET]`, `[ESTIMATED]`, or `[MEASURED]`.

---

## 1. Architectural Philosophy & Priorities

**VisionMate v2** is a modular, high-reliability, real-time multimodal assistive platform designed for visually impaired users.

### 1.1 Core Engineering Principles
1. **Backend First:** Core reliability, low latency, fault isolation, testability, and deterministic arbitration take precedence over UI cosmetics.
2. **Strict Dual-Path Processing:**
   - **Fast-Perception Path (Reflex):** High-frequency, lightweight object detection and spatial relative depth running in a dedicated loop.
   - **Deep-Cognition Path (Reasoning):** Heavy Multimodal Vision-Language Models (VLM) and Optical Character Recognition (OCR) run strictly **on-demand** or event-triggered in isolated background workers without blocking camera ingestion or the fast perception loop.
3. **Non-Blocking Decoupled Audio:** Speech recognition (ASR) and synthesis (TTS) run asynchronously. Speech synthesis does not delay vision capture; urgent visual alerts can immediately preempt ongoing conversational speech.
4. **Structured World State as Single Source of Truth:** A centralized, structured World State powers Awareness, Find Mode, Ask Mode, Spatial Context, and Priority Arbitration.
5. **Rigorous Epistemic & Safety Distinctions:**
   - **`[Observed]`**: Directly sensed data (e.g., bounding boxes, camera pixel arrays, detected acoustic power).
   - **`[Estimated]`**: Statistically derived indicators (e.g., image-space relative depth gradients, bounding box motion vectors). *Monocular visual depth is explicitly treated as relative spatial estimation, not certified physical metric distance.*
   - **`[Inferred]`**: Semantic reasoning produced by generative models (e.g., VLM scene narrative, OCR text synthesis).
6. **Hardware-Constrained Resource Management:** GPU VRAM (8.0 GB) is a hard ceiling. Models are managed via a dedicated **Model Lifecycle Manager** to load, reuse, or unload weights dynamically.

---

## 2. System Component Boundaries & Provider Interfaces

To ensure high modularity, testability, and future extensibility without vendor lock-in, all capabilities are defined via clean Abstract Base Classes (Provider Interfaces):

```
+---------------------------------------------------------------------------------------+
|                                    PROVIDER INTERFACES                                |
+---------------------------------------------------------------------------------------+
| CameraProvider   : Ingests frames (ESP32-CAM, DirectShow Webcam, Synthetic Mock)     |
| ObjectDetector   : Bounding boxes, classes, confidences (YOLOv8/v11, MobileNet)       |
| RelativeDepth    : Spatial depth maps & proximity trends (Depth Anything V2 Small)    |
| Tracker          : Assigns persistent track IDs & tracks motion across frames         |
| OCRProvider      : Text extraction on-demand (PP-OCRv5 / PaddleOCR 3.x)               |
| VisionReasoner   : Multimodal visual Q&A / scene captioning (Qwen2.5-VL-3B / Moondream)|
| ASRProvider      : Streaming speech-to-text & VAD (Silero VAD + Faster-Whisper)       |
| TTSProvider      : Low-latency speech synthesis (Windows SAPI5 / OneCore / Piper)     |
| MemoryProvider   : Ephemeral & session context store for World State queries          |
| ModelLifecycle   : Manages VRAM allocation, model loading, caching, and unloading     |
+---------------------------------------------------------------------------------------+
```

### 2.1 Interface Definitions (Code Contract)

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np

class CameraProvider(ABC):
    @abstractmethod
    def start(self) -> bool: ...
    @abstractmethod
    def get_latest_frame(self) -> Optional[np.ndarray]: ...
    @abstractmethod
    def is_connected(self) -> bool: ...
    @abstractmethod
    def stop(self) -> None: ...

class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]: ...

class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]], frame_shape: tuple) -> List[Dict[str, Any]]: ...

class OCRProvider(ABC):
    @abstractmethod
    def extract_text(self, image_crop: np.ndarray) -> str: ...

class VisionReasoner(ABC):
    @abstractmethod
    def reason(self, frame: np.ndarray, prompt: str) -> str: ...

class ASRProvider(ABC):
    @abstractmethod
    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[str]: ...

class TTSProvider(ABC):
    @abstractmethod
    def speak(self, text: str, interrupt: bool = False) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...

class ModelLifecycleManager(ABC):
    @abstractmethod
    def acquire_model(self, model_name: str) -> Any: ...
    @abstractmethod
    def release_model(self, model_name: str) -> None: ...
    @abstractmethod
    def get_vram_usage_mb(self) -> float: ...
```

---

## 3. High-Level System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                      INPUT HARDWARE LAYER                                         |
|  +--------------------+   +-----------------------+   +-------------------+   +----------------+  |
|  | ESP32-CAM (Wi-Fi)  |   | ESP32-CAM (Direct AP) |   | Integrated Webcam |   | Mic Array (PCM)|  |
|  +---------+----------+   +-----------+-----------+   +---------+---------+   +-------+--------+  |
+------------|--------------------------|-------------------------|---------------------|-----------+
             |                          |                         |                     |
             +--------------------------+-------------------------+                     |
                                        | (Frames)                                      | (Audio)
                                        v                                               v
+-----------------------------------------------------------------------+   +-----------------------+
|                 ZERO-COPY RING BUFFER (CameraProvider)                |   |    ASRProvider / VAD  |
|   - Thread-safe circular buffer (Depth = 3)                           |   | (Silero + Whisper)    |
|   - Non-blocking drops on lag; Auto-reconnect watchdog                |   | Runs in worker thread |
+-----------------------------------+-----------------------------------+   +-----------+-----------+
                                    |                                                   |
                     +--------------+--------------+                                    |
                     |                             | (Triggered Keyframes)              | (User Intent)
                     v                             v                                    |
+------------------------------------+   +------------------------------------+         |
|    FAST PERCEPTION PATH (TIER 1)   |   |   ON-DEMAND REASONING PATH (TIER 2)|         |
|  - ObjectDetector (YOLOv8/11n)     |   |  - Triggered by Intent/Read Mode   |         |
|  - RelativeDepth (Spatial cues)    |   |  - VisionReasoner (Qwen2.5-VL-3B)  |         |
|  - Tracker (Persistent Track IDs)  |   |  - OCRProvider (PP-OCRv5/PaddleOCR)|         |
|  - Target: 20-30 FPS, non-blocking |   |  - Managed via ModelLifecycleMgr   |         |
+--------------------+---------------+   +-----------------+------------------+         |
                     |                                     |                            |
                     +------------------+------------------+                            |
                                        v                                               v
+---------------------------------------------------------------------------------------------------+
|                                 STRUCTURED WORLD STATE (MVP)                                      |
|   - Single Source of Truth for Awareness, Find Mode, Ask Mode, Context & Priority                 |
|   - tracked_objects: List[TrackedObject]                                                          |
|       * track_id, class_name, bbox [x1,y1,x2,y2], confidence                                     |
|       * image_space_direction ("left", "center", "right", "11 o'clock", "12 o'clock")             |
|       * relative_depth ("near", "mid", "far", relative gradient value)                            |
|       * motion ("approaching", "stationary", "receding")                                          |
|       * first_seen, last_seen timestamps                                                          |
|   - recent_events: RingBuffer[Event]                                                              |
|   - current_mode: Mode (AWARENESS | FIND | READ | ASK)                                            |
+---------------------------------------------------+-----------------------------------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------+
|                                 PRIORITY ARBITRATION & COOLDOWN                                   |
|   - Level 1 [CRITICAL]: Imminent hazard / drop-off -> Instant audio preemption & alert chime       |
|   - Level 2 [NAVIGATION]: Doorway, clear path, new obstacle -> Queued guidance                    |
|   - Level 3 [INTERACTION]: User question answer, OCR readout -> Conversational response           |
|   - Spatial Hysteresis & Cooldown Suppression (Prevents repetitive audio chatter)                 |
+---------------------------------------------------+-----------------------------------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------+
|                                     TTSProvider (AUDIO ENGINE)                                    |
|   - Native Windows SAPI5 / OneCore low-latency output                                             |
|   - Hardware interrupt support (PurgeBeforeSpeak on Level 1 Alert)                                |
|   - Earcon auditory icons (direction-panned proximity beeps)                                      |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Structured World State Specification (MVP)

The World State is the central, shared state representation accessed concurrently by all reasoning, navigation, and speech modules:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import time

class SystemMode(Enum):
    AWARENESS = "awareness"
    FIND = "find"
    READ = "read"
    ASK = "ask"

class RelativeDepthBand(Enum):
    VERY_NEAR = "very_near"  # Imminent proximity in camera view
    NEAR = "near"
    MID = "mid"
    FAR = "far"
    UNKNOWN = "unknown"

class MotionTrend(Enum):
    APPROACHING = "approaching"
    STATIONARY = "stationary"
    RECEDING = "receding"
    UNKNOWN = "unknown"

@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox: List[float]               # [x1, y1, x2, y2] normalized 0.0-1.0
    confidence: float               # [Observed] Detection confidence
    image_space_direction: str      # [Observed] "10 o'clock", "12 o'clock", "2 o'clock"
    relative_depth: RelativeDepthBand # [Estimated] Relative proximity band
    relative_depth_score: float     # [Estimated] Raw relative depth gradient (0.0 - 1.0)
    motion: MotionTrend             # [Estimated] Derived from bbox/depth delta over time
    first_seen: float               # Epoch timestamp
    last_seen: float                # Epoch timestamp

@dataclass
class WorldState:
    timestamp: float = field(default_factory=time.time)
    current_mode: SystemMode = SystemMode.AWARENESS
    tracked_objects: Dict[int, TrackedObject] = field(default_factory=dict)
    target_find_query: Optional[str] = None
    latest_ocr_text: Optional[str] = None
    latest_vlm_summary: Optional[str] = None
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    camera_connected: bool = False
    fps_telemetry: float = 0.0
```

---

## 5. Model Lifecycle & Resource Manager

With a dedicated **8.0 GB VRAM limit**, models cannot all reside in GPU memory at peak precision concurrently. The `ModelLifecycleManager` orchestrates model memory residency:

```
+---------------------------------------------------------------------------------------+
|                               MODEL RESIDENCY STRATEGY                                |
+---------------------------------------------------------------------------------------+
| RESIDENT POOL (Always Loaded):                                                        |
|  - YOLOv8n / v11n (ObjectDetector)            : [ESTIMATED] ~1.1 GB VRAM              |
|  - Depth-Anything-V2-Small (RelativeDepth)    : [ESTIMATED] ~0.7 GB VRAM              |
|  - Silero VAD (VAD)                           : [MEASURED] CPU execution (<10 MB RAM) |
|  - Faster-Whisper-tiny.en (ASRProvider)       : [ESTIMATED] ~0.5 GB VRAM              |
|  Subtotal Base Resident VRAM                   : [ESTIMATED] ~2.3 GB VRAM              |
+---------------------------------------------------------------------------------------+
| ON-DEMAND POOL (Scheduled / Mutually Exclusive or Quantized):                         |
|  - PP-OCRv5 / PaddleOCR 3.x (OCRProvider)     : [ESTIMATED] ~0.9 GB VRAM (Triggered)  |
|  - Qwen2.5-VL-3B-Instruct Q4 (VisionReasoner) : [ESTIMATED] ~3.4 GB VRAM (On-Demand)   |
|  Total Peak Allocated VRAM                     : [ESTIMATED] ~6.6 GB / 8.0 GB          |
|  VRAM Headroom / Safety Margin                 : [ESTIMATED] ~1.4 GB Free              |
+---------------------------------------------------------------------------------------+
```

### Scheduling Rules
1. **Fast Perception Priority:** Base resident models remain pinned in memory during active navigation.
2. **On-Demand OCR Triggering:** OCR runs only when:
   - User explicitly enters `READ` mode ("VisionMate, read this").
   - A high text-likelihood heuristic or specific text-bearing object (`sign`, `book`, `screen`, `label`) is centered in view.
3. **On-Demand VLM Triggering:** The Vision Reasoner is invoked only upon explicit user queries ("What is in front of me?") or significant scene transition triggers.
4. **OOM Guard:** The Model Lifecycle Manager queries `nvidia-smi` / `torch.cuda.memory_allocated()` before allocating new contexts.

---

## 6. OCR Architecture: PP-OCRv5 / Modern PaddleOCR 3.x

VisionMate v2 adopts modern **PP-OCRv5 / PaddleOCR 3.x** architectures:
- **Text Detection:** DBNet++ with lightweight MobileNetV4 / LCNet backbone for robust multi-angle bounding box detection.
- **Text Recognition:** SVTRv2 / LCNet-based text recognizer.
- **Trigger Filtering:** Pre-filtered by text region confidence $>0.75$ and image sharpness score (Laplacian variance $>100$) before passing crops to OCR, saving compute cycles.
- **Compatibility Testing:** Validated for direct local Windows DirectML / CUDA inference.

---

## 7. Operational Modes & Intent Routing

The system operates across 4 user-selectable or intent-triggered modes:

```
[ User Audio / Command ] ──► [ Intent Router ]
                                   │
       ┌───────────────────────────┼───────────────────────────┬───────────────────────────┐
       ▼                           ▼                           ▼                           ▼
[ AWARENESS MODE ]          [ FIND MODE ]               [ READ MODE ]               [ ASK MODE ]
Continuous obstacle         Tracks target item          Triggered OCR               On-demand VLM scene
detection & spatial         (e.g., "Find my keys")      (PP-OCRv5) on text          reasoning for open-ended
relative depth alerts       Alerts when in view         regions & reads aloud       conversational queries
```

---

## 8. Priority Engine & Preemption Matrix

To protect user safety and prevent auditory fatigue:

| Priority Level | Event Trigger | Speech Action | Cooldown Hysteresis |
| :--- | :--- | :--- | :--- |
| **Level 1 (CRITICAL)** | Imminent obstacle collision, descending step/drop-off, rapid approaching entity | **Instant Preempt:** Purge ongoing TTS buffer, play urgent earcon, announce warning | 1.0 s spatial cooldown |
| **Level 2 (NAVIGATION)**| Clear path ahead, doorway located, target object found in FIND mode | **Queue / Soft-Interrupt:** Speak after current sentence finishes | 4.0 s deduplication cooldown |
| **Level 3 (INTERACTION)**| Read Mode OCR readout, Ask Mode VLM description, battery status | **Conversational:** Play in standard conversational queue; cancelable by user speech or Level 1 alert | None |

---

## 9. Failure Isolation & Resiliency

1. **Camera Loss Isolation:** If ESP32-CAM drops connection, the `CameraProvider` serves cached frames with visual state warnings, triggers exponential reconnect backoff (100ms $\to$ 500ms $\to$ 2s), and falls back seamlessly to the integrated ACER HD webcam after 3 seconds.
2. **Deep Reasoning Isolation:** VLM and OCR run in isolated background threads. A timeout or exception in a deep reasoning query never interrupts the Fast Perception loop or drops camera frame rate.
3. **Audio Subsystem Isolation:** Audio playback utilizes asynchronous, non-blocking COM dispatch (`SAPI.SpVoice` with `SVSFlagsAsync`). A blocked sound device will not stall the vision or state tracking loop.

---

## 10. Benchmarking Infrastructure & Verification Suite

All system targets must be verifiable through automated benchmarking scripts located in `tests/benchmarks/`:

1. **`bench_reflex_latency.py`**: Injects synthetic video frames into `ObjectDetector` + `Tracker` + `PriorityArbitrator` + `TTSProvider`, measuring end-to-end trigger-to-audio latency (P50, P95, P99).
2. **`bench_vram_residency.py`**: Measures exact GPU memory before and after loading models, validating the VRAM budget on the physical RTX 5050 GPU.
3. **`bench_camera_resilience.py`**: Simulates Wi-Fi dropouts and packet corruption to measure recovery time and fallback behavior.
4. **`bench_concurrency_isolation.py`**: Executes heavy VLM queries concurrently with continuous 30 FPS camera detection to verify that Fast Perception frame rate does not drop by $>5\%$.
