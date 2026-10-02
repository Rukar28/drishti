# VisionMate v2 - Performance Targets & Resource Budget

**Document Version:** 1.0.0  
**Target Hardware:** Intel Core i7-13620H (16 threads) | NVIDIA RTX 5050 Laptop GPU (8GB VRAM) | 16GB RAM  
**Standard:** Strict epistemic labeling of `[TARGET]`, `[ESTIMATED]`, and `[MEASURED]`.

---

## 1. Latency Targets & Benchmarking Specifications

| Processing Pipeline Stage | Metric Type | Target Value [TARGET] | Max Allowable SLA [TARGET] | Benchmark Script |
| :--- | :--- | :--- | :--- | :--- |
| **Camera Ingest to Frame Buffer** | P95 Latency | $\le$ **5 ms** | 10 ms | `bench_camera_resilience.py` |
| **Fast Object Detection (YOLOv8/11n)** | P95 Latency | $\le$ **20 ms** | 35 ms | `bench_reflex_latency.py` |
| **Relative Spatial Depth Estimation** | P95 Latency | $\le$ **25 ms** | 40 ms | `bench_reflex_latency.py` |
| **World State Tracking & Hungarian Match**| P95 Latency | $\le$ **3 ms** | 8 ms | `test_world_state.py` |
| **Priority Arbitration Decision** | P99 Latency | $\le$ **2 ms** | 5 ms | `test_priority_arbitration.py` |
| **TTS Speech Initiation (SAPI5/OneCore)** | P95 Latency | $\le$ **20 ms** | 40 ms | `test_speech_preemption.py` |
| **End-to-End Urgent Hazard Reflex** *(Frame Capture $\to$ Audio Output)* | **P95 Latency** | $\le$ **75 ms** | **< 110 ms** | `bench_reflex_latency.py` |
| **ASR Streaming Latency (Whisper Tiny INT8)**| P95 Latency | $\le$ **120 ms** | 200 ms | `bench_asr_latency.py` |
| **On-Demand OCR Extraction (PP-OCRv5)** | P95 Latency | $\le$ **250 ms** | 450 ms | `bench_ocr_latency.py` |
| **On-Demand VLM Reasoning (Qwen2.5-VL-3B Q4)**| P95 Latency | $\le$ **500 ms** | 900 ms | `bench_vlm_latency.py` |

> [!IMPORTANT]
> The numbers above represent strict engineering **[TARGET]** values. The actual **[MEASURED]** values will be recorded and added to this table only after executing the benchmark scripts on the physical RTX 5050 GPU.

---

## 2. Hardware Resource Budgets

### 2.1 GPU VRAM Allocation Budget (8,151 MiB Total [MEASURED])
```
+-------------------------------------------------------------------------------+
|                      GPU VRAM HARD BUDGET (8.0 GB TOTAL)                      |
+-------------------------------------------------------------------------------+
| BASE RESIDENT MODELS (Continuous Awareness & Fast Perception):                |
|  - ObjectDetector (YOLOv8n / v11n FP16)             : [TARGET] <= 1.2 GB       |
|  - RelativeDepth (Depth-Anything-V2-Small FP16)     : [TARGET] <= 0.8 GB       |
|  - ASRProvider (Faster-Whisper-tiny.en INT8)        : [TARGET] <= 0.6 GB       |
|  - CUDA Runtime Buffers & PyTorch Context Overhead  : [TARGET] <= 1.0 GB       |
|  Subtotal Resident Budget                           : [TARGET] <= 3.6 GB       |
+-------------------------------------------------------------------------------+
| ON-DEMAND / TRIGGERED WORKLOADS (ModelLifecycleManager Orchestrated):        |
|  - OCRProvider (PP-OCRv5 / PaddleOCR 3.x)           : [TARGET] <= 1.0 GB       |
|  - VisionReasoner (Qwen2.5-VL-3B-Instruct Q4)       : [TARGET] <= 3.4 GB       |
|  Peak Concurrent Allocation                         : [TARGET] <= 6.6 GB / 8.0 GB |
|  Guaranteed Free VRAM Headroom / Safety Margin      : [TARGET] >= 1.4 GB Free  |
+-------------------------------------------------------------------------------+
```

### 2.2 System RAM & CPU Budget
- **[MEASURED] Total System RAM:** 16,072 MB (~2,088 MB Free at idle Windows desktop).
- **[TARGET] VisionMate Process Working Set:** $\le$ **2.5 GB RAM**.
- **[TARGET] CPU Core Utilization (10 Cores / 16 Threads):**
  - Camera Ingest Worker: 1 Thread ($\le$ 5% CPU load)
  - Fast Perception & Tracker: 2 Threads ($\le$ 15% CPU load)
  - Voice VAD / Audio Output: 1 Thread ($\le$ 5% CPU load)
  - System Total CPU Load: $\le$ **30% average CPU utilization** during continuous operation.

---

## 3. Frame-Rate & Throughput Targets

| Subsystem | Input Source | Target Throughput [TARGET] | Min Acceptable FPS [TARGET] |
| :--- | :--- | :--- | :--- |
| **Camera Ingest (ESP32-CAM)** | Wi-Fi 6 MJPEG Stream | **25 – 30 FPS** | 15 FPS (under heavy Wi-Fi congestion) |
| **Camera Ingest (Webcam)** | DirectShow USB / Integrated | **30 FPS** | 30 FPS |
| **Fast Perception (Reflex)** | Camera Ingestion Ring Buffer | **20 – 30 FPS** | 15 FPS |
| **World State Tracking Updates**| Detections Stream | **30 Hz** | 20 Hz |
| **Dashboard WebSocket Feed** | Annotated Stream + HUD | **20 – 30 FPS** | 15 FPS |

---

## 4. Queue Capacity & Buffer Limits

To prevent memory leaks and latency blowups, all queues are strictly bounded:

```python
# System Queue Specifications [TARGET]
CAMERA_FRAME_BUFFER_DEPTH: int = 3       # Lock-free circular ring buffer (drops oldest frame on overrun)
DETECTION_EVENT_QUEUE_MAX: int = 10      # Bounded queue for spatial tracking
OCR_REQUEST_QUEUE_MAX: int = 2           # Single active + 1 pending read request
VLM_REQUEST_QUEUE_MAX: int = 1           # Mutually exclusive on-demand reasoning
AUDIO_COMMAND_QUEUE_MAX: int = 5         # Priority speech queue
WORLD_STATE_EVENT_HISTORY_MAX: int = 50  # Rolling telemetry history for UI / Context
```

---

## 5. Startup & Model Load Targets

| Lifecycle Milestone | Target Duration [TARGET] | Failure Threshold [TARGET] |
| :--- | :--- | :--- |
| **System Cold Start (Config & Event Bus Init)** | $\le$ **1.5 seconds** | 3.0 seconds |
| **Base Model Loading (YOLO + Depth + Whisper)** | $\le$ **4.0 seconds** | 8.0 seconds |
| **Camera Connection Establishment** | $\le$ **1.0 seconds** | 3.0 seconds |
| **Total Cold-Start to Active Awareness Ready** | $\le$ **6.5 seconds** | 12.0 seconds |
| **On-Demand VLM Warm Activation** | $\le$ **0.8 seconds** | 2.0 seconds |
| **On-Demand OCR Warm Activation** | $\le$ **0.4 seconds** | 1.0 seconds |

---

## 6. Failure Recovery Targets

| Failure Scenario | Target Recovery Time [TARGET] | Fallback Behavior [TARGET] |
| :--- | :--- | :--- |
| **ESP32-CAM Wi-Fi Disconnect** | $\le$ **500 ms** for first retry | Retries with backoff (100ms, 500ms, 2s); falls back to webcam after 3.0s |
| **Inference Thread Crash / Exception** | $\le$ **200 ms** actor restart | Watchdog restarts worker thread; serves last valid world state |
| **VLM Timeout (> 2.0s)** | $\le$ **2.1 seconds** | Cancels inference; alerts user ("Scene analysis timed out"); fast perception continues |
| **Audio Device Disconnect / Stall** | $\le$ **100 ms** reset | Resets COM SAPI voice instance without blocking perception loop |

---

## 7. Verification & Telemetry Table (To be filled during benchmarking)

| Pipeline Benchmark | Target Latency [TARGET] | Measured P50 [MEASURED] | Measured P95 [MEASURED] | Measured P99 [MEASURED] | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fast Object Detection (640x480) | $\le$ 20 ms | **16.76 ms** | **19.87 ms** | **21.12 ms** | **PASSED** |
| YOLOv8n GPU Inference Only | $\le$ 15 ms | **13.78 ms** | **16.37 ms** | **18.16 ms** | **PASSED** |
| Peak Detector VRAM Footprint | $\le$ 1.2 GB | **142.0 MiB** | **142.0 MiB** | **142.0 MiB** | **PASSED** |
| Memory Leak Growth (100 runs) | $\le$ 0.5 MB | **0.0000 MB** | **0.0000 MB** | **0.0000 MB** | **PASSED (Zero Leak)** |
| Relative Depth Estimation | $\le$ 25 ms | *Pending Phase 3* | *Pending Phase 3* | *Pending Phase 3* | Scheduled |
| Urgent Alert End-to-End | $\le$ 75 ms | *Pending Phase 4* | *Pending Phase 4* | *Pending Phase 4* | Scheduled |
| Whisper Streaming ASR | $\le$ 120 ms | *Pending Phase 4* | *Pending Phase 4* | *Pending Phase 4* | Scheduled |
| PP-OCRv5 Text Extraction | $\le$ 250 ms | *Pending Phase 5* | *Pending Phase 5* | *Pending Phase 5* | Scheduled |
| Qwen2.5-VL-3B VLM Q&A | $\le$ 500 ms | *Pending Phase 5* | *Pending Phase 5* | *Pending Phase 5* | Scheduled |
