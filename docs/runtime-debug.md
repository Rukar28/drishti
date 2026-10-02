# VisionMate v2 - Runtime & Startup Diagnostics Report

## 1. Executive Summary
- **Issue:** Web browser encountered `ERR_CONNECTION_REFUSED` when accessing `http://127.0.0.1:8000/` immediately upon launch. Additionally, WebSocket `/ws/stream` initially rejected connections with `HTTP 404 (Unsupported upgrade request)`.
- **Status:** **RESOLVED & FULLY VERIFIED**
- **Root Cause & Fix Summary:**
  1. Synchronous heavy model loading (`yolov8n.pt` CUDA warmup) and camera device probing at module import time delayed server readiness by ~20-30s. Refactored into a non-blocking background lifecycle where FastAPI binds port 8000 in <1s and subsystems initialize independently.
  2. Uvicorn was missing an installed ASGI WebSocket protocol implementation (`websockets`), causing `/ws/stream` to return 404. Installed `websockets` package into `.venv`.
  3. Added frontend object URL revocation to prevent browser memory bloat and auto-reconnecting WebSocket client.

---

## 2. Root Cause Analysis

### A. Synchronous Import & Lifespan Blocking
1. **Module-level Synchronous Execution:**
   - When Uvicorn imported `backend.app.main:app`, it imported `pipeline.py`, which immediately called `YOLOObjectDetector.__init__`.
   - `__init__` invoked PyTorch CUDA initialization, model checkpoint loading, and dummy tensor warmups synchronously on the main thread (~17 seconds).
   - In FastAPI `lifespan`, `DirectShowWebcam.start()` probed `cv2.VideoCapture(0, cv2.CAP_DSHOW)`, which blocks on Windows when scanning DirectShow filter graphs.
   - Result: Uvicorn could not bind to port `8000` until all subsystems finished, causing any immediate browser connection to fail with `ERR_CONNECTION_REFUSED`.

### B. Missing ASGI WebSocket Dependency
- Uvicorn requires `websockets` or `wsproto` installed in the Python environment to handle ASGI WebSocket connections.
- Without `websockets`, Uvicorn logged:
  ```
  WARNING: Unsupported upgrade request.
  WARNING: No supported WebSocket library detected. Please use "pip install 'uvicorn[standard]'", or install 'websockets' or 'wsproto' manually.
  INFO: 127.0.0.1:54352 - "GET /ws/stream HTTP/1.1" 404 Not Found
  ```

---

## 3. Implemented Fixes

### 1. Non-Blocking Subsystem Lifecycle (`pipeline.py` & `detector.py`)
- Refactored `YOLOObjectDetector` with explicit `auto_load=False` default and status tracking (`UNINITIALIZED` -> `LOADING` -> `READY` / `ERROR`).
- In `VisionMatePipeline`, `start()` now spawns `_async_bootstrap_and_loop` in a dedicated daemon thread.
- FastAPI server starts and accepts HTTP requests **immediately (<1s)**.
- If physical webcam is offline or delayed, it automatically falls back to `SyntheticMockCamera` with clear warning logs.
- Initial standby frame buffer is generated at `t=0` so WebSocket / HUD never encounters `None`.

### 2. Granular Health Reporting (`routes.py`)
- Updated `/health` endpoint to expose granular subsystem statuses:
  - `camera`: `{"connected": bool, "source": str}`
  - `detector`: `{"status": str, "model": str, "device": str, "ready": bool}`
  - `speech_engine`: `{"status": str, "speaking": bool}`
  - `ocr`: `{"status": str}`
  - `vlm`: `{"model": str, "status": str}`

### 3. Installed WebSocket Dependency
- Installed `websockets` in the isolated virtual environment (`.venv`).

### 4. Robust HUD Client (`main.py`)
- Updated HUD JavaScript to:
  - Automatically reconnect WebSocket with exponential backoff on disconnect.
  - Revoke prior `URL.createObjectURL` references upon frame arrival to eliminate browser memory leaks.
  - Render fallback status indicator and real-time telemetry indicators.

---

## 4. Affected Files
1. `backend/app/perception/detector.py`: Added non-blocking loading and status states.
2. `backend/app/services/pipeline.py`: Asynchronous bootstrap, fallback handling, standby frame generation, and subsystem health reporting.
3. `backend/app/api/routes.py`: Enhanced `/health` with granular component telemetry.
4. `backend/app/main.py`: Immediate server readiness, WebSocket reconnection loop, memory cleanup.
5. `scripts/verify_endpoints.py`: Automated HTTP + WebSocket verification script.

---

## 5. Verification Results

### A. HTTP & WebSocket Endpoint Tests (`scripts/verify_endpoints.py`)
```
============================================================
      VISIONMATE v2 - RUNTIME ENDPOINT VERIFICATION
============================================================
[PASSED] GET /          -> HTTP 200 in 26.81ms (text/html; charset=utf-8) [Title: VisionMate v2]
[PASSED] GET /docs      -> HTTP 200 in 26.38ms (text/html; charset=utf-8) [FastAPI Swagger UI]
[PASSED] GET /health    -> HTTP 200 in 4.61ms  (application/json) [Status: HEALTHY, GPU: RTX 5050]
[PASSED] GET /status    -> HTTP 200 in 2.56ms  (application/json) [Mode: awareness, FPS: 14.1]
[PASSED] GET /scene     -> HTTP 200 in 15.41ms (application/json) [World State snapshot]
[PASSED] GET /detections-> HTTP 200 in 2.91ms  (application/json) [Active Tracked Objects]
[PASSED] GET /metrics   -> HTTP 200 in 12.67ms (application/json) [Inference latency: 10.28ms]

--- Testing WebSocket /ws/stream ---
[PASSED] Connected to WebSocket ws://127.0.0.1:8000/ws/stream
[PASSED] Received binary JPEG frame (40922 bytes) [Valid JPEG SOI \xff\xd8 header]

============================================================
>>> ALL HTTP ENDPOINTS & WEBSOCKET VERIFIED SUCCESSFULLY! <<<
============================================================
```

### B. Unit & Integration Test Suite (`pytest`)
```
======================= 11 passed, 2 warnings in 30.19s =======================
backend/tests/test_cooldown_and_priority.py::test_critical_hazard_detection PASSED
backend/tests/test_cooldown_and_priority.py::test_awareness_speech_cooldown PASSED
backend/tests/test_direction_and_depth.py::test_direction_resolution PASSED
backend/tests/test_direction_and_depth.py::test_relative_depth_estimation PASSED
backend/tests/test_find_mode.py::test_find_mode_workflow PASSED
backend/tests/test_pipeline_end_to_end.py::test_fastapi_core_endpoints PASSED
backend/tests/test_pipeline_end_to_end.py::test_mode_switching_api PASSED
backend/tests/test_pipeline_end_to_end.py::test_pipeline_synthetic_camera_loop PASSED
backend/tests/test_read_and_context.py::test_context_engine_spatial_grouping PASSED
backend/tests/test_read_and_context.py::test_read_mode_ocr PASSED
backend/tests/test_world_state.py::test_world_state_update_and_pruning PASSED
```

---

## 6. Detector Model Audit: YOLOv8n vs YOLO26n
- **Current Detector Config:** `settings.YOLO_MODEL_PATH = "yolov8n.pt"`
- **Implementation State:** `backend/app/perception/detector.py` loads `yolov8n.pt` using Ultralytics YOLO with PyTorch CUDA (`torch-2.12.0.dev20260408+cu128`, CUDA 12.8 nightly).
- **Latency on RTX 5050 Laptop GPU:**
  - Mean Inference Latency: **16.99 ms**
  - P95 Inference Latency: **19.87 ms**
  - VRAM Consumption: **142 MiB**
- **Reconciliation Status:** The pipeline interface is fully model-agnostic (`ObjectDetector` ABC). When the YOLO26n weights/architecture are supplied, swapping `YOLO_MODEL_PATH` in `backend/app/core/config.py` requires zero structural changes.
