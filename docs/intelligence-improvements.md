# VisionMate v2 - Backend Intelligence, Guidance & Reliability Improvements

## 1. Problem Diagnosis & Engineering Solutions

### Problem 1: Object Detection Instabilities & False Positives
- **Cause:** Raw 1-frame detections at low confidence thresholds (e.g. spurious `tie`, `remote`, `fork` detections caused by clothing wrinkles or shadows) were immediately converted into World State active objects.
- **Solution:**
  - Implemented **3-State Hierarchy**: `RAW_DETECTION` → `STABLE_OBJECT` → `EVENT`.
  - In `SpatialTracker`, tracks require at least 3-4 consecutive frames ($\ge 0.25$s) of consistent IoU overlap before becoming `is_stable = True`.
  - Added **EMA Smoothing** ($\alpha = 0.60$) on bounding box coordinates and centroids to eliminate frame-to-frame pixel jitter.
  - Added noise-class confidence calibration in `YOLOObjectDetector`.

### Problem 2: OCR Reliability & On-Demand Execution
- **Cause:** OCR was previously not properly isolated from the pipeline lifecycle, and error handling for blurry frames was missing.
- **Solution:**
  - **Default: `OCR = OFF`** in Guidance Mode. Consumes 0% GPU/CPU during regular wearer operation.
  - Activated strictly on-demand when the user commands "Read this" / POST `/api/v1/read`.
  - Preprocessing with **CLAHE** contrast equalization and Laplacian sharpness validation.
  - If text is blurry or unreadable, speaks: `"I couldn't read the text clearly. Please hold the text steady."` rather than producing raw noise.
  - Automatically turns OFF upon reading completion and resumes Guidance Mode.

### Problem 3: Continuous "Caution" Spam on Ordinary Objects
- **Cause:** Proximity heuristics previously conflated optical scale with hazard severity, treating ordinary stationary objects (chairs, tables, laptops, bottles, standing persons) as approaching collision hazards.
- **Solution:**
  - **Completely separated Object Detection from Hazard Detection**: Object detection answers *"What is there?"*; Hazard Engine answers *"Does it threaten the wearer right now?"*.
  - Introduced **Forward Walking Corridor (PathZone)**:
    - `INSIDE_PATH` ($cx \in [0.34, 0.66]$)
    - `NEAR_PATH` ($cx \in [0.20, 0.34) \cup (0.66, 0.80]$)
    - `OUTSIDE_PATH` ($cx < 0.20 \cup cx > 0.80$)
  - Static objects (chairs, desks, bottles, laptops, standing persons) receive `HazardSeverity.NONE` and `hazard_score = 0.0`.
  - Hazards are strictly limited to:
    1. Structural fall risks: `stairs`, `dropoff`, `step`, `hole` (`EMERGENCY`).
    2. Fast approaching vehicles/bicycles crossing or entering path (`WARNING` / `EMERGENCY`).
    3. Dynamic objects rapidly looming inside `INSIDE_PATH` with verified approaching motion vectors (`WARNING`).

### Problem 4: Speech Repeating Without Stopping
- **Cause:** Continuous frame-level speech synthesis lacked semantic identity and silence hysteresis.
- **Solution:**
  - Implemented **Semantic Event Identity & State Machine**:
    - States: `DETECTED` → `CONFIRMED` → `ANNOUNCED` → `COOLDOWN` → `MONITORED` → `CHANGED` → `CLEARED`.
  - **Post-Speech Silence Enforcement**:
    - Once VisionMate speaks a scene summary (e.g. *"You are facing a desk with a laptop and bottle ahead. A person is on your left."*), it enters **SILENCE**.
    - It remains quiet until a material change occurs (e.g. person crosses path), a hazard escalates, or a new distinct event occurs.

---

## 2. Regression Test Matrix (22 Tests Passing)

| Test ID | Scenario | Result |
| :--- | :--- | :--- |
| **TEST 1** | Normal chair detection must NOT create "Caution" | **PASSED** |
| **TEST 2** | Normal bottle detection must NOT create "Caution" | **PASSED** |
| **TEST 3** | Normal laptop detection must NOT create "Caution" | **PASSED** |
| **TEST 4** | Repeated detection of the same object must NOT repeatedly speak | **PASSED** |
| **TEST 5** | After a spoken event, speech must enter cooldown / silence | **PASSED** |
| **TEST 6** | A materially changed event may trigger a new announcement | **PASSED** |
| **TEST 7** | OCR must remain OFF in normal Guidance Mode | **PASSED** |
| **TEST 8** | OCR activates only after Read command | **PASSED** |
| **TEST 9** | OCR automatically turns OFF after reading | **PASSED** |
| **TEST 10** | Poor OCR result must not be spoken as confident text | **PASSED** |
| **TEST 11** | VLM must not block Guidance Mode | **PASSED** |
| **TEST 12** | Find Mode must reuse World State and tracking | **PASSED** |
| **TEST 13** | Emergency speech can interrupt normal speech | **PASSED** |
| **TEST 14** | Emergency speech does not repeat indefinitely without a changed event | **PASSED** |
| **TESTS 15-22** | Core API endpoints, direction resolution, relative depth, world state pruning | **PASSED** |

---

## 3. Measured System Performance

- **Fast Perception Latency:** ~9.96 ms on NVIDIA RTX 5050 Laptop GPU (`torch-2.12.0.dev20260408+cu128`, CUDA 12.8 nightly).
- **Pipeline Frame Rate:** ~20–25 FPS stable.
- **VRAM Residency:** ~142 MiB (perception & tracking), leaving >7.8 GB free headroom for Qwen3-VL 4B VLM.
