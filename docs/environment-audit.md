# VisionMate v2 - Environment & Hardware Audit

**Audit Date:** September 17, 2026  
**Audited By:** Principal System Architect  
**Target Project:** VisionMate v2 (Real-Time Multimodal Assistive AI Platform)  
**Standard Compliance:** Strict separation of `[MEASURED]`, `[ESTIMATED]`, and `[TARGET]`.

---

## 1. Executive Summary & Hardware State

The host development machine is an **Acer Nitro ANV15-52 Gaming Laptop** featuring a 10-core/16-thread Intel Core i7 CPU and an **NVIDIA GeForce RTX 5050 Laptop GPU** (Blackwell generation, Compute Capability 12.0) with **8,151 MiB (~8.0 GB GDDR6) VRAM**.

### Hardware Verification Status
- **[MEASURED] OS:** Microsoft Windows 11 Home Single Language (64-bit, Build 26200)
- **[MEASURED] CPU:** 13th Gen Intel(R) Core(TM) i7-13620H (10 Cores: 6 P-Cores + 4 E-Cores, 16 Threads, Base 2.40 GHz)
- **[MEASURED] GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (Driver Version: 595.97, CUDA 13.2 support, Compute Capability 12.0)
- **[MEASURED] Dedicated VRAM:** 8,151 MiB Total, 7,910 MiB Free at idle
- **[MEASURED] System RAM:** 16,072 MB Total Physical Memory (~2,088 MB Free under active Windows desktop workload; 16,457 MB Total Visible)
- **[MEASURED] Storage (C:):** 237.7 GB Total (203.3 GB Used / 48.35 GB Free NVMe SSD)
- **[MEASURED] Storage (D:):** 237.7 GB Total (182.5 GB Used / 55.22 GB Free NVMe SSD)
- **[MEASURED] Audio Devices:** Intel Smart Sound Digital Microphone Array, Realtek Audio output, NVIDIA High Definition Audio
- **[MEASURED] Native Windows TTS Voices:** SAPI5 Desktop (`Microsoft David`, `Microsoft Zira`), OneCore (`DavidM`, `MarkM`, `ZiraM`)
- **[MEASURED] Integrated Camera:** ACER HD User Facing Webcam (Present & Active)
- **[MEASURED] Network Interfaces:** MediaTek Wi-Fi 6 MT7920 Wireless LAN (`10.10.10.215`), Killer E2600 Gigabit Ethernet (Disconnected), Bluetooth PAN
- **[MEASURED] Python Environment:** Python 3.11.9 64-bit in dedicated virtual environment (`c:\Users\Srushti Gole\VisionMate-v2\.venv\Scripts\python.exe`)
- **[MEASURED] PyTorch & CUDA:** PyTorch 2.12.0.dev20260408+cu128 with native CUDA 12.8 runtime & sm_120 (Blackwell) support
- **[MEASURED] Package Manager / PyLauncher:** `winget.exe` available, `py.exe` available
- **[MEASURED] Fast Perception Latency:** Mean 16.99 ms / P95 19.87 ms (at 640x480 resolution, 58.9 FPS max throughput)
- **[MEASURED] Peak Detector VRAM:** 142.0 MiB (<2% of 8.0 GB VRAM; 8,008 MiB free headroom)

---

## 2. Capability & Resource Analysis

### 2.1 GPU Acceleration & VRAM Envelope
- **[MEASURED]** Dedicated VRAM is strictly **8.0 GB (8,151 MiB)**. 
- **[ESTIMATED]** Loading multiple large deep models simultaneously (e.g. YOLO + Depth + VLM + Whisper + OCR) in full precision will exceed 8 GB VRAM, causing CUDA out-of-memory (OOM) or fallback to slow shared system memory.
- **Architectural Requirement:** A **Model Lifecycle & Resource Manager** is mandatory to enforce a dynamic scheduling strategy (e.g., loading the heavy VLM on-demand or unloading non-critical weights when fast perception requires priority). Concurrent model placement must be explicitly benchmarked on this exact RTX 5050 before finalizing runtime residency.

### 2.2 Python & Toolchain Compatibility
- **[MEASURED]** System Python is version **3.14.7**.
- **[ESTIMATED]** Standard pre-compiled PyTorch CUDA wheels, TorchVision, ONNX Runtime GPU, and Fast-Whisper wheels have verified binary support on Python **3.11** or **3.12**.
- **Architectural Requirement:** VisionMate v2 will establish an isolated virtual environment (`.venv`) using Python 3.11/3.12 (managed via `winget` / `py` launcher) to guarantee flawless binary wheel compatibility and CUDA runtime stability.

### 2.3 Storage & Cache Management
- **[MEASURED]** Combined free NVMe storage is **~103.5 GB** (C: 48.3 GB, D: 55.2 GB).
- **Architectural Requirement:** Model weights, caches, and test fixtures must be budgeted tightly. Unquantized FP32 weights (>15 GB) are prohibited; 4-bit/8-bit quantized models and compact checkpoints (totaling < 12 GB on disk) will be used.

### 2.4 Audio & Speech Subsystem
- **[MEASURED]** Built-in Intel Smart Sound dual-mic array and native SAPI5 / OneCore synthesis are verified and operational without third-party network dependencies.
- **[TARGET]** Speech input (ASR) and speech output (TTS) must execute in isolated asynchronous threads/processes so that audio I/O never blocks the camera frame ingestion or real-time perception loops.

### 2.5 Camera & ESP32-CAM Connectivity
- **[MEASURED]** MediaTek Wi-Fi 6 adapter is operational on local LAN `10.10.10.215`. Built-in Acer HD Webcam is present.
- **[ESTIMATED]** ESP32-CAM can stream MJPEG over HTTP (`http://<esp32-ip>:81/stream`) on the same subnet, via ESP32 Direct AP (`192.168.4.1`), or via high-speed USB-UART (COM ports).
- **Architectural Requirement:** The `CameraProvider` interface must support seamless dynamic fallback to local webcam or synthetic test streams if network jitter or packet loss disrupts the wireless feed.

---

## 3. Measured vs. Target Resource Matrix

| Resource Category | Current System State [MEASURED] | MVP Target Allocation [TARGET] | Status / Action |
| :--- | :--- | :--- | :--- |
| **GPU VRAM** | 8,151 MiB Total, 7,910 MiB Free | $\le$ 6.2 GB Peak Allocated | Model Lifecycle Manager required to prevent OOM |
| **System RAM** | 16,072 MB Total, ~2,088 MB Free | $\le$ 3.5 GB App Footprint | Zero-copy ring buffers & strict memory pooling |
| **Storage Footprint**| ~103.5 GB Free across C: & D: | $\le$ 15 GB for all models/data | Quantized models (Q4/INT8/FP16) |
| **Fast Loop Latency**| Not yet benchmarked | [TARGET] $\le$ 45 ms per frame | Benchmarking suite to measure actual latency |
| **Urgent Alert Loop** | Not yet benchmarked | [TARGET] $\le$ 100 ms trigger-to-audio | Preemption arbitration & benchmark harness |
| **VLM On-Demand** | Not yet benchmarked | [TARGET] $\le$ 600 ms response | On-demand async worker (non-blocking) |
| **OCR Ingestion** | Not yet benchmarked | [TARGET] On-demand / Triggered only | PP-OCRv5/PaddleOCR 3.x on ROI / Read Mode |
