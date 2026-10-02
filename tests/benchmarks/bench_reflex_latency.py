"""
Benchmark 3: Fast Reflex Perception Latency Benchmark
Measures granular breakdown of detection latency:
1. Frame Preprocessing (resize, normalize, tensor to CUDA)
2. Model Inference (FP16 vs FP32 on RTX 5050 GPU)
3. Postprocessing (NMS & bounding box decode)
4. End-to-End Total Detector Latency
Tests across resolutions: 640x480, 640x640, 320x320.
Computes Mean, Median, P95, P99, Min, Max over multiple iterations.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

def run_reflex_latency_benchmark():
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)

    results = {
        "benchmark": "Fast Reflex Latency Benchmark",
        "timestamp": datetime.now().isoformat(),
        "gpu_name": gpu_name,
        "configurations": []
    }

    # Load YOLOv8n model
    model_name = "yolov8n.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    model.to("cuda")

    # Configurations to test: (resolution_h_w, half_precision, name)
    configs = [
        ((480, 640), True, "640x480_FP16"),
        ((480, 640), False, "640x480_FP32"),
        ((640, 640), True, "640x640_FP16"),
        ((320, 320), True, "320x320_FP16"),
    ]

    total_iters = 120
    warmup_iters = 20
    bench_iters = total_iters - warmup_iters

    print(f"Starting Reflex Latency Suite ({total_iters} runs per config, {warmup_iters} warmup)...")

    for (h, w), is_half, cfg_name in configs:
        print(f"\n--- Testing Configuration: {cfg_name} (Resolution {w}x{h}, FP16={is_half}) ---")
        
        # Create synthetic realistic frame with mock objects
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Add random rectangular textures to simulate real scene complexity
        for _ in range(8):
            x1, y1 = np.random.randint(0, w-50), np.random.randint(0, h-50)
            x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            frame[y1:y2, x1:x2] = color

        preprocess_times = []
        inference_times = []
        postprocess_times = []
        total_times = []

        torch.cuda.reset_peak_memory_stats()

        for i in range(total_iters):
            t_start = time.perf_counter()

            # Execute model run with internal timing profiling
            preds = model(frame, imgsz=(h, w), half=is_half, verbose=False, device="cuda")
            torch.cuda.synchronize()

            t_end = time.perf_counter()

            if i >= warmup_iters:
                # Extract Ultralytics speed dict (in milliseconds)
                speed = preds[0].speed
                p_pre = speed.get("preprocess", 0.0)
                p_inf = speed.get("inference", 0.0)
                p_post = speed.get("postprocess", 0.0)
                p_tot = (t_end - t_start) * 1000.0

                preprocess_times.append(p_pre)
                inference_times.append(p_inf)
                postprocess_times.append(p_post)
                total_times.append(p_tot)

        # Calculate statistics
        def calc_stats(arr):
            a = np.array(arr)
            return {
                "mean_ms": round(float(np.mean(a)), 3),
                "median_ms": round(float(np.median(a)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
                "p99_ms": round(float(np.percentile(a, 99)), 3),
                "min_ms": round(float(np.min(a)), 3),
                "max_ms": round(float(np.max(a)), 3),
                "std_ms": round(float(np.std(a)), 3)
            }

        cfg_result = {
            "config_name": cfg_name,
            "resolution": f"{w}x{h}",
            "precision": "FP16" if is_half else "FP32",
            "iterations_sampled": bench_iters,
            "preprocess_stats": calc_stats(preprocess_times),
            "inference_stats": calc_stats(inference_times),
            "postprocess_stats": calc_stats(postprocess_times),
            "total_detector_stats": calc_stats(total_times),
            "max_achievable_fps": round(1000.0 / float(np.mean(total_times)), 1),
            "vram_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        }

        results["configurations"].append(cfg_result)

        tot = cfg_result["total_detector_stats"]
        inf = cfg_result["inference_stats"]
        print(f"  [MEASURED] Total Latency:  Mean={tot['mean_ms']}ms | P95={tot['p95_ms']}ms | P99={tot['p99_ms']}ms (Max FPS={cfg_result['max_achievable_fps']})")
        print(f"  [MEASURED] GPU Inference:  Mean={inf['mean_ms']}ms | P95={inf['p95_ms']}ms")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "reflex_latency_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("REFLEX LATENCY BENCHMARK COMPLETED")
    print(f"Results saved to: {out_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_reflex_latency_benchmark()
