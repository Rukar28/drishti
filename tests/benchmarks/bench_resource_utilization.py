"""
Benchmark 4: CPU, GPU, RAM Resource Utilization & Bottleneck Analyzer
Runs a continuous 15-second perception loop while sampling system metrics at 10 Hz:
- Process CPU % and System Total CPU %
- Process RAM Working Set (MB) and System RAM Free (MB)
- GPU Utilization % and GPU VRAM Usage (MiB)
- Pipeline Frame Rate and Frame Processing Latency
Saves detailed time-series telemetry and summary statistics.
"""

import sys
import os
import json
import time
import subprocess
import threading
import numpy as np
from datetime import datetime

def run_resource_benchmark():
    import psutil
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    pid = os.getpid()
    proc = psutil.Process(pid)

    results = {
        "benchmark": "Resource Utilization & Bottleneck Analysis",
        "timestamp": datetime.now().isoformat(),
        "gpu_name": gpu_name,
        "pid": pid,
        "telemetry_samples": [],
        "summary": {}
    }

    print(f"Initializing YOLO model on {gpu_name} for resource stress test...")
    model = YOLO("yolov8n.pt")
    model.to("cuda")

    # Warmup
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model(dummy_frame, imgsz=(480, 640), half=True, verbose=False, device="cuda")
    torch.cuda.synchronize()

    # Telemetry sampler thread
    stop_event = threading.Event()
    telemetry = []

    def sampler():
        while not stop_event.is_set():
            t_now = time.time()
            cpu_proc = proc.cpu_percent(interval=None)
            cpu_sys = psutil.cpu_percent(interval=None)
            mem_info = proc.memory_info()
            ram_proc_mb = mem_info.rss / (1024 * 1024)
            ram_sys_free_mb = psutil.virtual_memory().available / (1024 * 1024)

            # Query GPU
            try:
                out = subprocess.check_output(
                    "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
                    shell=True, text=True
                ).strip()
                gpu_util, gpu_mem = [float(x.strip()) for x in out.split(",")]
            except Exception:
                gpu_util, gpu_mem = -1.0, -1.0

            telemetry.append({
                "time_offset_sec": 0, # calculated later
                "timestamp": t_now,
                "process_cpu_pct": cpu_proc,
                "system_cpu_pct": cpu_sys,
                "process_ram_mb": round(ram_proc_mb, 2),
                "system_ram_free_mb": round(ram_sys_free_mb, 2),
                "gpu_utilization_pct": gpu_util,
                "gpu_vram_used_mib": gpu_mem
            })
            time.sleep(0.1)  # 10 Hz sampling

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    # Run continuous perception loop for 15 seconds
    duration_sec = 15.0
    start_time = time.perf_counter()
    frame_latencies = []
    frames_processed = 0

    print(f"Running continuous perception workload for {duration_sec} seconds...")
    while (time.perf_counter() - start_time) < duration_sec:
        t0 = time.perf_counter()
        _ = model(dummy_frame, imgsz=(480, 640), half=True, verbose=False, device="cuda")
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        frame_latencies.append(dt_ms)
        frames_processed += 1

    stop_event.set()
    sampler_thread.join(timeout=2.0)

    t_start_ref = telemetry[0]["timestamp"] if telemetry else time.time()
    for item in telemetry:
        item["time_offset_sec"] = round(item["timestamp"] - t_start_ref, 2)
        del item["timestamp"]

    results["telemetry_samples"] = telemetry

    # Compute summary metrics
    lat = np.array(frame_latencies)
    proc_cpus = [s["process_cpu_pct"] for s in telemetry if s["process_cpu_pct"] > 0]
    sys_cpus = [s["system_cpu_pct"] for s in telemetry]
    proc_rams = [s["process_ram_mb"] for s in telemetry]
    gpu_utils = [s["gpu_utilization_pct"] for s in telemetry if s["gpu_utilization_pct"] >= 0]
    gpu_mems = [s["gpu_vram_used_mib"] for s in telemetry if s["gpu_vram_used_mib"] >= 0]

    results["summary"] = {
        "duration_seconds": round(duration_sec, 2),
        "total_frames_processed": frames_processed,
        "effective_fps": round(frames_processed / duration_sec, 2),
        "latency_stats_ms": {
            "mean": round(float(np.mean(lat)), 2),
            "median": round(float(np.median(lat)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "min": round(float(np.min(lat)), 2),
            "max": round(float(np.max(lat)), 2)
        },
        "process_cpu_pct": {
            "mean": round(float(np.mean(proc_cpus)), 2) if proc_cpus else 0.0,
            "max": round(float(np.max(proc_cpus)), 2) if proc_cpus else 0.0
        },
        "system_cpu_pct": {
            "mean": round(float(np.mean(sys_cpus)), 2) if sys_cpus else 0.0,
            "max": round(float(np.max(sys_cpus)), 2) if sys_cpus else 0.0
        },
        "process_ram_mb": {
            "mean": round(float(np.mean(proc_rams)), 2) if proc_rams else 0.0,
            "max": round(float(np.max(proc_rams)), 2) if proc_rams else 0.0
        },
        "gpu_utilization_pct": {
            "mean": round(float(np.mean(gpu_utils)), 2) if gpu_utils else 0.0,
            "max": round(float(np.max(gpu_utils)), 2) if gpu_utils else 0.0
        },
        "gpu_vram_used_mib": {
            "mean": round(float(np.mean(gpu_mems)), 2) if gpu_mems else 0.0,
            "peak": round(float(np.max(gpu_mems)), 2) if gpu_mems else 0.0
        },
        "primary_bottleneck_analysis": (
            "GPU Compute Bound" if np.mean(gpu_utils) > 85 else
            "CPU Bound" if (proc_cpus and np.mean(proc_cpus) > 80) else
            "Extremely Low Resource Load (Balanced High Throughput)"
        )
    }

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "resource_utilization_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print("\n" + "="*60)
    print("RESOURCE UTILIZATION & BOTTLENECK REPORT [MEASURED]")
    print("="*60)
    print(f"Frames Processed:       {s['total_frames_processed']} in {s['duration_seconds']}s")
    print(f"Effective Throughput:   {s['effective_fps']} FPS")
    print(f"Per-Frame Latency:      Mean={s['latency_stats_ms']['mean']}ms | P95={s['latency_stats_ms']['p95']}ms")
    print(f"Process CPU Load:       Mean={s['process_cpu_pct']['mean']}% (Max={s['process_cpu_pct']['max']}%)")
    print(f"System CPU Load:        Mean={s['system_cpu_pct']['mean']}% (Max={s['system_cpu_pct']['max']}%)")
    print(f"Process RAM Working Set:Mean={s['process_ram_mb']['mean']} MB (Peak={s['process_ram_mb']['max']} MB)")
    print(f"GPU Utilization:        Mean={s['gpu_utilization_pct']['mean']}% (Peak={s['gpu_utilization_pct']['max']}%)")
    print(f"GPU VRAM Usage:         Mean={s['gpu_vram_used_mib']['mean']} MiB (Peak={s['gpu_vram_used_mib']['peak']} MiB)")
    print(f"Bottleneck Diagnosis:   {s['primary_bottleneck_analysis']}")
    print(f"Saved to:               {out_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_resource_benchmark()
