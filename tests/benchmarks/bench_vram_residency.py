"""
Benchmark 2: GPU VRAM Baseline & Residency Benchmark
Measures actual VRAM allocations across lifecycle stages:
1. System/Windows Baseline
2. Python Process Baseline
3. CUDA Context Init Baseline
4. YOLO Model Load Memory
5. YOLO Inference Memory (Single & Batch)
6. Peak VRAM during continuous inference
7. Memory Growth / Leak Check (100 sequential inferences)
8. Model Unload / Cache Release Baseline
"""

import sys
import os
import json
import time
import subprocess
from datetime import datetime

def get_nvidia_smi_memory():
    """Query total, used, free memory directly from nvidia-smi in MiB."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        parts = [float(x.strip()) for x in out.split(",")]
        return {"total_mib": parts[0], "used_mib": parts[1], "free_mib": parts[2]}
    except Exception as e:
        return {"error": str(e)}

def run_vram_benchmark():
    results = {
        "benchmark": "GPU VRAM Residency & Lifecycle",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "interpreter": sys.executable,
        "measurements": {}
    }

    # Step 1: System Baseline
    sys_base = get_nvidia_smi_memory()
    results["measurements"]["1_system_baseline_nvidia_smi"] = sys_base

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    results["gpu_name"] = gpu_name
    results["total_vram_mb"] = round(total_vram_mb, 2)

    # Step 2: Python Process & CUDA Context Init
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    cuda_init_smi = get_nvidia_smi_memory()
    cuda_init_torch_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    cuda_init_torch_res = torch.cuda.memory_reserved(device) / (1024 * 1024)

    results["measurements"]["2_cuda_init"] = {
        "nvidia_smi_used_mib": cuda_init_smi.get("used_mib"),
        "torch_allocated_mb": round(cuda_init_torch_alloc, 2),
        "torch_reserved_mb": round(cuda_init_torch_res, 2),
        "cuda_context_overhead_mib": round(cuda_init_smi.get("used_mib", 0) - sys_base.get("used_mib", 0), 2)
    }

    # Step 3: Load YOLO Model
    from ultralytics import YOLO
    model_name = "yolov8n.pt"
    print(f"Loading {model_name} onto CUDA...")
    t0 = time.perf_counter()
    model = YOLO(model_name)
    model.to("cuda")
    load_time_sec = time.perf_counter() - t0

    model_loaded_smi = get_nvidia_smi_memory()
    model_loaded_torch_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    model_loaded_torch_res = torch.cuda.memory_reserved(device) / (1024 * 1024)

    results["measurements"]["3_yolo_model_loaded"] = {
        "model_name": model_name,
        "load_time_sec": round(load_time_sec, 3),
        "nvidia_smi_used_mib": model_loaded_smi.get("used_mib"),
        "torch_allocated_mb": round(model_loaded_torch_alloc, 2),
        "torch_reserved_mb": round(model_loaded_torch_res, 2),
        "model_weight_vram_mb": round(model_loaded_torch_alloc - cuda_init_torch_alloc, 2)
    }

    # Step 4: First Inference & Warmup
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _ = model(dummy_frame, verbose=False, half=True)
    torch.cuda.synchronize()

    first_infer_smi = get_nvidia_smi_memory()
    first_infer_torch_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    first_infer_torch_res = torch.cuda.memory_reserved(device) / (1024 * 1024)

    results["measurements"]["4_first_inference_warmup"] = {
        "nvidia_smi_used_mib": first_infer_smi.get("used_mib"),
        "torch_allocated_mb": round(first_infer_torch_alloc, 2),
        "torch_reserved_mb": round(first_infer_torch_res, 2)
    }

    # Step 5: Continuous Inference & Memory Leak Test (100 iterations)
    print("Running 100 consecutive inferences to test for memory leaks...")
    alloc_history = []
    res_history = []
    for i in range(100):
        _ = model(dummy_frame, verbose=False, half=True)
        if (i + 1) % 20 == 0:
            torch.cuda.synchronize()
            alloc_history.append(torch.cuda.memory_allocated(device) / (1024 * 1024))
            res_history.append(torch.cuda.memory_reserved(device) / (1024 * 1024))

    torch.cuda.synchronize()
    peak_vram_torch = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_res_torch = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    post_infer_smi = get_nvidia_smi_memory()

    # Memory growth delta between iteration 20 and iteration 100
    memory_growth_mb = alloc_history[-1] - alloc_history[0]

    results["measurements"]["5_continuous_inference_100_runs"] = {
        "alloc_history_checkpoints_mb": [round(x, 2) for x in alloc_history],
        "reserved_history_checkpoints_mb": [round(x, 2) for x in res_history],
        "memory_growth_mb": round(memory_growth_mb, 4),
        "memory_leak_detected": bool(abs(memory_growth_mb) > 0.5),
        "peak_torch_allocated_mb": round(peak_vram_torch, 2),
        "peak_torch_reserved_mb": round(peak_res_torch, 2),
        "nvidia_smi_used_mib": post_infer_smi.get("used_mib"),
        "vram_headroom_remaining_mib": round(total_vram_mb - post_infer_smi.get("used_mib", 0), 2)
    }

    # Step 6: Model Cleanup and Release
    print("Testing model cleanup and memory release...")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    cleanup_smi = get_nvidia_smi_memory()
    cleanup_torch_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    cleanup_torch_res = torch.cuda.memory_reserved(device) / (1024 * 1024)

    results["measurements"]["6_cleanup_and_release"] = {
        "nvidia_smi_used_mib": cleanup_smi.get("used_mib"),
        "torch_allocated_mb": round(cleanup_torch_alloc, 2),
        "torch_reserved_mb": round(cleanup_torch_res, 2),
        "memory_reclaimed_mb": round(post_infer_smi.get("used_mib", 0) - cleanup_smi.get("used_mib", 0), 2)
    }

    results["status"] = "PASSED"

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "vram_residency_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("GPU VRAM RESIDENCY & LIFECYCLE REPORT [MEASURED]")
    print("="*60)
    print(f"GPU:                     {gpu_name} ({total_vram_mb:.0f} MB Total)")
    print(f"System Baseline VRAM:    {sys_base.get('used_mib')} MiB")
    print(f"CUDA Init VRAM (SMI):    {cuda_init_smi.get('used_mib')} MiB (Context: +{results['measurements']['2_cuda_init']['cuda_context_overhead_mib']} MiB)")
    print(f"YOLOv8n Load Weight VRAM:{results['measurements']['3_yolo_model_loaded']['model_weight_vram_mb']} MB (Load time: {load_time_sec:.3f}s)")
    print(f"Peak VRAM during infer:  {post_infer_smi.get('used_mib')} MiB (Torch Alloc: {peak_vram_torch:.2f} MB)")
    print(f"Memory Growth (100 run): {memory_growth_mb:.4f} MB (Leak: {results['measurements']['5_continuous_inference_100_runs']['memory_leak_detected']})")
    print(f"VRAM Headroom Remaining: {results['measurements']['5_continuous_inference_100_runs']['vram_headroom_remaining_mib']} MiB")
    print(f"Reclaimed after Unload:  {results['measurements']['6_cleanup_and_release']['memory_reclaimed_mb']} MiB")
    print(f"Saved to:                {out_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_vram_benchmark()
