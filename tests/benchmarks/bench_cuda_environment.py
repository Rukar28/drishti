"""
Benchmark 1: CUDA Environment Validation
Tests PyTorch version, CUDA runtime availability, GPU properties, memory baseline,
and performs tensor allocation and matrix multiplication verification.
"""

import sys
import os
import json
import time
from datetime import datetime

def run_cuda_benchmark():
    results = {
        "benchmark": "CUDA Environment Validation",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "interpreter": sys.executable,
        "status": "FAILED",
        "details": {}
    }

    try:
        import torch
        results["details"]["pytorch_version"] = torch.__version__
        results["details"]["cuda_available"] = torch.cuda.is_available()

        if not torch.cuda.is_available():
            results["error"] = "CUDA is not available to PyTorch. Check NVIDIA driver and PyTorch build."
            print(json.dumps(results, indent=2))
            sys.exit(1)

        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        compute_cap = torch.cuda.get_device_capability(current_device)
        total_memory_bytes = torch.cuda.get_device_properties(current_device).total_memory
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"

        results["details"]["device_count"] = device_count
        results["details"]["current_device_id"] = current_device
        results["details"]["gpu_name"] = gpu_name
        results["details"]["compute_capability"] = f"{compute_cap[0]}.{compute_cap[1]}"
        results["details"]["total_vram_mb"] = round(total_memory_mb, 2)
        results["details"]["cuda_version"] = cuda_version
        results["details"]["cudnn_version"] = str(cudnn_version)

        # Baseline memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_alloc_before = torch.cuda.memory_allocated(current_device) / (1024 * 1024)
        mem_res_before = torch.cuda.memory_reserved(current_device) / (1024 * 1024)

        # MatMul benchmark: 4096 x 4096 FP32 and FP16
        print("Running FP32 & FP16 CUDA Tensor MatMul tests...")
        N = 4096
        # Warmup
        a = torch.randn(N, N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, N, device="cuda", dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Timed FP32
        start_fp32 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed_fp32 = (time.perf_counter() - start_fp32) / iters * 1000.0  # ms per matmul
        tflops_fp32 = (2 * (N ** 3)) / (elapsed_fp32 / 1000.0) / 1e12

        # Timed FP16
        a_fp16 = a.to(dtype=torch.float16)
        b_fp16 = b.to(dtype=torch.float16)
        start_fp16 = time.perf_counter()
        for _ in range(iters):
            c_fp16 = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        elapsed_fp16 = (time.perf_counter() - start_fp16) / iters * 1000.0
        tflops_fp16 = (2 * (N ** 3)) / (elapsed_fp16 / 1000.0) / 1e12

        mem_alloc_after = torch.cuda.memory_allocated(current_device) / (1024 * 1024)
        mem_res_after = torch.cuda.memory_reserved(current_device) / (1024 * 1024)
        peak_mem = torch.cuda.max_memory_allocated(current_device) / (1024 * 1024)

        results["details"]["matmul_4096_fp32_ms"] = round(elapsed_fp32, 3)
        results["details"]["matmul_4096_fp32_tflops"] = round(tflops_fp32, 2)
        results["details"]["matmul_4096_fp16_ms"] = round(elapsed_fp16, 3)
        results["details"]["matmul_4096_fp16_tflops"] = round(tflops_fp16, 2)
        results["details"]["vram_allocated_mb"] = round(mem_alloc_after, 2)
        results["details"]["vram_reserved_mb"] = round(mem_res_after, 2)
        results["details"]["vram_peak_mb"] = round(peak_mem, 2)

        del a, b, c, a_fp16, b_fp16, c_fp16
        torch.cuda.empty_cache()

        results["status"] = "PASSED"

    except Exception as e:
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "cuda_environment_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("CUDA ENVIRONMENT VALIDATION REPORT [MEASURED]")
    print("="*60)
    print(f"Status:             {results['status']}")
    if results["status"] == "PASSED":
        d = results["details"]
        print(f"PyTorch Version:    {d['pytorch_version']}")
        print(f"CUDA Runtime:       CUDA {d['cuda_version']} (cuDNN {d['cudnn_version']})")
        print(f"GPU Device:         {d['gpu_name']} (Compute Capability {d['compute_capability']})")
        print(f"Total GPU VRAM:     {d['total_vram_mb']} MB (~{d['total_vram_mb']/1024:.2f} GB)")
        print(f"FP32 MatMul (4K):   {d['matmul_4096_fp32_ms']} ms ({d['matmul_4096_fp32_tflops']} TFLOPS)")
        print(f"FP16 MatMul (4K):   {d['matmul_4096_fp16_ms']} ms ({d['matmul_4096_fp16_tflops']} TFLOPS)")
        print(f"Peak VRAM used:     {d['vram_peak_mb']} MB")
        print(f"Saved to:           {out_file}")
    else:
        print(f"Error: {results.get('error')}")
    print("="*60 + "\n")

    if results["status"] != "PASSED":
        sys.exit(1)

if __name__ == "__main__":
    run_cuda_benchmark()
