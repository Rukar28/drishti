"""
Master Benchmark Suite Runner for VisionMate v2
Executes all Phase 1 baseline benchmarks and produces an aggregated summary.
"""

import sys
import os
import json
import subprocess
import time
from datetime import datetime

def run_suite():
    benchmarks = [
        ("CUDA Environment Validation", "bench_cuda_environment.py"),
        ("GPU VRAM Residency Baseline", "bench_vram_residency.py"),
        ("Reflex Perception Latency", "bench_reflex_latency.py"),
        ("Resource Utilization Stress Test", "bench_resource_utilization.py")
    ]

    base_dir = os.path.dirname(__file__)
    python_exe = sys.executable

    print("\n" + "="*70)
    print(f"VISIONMATE v2 - PHASE 1 BENCHMARK SUITE")
    print(f"Timestamp:   {datetime.now().isoformat()}")
    print(f"Interpreter: {python_exe}")
    print("="*70 + "\n")

    overall_results = {}

    for name, script in benchmarks:
        script_path = os.path.join(base_dir, script)
        print(f"\n[RUNNING BENCHMARK]: {name} ({script})...")
        t0 = time.perf_counter()
        proc = subprocess.run([python_exe, script_path], capture_output=False)
        dt = time.perf_counter() - t0
        status = "PASSED" if proc.returncode == 0 else "FAILED"
        overall_results[name] = {"script": script, "status": status, "duration_sec": round(dt, 2)}
        print(f"--> Result: {status} in {dt:.2f}s\n")
        if proc.returncode != 0:
            print(f"CRITICAL FAILURE in {script}. Halting suite.")
            sys.exit(1)

    print("\n" + "="*70)
    print("ALL PHASE 1 BASELINE BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    run_suite()
