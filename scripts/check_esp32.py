"""
VisionMate v2 — ESP32-CAM Snapshot & Stream Benchmark Tool
==========================================================
Tests the primary AI camera input (/capture at ~10 FPS) and validates
the MJPEG preview stream (:81/stream) against the physical ESP32-CAM.

Usage:
    python scripts/check_esp32.py [--ip 192.168.4.1] [--duration 10]
"""

import argparse
import socket
import time
import sys
import json
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, Any, List

try:
    import cv2
    import numpy as np
    _cv2_available = True
except ImportError:
    _cv2_available = False


# ANSI colours (safe for Windows terminals)
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):   return f"  {GREEN}[PASS]{RESET}  {msg}"
def fail(msg): return f"  {RED}[FAIL]{RESET}  {msg}"
def warn(msg): return f"  {YELLOW}[WARN]{RESET}  {msg}"
def info(msg): return f"  {CYAN}[INFO]{RESET}  {msg}"
def hdr(msg):  return f"\n{YELLOW}>> {msg}{RESET}"


def check_tcp_reachable(ip: str, port: int, timeout: float = 3.0) -> bool:
    """TCP connect check — is the port open?"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def fetch_http_json(url: str, timeout: float = 4.0) -> Tuple[bool, Dict[str, Any], str]:
    """Fetch and parse JSON endpoint."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VisionMate-Diag"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, data, ""
    except Exception as e:
        return False, {}, str(e)


def benchmark_snapshot_loop(
    capture_url: str,
    target_fps: float = 10.0,
    test_duration_sec: float = 10.0,
    timeout: float = 3.0,
) -> dict:
    """
    Benchmarks continuous /capture snapshot acquisition at target 10 FPS (100 ms interval):
    - Measures exact request latency per frame
    - Measures JPEG decode latency
    - Computes average and p95 latency
    - Measures actual acquired FPS
    - Validates image dimensions and byte integrity
    """
    target_interval = 1.0 / target_fps
    req_latencies: List[float] = []
    dec_latencies: List[float] = []
    total_latencies: List[float] = []
    frame_sizes: List[int] = []

    results = {
        "snapshots_requested": 0,
        "snapshots_received":  0,
        "decode_success":      0,
        "decode_failures":     0,
        "actual_fps":          0.0,
        "avg_req_ms":          0.0,
        "p95_req_ms":          0.0,
        "min_req_ms":          0.0,
        "max_req_ms":          0.0,
        "avg_dec_ms":          0.0,
        "avg_total_ms":        0.0,
        "p95_total_ms":        0.0,
        "avg_jpeg_kb":         0.0,
        "width":               0,
        "height":              0,
        "error":               None,
    }

    t_start = time.perf_counter()
    req = urllib.request.Request(
        capture_url,
        headers={
            "User-Agent": "VisionMate-SnapshotBenchmark",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

    while (time.perf_counter() - t_start) < test_duration_sec:
        results["snapshots_requested"] += 1
        t0 = time.perf_counter()

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            t1 = time.perf_counter()

            req_ms = (t1 - t0) * 1000.0
            req_latencies.append(req_ms)
            frame_sizes.append(len(data))
            results["snapshots_received"] += 1

            if len(data) > 100 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9":
                if _cv2_available:
                    t_dec_start = time.perf_counter()
                    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    dec_ms = (time.perf_counter() - t_dec_start) * 1000.0
                    dec_latencies.append(dec_ms)

                    if frame is not None:
                        results["decode_success"] += 1
                        if results["height"] == 0:
                            results["height"], results["width"] = frame.shape[:2]
                    else:
                        results["decode_failures"] += 1
                else:
                    results["decode_success"] += 1
            else:
                results["decode_failures"] += 1

            total_ms = (time.perf_counter() - t0) * 1000.0
            total_latencies.append(total_ms)

        except Exception as e:
            results["error"] = str(e)
            results["decode_failures"] += 1

        # Maintain 100 ms target interval
        elapsed = time.perf_counter() - t0
        sleep_sec = target_interval - elapsed
        if sleep_sec > 0.001:
            time.sleep(sleep_sec)

    total_test_time = time.perf_counter() - t_start
    if total_test_time > 0 and results["snapshots_received"] > 0:
        results["actual_fps"] = round(results["snapshots_received"] / total_test_time, 1)

    if req_latencies:
        results["avg_req_ms"] = round(sum(req_latencies) / len(req_latencies), 1)
        results["min_req_ms"] = round(min(req_latencies), 1)
        results["max_req_ms"] = round(max(req_latencies), 1)
        s_req = sorted(req_latencies)
        results["p95_req_ms"] = round(s_req[min(int(len(s_req) * 0.95), len(s_req) - 1)], 1)

    if dec_latencies:
        results["avg_dec_ms"] = round(sum(dec_latencies) / len(dec_latencies), 1)

    if total_latencies:
        results["avg_total_ms"] = round(sum(total_latencies) / len(total_latencies), 1)
        s_tot = sorted(total_latencies)
        results["p95_total_ms"] = round(s_tot[min(int(len(s_tot) * 0.95), len(s_tot) - 1)], 1)

    if frame_sizes:
        results["avg_jpeg_kb"] = round((sum(frame_sizes) / len(frame_sizes)) / 1024.0, 1)

    return results


def check_preview_stream(stream_url: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """Verify that :81/stream is accessible for browser preview without locking."""
    try:
        req = urllib.request.Request(stream_url, headers={"User-Agent": "VisionMate-Diag"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        chunk = resp.read(512)
        resp.close()
        if len(chunk) > 0 and b"--frame" in chunk:
            return True, "MJPEG Preview stream active and responsive"
        return True, "Stream reachable"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="VisionMate ESP32-CAM Snapshot & Stream Diagnostic")
    parser.add_argument("--ip",       default="192.168.4.1", help="ESP32 IP (default: 192.168.4.1)")
    parser.add_argument("--port",     default=81, type=int,   help="Stream preview port (default: 81)")
    parser.add_argument("--duration", default=10, type=float, help="Benchmark duration in seconds (default: 10s)")
    args = parser.parse_args()

    ip          = args.ip
    stream_port = args.port
    http_port   = 80
    stream_url  = f"http://{ip}:{stream_port}/stream"
    capture_url = f"http://{ip}/capture"
    info_url    = f"http://{ip}/info"

    overall_pass = True

    print("\n" + "=" * 65)
    print("  VisionMate v2 -- ESP32-CAM Snapshot (10 FPS) & Stream Benchmark")
    print("=" * 65)
    print(f"  Target IP         : {ip}")
    print(f"  AI Capture URL    : {capture_url}")
    print(f"  Preview Stream    : {stream_url}")
    print(f"  Target Interval   : 100 ms (~10 snapshots/sec)")
    print(f"  Test Duration     : {args.duration:.0f}s")
    print(f"  OpenCV Backend    : {'Available' if _cv2_available else 'Not installed (fallback mode)'}")

    # ── CHECK 1: TCP reachability ─────────────────────────────
    print(hdr("CHECK 1 -- Network Reachability"))
    tcp80 = check_tcp_reachable(ip, http_port, timeout=3.0)
    tcp81 = check_tcp_reachable(ip, stream_port, timeout=3.0)
    if tcp80:
        print(ok(f"Port 80 (Snapshot /capture & /info) open on {ip}"))
    else:
        print(fail(f"Port 80 not reachable on {ip}"))
        overall_pass = False

    if tcp81:
        print(ok(f"Port {stream_port} (Preview /stream) open on {ip}"))
    else:
        print(warn(f"Port {stream_port} (Preview /stream) not open on {ip} (optional)"))

    # ── CHECK 2: /info Telemetry ──────────────────────────────
    print(hdr("CHECK 2 -- Firmware Telemetry (/info)"))
    info_ok, info_data, info_err = fetch_http_json(info_url, timeout=4.0)
    if info_ok:
        print(ok(f"/info reachable"))
        print(info(f"Firmware           : {info_data.get('firmware', 'unknown')}"))
        print(info(f"Resolution         : {info_data.get('resolution', 'unknown')}"))
        print(info(f"JPEG Quality       : {info_data.get('quality', 'unknown')}"))
        print(info(f"PSRAM Available    : {info_data.get('psram', 'unknown')}"))
        print(info(f"WiFi Sleep Disabled: {info_data.get('wifi_sleep') is False}"))
        print(info(f"Sensor Capture Time: {info_data.get('last_capture_ms', 'n/a')} ms"))
        print(info(f"Free Heap / PSRAM  : {info_data.get('free_heap', 0)} / {info_data.get('free_psram', 0)} bytes"))
        print(info(f"Uptime             : {info_data.get('uptime_sec', 0)} s"))
    else:
        print(warn(f"/info endpoint: {info_err}"))

    # ── CHECK 3: Snapshot Loop Benchmark ──────────────────────
    print(hdr("CHECK 3 -- Continuous Snapshot Benchmark (10 FPS Target)"))
    if not tcp80:
        print(fail("Skipping snapshot benchmark -- Port 80 is not reachable."))
        overall_pass = False
    else:
        print(f"     Acquiring /capture snapshots at 100ms cadence for {args.duration:.0f}s...", flush=True)
        sb = benchmark_snapshot_loop(capture_url, target_fps=10.0, test_duration_sec=args.duration, timeout=3.0)

        if sb["snapshots_received"] == 0:
            print(fail(f"No snapshots received: {sb['error']}"))
            overall_pass = False
        else:
            print(ok(f"Snapshots received     : {sb['snapshots_received']}/{sb['snapshots_requested']}"))
            print(ok(f"Acquired Snapshot FPS  : {sb['actual_fps']} FPS (Target: 10.0 FPS)"))
            print(ok(f"Resolution             : {sb['width']}x{sb['height']} px"))
            print(ok(f"Average JPEG Size      : {sb['avg_jpeg_kb']} KB"))
            print(ok(f"Avg Request Latency    : {sb['avg_req_ms']} ms (min: {sb['min_req_ms']} ms, max: {sb['max_req_ms']} ms)"))
            print(ok(f"p95 Request Latency    : {sb['p95_req_ms']} ms"))
            print(ok(f"Avg JPEG Decode Time   : {sb['avg_dec_ms']} ms"))
            print(ok(f"Total Capture Latency  : {sb['avg_total_ms']} ms (p95: {sb['p95_total_ms']} ms)"))

            dec_rate = (sb["decode_success"] / max(1, sb["snapshots_received"])) * 100
            if dec_rate >= 95.0:
                print(ok(f"Decode Integrity Rate  : {dec_rate:.1f}% ({sb['decode_success']}/{sb['snapshots_received']})"))
            else:
                print(fail(f"Decode Integrity Rate  : {dec_rate:.1f}% ({sb['decode_failures']} failures)"))
                overall_pass = False

            if sb["actual_fps"] >= 8.5:
                print(ok(f"10 FPS Stability       : STABLE ({sb['actual_fps']:.1f} FPS)"))
            elif sb["actual_fps"] >= 6.0:
                print(warn(f"10 FPS Stability       : SLIGHT LAG ({sb['actual_fps']:.1f} FPS)"))
            else:
                print(fail(f"10 FPS Stability       : LOW ({sb['actual_fps']:.1f} FPS)"))
                overall_pass = False

    # ── CHECK 4: Preview Stream Availability ──────────────────
    print(hdr("CHECK 4 -- MJPEG Preview Stream (:81/stream)"))
    if tcp81:
        st_ok, st_msg = check_preview_stream(stream_url, timeout=3.0)
        if st_ok:
            print(ok(f"Preview Stream         : {st_msg}"))
            print(info("Note: Preview stream is reserved for browser/debugging, not AI input."))
        else:
            print(warn(f"Preview Stream issue   : {st_msg}"))
    else:
        print(warn("Port 81 closed -- MJPEG preview stream not active."))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    if overall_pass:
        print(f"  {GREEN}RESULT: ALL CHECKS PASSED [OK]{RESET}")
        print(f"  ESP32-CAM /capture snapshot pipeline is ready for AI perception.")
    else:
        print(f"  {RED}RESULT: CHECKS INCOMPLETE / FAILED{RESET}")
        print(f"  Ensure:")
        print(f"    1. Laptop Wi-Fi is connected to 'VisionMate' network.")
        print(f"    2. ESP32-CAM is powered and running firmware.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
