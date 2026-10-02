"""
VisionMate v2 - Camera Hardware Abstractions & Providers
Implements:
- DirectShowWebcam   : Built-in or USB webcam via DirectShow
- ESP32MjpegCamera   : ESP32-CAM Wi-Fi SoftAP MJPEG stream with full diagnostics,
                       auto-discovery (fixed IP + mDNS fallback), exponential
                       reconnect backoff, bounded latest-frame buffer, and rich
                       per-frame telemetry.
- SyntheticMockCamera: Deterministic simulation stream for tests
"""

import cv2
import time
import threading
import socket
import urllib.request
import urllib.error
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from backend.app.core.interfaces import CameraProvider

logger = logging.getLogger("visionmate.camera")


# ──────────────────────────────────────────────────────────────
# DirectShow Webcam
# ──────────────────────────────────────────────────────────────
class DirectShowWebcam(CameraProvider):
    """Local USB or built-in webcam provider using OpenCV DirectShow backend."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_connected = False

    def start(self) -> bool:
        if self._running:
            return True
        logger.info(f"Opening DirectShow webcam on index {self.device_index}...")
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open webcam at index {self.device_index}")
            self._is_connected = False
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._is_connected = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        while self._running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                    self._is_connected = True
                else:
                    self._is_connected = False
                    time.sleep(0.05)
            else:
                self._is_connected = False
                time.sleep(0.1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def is_connected(self) -> bool:
        return self._is_connected

    def get_resolution(self) -> Tuple[int, int]:
        return (self.target_width, self.target_height)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self._is_connected = False
        logger.info("Webcam stopped.")

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "source": "webcam",
            "connected": self._is_connected,
            "connection_state": "ONLINE" if self._is_connected else "OFFLINE",
        }


# ──────────────────────────────────────────────────────────────
# ESP32-CAM Snapshot Provider (~10 FPS / 100ms interval)
# ──────────────────────────────────────────────────────────────
class ESP32SnapshotCamera(CameraProvider):
    """
    Production ESP32-CAM Snapshot consumer for VisionMate v2.

    Pulls single-frame JPEG snapshots from /capture at ~10 FPS (100ms cadence)
    using non-blocking latest-frame semantics.

    Key design properties:
    - Fixed SoftAP IP (192.168.4.1/capture) — zero DHCP or manual configuration.
    - Low-overhead snapshot scheduler targeting ~10 FPS (100 ms interval).
    - Latest-frame semantics: newest frame always overwrites buffer; stale frames dropped.
    - Full telemetry: request latency, decode latency, p95 latency, snapshot FPS, dropped frames.
    - Resilient reconnection with exponential backoff on network hiccups.
    - Non-blocking get_latest_frame() returns immediately for 18 FPS pipeline ingestion.
    """

    STATE_OFFLINE      = "OFFLINE"
    STATE_ONLINE       = "ONLINE"
    STATE_RECONNECTING = "RECONNECTING"

    def __init__(
        self,
        capture_url: str = "http://192.168.4.1/capture",
        mdns_host:   str = "visionmatecam.local",
        target_fps:  float = 10.0,
        connect_timeout_sec: float = 3.0,
        max_reconnect_delay_sec: float = 6.0,
        mirror_horizontal: bool = True,
    ):
        self.capture_url = capture_url
        self.mdns_host   = mdns_host
        self.target_fps  = target_fps
        self.target_interval_sec     = 1.0 / max(1.0, target_fps)
        self.connect_timeout_sec     = connect_timeout_sec
        self.max_reconnect_delay_sec = max_reconnect_delay_sec
        self.mirror_horizontal       = mirror_horizontal

        # Frame buffer — single slot, latest always wins
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Connection state
        self._connection_state: str = self.STATE_OFFLINE
        self._active_url: str = capture_url

        # Diagnostic counters
        self._frames_received: int = 0
        self._frames_dropped: int  = 0
        self._reconnect_count: int = 0
        self._decode_failures: int = 0
        self._last_frame_time: float = 0.0
        self._session_start: float  = 0.0

        # Latency tracking
        self._last_request_latency_ms: float = 0.0
        self._last_decode_latency_ms: float  = 0.0
        self._request_latency_history: List[float] = []
        self._avg_request_latency_ms: float = 0.0
        self._p95_request_latency_ms: float = 0.0

        # FPS calculation
        self._fps_window_count: int   = 0
        self._fps_window_start: float = 0.0
        self._received_fps: float     = 0.0

        # Actual frame dimensions (detected from stream)
        self._frame_width:  int = 0
        self._frame_height: int = 0

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._session_start = time.time()
        self._fps_window_start = time.perf_counter()
        self._thread = threading.Thread(
            target=self._snapshot_worker, daemon=True, name="esp32-snapshot"
        )
        self._thread.start()
        logger.info(f"ESP32SnapshotCamera started (Target: {self.target_fps:.1f} FPS, URL: {self.capture_url})")
        return True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def is_connected(self) -> bool:
        return self._connection_state == self.STATE_ONLINE

    def get_resolution(self) -> Tuple[int, int]:
        return (self._frame_width or 800, self._frame_height or 600)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._connection_state = self.STATE_OFFLINE
        logger.info("ESP32SnapshotCamera stopped.")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns rich snapshot telemetry consumed by /status and /metrics."""
        age_ms = 0.0
        if self._last_frame_time > 0:
            age_ms = round((time.time() - self._last_frame_time) * 1000.0, 1)
        return {
            "source": "esp32_snapshot",
            "active_url": self._active_url,
            "connection_state": self._connection_state,
            "connected": self._connection_state == self.STATE_ONLINE,
            "camera_received_fps": round(self._received_fps, 1),
            "snapshot_fps": round(self._received_fps, 1),
            "snapshot_request_latency_ms": round(self._last_request_latency_ms, 1),
            "avg_request_latency_ms": round(self._avg_request_latency_ms, 1),
            "p95_request_latency_ms": round(self._p95_request_latency_ms, 1),
            "jpeg_decode_latency_ms": round(self._last_decode_latency_ms, 1),
            "frame_width": self._frame_width,
            "frame_height": self._frame_height,
            "frames_received": self._frames_received,
            "dropped_frames": self._frames_dropped,
            "reconnect_count": self._reconnect_count,
            "decode_failures": self._decode_failures,
            "last_frame_timestamp": round(self._last_frame_time, 3),
            "frame_age_ms": age_ms,
        }

    def _resolve_active_url(self) -> Optional[str]:
        """Verify fixed IP primary or fallback to mDNS."""
        if self._probe_url(self.capture_url):
            return self.capture_url
        if self.mdns_host:
            mdns_url = self._build_mdns_url()
            if mdns_url and self._probe_url(mdns_url):
                return mdns_url
        return None

    def _probe_url(self, url: str) -> bool:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VisionMate-Probe"})
            with urllib.request.urlopen(req, timeout=self.connect_timeout_sec) as resp:
                data = resp.read(64)
                return len(data) > 0
        except Exception:
            return False

    def _build_mdns_url(self) -> Optional[str]:
        try:
            ip = socket.gethostbyname(self.mdns_host)
            return f"http://{ip}/capture"
        except Exception:
            return None

    def _snapshot_worker(self):
        """
        Snapshot Acquisition Loop:
        - Requests /capture every ~100 ms (~10 FPS).
        - Validates and decodes JPEG into BGR numpy array.
        - Places frame in single-slot latest buffer.
        - Measures request latency and p95 timing metrics.
        """
        reconnect_delay = 0.5

        while self._running:
            self._connection_state = self.STATE_RECONNECTING
            active_url = self._resolve_active_url()

            if active_url is None:
                logger.warning(
                    f"ESP32-CAM capture endpoint unreachable ({self.capture_url}). "
                    f"Retry in {reconnect_delay:.1f}s..."
                )
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self.max_reconnect_delay_sec)
                self._reconnect_count += 1
                continue

            self._active_url = active_url
            self._connection_state = self.STATE_ONLINE
            reconnect_delay = 0.5
            logger.info(f"ESP32-CAM snapshot loop active: {active_url} (Cadence: {self.target_interval_sec*1000:.0f}ms)")

            req = urllib.request.Request(
                active_url,
                headers={
                    "User-Agent": "VisionMate-Snapshot",
                    "Cache-Control": "no-cache, no-store",
                    "Connection": "keep-alive"
                }
            )

            while self._running:
                t0 = time.perf_counter()

                try:
                    with urllib.request.urlopen(req, timeout=self.connect_timeout_sec) as resp:
                        data = resp.read()
                    t1 = time.perf_counter()

                    req_ms = (t1 - t0) * 1000.0
                    self._last_request_latency_ms = req_ms

                    # Track latency history for avg & p95
                    self._request_latency_history.append(req_ms)
                    if len(self._request_latency_history) > 100:
                        self._request_latency_history.pop(0)
                    self._avg_request_latency_ms = sum(self._request_latency_history) / len(self._request_latency_history)
                    sorted_l = sorted(self._request_latency_history)
                    p95_idx = int(len(sorted_l) * 0.95)
                    self._p95_request_latency_ms = sorted_l[min(p95_idx, len(sorted_l) - 1)]

                    # Validate JPEG magic bytes
                    if len(data) > 100 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9":
                        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        t2 = time.perf_counter()
                        self._last_decode_latency_ms = (t2 - t1) * 1000.0

                        if frame is not None:
                            # Apply horizontal mirror BEFORE buffering and AI processing
                            if self.mirror_horizontal:
                                frame = cv2.flip(frame, 1)

                            if self._frame_height == 0:
                                self._frame_height, self._frame_width = frame.shape[:2]
                                logger.info(f"ESP32 snapshot received — {self._frame_width}×{self._frame_height} (mirrored={self.mirror_horizontal})")

                            with self._lock:
                                if self._latest_frame is not None:
                                    self._frames_dropped += 1
                                self._latest_frame = frame

                            self._frames_received += 1
                            self._last_frame_time = time.time()

                            # Rolling FPS calculation
                            self._fps_window_count += 1
                            now = time.perf_counter()
                            elapsed = now - self._fps_window_start
                            if elapsed >= 1.0:
                                self._received_fps = self._fps_window_count / elapsed
                                self._fps_window_count = 0
                                self._fps_window_start = now
                        else:
                            self._decode_failures += 1
                    else:
                        self._decode_failures += 1

                except Exception as e:
                    self._connection_state = self.STATE_OFFLINE
                    logger.warning(f"ESP32 snapshot error: {e}. Reconnecting...")
                    break

                # Precise sleep to maintain target 10 FPS cadence (~100ms interval)
                elapsed_total = time.perf_counter() - t0
                sleep_time = self.target_interval_sec - elapsed_total
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, self.max_reconnect_delay_sec)


# ──────────────────────────────────────────────────────────────
# ESP32-CAM MJPEG Stream Provider (Preview / Debug)
# ──────────────────────────────────────────────────────────────
class ESP32MjpegCamera(CameraProvider):
    """
    ESP32-CAM MJPEG stream consumer for preview / diagnostic inspection.
    """

    # Connection state constants
    STATE_OFFLINE      = "OFFLINE"
    STATE_ONLINE       = "ONLINE"
    STATE_RECONNECTING = "RECONNECTING"

    def __init__(
        self,
        stream_url: str = "http://192.168.4.1:81/stream",
        mdns_host:  str = "visionmatecam.local",
        connect_timeout_sec: float = 3.0,
        max_reconnect_delay_sec: float = 8.0,
        read_chunk_bytes: int = 8192,
        mirror_horizontal: bool = True,
    ):
        self.stream_url = stream_url
        self.mdns_host  = mdns_host
        self.connect_timeout_sec     = connect_timeout_sec
        self.max_reconnect_delay_sec = max_reconnect_delay_sec
        self.read_chunk_bytes        = read_chunk_bytes
        self.mirror_horizontal       = mirror_horizontal

        # Frame buffer — single slot, latest always wins
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Connection state
        self._connection_state: str = self.STATE_OFFLINE
        self._active_url: str = stream_url

        # Diagnostic counters
        self._frames_received: int = 0
        self._frames_dropped: int  = 0      # processor slower than camera
        self._reconnect_count: int = 0
        self._decode_failures: int = 0
        self._last_frame_time: float = 0.0
        self._session_start: float  = 0.0

        # FPS calculation
        self._fps_window_count: int   = 0
        self._fps_window_start: float = 0.0
        self._received_fps: float     = 0.0

        # Actual frame dimensions (detected from stream)
        self._frame_width:  int = 0
        self._frame_height: int = 0

    # ── public API ────────────────────────────────────────────

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._session_start = time.time()
        self._fps_window_start = time.perf_counter()
        self._thread = threading.Thread(
            target=self._stream_worker, daemon=True, name="esp32-mjpeg"
        )
        self._thread.start()
        logger.info(f"ESP32MjpegCamera started — primary URL: {self.stream_url}")
        return True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def is_connected(self) -> bool:
        return self._connection_state == self.STATE_ONLINE

    def get_resolution(self) -> Tuple[int, int]:
        return (self._frame_width or 800, self._frame_height or 600)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._connection_state = self.STATE_OFFLINE
        logger.info("ESP32MjpegCamera stopped.")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns a rich diagnostics dict consumed by /status and /metrics."""
        age_ms = 0.0
        if self._last_frame_time > 0:
            age_ms = round((time.time() - self._last_frame_time) * 1000.0, 1)
        return {
            "source": "esp32",
            "active_url": self._active_url,
            "connection_state": self._connection_state,
            "connected": self._connection_state == self.STATE_ONLINE,
            "camera_received_fps": round(self._received_fps, 1),
            "frame_width": self._frame_width,
            "frame_height": self._frame_height,
            "frames_received": self._frames_received,
            "dropped_frames": self._frames_dropped,
            "reconnect_count": self._reconnect_count,
            "decode_failures": self._decode_failures,
            "last_frame_timestamp": round(self._last_frame_time, 3),
            "frame_age_ms": age_ms,
        }

    # ── discovery ─────────────────────────────────────────────

    def _resolve_active_url(self) -> Optional[str]:
        """
        Try the primary fixed-IP URL first.
        If unreachable, attempt mDNS resolution of the configured hostname.
        Returns the working stream URL or None.
        """
        # 1. Fixed SoftAP IP
        if self._probe_url(self.stream_url):
            logger.info(f"ESP32 reachable at primary URL: {self.stream_url}")
            return self.stream_url

        # 2. mDNS fallback
        if self.mdns_host:
            mdns_url = self._build_mdns_stream_url()
            if mdns_url and self._probe_url(mdns_url):
                logger.info(f"ESP32 reachable via mDNS: {mdns_url}")
                return mdns_url

        return None

    def _probe_url(self, url: str) -> bool:
        """Quick TCP probe to see if the HTTP server is reachable."""
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=self.connect_timeout_sec)
            return True
        except Exception:
            try:
                # Some MJPEG servers don't support HEAD — try GET with 1-byte read
                resp = urllib.request.urlopen(url, timeout=self.connect_timeout_sec)
                resp.read(1)
                resp.close()
                return True
            except Exception:
                return False

    def _build_mdns_stream_url(self) -> Optional[str]:
        """Resolve mDNS hostname to IP and build stream URL."""
        try:
            ip = socket.gethostbyname(self.mdns_host)
            # Preserve the port from the primary URL
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.stream_url)
            mdns_parsed = parsed._replace(netloc=f"{ip}:{parsed.port}" if parsed.port else ip)
            return urlunparse(mdns_parsed)
        except Exception:
            return None

    # ── main stream worker ────────────────────────────────────

    def _stream_worker(self):
        """
        Persistent MJPEG stream consumer.
        - Tries auto-discovery each reconnect cycle.
        - Parses MJPEG boundary frames by JPEG SOI/EOI byte markers.
        - Writes only the newest frame into the single-slot buffer.
        - Tracks full diagnostics throughout.
        """
        reconnect_delay = 0.5

        while self._running:
            # Discovery phase
            self._connection_state = self.STATE_RECONNECTING
            active_url = self._resolve_active_url()

            if active_url is None:
                logger.warning(
                    f"ESP32-CAM not reachable (primary={self.stream_url}, "
                    f"mDNS={self.mdns_host}). Retry in {reconnect_delay:.1f}s..."
                )
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self.max_reconnect_delay_sec)
                if self._reconnect_count > 0:
                    self._reconnect_count += 1
                continue

            self._active_url = active_url

            # Stream reading phase
            try:
                logger.info(f"Opening MJPEG stream: {active_url}")
                stream = urllib.request.urlopen(
                    active_url, timeout=self.connect_timeout_sec
                )
                self._connection_state = self.STATE_ONLINE
                reconnect_delay = 0.5  # reset backoff on successful connect
                if self._frames_received > 0:
                    self._reconnect_count += 1
                    logger.info(f"ESP32-CAM reconnected (reconnect #{self._reconnect_count})")

                buf = b""
                self._fps_window_start = time.perf_counter()
                self._fps_window_count = 0

                while self._running:
                    chunk = stream.read(self.read_chunk_bytes)
                    if not chunk:
                        logger.warning("ESP32-CAM: stream ended (zero-byte read).")
                        break

                    buf += chunk

                    # Extract all complete JPEG frames from buffer
                    while True:
                        soi = buf.find(b"\xff\xd8")  # JPEG Start Of Image
                        if soi == -1:
                            break
                        eoi = buf.find(b"\xff\xd9", soi + 2)  # JPEG End Of Image
                        if eoi == -1:
                            break  # partial frame — wait for more data

                        jpg_bytes = buf[soi : eoi + 2]
                        buf = buf[eoi + 2:]  # consume frame from buffer

                        frame = cv2.imdecode(
                            np.frombuffer(jpg_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )

                        if frame is None:
                            self._decode_failures += 1
                            continue

                        # Apply horizontal mirror BEFORE buffering and AI processing
                        if self.mirror_horizontal:
                            frame = cv2.flip(frame, 1)

                        # Update dimensions on first real frame
                        if self._frame_height == 0:
                            self._frame_height, self._frame_width = frame.shape[:2]
                            logger.info(
                                f"ESP32-CAM stream active — "
                                f"resolution: {self._frame_width}×{self._frame_height}"
                            )

                        # Write to bounded single-slot buffer
                        # If processor hasn't consumed previous frame, count as drop
                        with self._lock:
                            if self._latest_frame is not None:
                                self._frames_dropped += 1
                            self._latest_frame = frame

                        self._frames_received += 1
                        self._last_frame_time = time.time()

                        # Rolling FPS calculation (1-second window)
                        self._fps_window_count += 1
                        now = time.perf_counter()
                        elapsed = now - self._fps_window_start
                        if elapsed >= 1.0:
                            self._received_fps = self._fps_window_count / elapsed
                            self._fps_window_count = 0
                            self._fps_window_start = now

            except urllib.error.URLError as e:
                self._connection_state = self.STATE_OFFLINE
                logger.warning(
                    f"ESP32-CAM URLError: {e}. Reconnecting in {reconnect_delay:.1f}s..."
                )
            except Exception as e:
                self._connection_state = self.STATE_OFFLINE
                logger.warning(
                    f"ESP32-CAM stream error: {e}. Reconnecting in {reconnect_delay:.1f}s..."
                )
            finally:
                self._connection_state = self.STATE_OFFLINE

            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, self.max_reconnect_delay_sec)


# ──────────────────────────────────────────────────────────────
# Synthetic Mock Camera (Tests / Demo)
# ──────────────────────────────────────────────────────────────
class SyntheticMockCamera(CameraProvider):
    """Generates synthetic dynamic frames with simulated obstacles, persons, and text."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._step = 0
        self.scenario: str = "office_desk"

    def start(self) -> bool:
        self._running = True
        self._thread = threading.Thread(target=self._generator_loop, daemon=True)
        self._thread.start()
        return True

    def set_scenario(self, scenario_name: str):
        self.scenario = scenario_name

    def _generator_loop(self):
        dt = 1.0 / self.fps
        while self._running:
            frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
            self._step += 1
            t = self._step * 0.05

            if self.scenario == "office_desk":
                cv2.rectangle(frame, (100, 260), (540, 460), (70, 70, 70), -1)
                cv2.rectangle(frame, (260, 240), (380, 340), (120, 120, 120), -1)
                cv2.putText(frame, "laptop", (270, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (430, 250), (470, 350), (200, 100, 50), -1)
                cv2.putText(frame, "bottle", (420, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                px = int(80 + np.sin(t) * 20)
                cv2.rectangle(frame, (px, 120), (px + 80, 360), (50, 180, 50), -1)
                cv2.putText(frame, "person", (px, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif self.scenario == "hallway_obstacle":
                cx = 320
                scale = min(1.0 + (t % 6) * 0.3, 2.5)
                w_box = int(80 * scale)
                h_box = int(100 * scale)
                y_box = int(240 + (t % 6) * 25)
                cv2.rectangle(frame, (cx - w_box // 2, y_box - h_box), (cx + w_box // 2, y_box), (0, 0, 200), -1)
                cv2.putText(frame, "chair", (cx - 20, y_box - h_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif self.scenario == "reading_text":
                cv2.rectangle(frame, (140, 100), (500, 380), (240, 240, 240), -1)
                cv2.putText(frame, "VISIONMATE ASSISTANT", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
                cv2.putText(frame, "Room 302 - AI Lab Entrance", (160, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
                cv2.putText(frame, "Caution: Automated Doors Ahead", (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

            elif self.scenario == "dropoff_hazard":
                cv2.rectangle(frame, (80, 320), (560, 480), (20, 20, 180), -1)
                cv2.putText(frame, "stairs", (270, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            with self._lock:
                self._latest_frame = frame

            time.sleep(dt)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def is_connected(self) -> bool:
        return self._running

    def get_resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "source": "synthetic",
            "connected": self._running,
            "connection_state": "ONLINE" if self._running else "OFFLINE",
        }


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def create_camera_provider(source_type: str = "webcam", **kwargs) -> CameraProvider:
    """
    Factory to instantiate the configured CameraProvider.

    source_type:
        'esp32'          → ESP32SnapshotCamera (~10 FPS / 100ms /capture endpoint for AI)
        'esp32_snapshot' → ESP32SnapshotCamera (explicit alias)
        'esp32_mjpeg'    → ESP32MjpegCamera (continuous stream for preview/debugging)
        'webcam'         → DirectShowWebcam (local USB/built-in)
        'mock'           → SyntheticMockCamera (tests / demo)
        'synthetic'      → SyntheticMockCamera (alias)
    """
    s = source_type.lower().strip()
    from backend.app.core.config import settings as _s

    if s in ("esp32", "esp32_snapshot", "snapshot"):
        return ESP32SnapshotCamera(
            capture_url=kwargs.get("capture_url", _s.ESP32_CAPTURE_URL),
            mdns_host=kwargs.get("mdns_host", _s.ESP32_MDNS_HOST),
            target_fps=kwargs.get("target_fps", _s.ESP32_SNAPSHOT_TARGET_FPS),
            connect_timeout_sec=kwargs.get("connect_timeout_sec", _s.ESP32_CONNECT_TIMEOUT_SEC),
            max_reconnect_delay_sec=kwargs.get("max_reconnect_delay_sec", _s.ESP32_RECONNECT_MAX_DELAY_SEC),
            mirror_horizontal=kwargs.get("mirror_horizontal", _s.ESP32_MIRROR_HORIZONTAL),
        )
    elif s in ("esp32_mjpeg", "esp32_stream", "mjpeg", "stream"):
        return ESP32MjpegCamera(
            stream_url=kwargs.get("stream_url", _s.ESP32_STREAM_URL),
            mdns_host=kwargs.get("mdns_host",   _s.ESP32_MDNS_HOST),
            connect_timeout_sec=kwargs.get("connect_timeout_sec", _s.ESP32_CONNECT_TIMEOUT_SEC),
            max_reconnect_delay_sec=kwargs.get("max_reconnect_delay_sec", _s.ESP32_RECONNECT_MAX_DELAY_SEC),
            read_chunk_bytes=kwargs.get("read_chunk_bytes", _s.ESP32_READ_CHUNK_BYTES),
            mirror_horizontal=kwargs.get("mirror_horizontal", _s.ESP32_MIRROR_HORIZONTAL),
        )
    elif s in ("mock", "synthetic"):
        return SyntheticMockCamera(
            width=kwargs.get("width", 640),
            height=kwargs.get("height", 480),
        )
    else:
        return DirectShowWebcam(device_index=kwargs.get("device_index", 0))

