"""
VisionMate v2 - Master Pipeline Orchestrator

Camera -> YOLO -> Tracker -> World State -> Priority -> Modes -> Command Bus -> TTS
Voice: Mic -> VAD -> ASR (loaded once) -> Intent -> Command Bus -> Features
"""

import os
import time
import threading
from typing import Optional, Dict, Any

import numpy as np
import cv2
import logging

from backend.app.core.config import settings
from backend.app.core.interfaces import (
    CameraProvider,
    ObjectDetector,
    Tracker,
    OCRProvider,
    VisionReasoner,
    TTSProvider,
    ASRProvider,
    DepthProvider,
    GPSProvider,
)
from backend.app.core.events import EventType, event_broker
from backend.app.hardware.camera import create_camera_provider
from backend.app.hardware.depth import create_depth_provider
from backend.app.hardware.gps import create_gps_provider
from backend.app.perception.detector import YOLOObjectDetector
from backend.app.tracking.tracker import SpatialTracker
from backend.app.world_state.manager import world_state_mgr, WorldStateManager
from backend.app.priority.engine import PriorityEngine, PriorityLevel
from backend.app.speech.tts import tts_engine, WindowsSAPITTSProvider
from backend.app.speech.asr import LocalASRProvider
from backend.app.reasoning.ocr import PPOCRv5Provider
from backend.app.reasoning.vlm import OllamaQwen3VLReasoner
from backend.app.modes.awareness import AwarenessModeHandler
from backend.app.modes.find import FindModeHandler
from backend.app.modes.read import ReadModeHandler
from backend.app.modes.ask import AskModeHandler
from backend.app.intelligence.command_bus import CommandBus
from backend.app.schemas.world_state import SystemMode, WorldState

logger = logging.getLogger("visionmate.pipeline")


class VisionMatePipeline:
    """Master perceptual, voice, and reasoning pipeline."""

    def __init__(self, camera_source: str = "webcam", **camera_kwargs):
        self.camera_source_type = camera_source
        self.camera_kwargs = camera_kwargs

        self.camera: CameraProvider = create_camera_provider(camera_source, **camera_kwargs)
        self.depth: DepthProvider = create_depth_provider(settings.TOF_SOURCE)
        self.gps: GPSProvider = create_gps_provider(settings.GPS_SOURCE)

        self.detector: YOLOObjectDetector = YOLOObjectDetector(
            model_path=settings.YOLO_MODEL_PATH,
            device=settings.DETECTION_DEVICE,
            conf_thresh=settings.DETECTION_CONF_THRESHOLD,
            auto_load=False,
        )
        self.tracker: Tracker = SpatialTracker(
            max_disappeared=settings.TRACKER_MAX_DISAPPEARED,
            iou_threshold=settings.TRACKER_IOU_THRESHOLD,
            min_stable_frames=3,
        )
        self.world_state_mgr: WorldStateManager = world_state_mgr
        self.priority_engine = PriorityEngine(
            guidance_cooldown=settings.GUIDANCE_SPEECH_COOLDOWN_SEC,
            hazard_cooldown=settings.HAZARD_SPEECH_COOLDOWN_SEC,
        )
        self.tts: WindowsSAPITTSProvider = tts_engine
        self.asr: ASRProvider = LocalASRProvider(
            model_size=settings.ASR_MODEL,
            device=settings.ASR_DEVICE,
            compute_type=settings.ASR_COMPUTE_TYPE,
            sample_rate=settings.ASR_SAMPLE_RATE,
        )
        self.ocr: OCRProvider = PPOCRv5Provider()
        self.vlm: VisionReasoner = OllamaQwen3VLReasoner()

        self.guidance_handler = AwarenessModeHandler(self.priority_engine)
        self.find_handler = FindModeHandler(update_interval_sec=settings.FIND_MODE_UPDATE_INTERVAL_SEC)
        self.read_handler = ReadModeHandler(self.ocr)
        self.ask_handler = AskModeHandler(self.vlm)

        self.command_bus = CommandBus(self.asr.parse_intent)
        self._register_command_handlers()

        self._running = False
        self._perception_thread: Optional[threading.Thread] = None
        self._voice_thread: Optional[threading.Thread] = None
        self._voice_running = False
        self.voice_state = "LISTENING"
        self._lock = threading.Lock()

        self._latest_annotated_frame = self._create_standby_frame(
            "VisionMate v2 - Initializing 18 FPS pipeline..."
        )

        self.frames_processed = 0
        self.start_time = 0.0
        self.current_fps = 0.0
        self.last_latency_ms = 0.0
        self.last_yolo_latency_ms = 0.0

    def _register_command_handlers(self) -> None:
        self.command_bus.register("STOP", lambda parsed: self.handle_stop_command())
        self.command_bus.register("FIND", self._cmd_find)
        self.command_bus.register("READ", self._cmd_read)
        self.command_bus.register("ASK", self._cmd_ask)
        self.command_bus.register("GUIDANCE", self._cmd_guidance)
        self.command_bus.register("COLOR", self._cmd_not_available)
        self.command_bus.register("MEDICINE", self._cmd_not_available)
        self.command_bus.register("CURRENCY", self._cmd_not_available)
        self.command_bus.register("FACE", self._cmd_not_available)
        self.command_bus.register("NAVIGATION", self._cmd_not_available)
        self.command_bus.register("SOS", self._cmd_not_available)

    def _cmd_find(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        target = parsed.get("target") or "object"
        msg = self.start_find(target)
        return {"status": "started", "mode": "find", "message": msg, "target": target}

    def _cmd_read(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        is_full = parsed.get("full_reading", False)
        res = self.trigger_read(full_reading=is_full)
        return {"status": "started", "mode": "read", "result": res}

    def _cmd_ask(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        query = parsed.get("query") or parsed.get("raw_text") or ""
        res = self.trigger_ask(query)
        return {"status": "started", "mode": "ask", "result": res}

    def _cmd_guidance(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        msg = self.set_mode("guidance")
        return {"status": "started", "mode": "guidance", "message": msg}

    def _cmd_not_available(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        intent = parsed.get("intent", "FEATURE")
        return {
            "status": "not_available",
            "message": f"{intent} is not enabled in this backend phase.",
        }

    def _set_voice_state(self, state: str) -> None:
        self.voice_state = state
        event_broker.publish(EventType.VOICE_STATE, {"state": state})

    def _create_standby_frame(self, message: str) -> np.ndarray:
        img = np.full((480, 640, 3), 20, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (640, 40), (35, 39, 46), -1)
        cv2.putText(
            img,
            "VISIONMATE v2 | GUIDANCE HUD (18 FPS)",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(img, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    def start(self) -> bool:
        if self._running:
            return True

        self._running = True
        self.start_time = time.time()
        self._perception_thread = threading.Thread(target=self._async_bootstrap_and_loop, daemon=True)
        self._perception_thread.start()

        voice_enabled = settings.VOICE_LISTENER_ENABLED and not os.environ.get("PYTEST_CURRENT_TEST")
        if voice_enabled:
            self._voice_running = True
            self._voice_thread = threading.Thread(
                target=self._voice_loop,
                daemon=True,
                name="VisionMateVoiceListener",
            )
            self._voice_thread.start()

        logger.info(
            "[VISION] Pipeline worker spawned (target %s FPS, camera=%s).",
            settings.VISIONMATE_TARGET_FPS,
            self.camera_source_type,
        )
        return True

    def _async_bootstrap_and_loop(self):
        logger.info("[VISION] Bootstrapping hardware & model subsystems...")
        logger.info("[VISION] Primary camera: %s", self.camera_source_type)
        try:
            self.camera.start()
            if not self.camera.is_connected():
                logger.warning(
                    "[VISION] Camera unavailable (%s). Background reconnect active.",
                    self.camera_source_type,
                )
        except Exception as e:
            logger.warning("[VISION] Camera startup notice: %s. Background reconnect active.", e)

        try:
            self.detector.load()
        except Exception as e:
            logger.error("[VISION] Error loading YOLO detector: %s", e)

        try:
            self.tts.speak("VisionMate ready. Guidance mode active.", priority=PriorityLevel.INTERACTION, source="GUIDANCE")
        except Exception as e:
            logger.warning("[TTS] Greeting skipped: %s", e)

        target_period = 1.0 / float(settings.VISIONMATE_TARGET_FPS)
        last_time = time.perf_counter()

        while self._running:
            t0 = time.perf_counter()
            frame = self.camera.get_latest_frame()

            if frame is not None:
                h, w = frame.shape[:2]
                detections = []
                t_yolo_start = time.perf_counter()
                if self.detector.is_ready():
                    detections = self.detector.detect(frame)
                self.last_yolo_latency_ms = (time.perf_counter() - t_yolo_start) * 1000.0

                active_tracks = self.tracker.update(detections, (h, w))
                world_state = self.world_state_mgr.update_from_tracks(active_tracks, fps=self.current_fps)

                current_mode = world_state.current_mode
                speech_cmd = None
                if current_mode in [SystemMode.GUIDANCE, SystemMode.AWARENESS]:
                    speech_cmd = self.guidance_handler.process_frame_state(world_state)
                elif current_mode == SystemMode.FIND:
                    hazard_event = self.priority_engine.evaluate_hazard(world_state)
                    if hazard_event:
                        speech_cmd = hazard_event
                    else:
                        speech_cmd = self.find_handler.process_frame_state(world_state)

                if speech_cmd and "text" in speech_cmd:
                    txt = speech_cmd["text"]
                    pri = speech_cmd.get("priority", PriorityLevel.INTERACTION)
                    inter = speech_cmd.get("interrupt", False)
                    src = speech_cmd.get("source", "GUIDANCE")
                    self.tts.speak(txt, priority=pri, interrupt=inter, source=src)
                    self.world_state_mgr.record_speech(txt)

                self._draw_hud_annotations(frame, world_state)
                self.frames_processed += 1
                self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            else:
                with self._lock:
                    self._latest_annotated_frame = self._create_standby_frame(
                        f"Waiting for camera ({self.camera_source_type})..."
                    )

            dt = time.perf_counter() - last_time
            if dt > 0.5:
                self.current_fps = round(self.frames_processed / max(1e-3, (time.time() - self.start_time)), 1)

            elapsed = time.perf_counter() - t0
            sleep_time = target_period - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def _voice_loop(self):
        logger.info("[VOICE] Command listener started (VAD, not fixed 3s chunks).")
        try:
            self.asr.preload()
        except Exception as e:
            logger.error("[VOICE] ASR preload failed: %s", e)

        while self._voice_running:
            try:
                self._set_voice_state("LISTENING")
                event_broker.publish(EventType.VOICE_STARTED, {"state": "LISTENING"})
                transcript = self.asr.listen_utterance(stop_flag=lambda: not self._voice_running)
                if not transcript:
                    continue

                transcript = transcript.strip()
                logger.info("[VOICE] Heard: %s", transcript)
                self._set_voice_state("PROCESSING")
                event_broker.publish(EventType.VOICE_TRANSCRIPT, {"transcript": transcript})
                self._set_voice_state("THINKING")
                result = self.handle_voice_command(transcript)
                logger.info("[VOICE] Result: %s", result)
                if self.tts.is_speaking():
                    self._set_voice_state("SPEAKING")
            except Exception as e:
                logger.error("[VOICE] Listener error: %s", e)
                event_broker.publish(EventType.ERROR, {"source": "voice", "message": str(e)})
                time.sleep(0.5)

        logger.info("[VOICE] Command listener stopped.")

    def _draw_hud_annotations(self, frame: np.ndarray, world_state: WorldState):
        annotated = frame.copy()
        h, w = frame.shape[:2]
        cv2.line(annotated, (int(w * 0.34), 0), (int(w * 0.34), h), (45, 45, 45), 1)
        cv2.line(annotated, (int(w * 0.66), 0), (int(w * 0.66), h), (45, 45, 45), 1)

        for obj in world_state.get_active_objects():
            x1, y1, x2, y2 = obj.bounding_box
            cls = obj.class_name
            tid = obj.tracking_id
            dir_text = obj.direction.value
            if obj.hazard_severity.value in ["emergency", "warning"]:
                color = (0, 0, 240)
            elif obj.is_stable:
                color = (0, 220, 80)
            else:
                color = (120, 120, 120)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid} {cls} [{dir_text}]"
            if not obj.is_stable:
                label += " (confirming...)"
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        det_status = "OK" if self.detector.is_ready() else self.detector.status
        mode_name = world_state.current_mode.value.upper()
        mode_str = f"MODE: {mode_name} | FPS: {self.current_fps:.1f}/18 | Lat: {self.last_latency_ms:.1f}ms | Det: {det_status} | Voice: {self.voice_state}"
        cv2.rectangle(annotated, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(annotated, mode_str, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 2, cv2.LINE_AA)

        with self._lock:
            self._latest_annotated_frame = annotated

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated_frame is not None:
                return self._latest_annotated_frame.copy()
            return None

    def public_world_view(self) -> Dict[str, Any]:
        state = self.world_state_mgr.get_snapshot()
        objects = []
        for obj in state.get_active_objects():
            objects.append({
                "id": f"{obj.class_name}_{obj.tracking_id}",
                "label": obj.class_name,
                "position": obj.direction.value,
                "distance_m": None,
                "velocity": obj.motion_estimate.value,
                "confidence": obj.confidence,
            })
        hazards = []
        if state.active_hazard:
            hazards.append(state.active_hazard)
        return {
            "mode": state.current_mode.value.upper(),
            "objects": objects,
            "hazards": hazards,
            "navigation": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(state.timestamp)),
            "voice_state": self.voice_state,
            "camera_connected": self.camera.is_connected(),
            "depth": self.depth.health(),
            "gps": self.gps.health(),
        }

    def get_subsystems_health(self) -> Dict[str, Any]:
        cam_diag = {}
        if hasattr(self.camera, "get_diagnostics"):
            cam_diag = self.camera.get_diagnostics()
        else:
            cam_diag = {
                "source": self.camera_source_type,
                "connected": self.camera.is_connected(),
                "connection_state": "ONLINE" if self.camera.is_connected() else "OFFLINE",
            }

        asr_ready = getattr(self.asr, "model", None) is not None
        return {
            "camera": {
                "configured_source": settings.CAMERA_SOURCE,
                "active_source": self.camera_source_type,
                "connected": self.camera.is_connected(),
                "connection_state": cam_diag.get("connection_state", "UNKNOWN"),
                "camera_received_fps": cam_diag.get("camera_received_fps", 0.0),
                "snapshot_fps": cam_diag.get("snapshot_fps", cam_diag.get("camera_received_fps", 0.0)),
                "snapshot_request_latency_ms": cam_diag.get("snapshot_request_latency_ms", 0.0),
                "avg_request_latency_ms": cam_diag.get("avg_request_latency_ms", 0.0),
                "p95_request_latency_ms": cam_diag.get("p95_request_latency_ms", 0.0),
                "jpeg_decode_latency_ms": cam_diag.get("jpeg_decode_latency_ms", 0.0),
                "frame_width": cam_diag.get("frame_width", 0),
                "frame_height": cam_diag.get("frame_height", 0),
                "dropped_frames": cam_diag.get("dropped_frames", 0),
                "reconnect_count": cam_diag.get("reconnect_count", 0),
                "decode_failures": cam_diag.get("decode_failures", 0),
                "last_frame_timestamp": cam_diag.get("last_frame_timestamp", 0.0),
                "frame_age_ms": cam_diag.get("frame_age_ms", 0.0),
                "active_url": cam_diag.get("active_url", ""),
            },
            "detector": {
                "status": self.detector.status,
                "model": settings.YOLO_MODEL_PATH,
                "device": self.detector.device,
                "ready": self.detector.is_ready(),
            },
            "speech_engine": {
                "status": "READY" if self.tts is not None else "UNAVAILABLE",
                "speaking": self.tts.is_speaking() if self.tts else False,
                "speech_generation": self.tts.speech_generation,
            },
            "asr": {
                "model": settings.ASR_MODEL,
                "device": settings.ASR_DEVICE,
                "loaded": asr_ready,
                "voice_state": self.voice_state,
                "capture": "vad",
            },
            "depth": self.depth.health(),
            "gps": self.gps.health(),
            "ocr": {
                "status": "ON_DEMAND",
                "engine_loaded": self.ocr.ocr_engine is not None,
            },
            "vlm": {
                "model": settings.OLLAMA_VLM_MODEL,
                "status": "ON_DEMAND",
            },
            "target_fps": settings.VISIONMATE_TARGET_FPS,
            "current_fps": self.current_fps,
        }

    def handle_voice_command(self, transcript: str) -> Dict[str, Any]:
        return self.command_bus.dispatch(transcript)

    def handle_stop_command(self) -> Dict[str, Any]:
        self.tts.stop()
        self.find_handler.stop_find()
        self.world_state_mgr.set_find_target(None)
        self.world_state_mgr.set_mode(SystemMode.GUIDANCE)
        self.priority_engine.last_speech_time = time.time()
        event_broker.publish(EventType.MODE_CHANGED_PUBLIC, {"mode": "GUIDANCE"})
        logger.info("[GLOBAL STOP] Speech purged, Find terminated, Guidance silent.")
        return {
            "status": "STOPPED",
            "mode": "guidance",
            "message": "System stopped. Guidance active.",
            "speech_generation": self.tts.speech_generation,
        }

    def set_mode(self, mode_name: str) -> str:
        m = mode_name.lower().strip()
        if m in ["guidance", "awareness"]:
            self.find_handler.stop_find()
            self.world_state_mgr.set_find_target(None)
            self.world_state_mgr.set_mode(SystemMode.GUIDANCE)
            msg = "Switched to Guidance mode."
        elif m == "find":
            self.world_state_mgr.set_mode(SystemMode.FIND)
            msg = "Switched to Find mode."
        elif m == "read":
            self.world_state_mgr.set_mode(SystemMode.READ)
            msg = "Switched to Read mode."
        elif m == "ask":
            self.world_state_mgr.set_mode(SystemMode.ASK)
            msg = "Switched to Ask mode."
        else:
            return f"Unknown mode: {mode_name}"

        self.tts.purge_mode_speech("GUIDANCE" if m == "find" else "FIND")
        self.tts.speak(msg, priority=PriorityLevel.INTERACTION, source="SYSTEM")
        event_broker.publish(EventType.MODE_CHANGED_PUBLIC, {"mode": m.upper()})
        return msg

    def start_find(self, target_query: str) -> str:
        self.tts.stop()
        self.world_state_mgr.set_find_target(target_query)
        msg, session_id = self.find_handler.set_target(target_query)
        self.tts.speak(msg, priority=PriorityLevel.NAVIGATION, source="FIND")
        event_broker.publish(EventType.FIND_TARGET_UPDATED, {"target": target_query, "session_id": session_id})
        return msg

    def trigger_read(self, full_reading: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        frame = self.camera.get_latest_frame()
        if frame is None:
            frame = self.get_annotated_frame()
        capture_ms = (time.perf_counter() - t0) * 1000.0

        res = self.read_handler.trigger_read(frame, full_reading=full_reading)
        if "metrics" in res:
            res["metrics"]["ocr_capture_ms"] = round(capture_ms, 2)
            res["metrics"]["ocr_total_ms"] = round(res["metrics"].get("ocr_total_ms", 0.0) + capture_ms, 2)

        self.tts.speak(res["text"], priority=PriorityLevel.INTERACTION, source="READ")
        self.world_state_mgr.set_mode(SystemMode.GUIDANCE)
        return res

    def trigger_ask(self, user_query: str) -> Dict[str, Any]:
        frame = self.camera.get_latest_frame()
        if frame is None:
            frame = self.get_annotated_frame()
        res = self.ask_handler.ask(frame, user_query)
        self.tts.speak(res["text"], priority=PriorityLevel.INTERACTION, source="ASK")
        self.world_state_mgr.set_mode(SystemMode.GUIDANCE)
        return res

    def stop(self):
        self._running = False
        self._voice_running = False
        if self._perception_thread and self._perception_thread.is_alive():
            self._perception_thread.join(timeout=1.5)
        if self._voice_thread and self._voice_thread.is_alive():
            self._voice_thread.join(timeout=1.5)
        self.camera.stop()
        self.tts.stop()
        logger.info("VisionMate pipeline stopped.")


pipeline = VisionMatePipeline(camera_source=settings.CAMERA_SOURCE)
