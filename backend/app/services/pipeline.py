"""
VisionMate v2 - Master Pipeline Orchestrator (18 FPS Target & Voice Control)
Connects Camera -> YOLO -> Spatial Tracker -> Structured World State -> Priority/Event Engine -> Speech.
Executes multi-threaded, non-blocking asynchronous pipelines with generation invalidation and global STOP control.
"""

import time
import threading
import numpy as np
import cv2
from typing import Optional, Dict, Any, List, Tuple
import logging

from backend.app.core.config import settings
from backend.app.core.interfaces import CameraProvider, ObjectDetector, Tracker, OCRProvider, VisionReasoner, TTSProvider, ASRProvider
from backend.app.hardware.camera import create_camera_provider, SyntheticMockCamera
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
from backend.app.schemas.world_state import SystemMode, WorldState

logger = logging.getLogger("visionmate.pipeline")

class VisionMatePipeline:
    """Master VisionMate v2 asynchronous perceptual and reasoning pipeline running at 18 FPS."""
    def __init__(self, camera_source: str = "webcam", **camera_kwargs):
        self.camera_source_type = camera_source
        self.camera_kwargs = camera_kwargs

        # Instantiate Hardware & Providers
        # ESP32 config is sourced entirely from settings — no hardcoded URLs.
        self.camera: CameraProvider = create_camera_provider(camera_source, **camera_kwargs)
        self.detector: YOLOObjectDetector = YOLOObjectDetector(
            model_path=settings.YOLO_MODEL_PATH,
            device=settings.DETECTION_DEVICE,
            conf_thresh=settings.DETECTION_CONF_THRESHOLD,
            auto_load=False
        )
        self.tracker: Tracker = SpatialTracker(
            max_disappeared=settings.TRACKER_MAX_DISAPPEARED,
            iou_threshold=settings.TRACKER_IOU_THRESHOLD,
            min_stable_frames=3
        )
        self.world_state_mgr: WorldStateManager = world_state_mgr
        self.priority_engine = PriorityEngine(
            guidance_cooldown=settings.GUIDANCE_SPEECH_COOLDOWN_SEC,
            hazard_cooldown=settings.HAZARD_SPEECH_COOLDOWN_SEC
        )
        self.tts: WindowsSAPITTSProvider = tts_engine
        self.asr: ASRProvider = LocalASRProvider()
        self.ocr: OCRProvider = PPOCRv5Provider()
        self.vlm: VisionReasoner = OllamaQwen3VLReasoner()

        # Instantiate Mode Handlers
        self.guidance_handler = AwarenessModeHandler(self.priority_engine)
        self.find_handler = FindModeHandler(update_interval_sec=settings.FIND_MODE_UPDATE_INTERVAL_SEC)
        self.read_handler = ReadModeHandler(self.ocr)
        self.ask_handler = AskModeHandler(self.vlm)

        # Pipeline runtime state
        self._running = False
        self._perception_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Initialize default standby frame
        self._latest_annotated_frame = self._create_standby_frame("VisionMate v2 - Initializing 18 FPS pipeline...")
        
        # Performance Telemetry
        self.frames_processed = 0
        self.start_time = 0.0
        self.current_fps = 0.0
        self.last_latency_ms = 0.0
        self.last_yolo_latency_ms = 0.0

    def _create_standby_frame(self, message: str) -> np.ndarray:
        """Creates a clean standby canvas for HUD streaming before camera frames arrive."""
        img = np.full((480, 640, 3), 20, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (640, 40), (35, 39, 46), -1)
        cv2.putText(img, "VISIONMATE v2 | GUIDANCE HUD (18 FPS)", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(img, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    def start(self) -> bool:
        """Starts camera ingestion and background perception loop asynchronously at 18 FPS."""
        if self._running:
            return True

        self._running = True
        self.start_time = time.time()
        self._perception_thread = threading.Thread(target=self._async_bootstrap_and_loop, daemon=True)
        self._perception_thread.start()
        logger.info(f"VisionMate v2 Pipeline background worker spawned (Target: {settings.VISIONMATE_TARGET_FPS} FPS).")
        return True

    def _async_bootstrap_and_loop(self):
        """Asynchronously starts ESP32-CAM snapshot acquisition, loads YOLO model, and runs perception loop."""
        logger.info("Bootstrapping hardware & model subsystems...")

        # 1. Start Primary Camera Provider (ESP32-CAM /capture)
        logger.info("Primary camera: ESP32-CAM")
        logger.info(f"Snapshot endpoint: {settings.ESP32_CAPTURE_URL}")
        try:
            self.camera.start()
            if not self.camera.is_connected():
                logger.warning(
                    f"ESP32 camera unavailable at {settings.ESP32_CAPTURE_URL}. "
                    "Background reconnect worker active..."
                )
        except Exception as e:
            logger.warning(f"ESP32 camera startup notice: {e}. Background reconnect active...")

        # 2. Asynchronously load YOLO model
        try:
            self.detector.load()
        except Exception as e:
            logger.error(f"Error loading YOLO detector: {e}")

        # 3. Speak welcome
        try:
            self.tts.speak("VisionMate ready. Guidance mode active.", priority=PriorityLevel.INTERACTION, source="GUIDANCE")
        except Exception as e:
            logger.warning(f"TTS greeting skipped: {e}")

        # 4. Continuous Fast Perception Loop target: 18 FPS (~55.6 ms interval)
        target_period = 1.0 / float(settings.VISIONMATE_TARGET_FPS)
        last_time = time.perf_counter()

        while self._running:
            t0 = time.perf_counter()
            # Non-blocking get latest fresh frame (discards stale intermediate frames)
            frame = self.camera.get_latest_frame()

            if frame is not None:
                h, w = frame.shape[:2]
                
                # 1. Fast YOLO Detection
                detections = []
                t_yolo_start = time.perf_counter()
                if self.detector.is_ready():
                    detections = self.detector.detect(frame)
                self.last_yolo_latency_ms = (time.perf_counter() - t_yolo_start) * 1000.0

                # 2. Multi-object tracking with temporal confirmation
                active_tracks = self.tracker.update(detections, (h, w))

                # 3. Update central World State
                world_state = self.world_state_mgr.update_from_tracks(active_tracks, fps=self.current_fps)

                # 4. Mode-specific dispatch & Priority arbitration
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

                # 5. Dispatch speech if event generated
                if speech_cmd and "text" in speech_cmd:
                    txt = speech_cmd["text"]
                    pri = speech_cmd.get("priority", PriorityLevel.INTERACTION)
                    inter = speech_cmd.get("interrupt", False)
                    src = speech_cmd.get("source", "GUIDANCE")
                    self.tts.speak(txt, priority=pri, interrupt=inter, source=src)
                    self.world_state_mgr.record_speech(txt)

                # 6. Save annotated frame for Dashboard visualization
                self._draw_hud_annotations(frame, world_state)

                # Calculate metrics
                self.frames_processed += 1
                self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            else:
                with self._lock:
                    self._latest_annotated_frame = self._create_standby_frame(
                        f"Waiting for ESP32-CAM snapshots ({settings.ESP32_CAPTURE_URL})..."
                    )

            dt = time.perf_counter() - last_time
            if dt > 0.5:
                self.current_fps = round(self.frames_processed / max(1e-3, (time.time() - self.start_time)), 1)

            # Throttle to exact 18 FPS target period (~55.6 ms)
            elapsed = time.perf_counter() - t0
            sleep_time = target_period - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def _draw_hud_annotations(self, frame: np.ndarray, world_state: WorldState):
        """Draws spatial bounding boxes, path zone corridor, and mode HUD."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw forward walking corridor boundaries (center 34% - 66%)
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

        # HUD Banner
        det_status = "OK" if self.detector.is_ready() else self.detector.status
        mode_name = world_state.current_mode.value.upper()
        mode_str = f"MODE: {mode_name} | FPS: {self.current_fps:.1f}/18 | Lat: {self.last_latency_ms:.1f}ms | Det: {det_status}"
        cv2.rectangle(annotated, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(annotated, mode_str, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 2, cv2.LINE_AA)

        with self._lock:
            self._latest_annotated_frame = annotated

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated_frame is not None:
                return self._latest_annotated_frame.copy()
            return None

    def get_subsystems_health(self) -> Dict[str, Any]:
        """Returns granular health status for all subcomponents including full camera diagnostics."""
        # Pull diagnostics dict if provider supports it (ESP32 + webcam), else minimal dict
        cam_diag = {}
        if hasattr(self.camera, "get_diagnostics"):
            cam_diag = self.camera.get_diagnostics()
        else:
            cam_diag = {
                "source": self.camera_source_type,
                "connected": self.camera.is_connected(),
                "connection_state": "ONLINE" if self.camera.is_connected() else "OFFLINE",
            }

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
                "ready": self.detector.is_ready()
            },
            "speech_engine": {
                "status": "READY" if self.tts is not None else "UNAVAILABLE",
                "speaking": self.tts.is_speaking() if self.tts else False,
                "speech_generation": self.tts.speech_generation
            },
            "ocr": {
                "status": "ON_DEMAND",
                "engine_loaded": self.ocr.ocr_engine is not None
            },
            "vlm": {
                "model": settings.OLLAMA_VLM_MODEL,
                "status": "ON_DEMAND"
            },
            "target_fps": settings.VISIONMATE_TARGET_FPS,
            "current_fps": self.current_fps
        }

    # Voice Command & Interactive Actions
    def handle_voice_command(self, transcript: str) -> Dict[str, Any]:
        """Processes voice command and routes to the appropriate mode or cancellation action."""
        intent_res = self.asr.parse_intent(transcript)
        intent = intent_res.get("intent")
        
        logger.info(f"Voice Command Intent: {intent} from '{transcript}'")

        if intent == "STOP":
            return self.handle_stop_command()
        elif intent == "FIND":
            target = intent_res.get("target", "object")
            msg = self.start_find(target)
            return {"status": "SUCCESS", "mode": "find", "message": msg, "target": target}
        elif intent == "READ":
            is_full = intent_res.get("full_reading", False)
            res = self.trigger_read(full_reading=is_full)
            return {"status": "SUCCESS", "mode": "read", "result": res}
        elif intent == "ASK":
            query = intent_res.get("query", transcript)
            res = self.trigger_ask(query)
            return {"status": "SUCCESS", "mode": "ask", "result": res}
        elif intent == "GUIDANCE":
            msg = self.set_mode("guidance")
            return {"status": "SUCCESS", "mode": "guidance", "message": msg}
        else:
            return {"status": "UNRECOGNIZED", "intent": intent, "raw": transcript}

    def handle_stop_command(self) -> Dict[str, Any]:
        """
        Global STOP Command:
        1. Exits Find Mode and cancels Find target.
        2. Halts current TTS and purges all queued speech via generation invalidation.
        3. Returns mode to GUIDANCE.
        4. Enforces silence until new event occurs.
        """
        # 1. Purge & Halt Speech
        self.tts.stop()
        
        # 2. Deactivate Find Mode
        self.find_handler.stop_find()
        
        # 3. Restore Guidance Mode in World State
        self.world_state_mgr.set_find_target(None)
        self.world_state_mgr.set_mode(SystemMode.GUIDANCE)
        
        # 4. Suppress immediate chatter by setting speech cooldown
        self.priority_engine.last_speech_time = time.time()

        logger.info("[GLOBAL STOP] All pending speech purged, Find Mode terminated, Guidance mode active in silence.")
        return {
            "status": "STOPPED",
            "mode": "guidance",
            "message": "System stopped. Guidance active.",
            "speech_generation": self.tts.speech_generation
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
            msg = f"Unknown mode: {mode_name}"
            return msg

        # Invalidate old mode speech
        self.tts.purge_mode_speech("GUIDANCE" if m == "find" else "FIND")
        self.tts.speak(msg, priority=PriorityLevel.INTERACTION, source="SYSTEM")
        return msg

    def start_find(self, target_query: str) -> str:
        """
        Starts Find Mode for a specific target:
        Cancels previous normal/low speech, invalidates prior sessions, starts target tracking.
        """
        # 1. Invalidate previous normal speech
        self.tts.stop()
        
        # 2. Set Find target in handler and World State
        self.world_state_mgr.set_find_target(target_query)
        msg, session_id = self.find_handler.set_target(target_query)
        
        # 3. Speak announcement under new speech generation
        self.tts.speak(msg, priority=PriorityLevel.NAVIGATION, source="FIND")
        return msg

    def trigger_read(self, full_reading: bool = False) -> Dict[str, Any]:
        """
        Executes on-demand OCR reading on current fresh ESP32-CAM frame.
        Automatically turns OCR OFF and resumes Guidance Mode upon completion.
        """
        t0 = time.perf_counter()
        frame = self.camera.get_latest_frame()
        if frame is None:
            frame = self.get_annotated_frame()
        capture_ms = (time.perf_counter() - t0) * 1000.0

        # Run OCR on current frame
        res = self.read_handler.trigger_read(frame, full_reading=full_reading)
        if "metrics" in res:
            res["metrics"]["ocr_capture_ms"] = round(capture_ms, 2)
            res["metrics"]["ocr_total_ms"] = round(res["metrics"].get("ocr_total_ms", 0.0) + capture_ms, 2)

        self.tts.speak(res["text"], priority=PriorityLevel.INTERACTION, source="READ")
        
        # Automatically resume Guidance Mode
        self.world_state_mgr.set_mode(SystemMode.GUIDANCE)
        return res

    def trigger_ask(self, user_query: str) -> Dict[str, Any]:
        """Triggers on-demand Visual Q&A in Ask Mode with comprehensive scene reasoning."""
        frame = self.camera.get_latest_frame()
        if frame is None:
            frame = self.get_annotated_frame()
        res = self.ask_handler.ask(frame, user_query)
        self.tts.speak(res["text"], priority=PriorityLevel.INTERACTION, source="ASK")
        
        # Automatically resume Guidance Mode
        self.world_state_mgr.set_mode(SystemMode.GUIDANCE)
        return res

    def stop(self):
        self._running = False
        if self._perception_thread and self._perception_thread.is_alive():
            self._perception_thread.join(timeout=1.5)
        self.camera.stop()
        self.tts.stop()
        logger.info("VisionMate pipeline stopped.")

# Global Pipeline Singleton
pipeline = VisionMatePipeline(camera_source=settings.CAMERA_SOURCE)
