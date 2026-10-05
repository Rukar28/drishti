"""
VisionMate v2 - Global Pipeline Orchestrator

Main architecture:

    Camera
        ↓
    YOLO
        ↓
    Tracker
        ↓
    World State
        ↓
    Priority
        ↓
    Modes
        ↓
    Command Bus
        ↓
    TTS

Voice architecture:

    Microphone
        ↓
    Real Wake Word / Development Fallback
        ↓
    VAD
        ↓
    Faster-Whisper ASR
        ↓
    Intent
        ↓
    Command Bus
        ↓
    Features
        ↓
    TTS
"""

from __future__ import annotations

import os
import time
import threading
from typing import Optional, Dict, Any, List

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

from backend.app.core.events import (
    EventType,
    event_broker,
)

from backend.app.hardware.camera import (
    create_camera_provider,
)

from backend.app.hardware.depth import (
    create_depth_provider,
)

from backend.app.hardware.gps import (
    create_gps_provider,
)

from backend.app.perception.detector import (
    YOLOObjectDetector,
)

from backend.app.tracking.tracker import (
    SpatialTracker,
)

from backend.app.world_state.manager import (
    world_state_mgr,
    WorldStateManager,
)

from backend.app.priority.engine import (
    PriorityEngine,
    PriorityLevel,
)

from backend.app.speech.tts import (
    tts_engine,
    WindowsSAPITTSProvider,
)

from backend.app.speech.asr import (
    LocalASRProvider,
)

from backend.app.reasoning.ocr import (
    PPOCRv5Provider,
)

from backend.app.reasoning.vlm import (
    OllamaQwen3VLReasoner,
)

from backend.app.modes.awareness import (
    AwarenessModeHandler,
)

from backend.app.modes.find import (
    FindModeHandler,
)

from backend.app.modes.read import (
    ReadModeHandler,
)

from backend.app.modes.ask import (
    AskModeHandler,
)

from backend.app.modes.color import (
    ColorModeHandler,
)

from backend.app.intelligence.command_bus import (
    CommandBus,
)

from backend.app.speech.wakeword import (
    WakeWordProvider,
    wake_word_provider as _default_wake_provider,
)

from backend.app.speech.voice_state import (
    VoiceState,
    VoiceStateMachine,
)

from backend.app.reasoning.currency import (
    CurrencyRecognizer,
)

from backend.app.reasoning.medicine import (
    MedicineRecognizer,
)

from backend.app.navigation.navigator import (
    Navigator,
)

from backend.app.services.sos import (
    sos_service,
)

from backend.app.schemas.world_state import (
    SystemMode,
    WorldState,
)


logger = logging.getLogger(
    "visionmate.pipeline"
)


class VisionMatePipeline:
    """
    Master perceptual, voice, and reasoning pipeline.

    The pipeline owns:

    - camera
    - depth
    - GPS
    - YOLO
    - tracker
    - World State
    - priority engine
    - TTS
    - ASR
    - OCR
    - VLM
    - Command Bus
    - wake-word provider
    - voice state machine

    The pipeline does NOT implement wake-word detection itself.
    That responsibility belongs to WakeWordProvider.
    """

    def __init__(
        self,
        camera_source: str = "webcam",
        wake_provider: Optional[
            WakeWordProvider
        ] = None,
        **camera_kwargs,
    ):

        self.camera_source_type = (
            camera_source
        )

        self.camera_kwargs = (
            camera_kwargs
        )

        # =====================================================================
        # HARDWARE / PERCEPTION
        # =====================================================================

        self.camera: CameraProvider = (
            create_camera_provider(
                camera_source,
                **camera_kwargs,
            )
        )

        self.depth: DepthProvider = (
            create_depth_provider(
                settings.TOF_SOURCE
            )
        )

        self.gps: GPSProvider = (
            create_gps_provider(
                settings.GPS_SOURCE
            )
        )

        self.detector: YOLOObjectDetector = (
            YOLOObjectDetector(
                model_path=settings.YOLO_MODEL_PATH,
                device=settings.DETECTION_DEVICE,
                conf_thresh=settings.DETECTION_CONF_THRESHOLD,
                auto_load=False,
            )
        )

        self.tracker: Tracker = (
            SpatialTracker(
                max_disappeared=(
                    settings.TRACKER_MAX_DISAPPEARED
                ),
                iou_threshold=(
                    settings.TRACKER_IOU_THRESHOLD
                ),
                min_stable_frames=3,
            )
        )

        self.world_state_mgr: WorldStateManager = (
            world_state_mgr
        )

        self.priority_engine = (
            PriorityEngine(
                guidance_cooldown=(
                    settings.GUIDANCE_SPEECH_COOLDOWN_SEC
                ),
                hazard_cooldown=(
                    settings.HAZARD_SPEECH_COOLDOWN_SEC
                ),
            )
        )

        # =====================================================================
        # VOICE / REASONING
        # =====================================================================

        self.tts: WindowsSAPITTSProvider = (
            tts_engine
        )

        self.asr: ASRProvider = (
            LocalASRProvider(
                model_size=settings.ASR_MODEL,
                device=settings.ASR_DEVICE,
                compute_type=settings.ASR_COMPUTE_TYPE,
                sample_rate=settings.ASR_SAMPLE_RATE,
            )
        )

        self.ocr: OCRProvider = (
            PPOCRv5Provider()
        )

        self.vlm: VisionReasoner = (
            OllamaQwen3VLReasoner()
        )

        # =====================================================================
        # MODES
        # =====================================================================

        self.guidance_handler = (
            AwarenessModeHandler(
                self.priority_engine
            )
        )

        self.find_handler = (
            FindModeHandler(
                update_interval_sec=(
                    settings.FIND_MODE_UPDATE_INTERVAL_SEC
                )
            )
        )

        self.read_handler = (
            ReadModeHandler(
                self.ocr
            )
        )

        self.ask_handler = (
            AskModeHandler(
                self.vlm
            )
        )

        # =====================================================================
        # ON-DEMAND FEATURES
        # =====================================================================

        self.currency_recognizer = (
            CurrencyRecognizer(
                self.ocr
            )
        )

        self.medicine_recognizer = (
            MedicineRecognizer(
                self.ocr
            )
        )

        self.color_handler = ColorModeHandler()

        self.navigator = Navigator()

        # =====================================================================
        # COMMAND BUS
        # =====================================================================

        self.command_bus = (
            CommandBus(
                self.asr.parse_intent
            )
        )

        self._register_command_handlers()

        # =====================================================================
        # RUNTIME STATE
        # =====================================================================

        self._running = False
        self._navigation_thread = None
        self._navigation_stop = threading.Event()
        self._navigation_lock = threading.Lock()

        self._perception_thread: Optional[
            threading.Thread
        ] = None

        self._voice_thread: Optional[
            threading.Thread
        ] = None

        self._voice_running = False

        self.voice_state = "IDLE"

        # =====================================================================
        # REAL WAKE WORD PROVIDER
        # =====================================================================

        self.wake_provider: WakeWordProvider = (
            wake_provider
            or _default_wake_provider
        )

        self.voice_fsm = (
            VoiceStateMachine(
                on_change=(
                    self._on_voice_state_change
                )
            )
        )

        self._lock = threading.Lock()

        # =====================================================================
        # FRAME / PERFORMANCE STATE
        # =====================================================================

        self._latest_annotated_frame = (
            self._create_standby_frame(
                "VisionMate v2 - Initializing 18 FPS pipeline..."
            )
        )

        self.frames_processed = 0

        self.start_time = 0.0

        self.current_fps = 0.0

        self.last_latency_ms = 0.0

        self.last_yolo_latency_ms = 0.0

    # =========================================================================
    # COMMAND BUS
    # =========================================================================

    def _register_command_handlers(
        self,
    ) -> None:

        self.command_bus.register(
            "STOP",
            lambda parsed:
                self.handle_stop_command(),
        )

        self.command_bus.register(
            "FIND",
            self._cmd_find,
        )

        self.command_bus.register(
            "READ",
            self._cmd_read,
        )

        self.command_bus.register(
            "ASK",
            self._cmd_ask,
        )

        self.command_bus.register(
            "GUIDANCE",
            self._cmd_guidance,
        )

        self.command_bus.register(
            "COLOR",
            self._cmd_color,
        )

        self.command_bus.register(
            "MEDICINE",
            self._cmd_medicine,
        )

        self.command_bus.register(
            "CURRENCY",
            self._cmd_currency,
        )

        self.command_bus.register(
            "FACE",
            self._cmd_not_available,
        )

        self.command_bus.register(
            "NAVIGATION",
            self._cmd_navigation,
        )

        self.command_bus.register(
            "SOS",
            self._cmd_sos,
        )

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    def _cmd_find(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        target = (
            parsed.get("target")
            or "object"
        )

        msg = self.start_find(
            target
        )

        return {
            "status": "started",
            "mode": "find",
            "message": msg,
            "target": target,
        }

    def _cmd_read(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        is_full = parsed.get(
            "full_reading",
            False,
        )

        res = self.trigger_read(
            full_reading=is_full
        )

        return {
            "status": "started",
            "mode": "read",
            "result": res,
        }

    def _cmd_ask(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        query = (
            parsed.get("query")
            or parsed.get("raw_text")
            or ""
        )

        res = self.trigger_ask(
            query
        )

        return {
            "status": "started",
            "mode": "ask",
            "result": res,
        }

    def _cmd_guidance(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        msg = self.set_mode(
            "guidance"
        )

        return {
            "status": "started",
            "mode": "guidance",
            "message": msg,
        }

    def _cmd_color(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        target = parsed.get("target")

        res = self.trigger_color(
            target=target
        )

        return {
            "status": "ok" if res.get("success") else "failed",
            "mode": "color",
            "message": res.get("text", ""),
            "result": res,
        }

    def _cmd_not_available(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        intent = parsed.get(
            "intent",
            "FEATURE",
        )

        return {
            "status": "not_available",
            "message": (
                f"{intent} is not enabled "
                "in this backend phase."
            ),
        }

    def _cmd_currency(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        res = self.trigger_currency()

        return {
            "status": "started",
            "mode": "currency",
            "result": res,
        }

    def _cmd_medicine(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        res = self.trigger_medicine()

        return {
            "status": "started",
            "mode": "medicine",
            "result": res,
        }

    def _cmd_navigation(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        raw = (
            parsed.get("raw_text")
            or ""
        ).lower()

        destination = parsed.get(
            "destination"
        )

        if (
            "stop navigation"
            in raw
            or
            "cancel navigation"
            in raw
        ):

            return self.stop_navigation()

        if (
            "where am i"
            in raw
            or
            "where i am"
            in raw
        ):

            loc = (
                self.gps.get_location()
            )

            origin = (
                loc.get("lat"),
                loc.get("lon"),
            )

            text = (
                self.navigator.where_am_i(
                    origin
                )
            )

            self.tts.speak(
                text,
                priority=(
                    PriorityLevel.INTERACTION
                ),
                source="NAVIGATION",
            )

            return {
                "status": "ok",
                "mode": "navigation",
                "message": text,
            }

        if destination:

            res = self.start_navigation(
                destination
            )

            return {
                "status": res.get(
                    "status",
                    "ok",
                ).lower(),
                "mode": "navigation",
                "result": res,
            }

        text = (
            "Please tell me where to go. "
            "For example, navigate to home."
        )

        self.tts.speak(
            text,
            priority=(
                PriorityLevel.INTERACTION
            ),
            source="NAVIGATION",
        )

        return {
            "status": "needs_destination",
            "mode": "navigation",
            "message": text,
        }

    def _cmd_sos(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:

        sos_service.location_provider = self.gps
        result = sos_service.send_emergency_alert()
        sos_service.last_result = {**result, "location": self.gps.get_location(), "event_time": time.time()}

        if result["success"]:
            text = "Emergency alert sent."
            event_broker.publish(
                EventType.SOS_ALERT,
                {
                    "status": "sent",
                    "timestamp": result.get("timestamp"),
                },
            )
        else:
            status = result.get("status", "failed")
            if status == "not_configured":
                text = "Emergency notifications are not configured."
            elif status == "cooldown":
                text = "Emergency alert already sent recently."
            else:
                text = "I couldn't send the emergency alert."

            event_broker.publish(
                EventType.SOS_ALERT,
                {
                    "status": status,
                    "error": result.get("error"),
                },
            )

        self.tts.speak(
            text,
            priority=PriorityLevel.EMERGENCY,
            interrupt=True,
            source="SOS",
        )

        return {
            "status": result.get("status", "failed"),
            "mode": "sos",
            "success": result.get("success", False),
            "message": text,
        }

    # =========================================================================
    # VOICE STATE
    # =========================================================================

    def _on_voice_state_change(
        self,
        state: VoiceState,
    ) -> None:

        self.voice_state = (
            state.value
        )

        event_broker.publish(
            EventType.VOICE_STATE,
            {
                "state": state.value
            },
        )

    def _set_voice_state(
        self,
        state,
    ) -> None:

        """
        Back-compat setter.

        Prefer voice_fsm.transition()
        inside the voice loop.
        """

        try:

            vs = (
                state
                if isinstance(
                    state,
                    VoiceState,
                )
                else VoiceState(
                    str(state).upper()
                )
            )

        except ValueError:

            vs = VoiceState.IDLE

        self.voice_fsm.force(
            vs,
            reason="set_voice_state",
        )

    # =========================================================================
    # STANDBY FRAME
    # =========================================================================

    def _create_standby_frame(
        self,
        message: str,
    ) -> np.ndarray:

        img = np.full(
            (480, 640, 3),
            20,
            dtype=np.uint8,
        )

        cv2.rectangle(
            img,
            (0, 0),
            (640, 40),
            (35, 39, 46),
            -1,
        )

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

        cv2.putText(
            img,
            message,
            (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return img

    # =========================================================================
    # START
    # =========================================================================

    def start(self) -> bool:

        if self._running:
            return True

        self._running = True

        self.start_time = time.time()

        # ---------------------------------------------------------------------
        # Perception worker
        # ---------------------------------------------------------------------

        self._perception_thread = (
            threading.Thread(
                target=(
                    self._async_bootstrap_and_loop
                ),
                daemon=True,
                name="VisionMatePerception",
            )
        )

        self._perception_thread.start()
        self._navigation_stop.clear()
        self._navigation_thread = threading.Thread(target=self._navigation_loop, daemon=True, name="VisionMateNavigation")
        self._navigation_thread.start()

        # ---------------------------------------------------------------------
        # Voice worker
        # ---------------------------------------------------------------------

        voice_enabled = (
            settings.VOICE_LISTENER_ENABLED
            and not os.environ.get(
                "PYTEST_CURRENT_TEST"
            )
        )

        if voice_enabled:

            # IMPORTANT:
            #
            # wake_provider.start() returns False when the real
            # acoustic detector could not initialize.
            #
            # Do NOT silently pretend that wake-word detection
            # is working.
            try:

                wake_started = (
                    self.wake_provider.start()
                )

            except Exception as exc:

                wake_started = False

                logger.exception(
                    "[WAKE] Wake provider initialization "
                    "raised an exception: %s",
                    exc,
                )

            wake_health = (
                self.wake_provider.health()
            )

            if wake_started:

                logger.info(
                    "[WAKE] Provider ready | "
                    "provider=%s | "
                    "mode=%s | "
                    "real=%s | "
                    "phrase=%s",
                    wake_health.get(
                        "provider"
                    ),
                    wake_health.get(
                        "mode"
                    ),
                    wake_health.get(
                        "is_real_wake_word"
                    ),
                    wake_health.get(
                        "phrase"
                    ),
                )

                if (
                    self.wake_provider
                    .is_real_wake_word
                ):

                    logger.info(
                        "[WAKE] REAL_WAKE_WORD ACTIVE: "
                        "Hey Mycroft"
                    )

                else:

                    logger.warning(
                        "[WAKE] Voice provider is "
                        "NOT a real acoustic wake word: "
                        "%s",
                        wake_health.get(
                            "mode"
                        ),
                    )

                self._voice_running = True

                self._voice_thread = (
                    threading.Thread(
                        target=self._voice_loop,
                        daemon=True,
                        name=(
                            "VisionMateVoiceListener"
                        ),
                    )
                )

                self._voice_thread.start()

            else:

                # Real mode should fail visibly instead of
                # starting a useless voice loop.
                self._voice_running = False

                logger.error(
                    "[WAKE] Wake provider failed to start. "
                    "Voice listener will NOT start."
                )

                logger.error(
                    "[WAKE] Health: %s",
                    wake_health,
                )

        else:

            logger.info(
                "[VOICE] Voice listener disabled."
            )

        logger.info(
            "[VISION] Pipeline worker spawned "
            "(target %s FPS, camera=%s).",
            settings.VISIONMATE_TARGET_FPS,
            self.camera_source_type,
        )

        return True

    # =========================================================================
    # PERCEPTION LOOP
    # =========================================================================

    def _async_bootstrap_and_loop(
        self,
    ):

        logger.info(
            "[VISION] Bootstrapping hardware "
            "& model subsystems..."
        )

        logger.info(
            "[VISION] Primary camera: %s",
            self.camera_source_type,
        )

        try:

            self.camera.start()

            if not self.camera.is_connected():

                logger.warning(
                    "[VISION] Camera unavailable (%s). "
                    "Background reconnect active.",
                    self.camera_source_type,
                )

        except Exception as e:

            logger.warning(
                "[VISION] Camera startup notice: "
                "%s. Background reconnect active.",
                e,
            )

        try:

            self.detector.load()

        except Exception as e:

            logger.error(
                "[VISION] Error loading YOLO detector: %s",
                e,
            )

        try:

            self.tts.speak(
                "VisionMate ready. Guidance mode active.",
                priority=(
                    PriorityLevel.INTERACTION
                ),
                source="GUIDANCE",
            )

        except Exception as e:

            logger.warning(
                "[TTS] Greeting skipped: %s",
                e,
            )

        target_period = (
            1.0
            / float(
                settings.VISIONMATE_TARGET_FPS
            )
        )

        last_time = time.perf_counter()

        while self._running:

            t0 = time.perf_counter()

            frame = (
                self.camera.get_latest_frame()
            )

            if frame is not None:

                h, w = frame.shape[:2]

                detections = []

                # -------------------------------------------------------------
                # YOLO
                # -------------------------------------------------------------

                t_yolo_start = (
                    time.perf_counter()
                )

                if self.detector.is_ready():

                    detections = (
                        self.detector.detect(
                            frame
                        )
                    )

                self.last_yolo_latency_ms = (
                    time.perf_counter()
                    - t_yolo_start
                ) * 1000.0

                # -------------------------------------------------------------
                # TRACKING
                # -------------------------------------------------------------

                active_tracks = (
                    self.tracker.update(
                        detections,
                        (h, w),
                    )
                )

                # -------------------------------------------------------------
                # WORLD STATE
                # -------------------------------------------------------------

                world_state = (
                    self.world_state_mgr.update_from_tracks(
                        active_tracks,
                        fps=self.current_fps,
                    )
                )

                current_mode = (
                    world_state.current_mode
                )

                speech_cmd = None

                # -------------------------------------------------------------
                # DEPTH / TOF HAZARD
                # -------------------------------------------------------------

                depth_hazard = None

                try:

                    depth_hazard = (
                        self.priority_engine.evaluate_depth_hazard(
                            self.depth.get_zones()
                        )
                    )

                except Exception as e:

                    logger.debug(
                        "[DEPTH] hazard evaluation error: %s",
                        e,
                    )

                # -------------------------------------------------------------
                # PRIORITY
                # -------------------------------------------------------------

                if depth_hazard is not None:

                    speech_cmd = (
                        depth_hazard
                    )

                elif current_mode in [
                    SystemMode.GUIDANCE,
                    SystemMode.AWARENESS,
                ]:

                    speech_cmd = (
                        self.guidance_handler.process_frame_state(
                            world_state
                        )
                    )

                elif current_mode == SystemMode.FIND:

                    hazard_event = (
                        self.priority_engine.evaluate_hazard(
                            world_state
                        )
                    )

                    if hazard_event:

                        speech_cmd = (
                            hazard_event
                        )

                    else:

                        speech_cmd = (
                            self.find_handler.process_frame_state(
                                world_state
                            )
                        )

                # -------------------------------------------------------------
                # SPEECH
                # -------------------------------------------------------------

                if (
                    speech_cmd
                    and "text" in speech_cmd
                ):

                    txt = speech_cmd["text"]

                    pri = speech_cmd.get(
                        "priority",
                        PriorityLevel.INTERACTION,
                    )

                    inter = speech_cmd.get(
                        "interrupt",
                        False,
                    )

                    src = speech_cmd.get(
                        "source",
                        "GUIDANCE",
                    )

                    self.tts.speak(
                        txt,
                        priority=pri,
                        interrupt=inter,
                        source=src,
                    )

                    self.world_state_mgr.record_speech(
                        txt
                    )

                    self._publish_hazard_if_any(
                        speech_cmd,
                        pri,
                        src,
                    )

                # -------------------------------------------------------------
                # HUD
                # -------------------------------------------------------------

                self._draw_hud_annotations(
                    frame,
                    world_state,
                )

                self.frames_processed += 1

                self.last_latency_ms = (
                    time.perf_counter()
                    - t0
                ) * 1000.0

            else:

                with self._lock:

                    self._latest_annotated_frame = (
                        self._create_standby_frame(
                            f"Waiting for camera "
                            f"({self.camera_source_type})..."
                        )
                    )

            # -------------------------------------------------------------
            # FPS
            # -------------------------------------------------------------

            dt = (
                time.perf_counter()
                - last_time
            )

            if dt > 0.5:

                self.current_fps = round(
                    self.frames_processed
                    / max(
                        1e-3,
                        (
                            time.time()
                            - self.start_time
                        ),
                    ),
                    1,
                )

                last_time = (
                    time.perf_counter()
                )

            # -------------------------------------------------------------
            # TARGET FRAME PERIOD
            # -------------------------------------------------------------

            elapsed = (
                time.perf_counter()
                - t0
            )

            sleep_time = (
                target_period
                - elapsed
            )

            if sleep_time > 0.001:

                time.sleep(
                    sleep_time
                )

    # =========================================================================
    # VOICE LOOP
    # =========================================================================

    def _voice_loop(self):

        wake_mode = (
            "REAL_WAKE_WORD"
            if self.wake_provider.is_real_wake_word
            else self.wake_provider.mode
        )

        logger.info(
            "[VOICE] Wake-word subsystem: "
            "%s (%s). "
            "Adaptive VAD capture, not fixed 3s chunks.",
            self.wake_provider.name,
            wake_mode,
        )

        # ---------------------------------------------------------------------
        # Explicitly report REAL vs fallback.
        # ---------------------------------------------------------------------

        if (
            self.wake_provider.is_real_wake_word
        ):

            logger.info(
                "[VOICE] REAL acoustic wake-word "
                "detection is ACTIVE."
            )

            logger.info(
                "[VOICE] Wake phrase: %s",
                settings.WAKE_WORD_PHRASE,
            )

            logger.info(
                "[VOICE] Wake model: %s",
                settings.WAKE_WORD_PRETRAINED,
            )

        else:

            logger.warning(
                "[VOICE] DEVELOPMENT / NON-REAL "
                "wake-word provider active: %s",
                self.wake_provider.mode,
            )

        # ---------------------------------------------------------------------
        # Preload ASR once.
        # ---------------------------------------------------------------------

        try:

            self.asr.preload()

        except Exception as e:

            logger.error(
                "[VOICE] ASR preload failed: %s",
                e,
            )

        # ---------------------------------------------------------------------
        # Main voice lifecycle.
        # ---------------------------------------------------------------------

        while self._voice_running:

            try:

                # -------------------------------------------------------------
                # WAITING FOR WAKE
                # -------------------------------------------------------------

                self.voice_fsm.force(
                    VoiceState.IDLE,
                    "loop-start",
                )

                self.voice_fsm.transition(
                    VoiceState.LISTENING_FOR_WAKE,
                    "awaiting wake",
                )

                wake_detected = (
                    self.wake_provider.wait_for_wake(
                        stop_flag=(
                            lambda:
                            not self._voice_running
                        ),
                        timeout=1.0,
                    )
                )

                if not wake_detected:

                    continue

                # -------------------------------------------------------------
                # WAKE DETECTED
                # -------------------------------------------------------------

                gen_at_wake = (
                    self.voice_fsm.generation
                )

                self.voice_fsm.transition(
                    VoiceState.WAKE_DETECTED,
                    "wake detected",
                )

                event_broker.publish(
                    EventType.VOICE_STARTED,
                    {
                        "state": "WAKE_DETECTED",
                        "wake_mode": wake_mode,
                        "is_real_wake_word": (
                            self.wake_provider
                            .is_real_wake_word
                        ),
                    },
                )

                # -------------------------------------------------------------
                # Optional acknowledgement
                # -------------------------------------------------------------

                if settings.WAKE_WORD_ACK_TEXT:

                    self.tts.speak(
                        settings.WAKE_WORD_ACK_TEXT,
                        priority=(
                            PriorityLevel.USER_COMMAND
                        ),
                        source="VOICE",
                    )

                # -------------------------------------------------------------
                # CAPTURE COMMAND
                #
                # Wake-word provider releases the microphone after detection.
                # ASR/VAD can now acquire it.
                # -------------------------------------------------------------

                self.voice_fsm.transition(
                    VoiceState.LISTENING_FOR_COMMAND,
                    "capture command",
                )

                transcript = (
                    self.asr.listen_utterance(
                        stop_flag=(
                            lambda:
                            not self._voice_running
                        )
                    )
                )

                if not transcript:

                    self.voice_fsm.transition(
                        VoiceState.IDLE,
                        "no utterance",
                    )

                    continue

                transcript = (
                    transcript.strip()
                )

                # -------------------------------------------------------------
                # STOP / GENERATION SAFETY
                # -------------------------------------------------------------

                if (
                    self.voice_fsm.generation
                    != gen_at_wake
                ):

                    logger.info(
                        "[VOICE] Discarding stale "
                        "command captured before STOP."
                    )

                    continue

                logger.info(
                    "[VOICE] Heard: %s",
                    transcript,
                )

                # -------------------------------------------------------------
                # PROCESSING
                # -------------------------------------------------------------

                self.voice_fsm.transition(
                    VoiceState.PROCESSING,
                    "ASR complete",
                )

                event_broker.publish(
                    EventType.VOICE_TRANSCRIPT,
                    {
                        "transcript": transcript
                    },
                )

                # -------------------------------------------------------------
                # COMMAND BUS
                # -------------------------------------------------------------

                self.voice_fsm.transition(
                    VoiceState.EXECUTING,
                    "command bus dispatch",
                )

                result = (
                    self.handle_voice_command(
                        transcript
                    )
                )

                logger.info(
                    "[VOICE] Result: %s",
                    result,
                )

                # -------------------------------------------------------------
                # SPEAKING / RETURN TO IDLE
                # -------------------------------------------------------------

                if self.tts.is_speaking():

                    self.voice_fsm.transition(
                        VoiceState.SPEAKING,
                        "tts active",
                    )

                elif (
                    self.voice_fsm.state
                    != VoiceState.IDLE
                ):

                    self.voice_fsm.transition(
                        VoiceState.IDLE,
                        "session complete",
                    )

            except Exception as e:

                logger.error(
                    "[VOICE] Listener error: %s",
                    e,
                    exc_info=True,
                )

                event_broker.publish(
                    EventType.ERROR,
                    {
                        "source": "voice",
                        "message": str(e),
                    },
                )

                self.voice_fsm.force(
                    VoiceState.IDLE,
                    "error recovery",
                )

                time.sleep(0.5)

        logger.info(
            "[VOICE] Command listener stopped."
        )

    # =========================================================================
    # HAZARD EVENTS
    # =========================================================================

    def _publish_hazard_if_any(
        self,
        speech_cmd: Dict[str, Any],
        priority: int,
        source: str,
    ) -> None:

        """
        Publish HAZARD envelope for the WebSocket event stream.
        """

        event_key = str(
            speech_cmd.get(
                "event_key",
                "",
            )
            or ""
        )

        is_hazard = (
            event_key.startswith(
                "hazard"
            )
            or source == "DEPTH"
            or priority in (
                PriorityLevel.EMERGENCY,
                PriorityLevel.CRITICAL_HAZARD,
                PriorityLevel.OBSTACLE,
            )
        )

        if not is_hazard:

            return

        severity = (
            "emergency"
            if priority
            <= PriorityLevel.EMERGENCY
            else "warning"
        )

        event_broker.publish(
            EventType.HAZARD,
            {
                "text": speech_cmd.get(
                    "text",
                    "",
                ),
                "priority": priority,
                "severity": severity,
                "event_key": (
                    event_key
                    or None
                ),
                "source": source,
            },
        )

    # =========================================================================
    # NAVIGATION
    # =========================================================================

    def _navigation_loop(self):
        """Existing navigator gets its own clock, independent of camera availability."""
        while not self._navigation_stop.wait(1.0):
            try:
                self._navigation_tick()
            except Exception:
                logger.exception("Navigation update failed")

    def _navigation_tick(self) -> Optional[Dict[str, Any]]:
        if not self._navigation_lock.acquire(blocking=False):
            return None
        try:
            speech_generation = self.tts.speech_generation
            loc = self.gps.get_location()
            update = self.navigator.update((loc.get("lat"), loc.get("lon")))
            if update:
                event_broker.publish(EventType.NAVIGATION_UPDATE, update)
                if update.get("speak") and update.get("text"):
                    self.tts.speak(update["text"], priority=PriorityLevel.NAVIGATION,
                                   source="NAVIGATION", generation_id=speech_generation)
            return update
        finally:
            self._navigation_lock.release()

    # =========================================================================
    # HUD
    # =========================================================================

    def _draw_hud_annotations(
        self,
        frame: np.ndarray,
        world_state: WorldState,
    ):

        annotated = frame.copy()

        h, w = frame.shape[:2]

        cv2.line(
            annotated,
            (
                int(w * 0.34),
                0,
            ),
            (
                int(w * 0.34),
                h,
            ),
            (45, 45, 45),
            1,
        )

        cv2.line(
            annotated,
            (
                int(w * 0.66),
                0,
            ),
            (
                int(w * 0.66),
                h,
            ),
            (45, 45, 45),
            1,
        )

        for obj in (
            world_state.get_active_objects()
        ):

            x1, y1, x2, y2 = (
                obj.bounding_box
            )

            cls = obj.class_name

            tid = obj.tracking_id

            dir_text = (
                obj.direction.value
            )

            if (
                obj.hazard_severity.value
                in [
                    "emergency",
                    "warning",
                ]
            ):

                color = (
                    0,
                    0,
                    240,
                )

            elif obj.is_stable:

                color = (
                    0,
                    220,
                    80,
                )

            else:

                color = (
                    120,
                    120,
                    120,
                )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = (
                f"#{tid} {cls} "
                f"[{dir_text}]"
            )

            if not obj.is_stable:

                label += (
                    " (confirming...)"
                )

            cv2.putText(
                annotated,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        det_status = (
            "OK"
            if self.detector.is_ready()
            else self.detector.status
        )

        mode_name = (
            world_state.current_mode.value
            .upper()
        )

        mode_str = (
            f"MODE: {mode_name} | "
            f"FPS: {self.current_fps:.1f}/18 | "
            f"Lat: {self.last_latency_ms:.1f}ms | "
            f"Det: {det_status} | "
            f"Voice: {self.voice_state}"
        )

        cv2.rectangle(
            annotated,
            (0, 0),
            (w, 32),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            annotated,
            mode_str,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )

        with self._lock:

            self._latest_annotated_frame = (
                annotated
            )

    # =========================================================================
    # FRAME ACCESS
    # =========================================================================

    def get_annotated_frame(
        self,
    ) -> Optional[np.ndarray]:

        with self._lock:

            if (
                self._latest_annotated_frame
                is not None
            ):

                return (
                    self._latest_annotated_frame.copy()
                )

            return None

    # =========================================================================
    # PUBLIC WORLD VIEW
    # =========================================================================

    def public_world_view(
        self,
    ) -> Dict[str, Any]:

        state = (
            self.world_state_mgr.get_snapshot()
        )

        objects = []

        for obj in (
            state.get_active_objects()
        ):

            objects.append(
                {
                    "id": (
                        f"{obj.class_name}_"
                        f"{obj.tracking_id}"
                    ),
                    "label": obj.class_name,
                    "position": (
                        obj.direction.value
                    ),
                    "distance_m": None,
                    "velocity": (
                        obj.motion_estimate.value
                    ),
                    "confidence": (
                        obj.confidence
                    ),
                }
            )

        hazards = []

        if state.active_hazard:

            hazards.append(
                state.active_hazard
            )

        return {
            "mode": (
                state.current_mode.value
                .upper()
            ),
            "objects": objects,
            "hazards": hazards,
            "navigation": self.navigator.health(),
            "timestamp": (
                time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(
                        state.timestamp
                    ),
                )
            ),
            "voice_state": (
                self.voice_state
            ),
            "camera_connected": (
                self.camera.is_connected()
            ),
            "depth": (
                self.depth.health()
            ),
            "gps": (
                self.gps.health()
            ),
        }

    # =========================================================================
    # SUBSYSTEM HEALTH
    # =========================================================================

    def get_subsystems_health(
        self,
    ) -> Dict[str, Any]:

        cam_diag = {}

        if hasattr(
            self.camera,
            "get_diagnostics",
        ):

            cam_diag = (
                self.camera.get_diagnostics()
            )

        else:

            cam_diag = {
                "source": (
                    self.camera_source_type
                ),
                "connected": (
                    self.camera.is_connected()
                ),
                "connection_state": (
                    "ONLINE"
                    if self.camera.is_connected()
                    else "OFFLINE"
                ),
            }

        asr_ready = (
            getattr(
                self.asr,
                "model",
                None,
            )
            is not None
        )

        return {

            "camera": {
                "configured_source": (
                    settings.CAMERA_SOURCE
                ),
                "active_source": (
                    self.camera_source_type
                ),
                "connected": (
                    self.camera.is_connected()
                ),
                "connection_state": (
                    cam_diag.get(
                        "connection_state",
                        "UNKNOWN",
                    )
                ),
                "camera_received_fps": (
                    cam_diag.get(
                        "camera_received_fps",
                        0.0,
                    )
                ),
                "snapshot_fps": (
                    cam_diag.get(
                        "snapshot_fps",
                        cam_diag.get(
                            "camera_received_fps",
                            0.0,
                        ),
                    )
                ),
                "snapshot_request_latency_ms": (
                    cam_diag.get(
                        "snapshot_request_latency_ms",
                        0.0,
                    )
                ),
                "avg_request_latency_ms": (
                    cam_diag.get(
                        "avg_request_latency_ms",
                        0.0,
                    )
                ),
                "p95_request_latency_ms": (
                    cam_diag.get(
                        "p95_request_latency_ms",
                        0.0,
                    )
                ),
                "jpeg_decode_latency_ms": (
                    cam_diag.get(
                        "jpeg_decode_latency_ms",
                        0.0,
                    )
                ),
                "frame_width": (
                    cam_diag.get(
                        "frame_width",
                        0,
                    )
                ),
                "frame_height": (
                    cam_diag.get(
                        "frame_height",
                        0,
                    )
                ),
                "dropped_frames": (
                    cam_diag.get(
                        "dropped_frames",
                        0,
                    )
                ),
                "reconnect_count": (
                    cam_diag.get(
                        "reconnect_count",
                        0,
                    )
                ),
                "decode_failures": (
                    cam_diag.get(
                        "decode_failures",
                        0,
                    )
                ),
                "last_frame_timestamp": (
                    cam_diag.get(
                        "last_frame_timestamp",
                        0.0,
                    )
                ),
                "frame_age_ms": (
                    cam_diag.get(
                        "frame_age_ms",
                        0.0,
                    )
                ),
                "active_url": (
                    cam_diag.get(
                        "active_url",
                        "",
                    )
                ),
            },

            "detector": {
                "status": (
                    self.detector.status
                ),
                "model": (
                    settings.YOLO_MODEL_PATH
                ),
                "device": (
                    self.detector.device
                ),
                "ready": (
                    self.detector.is_ready()
                ),
            },

            "speech_engine": {
                "status": (
                    "READY"
                    if self.tts is not None
                    else "UNAVAILABLE"
                ),
                "speaking": (
                    self.tts.is_speaking()
                    if self.tts
                    else False
                ),
                "speech_generation": (
                    self.tts.speech_generation
                ),
            },

            "asr": {
                "model": (
                    settings.ASR_MODEL
                ),
                "device": (
                    settings.ASR_DEVICE
                ),
                "loaded": (
                    asr_ready
                ),
                "voice_state": (
                    self.voice_state
                ),
                "capture": "vad",
            },

            "depth": (
                self.depth.health()
            ),

            "gps": (
                self.gps.health()
            ),

            "ocr": {
                "status": "READY" if self.ocr.ocr_engine is not None else "UNAVAILABLE",
                "engine_loaded": (
                    self.ocr.ocr_engine
                    is not None
                ),
            },

            "vlm": {
                "model": (
                    settings.OLLAMA_VLM_MODEL
                ),
                "status": "ON_DEMAND",
            },

            # IMPORTANT:
            # This now exposes the real wake-word provider health,
            # including model, threshold, mode, and any initialization
            # error.
            "wake_word": (
                self.wake_provider.health()
            ),

            "navigation": (
                self.navigator.health()
            ),

            "voice_state_machine": (
                self.voice_fsm.snapshot()
            ),

            "currency": {
                "status": "ON_DEMAND"
            },

            "medicine": {
                "status": "ON_DEMAND"
            },

            "target_fps": (
                settings.VISIONMATE_TARGET_FPS
            ),

            "current_fps": (
                self.current_fps
            ),
        }

    # =========================================================================
    # VOICE COMMAND
    # =========================================================================

    def handle_voice_command(
        self,
        transcript: str,
    ) -> Dict[str, Any]:

        return (
            self.command_bus.dispatch(
                transcript
            )
        )

    # =========================================================================
    # GLOBAL STOP
    # =========================================================================

    def handle_stop_command(
        self,
    ) -> Dict[str, Any]:

        # Stop/purge speech immediately.
        self.tts.stop()

        # Stop FIND.
        self.find_handler.stop_find()

        self.world_state_mgr.set_find_target(
            None
        )

        # Return to guidance.
        self.world_state_mgr.set_mode(
            SystemMode.GUIDANCE
        )

        # Stop navigation.
        self.navigator.stop()

        # Invalidate any in-flight voice command
        # and reset the voice FSM.
        self.voice_fsm.abort(
            reason="STOP command"
        )

        self.priority_engine.last_speech_time = (
            time.time()
        )

        event_broker.publish(
            EventType.MODE_CHANGED_PUBLIC,
            {
                "mode": "GUIDANCE"
            },
        )

        logger.info(
            "[GLOBAL STOP] Speech purged, "
            "Find/Navigation terminated, "
            "voice session invalidated."
        )

        return {
            "status": "STOPPED",
            "mode": "guidance",
            "message": (
                "System stopped. "
                "Guidance active."
            ),
            "speech_generation": (
                self.tts.speech_generation
            ),
        }

    # =========================================================================
    # MODE CONTROL
    # =========================================================================

    def set_mode(
        self,
        mode_name: str,
    ) -> str:

        m = (
            mode_name
            .lower()
            .strip()
        )

        if m in [
            "guidance",
            "awareness",
        ]:

            self.find_handler.stop_find()

            self.world_state_mgr.set_find_target(
                None
            )

            self.world_state_mgr.set_mode(
                SystemMode.GUIDANCE
            )

            msg = (
                "Switched to Guidance mode."
            )

        elif m == "find":

            self.world_state_mgr.set_mode(
                SystemMode.FIND
            )

            msg = (
                "Switched to Find mode."
            )

        elif m == "read":

            self.world_state_mgr.set_mode(
                SystemMode.READ
            )

            msg = (
                "Switched to Read mode."
            )

        elif m == "ask":

            self.world_state_mgr.set_mode(
                SystemMode.ASK
            )

            msg = (
                "Switched to Ask mode."
            )

        else:

            return (
                f"Unknown mode: {mode_name}"
            )

        self.tts.purge_mode_speech(
            "GUIDANCE"
            if m == "find"
            else "FIND"
        )

        self.tts.speak(
            msg,
            priority=(
                PriorityLevel.INTERACTION
            ),
            source="SYSTEM",
        )

        event_broker.publish(
            EventType.MODE_CHANGED_PUBLIC,
            {
                "mode": m.upper()
            },
        )

        return msg

    # =========================================================================
    # FIND
    # =========================================================================

    def start_find(
        self,
        target_query: str,
    ) -> str:

        self.tts.stop()

        self.world_state_mgr.set_find_target(
            target_query
        )

        msg, session_id = (
            self.find_handler.set_target(
                target_query
            )
        )

        self.tts.speak(
            msg,
            priority=(
                PriorityLevel.NAVIGATION
            ),
            source="FIND",
        )

        event_broker.publish(
            EventType.FIND_TARGET_UPDATED,
            {
                "target": target_query,
                "session_id": session_id,
            },
        )

        return msg

    # =========================================================================
    # READ
    # =========================================================================

    def trigger_read(
        self,
        full_reading: bool = False,
    ) -> Dict[str, Any]:

        generation = self.tts.speech_generation
        t0 = time.perf_counter()

        frame = (
            self.camera.get_latest_frame()
        )

        capture_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        res = (
            self.read_handler.trigger_read(
                frame,
                full_reading=full_reading,
            )
        )

        if "metrics" in res:

            res["metrics"][
                "ocr_capture_ms"
            ] = round(
                capture_ms,
                2,
            )

            res["metrics"][
                "ocr_total_ms"
            ] = round(
                res["metrics"].get(
                    "ocr_total_ms",
                    0.0,
                )
                + capture_ms,
                2,
            )

        if generation != self.tts.speech_generation:
            return {"status": "CANCELLED", "success": False, "text": "Request cancelled."}

        self.tts.speak(
            res["text"],
            priority=(
                PriorityLevel.INTERACTION
            ),
            source="READ",
            generation_id=generation,
        )

        self.world_state_mgr.set_mode(
            SystemMode.GUIDANCE
        )

        return res

    # =========================================================================
    # ASK
    # =========================================================================

    def trigger_ask(
        self,
        user_query: str,
    ) -> Dict[str, Any]:

        generation = self.tts.speech_generation
        frame = (
            self.camera.get_latest_frame()
        )

        res = (
            self.ask_handler.ask(
                frame,
                user_query,
            )
        )

        if generation != self.tts.speech_generation:
            return {"status": "CANCELLED", "success": False, "text": "Request cancelled."}

        self.tts.speak(
            res["text"],
            priority=(
                PriorityLevel.INTERACTION
            ),
            source="ASK",
            generation_id=generation,
        )

        self.world_state_mgr.set_mode(
            SystemMode.GUIDANCE
        )

        return res

    # =========================================================================
    # LATEST FRAME
    # =========================================================================

    def _latest_frame(
        self,
    ):

        frame = (
            self.camera.get_latest_frame()
        )

        return frame

    # =========================================================================
    # COLOR
    # =========================================================================

    def trigger_color(
        self,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:

        """Run on-demand local color recognition without touching the 18 FPS loop."""

        generation = self.tts.speech_generation
        frame = self.camera.get_latest_frame()

        track_summaries: List[Dict[str, Any]] = []

        try:

            snapshot = self.world_state_mgr.get_snapshot()

            for obj in snapshot.get_active_objects():

                track_summaries.append(
                    {
                        "class_name": obj.class_name,
                        "bbox": list(obj.bounding_box),
                    }
                )

        except Exception as exc:

            logger.debug(
                "[COLOR] Track lookup skipped: %s",
                exc,
            )

        result = self.color_handler.analyze(
            frame,
            target=target,
            tracks=track_summaries,
        )

        if generation != self.tts.speech_generation:
            return {"status": "CANCELLED", "success": False, "text": "Request cancelled."}

        text = result.get("text", "")

        if text:

            self.tts.speak(
                text,
                priority=PriorityLevel.INTERACTION,
                source="COLOR",
            generation_id=generation,
            )

            try:

                self.world_state_mgr.record_speech(
                    text
                )

            except Exception:

                pass

            event_broker.publish(
                EventType.COLOR_RESULT_READY,
                {
                    "text": text,
                    "target": target,
                    "color": result.get("color"),
                    "confidence": result.get("confidence"),
                    "source": "COLOR",
                },
            )

        return result

    # =========================================================================
    # CURRENCY
    # =========================================================================

    def trigger_currency(
        self,
    ) -> Dict[str, Any]:

        """
        ON-DEMAND currency recognition.

        Never runs continuously.
        """

        generation = self.tts.speech_generation
        res = (
            self.currency_recognizer.recognize(
                self._latest_frame()
            )
        )

        if generation != self.tts.speech_generation:
            return {"status": "CANCELLED", "success": False, "text": "Request cancelled."}

        self.tts.speak(
            res["text"],
            priority=(
                PriorityLevel.INTERACTION
            ),
            source="CURRENCY",
            generation_id=generation,
        )

        self.world_state_mgr.set_mode(
            SystemMode.GUIDANCE
        )

        return res

    # =========================================================================
    # MEDICINE
    # =========================================================================

    def trigger_medicine(
        self,
    ) -> Dict[str, Any]:

        """
        ON-DEMAND medicine/product label recognition.

        Never runs continuously.
        """

        generation = self.tts.speech_generation
        res = (
            self.medicine_recognizer.recognize(
                self._latest_frame()
            )
        )

        if generation != self.tts.speech_generation:
            return {"status": "CANCELLED", "success": False, "text": "Request cancelled."}

        self.tts.speak(
            res["text"],
            priority=(
                PriorityLevel.INTERACTION
            ),
            source="MEDICINE",
            generation_id=generation,
        )

        self.world_state_mgr.set_mode(
            SystemMode.GUIDANCE
        )

        return res

    # =========================================================================
    # NAVIGATION
    # =========================================================================

    def start_navigation(
        self,
        destination: str,
    ) -> Dict[str, Any]:

        loc = (
            self.gps.get_location()
        )

        origin = (
            loc.get("lat"),
            loc.get("lon"),
        )

        speech_generation = self.tts.speech_generation
        res = (
            self.navigator.start(
                destination,
                origin,
            )
        )

        if res.get("text") and res.get("status") != "CANCELLED":

            self.tts.speak(
                res["text"],
                priority=(
                    PriorityLevel.NAVIGATION
                ),
                source="NAVIGATION",
                generation_id=speech_generation,
            )

        event_broker.publish(
            EventType.NAVIGATION_UPDATE,
            res,
        )

        return res

    def stop_navigation(
        self,
    ) -> Dict[str, Any]:

        self.navigator.stop()
        self.tts.stop()

        text = (
            "Navigation stopped."
        )

        self.tts.speak(
            text,
            priority=(
                PriorityLevel.NAVIGATION
            ),
            source="NAVIGATION",
        )

        event_broker.publish(
            EventType.NAVIGATION_UPDATE,
            {
                "status": "STOPPED",
                "navigation": self.navigator.health(),
                "text": text,
            },
        )

        return {
            "status": "stopped",
            "mode": "navigation",
            "message": text,
        }

    # =========================================================================
    # STOP PIPELINE
    # =========================================================================

    def stop(self):

        self._navigation_stop.set()
        self.navigator.stop()
        if self._navigation_thread and self._navigation_thread.is_alive():
            self._navigation_thread.join(timeout=1.5)
        self._running = False

        self._voice_running = False

        # ---------------------------------------------------------------------
        # Wait for perception worker.
        # ---------------------------------------------------------------------

        if (
            self._perception_thread
            and
            self._perception_thread.is_alive()
        ):

            self._perception_thread.join(
                timeout=1.5
            )

        # ---------------------------------------------------------------------
        # Wait for voice worker.
        # ---------------------------------------------------------------------

        if (
            self._voice_thread
            and
            self._voice_thread.is_alive()
        ):

            self._voice_thread.join(
                timeout=1.5
            )

        # ---------------------------------------------------------------------
        # Stop wake-word provider.
        # This releases its microphone stream.
        # ---------------------------------------------------------------------

        try:

            self.wake_provider.stop()

        except Exception as exc:

            logger.debug(
                "[WAKE] Provider shutdown notice: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # Hardware / TTS
        # ---------------------------------------------------------------------

        try:

            self.camera.stop()

        except Exception as exc:

            logger.debug(
                "[CAMERA] Shutdown notice: %s",
                exc,
            )

        try:

            self.tts.stop()

        except Exception as exc:

            logger.debug(
                "[TTS] Shutdown notice: %s",
                exc,
            )

        logger.info(
            "VisionMate pipeline stopped."
        )


# ============================================================================
# GLOBAL PIPELINE INSTANCE
# ============================================================================

pipeline = VisionMatePipeline(
    camera_source=settings.CAMERA_SOURCE
)