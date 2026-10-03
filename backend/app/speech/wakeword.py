"""
VisionMate / Drishti
Local Offline Wake-Word Detection

REAL wake word:
    "Hey Mycroft"

Engine:
    openWakeWord

Important architecture:

    Microphone
        ↓
    openWakeWord / hey_mycroft
        ↓
    Wake detected
        ↓
    VAD
        ↓
    Faster-Whisper
        ↓
    Command Bus
        ↓
    Feature
        ↓
    TTS

This module NEVER uses ASR transcript matching to detect the wake word.

VAD answers:
    "Is somebody speaking?"

ASR answers:
    "What did they say?"

Wake-word detection answers:
    "Did the user say the activation phrase?"

The real wake-word detector is fully local after the one-time
model download.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import sys
import time

import numpy as np

from backend.app.core.config import settings


logger = logging.getLogger("visionmate.wakeword")


# ============================================================================
# OPENWAKEWORD IMPORT
# ============================================================================

def _try_import_engine() -> Tuple[Any, Any]:
    """
    Import openWakeWord lazily.

    Lazy importing is intentional:
    - backend remains importable if openWakeWord is not installed
    - unit tests can inject a fake engine
    - actual model loading happens only when wake detection starts
    """

    import openwakeword  # type: ignore
    from openwakeword.model import Model  # type: ignore

    return openwakeword, Model


def _engine_importable() -> bool:
    try:
        _try_import_engine()
        return True
    except Exception:
        return False


# ============================================================================
# MODEL DOWNLOAD / RESOLUTION
# ============================================================================

def _download_pretrained_model(
    openwakeword_module: Any,
    model_name: str,
) -> None:
    """
    Download only the requested pretrained model and its required
    feature models.

    openWakeWord downloads models once and caches them locally.
    Runtime inference is then local/offline.
    """

    try:
        from openwakeword.utils import download_models  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openWakeWord is installed, but its model download utility "
            f"could not be imported: {exc}"
        ) from exc

    try:
        # Current openWakeWord API.
        download_models(model_names=[model_name])
        return
    except TypeError:
        # Compatibility with older releases.
        pass

    try:
        download_models([model_name])
        return
    except TypeError:
        pass

    # Last compatibility option: download all official pretrained models.
    # This is less efficient, but still valid.
    try:
        download_models()
    except Exception as exc:
        raise RuntimeError(
            f"Could not download pretrained wake-word model '{model_name}': {exc}"
        ) from exc


def _resolve_pretrained_model_path(
    openwakeword_module: Any,
    model_name: str,
    inference_framework: str = "onnx",
) -> Optional[str]:
    """
    Resolve the actual downloaded model file.

    This deliberately resolves the actual model path instead of assuming
    that every openWakeWord version accepts the friendly model name
    directly in Model(...).

    Expected result for Hey Mycroft is something similar to:

        .../hey_mycroft_v0.1.onnx
    """

    # Preferred API.
    try:
        get_paths = getattr(
            openwakeword_module,
            "get_pretrained_model_paths",
            None,
        )

        if callable(get_paths):
            paths = get_paths(inference_framework=inference_framework)

            for path in paths or []:
                path_str = str(path).lower()

                if model_name.lower() in path_str:
                    if Path(path).exists():
                        return str(Path(path).resolve())

    except Exception as exc:
        logger.debug(
            "[WAKE] get_pretrained_model_paths failed: %s",
            exc,
        )

    # Current package layout fallback.
    package_file = getattr(openwakeword_module, "__file__", None)

    if package_file:
        package_dir = Path(package_file).resolve().parent

        candidate_dirs = [
            package_dir / "resources" / "models",
            package_dir / "models",
        ]

        extensions = (
            [".onnx"]
            if inference_framework == "onnx"
            else [".tflite"]
        )

        for directory in candidate_dirs:
            if not directory.exists():
                continue

            for extension in extensions:
                matches = sorted(
                    directory.glob(f"*{model_name}*{extension}")
                )

                if matches:
                    return str(matches[0].resolve())

    return None


# ============================================================================
# STRUCTURED RESULT
# ============================================================================

@dataclass(frozen=True)
class WakeWordResult:
    """
    Result produced by one acoustic inference operation.
    """

    detected: bool
    score: float = 0.0
    phrase: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# BASE PROVIDER
# ============================================================================

class WakeWordProvider(ABC):
    """
    Base abstraction for wake-word providers.
    """

    name: str = "wakeword"

    # True ONLY when an actual acoustic wake-word model is used.
    is_real_wake_word: bool = False

    # Possible values:
    # REAL_WAKE_WORD
    # DEVELOPMENT_FALLBACK
    # DISABLED
    mode: str = "DEVELOPMENT_FALLBACK"

    @abstractmethod
    def start(self) -> bool:
        """
        Initialize resources/model.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """
        Release resources.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for the wake word.

        Returns:
            True  -> wake detected
            False -> timeout/stop/error
        """
        raise NotImplementedError

    def process_audio(
        self,
        chunk: np.ndarray,
    ) -> WakeWordResult:
        """
        Process an audio chunk.

        Base providers do not perform acoustic inference.
        """
        return WakeWordResult(
            detected=False,
            phrase=getattr(
                settings,
                "WAKE_WORD_PHRASE",
                None,
            ),
            error=f"{self.name} does not perform acoustic inference",
        )

    def is_available(self) -> bool:
        return True

    def reset(self) -> None:
        return None

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "mode": self.mode,
            "is_real_wake_word": self.is_real_wake_word,
            "phrase": getattr(
                settings,
                "WAKE_WORD_PHRASE",
                None,
            ),
            "available": self.is_available(),
        }


# ============================================================================
# REAL OPENWAKEWORD PROVIDER
# ============================================================================

class OpenWakeWordProvider(WakeWordProvider):
    """
    Genuine local acoustic wake-word detector.

    Target:
        Hey Mycroft

    Model:
        hey_mycroft

    Audio:
        16 kHz
        mono
        signed int16 PCM
        1280 samples / 80 ms

    The microphone stream is owned only while waiting for the wake word.
    Once the wake word fires, the stream is released so the ASR/VAD
    subsystem can acquire the microphone cleanly.
    """

    name = "openwakeword"

    is_real_wake_word = True

    mode = "REAL_WAKE_WORD"

    REQUIRED_SAMPLE_RATE = 16000

    FRAME_MS = 80

    FRAME_SAMPLES = 1280

    INFERENCE_FRAMEWORK = "onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        pretrained: Optional[str] = None,
        threshold: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> None:

        self.model_path = (
            model_path
            if model_path is not None
            else getattr(
                settings,
                "WAKE_WORD_MODEL_PATH",
                "",
            )
        )

        self.pretrained = (
            pretrained
            or getattr(
                settings,
                "WAKE_WORD_PRETRAINED",
                "hey_mycroft",
            )
        )

        self.threshold = float(
            threshold
            if threshold is not None
            else getattr(
                settings,
                "WAKE_WORD_THRESHOLD",
                0.5,
            )
        )

        self.sample_rate = int(
            sample_rate
            or getattr(
                settings,
                "ASR_SAMPLE_RATE",
                16000,
            )
        )

        self._model: Any = None

        self._model_names: List[str] = []

        self._stream: Any = None

        self._running = False

        self._load_attempted = False

        self._warned_malformed = False

        self._warned_mic = False

        self.last_error: Optional[str] = None

        self.last_result: Optional[WakeWordResult] = None

        self._resolved_model_path: Optional[str] = None

    # ========================================================================
    # MODEL LIFECYCLE
    # ========================================================================

    def _load_model(self) -> bool:
        """
        Load the REAL acoustic model exactly once per start lifecycle.
        """

        if self._model is not None:
            return True

        if self._load_attempted:
            return False

        self._load_attempted = True

        # ------------------------------------------------------------
        # Validate sample rate.
        # ------------------------------------------------------------

        if self.sample_rate != self.REQUIRED_SAMPLE_RATE:
            self.last_error = (
                "openWakeWord requires 16 kHz audio. "
                f"Configured sample rate is {self.sample_rate}. "
                "Set ASR_SAMPLE_RATE=16000."
            )

            logger.error(
                "[WAKE] REAL_WAKE_WORD initialization failed: %s",
                self.last_error,
            )

            return False

        # ------------------------------------------------------------
        # Validate target model.
        # ------------------------------------------------------------

        if self.pretrained != "hey_mycroft" and not self.model_path:
            self.last_error = (
                "Wake-word configuration mismatch: expected pretrained "
                "'hey_mycroft', got "
                f"'{self.pretrained}'."
            )

            logger.error(
                "[WAKE] REAL_WAKE_WORD initialization failed: %s",
                self.last_error,
            )

            return False

        # ------------------------------------------------------------
        # Import engine lazily.
        # ------------------------------------------------------------

        try:
            openwakeword_module, ModelClass = _try_import_engine()

        except Exception as exc:

            self.last_error = (
                "openWakeWord is required for real wake-word detection "
                "but could not be imported. "
                "Install it with: pip install openwakeword "
                f"(import error: {exc})"
            )

            logger.error(
                "[WAKE] REAL_WAKE_WORD initialization failed: %s",
                self.last_error,
            )

            return False

        # ------------------------------------------------------------
        # Load explicit model path OR pretrained Hey Mycroft model.
        # ------------------------------------------------------------

        try:

            if self.model_path:

                model_file = Path(self.model_path).expanduser().resolve()

                if not model_file.exists():

                    self.last_error = (
                        "Configured wake-word model path does not exist: "
                        f"{model_file}"
                    )

                    logger.error(
                        "[WAKE] REAL_WAKE_WORD initialization failed: %s",
                        self.last_error,
                    )

                    return False

                target_model = str(model_file)

            else:

                # Download/cached model.
                _download_pretrained_model(
                    openwakeword_module,
                    self.pretrained,
                )

                target_model = _resolve_pretrained_model_path(
                    openwakeword_module,
                    self.pretrained,
                    self.INFERENCE_FRAMEWORK,
                )

                if not target_model:

                    self.last_error = (
                        "openWakeWord downloaded/located its resources, "
                        "but the actual pretrained model file for "
                        f"'{self.pretrained}' could not be resolved."
                    )

                    logger.error(
                        "[WAKE] REAL_WAKE_WORD initialization failed: %s",
                        self.last_error,
                    )

                    return False

            # --------------------------------------------------------
            # Load actual model file.
            # --------------------------------------------------------

            self._model = ModelClass(
                wakeword_models=[target_model],
                inference_framework=self.INFERENCE_FRAMEWORK,
            )

            self._resolved_model_path = target_model

        except Exception as exc:

            self._model = None

            self.last_error = (
                f"Failed to load openWakeWord model "
                f"'{self.pretrained}': {exc}"
            )

            logger.error(
                "[WAKE] REAL_WAKE_WORD initialization failed: %s",
                self.last_error,
            )

            return False

        # ------------------------------------------------------------
        # Determine loaded model names.
        # ------------------------------------------------------------

        model_names = getattr(
            self._model,
            "model_names",
            None,
        )

        if model_names is None:

            models_dict = getattr(
                self._model,
                "models",
                {},
            )

            if isinstance(models_dict, dict):
                model_names = list(models_dict.keys())

        self._model_names = [
            str(name)
            for name in (model_names or [])
        ]

        # ------------------------------------------------------------
        # Verify Hey Mycroft was actually loaded.
        # ------------------------------------------------------------

        loaded_text = " ".join(
            self._model_names
        ).lower()

        resolved_text = str(
            self._resolved_model_path or ""
        ).lower()

        if (
            "mycroft" not in loaded_text
            and "mycroft" not in resolved_text
        ):
            self.last_error = (
                "Loaded wake-word model could not be verified as "
                "'hey_mycroft'. "
                f"Loaded names={self._model_names}, "
                f"path={self._resolved_model_path}"
            )

            self._model = None

            logger.error(
                "[WAKE] REAL_WAKE_WORD verification failed: %s",
                self.last_error,
            )

            return False

        logger.info(
            "[WAKE] REAL_WAKE_WORD READY | "
            "phrase='%s' | "
            "model='hey_mycroft' | "
            "path='%s' | "
            "loaded_names=%s | "
            "threshold=%.2f | "
            "audio=16kHz mono int16 | "
            "frame=%d samples",
            getattr(
                settings,
                "WAKE_WORD_PHRASE",
                "hey mycroft",
            ),
            self._resolved_model_path,
            self._model_names,
            self.threshold,
            self.FRAME_SAMPLES,
        )

        return True

    # ========================================================================
    # START / STOP
    # ========================================================================

    def start(self) -> bool:
        """
        Start the real wake-word provider.

        Returns True ONLY if the real acoustic model is loaded.
        """

        self._running = True

        # Allow an explicit restart after an earlier failure.
        self._load_attempted = False

        success = self._load_model()

        if not success:

            logger.error(
                "[WAKE] Real Hey Mycroft detection is NOT active."
            )

            return False

        return True

    def stop(self) -> None:
        """
        Clean shutdown.
        """

        self._running = False

        self._release_stream()

        self._model = None

        self._model_names = []

        self._resolved_model_path = None

        logger.info(
            "[WAKE] Wake-word detector stopped."
        )

    def is_available(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        """
        Reset internal openWakeWord state after a detection.
        """

        if self._model is None:
            return

        try:
            self._model.reset()

        except Exception as exc:

            logger.debug(
                "[WAKE] Acoustic model reset failed: %s",
                exc,
            )

    # ========================================================================
    # AUDIO PROCESSING
    # ========================================================================

    def process_audio(
        self,
        chunk: np.ndarray,
    ) -> WakeWordResult:
        """
        Run REAL acoustic inference on one PCM audio chunk.

        This is the ONLY method that can create a wake event.
        """

        if self._model is None:

            result = WakeWordResult(
                detected=False,
                score=0.0,
                phrase=getattr(
                    settings,
                    "WAKE_WORD_PHRASE",
                    "hey mycroft",
                ),
                error="Acoustic wake-word model is not loaded.",
            )

            self.last_result = result

            return result

        pcm = self._to_int16(chunk)

        if pcm.size == 0:

            result = WakeWordResult(
                detected=False,
                score=0.0,
                phrase=getattr(
                    settings,
                    "WAKE_WORD_PHRASE",
                    "hey mycroft",
                ),
            )

            self.last_result = result

            return result

        try:

            scores = self._model.predict(pcm)

        except Exception as exc:

            return self._malformed(
                f"openWakeWord inference error: {exc}"
            )

        if not isinstance(scores, dict) or not scores:

            return self._malformed(
                "openWakeWord returned an unexpected result."
            )

        # ------------------------------------------------------------
        # We loaded exactly one model, but keep this robust.
        # ------------------------------------------------------------

        target_scores = []

        for model_name, raw_score in scores.items():

            if (
                "mycroft"
                in str(model_name).lower()
            ):
                try:
                    target_scores.append(
                        float(raw_score)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        if not target_scores:

            return self._malformed(
                "Loaded openWakeWord model did not return "
                "a Hey Mycroft score."
            )

        score = max(target_scores)

        if np.isnan(score):

            return self._malformed(
                "openWakeWord returned NaN."
            )

        detected = score >= self.threshold

        if detected:

            logger.info(
                "[WAKE] Hey Mycroft DETECTED | score=%.3f | model=%s",
                score,
                self._model_names,
            )

            # Prevent immediate retriggering.
            self.reset()

            result = WakeWordResult(
                detected=True,
                score=score,
                phrase=getattr(
                    settings,
                    "WAKE_WORD_PHRASE",
                    "hey mycroft",
                ),
            )

        else:

            result = WakeWordResult(
                detected=False,
                score=score,
                phrase=getattr(
                    settings,
                    "WAKE_WORD_PHRASE",
                    "hey mycroft",
                ),
            )

        self.last_result = result

        return result

    # ========================================================================
    # LIVE MICROPHONE
    # ========================================================================

    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for Hey Mycroft from the live microphone.

        The microphone stream is opened ONCE while waiting.

        When wake is detected:
            stream is released
            ASR/VAD can take microphone ownership.
        """

        if not self._running:
            return False

        if not self._load_model():
            return False

        if not self._ensure_stream():
            return False

        deadline = (
            time.time() + timeout
            if timeout is not None
            else None
        )

        while self._running:

            if (
                stop_flag is not None
                and stop_flag()
            ):

                self._release_stream()

                return False

            if (
                deadline is not None
                and time.time() >= deadline
            ):

                # Keep stream alive so the next poll does not reopen
                # the microphone.
                return False

            try:

                frames, overflowed = self._stream.read(
                    self.FRAME_SAMPLES
                )

            except Exception as exc:

                self.last_error = (
                    "Microphone read failed: "
                    f"{exc}"
                )

                logger.error(
                    "[WAKE] %s",
                    self.last_error,
                )

                self._release_stream()

                return False

            if overflowed:

                logger.debug(
                    "[WAKE] Microphone input overflow."
                )

            if getattr(
                frames,
                "ndim",
                1,
            ) > 1:

                frames = frames[:, 0]

            result = self.process_audio(
                frames
            )

            if result.detected:

                # IMPORTANT:
                # Give the microphone to ASR/VAD.
                self._release_stream()

                return True

        self._release_stream()

        return False

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health(self) -> Dict[str, Any]:

        return {
            "provider": self.name,
            "mode": self.mode,
            "is_real_wake_word": True,
            "phrase": getattr(
                settings,
                "WAKE_WORD_PHRASE",
                "hey mycroft",
            ),
            "engine": "openWakeWord",
            "model": "hey_mycroft",
            "resolved_model_path": self._resolved_model_path,
            "model_names": self._model_names,
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "frame_samples": self.FRAME_SAMPLES,
            "available": self.is_available(),
            "last_error": self.last_error,
        }

    # ========================================================================
    # MICROPHONE INTERNALS
    # ========================================================================

    def _ensure_stream(self) -> bool:

        if self._stream is not None:
            return True

        try:

            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.FRAME_SAMPLES,
            )

            self._stream.start()

            self._warned_mic = False

            return True

        except Exception as exc:

            self.last_error = (
                "Microphone unavailable for "
                f"wake-word detection: {exc}"
            )

            if not self._warned_mic:

                self._warned_mic = True

                logger.error(
                    "[WAKE] %s",
                    self.last_error,
                )

            else:

                logger.debug(
                    "[WAKE] %s",
                    self.last_error,
                )

            self._stream = None

            return False

    def _release_stream(self) -> None:

        stream = self._stream

        self._stream = None

        if stream is None:
            return

        try:
            stream.stop()

        except Exception:
            pass

        try:
            stream.close()

        except Exception:
            pass

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    def _malformed(
        self,
        message: str,
    ) -> WakeWordResult:

        if not self._warned_malformed:

            self._warned_malformed = True

            logger.warning(
                "[WAKE] %s",
                message,
            )

        else:

            logger.debug(
                "[WAKE] %s",
                message,
            )

        result = WakeWordResult(
            detected=False,
            score=0.0,
            phrase=getattr(
                settings,
                "WAKE_WORD_PHRASE",
                "hey mycroft",
            ),
            error=message,
        )

        self.last_result = result

        return result

    @staticmethod
    def _to_int16(
        chunk: np.ndarray,
    ) -> np.ndarray:
        """
        Convert incoming audio into mono int16 PCM.
        """

        pcm = np.asarray(chunk)

        if pcm.ndim > 1:

            if pcm.shape[1] == 1:
                pcm = pcm[:, 0]
            else:
                pcm = pcm.reshape(-1)

        if pcm.dtype == np.int16:
            return pcm

        if np.issubdtype(
            pcm.dtype,
            np.floating,
        ):

            pcm = np.clip(
                pcm,
                -1.0,
                1.0,
            )

            return (
                pcm * 32767.0
            ).astype(np.int16)

        return pcm.astype(np.int16)


# ============================================================================
# DEVELOPMENT FALLBACK
# ============================================================================

class KeyboardWakeWordProvider(
    WakeWordProvider
):
    """
    DEVELOPMENT ONLY.

    Press Enter / Space to activate.

    This is NOT a real wake word.
    """

    name = "keyboard"

    is_real_wake_word = False

    mode = "DEVELOPMENT_FALLBACK"

    def start(self) -> bool:

        logger.warning(
            "[WAKE] DEVELOPMENT_FALLBACK enabled. "
            "Keyboard activation is NOT a real wake word."
        )

        return True

    def stop(self) -> None:
        return None

    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:

        # Non-interactive environments.
        if (
            not sys.stdin
            or not sys.stdin.isatty()
        ):

            end = time.time() + (
                timeout
                if timeout is not None
                else 1.0
            )

            while time.time() < end:

                if (
                    stop_flag is not None
                    and stop_flag()
                ):
                    return False

                time.sleep(0.05)

            return False

        try:

            if sys.platform.startswith("win"):

                import msvcrt

                while True:

                    if (
                        stop_flag is not None
                        and stop_flag()
                    ):
                        return False

                    if msvcrt.kbhit():

                        key = msvcrt.getch()

                        if key in (
                            b"\r",
                            b"\n",
                            b" ",
                        ):

                            logger.info(
                                "[WAKE] "
                                "Development fallback activated."
                            )

                            return True

                    time.sleep(0.03)

            else:

                import select

                while True:

                    if (
                        stop_flag is not None
                        and stop_flag()
                    ):
                        return False

                    ready, _, _ = select.select(
                        [sys.stdin],
                        [],
                        [],
                        0.1,
                    )

                    if ready:

                        sys.stdin.readline()

                        logger.info(
                            "[WAKE] "
                            "Development fallback activated."
                        )

                        return True

        except Exception as exc:

            logger.debug(
                "[WAKE] Keyboard fallback error: %s",
                exc,
            )

            return False


# ============================================================================
# DISABLED PROVIDER
# ============================================================================

class DisabledWakeWordProvider(
    WakeWordProvider
):
    """
    Wake word disabled.

    Voice session remains open.

    This is NOT a real wake word.
    """

    name = "disabled"

    is_real_wake_word = False

    mode = "DISABLED"

    def start(self) -> bool:

        logger.warning(
            "[WAKE] Wake word DISABLED."
        )

        return True

    def stop(self) -> None:
        return None

    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:

        if (
            stop_flag is not None
            and stop_flag()
        ):
            return False

        return True

    def health(self) -> Dict[str, Any]:

        return {
            "provider": self.name,
            "mode": self.mode,
            "is_real_wake_word": False,
            "phrase": None,
            "available": True,
        }


# ============================================================================
# FACTORY
# ============================================================================

def create_wake_word_provider(
    mode: Optional[str] = None,
    **kwargs: Any,
) -> WakeWordProvider:
    """
    Create the configured wake-word provider.

    Modes:

        real
            Real Hey Mycroft openWakeWord detector.

        auto
            Real detector when available, otherwise explicit
            development fallback.

        fallback
            Keyboard development fallback.

        disabled
            Wake gate disabled.
    """

    if not getattr(
        settings,
        "WAKE_WORD_ENABLED",
        True,
    ):

        logger.info(
            "[WAKE] Wake word disabled by configuration."
        )

        return DisabledWakeWordProvider()

    selected_mode = (
        mode
        or getattr(
            settings,
            "WAKE_WORD_MODE",
            "real",
        )
        or "real"
    ).lower().strip()

    # ------------------------------------------------------------
    # Disabled
    # ------------------------------------------------------------

    if selected_mode in (
        "disabled",
        "off",
        "none",
    ):

        return DisabledWakeWordProvider()

    # ------------------------------------------------------------
    # Development fallback
    # ------------------------------------------------------------

    if selected_mode in (
        "fallback",
        "keyboard",
        "push_to_talk",
        "pushtotalk",
    ):

        logger.warning(
            "[WAKE] DEVELOPMENT_FALLBACK selected."
        )

        return KeyboardWakeWordProvider()

    # ------------------------------------------------------------
    # Auto mode
    # ------------------------------------------------------------

    if (
        selected_mode == "auto"
        and not _engine_importable()
    ):

        logger.warning(
            "[WAKE] openWakeWord is unavailable. "
            "Using explicit DEVELOPMENT_FALLBACK."
        )

        return KeyboardWakeWordProvider()

    # ------------------------------------------------------------
    # Real mode
    # ------------------------------------------------------------

    return OpenWakeWordProvider(
        **kwargs
    )


# ============================================================================
# DEFAULT SINGLETON
# ============================================================================

wake_word_provider = create_wake_word_provider()