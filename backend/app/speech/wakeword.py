"""
VisionMate — Local / offline wake-word (keyword spotting) subsystem.

IMPORTANT DISTINCTION (do not conflate):
  * VAD            -> detects *that* a human is speaking, not *what* was said.
  * ASR substring  -> transcribes first, then string-matches. NOT a wake word.
  * Wake word (KWS) -> continuous, cheap, always-on acoustic keyword spotting
                       that fires ONLY when the enrolled trigger phrase is heard.

This module provides a `WakeWordProvider` interface with genuinely distinct modes:

  1. REAL wake word  — `OpenWakeWordProvider` uses the fully LOCAL, OFFLINE
     openWakeWord ONNX engine (no audio ever leaves the machine). It is a true
     acoustic keyword spotter. Requires the optional `openwakeword` dependency.

  2. DEVELOPMENT FALLBACK — `KeyboardWakeWordProvider` (Enter / push-to-talk) and
     `DisabledWakeWordProvider` (always-on session gate). These exist only so
     laptop development and CI work without an acoustic engine. They are clearly
     reported as `is_real_wake_word == False` and are never a real wake word.

The factory `create_wake_word_provider()` never silently claims a real wake word:
if the offline engine is unavailable it reports the fallback mode explicitly.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import logging
import sys
import time

import numpy as np

from backend.app.core.config import settings

logger = logging.getLogger("visionmate.wakeword")


class WakeWordProvider(ABC):
    """Abstract local wake-word detector. Must never require cloud audio."""

    name: str = "wakeword"
    #: True ONLY for a genuine acoustic keyword spotter (not VAD/ASR/fallback)
    is_real_wake_word: bool = False
    #: "REAL_WAKE_WORD" | "DEVELOPMENT_FALLBACK" | "DISABLED"
    mode: str = "DEVELOPMENT_FALLBACK"

    @abstractmethod
    def start(self) -> bool:
        """Allocate resources (open mic stream / load model)."""

    @abstractmethod
    def stop(self) -> None:
        """Release resources."""

    @abstractmethod
    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Block until the trigger phrase is heard (True) or timeout/stop (False)."""

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "mode": self.mode,
            "is_real_wake_word": self.is_real_wake_word,
            "phrase": settings.WAKE_WORD_PHRASE,
            "available": True,
        }


class OpenWakeWordProvider(WakeWordProvider):
    """
    REAL, fully-local acoustic wake-word detector backed by openWakeWord.

    Audio is processed on-device via an ONNX model; nothing is sent to any cloud
    service. If the optional dependency (or a model) is missing, `start()` fails
    cleanly and the factory falls back to an explicit development fallback.
    """

    name = "openwakeword"
    is_real_wake_word = True
    mode = "REAL_WAKE_WORD"

    def __init__(
        self,
        model_path: Optional[str] = None,
        pretrained: Optional[str] = None,
        threshold: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ):
        self.model_path = model_path if model_path is not None else settings.WAKE_WORD_MODEL_PATH
        self.pretrained = pretrained or settings.WAKE_WORD_PRETRAINED
        self.threshold = float(threshold if threshold is not None else settings.WAKE_WORD_THRESHOLD)
        self.sample_rate = int(sample_rate or settings.ASR_SAMPLE_RATE)
        # openWakeWord models expect 16 kHz audio in 80 ms (1280 sample) frames
        self._frame_samples = 1280
        self._model = None
        self._running = False
        self._last_error: Optional[str] = None

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from openwakeword.model import Model  # type: ignore
        except Exception as exc:  # optional dependency not installed
            self._last_error = f"openwakeword unavailable: {exc}"
            logger.warning(
                "[WAKE] openWakeWord engine not importable (%s). "
                "Falling back to an explicit DEVELOPMENT FALLBACK wake mode.",
                exc,
            )
            return False

        try:
            if self.model_path:
                self._model = Model(wakeword_models=[self.model_path])
            else:
                self._model = Model(wakeword_models=[self.pretrained])
            logger.info(
                "[WAKE] Local openWakeWord model loaded (%s). Phrase target: '%s'",
                self.model_path or self.pretrained,
                settings.WAKE_WORD_PHRASE,
            )
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.error("[WAKE] Failed to load openWakeWord model: %s", exc)
            return False

    def start(self) -> bool:
        self._running = True
        return self._load_model()

    def stop(self) -> None:
        self._running = False
        self._model = None

    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        if not self._load_model():
            return False

        import sounddevice as sd  # local dependency, already used by ASR

        start = time.time()
        logger.info("[WAKE] Listening for local wake phrase '%s'...", settings.WAKE_WORD_PHRASE)
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._frame_samples,
            ) as stream:
                while self._running:
                    if stop_flag is not None and stop_flag():
                        return False
                    if timeout is not None and (time.time() - start) >= timeout:
                        return False
                    frames, _ = stream.read(self._frame_samples)
                    pcm = np.asarray(frames[:, 0], dtype=np.int16)
                    scores = self._model.predict(pcm)
                    if scores and max(scores.values()) >= self.threshold:
                        logger.info("[WAKE] Wake phrase detected (score=%.2f).", max(scores.values()))
                        return True
        except Exception as exc:
            logger.exception("[WAKE] Wake-word audio capture failed: %s", exc)
            return False
        return False

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "mode": self.mode,
            "is_real_wake_word": True,
            "phrase": settings.WAKE_WORD_PHRASE,
            "engine": "openwakeword (local/offline)",
            "model": self.model_path or self.pretrained,
            "threshold": self.threshold,
            "available": self._model is not None,
            "last_error": self._last_error,
        }


class KeyboardWakeWordProvider(WakeWordProvider):
    """
    DEVELOPMENT FALLBACK ONLY — press Enter (or push-to-talk) to activate a session.

    This is emphatically NOT a real wake word. `is_real_wake_word` is False and the
    mode string says so. It exists so laptop development/CI works without an
    acoustic engine or microphone.
    """

    name = "keyboard"
    is_real_wake_word = False
    mode = "DEVELOPMENT_FALLBACK"

    def start(self) -> bool:
        logger.warning(
            "[WAKE] Using DEVELOPMENT FALLBACK wake mode (keyboard/push-to-talk). "
            "This is NOT a real wake word."
        )
        return True

    def stop(self) -> None:
        return None

    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        # Non-interactive contexts (pytest / no console) cannot block on input;
        # fall back to a short poll so the caller's stop_flag still terminates.
        if not sys.stdin or not sys.stdin.isatty():
            end = time.time() + (timeout if timeout is not None else 1.0)
            while time.time() < end:
                if stop_flag is not None and stop_flag():
                    return False
                time.sleep(0.05)
            return False

        try:
            if sys.platform.startswith("win"):
                import msvcrt
                while True:
                    if stop_flag is not None and stop_flag():
                        return False
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"\r", b"\n", b" "):
                            logger.info("[WAKE] (dev fallback) session activated by key press.")
                            return True
                    time.sleep(0.03)
            else:
                import select
                while True:
                    if stop_flag is not None and stop_flag():
                        return False
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        sys.stdin.readline()
                        logger.info("[WAKE] (dev fallback) session activated by Enter.")
                        return True
        except Exception as exc:
            logger.debug("[WAKE] keyboard fallback error: %s", exc)
            return False


class DisabledWakeWordProvider(WakeWordProvider):
    """
    Wake word intentionally disabled — the voice session gate is always open.

    Useful for competition demos and automated tests. Clearly NOT a wake word.
    """

    name = "disabled"
    is_real_wake_word = False
    mode = "DISABLED"

    def start(self) -> bool:
        logger.warning("[WAKE] Wake word disabled. Voice session is always active.")
        return True

    def stop(self) -> None:
        return None

    def wait_for_wake(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        if stop_flag is not None and stop_flag():
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


def create_wake_word_provider(mode: Optional[str] = None, **kwargs) -> WakeWordProvider:
    """
    Factory. NEVER silently pretends to be a real wake word:
      auto         -> openWakeWord if available, else explicit keyboard fallback
      openwakeword -> force real local engine (may still fall back if unavailable)
      keyboard     -> explicit development fallback
      disabled     -> gate disabled
    """
    if not settings.WAKE_WORD_ENABLED:
        return DisabledWakeWordProvider()

    m = (mode or settings.WAKE_WORD_MODE or "auto").lower().strip()

    if m in ("keyboard", "push_to_talk", "pushtotalk"):
        return KeyboardWakeWordProvider()
    if m in ("disabled", "off", "none"):
        return DisabledWakeWordProvider()

    # auto / openwakeword -> attempt the real local engine
    provider = OpenWakeWordProvider(**kwargs)
    if provider.start():
        return provider

    if m == "openwakeword":
        logger.error(
            "[WAKE] WAKE_WORD_MODE=openwakeword but the local engine is unavailable. "
            "Falling back to DEVELOPMENT FALLBACK (keyboard). Real wake word is NOT active."
        )
    return KeyboardWakeWordProvider()


# Module-level convenience singleton (mirrors tts_engine / asr_engine convention).
wake_word_provider = create_wake_word_provider()
