"""
Adaptive energy VAD for microphone capture.

Speech starts when RMS exceeds the start threshold, continues while speech is
present, and ends after a short silence hangover. No fixed 3-second window.
"""

from collections import deque
from typing import Deque, Optional
import numpy as np

from backend.app.core.config import settings


class EnergyVAD:
    def __init__(
        self,
        sample_rate: int = None,
        frame_ms: int = None,
        start_rms: float = None,
        end_rms: float = None,
        min_speech_ms: int = None,
        silence_end_ms: int = None,
        max_utterance_ms: int = None,
        pre_roll_ms: int = None,
    ):
        self.sample_rate = int(sample_rate or settings.ASR_SAMPLE_RATE)
        self.frame_ms = int(frame_ms or settings.VAD_FRAME_MS)
        self.start_rms = float(start_rms if start_rms is not None else settings.VAD_SPEECH_START_RMS)
        self.end_rms = float(end_rms if end_rms is not None else settings.VAD_SPEECH_END_RMS)
        self.min_speech_ms = int(min_speech_ms or settings.VAD_MIN_SPEECH_MS)
        self.silence_end_ms = int(silence_end_ms or settings.VAD_SILENCE_END_MS)
        self.max_utterance_ms = int(max_utterance_ms or settings.VAD_MAX_UTTERANCE_MS)
        self.pre_roll_ms = int(pre_roll_ms or settings.VAD_PRE_ROLL_MS)

        self.frame_samples = max(1, int(self.sample_rate * self.frame_ms / 1000.0))
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance_ms = 0
        pre_frames = max(1, int(self.pre_roll_ms / self.frame_ms))
        self._pre_roll: Deque[np.ndarray] = deque(maxlen=pre_frames)
        self._speech_frames: list = []

    def reset(self) -> None:
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance_ms = 0
        self._pre_roll.clear()
        self._speech_frames = []

    def rms(self, frame: np.ndarray) -> float:
        audio = np.asarray(frame, dtype=np.float32).flatten()
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def is_speech(self, frame: np.ndarray) -> bool:
        energy = self.rms(frame)
        if self._in_speech:
            return energy >= self.end_rms
        return energy >= self.start_rms

    def accept_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed one PCM frame. Returns a complete utterance array when speech ends,
        otherwise None.
        """
        audio = np.asarray(frame, dtype=np.float32).flatten()
        speaking = self.is_speech(audio)

        if not self._in_speech:
            self._pre_roll.append(audio)
            if speaking:
                self._in_speech = True
                self._speech_ms = self.frame_ms
                self._silence_ms = 0
                self._utterance_ms = self.frame_ms
                self._speech_frames = list(self._pre_roll)
            return None

        self._speech_frames.append(audio)
        self._utterance_ms += self.frame_ms
        if speaking:
            self._speech_ms += self.frame_ms
            self._silence_ms = 0
        else:
            self._silence_ms += self.frame_ms

        ended = False
        if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.silence_end_ms:
            ended = True
        if self._utterance_ms >= self.max_utterance_ms:
            ended = True

        if not ended:
            return None

        utterance = np.concatenate(self._speech_frames) if self._speech_frames else audio
        self.reset()
        if utterance.size < int(self.sample_rate * (self.min_speech_ms / 1000.0) * 0.5):
            return None
        return utterance
