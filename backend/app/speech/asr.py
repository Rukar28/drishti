"""
VisionMate local ASR: microphone VAD capture + Faster-Whisper + intent parser.

Whisper is loaded once and reused. Capture is adaptive (speech start → silence end),
not a fixed 3-second window.
"""

import logging
import re
from typing import Callable, Optional, Dict, Any

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from backend.app.core.config import settings
from backend.app.core.interfaces import ASRProvider
from backend.app.speech.vad import EnergyVAD


logger = logging.getLogger("visionmate.asr")


class LocalASRProvider(ASRProvider):
    """Microphone + Faster-Whisper ASR + VisionMate intent parser."""

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None,
        sample_rate: int = None,
    ):
        self.model_size = model_size or settings.ASR_MODEL
        self.device = device or settings.ASR_DEVICE
        self.compute_type = compute_type or settings.ASR_COMPUTE_TYPE
        self.sample_rate = int(sample_rate or settings.ASR_SAMPLE_RATE)
        self.model: Optional[WhisperModel] = None
        self.vad = EnergyVAD(sample_rate=self.sample_rate)

    def preload(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        logger.info(
            "[VOICE] Loading Faster-Whisper model '%s' on %s (%s) once...",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("[VOICE] Faster-Whisper model ready.")

    def listen_chunk(self, pcm_audio: np.ndarray) -> Optional[str]:
        if pcm_audio is None or len(pcm_audio) == 0:
            return None

        self._load_model()
        audio = np.asarray(pcm_audio, dtype=np.float32).flatten()

        try:
            segments, info = self.model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()
            if text:
                logger.info(
                    "[VOICE] ASR: '%s' (language=%s, probability=%.2f)",
                    text,
                    info.language,
                    info.language_probability,
                )
                return text
            return None
        except Exception as exc:
            logger.exception("[VOICE] ASR transcription failed: %s", exc)
            return None

    def capture_utterance(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        device: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Record until VAD detects end-of-speech. Returns float32 mono 16 kHz audio."""
        self.vad.reset()
        frame_samples = self.vad.frame_samples
        logger.info("[VOICE] Listening (VAD, %d Hz)...", self.sample_rate)
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=frame_samples,
                device=device,
            ) as stream:
                while True:
                    if stop_flag is not None and stop_flag():
                        return None
                    frames, overflowed = stream.read(frame_samples)
                    if overflowed:
                        logger.debug("[VOICE] input overflow")
                    utterance = self.vad.accept_frame(frames[:, 0])
                    if utterance is not None:
                        logger.info("[VOICE] Utterance captured (%.2fs)", len(utterance) / float(self.sample_rate))
                        return utterance
        except Exception as exc:
            logger.exception("[VOICE] Microphone capture failed: %s", exc)
            return None

    def listen_utterance(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        device: Optional[int] = None,
    ) -> Optional[str]:
        audio = self.capture_utterance(stop_flag=stop_flag, device=device)
        if audio is None:
            return None
        return self.listen_chunk(audio)

    def listen_once(
        self,
        duration: float = 3.0,
        device: Optional[int] = None,
    ) -> Optional[str]:
        """Legacy fixed-window capture. Prefer listen_utterance()."""
        return self.listen_utterance(device=device)

    def parse_intent(self, text: str) -> Dict[str, Any]:
        """Parse transcribed speech into a VisionMate command. Single parser."""
        if not text or not text.strip():
            return {"intent": "UNKNOWN", "confidence": 0.0}

        raw = text.strip()
        q = raw.lower()
        q_clean = re.sub(r"[^\w\s]", "", q)
        q_clean = re.sub(r"\s+", " ", q_clean).strip()

        stop_patterns = [
            r"\bstop\b",
            r"\bcancel\b",
            r"\bquiet\b",
            r"\bhalt\b",
            r"\bshut up\b",
            r"\bbe quiet\b",
        ]
        for pat in stop_patterns:
            if re.search(pat, q_clean):
                return {"intent": "STOP", "mode": "stop", "raw_text": raw, "confidence": 0.98}

        find_patterns = [
            r"^(?:can you\s+)?(?:please\s+)?find\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+)$",
            r"^(?:locate|search for|where is|wheres)\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+)$",
        ]
        for pat in find_patterns:
            m = re.match(pat, q_clean)
            if m:
                target = m.group(1).strip()
                if target:
                    return {
                        "intent": "FIND",
                        "mode": "find",
                        "target": target,
                        "raw_text": raw,
                        "confidence": 0.95,
                    }

        read_patterns = [
            r"\bread\s+(?:this|the\s+text|text|the\s+sign|sign|full|document)\b",
            r"\bwhat does this (?:say|sign say)\b",
            r"\bread\b",
        ]
        for pat in read_patterns:
            if re.search(pat, q_clean):
                is_full = "full" in q_clean or "document" in q_clean
                return {
                    "intent": "READ",
                    "mode": "read",
                    "full_reading": is_full,
                    "raw_text": raw,
                    "confidence": 0.95,
                }

        color_m = (
            re.search(r"\bwhat(?:'s|s| is)? (?:the )?colou?r\b(.*)$", q_clean)
            or re.search(
                r"\b(?:which|tell me the|show me the) colou?r\b(?:\s+of\b)?(.*)$",
                q_clean,
            )
        )
        if color_m or "what color" in q_clean:
            tail = (color_m.group(1) if color_m else "") or ""
            tail = re.sub(r"^(?:\s+is|\s+of|\s+are)\b", "", tail)
            tail = re.sub(r"^(?:\s+(?:the|my|this|that|a|an|it))+\b", "", tail)
            tail = re.sub(r"\b(please|now|exactly)\b$", "", tail).strip()
            return {
                "intent": "COLOR",
                "target": tail or None,
                "raw_text": raw,
                "confidence": 0.9,
            }

        if "medicine" in q_clean or "pill" in q_clean:
            return {"intent": "MEDICINE", "raw_text": raw, "confidence": 0.9}

        if any(w in q_clean for w in ("currency", "rupee", "note", "banknote")):
            return {"intent": "CURRENCY", "raw_text": raw, "confidence": 0.9}

        if re.search(r"\b(who is this|do i know this person)\b", q_clean):
            return {"intent": "FACE", "raw_text": raw, "confidence": 0.9}

        if re.search(r"\b(emergency|help me|call my caregiver)\b", q_clean):
            return {"intent": "SOS", "raw_text": raw, "confidence": 0.95}

        if re.search(r"\b(navigate to|take me to|go home|start navigation|stop navigation|where am i)\b", q_clean):
            dest = None
            m = re.search(r"(?:navigate to|take me to)\s+(.+)$", q_clean)
            if m:
                dest = m.group(1).strip()
            return {"intent": "NAVIGATION", "destination": dest, "raw_text": raw, "confidence": 0.9}

        ask_patterns = [
            r"what am i looking at",
            r"explain the scene",
            r"describe everything",
            r"describe the scene",
            r"what is in front of me",
            r"whats in front of me",
            r"what is around me",
            r"whats around me",
            r"what is that",
            r"what is this",
        ]
        for pat in ask_patterns:
            if pat in q_clean:
                return {"intent": "ASK", "mode": "ask", "query": raw, "confidence": 0.90}

        if any(word in q_clean for word in ("guidance", "awareness", "resume", "normal mode", "resume guidance")):
            return {"intent": "GUIDANCE", "mode": "guidance", "raw_text": raw, "confidence": 0.90}

        if q_clean.startswith(("what", "who", "where", "how", "why", "describe", "tell me")):
            return {"intent": "ASK", "mode": "ask", "query": raw, "confidence": 0.80}

        return {"intent": "UNKNOWN", "raw_text": raw, "confidence": 0.40}


asr_engine = LocalASRProvider()
