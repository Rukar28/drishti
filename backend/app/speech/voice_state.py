"""
VisionMate — Deterministic Voice State Machine (P2C).

Guarantees a single, serialized voice pipeline and prevents:
  * overlapping recordings
  * duplicate command execution
  * ASR running while another command is still executing
  * stale commands surviving a STOP

The pipeline's voice thread is the only caller, so transitions are naturally
serialized; this class additionally makes the legal transition graph explicit and
auditable, and exposes an atomic STOP/abort reset.
"""

from enum import Enum
from typing import Dict, List, Optional
import logging
import threading

logger = logging.getLogger("visionmate.voicestate")


class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING_FOR_WAKE = "LISTENING_FOR_WAKE"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING_FOR_COMMAND = "LISTENING_FOR_COMMAND"
    PROCESSING = "PROCESSING"      # ASR / intent parsing
    EXECUTING = "EXECUTING"        # Command Bus dispatch
    SPEAKING = "SPEAKING"          # TTS output
    COOLDOWN = "COOLDOWN"          # brief settle after TTS / after STOP


# Legal transition graph. Any target implicitly may also be entered from IDLE/COOLDOWN
# via reset paths; STOP may force IDLE/COOLDOWN from anywhere.
_TRANSITIONS: Dict[VoiceState, List[VoiceState]] = {
    VoiceState.IDLE: [
        VoiceState.LISTENING_FOR_WAKE, VoiceState.LISTENING_FOR_COMMAND,
        VoiceState.COOLDOWN, VoiceState.IDLE, VoiceState.SPEAKING,
    ],
    VoiceState.LISTENING_FOR_WAKE: [
        VoiceState.WAKE_DETECTED, VoiceState.IDLE, VoiceState.COOLDOWN, VoiceState.LISTENING_FOR_WAKE, VoiceState.SPEAKING,
    ],
    VoiceState.WAKE_DETECTED: [
        VoiceState.LISTENING_FOR_COMMAND, VoiceState.IDLE, VoiceState.COOLDOWN, VoiceState.SPEAKING,
    ],
    VoiceState.LISTENING_FOR_COMMAND: [
        VoiceState.PROCESSING, VoiceState.LISTENING_FOR_WAKE, VoiceState.IDLE,
        VoiceState.COOLDOWN, VoiceState.LISTENING_FOR_COMMAND,
    ],
    VoiceState.PROCESSING: [
        VoiceState.EXECUTING, VoiceState.LISTENING_FOR_WAKE, VoiceState.IDLE, VoiceState.COOLDOWN,
    ],
    VoiceState.EXECUTING: [
        VoiceState.SPEAKING, VoiceState.LISTENING_FOR_WAKE, VoiceState.IDLE, VoiceState.COOLDOWN,
    ],
    VoiceState.SPEAKING: [
        VoiceState.LISTENING_FOR_WAKE, VoiceState.LISTENING_FOR_COMMAND, VoiceState.IDLE, VoiceState.COOLDOWN,
    ],
    VoiceState.COOLDOWN: [
        VoiceState.LISTENING_FOR_WAKE, VoiceState.LISTENING_FOR_COMMAND, VoiceState.IDLE,
    ],
}

# States from which a new session may begin
SESSION_ENTRY_STATES = {VoiceState.IDLE, VoiceState.COOLDOWN, VoiceState.LISTENING_FOR_WAKE}


class VoiceStateMachine:
    """Thread-safe, deterministic voice state machine with an atomic abort/reset."""

    def __init__(self, on_change=None):
        self._state = VoiceState.IDLE
        self._lock = threading.RLock()
        self._generation = 0
        self._on_change = on_change

    @property
    def state(self) -> VoiceState:
        with self._lock:
            return self._state

    @property
    def value(self) -> str:
        return self.state.value

    @property
    def generation(self) -> int:
        """Incremented on every STOP/abort so late results can be discarded."""
        with self._lock:
            return self._generation

    def can_transition(self, target: VoiceState) -> bool:
        with self._lock:
            if target == self._state:
                return True
            return target in _TRANSITIONS.get(self._state, [])

    def transition(self, target: VoiceState, reason: str = "") -> bool:
        target = VoiceState(target)
        with self._lock:
            if target == self._state:
                return True
            if target not in _TRANSITIONS.get(self._state, []):
                # Illegal transitions are refused (never silently accepted) but logged.
                logger.warning(
                    "[VOICE-FSM] Illegal transition %s -> %s (%s). Refused.",
                    self._state.value, target.value, reason or "no reason",
                )
                return False
            prev = self._state
            self._state = target
        logger.debug("[VOICE-FSM] %s -> %s (%s)", prev.value, target.value, reason)
        if self._on_change is not None:
            try:
                self._on_change(target)
            except Exception as exc:  # never let telemetry break the FSM
                logger.debug("[VOICE-FSM] on_change error: %s", exc)
        return True

    def force(self, target: VoiceState, reason: str = "") -> None:
        """Unconditional transition (used only by abort/STOP paths)."""
        target = VoiceState(target)
        with self._lock:
            prev = self._state
            self._state = target
        logger.info("[VOICE-FSM] FORCE %s -> %s (%s)", prev.value, target.value, reason)
        if self._on_change is not None:
            try:
                self._on_change(target)
            except Exception:
                pass

    def abort(self, reason: str = "STOP") -> int:
        """Atomic STOP: bump generation, force IDLE. Returns the new generation."""
        with self._lock:
            self._generation += 1
            gen = self._generation
        self.force(VoiceState.IDLE, reason=reason)
        return gen

    def is_active(self) -> bool:
        with self._lock:
            return self._state not in (VoiceState.IDLE, VoiceState.LISTENING_FOR_WAKE)

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {"state": self._state.value, "generation": self._generation}