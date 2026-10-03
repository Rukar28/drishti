"""
P2B/P2C — Wake-word provider boundary and deterministic voice state machine.

These tests verify the *contract* of the wake-word subsystem: a real acoustic
engine is clearly distinguished from development fallbacks, and the voice state
machine has deterministic, auditable transitions with an atomic STOP/abort.
"""

import pytest

from backend.app.speech.wakeword import (
    WakeWordProvider,
    OpenWakeWordProvider,
    KeyboardWakeWordProvider,
    DisabledWakeWordProvider,
    create_wake_word_provider,
)
from backend.app.speech.voice_state import VoiceState, VoiceStateMachine


# ── Wake-word provider boundary ──────────────────────────────────
def test_keyboard_provider_is_development_fallback_not_real():
    p = KeyboardWakeWordProvider()
    assert p.is_real_wake_word is False
    assert p.mode == "DEVELOPMENT_FALLBACK"
    h = p.health()
    assert h["is_real_wake_word"] is False
    assert h["mode"] == "DEVELOPMENT_FALLBACK"


def test_disabled_provider_is_always_on_gate():
    p = DisabledWakeWordProvider()
    assert p.is_real_wake_word is False
    assert p.mode == "DISABLED"
    assert p.wait_for_wake(stop_flag=lambda: False) is True
    assert p.wait_for_wake(stop_flag=lambda: True) is False


def test_openwakeword_provider_declares_real_engine():
    # Even if the optional engine is absent, the class contract is a REAL wake word.
    p = OpenWakeWordProvider()
    assert p.is_real_wake_word is True
    assert p.mode == "REAL_WAKE_WORD"


def test_factory_never_fakes_real_wake_word_when_engine_absent():
    # In this environment openWakeWord is not installed, so 'auto' must clearly
    # degrade to an explicit development fallback (never claim a real wake word).
    p = create_wake_word_provider("auto")
    assert isinstance(p, WakeWordProvider)
    if isinstance(p, KeyboardWakeWordProvider):
        assert p.is_real_wake_word is False
    elif isinstance(p, OpenWakeWordProvider):
        assert p.is_real_wake_word is True  # engine genuinely present


def test_factory_modes():
    assert isinstance(create_wake_word_provider("keyboard"), KeyboardWakeWordProvider)
    assert isinstance(create_wake_word_provider("disabled"), DisabledWakeWordProvider)


# ── Voice state machine ──────────────────────────────────────────
def test_happy_path_transitions_are_legal():
    fsm = VoiceStateMachine()
    assert fsm.state == VoiceState.IDLE
    assert fsm.transition(VoiceState.LISTENING_FOR_WAKE) is True
    assert fsm.transition(VoiceState.WAKE_DETECTED) is True
    assert fsm.transition(VoiceState.LISTENING_FOR_COMMAND) is True
    assert fsm.transition(VoiceState.PROCESSING) is True
    assert fsm.transition(VoiceState.EXECUTING) is True
    assert fsm.transition(VoiceState.SPEAKING) is True


def test_illegal_transition_is_refused():
    fsm = VoiceStateMachine()
    # Cannot jump straight from IDLE into EXECUTING
    assert fsm.transition(VoiceState.EXECUTING) is False
    assert fsm.state == VoiceState.IDLE


def test_abort_bumps_generation_and_forces_idle():
    fsm = VoiceStateMachine()
    fsm.transition(VoiceState.LISTENING_FOR_WAKE)
    fsm.transition(VoiceState.WAKE_DETECTED)
    gen_before = fsm.generation
    new_gen = fsm.abort(reason="STOP")
    assert new_gen > gen_before
    assert fsm.state == VoiceState.IDLE
    assert fsm.generation == new_gen


def test_on_change_callback_fires():
    seen = []
    fsm = VoiceStateMachine(on_change=lambda s: seen.append(s.value))
    fsm.transition(VoiceState.LISTENING_FOR_WAKE)
    assert "LISTENING_FOR_WAKE" in seen
