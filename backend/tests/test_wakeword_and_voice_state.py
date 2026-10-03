"""
VisionMate / Drishti
Wake-word + Voice State Machine Tests

These tests verify:

1. Hey Mycroft configuration
2. Real-vs-fallback distinction
3. openWakeWord provider behavior
4. Acoustic inference
5. Threshold behavior
6. Microphone ownership
7. Reset behavior
8. Voice FSM transitions
9. Wake word does not directly execute commands
10. Wake word does not use ASR transcript matching

The real openWakeWord package is NOT required for the hermetic tests.
"""

from __future__ import annotations

import inspect
import types
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

import backend.app.speech.wakeword as wake_word_module

from backend.app.core.config import Settings, settings

from backend.app.speech.wakeword import (
    WakeWordProvider,
    WakeWordResult,
    OpenWakeWordProvider,
    KeyboardWakeWordProvider,
    DisabledWakeWordProvider,
    create_wake_word_provider,
)

from backend.app.speech.voice_state import (
    VoiceState,
    VoiceStateMachine,
)


# ============================================================================
# FAKE ACOUSTIC MODEL
# ============================================================================

class _FakeAcousticModel:
    """
    Deterministic fake openWakeWord model.

    It behaves like:

        openwakeword.model.Model

    but requires no microphone, model download, or network.
    """

    def __init__(
        self,
        scripted: Optional[List[Any]] = None,
    ) -> None:

        self._scripted = list(
            scripted or []
        )

        self.seen_chunks: List[
            np.ndarray
        ] = []

        self.reset_count = 0

    def predict(
        self,
        chunk,
    ):

        self.seen_chunks.append(
            np.asarray(chunk)
        )

        if self._scripted:

            result = self._scripted.pop(0)

        else:

            result = {
                "hey_mycroft_v0.1": 0.0
            }

        if isinstance(
            result,
            Exception,
        ):

            raise result

        return result

    def reset(self):

        self.reset_count += 1


# ============================================================================
# FAKE MICROPHONE STREAM
# ============================================================================

class _FakeStream:
    """
    Fake sounddevice InputStream.

    No physical microphone required.
    """

    def __init__(
        self,
        chunks: Optional[
            List[np.ndarray]
        ] = None,
    ) -> None:

        self._chunks = list(
            chunks or []
        )

        self.closed = False

        self.read_calls = 0

    def read(
        self,
        sample_count: int,
    ):

        self.read_calls += 1

        if self._chunks:

            chunk = self._chunks.pop(0)

        else:

            chunk = np.zeros(
                sample_count,
                dtype=np.int16,
            )

        chunk = np.asarray(
            chunk,
            dtype=np.int16,
        )

        return (
            chunk.reshape(-1, 1),
            False,
        )

    def stop(self):
        pass

    def close(self):

        self.closed = True


# ============================================================================
# TEST HELPERS
# ============================================================================

def _install_fake_engine(
    monkeypatch,
    model: _FakeAcousticModel,
) -> None:

    def fake_import():

        fake_openwakeword = types.SimpleNamespace()

        def fake_model(
            wakeword_models,
            inference_framework="onnx",
        ):

            return model

        return (
            fake_openwakeword,
            fake_model,
        )

    monkeypatch.setattr(
        wake_word_module,
        "_try_import_engine",
        fake_import,
    )

    # Avoid actual model download/path resolution.
    monkeypatch.setattr(
        wake_word_module,
        "_download_pretrained_model",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        wake_word_module,
        "_resolve_pretrained_model_path",
        lambda *_args, **_kwargs: (
            "C:/fake/hey_mycroft_v0.1.onnx"
        ),
    )


def _make_provider(
    **overrides: Any,
) -> OpenWakeWordProvider:

    kwargs = {
        "pretrained": "hey_mycroft",
        "threshold": 0.5,
        "sample_rate": 16000,
    }

    kwargs.update(overrides)

    return OpenWakeWordProvider(
        **kwargs
    )


def _attach_fake_stream(
    monkeypatch,
    provider: OpenWakeWordProvider,
    stream: _FakeStream,
) -> None:

    monkeypatch.setattr(
        provider,
        "_ensure_stream",
        lambda: True,
    )

    provider._stream = stream


# ============================================================================
# CONFIGURATION
# ============================================================================

def test_configuration_targets_hey_mycroft():

    fields = Settings.model_fields

    assert (
        fields[
            "WAKE_WORD_PHRASE"
        ].default
        == "hey mycroft"
    )

    assert (
        fields[
            "WAKE_WORD_PRETRAINED"
        ].default
        == "hey_mycroft"
    )

    assert (
        fields[
            "WAKE_WORD_PRETRAINED"
        ].default.replace(
            "_",
            " ",
        )
        == fields[
            "WAKE_WORD_PHRASE"
        ].default
    )

    assert (
        fields[
            "WAKE_WORD_PRETRAINED"
        ].default
        != "alexa"
    )

    assert (
        fields[
            "WAKE_WORD_MODE"
        ].default
        == "real"
    )

    threshold = fields[
        "WAKE_WORD_THRESHOLD"
    ].default

    assert 0.0 < threshold <= 1.0


# ============================================================================
# PROVIDER BASICS
# ============================================================================

def test_provider_defaults():

    provider = OpenWakeWordProvider()

    assert (
        provider.pretrained
        == settings.WAKE_WORD_PRETRAINED
    )

    assert (
        provider.threshold
        == settings.WAKE_WORD_THRESHOLD
    )

    assert (
        provider.sample_rate
        == settings.ASR_SAMPLE_RATE
    )

    assert (
        provider.is_real_wake_word
        is True
    )

    assert (
        provider.mode
        == "REAL_WAKE_WORD"
    )

    # Lazy initialization:
    # constructor must not load the model.
    assert (
        provider.is_available()
        is False
    )

    assert provider._model is None


def test_openwakeword_is_real_provider():

    provider = OpenWakeWordProvider()

    assert (
        provider.is_real_wake_word
        is True
    )

    assert (
        provider.mode
        == "REAL_WAKE_WORD"
    )


# ============================================================================
# FALLBACK PROVIDERS
# ============================================================================

def test_keyboard_is_not_real_wake_word():

    provider = (
        KeyboardWakeWordProvider()
    )

    assert (
        provider.is_real_wake_word
        is False
    )

    assert (
        provider.mode
        == "DEVELOPMENT_FALLBACK"
    )

    health = provider.health()

    assert (
        health[
            "is_real_wake_word"
        ]
        is False
    )


def test_disabled_provider():

    provider = (
        DisabledWakeWordProvider()
    )

    assert (
        provider.is_real_wake_word
        is False
    )

    assert (
        provider.mode
        == "DISABLED"
    )

    assert (
        provider.wait_for_wake(
            stop_flag=lambda: False
        )
        is True
    )

    assert (
        provider.wait_for_wake(
            stop_flag=lambda: True
        )
        is False
    )


# ============================================================================
# MISSING DEPENDENCY
# ============================================================================

def test_missing_engine_fails_loudly(
    monkeypatch,
):

    def missing_engine():

        raise ImportError(
            "No module named 'openwakeword'"
        )

    monkeypatch.setattr(
        wake_word_module,
        "_try_import_engine",
        missing_engine,
    )

    provider = _make_provider()

    assert provider.start() is False

    assert provider.last_error

    assert (
        "openwakeword"
        in provider.last_error.lower()
    )

    assert (
        provider.is_available()
        is False
    )

    health = provider.health()

    assert (
        health[
            "is_real_wake_word"
        ]
        is True
    )

    assert (
        health["mode"]
        == "REAL_WAKE_WORD"
    )


# ============================================================================
# FACTORY
# ============================================================================

def test_factory_modes():

    assert isinstance(
        create_wake_word_provider(
            "keyboard"
        ),
        KeyboardWakeWordProvider,
    )

    assert isinstance(
        create_wake_word_provider(
            "fallback"
        ),
        KeyboardWakeWordProvider,
    )

    assert isinstance(
        create_wake_word_provider(
            "disabled"
        ),
        DisabledWakeWordProvider,
    )

    assert isinstance(
        create_wake_word_provider(
            "real"
        ),
        OpenWakeWordProvider,
    )


def test_factory_auto_fallback_when_missing(
    monkeypatch,
):

    def missing():

        raise ImportError(
            "No module named 'openwakeword'"
        )

    monkeypatch.setattr(
        wake_word_module,
        "_try_import_engine",
        missing,
    )

    provider = create_wake_word_provider(
        "auto"
    )

    assert isinstance(
        provider,
        KeyboardWakeWordProvider,
    )

    assert (
        provider.is_real_wake_word
        is False
    )


# ============================================================================
# REAL ACOUSTIC RESULT BEHAVIOR
# ============================================================================

def test_detection_returns_structured_result(
    monkeypatch,
):

    model = _FakeAcousticModel(
        [
            {
                "hey_mycroft_v0.1": 0.87
            }
        ]
    )

    _install_fake_engine(
        monkeypatch,
        model,
    )

    provider = _make_provider()

    assert provider.start() is True

    result = provider.process_audio(
        np.zeros(
            1280,
            dtype=np.int16,
        )
    )

    assert isinstance(
        result,
        WakeWordResult,
    )

    assert (
        result.detected
        is True
    )

    assert (
        result.score
        == pytest.approx(0.87)
    )

    assert (
        result.phrase
        == "hey mycroft"
    )

    assert (
        model.seen_chunks[0].dtype
        == np.int16
    )

    assert (
        model.reset_count
        == 1
    )

    assert (
        provider.last_result
        is result
    )


def test_score_below_threshold_does_not_trigger(
    monkeypatch,
):

    model = _FakeAcousticModel(
        [
            {
                "hey_mycroft_v0.1": 0.42
            }
        ]
    )

    _install_fake_engine(
        monkeypatch,
        model,
    )

    provider = _make_provider(
        threshold=0.5
    )

    assert provider.start() is True

    result = provider.process_audio(
        np.zeros(
            1280,
            dtype=np.int16,
        )
    )

    assert (
        result.detected
        is False
    )

    assert (
        result.score
        == pytest.approx(0.42)
    )

    assert (
        model.reset_count
        == 0
    )


def test_threshold_is_configurable(
    monkeypatch,
):

    model = _FakeAcousticModel(
        [
            {
                "hey_mycroft_v0.1": 0.42
            }
        ]
    )

    _install_fake_engine(
        monkeypatch,
        model,
    )

    provider = _make_provider(
        threshold=0.3
    )

    assert provider.start() is True

    result = provider.process_audio(
        np.zeros(
            1280,
            dtype=np.int16,
        )
    )

    assert (
        result.detected
        is True
    )


# ============================================================================
# MALFORMED MODEL OUTPUT
# ============================================================================

def test_malformed_model_output_is_safe(
    monkeypatch,
):

    bad_outputs = [
        None,
        "not-a-dict",
        {
            "hey_mycroft_v0.1":
            "invalid"
        },
        {
            "hey_mycroft_v0.1":
            float("nan")
        },
        RuntimeError(
            "engine exploded"
        ),
    ]

    for bad_output in bad_outputs:

        model = _FakeAcousticModel(
            [bad_output]
        )

        _install_fake_engine(
            monkeypatch,
            model,
        )

        provider = _make_provider()

        assert (
            provider.start()
            is True
        )

        result = provider.process_audio(
            np.zeros(
                1280,
                dtype=np.int16,
            )
        )

        assert (
            result.detected
            is False
        )

        assert result.error


# ============================================================================
# MICROPHONE LOOP
# ============================================================================

def test_wait_for_wake_detects_and_releases_mic(
    monkeypatch,
):

    model = _FakeAcousticModel(
        [
            {
                "hey_mycroft_v0.1": 0.02
            },
            {
                "hey_mycroft_v0.1": 0.03
            },
            {
                "hey_mycroft_v0.1": 0.95
            },
        ]
    )

    _install_fake_engine(
        monkeypatch,
        model,
    )

    provider = _make_provider()

    assert provider.start() is True

    stream = _FakeStream(
        [
            np.zeros(
                1280,
                dtype=np.int16,
            )
            for _ in range(3)
        ]
    )

    _attach_fake_stream(
        monkeypatch,
        provider,
        stream,
    )

    assert (
        provider.wait_for_wake(
            stop_flag=lambda: False,
            timeout=5.0,
        )
        is True
    )

    assert (
        provider.last_result.detected
        is True
    )

    # Microphone must be released
    # before ASR/VAD starts.
    assert (
        provider._stream
        is None
    )

    assert (
        stream.closed
        is True
    )

    assert (
        model.reset_count
        == 1
    )


def test_wait_for_wake_timeout_keeps_stream(
    monkeypatch,
):

    model = _FakeAcousticModel()

    _install_fake_engine(
        monkeypatch,
        model,
    )

    provider = _make_provider()

    assert provider.start() is True

    stream = _FakeStream()

    _attach_fake_stream(
        monkeypatch,
        provider,
        stream,
    )

    assert (
        provider.wait_for_wake(
            stop_flag=lambda: False,
            timeout=0.05,
        )
        is False
    )

    assert (
        stream.closed
        is False
    )

    assert (
        provider.wait_for_wake(
            stop_flag=lambda: False,
            timeout=0.05,
        )
        is False
    )

    assert (
        stream.read_calls
        >= 1
    )


def test_shutdown_releases_microphone(
    monkeypatch,
):

    model = _FakeAcousticModel()

    _install_fake_engine(
        monkeypatch,
        model,
    )

    provider = _make_provider()

    assert provider.start() is True

    stream = _FakeStream()

    _attach_fake_stream(
        monkeypatch,
        provider,
        stream,
    )

    assert (
        provider.wait_for_wake(
            stop_flag=lambda: True
        )
        is False
    )

    assert (
        provider._stream
        is None
    )

    assert (
        stream.closed
        is True
    )


# ============================================================================
# STOP / RESET
# ============================================================================

def test_stop_releases_model(
    monkeypatch,
):

    model = _FakeAcousticModel(
        [
            {
                "hey_mycroft_v0.1": 0.9
            }
        ]
    )

    _install_fake_engine(
        monkeypatch,
        model,
    )

    provider = _make_provider()

    assert provider.start() is True

    assert (
        provider.is_available()
        is True
    )

    provider.stop()

    assert (
        provider.is_available()
        is False
    )

    assert (
        provider._model
        is None
    )


# ============================================================================
# ACOUSTIC ONLY
# ============================================================================

def test_wakeword_module_never_uses_asr_transcripts():

    source = inspect.getsource(
        wake_word_module
    )

    assert (
        "faster_whisper"
        not in source
    )

    assert (
        "transcribe"
        not in source
    )

    assert (
        "parse_intent"
        not in source
    )

    assert (
        '"hey mycroft" in'
        not in source.lower()
    )

    assert (
        "'hey mycroft' in"
        not in source.lower()
    )


# ============================================================================
# VOICE STATE MACHINE
# ============================================================================

def test_voice_fsm_happy_path():

    fsm = VoiceStateMachine()

    assert (
        fsm.state
        == VoiceState.IDLE
    )

    assert fsm.transition(
        VoiceState.LISTENING_FOR_WAKE
    )

    assert fsm.transition(
        VoiceState.WAKE_DETECTED
    )

    assert fsm.transition(
        VoiceState.LISTENING_FOR_COMMAND
    )

    assert fsm.transition(
        VoiceState.PROCESSING
    )

    assert fsm.transition(
        VoiceState.EXECUTING
    )

    assert fsm.transition(
        VoiceState.SPEAKING
    )


def test_illegal_direct_execution():

    fsm = VoiceStateMachine()

    assert (
        fsm.transition(
            VoiceState.EXECUTING
        )
        is False
    )

    assert (
        fsm.state
        == VoiceState.IDLE
    )


def test_abort_increments_generation():

    fsm = VoiceStateMachine()

    fsm.transition(
        VoiceState.LISTENING_FOR_WAKE
    )

    fsm.transition(
        VoiceState.WAKE_DETECTED
    )

    old_generation = (
        fsm.generation
    )

    new_generation = fsm.abort(
        reason="STOP"
    )

    assert (
        new_generation
        > old_generation
    )

    assert (
        fsm.state
        == VoiceState.IDLE
    )


def test_on_change_callback():

    states = []

    fsm = VoiceStateMachine(
        on_change=lambda state:
            states.append(
                state.value
            )
    )

    fsm.transition(
        VoiceState.LISTENING_FOR_WAKE
    )

    assert (
        "LISTENING_FOR_WAKE"
        in states
    )


# ============================================================================
# WAKE → FSM LIFECYCLE
# ============================================================================

def test_full_voice_session_lifecycle():

    fsm = VoiceStateMachine()

    fsm.force(
        VoiceState.IDLE,
        "test-start",
    )

    assert fsm.transition(
        VoiceState.LISTENING_FOR_WAKE
    )

    assert fsm.transition(
        VoiceState.WAKE_DETECTED
    )

    assert fsm.transition(
        VoiceState.LISTENING_FOR_COMMAND
    )

    assert fsm.transition(
        VoiceState.PROCESSING
    )

    assert fsm.transition(
        VoiceState.EXECUTING
    )

    assert fsm.transition(
        VoiceState.SPEAKING
    )


def test_wake_detection_cannot_execute_command_directly():

    fsm = VoiceStateMachine()

    assert fsm.transition(
        VoiceState.LISTENING_FOR_WAKE
    )

    assert fsm.transition(
        VoiceState.WAKE_DETECTED
    )

    # Wake alone cannot execute a command.
    assert (
        fsm.transition(
            VoiceState.EXECUTING
        )
        is False
    )

    assert (
        fsm.state
        == VoiceState.WAKE_DETECTED
    )

    assert fsm.transition(
        VoiceState.LISTENING_FOR_COMMAND
    )


def test_wake_cannot_repeat_during_command():

    fsm = VoiceStateMachine()

    fsm.transition(
        VoiceState.LISTENING_FOR_WAKE
    )

    fsm.transition(
        VoiceState.WAKE_DETECTED
    )

    fsm.transition(
        VoiceState.LISTENING_FOR_COMMAND
    )

    assert (
        fsm.transition(
            VoiceState.WAKE_DETECTED
        )
        is False
    )

    fsm.transition(
        VoiceState.PROCESSING
    )

    assert (
        fsm.transition(
            VoiceState.WAKE_DETECTED
        )
        is False
    )

    fsm.transition(
        VoiceState.EXECUTING
    )

    assert (
        fsm.transition(
            VoiceState.WAKE_DETECTED
        )
        is False
    )


# ============================================================================
# OPTIONAL REAL MODEL TEST
# ============================================================================

def test_real_openwakeword_model_optional():

    """
    Optional integration test.

    If openWakeWord is not installed, this test skips.

    If installed, it attempts to load the REAL Hey Mycroft model.

    It does not require a microphone.
    """

    pytest.importorskip(
        "openwakeword"
    )

    provider = OpenWakeWordProvider(
        pretrained="hey_mycroft",
        threshold=0.5,
        sample_rate=16000,
    )

    if not provider.start():

        pytest.skip(
            "Real Hey Mycroft model unavailable: "
            f"{provider.last_error}"
        )

    assert (
        provider.is_available()
        is True
    )

    health = provider.health()

    loaded_names = " ".join(
        map(
            str,
            health.get(
                "model_names",
                [],
            ),
        )
    ).lower()

    resolved_path = str(
        health.get(
            "resolved_model_path",
            "",
        )
    ).lower()

    assert (
        "mycroft" in loaded_names
        or "mycroft" in resolved_path
    )

    # Silence must not activate.
    for _ in range(10):

        result = provider.process_audio(
            np.zeros(
                1280,
                dtype=np.int16,
            )
        )

        assert (
            result.detected
            is False
        )

    provider.stop()