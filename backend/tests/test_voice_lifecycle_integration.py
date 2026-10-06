"""Controlled audio boundaries; actual parser, bus, color handler and events."""
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from backend.app.core.config import settings
from backend.app.core.events import event_broker
from backend.app.services.pipeline import VisionMatePipeline
from backend.app.speech.asr import LocalASRProvider
from backend.app.speech.voice_state import VoiceState, VoiceStateMachine
from backend.app.intelligence.command_bus import CommandBus
from backend.app.modes.color import ColorModeHandler
from backend.app.speech.tts import WindowsSAPITTSProvider


@pytest.mark.parametrize('phrase,cancel,expected', [
    ('What color is this?', False, 2),
    ('Hey Mycroft', False, 0),
    ('What color is this?', True, 0),
])
def test_two_color_sessions_wait_for_speech_before_next_wake(monkeypatch, phrase, cancel, expected):
    monkeypatch.setattr(settings, 'WAKE_WORD_ACK_TEXT', '')
    pipe = VisionMatePipeline.__new__(VisionMatePipeline)
    pipe._voice_running = True
    pipe.voice_fsm = VoiceStateMachine(pipe._on_voice_state_change)
    pipe.world_state_mgr = Mock()
    pipe.world_state_mgr.get_snapshot.return_value.get_active_objects.return_value = []
    pipe.camera = SimpleNamespace(get_latest_frame=lambda: np.full((80,80,3), (0,0,255), dtype=np.uint8))
    pipe.color_handler = ColorModeHandler()
    pending = [0]
    spoken = []
    def speak(text, **kwargs):
        spoken.append(text)
        pending[0] = 4
    def speaking():
        pending[0] = max(0, pending[0]-1)
        return pending[0] > 0
    pipe.tts = SimpleNamespace(speech_generation=1, is_busy=lambda: pending[0]>0,
                               is_speaking=speaking, speak=speak, stop=lambda: None)
    wakes = []
    def wake(**kwargs):
        assert not pending[0], 'wake capture reopened before playback ended'
        wakes.append(pipe.voice_fsm.value)
        if len(wakes) > 2:
            pipe._voice_running = False
            return False
        return True
    pipe.wake_provider = SimpleNamespace(is_real_wake_word=True, name='controlled audio fixture',
                                        wait_for_wake=wake, suspend_capture=lambda: None)
    def listen(**kwargs):
        assert pipe.voice_fsm.state == VoiceState.LISTENING_FOR_COMMAND
        kwargs['on_processing']()
        if cancel:
            pipe.voice_fsm.abort('STOP during capture')
        return phrase
    pipe.asr = SimpleNamespace(preload=lambda: None, listen_utterance=listen)
    pipe.command_bus = CommandBus(LocalASRProvider().parse_intent)
    pipe.command_bus.register('COLOR', pipe._cmd_color)
    events = event_broker.subscribe()
    try:
        pipe._voice_loop()
        captured = []
        while not events.empty():
            captured.append(events.get_nowait())
        results = [e['data'] for e in captured if e['type']=='VOICE_COMMAND']
        assert len(results)==expected
        assert all(r['intent']=='COLOR' and r['result']['success'] for r in results)
        assert all('red' in r['result']['text'].lower() for r in results)
        assert len(spoken)==expected
        states = [e['data']['state'] for e in captured if e['type']=='VOICE_STATE']
        assert states.count('SPEAKING')==expected
        assert states[-1]=='LISTENING_FOR_WAKE'
        assert states.count('EXECUTING')==expected
    finally:
        event_broker.unsubscribe(events)


@pytest.mark.parametrize('phrase,intent', [
    ('find the bottle','FIND'), ('read this','READ'), ('what is this','ASK'),
    ('what color is this','COLOR'), ('identify this currency','CURRENCY'),
    ('identify this medicine','MEDICINE'), ('take me to the pharmacy','NAVIGATION'),
    ('send SOS','SOS'), ('stop','STOP'),
])
def test_demo_phrases_route_once(phrase,intent):
    bus=CommandBus(LocalASRProvider().parse_intent)
    handler=Mock(return_value={'status':'test boundary'})
    bus.register(intent,handler)
    assert bus.dispatch(phrase)['intent']==intent
    handler.assert_called_once()


def test_tts_queued_work_is_busy_before_worker_starts(monkeypatch):
    monkeypatch.setattr(WindowsSAPITTSProvider,'_start_worker',lambda self: None)
    tts=WindowsSAPITTSProvider()
    assert not tts.is_busy()
    tts.speak('pending')
    assert tts.is_busy() and not tts.is_speaking()
    tts.stop()
    assert tts._queue.empty()


def test_speech_wait_does_not_finish_early():
    pipe=VisionMatePipeline.__new__(VisionMatePipeline)
    pipe._voice_running=True
    done=threading.Event()
    pipe.tts=SimpleNamespace(is_busy=lambda:not done.is_set(),is_speaking=lambda:True)
    pipe.wake_provider=SimpleNamespace(suspend_capture=Mock())
    pipe.voice_fsm=VoiceStateMachine()
    pipe.voice_fsm.force(VoiceState.EXECUTING)
    worker=threading.Thread(target=pipe._wait_for_voice_speech)
    worker.start()
    time.sleep(.08)
    assert worker.is_alive() and pipe.voice_fsm.state==VoiceState.SPEAKING
    done.set();worker.join(1)
    assert not worker.is_alive()


def test_cancelled_capture_never_transcribes(monkeypatch):
    asr=LocalASRProvider()
    monkeypatch.setattr(asr,'capture_utterance',lambda **kw:np.ones(100))
    transcribe=Mock()
    monkeypatch.setattr(asr,'listen_chunk',transcribe)
    assert asr.listen_utterance(stop_flag=lambda:True) is None
    transcribe.assert_not_called()
