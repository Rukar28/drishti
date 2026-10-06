"""
VisionMate v2 - Low-Latency Priority Speech Engine with Generation Invalidation
Implements TTSProvider using Windows Native SAPI5 / pywin32 with thread-safe
priority queueing, generation IDs, atomic speech preemption (PurgeBeforeSpeak),
and global cancellation.
"""

import queue
import threading
import time
import logging
from typing import Optional, Dict, Any
from backend.app.core.interfaces import TTSProvider
from backend.app.core.events import EventType, event_broker

logger = logging.getLogger("visionmate.tts")

class WindowsSAPITTSProvider(TTSProvider):
    """
    Asynchronous priority-managed Windows SAPI5 TTS engine with Generation Invalidation.
    Guarantees that once STOP is called or a mode changes, stale asynchronous speech is permanently purged.
    """
    def __init__(self, voice_rate: int = 2, volume: int = 100):
        self.voice_rate = voice_rate
        self.volume = volume
        
        # Priority queue stores tuples:
        # (priority_int, generation_id, timestamp, text, interrupt, source)
        # Priority 0 is Emergency, 1 is Navigation, 2 is Interaction
        self._queue = queue.PriorityQueue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_speaking = False
        self._voice = None
        self._lock = threading.Lock()
        
        # Generation counter for atomic invalidation
        self.speech_generation = 1

        # Purge is performed ONLY on the dedicated COM worker thread. Caller
        # threads (API/priority) must never touch the COM object directly, which
        # can otherwise block for the full duration of the current utterance.
        self._purge_requested = False
        self._speaking_until = 0.0

        self._start_worker()

    def _start_worker(self):
        self._running = True
        self._thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._thread.start()

    def _speech_loop(self):
        """Dedicated COM speech worker thread."""
        try:
            import pythoncom
            import win32com.client
            
            pythoncom.CoInitialize()
            self._voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._voice.Rate = self.voice_rate
            self._voice.Volume = self.volume
            logger.info("Windows SAPI5 Voice engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SAPI5 engine: {e}. Running in headless audio mode.")
            self._voice = None

        while self._running:
            # Honor any pending purge on the dedicated COM thread (non-blocking for callers).
            if self._purge_requested:
                self._purge_requested = False
                if self._voice is not None:
                    try:
                        self._voice.Speak("", 2)  # SVSFPurgeBeforeSpeak
                    except Exception:
                        pass
                self._is_speaking = False
                self._speaking_until = 0.0

            try:
                priority, gen_id, t_stamp, text, interrupt, source = self._queue.get(timeout=0.08)
                
                # Check Generation Invalidation: Discard if stale
                with self._lock:
                    if gen_id < self.speech_generation:
                        self._queue.task_done()
                        continue

                if text:
                    logger.info(f"[TTS SPEAK P{priority} G{gen_id} {source}]: '{text}'")
                    try:
                        if self._voice is None:
                            event_broker.publish(EventType.ERROR, {
                                "source": "tts", "message": "Windows speech output is unavailable.",
                            })
                            continue
                        self._is_speaking = True
                        self._voice.Speak(text, 1 | (2 if interrupt else 0))
                        event_broker.publish(EventType.TTS_STARTED, {
                            "text": text, "priority": priority, "source": source,
                            "generation": gen_id,
                        })
                        cancelled = False
                        # Poll SAPI's real completion, on its owning COM thread.
                        # Keep STOP/priority interruption responsive while waiting.
                        while self._running:
                            if self._purge_requested or gen_id < self.speech_generation:
                                self._voice.Speak("", 2)
                                self._purge_requested = False
                                cancelled = True
                                break
                            if self._voice.WaitUntilDone(20):
                                break
                        event_broker.publish(EventType.TTS_FINISHED, {
                            "text": text, "priority": priority, "source": source,
                            "generation": gen_id, "cancelled": cancelled,
                        })
                    except Exception as ex:
                        logger.error("SAPI speak exception: %s", ex)
                        event_broker.publish(EventType.ERROR, {
                            "source": "tts", "message": "Speech output failed.",
                        })
                    finally:
                        self._is_speaking = False
                        self._queue.task_done()
                else:
                    self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Unexpected error in TTS speech loop: {e}")

        if self._voice is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def speak(self, text: str, priority: int = 5, interrupt: bool = False, source: str = "GUIDANCE", generation_id: Optional[int] = None) -> None:
        """
        Enqueues text for speech tagged with generation ID.
        If interrupt=True (e.g. STOP or Emergency Hazard), purges queue and cuts off current audio immediately.
        """
        if not text or not text.strip():
            return

        clean_text = text.strip()

        with self._lock:
            current_gen = self.speech_generation if generation_id is None else generation_id
            
            # If the supplied generation is already obsolete, reject immediately
            if current_gen < self.speech_generation:
                return

            if interrupt:
                # Purge pending items (queue only). The COM purge is performed by
                # the worker thread so this call never blocks the caller.
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except Exception:
                        break

                self._is_speaking = False
                self._speaking_until = 0.0
                self._purge_requested = True

                self._queue.put((0, current_gen, time.time(), clean_text, True, source))
            else:
                self._queue.put((priority, current_gen, time.time(), clean_text, False, source))

    def stop(self) -> None:
        """
        Global Cancellation / Purge:
        Increments generation ID, purges all pending items, and halts playback immediately.
        """
        with self._lock:
            self.speech_generation += 1  # Invalidate all prior speech tasks
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except Exception:
                    break

            # Defer the COM purge to the worker thread — never block the caller.
            self._purge_requested = True
            self._is_speaking = False
            self._speaking_until = 0.0
            logger.info(f"TTS globally stopped. Incremented to speech generation {self.speech_generation}.")

    def purge_mode_speech(self, source_to_purge: str) -> None:
        """Purges speech events belonging to a specific mode when switching."""
        with self._lock:
            # Rebuild queue omitting the purged source
            items = []
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    self._queue.task_done()
                    if item[5] != source_to_purge:
                        items.append(item)
                except Exception:
                    break
            for it in items:
                self._queue.put(it)

    def is_busy(self) -> bool:
        """Include enqueued speech so a producer/worker race cannot reopen ASR."""
        return self._is_speaking or self._queue.unfinished_tasks > 0 or self._purge_requested

    def is_speaking(self) -> bool:
        return self._is_speaking

# Global TTS Provider instance
tts_engine = WindowsSAPITTSProvider()
