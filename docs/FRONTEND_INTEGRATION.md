# VisionMate integration verification — 2026-10-06

## Preserved implementation

Continued the existing FastAPI pipeline, React pages, camera providers, YOLO/tracking/world state, command bus, ASR, real Hey Mycroft provider, SAPI TTS, navigation, GPS, SOS, and AI handlers. No replacement dashboard or second voice/navigation architecture was introduced. The preceding frontend/navigation integration is preserved in commit `bf7f9ef` on branch `frontend`.

## Voice continuation

- The existing voice loop now waits for queued and active speech, then returns automatically to wake listening. It no longer forcibly resets every second while speech is running.
- SAPI's COM worker polls `WaitUntilDone(20)` instead of estimating completion from text length. STOP purges playback and invalidates late speech; queued work counts as busy before playback begins. Missing SAPI emits an error, not simulated speech success.
- Wake detection hands microphone ownership to command capture. Capture is bounded to 20 seconds, reports PROCESSING before actual transcription, and rejects commands captured before STOP. A wake-phrase-only transcript is discarded rather than executed.
- Background microphone capture is suspended and acoustic buffers reset during speech. Routine scene narration is suppressed during the active command; interrupting safety alerts remain available.
- Existing intent parser and handlers remain the execution path for COLOR, FIND, READ, ASK, CURRENCY, MEDICINE, NAVIGATION, SOS, STOP and explicitly unavailable FACE. `VOICE_EXECUTING` exposes the selected intent before processing; existing `VOICE_COMMAND` exposes the actual result.
- Assist now prominently shows actual backend state, wake time, transcript, recognized feature, result, speech state, errors, and a bounded recent-event list. Manual controls remain below it. React never records audio or runs an independent voice state machine.
- The frontend proxy uses backend 8000 for this session. Both REST and WebSocket traffic flow through Vite on 3000. Reconnection/poll recovery and existing camera stream cleanup are preserved.

## Dependency repairs

`requirements.txt` now includes openWakeWord 0.6.0, PaddleOCR 3.7.0 and PaddlePaddle 3.3.1. These are installed in the project `.venv`; `pip check` passes. Use that interpreter rather than an unrelated global Python environment.

Real Hey Mycroft ONNX, Faster-Whisper tiny.en, webcam and YOLO CPU initialized in the running backend. PaddleOCR's first real inference failed in the oneDNN path. The existing provider now explicitly selects PP-OCRv5, disables that incompatible CPU acceleration, and disables unnecessary document rotation/unwarping. A generated image containing **VISIONMATE READ TEST** was processed by the actual Paddle model and returned that exact text at 0.999 recognition confidence. This is a real model test on a fixture, not a live camera READ claim. Inference exceptions now return UNAVAILABLE rather than misleading NO_TEXT.

## Existing navigation and API contracts

- Home: GET `/health`, `/api/status`, `/api/world`, `/metrics`; `/ws/stream` annotated binary JPEG and `/ws/events` typed JSON envelopes.
- Assist: POST `/api/v1/find`, `/read`, `/ask`, `/color`, `/currency`, `/medicine`, `/stop`. Text command API and microphone sessions dispatch through the existing command bus.
- Navigation: GET/POST `/api/navigation`, POST `/api/navigation/stop`, GET/POST `/api/location`, GET `/api/navigation/map`.
- Google Geocoding resolves destinations; Routes `computeRoutes` WALK supplies steps, maneuvers, distance, duration and geometry; Static Maps images are proxied with private credentials kept on the backend.
- The existing Navigator projects fixes onto route geometry for step progression and remaining estimates, detects arrival, pauses on GPS loss, resumes on a fresh fix, and limits off-route reroutes. A dedicated one-second worker updates navigation independently of the camera. Guidance is spoken only on changes and is cancelled on STOP.
- Browser GPS carries measurement timestamps and accuracy, rejects stale fixes, and expires after 30 seconds. Sharing persists across pages until explicitly stopped, role switch or tab closure. Real/mock/unavailable sources remain distinguished.
- SOS status and submission use the actual configured service and active GPS. No emergency delivery is simulated.

## Verification

The full backend regression suite passed **167 tests** after the voice changes (two warnings: optional Paddle ccache and existing Starlette/httpx deprecation). This includes wake-echo rejection and stale-capture cancellation. The camera-loop test now waits for its actual first processed frame with a ten-second hard deadline instead of assuming CPU warm-up always finishes within 1.5 seconds; its existing assertions remain.

Frontend production build, TypeScript validation and five event-envelope tests passed. Controlled lifecycle tests exercise two sequential acoustic-boundary sessions through the actual parser, command bus, COLOR handler and pixel processing, verifying exactly one execution per session, actual computed color, speech wait, state ordering, and return to wake. Other supplied demo phrases are checked against the existing parser and registered dispatch. Test fixtures are never presented as live acoustic success.

Live browser checks across this continuation verified real webcam JPEG transport and detections, COLOR, actual unavailable service responses, STOP, role/page routes, GPS permission timeout, navigation start/stop without a fix, SOS not-configured response, backend offline/reconnection recovery, location-sharing lifecycle, mobile width and console state. Current live backend is `127.0.0.1:8000`, frontend `127.0.0.1:3000`. The Assist voice HUD renders real speech-start/finish and listening events.

An additional generated-audio fixture passed through the actual Hey Mycroft ONNX model (peak 0.9815, threshold 0.5); Faster-Whisper transcribed its separate command recording as **What color is this?**, parsed as COLOR. These generated inputs were local test fixtures, not live microphone recordings. SAPI completion behavior follows [Microsoft’s WaitUntilDone contract](https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ms723616(v=vs.85)).

**Live human wake → ASR → COLOR → spoken result → second wake acceptance is pending user execution/observation.** Model loading and controlled tests are not substituted for that acceptance test.

## Remaining external or physical acceptance

- Live Google route/map success needs a backend key with billing and Geocoding, Routes and Maps Static APIs enabled. No key was present; no real Google success is claimed.
- Browser geolocation timed out in the earlier live check. No user location or route was fabricated.
- ESP32 hardware acceptance remains outstanding; its provider is preserved. The real webcam is the active development source.
- Ollama/model availability controls ASK; previous live calls reported unavailable. No invented scene response was added.
- SOS is disabled/not configured; no real emergency email was sent. Existing controlled SMTP success/failure/cooldown tests pass.
- Acoustic STOP is available when the sequential microphone session is listening, not as simultaneous speech barge-in while a model/TTS runs. The persistent STOP button remains immediately available.
- Role selection is a local prototype, not authentication; history is bounded in-browser session data.
- Inherited npm audit findings in React Router/Tailwind dependencies were recorded previously (7 advisories); a major dependency migration was not performed in this voice-integration change.

## Key files

`backend/app/services/pipeline.py`, `speech/tts.py`, `speech/asr.py`, `speech/voice_state.py`, `speech/wakeword.py`, `intelligence/command_bus.py`, `core/events.py`, `reasoning/ocr.py`, `backend/tests/test_voice_lifecycle_integration.py`, `frontend/src/app/VoiceStatus.tsx`, `pages/Assist.tsx`, `realtime.ts`, `store.ts`, `requirements.txt`, `.env.example`, and `README.md`.

No credentials or captured audio are committed. One-time integration scripts from the earlier phase were removed in the preceding integration commit. An unrelated root `package-lock.json` observed during work is left untouched and excluded from this change.
