# VisionMate / Drishti — Backend Agent Progress

> Scope of this document: **BACKEND ONLY**. The React frontend, face recognition,
> SOS/caregiver and color recognition are intentionally **DEFERRED**.
> Status labels: IMPLEMENTED+TESTED · IMPLEMENTED+HARDWARE-UNVERIFIED · PARTIAL ·
> BLOCKED · DEFERRED.

_Last updated by backend completion pass._

## Source of truth

`docs/MASTER_PRD.md` was **not present** in the repository at the time of this
pass. The implementation was instead verified against the existing code, the
existing test suite, and the task specification. This file records the true state.

## Baseline (before this pass)

- Test suite: **42 passed** in ~43 s.
- Already implemented and working: camera providers (esp32/webcam/mock), YOLO
  detection, spatial tracking, World State, priority engine, TTS queue with
  generation invalidation, ASR (Faster-Whisper), adaptive energy VAD, Command Bus,
  FIND/READ/ASK, REST API + WebSockets, mock depth/GPS providers.

## Completed in this pass

| Area | Status | Notes |
|------|--------|-------|
| P0 priority hierarchy | IMPLEMENTED+TESTED | Expanded `PriorityLevel` (STOP/EMERGENCY/OBSTACLE/NAVIGATION/USER_COMMAND/INTERACTION/BACKGROUND), backward compatible. |
| P0 STOP interruption | IMPLEMENTED+TESTED | STOP purges TTS, exits FIND, stops navigation, aborts the voice FSM (generation invalidation) so stale commands cannot execute. |
| P0 duplicate/hazard cooldown/escalation | IMPLEMENTED+TESTED | Verified in `test_priority_and_interruption.py`. |
| P1 perception / World State / guidance | IMPLEMENTED+TESTED | Loop preserved; hazard events now published; depth hazard folded in. |
| P2A adaptive VAD | IMPLEMENTED+TESTED | Pre-existing; confirmed no fixed 3 s window. |
| P2B **wake word** | IMPLEMENTED+TESTED (fallback) / engine optional | New `WakeWordProvider` boundary: REAL local/offline `OpenWakeWordProvider` (no cloud audio) + explicit `KeyboardWakeWordProvider` DEVELOPMENT FALLBACK + `DisabledWakeWordProvider`. openWakeWord not installed in this environment, so runtime uses the clearly-labelled fallback. |
| P2C voice state machine | IMPLEMENTED+TESTED | New deterministic `VoiceStateMachine` with legal transition graph + atomic abort. |
| P2D Command Bus | IMPLEMENTED+TESTED | All voice commands route through the single Command Bus. |
| P3 FIND / READ / ASK | IMPLEMENTED+TESTED | Pre-existing; on-demand only. |
| P4 depth / VL53L5CX | IMPLEMENTED+HARDWARE-UNVERIFIED | Real HTTP-grid provider + zone math (closest-reading per zone) + depth-aware hazard reasoning. No sensor attached. |
| P5 currency | IMPLEMENTED+TESTED | On-demand OCR-based `CurrencyRecognizer`; never continuous; no invented values. |
| P5 medicine | IMPLEMENTED+TESTED | On-demand OCR-based `MedicineRecognizer`; recognition only, explicit "not medical advice". |
| P6 GPS/navigation | IMPLEMENTED+TESTED | Provider abstraction, offline mock routing (haversine/bearing), protected Google boundary (env-only key), Navigator session + voice guidance. |
| API/WebSocket | IMPLEMENTED+TESTED | Added currency/medicine/navigation endpoints; `HAZARD`, `TTS_STARTED/FINISHED`, `NAVIGATION_UPDATE` events now emitted. |
| Laptop dev mode | IMPLEMENTED+TESTED | `CAMERA_SOURCE=webcam`, `TOF_SOURCE=mock`, `GPS_SOURCE=mock`, `DETECTION_DEVICE=cpu` supported. |

## Wake-word status (explicit)

- There was **no** wake-word implementation before this pass (only VAD + ASR and,
  at most, transcript substring parsing, which is NOT a wake word).
- A real `WakeWordProvider` boundary now exists with a genuine **local/offline**
  engine path (`openwakeword`, ONNX, no cloud audio).
- Because `openwakeword` is **not installed** in this environment, the running
  system uses the **DEVELOPMENT FALLBACK** (keyboard/push-to-talk). This is
  reported honestly via `health()` (`is_real_wake_word: false`,
  `mode: DEVELOPMENT_FALLBACK`). It is **not** claimed to be a real wake word.
- To activate the real engine: `pip install openwakeword` and set
  `WAKE_WORD_MODE=openwakeword` (optionally `WAKE_WORD_MODEL_PATH=/path/model.onnx`).

## Blocked / dependency-limited

- `openwakeword` engine: **BLOCKED BY OPTIONAL DEPENDENCY** (not installed here).
- VL53L5CX depth: **IMPLEMENTED + HARDWARE-UNVERIFIED** (no sensor attached).
- Google Directions live routing: **BLOCKED BY CREDENTIALS** (no API key);
  boundary implemented, mock routing testable.
- Ollama VLM (`moondream`/`qwen3-vl`): ON-DEMAND; requires a local Ollama server.

## Deferred (intentionally)

Face recognition/enrollment · SOS/caregiver · color recognition · React frontend.

## Tests

- Existing suite: unchanged behaviour (regression-checked).
- New files: `test_priority_and_interruption.py`, `test_wakeword_and_voice_state.py`,
  `test_currency_medicine.py`, `test_navigation.py`, `test_depth_provider.py`.

## Known issues / next steps

1. Install `openwakeword` on the target device and enroll a "hey drishti" model;
   flip `WAKE_WORD_MODE=openwakeword`. Verify with a real microphone.
2. Wire a VL53L5CX to the companion MCU exposing `/depth` JSON; set
   `TOF_ENDPOINT_URL`; re-run depth tests on hardware.
3. Configure `GOOGLE_MAPS_API_KEY` (backend env only) and implement the live
   Directions call inside `GoogleDirectionsProvider` when on-site.
4. Echo/self-trigger tuning for push-to-talk vs always-on wake word on the final
   hardware.
