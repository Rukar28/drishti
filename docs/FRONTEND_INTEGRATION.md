# Frontend integration progress

Audit: existing frontend is incomplete (missing App, mismatched API methods/types and invented SOS endpoint). Backend REST and event contracts verified. No auth exists; role selection must be explicitly prototype-only. ASK is Ollama, not Gemini. Focus and Face are unavailable.

Contract map:
- Home: /api/world (WORLD_UPDATE), /api/status, /health, /metrics, /api/voice; binary annotated JPEG /ws/stream.
- Events: one /ws/events {type,timestamp,data}; world, voice, TTS, mode, navigation, hazards, SOS, errors. Bounded session history only.
- Assist: POST /api/v1/find {target}, /read {full_reading}, /ask {query}, /currency, /medicine; new thin /color {target} adapter.
- STOP: POST /api/v1/stop. GUIDANCE: POST /api/v1/mode {mode:'guidance'}.
- Navigation: /api/navigation GET, POST {destination}, /api/navigation/stop. Extend Google provider with Geocoding + Routes walking API, safe route/steps/polyline state. Proxy Google Static Map.
- Location: /api/location GET; new validated POST only when GPS_SOURCE=browser, fixes expire in 30 seconds.
- SOS: new GET /api/sos safe configuration/result, POST adapter to existing command bus; reuse live GPS for notification.

Implementation underway. Final validation/results will be recorded here.
## Implemented backend changes

- Google provider now uses backend-only Geocoding + current Routes `computeRoutes` with WALK, specific response field mask, real steps/distance/duration/polyline/warnings, bounded network timeouts and sanitized failures.
- Navigator exposes route, destination coordinates, current step/instruction, last update; advances steps, detects arrival/off-route, reroutes, and prevents cancelled requests from resurrecting navigation. Route distance/ETA remain the provider's calculation-time estimates, clearly labeled in the UI.
- Browser GPS validates coordinates/accuracy and expires fixes after 30 seconds. Location updates tick navigation even if the camera is unavailable.
- `/api/navigation/map` proxies Google Static Maps images with position/destination/polyline, keeping credentials private.
- `/api/v1/color` and `/api/sos` are thin adapters to existing pipeline/command-bus flows. SOS status exposes safe configuration/cooldown/latest result and uses active GPS for notification location.
- Public world snapshots include actual navigation health.
- Removed pre-existing fabricated OCR text and canned offline VLM descriptions. Assistance never processes the standby/HUD frame. Missing providers/frame report unavailable.
- Google HTTP client informational URL logging is disabled to prevent credential-bearing query strings entering logs.
- Two existing successful OCR/VLM unit tests now inject deterministic provider responses instead of relying on fabricated runtime fallbacks; their original assertions remain. Added explicit tests for missing engine/service, Google routes/errors, key privacy, GPS validation/expiry, map proxy, STOP races and route geometry.

## Frontend structure and routes

React 18 + TypeScript + Vite + React Router + Zustand + Tailwind + Lucide. `src/app/pages` holds role entry, layout, Home, Assist, Navigation, Safety, Overview, History and Settings. Shared components handle frames, maps, SOS, device status, async responses and status labels. `api.ts` centralizes origins, request bodies, JSON/error handling and bounded timeouts; `realtime.ts` manages one event connection with cleanup/backoff and periodic state recovery; `store.ts` contains shared state. JPEG frames remain isolated from the app store, with object URLs revoked on replacement/unmount. UI uses top navigation, keyboard controls, focus styles, mobile layouts, reduced-motion support and persistent STOP.

Routes: `/`, `/admin/login`, `/home`, `/assist`, `/navigate`, `/safety`, `/settings`, `/admin`, `/admin/location`, `/admin/safety`, `/admin/history`, `/admin/device`.

REST consumed: GET `/health`, `/api/status`, `/api/world`, `/metrics`, `/api/location`, `/api/navigation`, `/api/navigation/map`, `/api/sos`; POST `/api/v1/mode`, `/api/v1/find`, `/api/v1/read`, `/api/v1/ask`, `/api/v1/color`, `/api/v1/currency`, `/api/v1/medicine`, `/api/v1/stop`, `/api/navigation`, `/api/navigation/stop`, `/api/location`, `/api/sos`.

WebSockets: `/ws/events` JSON envelopes and `/ws/stream` binary JPEG. Frontend subscribes to actual event names; high-frequency world/voice-state updates are excluded from meaningful session history. Last spoken assistant text comes from actual TTS events or backend status.

## Validation on 2026-10-05

- Original baseline: 132 tests passed. Extended backend suite: 144 passed, one pre-existing Starlette/httpx deprecation warning.
- Frontend production build and TypeScript lint passed; final checks repeated after refinements.
- Real FastAPI and Vite ran together. Browser clicks tested role entry, Home, FIND, full READ, ASK, target COLOR, CURRENCY, MEDICINE, STOP, navigation start/stop with unavailable GPS, SOS confirmation/not-configured result, caregiver overview/location/safety/history/device, settings and larger text.
- Stopped and restarted the real backend: UI showed backend offline/stale data and event reconnecting, then recovered automatically.
- Both WebSocket transports connected. With ESP32 unreachable, binary stream carries backend standby frames; the UI hides these and shows Camera unavailable. Physical live camera/detection accuracy could not be verified.
- Mobile 390px layout and enlarged text inspected; no horizontal overflow. Desktop inspection and final console check recorded below.

## Current external limits / honest status

- This checkout's `.env` contains no Google key. Effective Google key presence is false; GPS and navigation are mock. Live Google billing/key/route/map success cannot be claimed. Controlled provider tests exercise real contract parsing and error paths without fabricating user data.
- ESP32 capture endpoint is unreachable. YOLO loaded on CPU, Faster-Whisper and the real Hey Mycroft wake-word model loaded. Physical camera, depth and live navigation remain hardware/configuration acceptance tests.
- PaddleOCR is missing in the installed runtime; OCR/currency/medicine correctly return unavailable/no frame. Ollama is unavailable in provider tests; ASK reports unavailable rather than invented answers.
- SOS is disabled/not configured; no real emergency email was sent during QA. Existing SMTP tests cover success, failure and cooldown with controlled transport.
- Focus and face recognition remain explicitly unavailable. No fake authentication or persisted history was added.
- npm audit reports 7 advisories in the inherited React Router 6 / Tailwind 3 dependency trees (2 moderate, 5 high), with no nonbreaking fix reported for the root packages. The app uses client routing, not SSR hydration; dev server is bound to loopback. Dependency-major migration remains a separate compatibility task.

Google API references: https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes and https://developers.google.com/maps/documentation/maps-static/start .
