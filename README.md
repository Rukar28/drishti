# VisionMate / Drishti

Integrated React + FastAPI assistive-vision prototype. The backend owns camera capture, perception, voice, OCR, visual reasoning, navigation and SOS.

## Run locally (Windows)

From the repository root in one terminal:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:3000 and choose **VisionMate User** or **Caregiver**. The Vite proxy carries REST and WebSocket traffic to FastAPI. `frontend/.env.example` documents custom backend origins. No backend credentials belong in frontend environment variables.

## Configuration

The root `.env` controls the backend. It is ignored by Git.

- Camera: `CAMERA_SOURCE=esp32` for the existing device, or `webcam` for a real local camera. `mock` is explicitly a demo provider.
- GPS: `GPS_SOURCE=browser`; restart FastAPI, then use **Share this device’s location** on Navigate. Requires browser permission and localhost/HTTPS. Fixes expire after 30 seconds; sharing stops when leaving Navigate.
- Live Google navigation: `NAVIGATION_PROVIDER=google` and `GOOGLE_MAPS_API_KEY` on the backend. Enable **Geocoding API**, **Routes API**, and **Maps Static API**, with billing and appropriate server key restrictions. Map images are proxied; the key is never sent to React. With `mock`, the UI explicitly says Demo Navigation.
- ASK: the existing Ollama service/model configured through `OLLAMA_BASE_URL` and `OLLAMA_VLM_MODEL`. Gemini is not required or called by the frontend.
- OCR: the backend implementation requires PaddleOCR 3.x/PP-OCRv5 and its platform-compatible PaddlePaddle runtime. Missing OCR now reports unavailable, including currency/medicine workflows; no fabricated text is produced.
- SOS: configure the `SOS_*` settings in `.env.example`. Confirmation triggers the existing notification service. A successful SMTP result is required before “sent” is displayed.

Role selection is **not authentication**. This prototype has no account/session authorization and must remain on a trusted local device/network. Do not expose it publicly without a proper authentication and authorization layer. Caregiver history is a bounded current-browser-session timeline, not a persisted audit log.

## Checks

```powershell
.venv\Scripts\python.exe -m pytest backend/tests -q
cd frontend
npm run build
npm run lint
```

The prior untracked, incomplete frontend draft is preserved locally in the Git-ignored `frontend/legacy-draft/` folder. Active code is in `frontend/src/app/`, with independent screens, shared components, typed API calls, one event connection/store, and a separate frame stream.

See [integration report](docs/FRONTEND_INTEGRATION.md) for endpoint contracts, backend changes, browser checks, and current hardware/configuration limits.
