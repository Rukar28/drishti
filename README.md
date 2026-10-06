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

For an explicit webcam/browser GPS development profile, the launcher can run both servers:

```powershell
.\scripts\start-dev.ps1 -Camera webcam -GPS browser
```

This preserves the root `.env`. Stop the previous frontend first so port 3000 is free. The launcher accepts `-FrontendPort` too and restores shell environment variables on exit. The default backend is **8000** and frontend is **3000**. A Git-ignored `frontend/.env.local` may override `BACKEND_PROXY_URL`; keep it aligned with the backend port.

## Configuration

The root `.env` controls the backend. It is ignored by Git.

- Camera: `CAMERA_SOURCE=esp32` for the existing device, or `webcam` for a real local camera. `mock` is explicitly a demo provider.
- GPS: `GPS_SOURCE=browser`; restart FastAPI, then use **Share this device’s location** on Navigate. Requires browser permission and localhost/HTTPS. Fixes expire after 30 seconds; sharing continues across pages until explicitly stopped, a role switch, or tab closure.
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
npm test
```

The prior untracked, incomplete frontend draft is preserved locally in the Git-ignored `frontend/legacy-draft/` folder. Active code is in `frontend/src/app/`, with independent screens, shared components, typed API calls, one event connection/store, and a separate frame stream.

See [integration report](docs/FRONTEND_INTEGRATION.md) for endpoint contracts, backend changes, browser checks, and current hardware/configuration limits.

## Voice-first demo

Use the project interpreter (`.venv\Scripts\python.exe`) for both installation and startup:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\start-dev.ps1 -Camera webcam -GPS browser
```

Set `WAKE_WORD_MODE=real`, `WAKE_WORD_PHRASE=hey mycroft`, and `WAKE_WORD_PRETRAINED=hey_mycroft` in the backend environment. ESP32 support is unchanged. Real wake detection uses the cached openWakeWord ONNX model, not transcript matching. The backend owns microphone capture; React does not record audio.

Open **Assist**. When it says **Listening for “Hey Mycroft”**, say the wake phrase, pause for **Listening for your command**, then say **What color is this?**. The HUD shows backend transcript, recognized feature, actual result, and speech state. After speech completes, repeat with **Find the bottle**, **Read this**, **What is this?**, **Identify this currency**, **Identify this medicine**, or **Take me to the pharmacy**. **Send SOS** invokes the actual configured emergency service; it is not a demo success action.

SAPI completion is polled on its COM worker; the microphone does not reopen during queued or active speech. The command-listening window times out after 20 seconds. Wake and ASR are sequential: speak the command after the wake phrase, rather than as one uninterrupted sentence. Spoken STOP is available at the next command capture; the persistent STOP control immediately cancels active tasks/TTS. Continuous acoustic barge-in during model execution or speech is not implemented.

PaddleOCR 3.7.0 + PaddlePaddle 3.3.1 run the existing PP-OCRv5 engine, explicitly selected to avoid upstream default-model changes. CPU oneDNN acceleration is disabled after an observed inference compatibility failure. Models download on first use, then cache locally. A generated printed-text fixture was recognized by the real engine; this is not a claim of live medicine identification.
