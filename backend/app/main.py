"""
VisionMate v2 - Main FastAPI Application Server & Minimal Competition HUD
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from backend.app.core.config import settings
from backend.app.api.routes import router as api_router
from backend.app.api.websocket import ws_router
from backend.app.services.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("visionmate.main")
# Upstream Google query strings include server credentials.
logging.getLogger("httpx").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start master pipeline asynchronously
    logger.info("Initializing VisionMate v2 backend pipeline...")
    pipeline.start()
    yield
    # Shutdown: Stop pipeline
    logger.info("Stopping VisionMate v2 backend pipeline...")
    pipeline.stop()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)

# Minimal Competition HUD Dashboard
HUD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VisionMate v2 - ESP32-CAM Live Dashboard</title>
  <style>
    body { margin: 0; background: #0b0e14; color: #e1e7ec; font-family: 'Segoe UI', Tahoma, sans-serif; display: flex; flex-direction: column; height: 100vh; }
    header { background: #161b22; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; }
    h1 { margin: 0; font-size: 1.25rem; color: #58a6ff; display: flex; align-items: center; gap: 8px; }
    .badge { padding: 4px 12px; border-radius: 4px; font-weight: bold; background: #238636; font-size: 0.85rem; letter-spacing: 0.5px; }
    .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; flex: 1; overflow: hidden; }
    .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; display: flex; flex-direction: column; }
    .panel h2 { margin: 0 0 12px 0; font-size: 1.0rem; color: #8b949e; border-bottom: 1px solid #21262d; padding-bottom: 6px; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.8px; }
    #videoContainer { position: relative; width: 100%; max-height: 520px; background: #000; border-radius: 6px; overflow: hidden; display: flex; justify-content: center; align-items: center; }
    #videoFeed { width: 100%; height: auto; max-height: 520px; object-fit: contain; }
    .source-tag { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.75); color: #58a6ff; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; border: 1px solid #30363d; }
    .btn-group { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; }
    button:hover { background: #30363d; color: #fff; }
    button.active { background: #1f6feb; border-color: #58a6ff; color: #fff; }
    .telemetry-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 0.88rem; }
    .speech-box { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; min-height: 60px; margin-top: 8px; color: #58a6ff; font-weight: 500; font-size: 0.9rem; }
    .status-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #238636; margin-right: 6px; }
    .status-dot.offline { background: #da3633; }
  </style>
</head>
<body>
  <header>
    <h1>VisionMate v2 <span>| Live ESP32-CAM AI Dashboard</span></h1>
    <div id="modeBadge" class="badge">GUIDANCE MODE</div>
  </header>
  <div class="main-grid">
    <div class="panel">
      <h2>Live Perception Stream (YOLO / Track Overlay)</h2>
      <div id="videoContainer">
        <div class="source-tag">AI Source: ESP32-CAM /capture</div>
        <img id="videoFeed" alt="Live Perception Stream" src="" />
      </div>
      <div class="btn-group">
        <button onclick="setMode('guidance')" class="active" id="btn-guidance">Guidance</button>
        <button onclick="promptFind()" id="btn-find">Find Target</button>
        <button onclick="triggerRead()" id="btn-read">Read Text (OCR)</button>
        <button onclick="promptAsk()" id="btn-ask">Ask AI (VLM)</button>
        <button onclick="stopSpeech()" style="background:#da3633; border-color:#f85149;">Stop Speech</button>
      </div>
    </div>
    <div class="panel">
      <h2>ESP32-CAM & AI Telemetry</h2>
      <div class="telemetry-row"><span>Camera Hardware:</span><strong id="camNameVal">ESP32-CAM (OV2640)</strong></div>
      <div class="telemetry-row"><span>Camera Status:</span><strong id="camStatusVal"><span id="camDot" class="status-dot"></span>ONLINE</strong></div>
      <div class="telemetry-row"><span>AI Source Endpoint:</span><strong id="camUrlVal">http://192.168.4.1/capture</strong></div>
      <div class="telemetry-row"><span>Snapshot Rate:</span><strong id="fpsVal">0.0 FPS</strong></div>
      <div class="telemetry-row"><span>Capture Latency:</span><strong id="capLatVal">0.0 ms</strong></div>
      <div class="telemetry-row"><span>YOLO Latency:</span><strong id="yoloLatVal">0.0 ms</strong></div>
      <div class="telemetry-row"><span>Total Perception Latency:</span><strong id="totLatVal">0.0 ms</strong></div>
      <div class="telemetry-row"><span>Detected Entities:</span><strong id="objCountVal">0</strong></div>
      <div class="telemetry-row"><span>Dropped / Stale Frames:</span><strong id="dropVal">0</strong></div>
      <h2 style="margin-top: 14px;">Latest Spoken Guidance</h2>
      <div id="speechBox" class="speech-box">Listening to scene...</div>
    </div>
  </div>

  <script>
    let currentObjectUrl = null;
    let ws = null;

    function initWebSocket() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      ws.onmessage = (event) => {
        if (typeof event.data !== 'string') {
          const blob = new Blob([event.data], { type: 'image/jpeg' });
          const newUrl = URL.createObjectURL(blob);
          const img = document.getElementById('videoFeed');
          img.src = newUrl;
          if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
          }
          currentObjectUrl = newUrl;
        }
      };

      ws.onclose = () => { setTimeout(initWebSocket, 1500); };
      ws.onerror = () => { ws.close(); };
    }

    initWebSocket();

    async function pollStatus() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        
        const isOnline = data.camera_connected || data.connected;
        const camDot = document.getElementById('camDot');
        if (isOnline) {
          camDot.className = 'status-dot';
          document.getElementById('camStatusVal').innerHTML = '<span class="status-dot"></span>ONLINE';
        } else {
          camDot.className = 'status-dot offline';
          document.getElementById('camStatusVal').innerHTML = '<span class="status-dot offline"></span>RECONNECTING';
        }

        document.getElementById('fpsVal').innerText = Number(data.snapshot_fps || data.camera_received_fps || data.fps || 0).toFixed(1) + ' FPS';
        document.getElementById('capLatVal').innerText = Number(data.capture_latency_ms || 0).toFixed(1) + ' ms';
        document.getElementById('totLatVal').innerText = Number(data.last_latency_ms || 0).toFixed(1) + ' ms';
        document.getElementById('objCountVal').innerText = data.active_object_count || 0;
        document.getElementById('dropVal').innerText = data.dropped_frames || 0;
        document.getElementById('modeBadge').innerText = (data.mode || 'GUIDANCE').toUpperCase() + ' MODE';
        
        if (data.camera_url) {
          document.getElementById('camUrlVal').innerText = data.camera_url;
        }
        if (data.last_spoken_narrative) {
          document.getElementById('speechBox').innerText = data.last_spoken_narrative;
        }

        // Fetch /metrics for YOLO latency
        const mRes = await fetch('/metrics');
        const mData = await mRes.json();
        if (mData.yolo_latency_ms !== undefined) {
          document.getElementById('yoloLatVal').innerText = Number(mData.yolo_latency_ms).toFixed(1) + ' ms';
        }
      } catch(e) {}
    }
    setInterval(pollStatus, 400);

    async function setMode(mode) {
      await fetch('/api/v1/mode', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ mode }) });
    }
    async function promptFind() {
      const target = prompt("What object would you like to find? (e.g. bottle, laptop, chair, person):", "bottle");
      if (target) {
        await fetch('/api/v1/find', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ target }) });
      }
    }
    async function triggerRead() {
      await fetch('/api/v1/read', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ full_reading: false }) });
    }
    async function promptAsk() {
      const query = prompt("What question would you like to ask the AI?", "What is in front of me?");
      if (query) {
        await fetch('/api/v1/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ query }) });
      }
    }
    async function stopSpeech() {
      await fetch('/api/v1/tts/stop', { method: 'POST' });
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def index():
    return HUD_HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
