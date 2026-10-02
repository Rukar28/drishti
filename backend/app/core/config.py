"""
VisionMate v2 - Global Configuration Settings
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "VisionMate v2"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # Server & API
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Camera Settings (ESP32-CAM is the only primary production camera provider)
    CAMERA_SOURCE: str = "esp32"  # 'esp32' (default & only production provider)
    WEBCAM_INDEX: int = 0
    CAMERA_FRAME_WIDTH: int = 640
    CAMERA_FRAME_HEIGHT: int = 480
    FRAME_BUFFER_DEPTH: int = 1  # Discard stale frames, consume latest only

    # ESP32-CAM SoftAP Settings
    # Fixed SoftAP IP — never changes between sessions (no DHCP dependency)
    ESP32_AP_IP: str = "192.168.4.1"
    ESP32_STREAM_URL: str = "http://192.168.4.1:81/stream"
    ESP32_CAPTURE_URL: str = "http://192.168.4.1/capture"
    ESP32_INFO_URL: str = "http://192.168.4.1/info"
    ESP32_MDNS_HOST: str = "visionmatecam.local"
    ESP32_CONNECT_TIMEOUT_SEC: float = 3.0
    ESP32_RECONNECT_MAX_DELAY_SEC: float = 6.0
    ESP32_READ_CHUNK_BYTES: int = 8192
    ESP32_SNAPSHOT_INTERVAL_SEC: float = 0.10  # 100 ms interval (~10 FPS)
    ESP32_SNAPSHOT_TARGET_FPS: float = 10.0
    ESP32_MIRROR_HORIZONTAL: bool = True  # Horizontally mirror ESP32 frames before AI perception

    # Perception & Tracking (Target: 18 FPS / ~55.6 ms per frame)
    VISIONMATE_TARGET_FPS: int = 18
    TARGET_PERCEPTION_FPS: int = 18
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    DETECTION_CONF_THRESHOLD: float = 0.40
    DETECTION_DEVICE: str = "cuda"  # fallback to 'cpu' if unavailable
    TRACKER_MAX_DISAPPEARED: int = 15
    TRACKER_IOU_THRESHOLD: float = 0.28

    # Priority & Cooldowns
    GUIDANCE_SPEECH_COOLDOWN_SEC: float = 6.0
    AWARENESS_SPEECH_COOLDOWN_SEC: float = 6.0
    HAZARD_SPEECH_COOLDOWN_SEC: float = 2.5
    FIND_MODE_UPDATE_INTERVAL_SEC: float = 1.5

    # Reasoning & OCR
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_VLM_MODEL: str = "qwen3-vl:4b"
    VLM_TIMEOUT_SEC: float = 8.0

    # Audio & TTS
    TTS_DEFAULT_RATE: int = 2  # SAPI voice rate (-10 to 10)
    TTS_VOLUME: int = 100

settings = Settings()

