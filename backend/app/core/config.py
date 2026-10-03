"""
VisionMate v2 - Global Configuration Settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "VisionMate v2"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # Server & API
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Camera Settings — switch by config, never hard-code in feature code
    CAMERA_SOURCE: str = "esp32"  # esp32 | webcam | mock
    WEBCAM_INDEX: int = 0
    TOF_SOURCE: str = "mock"      # mock | vl53l5cx
    GPS_SOURCE: str = "mock"      # mock | browser | hardware
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

    # Depth / ToF (VL53L5CX) — hardware is optional; laptop dev uses mock
    # TOF_ENDPOINT_URL: JSON depth source exposed by the companion MCU/ESP32, e.g.
    #   http://192.168.4.1/depth  -> {"zones": [[mm, ...8 cols], ...8 rows]}
    TOF_ENDPOINT_URL: str = ""
    TOF_TIMEOUT_SEC: float = 1.0
    TOF_CENTER_NEAR_THRESHOLD_MM: int = 800   # center-zone proximity hazard threshold
    TOF_VALID_MIN_MM: int = 20

    # Currency & Medicine (ON-DEMAND only; never continuous)
    CURRENCY_DEFAULT_REGION: str = "INR"  # INR | USD | EUR (affects denomination lexicon)

    # GPS / Navigation — backend-side protected configuration (never exposed to frontend)
    # NAVIGATION_PROVIDER: mock | google
    NAVIGATION_PROVIDER: str = "mock"
    NAVIGATION_ARRIVAL_RADIUS_M: float = 12.0
    NAVIGATION_OFF_ROUTE_RADIUS_M: float = 35.0
    # "name:lat,lon;name2:lat,lon" registry used by the mock provider (no live routing needed)
    NAVIGATION_DESTINATIONS: str = (
        "home:12.9716,77.5946;office:12.9352,77.6245;pharmacy:12.9600,77.6100"
    )
    # Google Directions key is ONLY read from the environment; never hard-code, never expose.
    GOOGLE_MAPS_API_KEY: str = ""

    # Audio & TTS
    TTS_DEFAULT_RATE: int = 2  # SAPI voice rate (-10 to 10)
    TTS_VOLUME: int = 100

    # ASR / real-time voice (Whisper loaded once, reused)
    ASR_MODEL: str = "tiny.en"
    ASR_DEVICE: str = "cpu"
    ASR_COMPUTE_TYPE: str = "int8"
    ASR_SAMPLE_RATE: int = 16000
    VOICE_LISTENER_ENABLED: bool = True

    # Wake-word (LOCAL / OFFLINE keyword spotting — NOT VAD, NOT transcript matching)
    # WAKE_WORD_MODE: auto | openwakeword | keyboard | disabled
    #   auto         -> use local openWakeWord engine if importable, else DEVELOPMENT FALLBACK
    #   openwakeword -> force local offline openWakeWord engine
    #   keyboard     -> explicit DEVELOPMENT FALLBACK (Enter key / push-to-talk)
    #   disabled     -> no gate; the session is always active (dev/competition convenience)
    WAKE_WORD_ENABLED: bool = True
    WAKE_WORD_MODE: str = "auto"
    WAKE_WORD_PHRASE: str = "hey drishti"
    WAKE_WORD_MODEL_PATH: str = ""  # optional custom openWakeWord model (.onnx)
    WAKE_WORD_PRETRAINED: str = "alexa"  # pretrained listen target when no custom model
    WAKE_WORD_THRESHOLD: float = 0.5
    WAKE_WORD_ACK_TEXT: str = ""  # optional short spoken acknowledgement on wake

    VAD_FRAME_MS: int = 30
    VAD_SPEECH_START_RMS: float = 0.018
    VAD_SPEECH_END_RMS: float = 0.010
    VAD_MIN_SPEECH_MS: int = 250
    VAD_SILENCE_END_MS: int = 550
    VAD_MAX_UTTERANCE_MS: int = 8000
    VAD_PRE_ROLL_MS: int = 240

settings = Settings()

