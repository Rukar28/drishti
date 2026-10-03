"""
VisionMate v2 - Global Configuration Settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    APP_NAME: str = "VisionMate v2"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # =========================================================================
    # SERVER & API
    # =========================================================================

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # =========================================================================
    # CAMERA SETTINGS
    # =========================================================================

    # esp32 | webcam | mock
    CAMERA_SOURCE: str = "esp32"

    WEBCAM_INDEX: int = 0

    # mock | vl53l5cx
    TOF_SOURCE: str = "mock"

    # mock | browser | hardware
    GPS_SOURCE: str = "mock"

    CAMERA_FRAME_WIDTH: int = 640
    CAMERA_FRAME_HEIGHT: int = 480

    # Discard stale frames; always consume latest frame.
    FRAME_BUFFER_DEPTH: int = 1

    # =========================================================================
    # ESP32-CAM SOFTAP SETTINGS
    # =========================================================================

    # Fixed SoftAP IP — never changes between sessions.
    ESP32_AP_IP: str = "192.168.4.1"

    ESP32_STREAM_URL: str = (
        "http://192.168.4.1:81/stream"
    )

    ESP32_CAPTURE_URL: str = (
        "http://192.168.4.1/capture"
    )

    ESP32_INFO_URL: str = (
        "http://192.168.4.1/info"
    )

    ESP32_MDNS_HOST: str = (
        "visionmatecam.local"
    )

    ESP32_CONNECT_TIMEOUT_SEC: float = 3.0

    ESP32_RECONNECT_MAX_DELAY_SEC: float = 6.0

    ESP32_READ_CHUNK_BYTES: int = 8192

    # 100 ms ≈ 10 FPS
    ESP32_SNAPSHOT_INTERVAL_SEC: float = 0.10

    ESP32_SNAPSHOT_TARGET_FPS: float = 10.0

    # Horizontally mirror ESP32 frames before AI perception.
    ESP32_MIRROR_HORIZONTAL: bool = True

    # =========================================================================
    # PERCEPTION & TRACKING
    # =========================================================================

    # Target: 18 FPS / ~55.6 ms per frame.
    VISIONMATE_TARGET_FPS: int = 18

    TARGET_PERCEPTION_FPS: int = 18

    YOLO_MODEL_PATH: str = "yolov8n.pt"

    DETECTION_CONF_THRESHOLD: float = 0.40

    # cuda | cpu
    # Runtime code should fall back to CPU if CUDA is unavailable.
    DETECTION_DEVICE: str = "cuda"

    TRACKER_MAX_DISAPPEARED: int = 15

    TRACKER_IOU_THRESHOLD: float = 0.28

    # =========================================================================
    # PRIORITY & COOLDOWNS
    # =========================================================================

    GUIDANCE_SPEECH_COOLDOWN_SEC: float = 6.0

    AWARENESS_SPEECH_COOLDOWN_SEC: float = 6.0

    HAZARD_SPEECH_COOLDOWN_SEC: float = 2.5

    FIND_MODE_UPDATE_INTERVAL_SEC: float = 1.5

    # =========================================================================
    # REASONING & OCR
    # =========================================================================

    OLLAMA_BASE_URL: str = (
        "http://127.0.0.1:11434"
    )

    OLLAMA_VLM_MODEL: str = "qwen3-vl:4b"

    VLM_TIMEOUT_SEC: float = 8.0

    # =========================================================================
    # DEPTH / TOF — VL53L5CX
    # =========================================================================

    # Optional companion MCU / ESP32 depth endpoint.
    #
    # Example:
    #
    # http://192.168.4.1/depth
    #
    # Expected:
    #
    # {
    #     "zones": [
    #         [mm, mm, ...],
    #         ...
    #     ]
    # }
    #
    TOF_ENDPOINT_URL: str = ""

    TOF_TIMEOUT_SEC: float = 1.0

    # Center-zone proximity hazard threshold.
    TOF_CENTER_NEAR_THRESHOLD_MM: int = 800

    TOF_VALID_MIN_MM: int = 20

    # =========================================================================
    # CURRENCY & MEDICINE
    # =========================================================================

    # INR | USD | EUR
    #
    # This affects denomination lexicon.
    #
    # These features are ON-DEMAND only.
    CURRENCY_DEFAULT_REGION: str = "INR"

    # =========================================================================
    # GPS / NAVIGATION
    # =========================================================================

    # mock | google
    NAVIGATION_PROVIDER: str = "mock"

    NAVIGATION_ARRIVAL_RADIUS_M: float = 12.0

    NAVIGATION_OFF_ROUTE_RADIUS_M: float = 35.0

    # Mock destination registry.
    #
    # Format:
    #
    # name:lat,lon;name2:lat,lon
    #
    NAVIGATION_DESTINATIONS: str = (
        "home:12.9716,77.5946;"
        "office:12.9352,77.6245;"
        "pharmacy:12.9600,77.6100"
    )

    # Google Directions key:
    # - read only from environment
    # - never hard-code
    # - never expose to frontend
    GOOGLE_MAPS_API_KEY: str = ""

    # =========================================================================
    # AUDIO & TTS
    # =========================================================================

    # SAPI voice rate (-10 to 10)
    TTS_DEFAULT_RATE: int = 2

    TTS_VOLUME: int = 100

    # =========================================================================
    # ASR / REAL-TIME VOICE
    # =========================================================================

    # Faster-Whisper model.
    ASR_MODEL: str = "tiny.en"

    # cpu | cuda
    ASR_DEVICE: str = "cpu"

    # int8 | float16 | etc.
    ASR_COMPUTE_TYPE: str = "int8"

    # IMPORTANT:
    # openWakeWord also expects 16 kHz audio.
    ASR_SAMPLE_RATE: int = 16000

    VOICE_LISTENER_ENABLED: bool = True

    # =========================================================================
    # WAKE WORD — REAL LOCAL / OFFLINE ACOUSTIC DETECTION
    # =========================================================================
    #
    # IMPORTANT:
    #
    # Wake word is NOT:
    #
    #     VAD
    #
    # and NOT:
    #
    #     Faster-Whisper transcript matching.
    #
    # The intended architecture is:
    #
    #     Microphone
    #          ↓
    #     openWakeWord
    #          ↓
    #     hey_mycroft
    #          ↓
    #     Wake detected
    #          ↓
    #     VAD
    #          ↓
    #     Faster-Whisper
    #          ↓
    #     Command Bus
    #
    # -------------------------------------------------------------------------
    # WAKE_WORD_MODE
    #
    # real:
    #     REAL acoustic Hey Mycroft detector.
    #
    # auto:
    #     Use the real detector when available.
    #     If unavailable, explicitly use DEVELOPMENT_FALLBACK.
    #
    # fallback:
    #     Keyboard / push-to-talk development mode.
    #
    # disabled:
    #     Wake gate disabled.
    #
    # -------------------------------------------------------------------------

    WAKE_WORD_ENABLED: bool = True

    # REAL wake-word mode is the default.
    #
    # Do NOT silently fall back to keyboard mode.
    WAKE_WORD_MODE: str = "real"

    # Human-readable phrase.
    WAKE_WORD_PHRASE: str = "hey mycroft"

    # Actual pretrained openWakeWord acoustic model.
    #
    # IMPORTANT:
    #
    # This MUST match the phrase above.
    #
    # Previous configuration was:
    #
    #     phrase = "hey drishti"
    #     model  = "alexa"
    #
    # which was incorrect.
    WAKE_WORD_PRETRAINED: str = "hey_mycroft"

    # Optional explicit path to a downloaded .onnx model.
    #
    # Leave empty to let the wake-word provider resolve the
    # pretrained Hey Mycroft model.
    WAKE_WORD_MODEL_PATH: str = ""

    # Starting threshold.
    #
    # This is configurable and should be tuned later against:
    # - microphone
    # - room noise
    # - speaker volume
    # - false-positive rate
    #
    # 0.5 is only a starting value.
    WAKE_WORD_THRESHOLD: float = 0.5

    # Optional acknowledgement after detecting the wake word.
    #
    # Keep empty initially so TTS does not accidentally feed
    # its own audio back into the microphone.
    WAKE_WORD_ACK_TEXT: str = ""

    # =========================================================================
    # VAD
    # =========================================================================

    VAD_FRAME_MS: int = 30

    VAD_SPEECH_START_RMS: float = 0.018

    VAD_SPEECH_END_RMS: float = 0.010

    VAD_MIN_SPEECH_MS: int = 250

    VAD_SILENCE_END_MS: int = 550

    VAD_MAX_UTTERANCE_MS: int = 8000

    VAD_PRE_ROLL_MS: int = 240

    # =========================================================================
    # SOS / EMERGENCY NOTIFICATIONS
    # =========================================================================

    SOS_ENABLED: bool = False

    SOS_COOLDOWN_SEC: float = 60.0

    SOS_RECIPIENT: str = ""

    SOS_SMTP_HOST: str = ""

    SOS_SMTP_PORT: int = 587

    SOS_SMTP_USERNAME: str = ""

    SOS_SMTP_PASSWORD: str = ""

    SOS_FROM_EMAIL: str = ""


# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================

settings = Settings()