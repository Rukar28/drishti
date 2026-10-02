# VisionMate v2 — ESP32-CAM Firmware
# Actual Wi-Fi credentials live only in secrets.h (gitignored).
# See secrets.example.h for the template.

## Required boards & libraries (Arduino IDE / PlatformIO)

### Board manager
- `esp32` by Espressif Systems ≥ 2.0.14
- Board: **AI Thinker ESP32-CAM**
- Upload speed: 115200
- Flash frequency: 80MHz
- Flash mode: QIO
- Partition scheme: *Huge APP (3MB No OTA)*

### Libraries (bundled in ESP32 Arduino core, no manual install needed)
- `esp_camera.h`
- `WiFi.h`
- `WebServer.h`
- `ESPmDNS.h`

## Setup steps

```
1. Install ESP32 board support in Arduino IDE.
2. Copy secrets.example.h → secrets.h
3. Edit secrets.h: set AP_PASSWORD, optionally adjust CAMERA_FRAMESIZE and CAMERA_JPEG_QUALITY.
4. Select board: Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM
5. Connect FTDI programmer (TX→RX, RX→TX, GND→GND, 5V→5V, IO0→GND for flash mode).
6. Press reset on ESP32-CAM while IO0 is grounded.
7. Upload sketch.
8. Disconnect IO0 from GND, press reset.
9. Check Serial Monitor (115200 baud) — should print:
       SoftAP started: SSID=VisionMate  IP=192.168.4.1
       mDNS: visionmatecam.local
       MJPEG stream server ready on :81/stream
10. Connect Windows laptop Wi-Fi to "VisionMate".
11. Test stream: open http://192.168.4.1:81/stream in browser.
12. Run: python scripts/check_esp32.py
```

## Endpoints

| Endpoint | Description |
|---|---|
| `http://192.168.4.1:81/stream` | MJPEG live stream (primary — used by VisionMate) |
| `http://192.168.4.1/capture` | Single JPEG snapshot |
| `http://192.168.4.1/info` | JSON info: fps, framesize, quality, uptime |
| `http://visionmatecam.local:81/stream` | mDNS fallback (same stream) |
