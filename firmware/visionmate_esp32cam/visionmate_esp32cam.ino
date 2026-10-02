/*
 * VisionMate v2 — ESP32-CAM Firmware (Low-Latency Optimized)
 * Target board: AI-Thinker ESP32-CAM (OV2640)
 *
 * Latency Optimizations:
 *   - CAMERA_GRAB_LATEST with double-buffering (fb_count = 2 in PSRAM)
 *   - TCP_NODELAY (client.setNoDelay(true)) to eliminate packet buffering lag
 *   - WiFi sleep disabled (WiFi.setSleep(false)) for continuous low-latency radio
 *   - Zero heap allocation in MJPEG stream loop (stack-allocated buffers)
 *   - Immediate frame buffer release (esp_camera_fb_return) to unblock sensor DMA
 *   - Non-blocking interleaved HTTP server processing during active streaming
 *   - Default FRAMESIZE_SVGA (800x600) + JPEG quality 12 for optimal YOLO/OCR detail
 *
 * Features:
 *   - Wi-Fi SoftAP (fixed IP 192.168.4.1)
 *   - MJPEG stream on port 81 at /stream
 *   - Single-frame JPEG capture at /capture
 *   - Camera info & metrics JSON at /info
 *   - mDNS hostname: visionmatecam.local
 *   - Configurable resolution & quality via secrets.h
 */

#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include "secrets.h"           // AP_SSID, AP_PASSWORD, CAMERA_JPEG_QUALITY, CAMERA_FRAMESIZE

// ── AI-Thinker ESP32-CAM pin map ─────────────────────────────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ── SoftAP IP (fixed — never changes) ────────────────────────
IPAddress AP_IP(192, 168, 4, 1);
IPAddress AP_GATEWAY(192, 168, 4, 1);
IPAddress AP_SUBNET(255, 255, 255, 0);

// ── HTTP servers ──────────────────────────────────────────────
WebServer streamServer(81);     // MJPEG stream  → :81/stream
WebServer httpServer(80);       // /capture, /info, /

// ── mDNS hostname ─────────────────────────────────────────────
const char* MDNS_HOSTNAME = "visionmatecam";

// ── Diagnostics & Telemetry ──────────────────────────────────
volatile uint32_t frameCount     = 0;
volatile uint32_t startMs        = 0;
volatile uint32_t lastCaptureUs  = 0;   // Capture time in microseconds
volatile size_t   lastJpegBytes  = 0;

// ─────────────────────────────────────────────────────────────
// Camera initialisation
// ─────────────────────────────────────────────────────────────
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;       // 20MHz high speed clock
  config.pixel_format = PIXFORMAT_JPEG;

  // Use PSRAM if available for double buffering with newest frame grab
  if (psramFound()) {
    config.frame_size   = CAMERA_FRAMESIZE;     // Default: FRAMESIZE_SVGA (800x600)
    config.jpeg_quality = CAMERA_JPEG_QUALITY;  // Default: 12
    config.fb_count     = 2;                    // Double buffer
    config.grab_mode    = CAMERA_GRAB_LATEST;   // Always return newest frame, discard stale
  } else {
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  // Sensor tuning for clear, stable real-time vision
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_special_effect(s, 0);
    s->set_whitebal(s, 1);          // Auto white balance ON
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);           // Auto WB
    s->set_exposure_ctrl(s, 1);     // Auto exposure ON
    s->set_aec2(s, 1);              // Night/day auto exposure
    s->set_gain_ctrl(s, 1);         // Auto gain ON
    s->set_agc_gain(s, 0);
    s->set_gainceiling(s, (gainceiling_t)2);
    s->set_bpc(s, 0);
    s->set_wpc(s, 1);
    s->set_raw_gma(s, 1);
    s->set_lenc(s, 1);
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
    s->set_dcw(s, 1);
    s->set_colorbar(s, 0);
  }

  Serial.println("Camera initialised successfully (Low-Latency mode).");
  return true;
}

// ─────────────────────────────────────────────────────────────
// Low-Latency MJPEG stream handler (:81/stream)
// ─────────────────────────────────────────────────────────────
void handleMjpegStream() {
  WiFiClient client = streamServer.client();
  if (!client) return;

  // 1. Disable Nagle's algorithm for immediate TCP packet transmission
  client.setNoDelay(true);
  client.setTimeout(2);

  // 2. Send standard multipart HTTP response header
  static const char HTTP_HEADER[] =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Cache-Control: no-cache, no-store, must-revalidate\r\n"
    "Pragma: no-cache\r\n"
    "Connection: close\r\n\r\n";

  client.write((const uint8_t*)HTTP_HEADER, sizeof(HTTP_HEADER) - 1);

  static const char BOUNDARY[] = "--frame\r\n";
  char part_buf[80];

  // 3. Stream loop with zero dynamic memory allocation
  while (client.connected()) {
    int64_t t0 = esp_timer_get_time();
    camera_fb_t* fb = esp_camera_fb_get();
    int64_t t1 = esp_timer_get_time();

    if (!fb) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }

    lastCaptureUs = (uint32_t)(t1 - t0);
    lastJpegBytes = fb->len;

    int hlen = snprintf(part_buf, sizeof(part_buf),
      "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
      (unsigned int)fb->len);

    size_t w1 = client.write((const uint8_t*)BOUNDARY, sizeof(BOUNDARY) - 1);
    size_t w2 = client.write((const uint8_t*)part_buf, (size_t)hlen);
    size_t w3 = client.write(fb->buf, fb->len);
    size_t w4 = client.write((const uint8_t*)"\r\n", 2);

    // Immediately return the frame buffer to the driver to prevent lag accumulation
    esp_camera_fb_return(fb);

    if (w1 == 0 || w2 == 0 || w3 == 0 || w4 == 0) {
      // Client disconnected or pipe broke
      break;
    }

    frameCount++;

    // Service HTTP requests on port 80 (e.g., /info or /capture) concurrently
    httpServer.handleClient();

    // Yield 1ms to RTOS scheduler and TCP/IP stack
    vTaskDelay(1);
  }

  client.stop();
}

// ─────────────────────────────────────────────────────────────
// Single capture handler (/capture)
// ─────────────────────────────────────────────────────────────
void handleCapture() {
  int64_t t0 = esp_timer_get_time();
  camera_fb_t* fb = esp_camera_fb_get();
  int64_t t1 = esp_timer_get_time();

  if (!fb) {
    httpServer.send(503, "text/plain", "Camera frame unavailable");
    return;
  }

  lastCaptureUs = (uint32_t)(t1 - t0);
  lastJpegBytes = fb->len;

  httpServer.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
  httpServer.sendHeader("Access-Control-Allow-Origin", "*");
  httpServer.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  httpServer.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);

  esp_camera_fb_return(fb);
}

// ─────────────────────────────────────────────────────────────
// Info endpoint (/info)
// ─────────────────────────────────────────────────────────────
void handleInfo() {
  sensor_t* s = esp_camera_sensor_get();
  uint32_t uptimeSec = millis() / 1000;
  float fps = (millis() - startMs) > 0
    ? (float)frameCount / ((millis() - startMs) / 1000.0f)
    : 0.0f;

  int fs = s ? s->status.framesize : -1;
  const char* resStr = "unknown";
  if (fs == FRAMESIZE_QVGA)  resStr = "320x240 (QVGA)";
  else if (fs == FRAMESIZE_VGA)  resStr = "640x480 (VGA)";
  else if (fs == FRAMESIZE_SVGA) resStr = "800x600 (SVGA)";
  else if (fs == FRAMESIZE_XGA)  resStr = "1024x768 (XGA)";
  else if (fs == FRAMESIZE_HD)   resStr = "1280x720 (HD)";
  else if (fs == FRAMESIZE_UXGA) resStr = "1600x1200 (UXGA)";

  String json = "{";
  json += "\"firmware\":\"VisionMate-v2-LowLatency\",";
  json += "\"ssid\":\"" + String(AP_SSID) + "\",";
  json += "\"ip\":\"192.168.4.1\",";
  json += "\"stream_url\":\"http://192.168.4.1:81/stream\",";
  json += "\"capture_url\":\"http://192.168.4.1/capture\",";
  json += "\"mdns\":\"visionmatecam.local\",";
  json += "\"resolution\":\"" + String(resStr) + "\",";
  json += "\"framesize\":" + String(fs) + ",";
  json += "\"quality\":" + String(s ? s->status.quality : -1) + ",";
  json += "\"psram\":" + String(psramFound() ? "true" : "false") + ",";
  json += "\"wifi_sleep\":false,";
  json += "\"tcp_nodelay\":true,";
  json += "\"last_capture_ms\":" + String(lastCaptureUs / 1000.0f, 2) + ",";
  json += "\"last_jpeg_kb\":" + String(lastJpegBytes / 1024.0f, 2) + ",";
  json += "\"frames_sent\":" + String(frameCount) + ",";
  json += "\"stream_fps\":" + String(fps, 1) + ",";
  json += "\"free_heap\":" + String(ESP.getFreeHeap()) + ",";
  json += "\"free_psram\":" + String(ESP.getFreePsram()) + ",";
  json += "\"uptime_sec\":" + String(uptimeSec);
  json += "}";

  httpServer.sendHeader("Access-Control-Allow-Origin", "*");
  httpServer.sendHeader("Cache-Control", "no-cache");
  httpServer.send(200, "application/json", json);
}

void handleRoot() {
  httpServer.sendHeader("Location", "/info", true);
  httpServer.send(302, "text/plain", "");
}

// ─────────────────────────────────────────────────────────────
// setup()
// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println("\n========================================");
  Serial.println("  VisionMate v2 ESP32-CAM (Low-Latency) ");
  Serial.println("========================================");

  // 1. Initialise camera
  if (!initCamera()) {
    Serial.println("FATAL: Camera init failed. Halting.");
    while (true) delay(1000);
  }

  // 2. Create Wi-Fi SoftAP with fixed IP & disable modem sleep
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);   // Disable power saving to eliminate radio latency jitter
  WiFi.softAPConfig(AP_IP, AP_GATEWAY, AP_SUBNET);
  if (!WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    Serial.println("FATAL: SoftAP creation failed. Halting.");
    while (true) delay(1000);
  }
  Serial.printf("SoftAP started: SSID=%s  IP=%s (WiFi Sleep: OFF)\n",
    AP_SSID, WiFi.softAPIP().toString().c_str());

  // 3. mDNS
  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    MDNS.addService("http", "tcp", 81);
    Serial.printf("mDNS: %s.local\n", MDNS_HOSTNAME);
  }

  // 4. Register route handlers
  streamServer.on("/stream", HTTP_GET, handleMjpegStream);
  streamServer.begin();
  Serial.println("MJPEG stream server ready on :81/stream (TCP_NODELAY: ON)");

  httpServer.on("/",        HTTP_GET, handleRoot);
  httpServer.on("/capture", HTTP_GET, handleCapture);
  httpServer.on("/info",    HTTP_GET, handleInfo);
  httpServer.begin();
  Serial.println("HTTP server ready on :80 (/info, /capture)");

  startMs = millis();
  Serial.println("VisionMate ESP32-CAM ready. Waiting for client connection...");
}

// ─────────────────────────────────────────────────────────────
// loop()
// ─────────────────────────────────────────────────────────────
void loop() {
  streamServer.handleClient();
  httpServer.handleClient();
  delay(1);  // Yield to RTOS scheduler
}
