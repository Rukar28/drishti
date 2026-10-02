/*
 * VisionMate v2 — ESP32-CAM Secrets (EXAMPLE)
 *
 * Copy this file to secrets.h and fill in your values.
 * secrets.h is gitignored and must NOT be committed.
 *
 * CAMERA_FRAMESIZE choices (from esp_camera.h):
 *   FRAMESIZE_QVGA   = 320x240  (fast, lower YOLO accuracy)
 *   FRAMESIZE_CIF    = 400x296
 *   FRAMESIZE_HVGA   = 480x320
 *   FRAMESIZE_VGA    = 640x480
 *   FRAMESIZE_SVGA   = 800x600  ← recommended for VisionMate
 *   FRAMESIZE_XGA    = 1024x768
 *   FRAMESIZE_UXGA   = 1600x1200 (large, slower)
 *
 * CAMERA_JPEG_QUALITY: 0 (best) – 63 (worst). 10–15 is practical range.
 *   Lower number = better quality = larger JPEG = more bandwidth.
 *   12 is recommended for VisionMate (balances YOLO accuracy & Wi-Fi bandwidth).
 */

#pragma once

// Wi-Fi SoftAP credentials
#define AP_SSID     "VisionMate"
#define AP_PASSWORD "your_password_here"   // Minimum 8 characters

// Camera configuration — optimised for real-time low latency & high accuracy
#define CAMERA_FRAMESIZE    FRAMESIZE_SVGA  // 800x600 — ideal balance of detail & speed
#define CAMERA_JPEG_QUALITY 12              // Balanced real-time quality & low latency (10-15 practical range)
