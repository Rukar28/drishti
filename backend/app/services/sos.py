"""
VisionMate v2 - SOS Emergency Notification Service

Sends emergency alerts via email with cooldown protection.
Never exposes credentials in logs or user-facing responses.
"""

import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, Optional
from datetime import datetime

from backend.app.core.config import settings

logger = logging.getLogger("visionmate.sos")


class SOSNotificationService:
    """
    Emergency notification service with cooldown protection.
    Sends SOS alerts via SMTP email when configured.
    """

    def __init__(self):
        self.last_result = None
        self.location_provider = None
        self._last_sent_time: float = 0.0
        self._cooldown_sec: float = 60.0  # Default 60s cooldown
        self._enabled: bool = False
        self._check_configuration()

    def _check_configuration(self) -> None:
        """Check if SOS is properly configured."""
        if not settings.SOS_ENABLED:
            self._enabled = False
            logger.info("[SOS] Emergency notifications disabled (SOS_ENABLED=false)")
            return

        required_fields = [
            settings.SOS_RECIPIENT,
            settings.SOS_SMTP_HOST,
            settings.SOS_SMTP_PORT,
            settings.SOS_SMTP_USERNAME,
            settings.SOS_SMTP_PASSWORD,
            settings.SOS_FROM_EMAIL,
        ]

        if all(required_fields):
            self._enabled = True
            self._cooldown_sec = settings.SOS_COOLDOWN_SEC
            logger.info(
                "[SOS] Emergency notifications enabled (cooldown=%.1fs)",
                self._cooldown_sec,
            )
        else:
            self._enabled = False
            logger.warning(
                "[SOS] Emergency notifications disabled - missing required configuration"
            )

    def _is_in_cooldown(self) -> bool:
        """Check if SOS is in cooldown period."""
        if not self._enabled:
            return False

        now = time.time()
        elapsed = now - self._last_sent_time
        if elapsed < self._cooldown_sec:
            logger.info(
                "[SOS] Cooldown active (%.1fs remaining)",
                self._cooldown_sec - elapsed,
            )
            return True
        return False

    def _get_location_info(self) -> Dict[str, Any]:
        """
        Get location information from GPS provider.
        Returns location status without fabricating coordinates.
        """
        # Import here to avoid circular dependency
        from backend.app.hardware.gps import create_gps_provider

        gps = self.location_provider or create_gps_provider(settings.GPS_SOURCE)
        location = gps.get_location()

        if location.get("lat") is not None and location.get("lon") is not None:
            return {
                "available": True,
                "lat": location["lat"],
                "lon": location["lon"],
                "accuracy": location.get("accuracy"),
                "source": location.get("source", "unknown"),
            }
        else:
            return {
                "available": False,
                "status": location.get("status", "unavailable"),
                "source": location.get("source", "unknown"),
            }

    def _build_emergency_message(self) -> str:
        """Build emergency notification message with available information."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        location_info = self._get_location_info()

        message_lines = [
            "EMERGENCY ALERT - VisionMate / Drishti",
            "",
            f"Timestamp: {timestamp}",
            "",
        ]

        if location_info["available"]:
            message_lines.extend([
                "Location Available:",
                f"  Latitude: {location_info['lat']}",
                f"  Longitude: {location_info['lon']}",
                f"  Accuracy: {location_info.get('accuracy', 'unknown')}",
                f"  Source: {location_info['source']}",
                "",
            ])
        else:
            message_lines.extend([
                "Location: Unavailable",
                f"  Status: {location_info.get('status', 'unknown')}",
                f"  Source: {location_info['source']}",
                "",
            ])

        message_lines.extend([
            "This is an automated emergency alert from VisionMate/Drishti.",
            "Please check on the user immediately.",
        ])

        return "\n".join(message_lines)

    def _send_email(self, message: str) -> Dict[str, Any]:
        """
        Send emergency notification via SMTP.
        Returns success/failure result without exposing credentials.
        """
        try:
            msg = MIMEMultipart()
            msg["From"] = settings.SOS_FROM_EMAIL
            msg["To"] = settings.SOS_RECIPIENT
            msg["Subject"] = "EMERGENCY ALERT - VisionMate / Drishti"

            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP(
                settings.SOS_SMTP_HOST,
                settings.SOS_SMTP_PORT,
                timeout=10.0,
            ) as server:
                server.starttls()
                server.login(
                    settings.SOS_SMTP_USERNAME,
                    settings.SOS_SMTP_PASSWORD,
                )
                server.send_message(msg)

            logger.info("[SOS] Emergency notification sent successfully")
            return {
                "success": True,
                "status": "sent",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        except smtplib.SMTPAuthenticationError as e:
            logger.error("[SOS] SMTP authentication failed: %s", e)
            return {
                "success": False,
                "status": "failed",
                "error": "authentication_error",
            }

        except smtplib.SMTPConnectError as e:
            logger.error("[SOS] SMTP connection failed: %s", e)
            return {
                "success": False,
                "status": "failed",
                "error": "connection_error",
            }

        except Exception as e:
            logger.exception("[SOS] Unexpected error sending notification: %s", e)
            return {
                "success": False,
                "status": "failed",
                "error": "unknown_error",
            }

    def send_emergency_alert(self) -> Dict[str, Any]:
        """
        Send emergency notification with cooldown protection.
        Returns structured result without exposing credentials.
        """
        if not self._enabled:
            logger.warning("[SOS] Attempted to send alert but SOS is not configured")
            return {
                "success": False,
                "status": "not_configured",
                "message": "Emergency notifications are not configured",
            }

        if self._is_in_cooldown():
            return {
                "success": False,
                "status": "cooldown",
                "message": "Emergency notification already sent recently",
                "cooldown_remaining_sec": max(
                    0,
                    self._cooldown_sec - (time.time() - self._last_sent_time),
                ),
            }

        message = self._build_emergency_message()
        result = self._send_email(message)

        if result["success"]:
            self._last_sent_time = time.time()

        return result

    def health(self):
        remaining = max(0, self._cooldown_sec - (time.time() - self._last_sent_time))
        return {"configured": self._enabled, "status": "not_configured" if not self._enabled else "cooldown" if remaining else "ready",
                "cooldown_remaining_sec": remaining, "last_result": self.last_result}

    def reset_cooldown(self) -> None:
        """Reset cooldown timer (useful for testing)."""
        self._last_sent_time = 0.0
        logger.info("[SOS] Cooldown reset")


# Global SOS service instance
sos_service = SOSNotificationService()
