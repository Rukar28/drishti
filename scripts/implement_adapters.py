from pathlib import Path
p=Path('backend/app/services/sos.py');s=p.read_text();s=s.replace('        self._last_sent_time: float = 0.0','        self.last_result = None\n        self.location_provider = None\n        self._last_sent_time: float = 0.0',1);s=s.replace('        gps = create_gps_provider(settings.GPS_SOURCE)','        gps = self.location_provider or create_gps_provider(settings.GPS_SOURCE)');i=s.index('    def reset_cooldown');s=s[:i]+'''    def health(self):
        remaining = max(0, self._cooldown_sec - (time.time() - self._last_sent_time))
        return {"configured": self._enabled, "status": "not_configured" if not self._enabled else "cooldown" if remaining else "ready",
                "cooldown_remaining_sec": remaining, "last_result": self.last_result}

'''+s[i:];p.write_text(s)
p=Path('backend/app/services/pipeline.py');s=p.read_text();s=s.replace('        result = sos_service.send_emergency_alert()','        sos_service.location_provider = self.gps\n        result = sos_service.send_emergency_alert()\n        sos_service.last_result = {**result, "location": self.gps.get_location(), "event_time": time.time()}');s=s.replace('            "navigation": None,','            "navigation": self.navigator.health(),');p.write_text(s)
p=Path('backend/app/main.py');s=p.read_text();s=s.replace('logger = logging.getLogger("visionmate.main")','logger = logging.getLogger("visionmate.main")\n# Upstream Google query strings include server credentials.\nlogging.getLogger("httpx").setLevel(logging.WARNING)');p.write_text(s)
