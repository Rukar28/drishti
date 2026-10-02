"""
VisionMate v2 - Local ASR & Voice Command Provider
Implements ASRProvider for local speech-to-text and confident voice intent parsing.
Supports natural voice variations for FIND, STOP, READ, ASK, and GUIDANCE.
"""

import re
import numpy as np
from typing import Optional, Dict, Any
import logging
from backend.app.core.interfaces import ASRProvider

logger = logging.getLogger("visionmate.asr")

class LocalASRProvider(ASRProvider):
    """Local ASR intent parser with high-precision keyword/pattern matching."""
    def __init__(self, model_size: str = "tiny.en", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self.model = None

    def listen_chunk(self, pcm_audio: np.ndarray) -> Optional[str]:
        return None

    def parse_intent(self, text: str) -> Dict[str, Any]:
        """
        Parses user spoken text into system commands with confident pattern matching.
        """
        if not text or not text.strip():
            return {"intent": "UNKNOWN", "confidence": 0.0}

        raw = text.strip()
        q = raw.lower()
        # Clean punctuation
        q_clean = re.sub(r'[^\w\s]', '', q)

        # 1. STOP Intent (Highest priority: immediately purge & cancel)
        stop_patterns = [r'\bstop\b', r'\bcancel\b', r'\bquiet\b', r'\bhypersilence\b', r'\bhalt\b', r'\bshut up\b', r'\bbe quiet\b']
        for pat in stop_patterns:
            if re.search(pat, q_clean):
                return {
                    "intent": "STOP",
                    "mode": "stop",
                    "raw_text": raw,
                    "confidence": 0.98
                }

        # 2. FIND Intent: "Find bottle", "Find the bottle", "Can you find my bottle", "Locate chair"
        find_patterns = [
            r'^(?:can you\s+)?(?:please\s+)?find\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+)$',
            r'^(?:locate|search for|where is)\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+)$'
        ]
        for pat in find_patterns:
            m = re.match(pat, q_clean)
            if m:
                target = m.group(1).strip()
                if target:
                    return {
                        "intent": "FIND",
                        "mode": "find",
                        "target": target,
                        "raw_text": raw,
                        "confidence": 0.95
                    }

        # 3. READ Intent: "Read this", "Read the text", "Read sign", "Read full"
        read_patterns = [r'\bread\s+(?:this|the\s+text|text|the\s+sign|sign|full|document)\b', r'\bread\b', r'\bwhat does this say\b']
        for pat in read_patterns:
            if re.search(pat, q_clean):
                is_full = "full" in q_clean or "document" in q_clean
                return {
                    "intent": "READ",
                    "mode": "read",
                    "full_reading": is_full,
                    "raw_text": raw,
                    "confidence": 0.95
                }

        # 4. ASK Intent: "What am I looking at?", "Explain the scene", "Describe everything in front of me"
        ask_patterns = [
            r'what am i looking at',
            r'explain the scene',
            r'describe everything',
            r'what is in front of me',
            r'what is that',
            r'what is this'
        ]
        for pat in ask_patterns:
            if pat in q_clean:
                return {
                    "intent": "ASK",
                    "mode": "ask",
                    "query": raw,
                    "confidence": 0.90
                }

        # 5. GUIDANCE / AWARENESS Mode return
        if any(w in q_clean for w in ["guidance", "awareness", "resume", "normal mode"]):
            return {
                "intent": "GUIDANCE",
                "mode": "guidance",
                "raw_text": raw,
                "confidence": 0.90
            }

        # Default fallback to ASK if interrogative
        if q_clean.startswith(("what", "who", "where", "how", "why", "describe", "tell me")):
            return {
                "intent": "ASK",
                "mode": "ask",
                "query": raw,
                "confidence": 0.80
            }

        return {"intent": "UNKNOWN", "raw_text": raw, "confidence": 0.40}
