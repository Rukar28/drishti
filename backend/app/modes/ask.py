"""
VisionMate v2 - Ask Mode Handler (Visual Question Answering)
Handles on-demand VLM visual question answering using Qwen3-VL 4B via Ollama.
Never runs continuously per frame; triggered strictly on user query.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from backend.app.core.interfaces import VisionReasoner
from backend.app.priority.engine import PriorityLevel

logger = logging.getLogger("visionmate.ask")

class AskModeHandler:
    """Manages on-demand Visual Q&A queries."""
    def __init__(self, vision_reasoner: VisionReasoner):
        self.vision_reasoner = vision_reasoner
        self.latest_answer: Optional[str] = None

    def ask(self, frame: np.ndarray, user_query: str) -> Dict[str, Any]:
        """Dispatches user query and active frame to the VLM reasoner."""
        if frame is None:
            return {
                "text": "Cannot answer question. Camera image is unavailable.",
                "priority": PriorityLevel.INTERACTION,
                "success": False
            }

        prompt = user_query if user_query and user_query.strip() else "What is in front of me?"
        logger.info(f"Triggering Ask Mode VLM reasoner with query: '{prompt}'...")

        answer = self.vision_reasoner.reason(frame, prompt)
        self.latest_answer = answer

        return {
            "text": answer,
            "query": prompt,
            "priority": PriorityLevel.INTERACTION,
            "success": not answer.startswith(("Visual AI is unavailable", "Visual scene analysis took too long", "Unable to process camera image"))
        }
