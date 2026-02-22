# memory/semantic_store.py
import os
import json
from datetime import datetime
from typing import Dict, Any, List


class SemanticStore:
    """
    Radiology semantic memory (study-level).

    Stores a bounded rolling summary of prior study impressions
    for a patient. NO decisions, NO adherence stats.
    """

    def __init__(self, root: str = "data/memory_db/semantic", max_history: int = 3):
        self.root = root
        self.max_history = max_history
        os.makedirs(root, exist_ok=True)

    def _path(self, patient_id: str) -> str:
        return os.path.join(self.root, f"{patient_id}.json")

    def get(self, patient_id: str) -> Dict[str, Any]:
        p = self._path(patient_id)
        if not os.path.exists(p):
            return {
                "patient_id": patient_id,
                "last_updated": None,
                "study_impressions": [],   # list[str]
                "patient_summary": "No prior imaging history available.",
            }
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def update(self, patient_id: str, impression_summary: str):
        """
        Write back ONE study-level impression summary.
        """
        state = self.get(patient_id)

        history: List[str] = state.get("study_impressions", [])

        # normalize text
        impression_summary = " ".join((impression_summary or "").split())
        if not impression_summary:
            return

        # append + truncate
        history = (history + [impression_summary])[-self.max_history :]

        state["study_impressions"] = history
        state["last_updated"] = datetime.utcnow().isoformat()

        # build clean semantic context for prompting
        state["patient_summary"] = " ".join(history)

        with open(self._path(patient_id), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
