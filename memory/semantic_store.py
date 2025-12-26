import os, json
from datetime import datetime
from typing import Dict, Any


class SemanticStore:
    """
    Semantic memory: per-patient rolling summary + stats.
    MVP update rule is deterministic (no extra LLM).
    """

    def __init__(self, root: str = "data/memory_db/semantic"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, patient_id: str) -> str:
        return os.path.join(self.root, f"{patient_id}.json")

    def get(self, patient_id: str) -> Dict[str, Any]:
        p = self._path(patient_id)
        if not os.path.exists(p):
            return {
                "patient_id": patient_id,
                "patient_summary": "No prior history.",
                "stats": {"total": 0, "adherent": 0, "nonadherent": 0, "ambiguous": 0, "last_seen": None},
                "last_summaries": [],
            }
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def update(self, patient_id: str, episode_summary: str, decision: str):
        state = self.get(patient_id)
        stats = state["stats"]

        stats["total"] += 1
        d = (decision or "").lower()
        if "adherent" in d and "non" not in d:
            stats["adherent"] += 1
        elif "non" in d:
            stats["nonadherent"] += 1
        else:
            stats["ambiguous"] += 1

        stats["last_seen"] = datetime.utcnow().isoformat()

        state["last_summaries"] = (state.get("last_summaries", []) + [episode_summary])[-3:]
        roll = " ".join(state["last_summaries"]).strip()
        state["patient_summary"] = (
            f"{roll} (Total={stats['total']}, A={stats['adherent']}, "
            f"NA={stats['nonadherent']}, Amb={stats['ambiguous']})."
            if roll else
            f"(Total={stats['total']}, A={stats['adherent']}, NA={stats['nonadherent']}, Amb={stats['ambiguous']})."
        )

        with open(self._path(patient_id), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
