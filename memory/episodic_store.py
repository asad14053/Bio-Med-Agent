import os, json, uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import faiss


class EpisodicStore:
    """
    Episodic memory:
      key   = z_episode (video embedding or image embedding)
      value = per-session record (question, short answer, decision, media path, optional clip_desc)
    """

    def __init__(self, dim: int, root: str = "data/memory_db"):
        os.makedirs(root, exist_ok=True)
        self.dim = dim
        self.meta_path = os.path.join(root, "episodic.jsonl")
        self.index_path = os.path.join(root, "episodic.faiss")

        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        self.meta: List[Dict[str, Any]] = []

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.meta.append(json.loads(line))

        if os.path.exists(self.index_path) and len(self.meta) > 0:
            self.index = faiss.read_index(self.index_path)
        elif len(self.meta) > 0:
            Z = np.array([m["z"] for m in self.meta], dtype=np.float32)
            Z = self._norm(Z)
            self.index.add(Z)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def add(
        self,
        patient_id: str,
        z: np.ndarray,
        question: str,
        answer_short: str,
        decision: str,
        media_path: str,
        clip_desc: Optional[str] = None,
    ) -> str:
        episode_id = str(uuid.uuid4())
        item = {
            "episode_id": episode_id,
            "patient_id": patient_id,
            "ts": datetime.utcnow().isoformat(),
            "z": z.astype(np.float32).tolist(),
            "question": question,
            "answer_short": answer_short,
            "decision": decision,
            "media_path": media_path,
            "clip_desc": clip_desc or "",
        }

        zz = self._norm(np.array(item["z"], dtype=np.float32).reshape(1, -1))
        self.index.add(zz)
        self.meta.append(item)

        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
        faiss.write_index(self.index, self.index_path)

        return episode_id

    def query(
        self,
        z: np.ndarray,
        k: int = 5,
        patient_id: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        if len(self.meta) == 0:
            return []
        z = z.astype(np.float32).reshape(1, -1)
        z = self._norm(z)

        D, I = self.index.search(z, min(len(self.meta), k * 5))
        out = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0:
                continue
            item = self.meta[idx]
            if patient_id is None or item["patient_id"] == patient_id:
                out.append((item, float(score)))
            if len(out) >= k:
                break
        return out
