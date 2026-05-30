import faiss
import numpy as np
import json
import os
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = Lock()


class FAISSStore:
    """Thread-safe FAISS index with metadata persistence."""

    INDEX_FILE = "index.faiss"
    META_FILE = "meta.json"

    def __init__(self, index_path: str = settings.FAISS_INDEX_PATH, dim: int = settings.EMBEDDING_DIM):
        self.index_path = Path(index_path)
        self.dim = dim
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._index: Optional[faiss.Index] = None
        # metadata[faiss_id] = {"paper_id": int, "chunk_db_id": int}
        self._metadata: dict[int, dict] = {}
        self._load()

    def _load(self):
        idx_file = self.index_path / self.INDEX_FILE
        meta_file = self.index_path / self.META_FILE

        if idx_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(idx_file))
            with open(meta_file, "r") as f:
                raw = json.load(f)
                self._metadata = {int(k): v for k, v in raw.items()}
            logger.info("faiss_index_loaded", vectors=self._index.ntotal)
        else:
            self._index = faiss.IndexFlatIP(self.dim)  # Inner product for cosine sim
            logger.info("faiss_index_created", dim=self.dim)

    def _save(self):
        faiss.write_index(self._index, str(self.index_path / self.INDEX_FILE))
        with open(self.index_path / self.META_FILE, "w") as f:
            json.dump(self._metadata, f)

    def add_vectors(self, embeddings: np.ndarray, metadata_list: list[dict]) -> list[int]:
        """Add vectors and return their FAISS IDs."""
        with _lock:
            start_id = self._index.ntotal
            self._index.add(embeddings)
            faiss_ids = list(range(start_id, self._index.ntotal))
            for fid, meta in zip(faiss_ids, metadata_list):
                self._metadata[fid] = meta
            self._save()
        return faiss_ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        paper_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Search and optionally filter by paper_ids."""
        with _lock:
            if self._index.ntotal == 0:
                return []

            # Over-fetch when filtering
            fetch_k = top_k * 10 if paper_ids else top_k
            fetch_k = min(fetch_k, self._index.ntotal)

            query = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self._index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata.get(int(idx))
            if not meta:
                continue
            if paper_ids and meta.get("paper_id") not in paper_ids:
                continue
            results.append({
                "faiss_id": int(idx),
                "score": float(score),
                **meta,
            })
            if len(results) >= top_k:
                break

        return results

    def delete_by_paper(self, paper_id: int):
        """Remove all vectors for a paper (rebuild index)."""
        with _lock:
            keep_ids = [fid for fid, m in self._metadata.items() if m.get("paper_id") != paper_id]
            if len(keep_ids) == len(self._metadata):
                return  # nothing to remove

            # Reconstruct index
            new_index = faiss.IndexFlatIP(self.dim)
            new_meta = {}

            if keep_ids:
                vecs = np.zeros((len(keep_ids), self.dim), dtype=np.float32)
                self._index.reconstruct_batch(keep_ids, vecs)
                new_index.add(vecs)
                for new_id, old_id in enumerate(keep_ids):
                    new_meta[new_id] = self._metadata[old_id]

            self._index = new_index
            self._metadata = new_meta
            self._save()

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0


# Singleton
_store_instance: Optional[FAISSStore] = None


def get_faiss_store() -> FAISSStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = FAISSStore()
    return _store_instance
