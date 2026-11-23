"""Persistent vector store of ESCO label embeddings.

Built once per (embedding model, ESCO file, label options) and reused on every
later run, so only the candidate's skills ever need to be embedded at query time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import CACHE_DIR, EmbeddingModel
from .embedder import DEVICE, Embedder
from .esco import EscoCatalog


def store_dir(spec: EmbeddingModel, catalog: EscoCatalog) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", spec.name).strip("_")
    return CACHE_DIR / f"{slug}__{catalog.fingerprint}"


class VectorStore:
    def __init__(self, embeddings: np.ndarray, catalog: EscoCatalog):
        self.catalog = catalog
        # Keep the matrix on the accelerator: a 100k x 768 search is a ~1 ms matmul.
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self.matrix = torch.from_numpy(embeddings).to(DEVICE, dtype=dtype)
        self._owners = torch.from_numpy(catalog.text_to_concept).to(DEVICE)

    @staticmethod
    def exists(spec: EmbeddingModel, catalog: EscoCatalog) -> bool:
        return (store_dir(spec, catalog) / "manifest.json").exists()

    @classmethod
    def load_or_build(
        cls,
        embedder: Embedder,
        catalog: EscoCatalog,
        progress: Callable[[float], None] | None = None,
    ) -> "VectorStore":
        folder = store_dir(embedder.spec, catalog)
        manifest, emb_path = folder / "manifest.json", folder / "embeddings.npy"
        if manifest.exists() and emb_path.exists():
            emb = np.load(emb_path)
            if emb.shape[0] == len(catalog.texts):
                return cls(emb, catalog)

        emb = embedder.encode(catalog.texts, progress=progress)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb)
        manifest.write_text(
            json.dumps({"model": embedder.spec.name, "entries": len(catalog.texts), "dim": int(emb.shape[1])})
        )
        return cls(emb, catalog)

    def search(self, query_emb: np.ndarray, top_k: int = 10, overfetch: int = 4):
        """Per query, return up to `top_k` (concept_idx, cosine) pairs, one per ESCO skill."""
        q = torch.from_numpy(query_emb).to(self.matrix.device, dtype=self.matrix.dtype)
        sims = q @ self.matrix.T
        k = min(top_k * overfetch, sims.shape[1])
        vals, idx = torch.topk(sims, k=k, dim=1)
        concept = self._owners[idx].cpu().numpy()
        vals = vals.float().cpu().numpy()

        results = []
        for c_row, v_row in zip(concept, vals):
            seen, hits = set(), []
            for c, v in zip(c_row, v_row):
                if c not in seen:  # a skill matched via several labels counts once; best label wins
                    seen.add(c)
                    hits.append((int(c), float(v)))
                    if len(hits) == top_k:
                        break
            results.append(hits)
        return results
