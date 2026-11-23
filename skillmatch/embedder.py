"""Thin wrappers around sentence-transformers models (bi-encoder and cross-encoder)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import EmbeddingModel, Reranker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """Produces L2-normalised embeddings so cosine similarity is a plain dot product."""

    def __init__(self, spec: EmbeddingModel):
        self.spec = spec
        self.model = SentenceTransformer(spec.name, device=DEVICE)
        if DEVICE == "cuda":
            self.model.half()
        self.model.max_seq_length = 64  # labels and skill phrases are short; keeps encoding fast

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        progress: Callable[[float], None] | None = None,
        chunk: int = 4096,
    ) -> np.ndarray:
        texts = [self.spec.prefix + t for t in texts]
        out = []
        for start in range(0, len(texts), chunk):
            out.append(
                self.model.encode(
                    texts[start : start + chunk],
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            )
            if progress:
                progress(min(1.0, (start + chunk) / max(len(texts), 1)))
        if not out:
            return np.zeros((0, 1), np.float16)
        return np.concatenate(out).astype(np.float16)


class CrossEncoderReranker:
    def __init__(self, spec: Reranker):
        self.model = CrossEncoder(spec.name, device=DEVICE, max_length=128)
        if DEVICE == "cuda":
            self.model.model.half()

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 128) -> np.ndarray:
        """Relevance probabilities in [0, 1]."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        logits = self.model.predict(
            pairs, batch_size=batch_size, activation_fn=torch.nn.Identity(), show_progress_bar=False
        )
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
