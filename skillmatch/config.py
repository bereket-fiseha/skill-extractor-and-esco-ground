"""Static configuration: model registry and default paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ESCO_CSV = ROOT / "skills_en.csv"
CACHE_DIR = ROOT / ".cache" / "vector_stores"


@dataclass(frozen=True)
class EmbeddingModel:
    name: str            # Hugging Face id
    label: str           # shown in the UI
    prefix: str = ""     # prepended to every text (e.g. E5 needs "query: ")
    note: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    m.label: m
    for m in [
        EmbeddingModel("BAAI/bge-small-en-v1.5", "BGE Small (fast)", note="33M params, quickest index build"),
        EmbeddingModel("BAAI/bge-base-en-v1.5", "BGE Base (balanced)", note="110M params"),
        EmbeddingModel("BAAI/bge-large-en-v1.5", "BGE Large (best quality)", note="335M params, slowest to index"),
        EmbeddingModel("intfloat/e5-base-v2", "E5 Base", prefix="query: ", note="110M params"),
        EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", "MiniLM L6 (fastest)", note="22M params"),
    ]
}
DEFAULT_EMBEDDING = "BGE Base (balanced)"


@dataclass(frozen=True)
class Reranker:
    name: str
    label: str
    note: str = ""


RERANKERS: dict[str, Reranker] = {
    r.label: r
    for r in [
        Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", "MiniLM L6 (fast)", "22M params"),
        Reranker("BAAI/bge-reranker-base", "BGE Reranker Base (accurate)", "278M params"),
        Reranker("BAAI/bge-reranker-large", "BGE Reranker Large (most accurate)", "560M params"),
    ]
}
DEFAULT_RERANKER = "BGE Reranker Base (accurate)"
