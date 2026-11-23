"""Skill span extraction with the SkillSpan JobBERT token classifiers.

Two models are run over the text and their spans are merged:
  * jjzha/jobbert_skill_extraction      -> skills ("developing mobile applications")
  * jjzha/jobbert_knowledge_extraction  -> knowledge/tools ("Node.js", "Docker")
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import pipeline

from .embedder import DEVICE

SKILL_MODEL = "jjzha/jobbert_skill_extraction"
KNOWLEDGE_MODEL = "jjzha/jobbert_knowledge_extraction"
CHUNK_WORDS = 250   # stays under the 512-token limit of BERT-style models
MAX_SPAN_WORDS = 8
_OUTSIDE = {"O", "LABEL_0"}


@dataclass(frozen=True)
class Span:
    text: str
    kind: str  # "skill" | "knowledge"


class JobBertExtractor:
    def __init__(self) -> None:
        device = 0 if DEVICE == "cuda" else -1
        make = lambda name: pipeline(  # noqa: E731
            "token-classification", model=name, aggregation_strategy="simple", device=device
        )
        self._pipes = {"skill": make(SKILL_MODEL), "knowledge": make(KNOWLEDGE_MODEL)}

    @staticmethod
    def _chunks(text: str) -> list[str]:
        # One chunk per line: BERT ignores newlines, so joining lines would fuse
        # neighbouring list items ("PostgreSQL MongoDB Git GitHub") into one span.
        chunks = []
        for line in text.splitlines():
            words = line.split()
            chunks += [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
        return chunks

    def extract(self, text: str, batch_size: int = 16) -> list[Span]:
        chunks = self._chunks(text)
        spans: dict[str, Span] = {}
        for kind, pipe in self._pipes.items():
            outputs = pipe(chunks, batch_size=batch_size) if chunks else []
            for chunk, ents in zip(chunks, outputs):
                for e in ents:
                    if e.get("entity_group") in _OUTSIDE:
                        continue
                    # Slice the original text by offsets: keeps casing and avoids "##" artefacts.
                    span = chunk[e["start"] : e["end"]].strip(" \t.,;:-–—()[]")
                    if len(span) > 1 and len(span.split()) <= MAX_SPAN_WORDS:
                        spans.setdefault(re.sub(r"\s+", " ", span).lower(), Span(span, kind))
        return list(spans.values())
