"""Two-stage matching: fast bi-encoder retrieval, then optional cross-encoder rerank."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .embedder import CrossEncoderReranker, Embedder
from .store import VectorStore


@dataclass
class Match:
    query: str
    label: str
    uri: str
    skill_type: str
    description: str
    bi_score: float
    rerank_score: float | None = None
    score: float = 0.0


@dataclass
class MatchResult:
    matches: list[Match] = field(default_factory=list)       # best match per ESCO skill, ranked
    per_query: dict[str, list[Match]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    n_queries: int = 0


class SkillMatcher:
    def __init__(self, embedder: Embedder, store: VectorStore, reranker: CrossEncoderReranker | None = None):
        self.embedder, self.store, self.reranker = embedder, store, reranker

    def match(
        self,
        queries: list[str],
        top_k: int = 5,
        retrieve_k: int = 15,
        min_bi_score: float = 0.0,
        min_final_score: float = 0.0,
        rerank_weight: float = 0.7,
        max_rerank_queries: int = 200,
    ) -> MatchResult:
        t0 = time.perf_counter()
        res = MatchResult(n_queries=len(queries))
        if not queries:
            return res

        # Stage 1: embed only the candidate phrases, search the pre-built store.
        q_emb = self.embedder.encode(queries)
        t1 = time.perf_counter()
        hits = self.store.search(q_emb, top_k=retrieve_k)
        t2 = time.perf_counter()
        res.timings.update(embed=t1 - t0, search=t2 - t1)

        concepts = self.store.catalog.concepts
        cands: dict[str, list[Match]] = {}
        for q, row in zip(queries, hits):
            ms = []
            for cidx, sim in row:
                if sim < min_bi_score:
                    break
                c = concepts.iloc[cidx]
                ms.append(Match(q, c.label, c.uri, c.skill_type, c.description, sim, score=sim))
            if ms:
                cands[q] = ms

        # Stage 2: cross-encoder rerank, limited to the most promising queries to bound latency.
        if self.reranker and cands:
            keep = sorted(cands, key=lambda q: cands[q][0].bi_score, reverse=True)[:max_rerank_queries]
            flat = [m for q in keep for m in cands[q]]
            probs = self.reranker.score([(m.query, m.label) for m in flat])
            for m, p in zip(flat, probs):
                m.rerank_score = float(p)
                m.score = rerank_weight * float(p) + (1 - rerank_weight) * m.bi_score
        res.timings["rerank"] = time.perf_counter() - t2

        for q, ms in cands.items():
            ms.sort(key=lambda m: m.score, reverse=True)
            kept = [m for m in ms[:top_k] if m.score >= min_final_score]
            if kept:
                res.per_query[q] = kept

        best: dict[str, Match] = {}
        for ms in res.per_query.values():
            top = ms[0]
            if top.uri not in best or top.score > best[top.uri].score:
                best[top.uri] = top
        res.matches = sorted(best.values(), key=lambda m: m.score, reverse=True)
        res.timings["total"] = time.perf_counter() - t0
        return res
