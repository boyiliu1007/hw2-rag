"""Hybrid retrieval: fuse a sparse and a dense ranking.

  rrf       Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d))    (Cormack et al., 2009)
  weighted  min-max normalize each retriever's scores over its own candidate list, then
            alpha * dense + (1 - alpha) * sparse; a document missing from one list gets 0 there
  rerank    take the sparse candidates and reorder them by dense similarity (dense "filters"
            nothing; sparse decides the pool)

Each retriever returns `depth` candidates; fusion happens over their union.
"""
from collections import defaultdict


def rrf(rankings, k=60):
    score = defaultdict(float)
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, 1):
            score[doc] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def _minmax(ranking):
    if not ranking:
        return {}
    vals = [s for _, s in ranking]
    lo, hi = min(vals), max(vals)
    return {d: (s - lo) / (hi - lo) if hi > lo else 1.0 for d, s in ranking}


def weighted(sparse, dense, alpha=0.5):
    s, d = _minmax(sparse), _minmax(dense)
    fused = {doc: alpha * d.get(doc, 0.0) + (1 - alpha) * s.get(doc, 0.0) for doc in set(s) | set(d)}
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, sparse, dense, method="rrf", alpha=0.5, rrf_k=60, depth=100):
        self.sparse, self.dense = sparse, dense
        self.method, self.alpha, self.rrf_k, self.depth = method, alpha, rrf_k, depth

    def search(self, queries, k=10):
        sp = self.sparse.search(queries, self.depth)
        if self.method == "rerank":
            out = []
            for q, cand in zip(queries, sp):
                ids = [i for i, _ in cand]
                qv = self.dense.encode([q], self.dense.cfg["query"])[0]
                vecs = self.dense.index.reconstruct_batch(ids) if ids else []
                sims = [(i, float(v @ qv)) for i, v in zip(ids, vecs)]
                out.append(sorted(sims, key=lambda x: -x[1])[:k])
            return out
        de = self.dense.search(queries, self.depth)
        if self.method == "rrf":
            return [rrf([a, b], self.rrf_k)[:k] for a, b in zip(sp, de)]
        if self.method == "weighted":
            return [weighted(a, b, self.alpha)[:k] for a, b in zip(sp, de)]
        raise ValueError(self.method)
