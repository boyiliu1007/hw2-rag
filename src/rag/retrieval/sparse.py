"""Sparse retrieval: BM25 over chunks with bm25s (English stopwords + Snowball stemming).

    python -m src.rag.retrieval.sparse --strategy structure      # build + save index
"""
import argparse
from pathlib import Path

import bm25s
import Stemmer

from ..chunking import ROOT, STRATEGIES, load_chunks

INDEX_DIR = ROOT / "indexes"


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75, method="lucene"):
        self.stemmer = Stemmer.Stemmer("english")
        self.params = {"k1": k1, "b": b, "method": method}
        self.model = None

    def _tok(self, texts):
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)

    def index(self, texts):
        self.model = bm25s.BM25(**self.params)
        self.model.index(self._tok(texts), show_progress=False)
        return self

    def search(self, queries, k=10):
        ids, scores = self.model.retrieve(self._tok(queries), k=k, show_progress=False, n_threads=0)
        return [[(int(i), float(s)) for i, s in zip(row_i, row_s)] for row_i, row_s in zip(ids, scores)]

    def save(self, path):
        self.model.save(str(path))

    @classmethod
    def load(cls, path):
        r = cls()
        r.model = bm25s.BM25.load(str(path))
        return r


def index_path(strategy, k1=1.5, b=0.75):
    return INDEX_DIR / f"bm25_{strategy}_k{k1}_b{b}"


def get_bm25(strategy, k1=1.5, b=0.75, chunks=None):
    """Load the saved index for a chunking strategy, building it on first use."""
    p = index_path(strategy, k1, b)
    chunk_file = ROOT / "indexes" / "chunks" / f"{strategy}.jsonl"
    if p.exists() and p.stat().st_mtime >= chunk_file.stat().st_mtime:
        return BM25Retriever.load(p)
    chunks = chunks or load_chunks(strategy)
    r = BM25Retriever(k1=k1, b=b).index([c["text"] for c in chunks])
    r.save(p)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, default="structure")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()
    get_bm25(args.strategy, args.k1, args.b)
    print("saved", index_path(args.strategy, args.k1, args.b))


if __name__ == "__main__":
    main()
