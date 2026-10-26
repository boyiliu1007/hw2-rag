"""Dense retrieval: sentence-transformers embeddings + FAISS inner-product index over
L2-normalized vectors (= cosine similarity).

    python -m src.rag.retrieval.dense --model bge-base --strategy structure   # build + save
"""
import argparse
import json

import faiss
import numpy as np

from ..chunking import ROOT, STRATEGIES, load_chunks

INDEX_DIR = ROOT / "indexes"

# name -> HF id, query prefix/prompt, passage prefix, max tokens. All released before 2026-01-01.
MODELS = {
    "minilm": {"hf": "sentence-transformers/all-MiniLM-L6-v2", "query": "", "doc": "", "max_len": 256},
    "bge-base": {"hf": "BAAI/bge-base-en-v1.5",
                 "query": "Represent this sentence for searching relevant passages: ", "doc": "",
                 "max_len": 512},
    "e5-base": {"hf": "intfloat/e5-base-v2", "query": "query: ", "doc": "passage: ", "max_len": 512},
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-Embedding-0.6B",
                   "query": "Instruct: Given a question about Pittsburgh or Carnegie Mellon University, "
                            "retrieve passages that answer the question\nQuery: ",
                   "doc": "", "max_len": 512},
}


def _load_model(name, device=None):
    from sentence_transformers import SentenceTransformer
    import torch
    cfg = MODELS[name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kw = {"model_kwargs": {"torch_dtype": torch.float16}} if device == "cuda" else {}
    m = SentenceTransformer(cfg["hf"], device=device, **kw)
    m.max_seq_length = cfg["max_len"]
    return m


class DenseRetriever:
    def __init__(self, name, model=None):
        self.name, self.cfg = name, MODELS[name]
        self._model = model
        self.index = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.name)
        return self._model

    def encode(self, texts, prefix, batch_size=64, progress=False):
        # longest-first so batches pad less; restore order afterwards
        order = np.argsort([-len(t) for t in texts])
        emb = self.model.encode([prefix + texts[i] for i in order], batch_size=batch_size,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=progress).astype(np.float32)
        out = np.empty_like(emb)
        out[order] = emb
        return out

    def build(self, texts, batch_size=64):
        emb = self.encode(texts, self.cfg["doc"], batch_size, progress=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        return self

    def search(self, queries, k=10):
        q = self.encode(list(queries), self.cfg["query"])
        scores, ids = self.index.search(q, k)
        return [[(int(i), float(s)) for i, s in zip(ri, rs) if i >= 0] for ri, rs in zip(ids, scores)]

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "meta.json").write_text(json.dumps({"model": self.cfg["hf"], "ntotal": self.index.ntotal}))

    @classmethod
    def load(cls, name, path):
        r = cls(name)
        r.index = faiss.read_index(str(path / "index.faiss"))
        return r


def index_path(name, strategy):
    return INDEX_DIR / f"dense_{name}_{strategy}"


def get_dense(name, strategy, chunks=None, batch_size=64):
    """Load the saved FAISS index for (model, chunking strategy), building it if missing or
    older than the chunk file."""
    p = index_path(name, strategy)
    chunk_file = INDEX_DIR / "chunks" / f"{strategy}.jsonl"
    if (p / "index.faiss").exists() and (p / "index.faiss").stat().st_mtime >= chunk_file.stat().st_mtime:
        return DenseRetriever.load(name, p)
    chunks = chunks or load_chunks(strategy)
    r = DenseRetriever(name).build([c["text"] for c in chunks], batch_size)
    r.save(p)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="bge-base")
    ap.add_argument("--strategy", choices=STRATEGIES + ("all",), default="structure")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    for s in (STRATEGIES if args.strategy == "all" else (args.strategy,)):
        get_dense(args.model, s, batch_size=args.batch_size)
        print("saved", index_path(args.model, s))


if __name__ == "__main__":
    main()
