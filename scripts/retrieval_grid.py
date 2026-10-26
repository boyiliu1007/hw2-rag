"""Retrieval comparison grid: every (retriever, chunking strategy) on a QA set, with a paired
McNemar test of hit@k against a baseline (BM25 on the same chunking).

    python scripts/retrieval_grid.py --embedders minilm bge-base qwen3-0.6b --k 5
"""
import argparse
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.chunking import STRATEGIES, load_chunks  # noqa: E402
from src.rag.evaluation import answer_in, is_short, normalize  # noqa: E402
from src.rag.retrieval.dense import get_dense, index_path  # noqa: E402
from src.rag.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.rag.retrieval.sparse import get_bm25  # noqa: E402

KS = (1, 5, 10)


def first_hit(results, chunks, qa, cache):
    ranks = []
    for q, hits in zip(qa, results):
        rank = None
        for r, (i, _) in enumerate(hits, 1):
            if i not in cache:
                cache[i] = normalize(chunks[i]["text"])
            if answer_in(cache[i], q["answers"]):
                rank = r
                break
        ranks.append(rank)
    return ranks


def mcnemar(a, b):
    """Exact McNemar p-value for paired binary outcomes a, b."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    return (binomtest(only_a, n, 0.5).pvalue if n else 1.0), only_a, only_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", default=["minilm", "bge-base"])
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--qa", type=Path, default=ROOT / "data/eval/public_evaluation_data.json")
    ap.add_argument("--k", type=int, default=5, help="hit@k used for the significance test")
    args = ap.parse_args()

    qa_all = json.load(args.qa.open())
    qa = [q for q in qa_all if is_short(q["answers"])]
    queries = [q["question"] for q in qa]
    rows = []
    for strategy in args.strategies:
        chunks = load_chunks(strategy)
        cache = {}
        bm25 = get_bm25(strategy, chunks=chunks)
        systems = [("bm25", bm25)]
        for e in args.embedders:
            if not (index_path(e, strategy) / "index.faiss").exists():
                print(f"  (skip {e}/{strategy}: index not built)")
                continue
            dense = get_dense(e, strategy, chunks)
            systems += [(f"dense:{e}", dense),
                        (f"rrf:{e}", HybridRetriever(bm25, dense, "rrf")),
                        (f"weighted:{e}", HybridRetriever(bm25, dense, "weighted", alpha=0.5)),
                        (f"rerank:{e}", HybridRetriever(bm25, dense, "rerank"))]
        base = None
        for name, r in systems:
            ranks = first_hit(r.search(queries, k=max(KS)), chunks, qa, cache)
            hit = [rk is not None and rk <= args.k for rk in ranks]
            base = base or hit
            p, wins, losses = mcnemar(hit, base)
            row = {"strategy": strategy, "system": name, "n": len(qa),
                   **{f"R@{k}": sum(1 for rk in ranks if rk and rk <= k) / len(qa) for k in KS},
                   "wins_vs_bm25": wins, "losses_vs_bm25": losses, "p_vs_bm25": p}
            rows.append(row)
            print(f"{strategy:<10}{name:<22}" + "".join(f"{row[f'R@{k}']:>8.3f}" for k in KS)
                  + f"   +{wins:<3}-{losses:<3} p={p:.3f}", flush=True)
            del r
    out = ROOT / "runs" / "retrieval" / "grid.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nn = {len(qa)} short-answer questions from {args.qa.name}; "
          f"significance: exact McNemar on hit@{args.k} vs BM25 on the same chunking -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
