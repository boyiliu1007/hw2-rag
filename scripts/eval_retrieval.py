"""Retrieval recall@k on a QA set: is any reference answer string inside the top-k chunks?

Reported separately for short-answer questions (<= 5 words; string match is meaningful) and
for all questions (long free-text answers rarely appear verbatim, so this undercounts).

    python scripts/eval_retrieval.py --retriever bm25 --strategy structure
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.chunking import STRATEGIES, load_chunks  # noqa: E402
from src.rag.evaluation import answer_in, is_short, normalize, qtype  # noqa: E402

KS = (1, 3, 5, 10, 20, 50)


def get_retriever(name, strategy, chunks, args):
    """bm25 | dense:<model> | hybrid-{rrf,weighted,rerank}:<model>, optionally followed by
    "+ce:<reranker>" for cross-encoder reranking of the top 50 (e.g. hybrid-rrf:bge-base+ce:bge-reranker-v2-m3)"""
    if "+ce:" in name:
        from src.rag.retrieval.rerank import RerankRetriever
        base_name, _, ce = name.partition("+ce:")
        return RerankRetriever(get_retriever(base_name, strategy, chunks, args), chunks, model=ce)
    from src.rag.retrieval.sparse import get_bm25
    if name == "bm25":
        return get_bm25(strategy, args.k1, args.b, chunks)
    kind, _, model = name.partition(":")
    from src.rag.retrieval.dense import get_dense
    if kind == "dense":
        return get_dense(model, strategy, chunks)
    if kind.startswith("hybrid-"):
        from src.rag.retrieval.hybrid import HybridRetriever
        return HybridRetriever(get_bm25(strategy, args.k1, args.b, chunks), get_dense(model, strategy, chunks),
                               method=kind.split("-", 1)[1], alpha=args.alpha, rrf_k=args.rrf_k)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="bm25",
                    help="bm25 | dense:<model> | hybrid-rrf:<model> | hybrid-weighted:<model> | hybrid-rerank:<model>")
    ap.add_argument("--strategy", choices=STRATEGIES, default="structure")
    ap.add_argument("--qa", type=Path, default=ROOT / "data/eval/public_evaluation_data.json")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=0.5, help="dense weight for hybrid-weighted")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--name", help="run name (default: <retriever>_<strategy>)")
    args = ap.parse_args()
    name = args.name or f"{args.retriever.replace(':', '-').replace('+', '_')}_{args.strategy}"

    chunks = load_chunks(args.strategy)
    qa = json.load(args.qa.open())
    retriever = get_retriever(args.retriever, args.strategy, chunks, args)
    results = retriever.search([q["question"] for q in qa], k=max(KS))

    norm_cache = {}
    rows = []
    for q, hits in zip(qa, results):
        rank = None
        for r, (idx, _) in enumerate(hits, 1):
            if idx not in norm_cache:
                norm_cache[idx] = normalize(chunks[idx]["text"])
            if answer_in(norm_cache[idx], q["answers"]):
                rank = r
                break
        rows.append({"id": q["id"], "question": q["question"], "type": qtype(q["question"]),
                     "short": is_short(q["answers"]), "first_hit_rank": rank,
                     "top_chunks": [chunks[i]["chunk_id"] for i, _ in hits[:10]]})

    out = ROOT / "runs" / "retrieval" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def recall(subset):
        n = len(subset)
        return n, [sum(1 for r in subset if r["first_hit_rank"] and r["first_hit_rank"] <= k) / max(n, 1)
                   for k in KS]

    print(f"\n{name}  ({args.qa.name}, {len(chunks)} chunks)")
    print(f"{'subset':<22}{'n':>5}" + "".join(f"{'R@' + str(k):>8}" for k in KS))
    groups = [("short answers", [r for r in rows if r["short"]]), ("all questions", rows)]
    by_type = defaultdict(list)
    for r in rows:
        if r["short"]:
            by_type[r["type"]].append(r)
    groups += [(f"  short / {t}", v) for t, v in sorted(by_type.items())]
    for label, subset in groups:
        n, rs = recall(subset)
        print(f"{label:<22}{n:>5}" + "".join(f"{x:>8.3f}" for x in rs))
    print(f"per-question results -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
