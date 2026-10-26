"""End-to-end QA: retrieve -> read -> (score if reference answers exist).

    python scripts/run_qa.py --reader Qwen/Qwen2.5-14B-Instruct --name rag_rrf_bge_14b
    python scripts/run_qa.py --reader Qwen/Qwen2.5-14B-Instruct --closed-book --name closed_14b
    python scripts/run_qa.py --qa data/eval/leaderboard_queries.json ...     # no answers: predict only

Writes runs/qa/<name>/{config.json, predictions.json, details.jsonl, scores.json}.
predictions.json is {"<id>": "<answer>"}, the submission format.
"""
import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.chunking import STRATEGIES, load_chunks  # noqa: E402
from src.rag.evaluation import METRICS, is_short, qtype, score  # noqa: E402

REPORTED = list(METRICS) + ["rouge", "total3"]  # rouge = mean of ROUGE-1/2/L; total3 = mean(F1, recall, rouge)
from src.rag.reader import Reader, build_messages  # noqa: E402


def retrieve(spec, strategy, questions, k):
    from eval_retrieval import get_retriever  # same retriever specs as the retrieval evaluation
    chunks = load_chunks(strategy)
    args = argparse.Namespace(k1=1.5, b=0.75, alpha=0.5, rrf_k=60)
    r = get_retriever(spec, strategy, chunks, args)
    hits = r.search(questions, k=k)
    out = [[{**chunks[i], "score": s} for i, s in row] for row in hits]
    del r, chunks
    gc.collect()
    try:  # free the embedder's GPU memory before vLLM claims the GPU
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return out


def summarize(rows):
    def mean(rs, m):
        return sum(r["scores"][m] for r in rs) / max(len(rs), 1)
    groups = {"all": rows, "short answers": [r for r in rows if r["short"]],
              "long answers": [r for r in rows if not r["short"]]}
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["type"]].append(r)
    groups.update({f"type/{t}": v for t, v in sorted(by_t.items())})
    return {g: {"n": len(v), **{m: round(mean(v, m), 4) for m in REPORTED}} for g, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=ROOT / "data/eval/public_evaluation_data.json")
    ap.add_argument("--retriever", default="hybrid-rrf:bge-base")
    ap.add_argument("--strategy", choices=STRATEGIES, default="fixed")
    ap.add_argument("--k", type=int, default=5, help="passages given to the reader")
    ap.add_argument("--reader", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs for vLLM")
    ap.add_argument("--closed-book", action="store_true")
    ap.add_argument("--thinking", action="store_true", help="reader is a reasoning model (strip its reasoning)")
    ap.add_argument("--quantization", help='e.g. "fp8": 8-bit weights so a 30B reader fits on one 48 GB GPU')
    ap.add_argument("--style", choices=["short", "phrase", "adaptive", "routed", "full", "restate"], default="short",
                    help="answer length style (see src/rag/reader.py STYLES)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    out = ROOT / "runs" / "qa" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    qa = json.load(args.qa.open())
    qa = qa if isinstance(qa, list) else [{"id": k, "question": v} for k, v in qa.items()]
    questions = [q["question"] for q in qa]

    passages = [None] * len(qa) if args.closed_book else retrieve(args.retriever, args.strategy, questions, args.k)
    reader = Reader(args.reader, tensor_parallel_size=args.tp, thinking=args.thinking, quantization=args.quantization)
    msgs = [build_messages(q, p, style=args.style) for q, p in zip(questions, passages)]
    closed = None if args.closed_book else [build_messages(q, None, style=args.style) for q in questions]
    raw = reader.answer(msgs, fallback_lists=closed)
    answers = list(raw)  # answer format (dates, numbers) is set by the prompt, not rewritten in code

    preds = {str(q["id"]): a for q, a in zip(qa, answers)}
    (out / "predictions.json").write_text(json.dumps(preds, indent=1, ensure_ascii=False))

    rows = []
    with (out / "details.jsonl").open("w") as f:
        for q, a, ra, p in zip(qa, answers, raw, passages):
            row = {"id": str(q["id"]), "question": q["question"], "type": qtype(q["question"]),
                   "prediction": a, "raw": ra, "passages": [c["chunk_id"] for c in p] if p else []}
            if q.get("answers"):
                row.update(answers=q["answers"], short=is_short(q["answers"]), scores=score(a, q["answers"]))
                rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        summary = summarize(rows)
        (out / "scores.json").write_text(json.dumps(summary, indent=1))
        print(f"\n{args.name}  ({args.qa.name})")
        print(f"{'subset':<24}{'n':>5}" + "".join(f"{m:>9}" for m in REPORTED))
        for g, s in summary.items():
            print(f"{g:<24}{s['n']:>5}" + "".join(f"{s[m]:>9.3f}" for m in REPORTED))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
