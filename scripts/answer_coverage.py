"""Answer-string coverage of the corpus: for each dev question, is any reference answer
present verbatim (after lowercasing and punctuation stripping) somewhere in docs.jsonl?

This is an upper bound on what an extractive reader can find, and a lower bound on real
coverage: long free-text reference answers rarely appear verbatim even when the facts do.

    python scripts/answer_coverage.py [--misses]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/processed/docs.jsonl")
    ap.add_argument("--dev", type=Path, default=ROOT / "data/eval/public_evaluation_data.json")
    ap.add_argument("--misses", action="store_true", help="print uncovered questions")
    args = ap.parse_args()

    corpus = " \n ".join(norm(json.loads(line)["text"]) for line in args.corpus.open())
    dev = json.load(args.dev.open())
    short = lambda d: min(len(a.split()) for a in d["answers"]) <= 5
    hit = {d["id"]: any(norm(a) and norm(a) in corpus for a in d["answers"]) for d in dev}
    by = Counter((short(d), hit[d["id"]]) for d in dev)
    n_short = by[(True, True)] + by[(True, False)]
    print(f"covered: {sum(hit.values())}/{len(dev)}")
    print(f"  short answers (<=5 words): {by[(True, True)]}/{n_short}")
    print(f"  long answers:              {by[(False, True)]}/{len(dev) - n_short}")
    if args.misses:
        for d in dev:
            if not hit[d["id"]]:
                print(f"{d['id']:>4} | {d['question']} | {d['answers'][0][:80]}")


if __name__ == "__main__":
    main()
