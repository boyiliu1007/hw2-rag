"""Turn a run's predictions into a submission file and validate it against the question set.

    # leaderboard: adds the "andrewid" key
    python scripts/make_submission.py runs/qa/<name> --qa data/eval/leaderboard_queries.json \
        --andrewid boyil2 --out system_outputs/leaderboard_<name>.json
    # unseen test set: no andrewid key
    python scripts/make_submission.py runs/qa/<name> --qa <test questions> --out system_outputs/system_output_1.json
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="runs/qa/<name> directory with predictions.json")
    ap.add_argument("--qa", type=Path, required=True, help="question file the run was made on")
    ap.add_argument("--andrewid", help="add for leaderboard submissions only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    preds = json.load((args.run / "predictions.json").open())
    qa = json.load(args.qa.open())
    ids = [str(q["id"]) for q in qa] if isinstance(qa, list) else list(map(str, qa))

    missing = [i for i in ids if i not in preds]
    extra = [i for i in preds if i not in set(ids)]
    bad = [i for i in ids if i in preds and (not isinstance(preds[i], str) or not preds[i].strip())]
    if missing or extra:
        raise SystemExit(f"id mismatch: missing {missing[:10]} extra {extra[:10]}")
    if bad:
        print(f"warning: {len(bad)} empty answers, e.g. ids {bad[:10]} (submitted as-is)")

    sub = {"andrewid": args.andrewid} if args.andrewid else {}
    sub.update({i: preds[i].strip() for i in ids})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sub, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(ids)} answers" + (f', andrewid="{args.andrewid}"' if args.andrewid else ""))


if __name__ == "__main__":
    main()
