"""Re-score saved runs (runs/qa/<name>/details.jsonl) with the current metrics, without
re-running retrieval or the reader.

    python scripts/score_run.py runs/qa/*                  # re-score
    python scripts/score_run.py --raw runs/qa/<run>          # use the reader's raw answers
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.evaluation import METRICS, is_short, score  # noqa: E402
from run_qa import REPORTED, summarize  # noqa: E402

use_raw = "--raw" in sys.argv
for d in map(Path, [a for a in sys.argv[1:] if not a.startswith("--")]):
    rows = [json.loads(l) for l in (d / "details.jsonl").open()]
    if use_raw:  # restore the reader's own answers (undo any earlier code-side rewriting)
        for r in rows:
            r["prediction"] = r.get("raw", r["prediction"])
        (d / "predictions.json").write_text(json.dumps({r["id"]: r["prediction"] for r in rows},
                                                       indent=1, ensure_ascii=False))
        if not any(r.get("answers") for r in rows):
            with (d / "details.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [r for r in rows if r.get("answers")]
    if not rows:
        continue
    for r in rows:
        r["short"], r["scores"] = is_short(r["answers"]), score(r["prediction"], r["answers"])
    with (d / "details.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(rows)
    (d / "scores.json").write_text(json.dumps(s, indent=1))
    print(f"{d.name:<44}" + "".join(f"{m} {s['all'][m]:.3f}  " for m in REPORTED)
          + f"| short F1 {s['short answers']['f1']:.3f} (n={s['short answers']['n']})")
