"""Per-question error report for a run on a QA set with reference answers.

For each question: our answer vs the reference, F1, and where the pipeline failed:
  correct                 F1 >= 0.8
  reader                  a reference answer is in the passages the reader saw
  retrieval               a reference answer is somewhere in the corpus, but not in those passages
  missing                 no reference answer appears anywhere in the corpus
  long (unchecked)        sentence-style reference; verbatim string checks are not meaningful

    python scripts/error_report.py runs/qa/<name>    # -> runs/qa/<name>/error_report.{md,csv}
"""
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.chunking import load_chunks  # noqa: E402
from src.rag.evaluation import answer_in, normalize, score  # noqa: E402

ORDER = ["missing", "retrieval", "reader", "long (unchecked)", "correct"]


def passage_view(chunk, chunk_id, answers, width=320):
    """Title, url and a snippet: around the first reference-answer match if there is one."""
    if chunk is None:
        return {"chunk_id": chunk_id, "title": "(not in current index)", "url": "", "snippet": "",
                "has_answer": False}
    text = " ".join(chunk["text"].split())
    has = answer_in(normalize(text), answers)
    start = 0
    if has:
        low = text.lower()
        hits = [low.find(a.lower()) for a in answers if a and low.find(a.lower()) >= 0]
        if hits:
            start = max(0, min(hits) - width // 2)
    snippet = ("…" if start else "") + text[start:start + width] + ("…" if start + width < len(text) else "")
    return {"chunk_id": chunk_id, "title": chunk.get("title") or "", "url": chunk.get("url") or "",
            "snippet": snippet, "has_answer": has}


def main():
    run = Path(sys.argv[1])
    cfg = json.load((run / "config.json").open())
    rows = [json.loads(l) for l in (run / "details.jsonl").open()]
    chunks = {c["chunk_id"]: c for c in load_chunks(cfg.get("strategy", "fixed"))}
    norm_chunk = {k: normalize(c["text"]) for k, c in chunks.items()}
    docs = [json.loads(l) for l in (ROOT / "data/processed/docs.jsonl").open()]
    norm_docs = [(d, normalize(d["text"] + " " + (d.get("title") or ""))) for d in docs]

    out, passages = [], {}
    for r in rows:
        f1 = score(r["prediction"], r["answers"])["f1"]
        ctx = " ".join(norm_chunk.get(p, "") for p in r["passages"])
        where = []
        if f1 >= 0.8:
            status = "correct"
        elif not r["short"]:
            status = "long (unchecked)"
        elif answer_in(ctx, r["answers"]):
            status = "reader"
        else:
            where = [d for d, t in norm_docs if answer_in(t, r["answers"])]
            status = "retrieval" if where else "missing"
        top = chunks.get(r["passages"][0], {}) if r["passages"] else {}
        passages[r["id"]] = [passage_view(chunks.get(pid), pid, r["answers"]) for pid in r["passages"]]
        sc = score(r["prediction"], r["answers"])
        out.append({
            "id": r["id"], "status": status, "type": r["type"], "f1": round(sc["f1"], 3),
            "recall": round(sc["recall"], 3), "rouge": round(sc["rouge"], 3),
            "question": r["question"], "reference": " || ".join(r["answers"]), "ours": r["prediction"],
            "answer_found_in": "; ".join(f"{d['title'][:60]} <{d['url']}>" for d in where[:2]),
            "top_passage": f"{top.get('title', '')[:60]} <{top.get('url', '')}>" if top else "",
            "passages_with_answer": ",".join(str(i) for i, v in enumerate(passages[r["id"]], 1) if v["has_answer"]),
            "retrieved": " || ".join(f"[{i}] {v['title'][:60]}" for i, v in enumerate(passages[r["id"]], 1)),
        })

    out.sort(key=lambda x: (ORDER.index(x["status"]), x["type"], x["id"]))
    with (run / "error_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    counts = Counter(x["status"] for x in out)
    lost = Counter()
    for x in out:
        lost[x["status"]] += 1 - x["f1"]
    def lb(items):
        """Leaderboard-style numbers in %; Total without the LLM judge = mean(F1, Recall, ROUGE)."""
        n = max(len(items), 1)
        f, rc, rg = (100 * sum(x[k] for x in items) / n for k in ("f1", "recall", "rouge"))
        return f, rc, rg, (f + rc + rg) / 3

    f, rc, rg, tot = lb(out)
    md = [f"# Error report: `{run.name}` ({len(out)} questions)\n",
          "## Leaderboard metrics\n",
          "| subset | n | F1 (%) | Recall (%) | ROUGE (%) | Total w/o judge (%) |", "|---|---|---|---|---|---|",
          f"| **all** | {len(out)} | **{f:.2f}** | **{rc:.2f}** | **{rg:.2f}** | **{tot:.2f}** |"]
    for t in sorted({x["type"] for x in out}):
        items = [x for x in out if x["type"] == t]
        md.append(f"| type: {t} | {len(items)} | " + " | ".join(f"{v:.2f}" for v in lb(items)) + " |")
    md.append("\nROUGE = mean of ROUGE-1, ROUGE-2 and ROUGE-L; each metric takes the best reference answer. "
              "The leaderboard Total also averages in the LLM judge rescaled to (judge - 1) / 4, which "
              "cannot be computed locally.\n")
    md += ["## Where answers fail\n",
           "| status | questions | F1 (%) | Recall (%) | ROUGE (%) | F1 points lost (of 100) |",
           "|---|---|---|---|---|---|"]
    for st in ORDER:
        items = [x for x in out if x["status"] == st]
        if items:
            f_, rc_, rg_, _ = lb(items)
            md.append(f"| {st} | {len(items)} | {f_:.1f} | {rc_:.1f} | {rg_:.1f} | {100 * lost[st] / len(out):.1f} |")
    md.append("\n`missing` = no reference answer string anywhere in the corpus; `retrieval` = in the corpus "
              "but not in the reader's passages; `reader` = in the passages but answered wrong.\n")
    for s in ORDER:
        items = [x for x in out if x["status"] == s]
        if not items:
            continue
        md.append(f"\n## {s} ({len(items)})\n")
        md.append("| id | type | F1 | Recall | ROUGE | question | reference | ours | " +
                  ("answer is on" if s == "retrieval" else "reader's top passage") + " |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for x in items:
            cell = lambda v: str(v).replace("|", "/").replace("\n", " ")
            loc = x["answer_found_in"] if s == "retrieval" else x["top_passage"]
            md.append(f"| {x['id']} | {x['type']} | {x['f1']:.2f} | {x['recall']:.2f} | {x['rouge']:.2f} | {cell(x['question'])} | "
                      f"{cell(x['reference'][:120])} | {cell(x['ours'][:80])} | {cell(loc)} |")
    md.append("\n## Retrieved passages (questions not fully correct)\n")
    md.append("Passages in the order the reader saw them. ✅ = contains a reference answer; the snippet is "
              "centred on it. Passages whose id is missing come from an older index.\n")
    for x in out:
        if x["status"] == "correct":
            continue
        md.append(f"<details><summary><b>{x['id']}</b> [{x['status']}] {html.escape(x['question'])} "
                  f"— ours: <i>{html.escape(x['ours'][:80])}</i></summary>\n")
        md.append(f"Reference: {html.escape(x['reference'][:300])}\n")
        for i, v in enumerate(passages[x["id"]], 1):
            mark = "✅" if v["has_answer"] else "▫️"
            md.append(f"{i}. {mark} **{html.escape(v['title'][:90])}** "
                      f"<{v['url']}>  \n   `{v['chunk_id']}` — {html.escape(v['snippet'])}")
        md.append("\n</details>\n")
    (run / "error_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:6]))
    print(f"-> {run / 'error_report.md'}, {run / 'error_report.csv'}")


if __name__ == "__main__":
    main()
