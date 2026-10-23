"""Merge everything collected into data/processed/docs.jsonl.

Inputs
  data/raw/manifest.jsonl + raw HTML/PDF   (crawl.py)
  data/processed/sources/*.jsonl           (wikipedia.py, events.py: already docs)
  data/baseline/**                         (course-provided baseline, any of .txt/.md/.html/.pdf/.json/.jsonl)

Output record: {doc_id, url, title, topic, source, text}

    python scripts/build_corpus.py
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.scrape.events import MIN_DATE, _ld_names, _ld_place, event_doc  # noqa: E402
from src.rag.scrape.html_clean import extract_jsonld_events, html_to_text  # noqa: E402
from src.rag.scrape.pdf_extract import pdf_to_text  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MIN_CHARS = 200          # drop near-empty pages (JS shells, redirects, error pages)
BOILER_FRAC = 0.3        # per-source line frequency above which a line is site chrome


def norm_key(text):
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def strip_site_boilerplate(docs):
    """Remove lines repeated across many pages of the same source (menus, footers, CTAs)."""
    by_src = defaultdict(list)
    for d in docs:
        by_src[d["source"]].append(d)
    for src, group in by_src.items():
        if len(group) < 5:
            continue
        freq = Counter()
        for d in group:
            freq.update({ln.strip() for ln in d["text"].split("\n") if ln.strip()})
        thresh = max(3, BOILER_FRAC * len(group))
        common = {ln for ln, c in freq.items() if c >= thresh and not ln.startswith("#")}
        for d in group:
            d["text"] = re.sub(r"\n{3,}", "\n\n", "\n".join(
                ln for ln in d["text"].split("\n") if ln.strip() not in common)).strip()


def jsonld_event_docs(html, source, page_url):
    docs = []
    for e in extract_jsonld_events(BeautifulSoup(html, "lxml")):
        start = str(e.get("startDate", ""))
        if not e.get("name") or start[:10] < MIN_DATE:
            continue
        venue, addr = _ld_place(e.get("location"))
        docs.append(event_doc(source, title=str(e["name"]), url=e.get("url") or page_url,
                              start=start, end=e.get("endDate"), venue=venue, address=addr,
                              performers=_ld_names(e.get("performer")),
                              organizer=_ld_names(e.get("organizer")),
                              description=e.get("description", "")))
    return docs


def from_manifest():
    recs = {}
    for line in (RAW / "manifest.jsonl").open():
        r = json.loads(line)
        if r.get("path"):
            recs[r["url"]] = r
    pages, events = [], []
    for r in tqdm(recs.values(), desc="raw pages"):
        path = ROOT / r["path"]
        if not path.exists():
            continue
        if r["content_type"] == "pdf":
            title, text = pdf_to_text(path.read_bytes())
            title = title or Path(r["url"]).stem.replace("_", " ").replace("-", " ")
        elif r["content_type"] == "html":
            html = path.read_text(errors="ignore")
            title, text = html_to_text(html, r["url"])
            events += jsonld_event_docs(html, r["source"], r["url"])
        else:
            continue
        pages.append({"doc_id": f"{r['source']}-{path.stem}", "url": r["url"], "title": title,
                      "topic": r["topic"], "source": r["source"], "text": text})
    strip_site_boilerplate(pages)
    return pages + events


def from_sources():
    docs = []
    for f in sorted((PROC / "sources").glob("*.jsonl")):
        docs += [json.loads(line) for line in f.open()]
    return docs


def from_baseline(base):
    """Course-provided baseline; format unknown in advance, so accept common file types."""
    docs = []
    if not base.exists():
        return docs
    for f in sorted(base.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        suf = f.suffix.lower()
        rel = str(f.relative_to(base))
        if suf in (".txt", ".md"):
            items = [(f.stem, f.read_text(errors="ignore"), "")]
        elif suf in (".html", ".htm"):
            t, x = html_to_text(f.read_text(errors="ignore"))
            items = [(t or f.stem, x, "")]
        elif suf == ".pdf":
            t, x = pdf_to_text(f.read_bytes())
            items = [(t or f.stem, x, "")]
        elif suf in (".json", ".jsonl"):
            raw = f.read_text(errors="ignore")
            try:
                data = json.loads(raw) if suf == ".json" else [json.loads(l) for l in raw.splitlines() if l.strip()]
            except json.JSONDecodeError:
                continue
            data = data if isinstance(data, list) else [data]
            items = []
            for d in data:
                if isinstance(d, dict):
                    text = d.get("text") or d.get("content") or d.get("body") or ""
                    items.append((d.get("title", f.stem), text, d.get("url", "")))
                elif isinstance(d, str):
                    items.append((f.stem, d, ""))
        else:
            continue
        for i, (title, text, url) in enumerate(items):
            docs.append({"doc_id": f"baseline-{hashlib.sha1(f'{rel}#{i}'.encode()).hexdigest()[:12]}",
                         "url": url, "title": title, "topic": "baseline", "source": "baseline",
                         "text": text.strip()})
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=ROOT / "data" / "baseline")
    ap.add_argument("--out", type=Path, default=PROC / "docs.jsonl")
    args = ap.parse_args()

    docs = from_sources() + from_manifest() + from_baseline(args.baseline)
    kept, seen, dropped = [], set(), Counter()
    for d in docs:
        is_event = "\nEvent: " in d["text"]
        if len(d["text"]) < (40 if is_event else MIN_CHARS):
            dropped["short"] += 1
            continue
        k = norm_key(d["text"])
        if k in seen or d["doc_id"] in seen:  # same event cross-posted by several CMU groups
            dropped["duplicate"] += 1
            continue
        seen.update((k, d["doc_id"]))
        kept.append(d)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(kept)} docs -> {args.out.relative_to(ROOT)}  (dropped: {dict(dropped)})")
    stats = defaultdict(lambda: [0, 0])
    for d in kept:
        stats[(d["topic"], d["source"])][0] += 1
        stats[(d["topic"], d["source"])][1] += len(d["text"])
    print(f"{'topic':<10} {'source':<24} {'docs':>6} {'chars':>10}")
    for (t, s), (n, c) in sorted(stats.items()):
        print(f"{t:<10} {s:<24} {n:>6} {c:>10,}")
    print(f"{'TOTAL':<35} {len(kept):>6} {sum(len(d['text']) for d in kept):>10,}")


if __name__ == "__main__":
    main()
