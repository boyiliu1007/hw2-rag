"""Crawl coverage audit: for each crawl source, how many in-scope pages did we see linked
(or listed in a sitemap) but never fetch?

    python scripts/coverage_audit.py [source ...]
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.scrape.crawl import _domain, extract_links, load_sources, normalize_url  # noqa: E402

ASSETS = re.compile(r"\.(jpg|jpeg|png|gif|svg|webp|avif|mp4|mp3|zip|ics|css|js|xml|pdf|docx?|xlsx?|pptx?)(\?|$)", re.I)


def main():
    only = set(sys.argv[1:])
    sources = {s["name"]: s for s in load_sources() if s["type"] == "crawl" and (not only or s["name"] in only)}
    fetched, failed, paths = defaultdict(set), defaultdict(set), defaultdict(list)
    for r in map(json.loads, (ROOT / "data/raw/manifest.jsonl").open()):
        if r["source"] not in sources:
            continue
        (fetched if r.get("path") else failed)[r["source"]].add(r["url"])
        if r.get("path") and r.get("content_type") == "html":
            paths[r["source"]].append((r["url"], ROOT / r["path"]))
    print(f"{'source':<26}{'fetched':>8}{'failed':>7}{'linked, not fetched':>21}  examples")
    report = {}
    for name, src in sources.items():
        allow = [re.compile(p) for p in src.get("allow", [])]
        deny = [re.compile(p) for p in src.get("deny", [])]
        domains = {_domain(u) for u in src["seeds"]}

        def in_scope(u):
            if ASSETS.search(u) or any(d.search(u) for d in deny):
                return False
            return any(a.search(u) for a in allow) if allow else _domain(u) in domains
        missing = Counter()
        have = {normalize_url(u) for u in fetched[name]} | {normalize_url(u) for u in failed[name]}
        for url, p in paths[name]:
            if not p.exists():
                continue
            for link in extract_links(p.read_text(errors="ignore"), url):
                if in_scope(link) and link not in have:
                    missing[link] += 1
        report[name] = sorted(missing, key=lambda u: -missing[u])
        ex = ", ".join(u.split("//", 1)[-1][:60] for u in report[name][:3])
        print(f"{name:<26}{len(fetched[name]):>8}{len(failed[name]):>7}{len(missing):>21}  {ex}")
    out = ROOT / "runs" / "coverage_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out.relative_to(ROOT)} (all unfetched in-scope links, most-linked first)")


if __name__ == "__main__":
    main()
