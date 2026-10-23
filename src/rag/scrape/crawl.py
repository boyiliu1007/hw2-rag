"""BFS crawler for `crawl` and `files` sources in sources.yaml.

Saves raw bytes to data/raw/<source>/<sha1(url)>.<ext> and appends a line per
fetched URL to data/raw/manifest.jsonl. Re-running skips URLs already in the
manifest, so an interrupted crawl can be resumed.

    python -m src.rag.scrape.crawl --only visitpittsburgh cmu_about
"""
import argparse
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

from .http import WAYBACK, Fetcher

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.jsonl"
SOURCES = Path(__file__).with_name("sources.yaml")


def load_sources(path=SOURCES):
    cfg = yaml.safe_load(Path(path).read_text())
    defaults = cfg.get("defaults", {})
    out = []
    for s in cfg["sources"]:
        merged = {**defaults, **s}
        merged["deny"] = defaults.get("deny", []) + s.get("deny", [])
        out.append(merged)
    return out


def normalize_url(url):
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    # strip Wayback prefix if a link was rewritten
    m = re.match(r"^/web/\d+(?:id_)?/(https?://.+)$", p.path)
    if p.netloc == "web.archive.org" and m:
        return normalize_url(m.group(1))
    netloc = p.netloc.lower()
    path = re.sub(r"/{2,}", "/", p.path) or "/"
    # drop tracking params
    query = "&".join(q for q in p.query.split("&") if q and not q.startswith(("utm_", "fbclid", "gclid")))
    return urlunparse((p.scheme.lower(), netloc, path, "", query, ""))


def url_id(url):
    return hashlib.sha1(url.encode()).hexdigest()[:16]


PERMANENT_FAIL = {401, 403, 404, 410, 451}


def load_manifest():
    """url -> record, keeping successes and permanent failures (transient ones are retried)."""
    seen = {}
    if MANIFEST.exists():
        for line in MANIFEST.open():
            rec = json.loads(line)
            if rec.get("path") or rec.get("status") in PERMANENT_FAIL:
                seen[rec["url"]] = rec
    return seen


def append_manifest(rec):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ext_for(content_type, url):
    ct = (content_type or "").lower()
    if "pdf" in ct or url.lower().endswith(".pdf"):
        return "pdf"
    if "json" in ct:
        return "json"
    if "html" in ct or "xml" in ct:
        return "html"
    return None  # skip images, binaries, etc.


def save(source, url, resp, wayback):
    ext = ext_for(resp.headers.get("content-type"), url)
    if ext is None:
        return None
    out = RAW / source["name"] / f"{url_id(url)}.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    rec = {
        "url": url,
        "fetched_url": resp.url,
        "source": source["name"],
        "topic": source["topic"],
        "path": str(out.relative_to(ROOT)),
        "content_type": ext,
        "status": resp.status_code,
        "wayback": wayback,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_manifest(rec)
    return rec


def extract_links(html, base):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        u = normalize_url(urljoin(base, href))
        if u:
            yield u


def _domain(url):
    return urlparse(url).netloc.removeprefix("www.")


def crawl_source(src, fetcher, seen):
    allow = [re.compile(p) for p in src.get("allow", [])]
    deny = [re.compile(p) for p in src.get("deny", [])]
    seed_domains = {_domain(s) for s in src["seeds"]}
    max_pages, max_depth = src["max_pages"], src["max_depth"]
    mode = src.get("fetch", "auto")

    def ok(u):
        if any(d.search(u) for d in deny):
            return False
        is_pdf = u.lower().endswith(".pdf")
        if is_pdf and not src.get("follow_pdfs"):
            return False
        if allow:
            return any(a.search(u) for a in allow)
        return _domain(u) in seed_domains

    # max_pages is per seed domain so multi-site sources (food festivals) share fairly
    per_domain = {d: 0 for d in seed_domains}
    queue = deque((normalize_url(s), 0) for s in src["seeds"])
    queued = {u for u, _ in queue}
    bar = tqdm(desc=src["name"], unit="pg")
    while queue:
        url, depth = queue.popleft()
        dom = _domain(url)
        if per_domain.get(dom, 0) >= max_pages:
            continue
        rec = seen.get(url)
        if rec is None:
            resp, wb = fetcher.get_with_fallback(url, mode)
            if resp is None or resp.status_code != 200:
                append_manifest({"url": url, "source": src["name"], "topic": src["topic"],
                                 "status": resp.status_code if resp is not None else None,
                                 "path": None})
                seen[url] = {"path": None}
                continue
            rec = save(src, url, resp, wb)
            seen[url] = rec or {"path": None}
            if rec is None:
                continue
            bar.update(1)
        per_domain[dom] = per_domain.get(dom, 0) + 1
        if not rec or not rec.get("path") or rec.get("content_type") != "html" or depth >= max_depth:
            continue
        html = (ROOT / rec["path"]).read_text(errors="ignore")
        base = rec.get("fetched_url", url)
        if base.startswith(WAYBACK[:28]):
            base = url
        for link in extract_links(html, base):
            if link not in queued and ok(link):
                queued.add(link)
                queue.append((link, depth + 1))
    bar.close()


def fetch_files(src, fetcher, seen):
    mode = src.get("fetch", "auto")
    for url in tqdm(src["urls"], desc=src["name"]):
        url = normalize_url(url)
        if url in seen:
            continue
        resp, wb = fetcher.get_with_fallback(url, mode)
        if resp is None or resp.status_code != 200:
            print(f"  ! failed {url} ({resp.status_code if resp is not None else 'error'})")
            continue
        seen[url] = save(src, url, resp, wb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="source names to run (default: all crawl/files sources)")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-pages", type=int, help="override max_pages (for quick tests)")
    args = ap.parse_args()

    seen = load_manifest()
    for src in load_sources():
        if src["type"] not in ("crawl", "files"):
            continue
        if args.only and src["name"] not in args.only:
            continue
        if args.max_pages:
            src["max_pages"] = args.max_pages
        fetcher = Fetcher(delay=src.get("delay", args.delay))
        print(f"== {src['name']} ({src['type']}, fetch={src.get('fetch')})")
        (crawl_source if src["type"] == "crawl" else fetch_files)(src, fetcher, seen)


if __name__ == "__main__":
    main()
