"""Wikipedia articles via the MediaWiki API (clean plaintext with section headings).

    python -m src.rag.scrape.wikipedia
"""
import json
import re

from tqdm import tqdm

from .crawl import ROOT, load_sources, url_id
from .html_clean import normalize_ws
from .http import Fetcher

API = "https://en.wikipedia.org/w/api.php"
OUT = ROOT / "data" / "processed" / "sources"
CACHE = ROOT / "data" / "raw" / "wikipedia"
# Trailing sections that are lists of links/citations rather than content
DROP_SECTIONS = re.compile(r"^(See also|References|Notes|Further reading|External links|"
                           r"Bibliography|Sources|Citations|Footnotes)$", re.I)


def clean_extract(text):
    """API plaintext marks headings as '== Heading =='; convert to markdown and drop tail sections."""
    out, skipping = [], False
    for line in text.split("\n"):
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            if level == 2:
                skipping = bool(DROP_SECTIONS.match(title))
            if not skipping:
                out.append(f"\n{'#' * level} {title}\n")
            continue
        if not skipping:
            out.append(line)
    return normalize_ws("\n".join(out))


def fetch(titles, fetcher):
    docs = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for title in tqdm(titles, desc="wikipedia"):
        cache = CACHE / f"{url_id(title)}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            r = fetcher.get(API, params={
                "action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "wiki",
                "redirects": 1, "inprop": "url", "titles": title, "format": "json", "formatversion": 2})
            if r is None or r.status_code != 200:
                print(f"  ! {title}: request failed")
                continue
            data = r.json()
            cache.write_text(json.dumps(data))
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            print(f"  ! {title}: missing")
            continue
        p = pages[0]
        text = clean_extract(p.get("extract", ""))
        docs.append({"doc_id": f"wiki-{p['pageid']}", "url": p.get("fullurl", ""),
                     "title": p["title"], "topic": None, "source": "wikipedia",
                     "text": f"# {p['title']}\n\n{text}"})
    return docs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = next(s for s in load_sources() if s["type"] == "wikipedia")
    docs = fetch(src["titles"], Fetcher(delay=1.0, retries=3, respect_robots=False))
    for d in docs:
        d["topic"] = src["topic"]
    with (OUT / "wikipedia.jsonl").open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} articles, {sum(len(d['text']) for d in docs)/1e6:.1f}M chars")


if __name__ == "__main__":
    main()
