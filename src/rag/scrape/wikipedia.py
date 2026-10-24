"""Wikipedia articles via the MediaWiki API (clean plaintext with section headings).

Titles come from three places in the `wikipedia` source of sources.yaml:
  titles      hand-picked articles
  categories  every article in a category (optionally one level of subcategories)
  players     current-roster player names written by datasets.py (sports), resolved by search
  linked      lead sections of articles linked from `titles` whose categories match a pattern

    python -m src.rag.scrape.wikipedia
"""
import json
import re

from tqdm import tqdm

from .crawl import RAW, ROOT, load_sources, url_id
from .html_clean import normalize_ws
from .http import Fetcher

API = "https://en.wikipedia.org/w/api.php"
OUT = ROOT / "data" / "processed" / "sources"
CACHE = RAW / "wikipedia"
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


def _api(fetcher, cache_name, params):
    cache = CACHE / cache_name
    if cache.exists():
        return json.loads(cache.read_text())
    r = fetcher.get(API, params={"format": "json", "formatversion": 2, **params})
    if r is None or r.status_code != 200:
        return None
    data = r.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def category_titles(cat, depth, limit, fetcher):
    """Articles (namespace 0) in Category:cat, descending `depth` levels of subcategories."""
    titles, frontier, seen = [], [cat], set()
    for level in range(depth + 1):
        nxt = []
        for c in frontier:
            if c in seen:
                continue
            seen.add(c)
            data = _api(fetcher, f"cat_{url_id(c)}.json", {
                "action": "query", "list": "categorymembers", "cmtitle": f"Category:{c}",
                "cmtype": "page|subcat", "cmlimit": 500})
            for m in (data or {}).get("query", {}).get("categorymembers", []):
                if m["ns"] == 0:
                    titles.append(m["title"])
                elif m["ns"] == 14 and level < depth:
                    nxt.append(m["title"].removeprefix("Category:"))
        frontier = nxt
    return titles[:limit]


def resolve_player(name, hint, fetcher):
    """Search '<name> <sport>' and accept the top hit only if its title is the player's name
    (optionally with a disambiguator), so a same-named non-athlete is not pulled in."""
    data = _api(fetcher, f"search_{url_id(name + hint)}.json", {
        "action": "query", "list": "search", "srsearch": f'"{name}" {hint}', "srlimit": 3})
    for hit in (data or {}).get("query", {}).get("search", []):
        if re.match(rf"^{re.escape(name)}( \(.*\))?$", hit["title"], re.I):
            return hit["title"]
    return None


def query_batches(fetcher, kind, titles, size, params):
    """Yield API responses for `titles` in batches of `size`, following `continue`."""
    for i in range(0, len(titles), size):
        chunk, cont, k = titles[i:i + size], {}, 0
        while True:
            data = _api(fetcher, f"{kind}_{url_id('|'.join(chunk))}_{k}.json", {
                "action": "query", "titles": "|".join(chunk), "redirects": 1, **params, **cont})
            if data is None:
                print(f"  ! {kind} batch {i}: request failed")
                break
            yield data
            if "continue" not in data:
                break
            cont, k = data["continue"], k + 1


def linked_titles(seeds, pattern, fetcher):
    """Articles linked from `seeds` that carry a non-hidden category matching `pattern`."""
    links = set()
    for data in tqdm(query_batches(fetcher, "links", seeds, 50,
                                   {"prop": "links", "plnamespace": 0, "pllimit": "max"}), desc="links"):
        for p in data.get("query", {}).get("pages", []):
            links.update(l["title"] for l in p.get("links", []))
    cats = {}
    for data in tqdm(query_batches(fetcher, "cats", sorted(links), 50,
                                   {"prop": "categories", "cllimit": "max", "clshow": "!hidden"}),
                     desc="categories", total=len(links) // 50 + 1):
        for p in data.get("query", {}).get("pages", []):
            if not p.get("missing"):
                cats.setdefault(p["title"], set()).update(c["title"] for c in p.get("categories", []))
    rx = re.compile(pattern)
    keep = sorted(t for t, cs in cats.items() if any(rx.search(c) for c in cs))
    print(f"  {len(links)} linked articles, {len(keep)} with a matching category")
    return keep


def fetch_intros(titles, fetcher):
    """Lead sections, 20 articles per request (whole-article extracts are limited to 1)."""
    pages = {}
    for data in tqdm(query_batches(fetcher, "intro", titles, 20, {
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": 20,
            "exsectionformat": "wiki", "inprop": "url"}), desc="intros", total=len(titles) // 20 + 1):
        for p in data.get("query", {}).get("pages", []):
            if p.get("missing"):
                continue
            q = pages.setdefault(p["pageid"], p)
            if p.get("extract"):
                q["extract"] = p["extract"]
    return [{"doc_id": f"wiki-{p['pageid']}", "url": p.get("fullurl", ""), "title": p["title"],
             "topic": None, "source": "wikipedia", "meta": {"lead_only": True},
             "text": f"# {p['title']}\n\n{normalize_ws(p['extract'])}"}
            for p in pages.values() if len(p.get("extract", "")) > 100]


def fetch(titles, fetcher):
    docs, seen_ids = [], set()
    for title in tqdm(titles, desc="wikipedia"):
        data = _api(fetcher, f"{url_id(title)}.json", {
            "action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "wiki",
            "redirects": 1, "inprop": "url", "titles": title})
        if data is None:
            print(f"  ! {title}: request failed")
            continue
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            print(f"  ! {title}: missing")
            continue
        p = pages[0]
        if p["pageid"] in seen_ids:  # two titles redirecting to one article
            continue
        seen_ids.add(p["pageid"])
        text = clean_extract(p.get("extract", ""))
        docs.append({"doc_id": f"wiki-{p['pageid']}", "url": p.get("fullurl", ""),
                     "title": p["title"], "topic": None, "source": "wikipedia",
                     "text": f"# {p['title']}\n\n{text}"})
    return docs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = next(s for s in load_sources() if s["type"] == "wikipedia")
    fetcher = Fetcher(delay=src.get("delay", 2.0), retries=5, respect_robots=False)
    titles = list(src["titles"])
    for c in src.get("categories", []):
        titles += category_titles(c["name"], c.get("depth", 0), c.get("limit", 200), fetcher)
    players = RAW / "sports" / "players.json"
    if src.get("players") and players.exists():
        for name, hint in tqdm(json.loads(players.read_text()), desc="resolve players"):
            t = resolve_player(name, hint, fetcher)
            if t:
                titles.append(t)
    titles = [t for t in dict.fromkeys(titles) if not t.startswith(("List of people", "Lists of"))]
    print(f"{len(titles)} titles")
    docs = fetch(titles, fetcher)
    if src.get("linked"):
        have = {d["title"] for d in docs}
        extra = [t for t in linked_titles(src["titles"], src["linked"]["category_pattern"], fetcher)
                 if t not in have][:src["linked"].get("max", 5000)]
        ids = {d["doc_id"] for d in docs}
        docs += [d for d in fetch_intros(extra, fetcher) if d["doc_id"] not in ids]
    for d in docs:
        d["topic"] = src["topic"]
    with (OUT / "wikipedia.jsonl").open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} articles, {sum(len(d['text']) for d in docs)/1e6:.1f}M chars")


if __name__ == "__main__":
    main()
