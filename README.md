# 11-711 HW2 — Pittsburgh/CMU RAG

Retrieval-augmented QA over a self-collected knowledge resource about Pittsburgh and CMU.
Status: **data collection pipeline done**; chunking / retrieval / reader / evaluation to follow.

## Setup

```bash
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Building the knowledge resource

```bash
scripts/scrape_all.sh                # everything below, then build_corpus.py
```

or step by step:

```bash
.venv/bin/python -m src.rag.scrape.wikipedia                 # MediaWiki API plaintext
.venv/bin/python -m src.rag.scrape.events                    # structured event feeds
.venv/bin/python -m src.rag.scrape.crawl --only cmu_about    # HTML/PDF crawl (any source names)
.venv/bin/python scripts/build_corpus.py                     # -> data/processed/docs.jsonl
```

All steps are resumable: raw responses are cached under `data/raw/` and listed in
`data/raw/manifest.jsonl`; transient failures (429/5xx) are retried on the next run.

If the course baseline knowledge resource is available, place it in `data/baseline/`
(`.txt/.md/.html/.pdf/.json/.jsonl` are all accepted) and `build_corpus.py` merges it in.

### Sources (`src/rag/scrape/sources.yaml`)

| Type | How it is collected | Sources |
|---|---|---|
| `wikipedia` | MediaWiki API `extracts` (clean text, section headings kept, References/See also dropped) | ~105 articles: city/history, CMU, culture, museums, food, sports |
| `events` | Site-specific structured feeds → one document per event | pittsburgh.events (sitemap → schema.org JSON-LD), downtownpittsburgh.com (WordPress REST API), events.cmu.edu (LiveWhale JSON API) |
| `crawl` | Polite BFS (robots.txt, per-domain delay, depth/page caps, allow/deny regexes) | Visit Pittsburgh, pittsburghpa.gov, Britannica, CMU about/history, Symphony, Opera, Cultural Trust, Carnegie Museums, Heinz History Center, The Frick, food festival sites, Pirates/Steelers/Penguins, City Paper |
| `files` | Direct PDF download | City tax regulations (5 regulation PDFs + taxpayer bill of rights, rates, due dates), 2025 Operating Budget |

Notes
- pittsburghpa.gov, britannica.com and pghcitypaper.com block scripted clients (HTTP 403), so
  they are fetched from the Internet Archive's Wayback Machine (`fetch: wayback`).
- Event questions only concern Oct 2026 onward, so event docs before 2026-10-01 are dropped.
- Wikipedia and Wayback throttle browser-style headers; those domains get an honest bot User-Agent.

### Processing

- **HTML** (`html_clean.py`, BeautifulSoup + lxml): removes scripts, nav/header/footer and
  cookie/newsletter/social containers; keeps headings (`#`), list items and paragraph breaks;
  flattens tables to `Header: value | Header: value` rows; also pulls schema.org Event JSON-LD
  from any crawled page into event docs.
- **PDF** (`pdf_extract.py`, pdfplumber with pypdf fallback): per-page text plus row-wise table
  text, running headers/footers and TOC dot-leaders removed, hyphenation repaired.
- **Event docs** (`events.py`): a lead sentence plus explicit fields, e.g.
  ```
  Shane Patrick in The Oracle runs from Friday, October 2, 2026 to Saturday, October 17, 2026 at Liberty Magic.
  Event: Shane Patrick in The Oracle
  Date: ...  Venue: ...  Organizer: ...  Description: ...
  ```
- **Corpus build** (`scripts/build_corpus.py`): strips lines repeated on ≥30% of a site's
  pages (residual menus/footers), drops near-empty pages, exact-dedupes on normalized text.

Output: `data/processed/docs.jsonl`, one `{doc_id, url, title, topic, source, text}` per line.

## Repository layout

```
src/rag/scrape/      http.py, crawl.py, wikipedia.py, events.py, html_clean.py, pdf_extract.py, sources.yaml
scripts/             scrape_all.sh, build_corpus.py
data/raw/            cached raw responses + manifest (gitignored)
data/processed/      sources/*.jsonl (Wikipedia + event docs), docs.jsonl (final corpus)
data/eval/           public_evaluation_data.json
```

## Tools and data credits

requests, Beautiful Soup 4, lxml, pdfplumber, pypdf, PyYAML, tqdm; Wikipedia (CC BY-SA);
Internet Archive Wayback Machine; the public websites listed in `sources.yaml`.
