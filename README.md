# 11-711 HW2 — Pittsburgh/CMU RAG

Retrieval-augmented QA over a self-collected knowledge resource about Pittsburgh and CMU:
BM25 + bge-base dense retrieval fused with RRF, a bge-reranker-v2-m3 cross-encoder, and a
Qwen3-30B-A3B-Instruct-2507 reader. The submitted test outputs are in `system_outputs/`
(see [Submitted test outputs](#submitted-test-outputs)).

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
| `events` | Site-specific structured feeds → one document per event | pittsburgh.events (sitemap → schema.org JSON-LD), downtownpittsburgh.com (WordPress REST API), events.cmu.edu (LiveWhale JSON API), calendar.pitt.edu (Localist API) |
| `crawl` | Polite BFS (robots.txt, per-domain delay, depth/page caps, allow/deny regexes) | Visit Pittsburgh, pittsburghpa.gov, Britannica, CMU about/history, Symphony, Opera, Cultural Trust, Carnegie Museums, Heinz History Center, The Frick, food festival sites, Pirates/Steelers/Penguins, City Paper |
| `files` | Direct PDF download | City tax regulations (5 regulation PDFs + taxpayer bill of rights, rates, due dates), 2025 Operating Budget, Allegheny County 2026 proposed fiscal plan |
| `dataset` | Structured data/APIs → one short doc per record (`datasets.py`) | Carnegie Museum of Art collection (28k artworks, CC0 2017 open-data release), Allegheny County municipal + school-district millage (2025–26), sports schedules from Oct 2026 (Penguins via NHL API; Steelers, Pitt, Duquesne, Robert Morris football and men's basketball via ESPN) and current Penguins/Pirates/Steelers roster bios (NHL API, MLB StatsAPI, ESPN) |

Beyond the handout list, more sources were added after checking which *kinds* of dev answers
were missing from the corpus (`scripts/answer_coverage.py --misses`) and which topics the leaderboard
questions cover. To avoid hand-picking documents for individual dev questions, every addition
is a whole resource or follows a fixed inclusion rule:

- whole datasets: the museum collection, all municipalities' millage, full team schedules/rosters;
- Wikipedia: whole Pittsburgh categories (museums, bridges, neighborhoods, venues, universities,
  college athletics, history, National Register / city historic designations, …), current-roster player articles, and the lead sections of
  every article linked from the original hand-picked articles whose categories mention
  Pittsburgh, Allegheny County or Carnegie Mellon;
- official sites of venues hosting ≥ 10 events in the event feeds (excluding venues already
  covered by a handout source), plus Pittsburgh museums/attractions/theaters;
- athletics sites (schedules, rosters, coaches) of every four-year college in the city with
  one (Pitt, Duquesne, Robert Morris, CMU, Chatham, Carlow, Point Park);
- the University of Pittsburgh events calendar (alongside the handout's CMU calendar);
- a deeper cmu.edu crawl (schools, music concerts, campus life, homecoming; ≤ 1,500 pages) and
  a larger Visit Pittsburgh crawl (≤ 1,500 pages, up from 400).

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
  The raw cache is not committed, so a source whose raw files are absent on the current
  machine keeps its documents from the previous `docs.jsonl` (`--no-carry` disables this).


Output: `data/processed/docs.jsonl`, one `{doc_id, url, title, topic, source, text}` per line. Git stores the gzip copy `docs.jsonl.gz` (the plain file exceeds GitHub's 100 MB limit); after cloning run `gunzip -k data/processed/docs.jsonl.gz`.

## Repository layout

```
src/rag/scrape/      http.py, crawl.py, wikipedia.py, events.py, datasets.py, html_clean.py, pdf_extract.py, sources.yaml
scripts/             scrape_all.sh, build_corpus.py, answer_coverage.py, entity_check.py, coverage_audit.py,
                     eval_retrieval.py, retrieval_grid.py, run_qa.py, score_run.py, error_report.py, make_submission.py, slurm/
data/raw/            cached raw responses + manifest (gitignored)
data/processed/      sources/*.jsonl (Wikipedia + event docs), docs.jsonl (final corpus)
data/eval/           public_evaluation_data.json
```

## Tools and data credits

requests, Beautiful Soup 4, lxml, pdfplumber, pypdf, PyYAML, tqdm; Wikipedia (CC BY-SA);
Internet Archive Wayback Machine; Carnegie Museum of Art collection data (CC0); Allegheny
County Treasurer millage tables; NHL, MLB StatsAPI and ESPN public JSON endpoints; the public
websites listed in `sources.yaml`.

## Running the QA system

```bash
# 1. chunk the corpus (fixed | sentence | structure) -> indexes/chunks/<strategy>.jsonl
.venv/bin/python -m src.rag.chunking --strategy all
# 2. indexes are built on first use (BM25 via bm25s; dense via sentence-transformers + FAISS)
.venv/bin/python -m src.rag.retrieval.sparse --strategy fixed
.venv/bin/python -m src.rag.retrieval.dense --model bge-base --strategy fixed
# 3. retrieval recall@k on the public dev set, and the retriever x chunking comparison grid
.venv/bin/python scripts/eval_retrieval.py --retriever hybrid-rrf:bge-base+ce:bge-reranker-v2-m3 --strategy fixed
.venv/bin/python scripts/retrieval_grid.py --embedders minilm bge-base qwen3-0.6b
# 4. end-to-end QA (needs GPUs; see scripts/slurm/qa.sbatch). Final system:
.venv/bin/python scripts/run_qa.py --retriever hybrid-rrf:bge-base+ce:bge-reranker-v2-m3 --k 10 \
    --reader Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 2 --style restate --name dev_restate
#    closed-book baseline: add --closed-book; other answer styles: --style short|routed|full
# 5. scores, per-question error report, and a validated submission file
.venv/bin/python scripts/score_run.py runs/qa/dev_restate
.venv/bin/python scripts/error_report.py runs/qa/dev_restate      # -> runs/qa/dev_restate/error_report.md
.venv/bin/python scripts/make_submission.py runs/qa/<run> --qa <questions.json> --out system_outputs/system_output_1.json
```

| Component | Implementation |
|---|---|
| Chunking | `src/rag/chunking.py`: fixed-size + overlap, sentence-aware, heading/paragraph-aware; per-record sources kept whole; title/section header on every chunk |
| Sparse | `src/rag/retrieval/sparse.py`: bm25s (Lucene BM25, k1=1.5, b=0.75), English stopwords + Snowball stemming |
| Dense | `src/rag/retrieval/dense.py`: sentence-transformers (all-MiniLM-L6-v2, bge-base-en-v1.5, Qwen3-Embedding-0.6B) + FAISS inner-product index |
| Hybrid | `src/rag/retrieval/hybrid.py`: reciprocal rank fusion, min-max weighted fusion, sparse-then-dense rerank |
| Reranker | `src/rag/retrieval/rerank.py`: cross-encoder over the top 50 (bge-reranker-v2-m3; Qwen3-Reranker compared) |
| Reader | `src/rag/reader.py`: Qwen3-30B-A3B-Instruct-2507 via vLLM, answer styles, refusal retry + closed-book fallback |
| Evaluation | `src/rag/evaluation.py`: SQuAD-normalized EM/F1, answer recall, ROUGE-1/2/L |
