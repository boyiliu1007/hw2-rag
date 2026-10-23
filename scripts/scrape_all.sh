#!/usr/bin/env bash
# Collect the full knowledge resource. Each step is resumable (raw files are cached),
# so re-running only fetches what is missing or previously failed transiently.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY -m src.rag.scrape.wikipedia                       # ~2 min, MediaWiki API
$PY -m src.rag.scrape.events                          # ~20 min, pittsburgh.events / Downtown / CMU feeds
# HTML/PDF crawl; groups hit disjoint domains so they run in parallel
$PY -m src.rag.scrape.crawl --only britannica pittsburghpa_gov city_tax_regulations city_operating_budget citypaper_events visitpittsburgh &
$PY -m src.rag.scrape.crawl --only cmu_about cmu_campus_events pittsburgh_symphony pittsburgh_opera cultural_trust &
$PY -m src.rag.scrape.crawl --only carnegie_museums heinz_history_center the_frick food_festivals pirates steelers penguins &
wait
$PY scripts/build_corpus.py
