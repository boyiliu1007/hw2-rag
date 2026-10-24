#!/usr/bin/env bash
# Collect the full knowledge resource. Each step is resumable (raw files are cached),
# so re-running only fetches what is missing or previously failed transiently.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY -m src.rag.scrape.datasets                        # ~2 min, museum CSV, millage, sports APIs
$PY -m src.rag.scrape.wikipedia                       # ~45 min, MediaWiki API (needs datasets' player list)
$PY -m src.rag.scrape.events                          # ~20 min, pittsburgh.events / Downtown / CMU / Pitt feeds
# HTML/PDF crawl; groups hit disjoint domains so they run in parallel
$PY -m src.rag.scrape.crawl --only britannica pittsburghpa_gov city_tax_regulations city_operating_budget citypaper_events visitpittsburgh &
$PY -m src.rag.scrape.crawl --only cmu_about cmu_site cmu_campus_events pittsburgh_symphony pittsburgh_opera cultural_trust &
$PY -m src.rag.scrape.crawl --only carnegie_museums heinz_history_center the_frick food_festivals pirates steelers penguins &
$PY -m src.rag.scrape.crawl --only county_budget college_athletics venues_culture nextpittsburgh_events &
wait
$PY scripts/build_corpus.py
