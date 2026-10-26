"""Structured event feeds -> one short document per event.

Event calendars rendered as generic page text lose the title/date/venue
association that event questions depend on ("Who performs at X on Y?"), so each
event becomes its own small document with explicit fields.

Writes data/processed/sources/<source>.jsonl (doc records) and caches API
responses under data/raw/<source>/.

    python -m src.rag.scrape.events --only pittsburgh_events
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

from .crawl import ROOT, load_sources
from .html_clean import extract_jsonld_events, normalize_ws
from .http import Fetcher

OUT = ROOT / "data" / "processed" / "sources"
RAW = ROOT / "data" / "raw"
MIN_DATE = "2026-10-01"  # handout: event questions are about Oct 2026 onward
# Sports schedules keep the whole 2026-27 season: questions about season totals, openers or
# records ("how many games are on the 2026 schedule") need the games before October too.
SEASON_START = "2026-07-01"


def strip_html(s):
    if not s:
        return ""
    return normalize_ws(BeautifulSoup(str(s), "lxml").get_text(" "))


def fmt_dt(iso):
    """'2026-10-13T19:30:00-04:00' -> ('Tuesday, October 13, 2026', '7:30 PM')."""
    if not iso:
        return "", ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso, ""
    date = dt.strftime("%A, %B %-d, %Y")
    has_time = "T" in iso and not iso.split("T")[1].startswith("00:00")
    return date, (dt.strftime("%-I:%M %p") if has_time else "")


def event_doc(source, *, title, url, start, end=None, venue="", address="", category="",
              organizer="", performers="", price="", recurrence="", description=""):
    date, time = fmt_dt(start)
    end_date, end_time = fmt_dt(end) if end else ("", "")
    lines = [f"Event: {title}"]
    when = date + (f" at {time}" if time else "")
    if end_date and end_date != date:
        when += f" to {end_date}" + (f" at {end_time}" if end_time else "")
    elif end_time and end_time != time:
        when += f" to {end_time}"
    if when:
        lines.append(f"Date: {when}")
    if recurrence:
        lines.append(f"Recurs: {recurrence}")
    if venue or address:
        lines.append("Venue: " + ", ".join(x for x in (venue, address) if x))
    for k, v in (("Performers", performers), ("Category", category), ("Organizer", organizer), ("Price", price)):
        if v:
            lines.append(f"{k}: {v}")
    desc = strip_html(description)
    if desc:
        lines.append(f"Description: {desc[:1500]}")
    # A natural-language lead sentence helps both BM25 and dense retrieval
    if end_date and end_date != date:
        lead = f"{title} runs from {date} to {end_date}"
    else:
        lead = f"{title} takes place on {date}" + (f" at {time}" if time else "")
    lead += (f" at {venue}" if venue else "") + "."
    text = lead + "\n" + "\n".join(lines)
    key = f"{source}|{title}|{start}|{venue}"
    return {
        "doc_id": f"{source}-{hashlib.sha1(key.encode()).hexdigest()[:12]}",
        "url": url, "title": f"{title} ({date})", "topic": "events", "source": source,
        "text": text, "meta": {"start": start, "venue": venue},
    }


def _cache(source, name, payload):
    p = RAW / source / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))


# ---------------------------------------------------------------- pittsburgh.events
def _ld_place(loc):
    if isinstance(loc, list):
        loc = loc[0] if loc else {}
    if not isinstance(loc, dict):
        return str(loc or ""), ""
    addr = loc.get("address", "")
    if isinstance(addr, dict):
        addr = ", ".join(x for x in (addr.get("streetAddress"), addr.get("addressLocality"),
                                     addr.get("addressRegion"), addr.get("postalCode")) if x)
    return loc.get("name", ""), addr


def _ld_names(x):
    if isinstance(x, list):
        return ", ".join(filter(None, (_ld_names(i) for i in x)))
    if isinstance(x, dict):
        return x.get("name", "")
    return str(x or "")


def pittsburgh_events(src, fetcher):
    """Every page in the sitemap embeds schema.org Event JSON-LD; month pages only show
    ~24 events, but performer/venue/category pages together cover the full calendar."""
    idx = fetcher.get("https://pittsburgh.events/sitemap/index.xml")
    sitemaps = re.findall(r"<loc>([^<]+)</loc>", idx.text)
    urls = []
    for sm in sitemaps:
        r = fetcher.get(sm)
        urls += re.findall(r"<loc>([^<]+)</loc>", r.text) if r is not None else []
    urls = list(dict.fromkeys(urls))
    print(f"  {len(urls)} sitemap URLs")
    events = {}
    for u in tqdm(urls, desc="pittsburgh.events"):
        r = fetcher.get(u)
        if r is None or r.status_code != 200:
            continue
        for e in extract_jsonld_events(BeautifulSoup(r.text, "lxml")):
            eid = e.get("@id") or e.get("url") or f"{e.get('name')}|{e.get('startDate')}"
            if eid not in events:
                events[eid] = e
    _cache(src["name"], "events_jsonld.json", list(events.values()))
    docs = []
    for e in events.values():
        start = e.get("startDate", "")
        if start[:10] < MIN_DATE:
            continue
        venue, addr = _ld_place(e.get("location"))
        t = e.get("@type", "Event")
        cat = re.sub(r"(?<!^)(?=[A-Z])", " ", t if isinstance(t, str) else t[0]).replace(" Event", "")
        page = e.get("url") or (e.get("mainEntityOfPage") or {}).get("url", "")
        docs.append(event_doc(src["name"], title=e.get("name", ""), url=page, start=start,
                              end=e.get("endDate"), venue=venue, address=addr, category=cat,
                              performers=_ld_names(e.get("performer")),
                              organizer=_ld_names(e.get("organizer")),
                              description=e.get("description", "")))
    return docs


# ---------------------------------------------------------------- downtownpittsburgh.com
def downtown_pittsburgh(src, fetcher):
    base = "https://downtownpittsburgh.com/wp-json/vibemap/v1/events-data"
    page, raw = 1, []
    while True:
        r = fetcher.get(f"{base}?page={page}")
        if r is None or r.status_code != 200:
            break
        d = r.json()
        raw += d["events"]
        if not d["pagination"].get("has_more"):
            break
        page += 1
    _cache(src["name"], "events_data.json", raw)
    docs, seen = [], set()
    for e in raw:
        m = e.get("meta") or {}
        start = m.get("vibemap_event_start_date", "")
        if start[:10] < MIN_DATE:
            continue
        key = (e.get("title"), start)
        if key in seen:
            continue
        seen.add(key)
        docs.append(event_doc(
            src["name"], title=strip_html(e.get("title", "")), url=e.get("permalink", ""),
            start=start, end=m.get("vibemap_event_end_date"),
            venue=m.get("vibemap_event_venue_name") or m.get("vibemap_event_hotspots_place", ""),
            address=m.get("vibemap_event_address", ""),
            category=m.get("vibemap_event_categories", ""),
            organizer=m.get("vibemap_event_organizer", ""),
            price=m.get("vibemap_event_price", ""),
            recurrence=m.get("vibemap_event_recurrence_display", ""),
            description=e.get("content") or m.get("vibemap_event_description", "")))
    return docs


# ---------------------------------------------------------------- events.cmu.edu
def cmu_livewhale(src, fetcher):
    fields = "location,summary,description,group_title,tags,event_types,cost,custom_admission_fee"
    url = (f"https://events.cmu.edu/live/json/events/start_date/{src['start']}"
           f"/end_date/{src['end']}/response_fields/{fields}")
    raw = []
    while url:
        r = fetcher.get(url)
        if r is None or r.status_code != 200:
            break
        d = r.json()
        raw += d.get("data", [])
        url = d.get("links", {}).get("next")
    _cache(src["name"], "events.json", raw)
    docs = []
    for e in raw:
        types = [strip_html(t) for t in (e.get("event_types") or []) if "Exclude" not in t]
        docs.append(event_doc(
            src["name"], title=strip_html(e.get("title", "")), url=e.get("url", ""),
            start=e.get("date_iso", ""), end=e.get("date2_iso"),
            venue=strip_html(e.get("location") or e.get("location_title") or ""),
            category=", ".join(types), organizer=e.get("group_title") or "",
            price=strip_html(e.get("cost") or e.get("custom_admission_fee") or ""),
            description=e.get("description") or e.get("summary") or ""))
    return docs


# ---------------------------------------------------------------- calendar.pitt.edu
def pitt_localist(src, fetcher, max_instances=10):
    """University of Pittsburgh calendar (Localist API v2); one doc per occurrence, capped so a
    daily recurring event does not flood the corpus."""
    raw, page = [], 1
    while page:
        r = fetcher.get("https://calendar.pitt.edu/api/2/events",
                        params={"start": src["start"], "end": src["end"], "pp": 100, "page": page})
        if r is None or r.status_code != 200:
            break
        d = r.json()
        raw += [x["event"] for x in d.get("events", [])]
        page = d.get("page", {}).get("next_page")
    _cache(src["name"], "events.json", raw)
    docs = []
    for e in raw:
        f = e.get("filters") or {}
        types = ", ".join(t["name"] for t in f.get("event_types", []))
        venue = ", ".join(x for x in (e.get("location_name"), e.get("room_number")) if x)
        if e.get("experience") == "virtual" and not venue:
            venue = "Online"
        org = ", ".join(x["name"] for x in e.get("departments", []))
        insts = [i["event_instance"] for i in e.get("event_instances", [])
                 if i["event_instance"]["start"][:10] >= MIN_DATE][:max_instances]
        for inst in insts:
            start = inst["start"][:10] if inst.get("all_day") else inst["start"]
            docs.append(event_doc(
                src["name"], title=strip_html(e.get("title", "")), url=e.get("localist_url", ""),
                start=start, end=inst.get("end"), venue=venue, address=e.get("address") or "",
                category=types, organizer=org, price=e.get("ticket_cost") or "",
                description=e.get("description_text") or ""))
    return docs


PARSERS = {"pittsburgh_events": pittsburgh_events,
           "downtown_pittsburgh": downtown_pittsburgh,
           "cmu_livewhale": cmu_livewhale,
           "pitt_localist": pitt_localist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for src in load_sources():
        if src["type"] != "events" or (args.only and src["name"] not in args.only):
            continue
        print(f"== {src['name']}")
        docs = PARSERS[src["parser"]](src, Fetcher(delay=args.delay))
        with (OUT / f"{src['name']}.jsonl").open("w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  wrote {len(docs)} event docs")


if __name__ == "__main__":
    main()
