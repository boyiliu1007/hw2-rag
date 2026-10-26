"""Structured datasets/APIs -> one short document per record.

  cmoa     Carnegie Museum of Art collection records (CC0 open-data release) -> one doc per artwork
  millage  Allegheny County Treasurer municipal + school-district real estate millage tables
  sports   Pro/college team schedules (games from Oct 2026 on) and current-roster player bios
           (NHL, MLB StatsAPI, ESPN); also writes player names for wikipedia.py to resolve

Writes data/processed/sources/<source>.jsonl and caches raw responses under data/raw/<source>/.

    python -m src.rag.scrape.datasets --only cmoa millage sports
"""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from tqdm import tqdm

from .crawl import ROOT, load_sources
from .events import SEASON_START, event_doc
from .http import Fetcher

OUT = ROOT / "data" / "processed" / "sources"
RAW = ROOT / "data" / "raw"
ET = ZoneInfo("America/New_York")


def _id(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _cached_get(fetcher, source, name, url, **kw):
    """Fetch url once; later runs reuse data/raw/<source>/<name>."""
    p = RAW / source / name
    if p.exists():
        return p.read_text()
    r = fetcher.get(url, **kw)
    if r is None or r.status_code != 200:
        print(f"  ! {url}: {r.status_code if r is not None else 'error'}")
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.text)
    return r.text


def _year(d):
    return (d or "")[:4] if re.match(r"^\d{4}", d or "") and not (d or "").startswith("0001") else ""


# ---------------------------------------------------------------- Carnegie Museum of Art
CMOA_FIELDS = [  # (label, csv column)
    ("Date", "creation_date"), ("Medium", "medium"), ("Classification", "classification"),
    ("Department", "department"), ("Accession number", "accession_number"),
    ("Credit line", "credit_line"), ("Date acquired", "date_acquired"),
    ("Location (2017 record)", "physical_location"),
]


def _creator(r):
    name = r.get("full_name") or r.get("cited_name") or ""
    if not name:
        return ""
    bits = [b for b in (r.get("nationality"), "-".join(filter(None, (_year(r.get("birth_date")),
                                                                    _year(r.get("death_date")))))) if b]
    role = r.get("role") or ""
    s = name + (f" ({', '.join(bits)})" if bits else "")
    return s + (f" [{role}]" if role and role.lower() not in ("artist", "") else "")


def acquisition_method(credit_line):
    """Gift / Bequest / Purchase from a credit line ("Bequest of Roy A. Hunt" -> Bequest)."""
    c = credit_line.lower()
    if "bequest" in c:
        return "Bequest"
    if "gift" in c or "donated" in c or "generosity" in c:
        return "Gift"
    if any(w in c for w in ("purchase", "fund", "acquired through", "museum acquisition")):
        return "Purchase"
    return ""


def cmoa(src, fetcher):
    """The museum's 2017 open-data release (CC0); the official repo was removed, so the
    unmodified fork is used. One row per (object, creator): group rows by object id."""
    text = _cached_get(fetcher, src["name"], "cmoa.csv", src["url"])
    if text is None:
        return []
    objects = OrderedDict()
    for r in csv.DictReader(io.StringIO(text)):
        objects.setdefault(r["id"], []).append(r)
    docs = []
    for oid, rows in objects.items():
        r = rows[0]
        title = (r["title"] or "Untitled").strip()
        creators = list(dict.fromkeys(c for c in map(_creator, rows) if c))
        names = list(dict.fromkeys(x["full_name"] for x in rows if x.get("full_name")))
        kind = (r.get("classification") or "work").rstrip("s") or "work"
        lead = f"“{title}” is a {kind} in the collection of the Carnegie Museum of Art (CMOA) in Pittsburgh"
        lead += (f", by {' and '.join(names)}" if names else "") + "."
        lines = [f"Title: {title}"]
        if creators:
            lines.append(f"Artist: {'; '.join(creators)}")
        for label, col in CMOA_FIELDS:
            v = (r.get(col) or "").strip()
            if v:
                lines.append(f"{label}: {v}")
        method = acquisition_method(r.get("credit_line") or "")
        if method:
            lines.append(f"Acquisition method: {method}")
        prov = re.sub(r"\s+", " ", r.get("provenance_text") or "").strip()
        if prov:
            lines.append(f"Provenance: {prov[:800]}")
        docs.append({"doc_id": f"cmoa-{_id(oid)}", "url": r.get("web_url", ""),
                     "title": f"{title} (Carnegie Museum of Art)", "topic": src["topic"],
                     "source": src["name"], "text": lead + "\n" + "\n".join(lines)})
    return docs


# ---------------------------------------------------------------- Allegheny County millage
MILLAGE_NOTES = {  # footnote digits appended to municipality names on the county page
    "1": "land millage applies (only in the Cities of Clairton and McKeesport)",
    "2": "municipal and school taxes are not based on current county assessed values",
    "3": "includes fire tax", "4": "includes road reconstruction tax", "5": "includes library tax",
}


def _table_rows(html):
    soup = BeautifulSoup(html, "lxml")
    table = max(soup.find_all("table"), key=lambda t: len(t.find_all("tr")))
    rows = []
    for tr in table.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all("td", recursive=False)]
        if cells and not cells[0].startswith(("Municipality", "School District")):
            rows.append(cells)
    return rows


def _mills(x):
    """'4.1300' -> '4.13 mills (4.1300)': the county table pads to 4 decimals."""
    short = f"{float(x):g}"
    return f"{short} mills" + (f" ({x})" if short != x else "")


def millage(src, fetcher):
    docs = []
    for year in src["years"]:
        muni_url = f"https://apps.alleghenycounty.us/website/MillMuni.asp?Year={year}"
        sd_url = f"https://apps.alleghenycounty.us/website/millsd.asp?Year={year}"
        muni_html = _cached_get(fetcher, src["name"], f"muni_{year}.html", muni_url)
        sd_html = _cached_get(fetcher, src["name"], f"sd_{year}.html", sd_url)
        if not muni_html or not sd_html:
            continue
        school_of = {}
        for cells in _table_rows(sd_html):
            if len(cells) < 3 or not re.match(r"^[\d.]+$", cells[2]):
                continue
            sd = cells[0].replace("º", "").strip()
            land = f" plus {cells[3]} mills on land" if len(cells) > 3 and re.match(r"^[\d.]+$", cells[3]) else ""
            members = [m.strip() for m in cells[1].split(",") if m.strip()]
            for m in members:
                school_of[m] = (sd, cells[2])
            text = (f"The {sd} School District's {year} real estate tax rate is {_mills(cells[2])}{land} "
                    f"(Allegheny County Treasurer, School District Tax Millages {year}).\n"
                    f"School district: {sd}\nYear: {year}\nMillage: {cells[2]} mills{land}\n"
                    f"Municipalities served: {', '.join(members)}")
            docs.append({"doc_id": f"millage-sd-{_id(sd, year)}", "url": sd_url,
                         "title": f"{sd} School District real estate tax millage {year}",
                         "topic": src["topic"], "source": src["name"], "text": text})
        county_rate = None
        for cells in _table_rows(muni_html):
            if len(cells) < 4 or not re.match(r"^[\d.]+$", cells[3]):
                continue
            m = re.match(r"^(.*?)\s*(\d)?$", cells[0])
            name, note = m.group(1).strip(), m.group(2)
            if name == "Allegheny County":
                county_rate = cells[3]
            land = cells[4] if len(cells) > 4 and re.match(r"^[\d.]+$", cells[4]) else ""
            lines = [f"{name}'s {year} municipal real estate tax rate is {_mills(cells[3])}"
                     + (f" (plus {land} mills on land)" if land else "")
                     + f" (Allegheny County Treasurer, Municipality Tax Millages {year}).",
                     f"Municipality: {name}", f"Year: {year}", f"Municipal millage: {cells[3]} mills"]
            if land:
                lines.append(f"Land millage: {land} mills")
            if note in MILLAGE_NOTES:
                lines.append(f"Note: {MILLAGE_NOTES[note]}")
            if name in school_of:
                lines.append(f"School district: {school_of[name][0]} ({school_of[name][1]} mills)")
            if county_rate and name != "Allegheny County":
                lines.append(f"Allegheny County millage: {county_rate} mills")
            if cells[1]:
                lines.append(f"Tax collector: {cells[1]}" + (f", phone {cells[2]}" if cells[2] else ""))
            docs.append({"doc_id": f"millage-muni-{_id(name, year)}", "url": muni_url,
                         "title": f"{name} real estate tax millage {year}", "topic": src["topic"],
                         "source": src["name"], "text": "\n".join(lines)})
    return docs


# ---------------------------------------------------------------- sports
def _local_iso(utc):
    """'2026-10-24T23:00Z' -> local Eastern ISO string so event_doc prints local time."""
    dt = datetime.fromisoformat(utc.replace("Z", "+00:00"))
    return dt.astimezone(ET).isoformat()


def _game_doc(source, team, opp, home, start_utc, venue, city, league, url, extra=""):
    title = f"{opp} at {team}" if home else f"{team} at {opp}"
    start = _local_iso(start_utc)
    doc = event_doc(source, title=title, url=url, start=start, venue=venue, address=city,
                    category=f"{league} game", description=extra)
    verb = "host" if home else "play at"
    date, = re.findall(r"^Date: (.*)$", doc["text"], re.M) or [""]
    doc["text"] = (f"The {team} {verb} the {opp} on {date} ET"
                   + (f" at {venue}" if venue else "") + (f" in {city}" if city else "") + ".\n"
                   + doc["text"].split("\n", 1)[1])
    doc["topic"] = "sports"
    return doc


def nhl(src, fetcher, players):
    docs = []
    raw = _cached_get(fetcher, src["name"], "nhl_schedule.json",
                      f"https://api-web.nhle.com/v1/club-schedule-season/PIT/{src['nhl_season']}")
    for g in json.loads(raw or '{"games": []}')["games"]:
        if g["gameDate"] < SEASON_START:
            continue
        home = g["homeTeam"]["abbrev"] == "PIT"
        opp_t = g["awayTeam"] if home else g["homeTeam"]
        opp = " ".join(filter(None, (opp_t.get("placeName", {}).get("default"),
                                     opp_t.get("commonName", {}).get("default"))))
        kind = {1: "NHL preseason", 2: "NHL regular season", 3: "NHL playoff"}.get(g.get("gameType"), "NHL")
        docs.append(_game_doc(src["name"], "Pittsburgh Penguins", opp, home, g["startTimeUTC"],
                              g.get("venue", {}).get("default", ""), "", kind,
                              "https://www.nhl.com/penguins/schedule"))
    raw = _cached_get(fetcher, src["name"], "nhl_roster.json", "https://api-web.nhle.com/v1/roster/PIT/current")
    roster = json.loads(raw or "{}")
    pos = {"C": "center", "L": "left wing", "R": "right wing", "D": "defenseman", "G": "goaltender"}
    for p in tqdm([p for k in ("forwards", "defensemen", "goalies") for p in roster.get(k, [])], desc="nhl players"):
        raw = _cached_get(fetcher, src["name"], f"nhl_player_{p['id']}.json",
                          f"https://api-web.nhle.com/v1/player/{p['id']}/landing")
        d = json.loads(raw or "{}")
        name = f"{p['firstName']['default']} {p['lastName']['default']}"
        born = ", ".join(filter(None, (d.get("birthCity", {}).get("default"),
                                       (d.get("birthStateProvince") or {}).get("default"), d.get("birthCountry"))))
        dr = d.get("draftDetails") or {}
        lines = [f"{name} is a {pos.get(p['positionCode'], 'player')} for the Pittsburgh Penguins (NHL), "
                 f"wearing number {p.get('sweaterNumber', '')}.",
                 f"Player: {name}", "Team: Pittsburgh Penguins",
                 f"Position: {pos.get(p['positionCode'], p['positionCode'])}",
                 f"Born: {d.get('birthDate', p.get('birthDate', ''))} in {born}",
                 f"Height: {p.get('heightInInches', '')} in; Weight: {p.get('weightInPounds', '')} lb",
                 f"Shoots/catches: {p.get('shootsCatches', '')}"]
        if dr:
            lines.append(f"Drafted: {dr.get('year')} round {dr.get('round')}, pick {dr.get('overallPick')} "
                         f"overall by {dr.get('teamAbbrev')}")
        docs.append({"doc_id": f"sports-nhl-{p['id']}", "url": f"https://www.nhl.com/player/{p['id']}",
                     "title": f"{name} (Pittsburgh Penguins)", "topic": "sports", "source": src["name"],
                     "text": "\n".join(lines)})
        players.append((name, "ice hockey"))
    return docs


def mlb(src, fetcher, players):
    docs = []
    raw = _cached_get(fetcher, src["name"], "mlb_roster.json",
                      "https://statsapi.mlb.com/api/v1/teams/134/roster?rosterType=40Man&hydrate=person")
    for r in json.loads(raw or '{"roster": []}')["roster"]:
        p = r["person"]
        born = ", ".join(filter(None, (p.get("birthCity"), p.get("birthStateProvince"), p.get("birthCountry"))))
        posn = p.get("primaryPosition", {}).get("name", "player").lower()
        lines = [f"{p['fullName']} is a {posn} for the Pittsburgh Pirates (MLB)"
                 + (f", wearing number {p['primaryNumber']}" if p.get("primaryNumber") else "") + ".",
                 f"Player: {p['fullName']}", "Team: Pittsburgh Pirates", f"Position: {posn}",
                 f"Born: {p.get('birthDate', '')} in {born}"]
        if p.get("mlbDebutDate"):
            lines.append(f"MLB debut: {p['mlbDebutDate']}")
        lines.append(f"Bats/throws: {p.get('batSide', {}).get('description', '')}/"
                     f"{p.get('pitchHand', {}).get('description', '')}")
        lines.append(f"Height: {p.get('height', '')}; Weight: {p.get('weight', '')} lb")
        if r.get("status", {}).get("description"):
            lines.append(f"Roster status: {r['status']['description']}")
        docs.append({"doc_id": f"sports-mlb-{p['id']}", "url": f"https://www.mlb.com/player/{p['id']}",
                     "title": f"{p['fullName']} (Pittsburgh Pirates)", "topic": "sports",
                     "source": src["name"], "text": "\n".join(lines)})
        players.append((p["fullName"], "baseball"))
    return docs


ESPN = "https://site.api.espn.com/apis/site/v2/sports"


def espn_schedules(src, fetcher):
    docs = []
    for t in src["espn_teams"]:
        for season in t["seasons"]:
            raw = _cached_get(fetcher, src["name"], f"espn_{t['sport'].replace('/', '_')}_{t['id']}_{season}.json",
                              f"{ESPN}/{t['sport']}/teams/{t['id']}/schedule?season={season}&seasontype=2")
            for e in json.loads(raw or '{"events": []}').get("events", []):
                if e["date"][:10] < SEASON_START:
                    continue
                comp = e["competitions"][0]
                me = next(c for c in comp["competitors"] if c["team"]["id"] == str(t["id"]))
                opp = next(c for c in comp["competitors"] if c["team"]["id"] != str(t["id"]))
                v = comp.get("venue") or {}
                addr = v.get("address") or {}
                city = ", ".join(filter(None, (addr.get("city"), addr.get("state"))))
                notes = "; ".join(n.get("headline", "") for n in comp.get("notes", []) if n.get("headline"))
                docs.append(_game_doc(src["name"], t["name"], opp["team"]["displayName"],
                                      me.get("homeAway") == "home", e["date"], v.get("fullName", ""), city,
                                      t["league"], (e.get("links") or [{}])[0].get("href", ""), notes))
    return docs


def espn_roster(src, fetcher, players):
    docs = []
    for t in src.get("espn_rosters", []):
        raw = _cached_get(fetcher, src["name"], f"espn_roster_{t['sport'].replace('/', '_')}_{t['id']}.json",
                          f"{ESPN}/{t['sport']}/teams/{t['id']}/roster")
        groups = json.loads(raw or '{"athletes": []}').get("athletes", [])
        for a in (a for g in groups for a in g.get("items", [])):
            bp = a.get("birthPlace") or {}
            born = ", ".join(filter(None, (bp.get("city"), bp.get("state"), bp.get("country"))))
            posn = (a.get("position") or {}).get("displayName", "player")
            lines = [f"{a['fullName']} is a {posn.lower()} for the {t['name']}"
                     + (f", wearing number {a['jersey']}" if a.get("jersey") else "") + ".",
                     f"Player: {a['fullName']}", f"Team: {t['name']}", f"Position: {posn}"]
            if a.get("dateOfBirth"):
                lines.append(f"Born: {a['dateOfBirth'][:10]}" + (f" in {born}" if born else ""))
            if (a.get("college") or {}).get("name"):
                lines.append(f"College: {a['college']['name']}")
            if (a.get("experience") or {}).get("years") is not None:
                lines.append(f"Experience: {a['experience']['years']} years")
            lines.append(f"Height: {a.get('displayHeight', '')}; Weight: {a.get('displayWeight', '')}")
            docs.append({"doc_id": f"sports-espn-{a['id']}", "url": "", "title": f"{a['fullName']} ({t['name']})",
                         "topic": "sports", "source": src["name"], "text": "\n".join(lines)})
            players.append((a["fullName"], t["wiki_hint"]))
    return docs


def sports(src, fetcher):
    players = []
    docs = nhl(src, fetcher, players) + mlb(src, fetcher, players)
    docs += espn_schedules(src, fetcher) + espn_roster(src, fetcher, players)
    p = RAW / src["name"] / "players.json"  # read by wikipedia.py to add player articles
    p.write_text(json.dumps(sorted(set(players))))
    return docs


# ---------------------------------------------------------------- OpenStreetMap places
OVERPASS = "https://overpass-api.de/api/interpreter"


def _osm_kind(t):
    """Human-readable category from OSM tags, e.g. 'Thai restaurant', 'bakery', 'museum'."""
    for key in ("amenity", "shop", "tourism", "leisure"):
        if t.get(key):
            kind = t[key].replace("_", " ")
            if key == "shop" and kind not in ("yes",):
                kind = f"{kind} shop" if kind in ("gift", "music", "coffee", "wine", "tea", "pet", "bicycle",
                                                   "frame", "craft", "variety store") else kind
            if key == "shop" and kind == "yes":
                kind = "shop"
            if t.get("cuisine") and key == "amenity":
                kind = f"{t['cuisine'].replace(';', ', ').replace('_', ' ')} {kind}"
            return kind
    return "place"


def osm(src, fetcher):
    """Named shops, restaurants, cafes, bars, venues, museums, hotels and parks in Allegheny County
    from OpenStreetMap (ODbL): one document per place."""
    query = (f"[out:json][timeout:300];area[\"name\"=\"{src['area']}\"][\"admin_level\"=\"{src['admin_level']}\"]->.a;("
             + "".join(f'nwr["{k}"~"{v}"]["name"](area.a);' for k, v in src["filters"].items())
             + ");out tags center;")
    p = RAW / src["name"] / "overpass.json"
    if not p.exists():
        r = fetcher.s.post(OVERPASS, data={"data": query}, timeout=360,
                           headers={"User-Agent": "CMU-11711-HW2-RAG/0.1 (https://github.com/boyiliu1007/hw2-rag)"})
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(r.text)
    docs, seen = [], set()
    for e in json.loads(p.read_text())["elements"]:
        t = e.get("tags", {})
        name = t.get("name", "").strip()
        if not name:
            continue
        addr = " ".join(filter(None, (t.get("addr:housenumber"), t.get("addr:street"))))
        place = ", ".join(filter(None, (t.get("addr:suburb") or t.get("addr:neighbourhood"),
                                        t.get("addr:city"), "PA" if t.get("addr:city") else "",
                                        t.get("addr:postcode"))))
        where = ", ".join(filter(None, (addr, place)))
        key = (name.lower(), where.lower())
        if key in seen:
            continue
        seen.add(key)
        kind = _osm_kind(t)
        article = "an" if kind[:1].lower() in "aeiou" else "a"
        lead = f"{name} is {article} {kind}" + (f" at {where}" if where else " in Allegheny County, Pennsylvania") + "."
        fields = [("Name", name), ("Type", kind), ("Address", where), ("Brand", t.get("brand")),
                  ("Opening hours", t.get("opening_hours")), ("Phone", t.get("phone") or t.get("contact:phone")),
                  ("Website", t.get("website") or t.get("contact:website")),
                  ("Description", t.get("description")), ("Operator", t.get("operator")),
                  ("Wheelchair accessible", t.get("wheelchair")),
                  ("Outdoor seating", t.get("outdoor_seating")), ("Takeaway", t.get("takeaway"))]
        text = lead + "\n" + "\n".join(f"{k}: {v}" for k, v in fields if v)
        docs.append({"doc_id": f"osm-{e['type']}{e['id']}", "url": f"https://www.openstreetmap.org/{e['type']}/{e['id']}",
                     "title": f"{name} ({kind})", "topic": src["topic"], "source": src["name"], "text": text})
    return docs


PARSERS = {"cmoa": cmoa, "millage": millage, "sports": sports, "osm": osm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for src in load_sources():
        if src["type"] != "dataset" or (args.only and src["name"] not in args.only):
            continue
        print(f"== {src['name']}")
        docs = PARSERS[src["parser"]](src, Fetcher(delay=args.delay))
        with (OUT / f"{src['name']}.jsonl").open("w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  wrote {len(docs)} docs")


if __name__ == "__main__":
    main()
