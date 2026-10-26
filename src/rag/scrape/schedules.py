"""College athletics schedule pages -> one event document per game.

As page text, a schedule loses the link between a date, an opponent and the team ("Dec 9",
"vs", "Eastern Michigan" end up on separate lines, often in different chunks). Three page
formats are handled: Sidearm "nextgen" game cards, classic Sidearm list items, PrestoSports
event rows.
"""
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .events import SEASON_START, event_doc

SCHOOLS = {
    "pittsburghpanthers.com": "Pitt Panthers (University of Pittsburgh)",
    "goduquesne.com": "Duquesne Dukes",
    "rmucolonials.com": "Robert Morris Colonials",
    "gochathamcougars.com": "Chatham Cougars (Chatham University)",
    "pointparksports.com": "Point Park Pioneers (Point Park University)",
    "athletics.cmu.edu": "Carnegie Mellon Tartans (CMU)",
    "athletics.carlow.edu": "Carlow Celtics (Carlow University)",
}
MONTHS = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


def _txt(el):
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el else ""


def _season(title, url):
    m = re.search(r"\b(20\d\d)(?:-(\d\d))?\b", f"{title} {url}")
    return (int(m.group(1)), int("20" + m.group(2)) if m.group(2) else None) if m else (None, None)


def _iso(date_s, time_s, season):
    """'Dec 9 (Wed)', '6:00 p.m.' + season (2026, 2027) -> '2026-12-09T18:00:00'."""
    m = re.search(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})", date_s or "")
    if not m or m.group(1).lower() not in MONTHS:
        return None
    month, day = MONTHS[m.group(1).lower()], int(m.group(2))
    first, second = season
    if first is None:
        return None
    year = second if (second and month < 7) else first
    iso = f"{year}-{month:02d}-{day:02d}"
    t = re.search(r"(\d{1,2})(?::(\d\d))?\s*([ap])\.?\s*m", (time_s or "").lower())
    if t:
        hour = int(t.group(1)) % 12 + (12 if t.group(3) == "p" else 0)
        iso += f"T{hour:02d}:{int(t.group(2) or 0):02d}:00"
    return iso


def _team(soup, url):
    host = urlparse(url).netloc.removeprefix("www.")
    title = _txt(soup.title) if soup.title else ""
    sport = re.search(r"(?:20\d\d(?:-\d\d)?\s+)?(?:Tartan\s+)?(.+?)\s+Schedule", title)
    sport = sport.group(1).strip() if sport else ""
    return SCHOOLS.get(host, host), sport, title


def _nextgen(soup):
    for c in soup.find_all(attrs={"data-test-id": "s-game-card-standard__root"}):
        f = lambda k: _txt(c.find(attrs={"data-test-id": k}))
        yield {"date": f("s-game-card-standard__header-game-date"),
               "time": f("s-game-card-standard__header-game-time"),
               "opponent": f("s-game-card-standard__header-team-opponent-link"),
               "where": "", "location": f("s-game-card-facility-and-location__details-standard"),
               "tv": f("s-game-card-standard__header-tv-and-radio"), "result": ""}


def _classic(soup):
    for li in soup.select("li.sidearm-schedule-game"):
        f = lambda cls: _txt(li.select_one("." + cls))
        cls = " ".join(li.get("class", []))
        where = "home" if "home-game" in cls else "away" if "away-game" in cls else \
            "neutral" if "neutral" in cls else ""
        yield {"date": f("sidearm-schedule-game-opponent-date"), "time": _txt(li),
               "opponent": f("sidearm-schedule-game-opponent-name"), "where": where,
               "location": f("sidearm-schedule-game-location"), "tv": "",
               "result": f("sidearm-schedule-game-result")}


def _presto_rows(soup):
    """Older theme: one .event-row per game; newer theme: a .card-body per game."""
    rows = soup.select(".event-row")
    if rows:
        return rows
    out = []
    for name in soup.select(".event-opponent-name"):
        card = next((p for p in name.parents if "card-body" in (p.get("class") or [])), None)
        if card is not None and card not in out:
            out.append(card)
    return out


def _presto(soup):
    for row in _presto_rows(soup):
        cls = row.get("class", [])
        badge = _txt(row.select_one(".event-location-badge")).lower()
        where = next((w for w in ("home", "away", "neutral") if w in cls), "") or \
            {"vs": "home", "at": "away"}.get(badge, "")
        info = [_txt(x) for x in row.select(".event-dateinfo li")] or [_txt(row.select_one(".event-dateinfo"))] * 2
        loc = next((_txt(x) for x in row.find_all(class_=re.compile(r"location|venue|neutralsite"))
                    if "badge" not in " ".join(x.get("class", [])) and _txt(x)), "")
        if "neutralsite" in " ".join(" ".join(x.get("class", [])) for x in row.find_all(class_=True)) and loc:
            where = "neutral"
        yield {"date": info[0] if info else "", "time": info[1] if len(info) > 1 else "",
               "opponent": _txt(row.select_one(".event-opponent-name")), "where": where,
               "location": loc.lstrip("@ ").strip(), "tv": "",
               "result": _txt(row.select_one(".event-result, .result"))}


def parse_schedule(html, url, source="college_athletics"):
    """-> list of event docs (games of the 2026-27 season, from SEASON_START); [] if the page is not a schedule."""
    soup = BeautifulSoup(html, "lxml")
    if soup.find(attrs={"data-test-id": "s-game-card-standard__root"}):
        games = _nextgen(soup)
    elif soup.select_one("li.sidearm-schedule-game"):
        games = _classic(soup)
    elif soup.select_one(".event-row"):
        games = _presto(soup)
    else:
        return []
    school, sport, page_title = _team(soup, url)
    season = _season(page_title, url)
    team = f"{school} {sport}".strip()
    docs = []
    for g in games:
        if not g["opponent"]:
            continue
        # classic Sidearm puts the time inside the date cell
        date_s, time_s = g["date"], g["time"] or g["date"]
        start = _iso(date_s, time_s, season)
        if not start or start[:10] < SEASON_START:
            continue
        sep = {"home": "vs.", "away": "at", "neutral": "vs."}.get(g["where"], "vs.")
        title = f"{team} {sep} {g['opponent']}"
        desc = "; ".join(x for x in (
            {"home": "Home game", "away": "Away game", "neutral": "Neutral site"}.get(g["where"], ""),
            g["tv"], f"Result: {g['result']}" if g["result"] else "") if x)
        d = event_doc(source, title=title, url=url, start=start, venue=g["location"],
                      category=f"college {sport.lower()} game".strip(), organizer=school,
                      description=desc)
        d["topic"] = "sports"
        docs.append(d)
    return docs
