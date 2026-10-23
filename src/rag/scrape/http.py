"""Polite HTTP fetching: per-domain rate limit, robots.txt, retries, Wayback fallback."""
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
)
WAYBACK = "https://web.archive.org/web/2026id_/"  # id_ = original bytes, no toolbar
BOT_UA = "CMU-11711-HW2-RAG/0.1 (student coursework; python-requests)"
# Wayback throttles hard (~15 req/min) and each fetch is a redirect + the page
DOMAIN_DELAY = {"web.archive.org": 6.0}
# These answer HTTP 429 to a browser-style header set; they expect an honest bot UA
BOT_DOMAINS = {"web.archive.org", "en.wikipedia.org"}
BOT_HEADERS = {"User-Agent": BOT_UA, "Accept": "*/*", "Accept-Language": None}


class Fetcher:
    def __init__(self, delay=1.0, timeout=30, retries=2, respect_robots=True):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.delay, self.timeout, self.retries = delay, timeout, retries
        self.respect_robots = respect_robots
        self._last = {}    # domain -> last request time
        self._robots = {}  # domain -> RobotFileParser | None

    def _wait(self, domain):
        delay = max(self.delay, DOMAIN_DELAY.get(domain, 0))
        dt = time.time() - self._last.get(domain, 0)
        if dt < delay:
            time.sleep(delay - dt)
        self._last[domain] = time.time()

    def allowed(self, url):
        if not self.respect_robots:
            return True
        p = urlparse(url)
        if p.netloc not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.s.get(f"{p.scheme}://{p.netloc}/robots.txt", timeout=self.timeout)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])
            self._robots[p.netloc] = rp
        return self._robots[p.netloc].can_fetch(UA, url)

    def get(self, url, wayback=False, **kw):
        """GET url; with wayback=True fetch the archived copy instead.

        Returns a requests.Response or None. Caller checks status_code.
        """
        target = WAYBACK + url if wayback else url
        domain = urlparse(target).netloc
        for attempt in range(self.retries + 1):
            self._wait(domain)
            try:
                if domain in BOT_DOMAINS:
                    kw.setdefault("headers", BOT_HEADERS)  # None values drop session headers
                r = self.s.get(target, timeout=self.timeout, **kw)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    ra = r.headers.get("retry-after", "")
                    time.sleep(min(int(ra), 120) if ra.isdigit() else 2 ** (attempt + 2))
                    continue
                return r
            except requests.RequestException:
                if attempt < self.retries:
                    time.sleep(2 ** (attempt + 1))
        return None

    def get_with_fallback(self, url, mode="auto"):
        """mode: 'live' | 'wayback' | 'auto' (live, then Wayback on 403/404/410/error).

        Returns (response, used_wayback).
        """
        if mode == "wayback":
            return self.get(url, wayback=True), True
        r = self.get(url) if self.allowed(url) else None
        if mode == "auto" and (r is None or r.status_code in (403, 404, 410, 451)):
            wb = self.get(url, wayback=True)
            if wb is not None and wb.status_code == 200:
                return wb, True
        return r, False
