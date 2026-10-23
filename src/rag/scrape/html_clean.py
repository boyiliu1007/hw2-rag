"""HTML -> clean plain text that preserves paragraph / heading / table structure.

Output uses blank lines between blocks so paragraph-based chunking downstream
can split on them. Headings are prefixed with '#' marks; tables become
"Header: value | Header: value" lines so each row survives as one unit.
"""
import json
import re

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

DROP_TAGS = ["script", "style", "noscript", "template", "svg", "iframe",
             "button", "select", "input", "dialog"]
# Usually boilerplate, but some CMSs (ASP.NET) wrap the entire page in <form>, so these
# are only removed when they hold a small share of the page text.
SOFT_DROP_TAGS = ["nav", "header", "footer", "aside", "form"]
# class/id words for boilerplate containers, matched as whole -/_ delimited parts of a
# token ("cookie-banner", "site_footer") so CamelCase content classes ("ContentSidebar") survive
NOISE = re.compile(
    r"(?:^|[-_])(cookies?|consent|gdpr|newsletter|subscribe|signup|social|share|sharing|"
    r"breadcrumbs?|skip|sidebar|popup|modal|advert|ads|promo|related|comments|"
    r"translate|language|back-to-top|menu|navbar|footer|header)(?:$|[-_])", re.I)
BLOCK = {"p", "div", "section", "article", "main", "li", "dd", "dt", "blockquote",
         "pre", "figcaption", "address", "tr", "br", "hr", "ul", "ol", "dl", "table"}
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _is_noise(tag):
    if not isinstance(tag, Tag) or tag.attrs is None:
        return False
    tokens = list(tag.get("class", []) or []) + [tag.get("id") or ""]
    role = tag.get("role", "")
    return any(NOISE.search(t.lower()) for t in tokens if t) or \
        role in ("navigation", "banner", "contentinfo", "search")


def _table_to_lines(table):
    rows = table.find_all("tr")
    if not rows:
        return ""
    headers = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
    has_header = bool(rows[0].find("th"))
    out = []
    for tr in rows[1:] if has_header else rows:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not any(cells):
            continue
        if has_header and len(headers) == len(cells):
            out.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, cells) if v))
        else:
            out.append(" | ".join(c for c in cells if c))
    return "\n".join(out)


PARA_BLOCK = {"p", "section", "article", "blockquote", "ul", "ol", "dl"}


def _walk(root, out):
    """Iterative pre-order walk (government CMS pages nest deeper than Python's recursion limit).

    Stack items are Tags to expand or ("T", text) tokens to emit.
    """
    stack = [root]
    while stack:
        item = stack.pop()
        if isinstance(item, tuple):
            out.append(item[1])
            continue
        pushed = []
        for child in item.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                pushed.append(("T", str(child)))
            elif not isinstance(child, Tag):
                continue
            elif child.name in HEADINGS:
                text = child.get_text(" ", strip=True)
                if text:
                    pushed.append(("T", f"\n\n{'#' * HEADINGS[child.name]} {text}\n\n"))
            elif child.name == "table":
                pushed.append(("T", "\n\n" + _table_to_lines(child) + "\n\n"))
            elif child.name == "li":
                pushed += [("T", "\n- "), child, ("T", "\n")]
            elif child.name in BLOCK:
                sep = "\n\n" if child.name in PARA_BLOCK else "\n"
                pushed += [("T", sep), child, ("T", sep)]
            else:
                pushed.append(child)
        stack.extend(reversed(pushed))


MOJIBAKE = re.compile(r"(?:[\u00c2-\u00f4][\u0080-\u00bf\u0152-\u2122]{1,3})+")


def fix_mojibake(text):
    """Repair UTF-8 text that was decoded as cp1252 upstream (e.g. 'Pittsburgh\u00e2\u20ac\u2122s')."""
    def repl(m):
        try:
            return m.group(0).encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group(0)
    return MOJIBAKE.sub(repl, text)


def normalize_ws(text):
    text = fix_mojibake(text)
    text = text.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop lines that are just bullets / punctuation
    text = re.sub(r"^\s*[-•|·]\s*$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def dedupe_lines(text, min_len=25):
    """Remove repeated long lines (menus/CTAs repeated within a page)."""
    seen, out = set(), []
    for ln in text.split("\n"):
        key = ln.strip().lower()
        if len(key) >= min_len and key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return "\n".join(out)


def extract_jsonld_events(soup):
    """Return schema.org Event objects embedded as JSON-LD (many event sites have them)."""
    events = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            d = stack.pop()
            if isinstance(d, list):
                stack.extend(d)
            elif isinstance(d, dict):
                t = d.get("@type", "")
                t = " ".join(t) if isinstance(t, list) else t
                if "Event" in t:
                    events.append(d)
                stack.extend(v for k, v in d.items() if k == "@graph" or isinstance(v, list))
    return events


def html_to_text(html, url=""):
    """Return (title, text)."""
    soup = BeautifulSoup(html, "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()

    for t in soup(DROP_TAGS):
        t.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    # Remove boilerplate containers, but never one holding a large share of the page:
    # broad class names ("has-sidebar", "modal-open") sometimes sit on the main wrapper.
    total = len(soup.get_text(" ", strip=True)) or 1
    h1 = soup.find("h1")
    for t in soup.find_all(lambda t: t.name in SOFT_DROP_TAGS or _is_noise(t)):
        if t.decomposed or t.name in ("body", "html", "main"):
            continue
        small = len(t.get_text(" ", strip=True)) < 0.3 * total
        if t.name in ("nav", "header", "footer", "aside") or t.get("role") == "navigation":
            # semantic chrome: drop unless it wraps the page title (some sites put <main> inside <header>)
            holds_title = h1 is not None and (t is h1 or h1 in t.descendants)
            if small or not holds_title:
                t.decompose()
        elif small:  # <form> wrappers and class-name matches: only drop if minor
            t.decompose()

    root = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.body or soup
    # if <main> is suspiciously small, fall back to body
    if root is not soup.body and soup.body and len(root.get_text(strip=True)) < 200:
        root = soup.body
    out = []
    _walk(root, out)
    text = dedupe_lines(normalize_ws("".join(out)))
    return title, text
