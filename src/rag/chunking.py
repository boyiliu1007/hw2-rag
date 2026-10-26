"""Document chunking: fixed-size with overlap, sentence-aware, and structure (heading/paragraph)
aware. Sizes are in whitespace words (~1.3 subword tokens each).

Every chunk starts with a context header ("<doc title> > <section heading>") so a chunk cut
from the middle of a long page still says what it is about, for BM25 and embedders alike.

Records that are already one fact-sheet each (events, artworks, millage rows, player bios,
games) are never split: splitting would separate e.g. a date from the event it belongs to.

    python -m src.rag.chunking --strategy structure     # -> indexes/chunks/structure.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "indexes" / "chunks"
WHOLE_SOURCES = {"cmoa", "millage", "sports", "pittsburgh_events", "downtown_events",
                 "cmu_events", "pitt_events"}
STRATEGIES = ("fixed", "sentence", "structure")

ABBREV = {"mr", "mrs", "ms", "dr", "st", "ave", "blvd", "rd", "no", "jr", "sr", "inc", "co",
          "corp", "ltd", "vs", "etc", "mt", "ft", "gen", "gov", "sen", "rep", "prof", "rev",
          "dept", "univ", "approx", "est", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
          "sep", "sept", "oct", "nov", "dec", "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "pa"}
SENT_END = re.compile(r"(?<=[.!?])[\"”')\]]*\s+(?=[\"“(\[]?[A-Z0-9])")


def n_words(s):
    return len(s.split())


# ------------------------------------------------------------------ splitting primitives
def split_sentences(text):
    """Split prose into sentences without breaking on abbreviations ("St.", "U.S.") or
    initials ("H. J. Heinz")."""
    out, start = [], 0
    for m in SENT_END.finditer(text):
        prev = re.search(r"(\S+)$", text[start:m.start()])
        tok = prev.group(1).rstrip(".\"”')]").lower() if prev else ""
        if tok in ABBREV or re.fullmatch(r"[a-z]", tok) or re.fullmatch(r"(?:[a-z]\.)+[a-z]", tok):
            continue
        out.append(text[start:m.start()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def _is_tabular(line):
    """Table rows ("Header: value | Header: value"), list items and short label lines stay
    one unit each instead of being merged into a sentence."""
    return (" | " in line or line.lstrip().startswith(("- ", "* ", "• "))
            or (n_words(line) <= 12 and not line.rstrip().endswith((".", ",", ";"))))


def paragraph_units(par):
    """A paragraph -> sentence-like units. Wrapped prose lines (PDF/HTML hard breaks) are
    re-joined before sentence splitting; tabular lines are kept as they are."""
    units, prose = [], []

    def flush():
        if prose:
            units.extend(split_sentences(" ".join(prose)))
            prose.clear()

    for line in par.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _is_tabular(line):
            flush()
            units.append(line)
        else:
            prose.append(line)
    flush()
    return units


def split_long(unit, max_words, overlap):
    """Last resort for a single unit longer than max_words: fixed word windows."""
    w = unit.split()
    step = max(1, max_words - overlap)
    return [" ".join(w[i:i + max_words]) for i in range(0, max(1, len(w) - overlap), step)]


def pack(units, max_words, overlap_words):
    """Greedily pack units into chunks of <= max_words; carry trailing units totalling about
    overlap_words into the next chunk so facts at a boundary appear in both."""
    chunks, cur, size = [], [], 0
    for u in units:
        if n_words(u) > max_words:
            if cur:
                chunks.append(cur)
            chunks.extend([[p] for p in split_long(u, max_words, overlap_words)])
            cur, size = [], 0
            continue
        if cur and size + n_words(u) > max_words:
            chunks.append(cur)
            carry, c = [], 0
            for x in reversed(cur):
                if c + n_words(x) > overlap_words:
                    break
                carry.insert(0, x)
                c += n_words(x)
            cur, size = carry, c
        cur.append(u)
        size += n_words(u)
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------ strategies
def chunk_fixed(text, max_words, overlap):
    return [(None, " ".join(c)) for c in pack(text.split(), max_words, overlap)]


def chunk_sentence(text, max_words, overlap):
    units = [u for par in re.split(r"\n\s*\n", text) for u in paragraph_units(par)]
    return [(None, " ".join(c)) for c in pack(units, max_words, overlap)]


def sections(text):
    """Split on markdown-style headings ("# ...", "## ...") kept by the cleaners; returns
    [(heading path, body)]."""
    out, path, buf = [], [], []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            if "".join(buf).strip():
                out.append((" > ".join(path), "\n".join(buf)))
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2)]
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((" > ".join(path), "\n".join(buf)))
    return out


def chunk_structure(text, max_words, overlap):
    """Sections never share a chunk; within a section, whole paragraphs are packed together
    and only an over-long paragraph is broken at sentence boundaries."""
    out = []
    for heading, body in sections(text):
        units = []
        for par in re.split(r"\n\s*\n", body):
            if not par.strip():
                continue
            if n_words(par) <= max_words:
                units.append(re.sub(r"[ \t]*\n[ \t]*", "\n", par.strip()))
            else:
                units.extend(paragraph_units(par))
        for c in pack(units, max_words, overlap):
            out.append((heading, "\n".join(c)))
    return out


CHUNKERS = {"fixed": chunk_fixed, "sentence": chunk_sentence, "structure": chunk_structure}


def chunk_doc(doc, strategy, max_words=180, overlap=30, min_words=8):
    """-> list of chunk records. Short docs and per-record sources stay whole."""
    title = (doc.get("title") or "").strip()
    text = doc["text"].strip()
    base = {k: doc.get(k) for k in ("doc_id", "url", "title", "topic", "source")}
    if doc["source"] in WHOLE_SOURCES or n_words(text) <= max_words * 1.2:
        pieces = [(None, text)]
    else:
        pieces = CHUNKERS[strategy](text, max_words, overlap)
    out = []
    for heading, body in pieces:
        body = body.strip()
        if n_words(body) < min_words and len(pieces) > 1:
            continue  # stray heading/label left at a boundary
        if heading and title and heading.split(" > ")[0].strip().lower() == title.lower():
            heading = heading.split(" > ", 1)[1] if " > " in heading else None
        header = " > ".join(x for x in (title, heading) if x)
        first = body.split("\n", 1)[0].strip().lstrip("# ")
        if header and first != header and first != title:
            body = f"{header}\n{body}"
        out.append({**base, "chunk_id": f"{doc['doc_id']}#{len(out)}", "text": body})
    return out


def build(strategy, corpus=ROOT / "data/processed/docs.jsonl", max_words=180, overlap=30):
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out = CHUNK_DIR / f"{strategy}.jsonl"
    n_docs = n_chunks = words = 0
    with out.open("w") as f:
        for line in tqdm(corpus.open(), desc=f"chunk[{strategy}]"):
            doc = json.loads(line)
            n_docs += 1
            for c in chunk_doc(doc, strategy, max_words, overlap):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_chunks += 1
                words += n_words(c["text"])
    print(f"{strategy}: {n_docs} docs -> {n_chunks} chunks, {words / n_chunks:.0f} words/chunk -> {out}")
    return out


def load_chunks(strategy):
    return [json.loads(line) for line in (CHUNK_DIR / f"{strategy}.jsonl").open()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES + ("all",), default="all")
    ap.add_argument("--max-words", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=30)
    args = ap.parse_args()
    for s in (STRATEGIES if args.strategy == "all" else (args.strategy,)):
        build(s, max_words=args.max_words, overlap=args.overlap)


if __name__ == "__main__":
    main()
