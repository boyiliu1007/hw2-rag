"""PDF -> plain text with pdfplumber (pypdf fallback), stripping running headers/footers."""
import io
import re
from collections import Counter

import pdfplumber
from pypdf import PdfReader

from .html_clean import normalize_ws


def _strip_running_lines(pages, min_frac=0.5):
    """Drop first/last lines that repeat on >= min_frac of pages (headers, footers, page numbers)."""
    if len(pages) < 4:
        return pages
    edge = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        for ln in set(lines[:2] + lines[-2:]):
            edge[re.sub(r"\d+", "#", ln)] += 1
    common = {k for k, v in edge.items() if v >= min_frac * len(pages)}
    out = []
    for p in pages:
        keep = [ln for ln in p.split("\n") if re.sub(r"\d+", "#", ln.strip()) not in common]
        out.append("\n".join(keep))
    return out


def _table_rows(table):
    rows = []
    for row in table:
        cells = [re.sub(r"\s+", " ", c).strip() for c in row if c and c.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def pdf_to_text(data: bytes, max_pages=None):
    """Return (title, text). Pages separated by blank lines."""
    pages, title = [], ""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            title = (pdf.metadata or {}).get("Title", "") or ""
            for i, page in enumerate(pdf.pages):
                if max_pages and i >= max_pages:
                    break
                txt = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
                # Budget docs are table-heavy; extract_text flattens tables poorly,
                # so append row-wise table text when tables are detected.
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                if tables:
                    txt += "\n\n" + "\n\n".join(_table_rows(t) for t in tables)
                pages.append(txt)
    except Exception:
        pages = []
    if sum(len(p) for p in pages) < 200:  # scanned or pdfplumber failure -> pypdf
        try:
            reader = PdfReader(io.BytesIO(data))
            title = title or (reader.metadata.title if reader.metadata else "") or ""
            pages = [(pg.extract_text() or "") for pg in reader.pages[:max_pages]]
        except Exception:
            pages = []
    pages = _strip_running_lines(pages)
    # de-hyphenate words broken across lines
    text = "\n\n".join(re.sub(r"(\w)-\n(\w)", r"\1\2", p) for p in pages)
    text = re.sub(r"\s*(?:\.\s?){4,}\s*\d*", " ", text)  # table-of-contents dot leaders
    return title.strip(), normalize_ws(text)
