"""Shared evaluation helpers: answer normalization, retrieval hit test, question types."""
import re
import string

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = set(string.punctuation) | set("“”‘’–—…")


def normalize(s):
    """SQuAD normalization: lowercase, drop punctuation and articles, squeeze whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in PUNCT)  # deleted, as in the SQuAD script: 3,500 -> 3500
    return " ".join(ARTICLES.sub(" ", s).split())


def is_short(answers, max_words=5):
    return min(len(a.split()) for a in answers) <= max_words


def answer_in(text_norm, answers):
    """True if any (normalized) reference answer occurs in an already-normalized text."""
    return any((a := normalize(x)) and f" {a} " in f" {text_norm} " for x in answers)


QTYPES = [  # first match wins
    ("museum_collection", r"Carnegie Museum of Art (list|record|catalog|painting)|accession|credit line"),
    ("event_dated", r"\b(20(26|27))\b|\b(January|February|March|April|May|June|July|August|September|"
                    r"October|November|December) \d{1,2}\b"),
    ("sports", r"Steelers|Penguins|Pirates|Panthers|Colonials|Dukes|Tartans|Cougars|basketball|football|"
               r"hockey|baseball|soccer|coach|season|NFL|NHL|MLB|NCAA|stadium|player"),
    ("tax_budget", r"\btax|budget|millage|mills\b|revenue|payroll|LST\b"),
    ("cmu", r"Carnegie Mellon|CMU|Tartan|Scotty"),
    ("food", r"restaurant|food|festival|cuisine|bakery|coffee|pizza|chocolate|dining|market"),
]


def qtype(question):
    for name, pat in QTYPES:
        if re.search(pat, question, re.I):
            return name
    return "general"


# ------------------------------------------------------------------ answer metrics
def _tokens(s):
    return normalize(s).split()


def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


def f1(pred, ref):
    """SQuAD token-level F1 (Rajpurkar et al., 2016, section 6.1)."""
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return float(p == r)
    common = sum(min(p.count(t), r.count(t)) for t in set(p) & set(r))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def recall(pred, ref):
    """Answer recall: fraction of reference tokens that appear in the prediction."""
    p, r = _tokens(pred), _tokens(ref)
    if not r:
        return float(not p)
    return sum(min(p.count(t), r.count(t)) for t in set(r)) / len(r)


def rouge_l(pred, ref):
    """ROUGE-L F-measure from the longest common subsequence of normalized tokens."""
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return float(p == r)
    prev = [0] * (len(r) + 1)
    for x in p:
        cur = [0]
        for j, y in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def _ngram_f(pred, ref, n):
    p, r = _tokens(pred), _tokens(ref)
    pg = [tuple(p[i:i + n]) for i in range(len(p) - n + 1)]
    rg = [tuple(r[i:i + n]) for i in range(len(r) - n + 1)]
    if not pg or not rg:
        return float(p == r) if n > 1 and len(p) < n and len(r) < n else 0.0
    common = sum(min(pg.count(g), rg.count(g)) for g in set(pg) & set(rg))
    if common == 0:
        return 0.0
    prec, rec = common / len(pg), common / len(rg)
    return 2 * prec * rec / (prec + rec)


def rouge_1(pred, ref):
    return _ngram_f(pred, ref, 1)


def rouge_2(pred, ref):
    """Bigram ROUGE F; when both answers are single words it counts as an exact-match check."""
    return _ngram_f(pred, ref, 2)


METRICS = {"em": exact_match, "f1": f1, "recall": recall, "rouge1": rouge_1, "rouge2": rouge_2, "rougeL": rouge_l}


def score(pred, refs):
    """Each metric is the max over the reference answers; rouge = mean of ROUGE-1/2/L, as the
    course leaderboard reports it."""
    s = {m: max(fn(pred, r) for r in refs) for m, fn in METRICS.items()}
    s["rouge"] = (s["rouge1"] + s["rouge2"] + s["rougeL"]) / 3
    # leaderboard Total = mean(F1, Recall, ROUGE, judge rescaled to 0-1); we have no judge locally
    s["total3"] = (s["f1"] + s["recall"] + s["rouge"]) / 3
    return s
