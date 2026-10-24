"""Answer-free coverage probe: does each question's named entity appear in the corpus at all?

Entities are found with simple rules (quoted titles; runs of capitalized words, allowing
connectors like "of", "and", "&"), so it works on questions without reference answers
(leaderboard, test set). A question whose entities are all absent cannot be answered from
the corpus, which points at missing sources.

    python scripts/entity_check.py data/eval/leaderboard_queries.json [--show all|missing]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GENERIC = {
    "pittsburgh", "pennsylvania", "pa", "western pa", "western pennsylvania", "cmu",
    "carnegie mellon", "carnegie mellon university", "allegheny county", "united states", "us",
    "what", "which", "who", "when", "where", "why", "how", "in", "on", "the", "a", "an", "is",
    "does", "did", "do", "are", "was", "were", "can", "for", "at", "of", "and", "if", "as",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "i", "it", "its", "this", "that", "there", "according",
}
CONNECTORS = r"(?:of|and|the|de|du|la|le|&|-)"
CAP = r"(?:[A-Z][\w.'’&-]*|\d+[\w-]*)"
# words that end an entity span: dates and times are not part of a name
BREAK = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
         "october", "november", "december", "monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday", "fall", "spring", "summer", "winter", "am", "pm"}
SPAN = re.compile(rf"{CAP}(?:\s+(?:{CONNECTORS}\s+)*{CAP})*")
QUOTED = re.compile(r"[\"“”']([^\"“”']{3,})[\"“”']")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def entities(q):
    ents = [m.group(1) for m in QUOTED.finditer(q)]
    rest = QUOTED.sub(" ", q)
    for m in SPAN.finditer(rest):
        words = [re.sub(r"(['’]s?|[.,:;])$", "", w) for w in m.group(0).split()]
        for i, w in enumerate(words):
            if norm(w) in BREAK or re.fullmatch(r"\d[\d:]*", w):
                words = words[:i]
                break
        while words and norm(words[0]) in GENERIC:   # "What", "In", "The" at span start
            words = words[1:]
        while words and re.fullmatch(CONNECTORS, words[-1]):
            words = words[:-1]
        e = " ".join(words)
        if e and norm(e) not in GENERIC and not re.fullmatch(r"[\d-]+", e):
            ents.append(e)
    return list(dict.fromkeys(ents))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", type=Path)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/processed/docs.jsonl")
    ap.add_argument("--show", choices=["missing", "all", "none"], default="missing")
    args = ap.parse_args()

    corpus = " " + " \n ".join(norm(json.loads(l)["text"] + " " + (json.loads(l).get("title") or ""))
                               for l in args.corpus.open()) + " "
    qs = json.load(args.questions.open())
    qs = qs if isinstance(qs, list) else [{"id": k, "question": v} for k, v in qs.items()]

    status = Counter()
    for q in qs:
        ents = entities(q["question"])
        found = {e: f" {norm(e)} " in corpus for e in ents}
        if not ents:
            s = "no entity"
        elif all(found.values()):
            s = "all found"
        elif any(found.values()):
            s = "some missing"
        else:
            s = "none found"
        status[s] += 1
        if args.show == "all" or (args.show == "missing" and s in ("some missing", "none found")):
            marks = ", ".join(("+" if ok else "-") + e for e, ok in found.items())
            print(f"[{s:12}] {q['id']:>4} {q['question']}\n{'':20}{marks}")
    print(f"\n{len(qs)} questions: " + ", ".join(f"{k} {v}" for k, v in status.most_common()))


if __name__ == "__main__":
    main()
