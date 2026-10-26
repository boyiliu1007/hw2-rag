"""Retrievers share one interface:

    r.search(queries: list[str], k: int) -> list[list[(chunk_index, score)]]

chunk_index is the row in indexes/chunks/<strategy>.jsonl; higher score = more relevant.
"""
