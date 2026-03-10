"""Shared term normalization helpers for cognition routing and retrieval."""

from __future__ import annotations

import re
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "do",
    "for",
    "from",
    "help",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "our",
    "please",
    "that",
    "the",
    "this",
    "to",
    "use",
    "we",
    "with",
    "you",
    "your",
}

ALIASES = {
    "alternative": "creative_explore",
    "alternatives": "creative_explore",
    "brainstorm": "creative_explore",
    "brainstorming": "creative_explore",
    "check": "verify",
    "checked": "verify",
    "checking": "verify",
    "earlier": "memory_reuse",
    "idea": "creative_explore",
    "ideas": "creative_explore",
    "previous": "memory_reuse",
    "prior": "memory_reuse",
    "recall": "memory_reuse",
    "remember": "memory_reuse",
    "summaries": "summarization",
    "summarise": "summarization",
    "summarised": "summarization",
    "summarising": "summarization",
    "summarize": "summarization",
    "summarized": "summarization",
    "summarizing": "summarization",
    "summary": "summarization",
    "validate": "verify",
    "validated": "verify",
    "validation": "verify",
    "verifying": "verify",
}


def normalize_terms(text: str) -> list[str]:
    """Return normalized non-stopword terms from free-form text."""
    return list(iter_normalized_terms(text))


def iter_normalized_terms(text: str) -> Iterable[str]:
    """Yield normalized non-stopword terms from free-form text."""
    for raw_token in TOKEN_PATTERN.findall(text.lower()):
        token = ALIASES.get(raw_token, raw_token)
        if token in STOPWORDS:
            continue
        yield token


def unique_terms(text: str) -> list[str]:
    """Return normalized terms with duplicates removed while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for term in iter_normalized_terms(text):
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered


def term_set(*parts: object) -> set[str]:
    """Return normalized unique terms gathered from multiple values."""
    terms: set[str] = set()
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            terms.update(iter_normalized_terms(part))
            continue
        if isinstance(part, dict):
            for value in part.values():
                if isinstance(value, str):
                    terms.update(iter_normalized_terms(value))
            continue
        if isinstance(part, (list, tuple, set)):
            for value in part:
                if isinstance(value, str):
                    terms.update(iter_normalized_terms(value))
            continue
        terms.update(iter_normalized_terms(str(part)))
    return terms


def overlap_score(query_terms: list[str], candidate_terms: set[str]) -> float:
    """Return normalized overlap score in [0, 1]."""
    if not query_terms or not candidate_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in candidate_terms)
    return hits / len(query_terms)
