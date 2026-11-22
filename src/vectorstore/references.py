"""Recognise reference lists so that search can rank them below findings.

A bibliography entry matches a question on meaning and on keywords alike,
because paper titles read like questions ("Cool-flame extinction during
n-alkane droplet combustion in microgravity"), yet it says nothing about what
the paper found. Search ranks such chunks after all others; it never removes
them and never changes their stored labels or evidence status.
"""

from __future__ import annotations

import re

# Each pattern is one citation mark: an author ("Nayagam, V." or "S.H. Won"),
# a year ("(2015)" or ", 2015."), or a volume/page reference.
_CITATION_MARKS = (
    re.compile(r"\b[A-Z][A-Za-z'’\-]+,\s+(?:[A-Z]\.\s?-?){1,3}"),
    re.compile(r"\b(?:[A-Z]\.\s?-?){1,3}\s?[A-Z][a-z]{2,}"),
    re.compile(r"\((?:19|20)\d{2}[a-z]?\)|,\s(?:19|20)\d{2}[a-z]?[.,;)]"),
    re.compile(r"\b\d+\s?\(\d+\),?\s?\d+\s?[-–]\s?\d+|\bpp?\.\s?\d+|\bVol\.\s?\d+", re.IGNORECASE),
)
# Citation marks per 100 words at which a chunk counts as a reference list.
# On the indexed corpus, reference and publication lists score 12-50 while
# prose with in-text citations scores under 4. Pages that mix findings with
# the start of a reference list score 6-11 and are deliberately not flagged.
REFERENCE_LIST_DENSITY = 12.0

_ASKS_FOR_PUBLICATIONS = re.compile(
    r"\b(?:papers?|publications?|published|references?|bibliograph\w*|cit(?:e|es|ed|ing|ations?)|"
    r"journals?|articles?|authors?)\b",
    re.IGNORECASE,
)


def citation_density(text: str) -> float:
    words = len(text.split())
    if not words:
        return 0.0
    marks = sum(len(pattern.findall(text)) for pattern in _CITATION_MARKS)
    return marks * 100 / words


def looks_like_reference_list(text: str) -> bool:
    return citation_density(text) >= REFERENCE_LIST_DENSITY


def asks_for_publications(question: str) -> bool:
    """True when reference lists are what the question is about."""

    return bool(_ASKS_FOR_PUBLICATIONS.search(question))
