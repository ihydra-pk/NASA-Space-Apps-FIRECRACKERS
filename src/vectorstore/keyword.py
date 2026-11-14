"""BM25 keyword scoring over stored chunk text, with no external dependencies.

Embeddings match meaning but blur exact identifiers. Keyword scoring is the
complement: test numbers, acronyms and formulae such as ``SF6`` or ``C7H16``
match literally.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")

# Question filler. Left in, these make every chunk a "keyword match" and bury
# the terms that identify what is being asked about.
STOPWORDS = frozenset(
    """a about above after all also an and any are as at be been being between both but by can
    compared could did do does during each for from had has have how if in into is it its many
    may more most much not of on or other over per same should so some such than that the their
    them then there these they this those through to under up use used using was were what when
    where which while who why will with would""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric runs: ``"PSI-69 (SF6)"`` -> ``psi, 69, sf6``."""

    return _TOKEN.findall(text.casefold())


def bm25_scores(
    query: str,
    texts: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Score every text against the query; 0.0 means no query term occurs."""

    query_terms = set(tokenize(query)) - STOPWORDS
    if not query_terms or not texts:
        return [0.0] * len(texts)

    term_counts: list[Counter[str]] = []
    lengths: list[int] = []
    document_frequency: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(text)
        counts = Counter(token for token in tokens if token in query_terms)
        term_counts.append(counts)
        lengths.append(len(tokens))
        document_frequency.update(counts.keys())

    total = len(texts)
    average_length = sum(lengths) / total or 1.0
    # The "+1" form keeps IDF positive, so a very common term adds a little
    # instead of subtracting.
    idf = {
        term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }

    scores: list[float] = []
    for counts, length in zip(term_counts, lengths):
        normalizer = k1 * (1.0 - b + b * length / average_length)
        scores.append(
            sum(
                idf[term] * count * (k1 + 1.0) / (count + normalizer)
                for term, count in counts.items()
            )
        )
    return scores
