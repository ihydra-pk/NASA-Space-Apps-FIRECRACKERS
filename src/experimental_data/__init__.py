"""Structured ingestion and condition matching for available NASA data."""

from .matching import MatchResult, match_experiments
from .models import EvidenceStatus, ExperimentRecord
from .service import ExperimentalDataService, IngestionReport

__all__ = [
    "EvidenceStatus",
    "ExperimentRecord",
    "ExperimentalDataService",
    "IngestionReport",
    "MatchResult",
    "match_experiments",
]
