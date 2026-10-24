"""Serializable evidence shared by policy regrading and launch finalization."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrialVerdict:
    task: str
    doc_id: str
    trial_id: str
    filter: str
    score: float
    audit: dict[str, Any]
    extracted: str
