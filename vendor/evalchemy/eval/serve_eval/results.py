# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Read Evalchemy's aggregate results from its FineStore archive."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval.contracts.finestore_output import DEFAULT_SOURCE_PREFIX, read_finestore_output
from eval.contracts.task_outcome import TaskOutcome, validate_result_document


class EvalResults(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    task_outcomes: Dict[str, TaskOutcome] = Field(default_factory=dict)
    task_preparations: Dict[str, Any] = Field(default_factory=dict)
    n_samples: Dict[str, Any] = Field(default_factory=dict, alias="n-samples")
    lm_eval_version: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None
    model_source: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def require_shared_result_contract(cls, value: Any) -> Any:
        """Reject legacy or malformed result dictionaries at every read path."""
        if isinstance(value, Mapping):
            validate_result_document(value)
        return value

    @classmethod
    def load_archive(cls, root: str, source_prefix: str = DEFAULT_SOURCE_PREFIX) -> "EvalResults":
        """Load one task group's committed aggregate from FineStore."""
        document = read_finestore_output(root, source_prefix)
        if document is None:
            raise FileNotFoundError(f"no Evalchemy result for {source_prefix!r} in FineStore archive {root!r}")
        return cls.model_validate(document)

    def metric(self, task: str, name: str) -> Optional[float]:
        value = (self.results.get(task) or {}).get(name)
        return float(value) if isinstance(value, (int, float)) else None

    def numeric_metrics(self, task: str) -> Dict[str, float]:
        task_results = self.results.get(task) or {}
        return {k: float(v) for k, v in task_results.items() if isinstance(v, (int, float))}

    def sample_count(self, task: str) -> Optional[int]:
        """Return completed coverage for a requested task or a native group leaf."""
        outcome = self.task_outcomes.get(task)
        if outcome is not None and outcome.scored_count is not None:
            return outcome.scored_count
        counts = self.n_samples.get(task, {})
        return counts.get("effective") if isinstance(counts, Mapping) else counts
