# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted contracts and review evidence for retained Atlas populations."""

from typing import Literal

from finestore.rl.curation import SourceQualityResult
from marin.execution.artifact import Artifact

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.retention_plan import RetentionPlan


class FullQCInputs(Artifact):
    source_id: str
    population: SourceTextPopulation
    retention: RetentionPlan
    drop_manifest_sha256: str
    index_path: str
    index_sha256: str
    retained_count: int
    task_bytes: int
    oracle_tasks: int
    missing_execution_requirements: int
    grading_environments: dict[str, int]


class FullQCReview(Artifact):
    source_id: str
    inputs: FullQCInputs
    reviewer: dict
    rubric: dict
    coverage_complete: bool
    qualification_status: Literal["pending"]
    attempted_count: int
    result: SourceQualityResult
    top_issues: dict[str, int]
    records_path: str
    records_sha256: str
    batch_receipts: list[str]
