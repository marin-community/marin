# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted contracts and review evidence for retained Atlas populations."""

from typing import Literal

from finestore.rl.curation import SourceQualityResult
from marin.execution.artifact import Artifact
from pydantic import BaseModel, Field

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation


class RetentionPlan(BaseModel):
    source_id: str
    source_dataset_id: str
    source_revision: str
    converter_revision: str
    converter_version: str
    input_artifact: str
    task_specs_sha256: str
    raw_count: int = Field(ge=0)
    combined_drop_count: int = Field(ge=0)
    retained_count: int = Field(ge=0)
    combined_drop_manifest_uri: str
    lexical_use: str


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
