# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable scored curriculum releases consumed by Atlas."""

from enum import StrEnum
from typing import Literal

from finestore.rl.curation import QualifiedSourceDeclaration
from pydantic import BaseModel, ConfigDict, Field

LEGACY_COVERAGE_VERSION = "2026.10.15.1"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbeddingSpec(_Record):
    """Numerical and text-rendering identity shared by tasks and curriculum anchors."""

    model: str
    revision: str
    tokenizer_revision: str
    dimensions: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    compute_dtype: Literal["float32", "float16", "bfloat16"]
    pooling: Literal["last"] = "last"
    normalization: Literal["l2"] = "l2"
    storage_dtype: Literal["float32"] = "float32"


class AlignmentSource(_Record):
    """A matched source or an inventory-only declaration in a curriculum release.

    Declared sources have no placement statistics or intended use.
    """

    source_id: str
    source_key: str
    population: Literal["converted", "admitted"]
    intended_use: str | None
    task_count: int
    ready_count: int
    insufficient_context_count: int
    basis: Literal["matched", "declared"]
    curation_root: str | None = None
    input_rows: int | None = None
    converted_rows: int | None = None
    rejections: dict[str, int] = Field(default_factory=dict)


class CurriculumNode(_Record):
    node_id: str
    subject_id: str
    parent_id: str | None
    kind: Literal["subject", "group", "capability"]
    name: str
    description: str


class AlignmentRelease(_Record):
    release_id: str
    created_at: str
    curriculum_key: str
    population: Literal["converted", "admitted"]
    sample_size: int | None
    task_count: int
    ready_count: int
    insufficient_context_count: int
    sources: list[AlignmentSource]
    inventory: list[QualifiedSourceDeclaration]
    nodes: list[CurriculumNode]
    paths: list[str]
    lineage: dict[str, str]
    embedding_spec: EmbeddingSpec


class CoverageMetric(StrEnum):
    DENOMINATOR = "denominator"
    ASSIGNED = "assigned"
    SUBJECT_ONLY = "subject_only"
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    SKIPPED = "skipped"
    PLACEMENT = "placement"
    CAPABILITY = "capability"


class CurriculumStatistic(_Record):
    source_id: str
    metric: CoverageMetric
    subject_id: str
    node_id: str
    instances: int


class CurriculumConfidenceStatistic(_Record):
    source_id: str
    metric: CoverageMetric
    subject_id: str
    node_id: str
    instances: list[int]


class CurriculumConfidenceLevel(_Record):
    label: str
    subject_min_similarity: float
    capability_min_similarity: float


class CurriculumCoverage(_Record):
    coverage_id: str
    alignment_release_id: str
    curriculum_key: str
    subject_distance: float
    capability_distance: float
    task_count: int
    paths: list[str]
    sample_paths: list[str]
    sample_task_count: int
    source_sample_size: int
    area_sample_size: int
    version: str = LEGACY_COVERAGE_VERSION
    confidence_levels: list[CurriculumConfidenceLevel] = Field(default_factory=list)
    confidence_paths: list[str] = Field(default_factory=list)
