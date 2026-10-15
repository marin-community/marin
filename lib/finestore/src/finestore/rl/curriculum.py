# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable scored curriculum releases consumed by Atlas."""

from typing import Any, Literal

from pydantic import Field

from finestore.rl.curation import CatalogRecord, SourceDeclaration


class AlignmentSource(CatalogRecord):
    source_id: str
    source_key: str
    population: Literal["converted", "admitted"]
    intended_use: str
    task_count: int
    ready_count: int
    insufficient_context_count: int
    unique_content_count: int
    curation_root: str | None = None
    input_rows: int | None = None
    converted_rows: int | None = None
    rejections: dict[str, int] = Field(default_factory=dict)


class CurriculumNode(CatalogRecord):
    node_id: str
    subject_id: str
    parent_id: str | None
    kind: Literal["subject", "group", "capability"]
    name: str
    description: str


class AlignmentRelease(CatalogRecord):
    release_id: str
    created_at: str
    curriculum_key: str
    population: Literal["converted", "admitted"]
    sample_size: int | None
    task_count: int
    ready_count: int
    insufficient_context_count: int
    unique_content_count: int
    unique_embedding_count: int
    sources: list[AlignmentSource]
    inventory: list[SourceDeclaration]
    nodes: list[CurriculumNode]
    paths: list[str]
    lineage: dict[str, str]
    embedding_spec: dict[str, Any]
