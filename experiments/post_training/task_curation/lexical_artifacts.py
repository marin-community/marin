# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable artifact types shared by Atlas lexical entrypoints."""

from typing import Any, Literal

from marin.datakit.decon import AllSourceDropSets, DeconAttributes
from marin.execution.artifact import Artifact
from pydantic import Field, field_validator

from experiments.post_training.task_curation.source_identity import SourceId, canonical_source_id, canonical_source_map


class SourceTextPopulation(Artifact):
    path: str = Field(min_length=1)
    source_id: SourceId
    dataset_id: str
    dataset_revision: str
    source_version: str
    converter_revision: str
    text_renderer_version: str
    task_count: int
    text_sha256: str
    task_specs_sha256: str
    main_output_dir: str
    specs_output_dir: str
    converter_rejections: dict[str, int]
    admission_rejections: dict[str, int]
    within_source_dedup: dict | None = None


class LexicalDropSets(Artifact):
    attributes: AllSourceDropSets


class SourceLexicalMarks(Artifact):
    source_id: SourceId
    population_path: str
    attributes: DeconAttributes


class LexicalMeasurements(Artifact):
    population_set: str
    source_ids: list[str]
    measurements: dict[str, dict]
    drop_manifests: dict[str, str]

    @field_validator("source_ids", mode="before")
    @classmethod
    def _source_list(cls, value: list[str]) -> list[str]:
        return [canonical_source_id(source_id) for source_id in value]

    @field_validator("measurements", "drop_manifests", mode="before")
    @classmethod
    def _source_maps(cls, value: dict[str, Any]) -> dict[str, Any]:
        return canonical_source_map(value)


class ActorCensus(Artifact):
    source_id: SourceId
    dataset_id: str
    dataset_revision: str
    converter_version: str
    converter_revision: str
    input_artifact: str
    task_specs_sha256: str
    task_count: int
    daytona_safety: Literal["safe", "unsafe", "not_applicable"]
    daytona_snapshot_count: int
    daytona_machine_slots: int
    daytona_execution_failures: int
    daytona_count_basis: str
    daytona_unsafe_reasons: list[str]
    daytona_unsafe_details: dict[str, str]
    daytona_unsafe_counts: dict[str, int]
    daytona_roles: dict
    actor_environment_counts: dict[str, int]
    empty_declared_environment_tasks: dict[str, int]
    snapshot_sizes: dict[str, int]
    execution_failures: dict[str, int]
    mapping_failures: dict[str, int]
    integrity_contracts: dict


class IntegrityCensus(Artifact):
    source_id: SourceId
    input_artifact: str
    converter_version: str
    converter_revision: str
    task_count: int
    summary: dict
