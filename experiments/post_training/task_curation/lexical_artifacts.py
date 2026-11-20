# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable artifact types shared by Atlas lexical entrypoints."""

from typing import Any, Literal

from marin.datakit.decon import AllSourceDropSets, DeconAttributes, EvalBloom
from marin.execution.artifact import RECORD_FILENAME, Artifact, read_record, result_type_name
from pydantic import Field, field_validator
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId, canonical_source_id, canonical_source_map


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


class EvalReferenceBloom(Artifact, EvalBloom):
    """A lazy artifact whose payload also supports the Datakit Bloom merger."""


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


def population_for_census(evidence_url: str, source_id: str) -> SourceTextPopulation:
    """Resolve the frozen population from typed actor or population evidence."""
    pointer = StoragePath(evidence_url)
    if pointer.name != RECORD_FILENAME:
        raise ValueError(f"Census evidence is not an artifact record: {source_id}")
    record = read_record(str(pointer.parent))
    if record is None:
        raise ValueError(f"Census evidence record is missing: {source_id}")
    if record.result_type == result_type_name(ActorCensus):
        actor = ActorCensus.raw_load(str(pointer.parent))
        if actor.source_id != canonical_source_id(source_id):
            raise ValueError(f"Actor census belongs to another source: {source_id}")
        population = SourceTextPopulation.raw_load(actor.input_artifact)
    elif record.result_type == result_type_name(SourceTextPopulation):
        population = SourceTextPopulation.raw_load(str(pointer.parent))
    else:
        raise ValueError(f"Unsupported census evidence type for {source_id}: {record.result_type}")
    if population.source_id != canonical_source_id(source_id):
        raise ValueError(f"Census population belongs to another source: {source_id}")
    return population
