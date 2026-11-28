# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve and fingerprint only the declaration's SHA-bound TaskSpec population."""

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import ActorCensusInput
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.source_fingerprints import SourceFingerprint

from experiments.post_training.task_curation.content_identity import artifact_spec_digest
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.published_census import population_rows
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.source_fingerprints import fingerprint_source
from experiments.post_training.task_curation.source_population_cache import PopulationLocations
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


@dataclass(frozen=True)
class PopulationFingerprint:
    population: SourceTextPopulation
    fingerprint: SourceFingerprint
    locations: PopulationLocations


def resolve_population(
    source_id: str,
    binding: ActorCensusInput,
    manifest: ArtifactRef,
    *,
    region_prefix: str,
) -> SourceTextPopulation:
    """Verify declared identity and count before locating or opening any shard.

    The manifest is a sealed population document produced by the graph. Repair
    mode obtains its binding through the declaration's maintained census chain.
    No staging-directory search or population-name convention is consulted.
    """
    population = SourceTextPopulation.model_validate(read_ref(manifest))
    if population.source_id != canonical_source_id(source_id):
        raise ValueError("population manifest belongs to another source")
    if population.task_count != binding.measured_tasks:
        raise ValueError(
            f"staged population count differs from declaration: {population.task_count} != {binding.measured_tasks}"
        )
    actual = (
        population.dataset_id,
        population.dataset_revision,
        population.source_version,
        population.converter_revision,
        population.task_specs_sha256,
    )
    expected = (
        binding.dataset_id,
        binding.dataset_revision,
        binding.converter_version,
        binding.converter_revision,
        binding.task_specs_sha256,
    )
    if actual != expected:
        raise ValueError("population manifest differs from the declaration pin or TaskSpec digest")
    for uri in (population.path, population.specs_output_dir, population.main_output_dir):
        StoragePath(uri).relative_to(StoragePath(region_prefix))
    return population


def population_inputs(population: SourceTextPopulation) -> dict[str, str]:
    """Check every shard's footer count before hashing payload bytes in region."""
    paths = sorted((StoragePath(population.specs_output_dir) / "*.parquet").glob(), key=str)
    if not paths:
        raise ValueError("declared population has no TaskSpec shards")
    count = 0
    for path in paths:
        with path.open("rb") as stream:
            count += pq.ParquetFile(stream).metadata.num_rows
    if count != population.task_count:
        raise ValueError(f"staged shard count differs from declaration: {count} != {population.task_count}")
    return {
        (str(path) if path.scheme is not None else Path(str(path)).resolve().as_uri()): file_sha256(path)
        for path in paths
    }


def fingerprint_population(
    source_id: str,
    binding: ActorCensusInput,
    manifest: ArtifactRef,
    *,
    region_prefix: str,
) -> PopulationFingerprint:
    """Measure complete declared bytes and emit co-located cache locators.

    Run on a worker in the population's region. Local preflight supplies small
    storage fixtures at the same I/O boundary.
    """
    population = resolve_population(source_id, binding, manifest, region_prefix=region_prefix)
    inputs = population_inputs(population)
    if artifact_spec_digest(population) != binding.task_specs_sha256:
        raise ValueError("population TaskSpec bytes differ from the declaration digest")
    fingerprint = fingerprint_source((json.loads(row["task_spec"]) for row in population_rows(population)), inputs)
    if fingerprint.task_count != binding.measured_tasks or fingerprint.within_source_duplicates:
        raise ValueError("fingerprint must cover the complete within-source-deduplicated population")
    if population_inputs(population) != inputs:
        raise ValueError("population shards changed during fingerprinting")
    return PopulationFingerprint(
        population=population,
        fingerprint=fingerprint,
        locations=PopulationLocations(files={digest: uri for uri, digest in inputs.items()}),
    )
