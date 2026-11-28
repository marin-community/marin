# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The declared population, rather than a staging convention, controls survey I/O."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ActorCensusInput

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.population import fingerprint_population, resolve_population
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation


@pytest.fixture
def declared_population(qc_inputs, tmp_path):
    config, _ = qc_inputs
    population = SourceTextPopulation.raw_load(config.population_path)
    binding = ActorCensusInput(
        dataset_id=population.dataset_id,
        dataset_revision=population.dataset_revision,
        converter_version=population.source_version,
        converter_revision=population.converter_revision,
        measured_tasks=population.task_count,
        task_specs_sha256=population.task_specs_sha256,
        evidence_url=str(tmp_path / "census.json"),
    )
    return population, binding


def test_population_fingerprint_emits_only_file_uri_digests_and_colocated_locators(declared_population, tmp_path):
    population, binding = declared_population
    manifest = seal_manifest(population.model_dump(mode="json"), str(tmp_path / "population.json"))

    result = fingerprint_population(population.source_id, binding, manifest, region_prefix=str(tmp_path))

    inputs = {
        path.resolve().as_uri(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(population.specs_output_dir).glob("*.parquet"))
    }
    assert result.fingerprint.task_count == binding.measured_tasks
    assert result.fingerprint.inputs == inputs
    assert result.locations.files == {digest: uri for uri, digest in inputs.items()}


def test_raw_staging_count_is_refused_before_accessing_nonexistent_shards(declared_population, tmp_path):
    population, binding = declared_population
    raw = population.model_copy(
        update={"task_count": binding.measured_tasks * 106, "specs_output_dir": str(tmp_path / "must-not-open")}
    )
    manifest = seal_manifest(raw.model_dump(mode="json"), str(tmp_path / "raw-population.json"))

    with pytest.raises(ValueError, match="staged population count differs from declaration"):
        fingerprint_population(population.source_id, binding, manifest, region_prefix=str(tmp_path))


def test_raw_shard_rows_are_refused_before_task_decoding(declared_population, tmp_path):
    population, binding = declared_population
    # The footer reveals a pre-dedup population even when its schema cannot be
    # decoded as TaskSpecs. Rejection must precede hashing or row decoding.
    for path in Path(population.specs_output_dir).glob("*.parquet"):
        path.unlink()
    pq.write_table(
        pa.table({"raw_row": range(binding.measured_tasks * 106)}),
        Path(population.specs_output_dir) / "raw.parquet",
    )
    manifest = seal_manifest(population.model_dump(mode="json"), str(tmp_path / "population.json"))

    with pytest.raises(ValueError, match="staged shard count differs from declaration"):
        fingerprint_population(population.source_id, binding, manifest, region_prefix=str(tmp_path))


def test_population_from_another_pin_is_refused_before_shard_access(declared_population, tmp_path):
    population, binding = declared_population
    wrong_pin = population.model_copy(
        update={"dataset_revision": "another-pin", "specs_output_dir": str(tmp_path / "must-not-open")}
    )
    manifest = seal_manifest(wrong_pin.model_dump(mode="json"), str(tmp_path / "wrong-pin.json"))

    with pytest.raises(ValueError, match="declaration pin or TaskSpec digest"):
        resolve_population(population.source_id, binding, manifest, region_prefix=str(tmp_path))


def test_same_count_with_changed_taskspec_bytes_cannot_be_fingerprinted(declared_population, tmp_path):
    population, binding = declared_population
    path = next(Path(population.specs_output_dir).glob("*.parquet"))
    table = pq.read_table(path)
    rows = table.to_pylist()
    document = json.loads(rows[0]["task_spec"])
    document["context"]["events"][0]["content"] = "A changed public question"
    rows[0]["task_spec"] = json.dumps(document)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest = seal_manifest(population.model_dump(mode="json"), str(tmp_path / "population.json"))

    with pytest.raises(ValueError, match="TaskSpec bytes differ from the declaration digest"):
        fingerprint_population(population.source_id, binding, manifest, region_prefix=str(tmp_path))
