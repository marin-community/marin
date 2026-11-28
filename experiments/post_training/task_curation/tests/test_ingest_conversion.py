# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real QUICK conversion feeds dedup and census from the sealed upstream map."""

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.conversion import ConvertedPopulation, convert_stage
from experiments.post_training.task_curation.ingest_stages.graph import execute_stage
from experiments.post_training.task_curation.ingest_stages.normalized_population import census_stage, dedup_stage
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage, load_receipt
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.release_stages.manifest import read_ref
from experiments.post_training.task_curation.tests.ingest_fixtures import next_job


def test_conversion_dedup_census_preserve_the_complete_duplicate_drop_proof(staged_job):
    dedup_job, converted = next_job(staged_job, convert_stage, IngestStage.DEDUP)
    census_job, dedup = next_job(dedup_job, dedup_stage, IngestStage.CENSUS)
    artifact = execute_stage(census_stage, census_job)
    census = load_receipt(artifact.receipt, census_job.inputs)
    population = SourceTextPopulation.model_validate(read_ref(dedup.outputs["population"]))
    measured = ActorCensus.model_validate(read_ref(census.outputs["actor-integrity"]))
    evidence = population.within_source_dedup
    assert (evidence["raw_count"], evidence["unique_count"], evidence["within_source_duplicates"]) == (2, 1, 1)
    assert population.task_count == measured.task_count == 1
    assert population.task_specs_sha256 == measured.task_specs_sha256
    conversion_evidence = ConvertedPopulation.model_validate(read_ref(converted.outputs["converted"]))
    assert conversion_evidence.drops
    assert SourceTextPopulation.raw_load(population.path) == population
    # A retry reuses sealed outputs instead of mixing another conversion attempt.
    assert convert_stage(staged_job) == converted.outputs


def test_conversion_refuses_a_replaced_upstream_map_before_writing_tasks(staged_job, tmp_path):
    impostor = seal_manifest({"not": "the dependency's staged map"}, str(tmp_path / "impostor.json"))
    inputs = staged_job.inputs.model_copy(
        update={"artifacts": {**staged_job.inputs.artifacts, "stage-upstream/staged": impostor}}
    )
    with pytest.raises(ValueError, match="differs from the staged upstream dependency"):
        execute_stage(convert_stage, replace(staged_job, inputs=inputs))
    assert not Path(staged_job.output_root).exists()
