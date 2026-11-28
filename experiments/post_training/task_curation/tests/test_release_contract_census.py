# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Release contracts follow frozen populations rather than older registry counts."""

from experiments.post_training.task_curation.derived_contract_census import (
    DerivedContractReceipt,
    census_source,
    scan_population,
)
from experiments.post_training.task_curation.release_stages import contract_census as stage
from experiments.post_training.task_curation.release_stages.census import seal
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.tests.test_derived_contract import (
    EXACT,
    JUDGE,
    population_fixture,
    reply_task,
)


def inputs_for(config, tmp_path):
    _, digest, _ = scan_population(config.population)
    census = seal(
        str(tmp_path / "frozen-census.json"),
        {
            "results": [
                {
                    "declaration": {
                        "id": config.population.source_id,
                        "actor_census_input": config.input.model_dump(mode="json"),
                    },
                    "population": config.population.model_dump(mode="json"),
                    "inventory": {"scanned_task_specs_sha256": digest},
                }
            ]
        },
    )
    return stage.ContractCensusInputs(
        release_id="testrelease",
        source_revision="a" * 40,
        census=census,
        targets=(config.population.source_id,),
        output_root=str(tmp_path / "contracts"),
    )


def test_changed_population_replaces_the_old_contract_and_resumes_without_rescanning(tmp_path, monkeypatch):
    old = population_fixture(tmp_path / "old", (reply_task("old", EXACT),))
    new = population_fixture(tmp_path / "new", (reply_task("new1", EXACT), reply_task("new2", JUDGE)))
    prior = census_source(old, str(tmp_path / "prior"))
    monkeypatch.setattr(stage, "derived_contract_receipts", lambda: {prior.source_id: prior})
    inputs = inputs_for(new, tmp_path)
    result = stage.run_contract_census(inputs, dry_run=False)
    receipt = DerivedContractReceipt.model_validate(read_ref(ArtifactRef(**result["contracts"]))["contracts"][0])
    assert result["contract_tasks"] == 2
    assert result["rescanned_sources"] == [prior.source_id]
    assert receipt.input == new.input and receipt.summary.task_count == 2

    def no_rescan(*args, **kwargs):
        raise AssertionError("A completed contract must resume from its sealed receipt")

    monkeypatch.setattr(stage, "census_source", no_rescan)
    assert stage.run_contract_census(inputs, dry_run=False) == result


def test_exact_population_reuses_its_registered_contract(tmp_path, monkeypatch):
    config = population_fixture(tmp_path / "population", (reply_task("one", EXACT),))
    prior = census_source(config, str(tmp_path / "prior"))
    monkeypatch.setattr(stage, "derived_contract_receipts", lambda: {prior.source_id: prior})
    inputs = inputs_for(config, tmp_path)

    def no_rescan(*args, **kwargs):
        raise AssertionError("An exact registered receipt needs no population rescan")

    monkeypatch.setattr(stage, "census_source", no_rescan)
    result = stage.run_contract_census(inputs, dry_run=False)
    assert result["reused_sources"] == [prior.source_id] and result["rescanned_sources"] == []
    assert read_ref(ArtifactRef(**result["contracts"]))["contracts"] == [prior.model_dump(mode="json")]
