# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen lexical metrics follow their measured inputs without granting admission."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    DaytonaMeasurement,
    LexicalContaminationEvidence,
    LexicalUse,
    SourceDeclaration,
)
from finestore.schema import arrow_schema

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_binding import (
    LexicalMeasurementReceipt,
    bind_lexical_measurement,
)
from experiments.post_training.task_curation.lexical_policy import LexicalInputUse
from experiments.post_training.task_curation.lexical_registration import main as register_measurements
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.source import RlDataSource, SourceOrigin
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence


def measured_source(path: Path) -> tuple[RlDataSource, LexicalMeasurementReceipt]:
    source = number_source(path)
    dataset = source.dataset
    assert dataset is not None
    document = decontamination_evidence(dataset.name, dataset.revision, 1).model_dump(mode="json")
    document.update(
        receipt=None,
        contaminated_tasks=1,
        contaminated_fraction=1,
        max_overlap=1,
        mean_max_overlap=1,
        p95_max_overlap=1,
        matched_evals={"asdiv": 1},
    )
    evidence = LexicalContaminationEvidence.model_validate(document)
    population_input = ActorCensusInput(
        dataset_id=dataset.name,
        dataset_revision=dataset.revision,
        converter_version=source.version,
        converter_revision=evidence.converter_revision,
        measured_tasks=1,
        task_specs_sha256="b" * 64,
        evidence_url="s3://fixture/normalization/.artifact.json",
    )
    return (
        replace(source, info=replace(source.info, lexical_contamination=None, actor_census_input=population_input)),
        LexicalMeasurementReceipt(
            source_id=source.info.id, input=population_input, use=LexicalUse.TRAIN, measurement=evidence
        ),
    )


def test_historical_measurement_reaches_canonical_catalog_without_admitting_training(tmp_path):
    source, receipt = measured_source(tmp_path / "input")
    historical_id = "MarinSkyRL:apps"
    source = replace(
        source,
        info=replace(
            source.info,
            id=canonical_source_id(historical_id),
            origin=SourceOrigin("CodeParrot"),
            upstream_dataset="apps",
        ),
    )
    record = {**receipt.model_dump(mode="json"), "source_id": historical_id}
    (tmp_path / "part-000.json").write_text(json.dumps({historical_id: record}))
    receipts = read_receipt_shards(
        tmp_path, LexicalMeasurementReceipt, "lexical", load_json=lambda path: json.loads(path.read_text())
    )
    bound = bind_lexical_measurement(source, receipts[source.info.id], LexicalUse.TRAIN)
    table = pa.Table.from_pylist([bound.declaration.model_dump(mode="json")], schema=arrow_schema(SourceDeclaration))
    restored = SourceDeclaration.model_validate(table.to_pylist()[0])
    assert restored.lexical_contamination is not None and source.dataset is not None
    assert restored.id == canonical_source_id(historical_id)
    assert restored.lexical_contamination.contaminated_tasks == 1
    assert restored.lexical_contamination.matched_evals == {"asdiv": 1}
    assert restored.lexical_contamination.receipt is None
    assert restored.lexical_contamination.admission_blocker(source.dataset.name, source.dataset.revision) is not None


@pytest.mark.parametrize("changed", ["pin", "version", "converter", "digest", "use", "missing"])
def test_stale_or_missing_measurement_clears_prior_metrics(tmp_path, changed):
    source, receipt = measured_source(tmp_path)
    source = bind_lexical_measurement(source, receipt, LexicalUse.TRAIN)
    use = LexicalUse.TRAIN
    assert source.config is not None and source.dataset is not None
    if changed == "pin":
        source = replace(source, config=replace(source.config, dataset=replace(source.dataset, revision="c" * 40)))
    elif changed == "version":
        source = replace(source, config=replace(source.config, version="2"))
    elif changed in {"converter", "digest"}:
        field = "converter_revision" if changed == "converter" else "task_specs_sha256"
        updated = receipt.input.model_copy(update={field: "c" * 64})
        source = replace(source, info=replace(source.info, actor_census_input=updated))
    elif changed == "use":
        use = LexicalUse.EVAL
    else:
        receipt = None
    rebound = bind_lexical_measurement(source, receipt, use)
    assert rebound.info.lexical_contamination is None


def test_measurement_registration_preserves_current_release_proof(tmp_path):
    source, receipt = measured_source(tmp_path)
    assert source.dataset is not None and source.config is not None
    dataset = replace(source.dataset, revision="c" * 40)
    proof = decontamination_evidence(dataset.name, dataset.revision, 1)
    source = replace(
        source, config=replace(source.config, dataset=dataset), info=replace(source.info, lexical_contamination=proof)
    )
    bound = bind_lexical_measurement(source, receipt, LexicalUse.TRAIN)
    assert bound.info.lexical_contamination is not None
    assert bound.info.lexical_contamination == proof
    assert bound.info.lexical_contamination.admission_blocker(dataset.name, dataset.revision) is None


def test_registration_writes_fresh_shards_and_preserves_occupied_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.post_training.task_curation.lexical_registration.read_registry_json",
        lambda path: json.loads(path.read_text()),
    )
    source, receipt = measured_source(tmp_path / "input")
    population = SourceTextPopulation(
        path="s3://fixture/normalization",
        source_id=source.info.id,
        dataset_id=receipt.input.dataset_id,
        dataset_revision=receipt.input.dataset_revision,
        source_version=receipt.input.converter_version,
        converter_revision=receipt.input.converter_revision,
        text_renderer_version=receipt.measurement.text_renderer_version,
        task_count=receipt.input.measured_tasks,
        text_sha256="a" * 64,
        task_specs_sha256=receipt.input.task_specs_sha256,
        main_output_dir="s3://fixture/text",
        specs_output_dir="s3://fixture/specs",
        converter_rejections={},
        admission_rejections={},
    )
    actor = ActorCensusReceipt(
        source_id=source.info.id,
        input=receipt.input,
        measurement=DaytonaMeasurement(
            dataset_id=receipt.input.dataset_id,
            dataset_revision=receipt.input.dataset_revision,
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Full fixture census",
            daytona_unsafe_reasons=[],
            daytona_unsafe_details={},
            daytona_unsafe_counts={},
        ),
        integrity_contracts=None,
    )
    use = LexicalInputUse(
        source_id=source.info.id,
        dataset_id=receipt.input.dataset_id,
        dataset_revision=receipt.input.dataset_revision,
        converter_revision=receipt.input.converter_revision,
        use=receipt.use,
    )
    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    (actor_dir / "part-000.json").write_text(json.dumps({source.info.id: actor.model_dump(mode="json")}))
    inputs = {
        "outcomes": {"measurements": {source.info.id: receipt.measurement.model_dump(mode="json")}},
        "populations": [population.model_dump(mode="json")],
        "source-uses": [use.model_dump(mode="json")],
    }
    arguments = ["lexical-registration"]
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        arguments.extend((f"--{name}", str(path)))
    output = tmp_path / "fresh"
    monkeypatch.setattr(sys, "argv", [*arguments, "--actor-receipts-dir", str(actor_dir), "--output-dir", str(output)])
    register_measurements()
    written = read_receipt_shards(
        output, LexicalMeasurementReceipt, "lexical", load_json=lambda path: json.loads(path.read_text())
    )[source.info.id]
    assert written.measurement.contaminated_tasks == 1 and written.measurement.receipt is None
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    previous = occupied / "part-000.json"
    previous.write_bytes(b'{"older": "immutable receipt"}\n')
    before = previous.read_bytes()
    monkeypatch.setattr(sys, "argv", [*arguments, "--actor-receipts-dir", str(actor_dir), "--output-dir", str(occupied)])
    with pytest.raises(FileExistsError):
        register_measurements()
    assert previous.read_bytes() == before
