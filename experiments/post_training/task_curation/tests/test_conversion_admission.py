# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The oracle-flow conversion admission binds typed receipts and refuses drift."""

import json
from pathlib import Path

import pytest
from finestore.rl.curation import (
    ActorCensusInput as ActorCensusInputModel,
)
from finestore.rl.curation import (
    DatasetRelease,
    DaytonaMeasurement,
    LexicalContaminationEvidence,
    LexicalUse,
    SourceOrigin,
    TaskFormat,
)
from finestore.rl.status_taxonomy import DataRestriction
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation import conversion_admission
from experiments.post_training.task_curation.census_binding import ActorCensusReceipt
from experiments.post_training.task_curation.conversion_admission import (
    ConversionAdmissionReceipt,
    bind_conversion_admission,
)
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_binding import LexicalMeasurementReceipt
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_cohort
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

ORIGINAL_REPO = "open-athena/task-trove-laion__exp_rpt_fixture-oracle-filter-20261114-v2"
CONVERSION_REPO = "open-athena/laion__fixture-b0101045-taskspec-20261126b"
CONVERSION_REV = "d" * 40
ACTOR_REV = "010b92a14fd4607db55c4462cdcf337ee8b96517"


def _receipt(**overrides) -> ConversionAdmissionReceipt:
    release = DatasetRelease(
        repo=CONVERSION_REPO,
        revision=CONVERSION_REV,
        url=f"https://huggingface.co/datasets/{CONVERSION_REPO}/tree/{CONVERSION_REV}",
    )
    actor_input = ActorCensusInputModel(
        dataset_id=CONVERSION_REPO,
        dataset_revision=CONVERSION_REV,
        converter_version="harbor-cohort-taskspec-v1",
        converter_revision=ACTOR_REV,
        measured_tasks=12,
        task_specs_sha256="c" * 64,
        evidence_url="s3://fixture/staged-census/.artifact.json",
    )
    measurement = DaytonaMeasurement(
        dataset_id=CONVERSION_REPO,
        dataset_revision=CONVERSION_REV,
        daytona_snapshot_count=0,
        daytona_machine_slots=12,
        daytona_execution_failures=0,
        daytona_count_basis="Declared demands measured from the staged population artifact.",
        daytona_unsafe_reasons=[],
        daytona_unsafe_details={},
        daytona_unsafe_counts={},
        daytona_roles={},
    )
    fields = dict(
        source_id="TaskTrove:laion__fixture-v4",
        intended_use=IntendedUse.TRAIN,
        original_release=DatasetRelease(
            repo=ORIGINAL_REPO,
            revision="e" * 40,
            url=f"https://huggingface.co/datasets/{ORIGINAL_REPO}/tree/{'e' * 40}",
        ),
        original_pipeline_revision="1509fc8638f6be3840b5c82b53348f93152a9c31",
        conversion_release=release,
        acceptance="basepin-acceptance-fixture",
        actor=ActorCensusReceipt(
            source_id="TaskTrove:laion__fixture-v4",
            input=actor_input,
            measurement=measurement,
            integrity_contracts=None,
        ),
        lexical_measurement_uri="s3://fixture/lexical-measurement-receipts.json",
        grading_evidence_uri="s3://fixture/grading-receipts.json",
        population_record="s3://fixture/lexical/text/popkey/2026.11.26.1",
    )
    fields.update(overrides)
    return ConversionAdmissionReceipt(**fields)


def _staged_population(tmp_path, receipt: ConversionAdmissionReceipt) -> Path:
    """A real staged population record at a local path, matching the census binding."""

    population = SourceTextPopulation(
        path="placeholder",
        source_id=receipt.source_id,
        dataset_id=receipt.conversion_release.repo,
        dataset_revision=receipt.conversion_release.revision,
        source_version="harbor-cohort-taskspec-v1",
        converter_revision=ACTOR_REV,
        text_renderer_version="taskcompendium-format-conversation-v1",
        task_count=12,
        text_sha256="a" * 64,
        task_specs_sha256="c" * 64,
        main_output_dir=str(tmp_path / "outputs"),
        specs_output_dir=str(tmp_path / "specs"),
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": 12,
            "unique_count": 12,
            "within_source_duplicates": 0,
            "drop_manifest_uri": "s3://fixture/within-source-dedup/drops.parquet",
        },
    )

    root = tmp_path / "staged-population"
    write_record(
        ArtifactRecord(
            output_path=str(root),
            result_type=result_type_name(SourceTextPopulation),
            result=population.result_payload(),
        )
    )
    return root


def _original_source(receipt: ConversionAdmissionReceipt) -> RlDataSource:
    """A configured oracle-flow declaration bound to the receipt's original zero-task pin."""
    config = PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="fixture",
        version="2026.11.14.1",
        repo=receipt.original_release.repo,
        revision=receipt.original_release.revision,
        expected_rows=0,
    )
    info = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        id=receipt.source_id,
        upstream_dataset="laion__fixture-v4",
        upstream_name="laion__fixture-v4",
        build_path="Task Trove",
        origin=SourceOrigin.TASKTROVE,
        family="competitive-programming",
        tags=("agentic", "single-turn", "excluded"),
        task_format=TaskFormat.TASKSPEC,
        count=0,
        notes="Excluded: the mechanical wave retained zero tasks.",
    )
    return RlDataSource(config=config, pipeline=_original_pipeline(), info=info)


def _original_pipeline():

    return import_taskcompendium_cohort


def _inject_lexical(monkeypatch, receipt: ConversionAdmissionReceipt) -> None:
    """A registered lexical measurement at the conversion release (0 contaminated)."""

    evidence = receipt_lexical_evidence(receipt)
    monkeypatch.setattr(
        conversion_admission,
        "lexical_measurements",
        lambda: {
            receipt.source_id: LexicalMeasurementReceipt(
                source_id=receipt.source_id,
                input=receipt.actor.input,
                use=LexicalUse.TRAIN,
                measurement=evidence,
            )
        },
    )


def receipt_lexical_evidence(receipt: ConversionAdmissionReceipt):

    return LexicalContaminationEvidence.model_validate(
        {
            "version": 1,
            "input_dataset_id": receipt.conversion_release.repo,
            "input_dataset_revision": receipt.conversion_release.revision,
            "converter_revision": ACTOR_REV,
            "text_renderer_version": "taskcompendium-format-conversation-v1",
            "measured_tasks": 12,
            "contaminated_tasks": 0,
            "contaminated_fraction": 0.0,
            "max_overlap": 0.0,
            "mean_max_overlap": 0.0,
            "p95_max_overlap": 0.0,
            "matched_evals": {},
            "eval_corpus_version": "aa-index-v4.1.1-v3",
            "eval_bloom_uri": "s3://fixture/bloom",
            "eval_hash_index_uri": "s3://fixture/bloom/index",
            "feature_filter_version": 4,
            "ngram_length": 13,
            "overlap_threshold": 0.5,
            "min_matched_features": 2,
            "drop_set_uris": [],
            "task_flags_uri": "s3://fixture/flags",
            "evidence_url": "s3://fixture/lexical/report",
            "receipt": None,
        }
    )


def _grading_receipts_document(tmp_path, receipt: ConversionAdmissionReceipt) -> Path:
    """A local grading census receipt document covering the fixture source."""
    evidence = {
        "assessed_tasks": 12,
        "required_counts": {"separate": 12},
        "declared_counts": {"separate": 12},
        "status_counts": {"matched": 12},
        "reason_counts": {
            "The typed grader reads the declared submitted answer or captured files in its own environment": 12
        },
        "missing_artifact_counts": {},
        "declared_artifact_counts": {},
        "evidence_url": "s3://fixture/grading/receipt.json",
    }
    row = {"source_id": receipt.source_id, "evidence": evidence}
    path = tmp_path / "grading-receipts.json"
    path.write_text(json.dumps([row], indent=1))
    return path


def test_conversion_admission_binds_the_release_pin_as_executable_taskspec(tmp_path, monkeypatch):
    receipt = _receipt()
    receipt = receipt.model_copy(update={"population_record": str(_staged_population(tmp_path, receipt))})
    receipt = receipt.model_copy(update={"grading_evidence_uri": str(_grading_receipts_document(tmp_path, receipt))})
    _inject_lexical(monkeypatch, receipt)
    bound = bind_conversion_admission(_original_source(receipt), receipt)
    assert bound.info.task_format is TaskFormat.TASKSPEC
    assert bound.dataset is not None and bound.dataset.name == CONVERSION_REPO
    assert bound.info.count == 12
    assert "excluded" not in bound.info.tags
    assert bound.info.actor_census_input is not None
    assert bound.info.actor_census_input.dataset_revision == CONVERSION_REV
    assert bound.info.grading_environment is not None


def test_conversion_admission_refuses_another_population():
    receipt = _receipt()
    drifted = receipt.model_copy(
        update={
            "original_release": DatasetRelease(
                repo=ORIGINAL_REPO + "-other",
                revision="f" * 40,
                url=f"https://huggingface.co/datasets/{ORIGINAL_REPO}-other/tree/{'f' * 40}",
            )
        }
    )
    with pytest.raises(ValueError, match="another configured original population"):
        bind_conversion_admission(_original_source(receipt), drifted)


def test_conversion_admission_refuses_a_cross_pin_census(tmp_path, monkeypatch):
    receipt = _receipt()
    receipt = receipt.model_copy(update={"population_record": str(_staged_population(tmp_path, receipt))})
    _inject_lexical(monkeypatch, receipt)
    drifted = receipt.model_copy(
        update={
            "conversion_release": receipt.conversion_release.model_copy(
                update={
                    "revision": "0" * 40,
                    "url": f"https://huggingface.co/datasets/{CONVERSION_REPO}/tree/{'0' * 40}",
                }
            )
        }
    )
    with pytest.raises(ValueError, match="another release pin"):
        bind_conversion_admission(_original_source(receipt), drifted)
