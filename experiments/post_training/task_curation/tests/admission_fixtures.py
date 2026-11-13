# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census records for local synthetic source populations."""

import hashlib
import json
from dataclasses import replace

import pytest
from finestore.rl.curation import ActorCensusInput

from experiments.post_training.task_curation import registry, retirement
from experiments.post_training.task_curation.admission_binding import AdmissionCensusReceipt
from experiments.post_training.task_curation.datasets import admission_censuses
from experiments.post_training.task_curation.datasets.admission_censuses import admission_census_receipts
from experiments.post_training.task_curation.source import RlDataSource


@pytest.fixture
def isolated_admission_census(tmp_path_factory, monkeypatch, request):
    if request.node.get_closest_marker("data_integration") is not None:
        yield
        return
    root = tmp_path_factory.mktemp("admission-registry")
    document = root / "document.json"
    document.write_text("{}\n")
    (root / "part-000.json").write_text(
        json.dumps(
            {
                "repo": "fixture/admission-registry",
                "revision": "a" * 40,
                "path": "admission.json",
                "sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
            }
        )
    )

    def download(**kwargs):
        if kwargs["repo_id"] != "fixture/admission-registry" or kwargs["filename"] not in {
            "admission.json",
            "retirements.json",
        }:
            raise AssertionError("Unit tests must inject local registry documents at the download boundary")
        return str(document if kwargs["filename"] == "admission.json" else ledger_document)

    ledger = root / "retirements.json"
    ledger_document = root / "retirement-document.json"
    ledger_document.write_text('{"entries": []}\n')
    ledger.write_text(
        json.dumps(
            {
                "repo": "fixture/admission-registry",
                "revision": "a" * 40,
                "path": "retirements.json",
                "sha256": hashlib.sha256(ledger_document.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(retirement, "RETIREMENTS_PATH", ledger)
    retirement.retired_admission_source_ids.cache_clear()
    monkeypatch.setattr(admission_censuses, "RECEIPTS_DIRECTORY", root)
    monkeypatch.setattr(registry, "hf_hub_download", download)
    admission_census_receipts.cache_clear()
    yield
    admission_census_receipts.cache_clear()
    retirement.retired_admission_source_ids.cache_clear()


def register_admission_census(source_id: str, dataset: str, revision: str, version: str, count: int) -> ActorCensusInput:
    binding = ActorCensusInput(
        dataset_id=dataset,
        dataset_revision=revision,
        converter_version=version,
        converter_revision="fixture",
        measured_tasks=count,
        task_specs_sha256="a" * 64,
        evidence_url="https://example.test/census",
    )
    admission_census_receipts()[source_id] = AdmissionCensusReceipt(
        source_id=source_id,
        input=binding,
        population_path="https://example.test/population",
        scanned_task_specs_sha256="a" * 64,
        retained_tasks=count,
        runnable_grader_tasks=count,
        no_grader_tasks=0,
        grader_kinds={"verifyit": count},
        missing_grader_reasons={},
        evidence_url="https://example.test/admission",
    )
    return binding


def with_admission_census(source: RlDataSource) -> RlDataSource:
    assert source.dataset is not None and source.info.within_source_dedup is not None
    binding = register_admission_census(
        source.info.id,
        source.dataset.name,
        source.dataset.revision,
        source.version,
        source.info.within_source_dedup.unique_count,
    )
    return replace(source, info=replace(source.info, actor_census_input=binding))
