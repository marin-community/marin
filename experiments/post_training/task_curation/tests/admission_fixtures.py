# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census records for local synthetic source populations."""

import hashlib
import json
from dataclasses import replace

import pytest
from finestore.rl.curation import ActorCensusInput, GradingEnvironmentEvidence, SourceDeclaration

from experiments.post_training.task_curation import registry, retirement
from experiments.post_training.task_curation.source import RlDataSource


@pytest.fixture
def isolated_admission_census(tmp_path_factory, monkeypatch, request):
    if request.node.get_closest_marker("data_integration") is not None:
        yield
        return
    root = tmp_path_factory.mktemp("admission-registry")

    def download(**kwargs):
        if kwargs["repo_id"] != "fixture/admission-registry" or kwargs["filename"] != "retirements.json":
            raise AssertionError("Unit tests must inject local registry documents at the download boundary")
        return str(ledger_document)

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
    monkeypatch.setattr(registry, "hf_hub_download", download)
    yield
    retirement.retired_admission_source_ids.cache_clear()


def actor_census_input(dataset: str, revision: str, version: str, count: int) -> ActorCensusInput:
    binding = ActorCensusInput(
        dataset_id=dataset,
        dataset_revision=revision,
        converter_version=version,
        converter_revision="fixture",
        measured_tasks=count,
        task_specs_sha256="a" * 64,
        evidence_url="https://example.test/census",
    )
    return binding


def with_admission_census(source: RlDataSource) -> RlDataSource:
    assert source.dataset is not None and source.info.within_source_dedup is not None
    binding = actor_census_input(
        source.dataset.name,
        source.dataset.revision,
        source.version,
        source.info.within_source_dedup.unique_count,
    )
    return replace(
        source,
        info=replace(
            source.info,
            actor_census_input=binding,
            grading_environment=source.info.grading_environment or grading_evidence(binding.measured_tasks),
        ),
    )


def with_declaration_admission_census(declaration: SourceDeclaration) -> SourceDeclaration:
    assert declaration.dataset_id is not None and declaration.dataset_revision is not None
    assert declaration.pipeline is not None
    binding = actor_census_input(
        declaration.dataset_id,
        declaration.dataset_revision,
        declaration.pipeline.version,
        declaration.within_source_dedup.unique_count,
    )
    return SourceDeclaration.model_validate(
        {
            **declaration.model_dump(),
            "actor_census_input": binding,
            "grading_environment": declaration.grading_environment or grading_evidence(binding.measured_tasks),
        }
    )


def grading_evidence(count: int) -> GradingEnvironmentEvidence:
    return GradingEnvironmentEvidence(
        assessed_tasks=count,
        required_counts={"separate": count},
        declared_counts={"separate": count},
        status_counts={"matched": count},
        reason_counts={"Typed grader": count},
        evidence_url="https://example.test/grading",
    )
