# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census records for local synthetic source populations."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import huggingface_hub
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    ConsumerNeed,
    ConsumerReceipt,
    DatasetRelease,
    GradingEnvironmentEvidence,
)
from finestore.rl.curation import (
    QualifiedSourceDeclaration as SourceDeclaration,
)
from finestore.rl.status_taxonomy import DataRestriction
from pydantic import TypeAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation import registry, retirement
from experiments.post_training.task_curation.consumer_scope import CONSUMER_RUNTIME_COMMIT
from experiments.post_training.task_curation.datasets.workplace import CONSUMER_RUNTIME
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    source_identifier,
)
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license

ROOT = Path(__file__).resolve().parents[4]


def synthetic_workplace_document() -> dict:
    """A minimal consistent workplace qualification record for tests that enumerate the
    pending catalog: the same identity checks ``workplace.sources`` applies, at fixture
    values, so no test needs the real registry document."""

    dataset = DatasetRelease(
        repo="open-athena/fixture-workplace",
        revision="a" * 40,
        url="https://huggingface.co/datasets/open-athena/fixture-workplace/tree/" + "a" * 40,
    )
    sid = source_identifier(SourceOrigin.MARIN, "fixture-workplace", None, None)
    info = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        id=sid,
        origin=SourceOrigin.MARIN,
        upstream_dataset="fixture-workplace",
        upstream_name="Fixture Workplace",
        build_path="Fixture",
        count=3,
        current_release=dataset,
        actor_census_input=ActorCensusInput(
            dataset_id=dataset.repo,
            dataset_revision=dataset.revision,
            converter_version="fixture-1",
            converter_revision="b" * 40,
            measured_tasks=3,
            task_specs_sha256="c" * 64,
            evidence_url="https://example.test/census",
        ),
        consumer_receipt=ConsumerReceipt(
            source_id=sid,
            dataset_id=dataset.repo,
            dataset_revision=dataset.revision,
            task_specs_sha256="c" * 64,
            runtime_commit=CONSUMER_RUNTIME,
            checked_tasks=3,
            failed_tasks=0,
            failures={},
            evidence_url="https://example.test/consumer",
            requirements={ConsumerNeed.TOOL_PROVIDER: 3},
        ),
        consumer_requirements=(ConsumerNeed.TOOL_PROVIDER,),
        consumer_runtime_commit=CONSUMER_RUNTIME,
    )
    cohort = PrebuiltCohort(
        name="workplace-assistant",
        version="fixture-1",
        repo=dataset.repo,
        revision=dataset.revision,
        expected_rows=3,
        intended_use=IntendedUse.TRAIN,
    )
    return {
        "source_id": info.id,
        "info": TypeAdapter(SourceInfo).dump_python(info, mode="json"),
        "cohort": TypeAdapter(PrebuiltCohort).dump_python(cohort, mode="json"),
    }


@pytest.fixture
def isolated_admission_census(tmp_path_factory, monkeypatch, request):
    if request.node.get_closest_marker("data_integration") is not None:
        yield
        return
    root = tmp_path_factory.mktemp("admission-registry")

    workplace_pointer = json.loads(
        (ROOT / "experiments" / "post_training" / "task_curation" / "datasets" / "workplace.json").read_text()
    )
    workplace_document = root / "workplace-qualification.json"
    workplace_document.write_text(json.dumps(synthetic_workplace_document(), indent=2, sort_keys=True) + "\n")
    workplace_local_pointer = root / "workplace-pointer.json"
    workplace_local_pointer.write_text(
        json.dumps(
            {
                "repo": workplace_pointer["repo"],
                "revision": workplace_pointer["revision"],
                "path": workplace_pointer["path"],
                "sha256": hashlib.sha256(workplace_document.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    from experiments.post_training.task_curation.datasets import workplace as workplace_module  # noqa: PLC0415

    monkeypatch.setattr(workplace_module, "QUALIFICATION", workplace_local_pointer)
    workplace_module.qualification.cache_clear()

    def download(**kwargs):
        if kwargs.get("repo_id") == workplace_pointer["repo"] and kwargs.get("filename") == workplace_pointer["path"]:
            return str(workplace_document)
        if kwargs["repo_id"] == "open-athena/marin-rl-data-atlas-registry":
            # The pending sources' pinned qualification documents are deterministic
            # registry data; serve them from the pinned revision rather than asking
            # every pending module's test to stage a synthetic copy.
            return huggingface_hub.hf_hub_download(**kwargs)
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


def consumer_receipt(source_id: str, binding: ActorCensusInput) -> ConsumerReceipt:
    return ConsumerReceipt(
        source_id=source_id,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
        task_specs_sha256=binding.task_specs_sha256,
        runtime_commit=CONSUMER_RUNTIME_COMMIT,
        checked_tasks=binding.measured_tasks,
        failed_tasks=0,
        failures={},
        evidence_url="https://example.test/consumer",
    )


def with_consumer_receipt(source: RlDataSource) -> RlDataSource:
    assert source.info.actor_census_input is not None
    return replace(
        source,
        info=replace(
            source.info,
            consumer_receipt=consumer_receipt(source.info.id, source.info.actor_census_input),
        ),
    )


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
            license_receipt=fixture_license(source.info.id, source.dataset.name, source.dataset.revision),
            actor_census_input=binding,
            consumer_receipt=consumer_receipt(source.info.id, binding),
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
            "license_receipt": fixture_license(declaration.id, declaration.dataset_id, declaration.dataset_revision),
            "actor_census_input": binding,
            "consumer_receipt": consumer_receipt(declaration.id, binding),
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
