# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import asdict

import pytest
from finestore.rl.curation import ConsumerNeed, DatasetRelease, WithinSourceDedupEvidence
from finestore.rl.status_taxonomy import DataRestriction
from pydantic import TypeAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets import nemo_structured, workplace
from experiments.post_training.task_curation.datasets.terminalworld import qualified
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import SourceInfo, SourceOrigin, TaskFormat
from experiments.post_training.task_curation.tests.admission_fixtures import actor_census_input, consumer_receipt


@pytest.mark.parametrize(
    "module,origin,name,count",
    [
        (qualified, SourceOrigin.TERMINALWORLD, "TerminalWorld", 4),
        (nemo_structured, SourceOrigin.NEMOTRON, "Nemotron-RL-instruction_following-structured_outputs", 9213),
    ],
)
def test_qualification_selects_original_train_population_before_receipt_binding(
    monkeypatch, module, origin, name, count
):
    release = DatasetRelease(
        repo="open-athena/native-intake",
        revision="a" * 40,
        url="https://huggingface.co/datasets/open-athena/native-intake/tree/" + "a" * 40,
    )
    config = PrebuiltCohort(
        name="native-intake",
        version="native-v1",
        repo=release.repo,
        revision=release.revision,
        expected_rows=count,
        intended_use=IntendedUse.TRAIN,
    )
    info = SourceInfo(
        origin=origin,
        upstream_dataset=name,
        upstream_name=name,
        build_path="native conversion",
        family="terminal",
        count=count,
        task_format=TaskFormat.TASKSPEC,
        data_restriction=DataRestriction.OPEN,
        current_release=release,
    )
    record = {
        "source_id": info.id,
        "original_info": asdict(info),
        "config": asdict(config),
        "admission": {"filtration": {"original_count": count}},
        "retained": {
            "publication": {
                "plan": {
                    "source_dataset_id": release.repo,
                    "source_revision": release.revision,
                    "converter_version": "native-v1",
                    "raw_count": count,
                }
            }
        },
    }
    monkeypatch.setattr(module, "read_registry_json", lambda path: record)
    (source,) = module.sources()
    assert source.current_release == release
    assert source.config.expected_rows == count
    assert source.intended_use.value == "train"
    assert source.pipeline_reference.source == release.repo
    assert source.pipeline_reference.revision == release.revision
    assert source.info.task_format is TaskFormat.TASKSPEC


@pytest.fixture
def workplace_qualification():
    release = DatasetRelease(
        repo="open-athena/prepared-workplace",
        revision="a" * 40,
        url="https://huggingface.co/datasets/open-athena/prepared-workplace/tree/" + "a" * 40,
    )
    actor = actor_census_input(release.repo, release.revision, "prepared-v1", 3)
    source_id = "Nvidia:Nemotron-RL-agent-workplace_assistant"
    consumer = consumer_receipt(source_id, actor).model_copy(
        update={"runtime_commit": workplace.CONSUMER_RUNTIME, "requirements": {ConsumerNeed.TOOL_PROVIDER: 3}}
    )
    dedup = WithinSourceDedupEvidence(
        raw_count=3,
        unique_count=3,
        within_source_duplicates=0,
        source_dataset_id=actor.dataset_id,
        source_revision=actor.dataset_revision,
        converter_revision=actor.converter_revision,
        converter_version=actor.converter_version,
        drop_manifest_uri="s3://test/duplicates.parquet",
        evidence_url="s3://test/dedup.json",
    )
    info = SourceInfo(
        origin=SourceOrigin.NVIDIA,
        upstream_dataset="Nemotron-RL-agent-workplace_assistant",
        upstream_name="Workplace",
        build_path="Prepared native image",
        current_release=release,
        data_restriction=DataRestriction.OPEN,
        actor_census_input=actor,
        consumer_runtime_commit=workplace.CONSUMER_RUNTIME,
        consumer_receipt=consumer,
        consumer_requirements=(ConsumerNeed.TOOL_PROVIDER,),
        within_source_dedup=dedup,
    )
    cohort = PrebuiltCohort(
        name="prepared-workplace",
        version=actor.converter_version,
        repo=release.repo,
        revision=release.revision,
        expected_rows=3,
        intended_use=IntendedUse.TRAIN,
    )
    return {
        "source_id": source_id,
        "info": TypeAdapter(SourceInfo).dump_python(info, mode="json"),
        "cohort": asdict(cohort),
    }


def test_workplace_qualified_population_uses_its_matching_consumer_runtime(workplace_qualification, monkeypatch):
    monkeypatch.setattr(workplace, "read_registry_json", lambda path: workplace_qualification)
    (source,) = workplace.sources()
    assert source.consumer_findings == ()
    assert source.consumer_runtime_commit == source.info.consumer_receipt.runtime_commit


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_commit", "b" * 40),
        ("dataset_revision", "b" * 40),
        ("task_specs_sha256", "b" * 64),
        ("checked_tasks", 4),
        ("requirements", {ConsumerNeed.TOOL_PROVIDER: 2}),
    ],
)
def test_workplace_refuses_a_consumer_receipt_from_another_population(
    workplace_qualification, monkeypatch, field, value
):
    record = deepcopy(workplace_qualification)
    record["info"]["consumer_receipt"][field] = value
    monkeypatch.setattr(workplace, "read_registry_json", lambda path: record)
    with pytest.raises(ValueError, match="selected runtime or private population"):
        workplace.sources()
