# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import pytest
from finestore.rl.curation import DatasetRelease
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets import nemo_structured
from experiments.post_training.task_curation.datasets.terminalworld import qualified
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import SourceInfo, SourceOrigin, TaskFormat


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
