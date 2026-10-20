# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned source declarations convert and verify native structural rows."""

import json
from pathlib import Path

import pytest
from taskcompendium.models import TaskResource, TextMessage
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import CheckStatus, IntendedUse
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.structure_closure import sources
from experiments.post_training.task_curation.tests.conversion import converted_task

FIXTURE = Path(__file__).parent / "fixtures/structure_closure.json"


@pytest.mark.parametrize("intended_use", [IntendedUse.TRAIN, IntendedUse.EVAL])
def test_structure_closure_declared_split_uses_text_prefill_and_passes_grader_controls(intended_use):
    source = next(
        source for source in sources() if source.config is not None and source.config.intended_use == intended_use
    )
    config = source.config
    assert config is not None and config.controls is not None and config.source.select is not None
    row = json.loads(FIXTURE.read_text())
    native = json.loads(row["lowered_task_spec"])["task"]
    native_config = json.loads(resource_bytes(TaskResource.model_validate(native["resources"]["verifier"][0])))
    task = converted_task(config, row)
    assert task.source.dataset == config.source.name
    assert task.source.revision == config.source.revision
    assert task.id != native["id"]
    assert task.context.events[-1] == TextMessage(role="assistant", content=native_config["extras"]["prefix"])
    verification = run_controls(task, controls=config.controls, machines=None)
    assert verification.checks and all(check.status == CheckStatus.PASS for check in verification.checks)
    other_family = {"lowered_task_spec": json.dumps({"task": {"source": {"dataset": "NVIDIA"}}})}
    assert not config.source.select(other_family, ConversionContext({}, None))
