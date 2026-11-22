# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tarfile
from pathlib import Path

import pytest
from taskcompendium.harbor.adapter import TASK_SPEC_FILE
from taskcompendium.harbor.negative_control import NegativeControlSource
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec

from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.forge_nop_wave import control_request

FIXTURES = Path(__file__).parent / "fixtures/negative_controls"


def test_nop_request_binds_calibrated_contract_without_replacing_source_identity(tmp_path):
    with tarfile.open(FIXTURES / "packet-runtime-fallback.tar.gz") as archive:
        source = archive.extractfile("nop-control/source-task-spec.json").read()
    task = TaskSpec.model_validate_json(source)
    (tmp_path / TASK_SPEC_FILE).write_bytes(source)
    configuration = b"""[environment]
cpus = 1
memory_mb = 2048
storage_mb = 2048
[agent]
timeout_sec = 300
[verifier]
timeout_sec = 60
"""
    (tmp_path / "task.toml").write_bytes(configuration)
    actor = ExecutionRequirements(
        cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
    )
    verifier = actor.model_copy(update={"memory_mb": 4096})
    original_hash = task_spec_sha256(task)
    entry = {"task_sha256": original_hash, "source_task_checksum": "a" * 64, "task_checksum": "b" * 64}
    request = control_request(task.id, entry, tmp_path, verifier.model_dump(mode="json"), None, tmp_path / "output")
    bound = NegativeControlSource.model_validate(request["source"])
    published = calibrated_task_spec(task, actor, verifier, configuration)
    assert bound.task_sha256 == task_spec_sha256(published)
    assert bound.task_sha256 != bound.source_task_sha256
    assert bound.source_task_sha256 == original_hash
    assert (tmp_path / TASK_SPEC_FILE).read_bytes() == source

    changed = {**entry, "task_sha256": "c" * 64}
    with pytest.raises(ValueError, match="differs from its source task"):
        control_request(task.id, changed, tmp_path, verifier.model_dump(mode="json"), None, tmp_path / "output")
