# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RST conversion preserves source declarations through execution calibration."""

import base64
import hashlib
import tomllib

import pytest
from taskcompendium.harbor.adapter import declared_execution_limits, source_task_configurations, write_harbor_task
from taskcompendium.models import DockerBuildExecution, Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.datasets.rst import DATASET, REVISION, VERSION, convert_rst


@pytest.mark.parametrize("deadline", [900, None])
def test_rst_source_configuration_and_verifier_survive_calibrated_export(tmp_path, deadline):
    config = (
        b"[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=6144\nbuild_timeout_sec=1200\n[agent]\ntimeout_sec=900\n"
    )
    if deadline is not None:
        config += b"[verifier]\ntimeout_sec=900\n"
    files = {
        "instruction.md": b"Create /app/result.txt with the requested result.\n",
        "task.toml": config,
        "environment/Dockerfile": b"FROM ubuntu:24.04\nWORKDIR /app\n",
        "tests/test.sh": b"#!/bin/bash\npython3 /tests/test_state.py\n",
        "tests/test.sh.uvx_bak": b"#!/bin/bash\nuvx pytest /tests/test_state.py\n",
        "tests/test_state.py": b"from pathlib import Path\nassert Path('/app/result.txt').read_text() == 'done'\n",
        "solution/solve.sh": b"#!/bin/bash\nprintf done > result.txt\n",
    }
    task = convert_rst(
        RawRow(
            id="rst-fixture",
            source=Source(dataset=DATASET, revision=REVISION, row="fixture", importer_revision=VERSION),
            data={
                "archive_links": {},
                "archive_sha256": hashlib.sha256(b"fixture archive").hexdigest(),
                "files": {name: base64.b64encode(value).decode() for name, value in files.items()},
                "file_metadata": {
                    name: {"mode": "0755" if name.endswith(".sh") else "0644", "mtime_ns": 123000000000}
                    for name in files
                },
            },
        ),
        ConversionContext(inputs={}, grader_environment=None),
    )
    assert isinstance(task, TaskSpec)
    assert task.context.events[0].content == files["instruction.md"].decode()
    assert task.environment_requirements.docker_build.files[0].path == "Dockerfile"
    assert resource_bytes(task.environment_requirements.docker_build.files[0]) == files["environment/Dockerfile"]
    assert task.grader.timeout == deadline
    assert task.integrity_contract.agent_user is None
    assert task.integrity_contract.undeclared_deadline == (deadline is None)
    assert {item.name: item.sha256 for item in task.integrity_contract.verifier_files} == {
        name.removeprefix("tests/"): hashlib.sha256(value).hexdigest()
        for name, value in files.items()
        if name.startswith("tests/")
    }
    limits = declared_execution_limits(task)
    assert limits == {
        "cpus": 1,
        "memory_mb": 2048,
        "storage_mb": 6144,
        "agent_timeout_sec": 900,
        **({"verifier_timeout_sec": 900} if deadline is not None else {}),
    }
    shape = INITIAL_REQUIREMENTS.model_copy(update=limits)
    packet = tmp_path / "task"
    write_harbor_task(task, packet, shape, docker_build=DockerBuildExecution.DECLARATIVE)
    assert tomllib.loads((packet / "task.toml").read_text())["environment"]["build_timeout_sec"] == 1200
    calibrated = calibrated_task_spec(task, shape, None, (packet / "task.toml").read_bytes())
    assert tuple(resource_bytes(resource) for resource in source_task_configurations(calibrated)) == (config,)
    assert calibrated.grader.timeout == deadline
    assert calibrated.integrity_contract == task.integrity_contract
    assert calibrated.resources.oracle == task.resources.oracle
