# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.cli_gym import score, tasks


def source_row(yaml):
    return RawRow(
        "task",
        Source(dataset="CLI-Gym", revision="pin", row="task", importer_revision="test"),
        {
            "task_id": "task",
            "task_yaml": yaml,
            "dockerfile": "FROM source@sha256:" + "a" * 64 + "\n",
            "docker_compose": (
                """services:
  CLIGYM:
    build: {context: ., dockerfile: Dockerfile}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: [sh, -c, sleep infinity]
    environment: ['TEST_DIR=${T_BENCH_TEST_DIR}']
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
"""
            ),
            "run_tests": "#!/bin/bash\nprintf 'original test output'\nexit 0\n",
        },
    )


def test_cli_keeps_source_test_bytes_private_and_deadlines_declared():
    row = source_row(
        """instruction: Repair the environment
parser_name: pytest
max_agent_timeout_sec: 3000
max_test_timeout_sec: 3000
run_tests_in_same_shell: false
"""
    )
    task = tasks.convert_cli(row, ConversionContext({}, None))
    assert not isinstance(task, ImportRejection)
    assert not task.resources.oracle and task.execution_requirements is None
    assert task.grader.timeout == 3000
    private = {r.path: resource_bytes(r) for r in task.resources.verifier}
    assert private["source_archive/run-tests.sh"] == row.data["run_tests"].encode()
    assert resource_bytes(task.environment_requirements.docker_build.files[0]) == row.data["dockerfile"].encode()
    assert task.integrity_contract.verifier_files[0].sha256 == hashlib.sha256(row.data["run_tests"].encode()).hexdigest()
    assert task.grader.collect[-1].cwd is None  # Keep the native image WORKDIR.
    assert task.grader.artifacts[0].source == tasks.LOG
    assert task.grader.argv[4] == "pytest"


def test_malformed_source_yaml_retains_raw_identity_as_typed_rejection():
    row = source_row("instruction: |\n  valid line\n bad indentation\n")
    result = tasks.convert_cli(row, ConversionContext({}, None))
    assert isinstance(result, ImportRejection)
    assert result.kind.value == "source_defect" and result.reason == "malformed_task_yaml"
    assert json.loads(result.detail)["yaml_sha256"] == hashlib.sha256(row.data["task_yaml"].encode()).hexdigest()


@pytest.mark.parametrize(
    "parser,output,reward",
    [
        ("pytest", "=== short test summary info ===\nSKIPPED a.py::skip\nXFAIL a.py::xfail\n", 1.0),
        ("pytest", "=== short test summary info ===\nXPASS a.py::xpass\n", 0.0),
        ("pytest", "=== short test summary info ===\n", 1.0),
        ("swebench", "SWEBench results starts here\nPASSED\nSWEBench results ends here", 1.0),
        ("swebench", "SWEBench results starts here\nFAILED\nSWEBench results ends here", 0.0),
    ],
)
def test_original_parser_semantics_override_test_script_exit_status(tmp_path, parser, output, reward):
    score.materialize_upstream(Path(tasks.__file__).with_name("upstream.json"), tmp_path)
    assert score.score(parser, output, tmp_path)[0] == reward


@pytest.mark.parametrize("parser", ["pytest", "swebench"])
def test_missing_source_markers_remain_parser_errors(tmp_path, parser):
    score.materialize_upstream(Path(tasks.__file__).with_name("upstream.json"), tmp_path)
    with pytest.raises(ValueError):
        score.score(parser, "test command exited zero but emitted no source result", tmp_path)
