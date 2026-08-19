# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import Entrypoint, JobName

from levanter import distributed
from levanter.distributed import _square_brace_expand


def _run_with_iris_completion(marker: Path, outcome: str) -> None:
    class RecordingClient:
        def complete(self, _job_id: JobName) -> None:
            marker.touch()

    distributed.get_job_info = lambda: JobInfo(task_id=JobName.from_wire("/test/training/0"))
    distributed.configure_megascale_from_iris = lambda: None
    distributed.initialize_iris_jax = lambda: None
    distributed.iris_ctx = lambda: SimpleNamespace(client=RecordingClient())
    distributed.jax.process_index = lambda: 0

    distributed.DistributedConfig().initialize()
    if outcome == "exception":
        raise RuntimeError("training failed")
    if outcome == "system-exit":
        raise SystemExit(7)


@pytest.mark.parametrize(
    ("outcome", "expected_returncode", "expected_completion"),
    [
        pytest.param("return", 0, True, id="clean-return"),
        pytest.param("exception", 1, False, id="uncaught-exception"),
        pytest.param("system-exit", 7, False, id="nonzero-system-exit"),
    ],
)
def test_callable_runner_only_completes_iris_job_after_clean_exit(
    tmp_path: Path,
    outcome: str,
    expected_returncode: int,
    expected_completion: bool,
) -> None:
    marker = tmp_path / "completed"
    entrypoint = Entrypoint.from_callable(_run_with_iris_completion, marker, outcome)
    for name, contents in entrypoint.workdir_files.items():
        (tmp_path / name).write_bytes(contents)

    env = {**os.environ, "IRIS_WORKDIR": str(tmp_path)}
    result = subprocess.run([sys.executable, tmp_path / "_callable_runner.py"], env=env, check=False)

    assert result.returncode == expected_returncode
    assert marker.exists() is expected_completion


def test_square_brace_expand():
    custom_sequence = "node[001-004,007]suffix"
    expanded_nodes = _square_brace_expand(custom_sequence)
    assert expanded_nodes == ["node001suffix", "node002suffix", "node003suffix", "node004suffix", "node007suffix"]

    custom_sequence_2 = "prefix[001-002]node[005-006]suffix"
    expanded_nodes_2 = _square_brace_expand(custom_sequence_2)
    assert expanded_nodes_2 == [
        "prefix001node005suffix",
        "prefix001node006suffix",
        "prefix002node005suffix",
        "prefix002node006suffix",
    ]

    custom_sequence_3 = "node[1-11]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)]

    custom_sequence_3 = "node[1-11,21]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)] + ["node21suffix"]
