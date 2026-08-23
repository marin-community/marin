# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import Entrypoint, JobName
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV, MultiGpuHook

from levanter import distributed
from levanter.distributed import _square_brace_expand


def _run_with_iris_completion(marker: Path, outcome: str) -> None:
    class RecordingClient:
        def complete_job(self, _job_id: JobName) -> None:
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


def _run_supervised_iris_rank(output_dir: Path, failing_rank: int | None) -> None:
    process_index = int(os.environ[IRIS_MULTIGPU_PROCESS_INDEX_ENV])
    release_fifo = output_dir / "release-failure"

    def release_failure() -> None:
        if failing_rank is not None and process_index == 0:
            with release_fifo.open("w") as stream:
                stream.write("1")

    class RecordingClient:
        def complete_job(self, _job_id: JobName) -> None:
            (output_dir / "completed").touch()
            release_failure()

    def shutdown() -> None:
        (output_dir / f"shutdown-{process_index}").touch()
        release_failure()

    distributed.get_job_info = lambda: JobInfo(task_id=JobName.from_wire("/test/training/0"))
    distributed.configure_megascale_from_iris = lambda: None
    distributed.initialize_iris_jax = lambda: None
    distributed.iris_ctx = lambda: SimpleNamespace(client=RecordingClient())
    distributed.jax.process_index = lambda: process_index
    distributed.jax.distributed.shutdown = shutdown

    distributed.DistributedConfig().initialize()
    if process_index == failing_rank:
        with release_fifo.open("r") as stream:
            assert stream.read(1) == "1"
        os.kill(os.getpid(), signal.SIGKILL)


def _run_supervised_callable(tmp_path: Path, failing_rank: int | None) -> subprocess.CompletedProcess:
    os.mkfifo(tmp_path / "release-failure")
    entrypoint = Entrypoint.from_callable(_run_supervised_iris_rank, tmp_path, failing_rank)
    for name, contents in entrypoint.workdir_files.items():
        (tmp_path / name).write_bytes(contents)

    command = MultiGpuHook(nproc=2).wrap(entrypoint.command)
    env = {**os.environ, "IRIS_PYTHON": sys.executable, "IRIS_WORKDIR": str(tmp_path)}
    return subprocess.run(command, env=env, check=False)


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


def test_multigpu_clean_teardown_exits_zero_after_every_rank_shuts_down(tmp_path: Path) -> None:
    result = _run_supervised_callable(tmp_path, failing_rank=None)

    assert result.returncode == 0
    assert not (tmp_path / "completed").exists()
    assert {path.name for path in tmp_path.glob("shutdown-*")} == {"shutdown-0", "shutdown-1"}


def test_multigpu_late_rank_sigkill_does_not_complete_iris_job(tmp_path: Path) -> None:
    result = _run_supervised_callable(tmp_path, failing_rank=1)

    assert result.returncode != 0
    assert not (tmp_path / "completed").exists()
    assert {path.name for path in tmp_path.glob("shutdown-*")} == {"shutdown-0"}


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
