# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The callable runner executes a pickled callable exactly once, even under spawn."""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import cloudpickle
from iris.cluster.types import CALLABLE_RUNNER


def test_callable_runner_supports_spawn_process_pools(tmp_path):
    marker = tmp_path / "calls.txt"

    def callable_with_spawn_pool(path: str) -> None:
        with open(path, "a") as f:
            f.write("call\n")
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
            assert list(pool.map(abs, [-1, -2])) == [1, 2]

    (tmp_path / "_callable_runner.py").write_text(CALLABLE_RUNNER)
    (tmp_path / "_callable.pkl").write_bytes(cloudpickle.dumps((callable_with_spawn_pool, (str(marker),), {})))
    result = subprocess.run(
        [sys.executable, "-u", str(tmp_path / "_callable_runner.py")],
        env={**os.environ, "IRIS_WORKDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert marker.read_text() == "call\n"
