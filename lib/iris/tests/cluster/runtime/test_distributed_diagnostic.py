# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from iris.cluster.runtime.distributed_diagnostic import capture_distributed_diagnostic
from iris.cluster.runtime.profile import ExecResult


class FakeDispatch:
    pyspy_bin = "py-spy"
    memray_bin = "memray"

    def scratch(self, *suffixes):
        raise AssertionError("not used")

    def exec_profiler(self, command, *, sample_timeout):
        return ExecResult(0, b"Thread 1\n  train.py:12\n", "")

    def exec(self, command, *, timeout):
        if command[0] == "python" and command[-2] == "json":
            return ExecResult(
                0,
                b'OK\n{"communicators":[{"hash":"a","ranks":[{"rank":0,"collective_counts":{"AllReduce":2}},{"rank":1,"collective_counts":{"AllReduce":1}}]}]}',
                "",
            )
        if command[0] == "python" and command[-2] == "text":
            return ExecResult(0, b"OK\ntext", "")
        if command[0] == "nvidia-smi":
            return ExecResult(1, b"", "nvidia-smi unavailable")
        return ExecResult(0, b"evidence\n", "")

    def read_file(self, path):
        raise AssertionError("not used")


def test_capture_distributed_diagnostic_preserves_partial_evidence_and_skew():
    bundle = json.loads(capture_distributed_diagnostic(FakeDispatch(), pid="1", source="/job/0", attempt_id=2))

    assert bundle["source"] == "/job/0"
    assert bundle["nccl_ras"]["raw_response"].startswith("OK")
    assert bundle["nccl_ras"]["collective_count_skews"][0]["lagging_ranks"] == [1]
    assert bundle["threads"]["text"].startswith("Thread 1")
    assert bundle["gpu"]["status"] == "unavailable"
    assert bundle["errors"][0]["collector"] == "gpu"
