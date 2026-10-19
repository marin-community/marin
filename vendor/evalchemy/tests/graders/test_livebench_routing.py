# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage that the ``LiveBench/lcb_runner`` stack is bounded.

Issue #147: ``compute_code_generation_metrics.check_correctness`` wrapped ``run_test`` in a
``multiprocessing.Manager`` plus a ``Process`` and joined on
``(timeout + 1) * len(inputs) + 5`` -- unbounded in both memory and time. The rewrite uses a
pipe result transport with a parent-side wall-clock watchdog and deterministic reap, so no
Manager server or worker survives a timeout.

These tests need the ``livebench`` extra (``pyext``); ``pytest.importorskip`` keeps them
runnable in a full env and skipped (not failing) in the lean ``graders`` CI job, which runs
``tests/graders/`` with ``--extra dev`` only. Same opt-in shape as the HumanEval reference tests.
"""

import importlib
import json
import pathlib
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytest.importorskip("pyext", reason="LiveBench code-exec grading needs the `livebench` extra")

from lcb_test_support import living_children  # noqa: E402


def _compute_module():
    sys.path.insert(0, str(REPO / "eval" / "chat_benchmarks" / "LiveBench"))
    return importlib.import_module("livebench.lcb_runner.evaluation.compute_code_generation_metrics")


def test_livebench_check_correctness_is_bounded():
    """A runaway through LiveBench's routed ``check_correctness`` is cut to the deadline and reaps."""
    module = _compute_module()
    sample = {"input_output": json.dumps({"inputs": ["[1]"], "outputs": ["[0]"], "fn_name": "hang"})}
    runaway = "import time\ndef hang(a):\n    while True:\n        time.sleep(1)\n"

    start = time.perf_counter()
    before = living_children()
    module.check_correctness(sample, runaway, timeout=1, debug=False, deadline=5)
    elapsed = time.perf_counter() - start

    assert elapsed < 8.0, f"runaway should be cut at the 5s deadline, took {elapsed:.1f}s"
    time.sleep(0.3)  # allow the join-grace to settle before we measure
    assert (living_children() - before) == set(), "no LiveBench worker may survive a timeout"
