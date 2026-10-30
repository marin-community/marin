# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for functional-correctness sandbox launch isolation."""

import json
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest


_EVALUATION_MODULES = [
    "eval.chat_benchmarks.HumanEvalPlus.human_eval_plus.execution",
    "eval.chat_benchmarks.MBPPPlus.mbpp_plus.execution",
]
_PLAIN_TEST_CODE = "def answer():\n    return 42\nassert answer() == 42"
# The evalplus test harness (HumanEvalPlus and MBPPPlus data) starts every test with this
# import; numpy's first import writes os.environ, which the sandbox's reliability_guard disables.
_NUMPY_TEST_CODE = "import numpy as np\n" + _PLAIN_TEST_CODE


def _run_check_in_spawned_worker(check_correctness, tmp_path, test_code):
    sample = {"test_code": test_code}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        return executor.submit(
            check_correctness,
            "python/0",
            sample,
            "python",
            1.0,
            str(tmp_path),
            0,
        ).result()


@pytest.mark.parametrize("module_path", _EVALUATION_MODULES)
@pytest.mark.parametrize("test_code", [_PLAIN_TEST_CODE, _NUMPY_TEST_CODE], ids=["plain", "imports_numpy"])
def test_spawned_worker_can_launch_functional_correctness_sandbox(tmp_path, module_path, test_code):
    module = __import__(module_path, fromlist=["evaluation"])
    result = _run_check_in_spawned_worker(module.check_correctness, tmp_path, test_code)

    assert result["passed"], result["result"]


@pytest.mark.skipif(sys.platform != "linux", reason="Native address-space limits are enforced on Linux")
def test_shared_grader_pool_preserves_native_memory_budget_after_cli_and_source_imports(tmp_path):
    output = tmp_path / "native-grades.json"
    result = subprocess.run(
        [sys.executable, "-m", "tests.contracts.code_grader_pool_control", str(output)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == [
        ["lcb_initial_True", 1.0], ["lcb_initial_False", 0.0],
        ["humaneval_42", 1.0], ["humaneval_41", 0.0],
        ["livebench_amps_42", 100.0], ["livebench_amps_41", 0.0],
        ["lcb_after_imports_stdin_True", 1.0], ["lcb_after_imports_stdin_False", 0.0],
        ["lcb_after_imports_functional_True", 1.0], ["lcb_after_imports_functional_False", 0.0],
        ["livebench_coding_True", 100.0], ["livebench_coding_False", 0.0],
    ]
