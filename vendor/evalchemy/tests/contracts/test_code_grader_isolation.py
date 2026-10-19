# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for functional-correctness sandbox launch isolation."""

import json
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest



def _record_worker_process(task_id, sample, _language, _timeout, _tmp_dir, completion_id):
    Path(sample["pid_file"]).write_text(str(os.getpid()))
    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "passed": True,
    }


def _write_evaluation_fixture(tmp_path, is_mbpp):
    problem = {
        "task_id": "python/0",
        "prompt": "def answer():\n",
        "test": ["assert answer() == 42"] if is_mbpp else "assert answer() == 42",
    }
    sample = {
        "task_id": "python/0",
        "prompt": problem["prompt"],
        "generation": "def answer():\n    return 42",
    }
    problem_file = tmp_path / "problems.jsonl"
    input_file = tmp_path / "samples.jsonl"
    problem_file.write_text(json.dumps(problem) + "\n")
    input_file.write_text(json.dumps(sample) + "\n")
    return problem_file, input_file, sample


_EVALUATION_MODULES = [
    ("eval.chat_benchmarks.HumanEvalPlus.human_eval_plus.evaluation", False),
    ("eval.chat_benchmarks.MBPPPlus.mbpp_plus.evaluation", True),
]
_PLAIN_TEST_CODE = "def answer():\n    return 42\nassert answer() == 42"
# The evalplus test harness (HumanEvalPlus and MBPPPlus data) starts every test with this
# import; numpy's first import writes os.environ, which the sandbox's reliability_guard disables.
_NUMPY_TEST_CODE = "import numpy as np\n" + _PLAIN_TEST_CODE


def _run_check_in_spawned_worker(check_correctness, tmp_path, is_mbpp, test_code):
    _problem_file, _input_file, sample = _write_evaluation_fixture(tmp_path, is_mbpp)
    sample["test_code"] = test_code
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


@pytest.mark.parametrize(("module_path", "is_mbpp"), _EVALUATION_MODULES)
def test_functional_correctness_checks_run_outside_evaluator_process(
    monkeypatch,
    tmp_path,
    module_path,
    is_mbpp,
):
    module = __import__(module_path, fromlist=["evaluation"])
    pid_file = tmp_path / "worker.pid"
    problem_file, input_file, sample = _write_evaluation_fixture(tmp_path, is_mbpp)
    sample["pid_file"] = str(pid_file)
    input_file.write_text(json.dumps(sample) + "\n")
    monkeypatch.setattr(module, "check_correctness", _record_worker_process)

    result = module.evaluate_functional_correctness(
        input_file=str(input_file),
        tmp_dir=str(tmp_path),
        n_workers=1,
        problem_file=str(problem_file),
        is_mbpp=is_mbpp,
        k=[1],
    )

    assert result["scored_count"] == 1
    assert "pass@1" in result
    assert int(pid_file.read_text()) != os.getpid()


@pytest.mark.parametrize(("module_path", "is_mbpp"), _EVALUATION_MODULES)
@pytest.mark.parametrize("test_code", [_PLAIN_TEST_CODE, _NUMPY_TEST_CODE], ids=["plain", "imports_numpy"])
def test_spawned_worker_can_launch_functional_correctness_sandbox(tmp_path, module_path, is_mbpp, test_code):
    module = __import__(module_path, fromlist=["evaluation"])
    result = _run_check_in_spawned_worker(module.check_correctness, tmp_path, is_mbpp, test_code)

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
