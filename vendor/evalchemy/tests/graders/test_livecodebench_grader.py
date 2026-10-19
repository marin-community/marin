# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Contract and regression coverage for the bounded LiveCodeBench grader in ``eval.graders``.

Covers the issue #147 bounds the four vendored stacks lack:

- a memory cap wired by default (``RLIMIT_AS``/``RLIMIT_DATA``/``RLIMIT_STACK``);
- a wall-clock deadline independent of the test count, in addition to the per-test alarm;
- deterministic child reaping -- no child or ``multiprocessing.Manager`` process survives
  a timeout;
- child identification at spawn (pid, test count, deadline).

The memory-cap *enforcement* tests are Linux-only: macOS cannot lower ``RLIMIT_AS``, so
there the cap degrades to best-effort and the cross-platform bound is the wall-clock
deadline plus kill. The "cap is wired" tests run everywhere (they only check the default
is non-None).
"""

import concurrent.futures
import inspect
import multiprocessing
import pathlib
import platform
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The test directory is on ``sys.path`` via pytest, so the shared helpers here are importable.
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval.runtime.resources import ExecutionMode  # noqa: E402
from eval.graders import livecodebench  # noqa: E402
from lcb_test_support import living_children  # noqa: E402

# macOS cannot lower RLIMIT_AS (setrlimit raises), so the cap is not enforced there;
# the enforcement tests need a platform that honors it.
LINUX = platform.system() == "Linux"


@pytest.fixture
def assert_no_child_leak():
    """Snapshot this process's children; fail on teardown if any were left running."""
    before = living_children()
    yield
    time.sleep(0.2)  # allow a killed child's join-grace to settle before we measure
    leaked = living_children() - before
    assert not leaked, f"grader leaked child process(es): {leaked}"


def test_module_does_not_import_a_vendored_benchmark_module():
    """Importing the grader pulls in no chat_benchmarks / LiveBench code.

    The RL reward path imports ``eval.graders.livecodebench`` and must not load the vendored
    LCB stacks as a side effect. A fresh subprocess is a clean interpreter, so the assertion
    is independent of what the rest of the session has already imported.
    """
    script = (
        "import sys, eval.graders.livecodebench as g;"
        "bad=[n for n in sys.modules if n=='pyext' or n.startswith(('livebench','eval.chat_benchmarks.Live'))];"
        "assert g.run is not None and not bad, bad"
    )
    probe = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(REPO))
    assert probe.returncode == 0, probe.stderr


def test_grader_declares_process_isolated_execution():
    assert livecodebench.GRADER_EXECUTION_MODE is ExecutionMode.PROCESS_ISOLATED


def test_memory_cap_is_wired_by_default():
    """Pre-#147 the cap default was ``None`` so the rlimit was never applied. This is the
    direct regression: the parameter defaults to a non-None cap so a caller who never
    passes one still gets a cap."""
    defaults = inspect.signature(livecodebench.run).parameters["maximum_memory_bytes"].default
    assert defaults is not None, "run() must wire the memory cap by default"


def _grader_pass_in_child():
    """Module-level so it is picklable; the RL path grades inside spawned env workers."""
    from eval.graders import livecodebench

    return livecodebench.run(
        [{"input": [1, 2], "output": 3}],
        "def add(a, b):\n    return a + b\n",
        fn_name="add",
    )["passed"]


def test_grader_is_spawning_and_safe_to_import_in_a_child():
    """The RL path grades per-trajectory inside a spawned env worker. That worker itself
    launches the bounded child, so the grader must spawn cleanly from inside a
    ProcessPoolExecutor (the contract's ``execute_grading_jobs`` dispatcher, whose workers
    -- unlike daemon ``multiprocessing.Pool`` workers -- may spawn children)."""
    context = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        assert executor.submit(_grader_pass_in_child).result() is True


# --- scoring semantics (the contract a binary/fractional reward is computed from) ---


def _one_functional(input_value, output_value):
    return [{"input": input_value, "output": output_value}]


def test_passing_functional_candidate_scores_all_pass():
    result = livecodebench.run(
        _one_functional([1, 2], 3),
        "def add(a, b):\n    return a + b\n",
        fn_name="add",
    )
    assert result["passed"] is True
    assert result["num_passed"] == 1
    assert result["num_tests"] == 1
    assert result["timed_out"] is False


@pytest.mark.parametrize(
    ("completion", "expected_accuracy"),
    [
        ("```python\ndef add(a, b):\n    return a + b\n```", 1.0),
        ("```python\ndef add(a, b):\n    return a + b\n```\n```python\n\n```", 0.0),
    ],
)
def test_live_code_scores_an_empty_final_code_block_as_incorrect(completion, expected_accuracy):
    from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
    from eval.graders.programs import live_code

    spec = EvalSpec(
        task="LiveCodeBench", source=SourceRef("fixture"), source_id="functional", ordinal=0,
        requests=(), grader=GraderSpec("live_code"),
        doc={"test": [{"input": [1, 2], "output": 3, "testtype": "functional"}]},
    )

    assert live_code(spec, [completion]).metrics == {"accuracy": expected_accuracy}


@pytest.mark.parametrize("comparison", ["exact", "stdout_normalized"])
@pytest.mark.parametrize(
    ("completion", "expected_accuracy"),
    [
        ('if __name__ == "__main__":\n    print(input())', 1.0),
        ('if __name__ == "__main__":\n    print(float(input()))', 0.0),
        ("if __name__ == '__main__':\n    print(input())", 0.0),
    ],
)
def test_competition_programs_preserve_original_main_guard(comparison, completion, expected_accuracy):
    from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
    from eval.graders.programs import program

    spec = EvalSpec(
        task="competition", source=SourceRef("fixture"), source_id="echo", ordinal=0,
        requests=(), grader=GraderSpec("program", reference={
            "tests": [{"input": "2", "output": "2"}], "stdin": True,
            "timeout": 1, "comparison": comparison,
        }),
    )

    assert program(spec, [f"```python\n{completion}\n```"]).metrics == {"accuracy": expected_accuracy}


def test_failing_candidate_scores_failed_and_short_circuits():
    """Stop-on-failure: a wrong answer stops the run without running the later test."""
    result = livecodebench.run(
        [{"input": [1, 2], "output": 999}, {"input": [1, 2], "output": 3}],
        "def add(a, b):\n    return a + b\n",
        fn_name="add",
    )
    assert result["passed"] is False
    assert result["num_passed"] == 0
    assert result["outcomes"][0]["outcome"] == "failed"


def test_collect_all_runs_remaining_tests_after_a_failure():
    result = livecodebench.run(
        [{"input": [1, 2], "output": 999}, {"input": [1, 2], "output": 3}],
        "def add(a, b):\n    return a + b\n",
        fn_name="add",
        collect_all=True,
    )
    assert result["passed"] is False
    assert result["num_passed"] == 1
    assert [t["outcome"] for t in result["outcomes"]] == ["failed", "passed"]


@pytest.mark.parametrize("count,completion,expected_passed", [
    (4096, "def identity(value):\n    return value\n", True),
    (16, "def identity(value):\n    raise ValueError('x' * 8192)\n", False),
])
def test_all_test_verdicts_survive_pipe_backpressure(count, completion, expected_passed):
    result = livecodebench.run(
        [{"input": [index], "output": index} for index in range(count)],
        completion, fn_name="identity", collect_all=True, deadline=3,
    )

    assert result["passed"] is expected_passed
    assert result["timed_out"] is False
    assert len(result["outcomes"]) == count
    assert result["num_passed"] == (count if expected_passed else 0)


def test_stdin_mode_feeds_input_and_compares_stdout():
    result = livecodebench.run(
        [{"input": "1 2\n", "output": "3\n"}],
        "a, b = input().split()\nprint(int(a) + int(b))\n",
        stdin_mode=True,
    )
    assert result["passed"] is True


def test_stdout_normalized_comparison_preserves_codeforces_line_handling():
    tests = [{"input": "", "output": "1 2"}]
    completion = "print(1)\nprint(2)\n"
    assert livecodebench.run(tests, completion, stdin_mode=True, comparison="stdout_normalized")["passed"] is True
    assert livecodebench.run(tests, completion, stdin_mode=True, comparison="exact")["passed"] is False


def test_fn_name_overrides_inferred_name():
    """The caller-supplied fn_name is the function to call, not the inferred one."""
    completion = "def helper(a, b):\n    return a + b\n"
    result = livecodebench.run(_one_functional([3, 4], 7), completion, fn_name="helper")
    assert result["passed"] is True


def test_grade_returns_fraction_of_passed_tests():
    """MarinSkyRL computes binary and fractional rewards from per-test results."""
    assert livecodebench.grade(_one_functional([1, 2], 3), "def add(a, b):\n    return a + b\n", fn_name="add") == 1.0
    assert livecodebench.grade(_one_functional([1, 2], 999), "def add(a, b):\n    return a + b\n", fn_name="add") == 0.0


# --- bounds: the four things the vendored stacks lack ---


def test_per_test_alarm_bounds_a_hung_test(assert_no_child_leak):
    """A single wedged test is cut off at its alarm, not left to the whole-budget join."""
    start = time.perf_counter()
    result = livecodebench.run(
        [{"input": [0], "output": 0}],
        "import time\ndef hang(a):\n    time.sleep(30)\n    return a\n",
        fn_name="hang",
        test_timeout=1.0,
        deadline=12.0,
    )
    elapsed = time.perf_counter() - start
    assert result["passed"] is False
    assert result["timed_out"] is True
    assert elapsed < 8.0, f"a hung test should be cut off near its 1s alarm, not at {elapsed:.1f}s"


def test_wall_clock_deadline_fires_below_test_count_formula(assert_no_child_leak):
    """The deadline is independent of the test count: the old join was (timeout+1)*N+5."""
    n = 40
    tests = [{"input": [i], "output": i} for i in range(n)]
    # Each test sleeps 0.15s (well under test_timeout) so the per-test alarm never fires;
    # only the wall-clock deadline can stop the run. total = n * 0.15s = 6s > deadline.
    completion = "import time\ndef f(a):\n    time.sleep(0.15)\n    return a\n"
    deadline = 2.0
    test_timeout = 10.0
    old_formula = (test_timeout + 1) * n + 5

    start = time.perf_counter()
    result = livecodebench.run(tests, completion, fn_name="f", test_timeout=test_timeout, deadline=deadline)
    elapsed = time.perf_counter() - start

    assert result["passed"] is False, "the deadline must score the run failed"
    assert result["timed_out"] is True
    assert result["num_passed"] < n, "partial work before the deadline must be visible"
    assert elapsed < old_formula, f"must finish far below the test-count formula ({old_formula:.0f}s)"
    assert elapsed < 5.0, f"deadline is {deadline}s, took {elapsed:.1f}s"


def test_timeout_child_is_deterministically_reaped_and_scores_failed(assert_no_child_leak):
    start = time.perf_counter()
    result = livecodebench.run(
        [{"input": [0], "output": 0}],
        "import time\ndef hang(a):\n    while True:\n        time.sleep(1)\n",
        fn_name="hang",
        test_timeout=0.5,
        deadline=1.5,
    )
    elapsed = time.perf_counter() - start
    assert result["passed"] is False
    assert result["timed_out"] is True
    assert elapsed < 4.0
    # the assert_no_child_leak fixture guarantees no direct child is left running.
    assert result["child_pid"] is not None, "the grader must report the child pid it spawned"


def test_child_allocation_identifies_its_pid():
    """The run reports the pid of the child it spawned, so a runaway is attributable.

    The bounded grader returns ``child_pid`` as its public contract (the RL reward path
    uses it to attribute a runaway); the test count is likewise part of the result.
    """
    result = livecodebench.run(_one_functional([1, 2], 3), "def add(a,b):\n return a+b", fn_name="add")
    assert isinstance(result["child_pid"], int) and result["child_pid"] > 0
    assert result["num_tests"] == 1


def test_candidate_crash_does_not_take_down_the_grader(assert_no_child_leak):
    """A hard exit (os._exit) in the child is just a failure, not a grader crash."""
    assert livecodebench.run(
        _one_functional([1, 2], 3),
        "import os\ndef add(a,b):\n    os._exit(3)\n",
        fn_name="add",
    )["passed"] is False
    # the grader still works for a follow-up
    assert livecodebench.run(_one_functional([1, 2], 3), "def add(a,b):\n return a+b", fn_name="add")["passed"] is True


def test_destructive_calls_are_disabled(assert_no_child_leak):
    outcome = livecodebench.run(
        _one_functional([1], 1),
        "import os\ndef f(a):\n    os.system('true')\n    return a\n",
        fn_name="f",
    )
    assert outcome["passed"] is False
    assert outcome["outcomes"][0]["outcome"] == "failed"


def test_large_failure_output_is_not_a_denial_of_service():
    """A candidate-controlled failure output must not outgrow the pipe and wedge the child."""
    start = time.perf_counter()
    assert livecodebench.run(
        _one_functional([1, 2], 3),
        "def add(a, b):\n    return 'x' * 70000\n",
        fn_name="add",
    )["passed"] is False
    assert time.perf_counter() - start < 1.5


@pytest.mark.skipif(not LINUX, reason="macOS cannot lower RLIMIT_AS; the cap is not enforced there")
def test_child_allocating_past_the_cap_is_killed_and_scores_failed(assert_no_child_leak):
    cap = 128 * 1024 * 1024
    result = livecodebench.run(
        [{"input": [0], "output": 0}],
        "def boom(a):\n    return bytearray(512 * 1024 * 1024)\n",
        fn_name="boom",
        maximum_memory_bytes=cap,
    )
    assert result["passed"] is False
    assert result["outcomes"][0]["outcome"] in (livecodebench.FAILED, livecodebench.TIMED_OUT)
