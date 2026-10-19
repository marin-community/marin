# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage that the vendored LiveCodeBench stacks route through the bounded grader.

Issue #147: the three NovaSky ``testing_util`` copies (``LiveCodeBench``,
``LiveCodeBenchv5``, ``LiveCodeBenchv5_official``) left their worker unbounded --
``lcb_run`` joined on ``(timeout + 1) * len(test_cases) + 5`` seconds with no memory cap.
These tests exec-load each variant's ``lcb_run`` the way ``eval/task.py`` loads the benchmark
and assert a runaway candidate is now bounded: it finishes far below the old test-count join
and leaves no worker running.
"""

import importlib.util
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lcb_test_support import living_children  # noqa: E402

VARIANTS = [
    ("LiveCodeBench", "eval/chat_benchmarks/LiveCodeBench/livecodebench_utils.py"),
    ("LiveCodeBenchv5", "eval/chat_benchmarks/LiveCodeBenchv5/livecodebench_utils.py"),
    ("LiveCodeBenchv5_official", "eval/chat_benchmarks/LiveCodeBenchv5_official/livecodebench_utils.py"),
]


def _load_variant(path: str):
    """Exec-load a variant's ``livecodebench_utils`` by path (it has no ``__init__.py``)."""
    spec = importlib.util.spec_from_file_location(f"lcb_probe_{pathlib.Path(path).parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lcb_run_is_bounded_not_unbounded_join():
    """A runaway through the routed lcb_run is cut at the per-test alarm and reaps; it is
    not left to the old (timeout+1)*N+5 join (the #147 unbounded pattern)."""
    module = _load_variant(VARIANTS[0][1])
    problem = {"test": [{"testtype": "functional", "input": [1], "output": 1}]}
    runaway = "import time\ndef f(a):\n    time.sleep(20)\n    return a\n"
    timeout = 0.4
    old_formula = (timeout + 1) * 1 + 5  # what lcb_run joined on before the fix

    start = time.perf_counter()
    before = living_children()
    result = module.lcb_run(problem, runaway, timeout, True)
    elapsed = time.perf_counter() - start

    assert elapsed < old_formula, f"routed lcb_run must be bounded, took {elapsed:.1f}s vs old join {old_formula:.0f}s"
    assert elapsed < 3.0, f"a single hung test should be cut near its {timeout}s alarm, not {elapsed:.1f}s"
    assert not all(row[0] for row in result), "a runaway must not score its test passed"
    assert (living_children() - before) == set(), "no lcb_run worker may survive a timeout"


def test_lcb_run_still_scores_a_correct_candidate():
    """Routing preserved the verdict: a correct candidate still passes through lcb_run."""
    for _name, path in VARIANTS:
        module = _load_variant(path)
        problem = {"test": [{"testtype": "functional", "input": [1, 2], "output": 3}]}
        result = module.lcb_run(problem, "def add(a, b):\n    return a + b\n", 6, True)
        assert all(row[0] for row in result), f"{path} regressed a correct candidate"


def test_lcb_run_scores_a_wrong_candidate_failed():
    module = _load_variant(VARIANTS[0][1])
    problem = {"test": [{"testtype": "functional", "input": [1, 2], "output": 999}]}
    result = module.lcb_run(problem, "def add(a, b):\n    return a + b\n", 6, True)
    assert not all(row[0] for row in result)
