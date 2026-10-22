"""Parity tests for the self-contained HumanEval grader in eval/graders.

``data/humaneval_grader_benchmark.jsonl`` carries the verdict the pinned
lm-evaluation-harness ``v0.4.12`` produced for each of its 100 problems through
``evaluate.load("code_eval")``. Unlike the MATH and GSM8K tests, the reference is
not re-run by default: ``code_eval`` downloads a metric module from the Hub on
first use and demands ``HF_ALLOW_CODE_EVAL=1``, so recording its verdicts keeps
this suite hermetic. ``scripts/benchmarks/build_humaneval_benchmark.py``
regenerates them from a pinned dataset revision against a pinned ``code_eval``
revision, and ``test_matches_live_reference`` re-checks them against that same
pinned reference when ``HF_ALLOW_CODE_EVAL=1`` is set.

Every candidate runs in its own process, so this file is slower than its
siblings. Most of that is the timeout candidates, which cost a full SIGALRM wait
in the reference and the port alike.
"""

import importlib.util
import json
import os
import pathlib
import sys
import time

import pytest

# Re-running the reference executes model-generated code and pulls a metric
# module from the Hub, so it stays opt-in behind the same switch code_eval uses.
requires_code_eval = pytest.mark.skipif(
    os.environ.get("HF_ALLOW_CODE_EVAL") != "1",
    reason="set HF_ALLOW_CODE_EVAL=1 to re-run the code_eval reference",
)

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import humaneval  # noqa: E402

BENCHMARK = pathlib.Path(__file__).parent / "data" / "humaneval_grader_benchmark.jsonl"
GENERATOR = REPO / "scripts/benchmarks/build_humaneval_benchmark.py"


def _generator_module():
    """Load the fixture generator by path; scripts/ is not an importable package.

    The Hub pins live there and are read from it rather than restated here, so
    the live check can only ever run against the revision the fixture was
    recorded with.
    """
    spec = importlib.util.spec_from_file_location("build_humaneval_benchmark", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A complete, self-contained problem used by the behavior tests below.
PROMPT = "def add(a, b):\n"
TESTS = "def check(candidate):\n    assert candidate(2, 3) == 5\n\ncheck(add)"


@pytest.fixture(scope="module")
def benchmark():
    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    assert len(records) == 100
    return records


def test_matches_harness_on_benchmark(benchmark):
    mismatched = [
        (record["task_id"], record["kind"])
        for record in benchmark
        if humaneval.grade(record["problem"], record["solution"], record["reference_answer"])
        != record["reference_grade"]
    ]
    assert mismatched == []


@requires_code_eval
def test_matches_live_reference(benchmark):
    """Confirm the recorded verdicts still match a real code_eval run."""
    evaluate = pytest.importorskip("evaluate")
    # Pinned to the revision the fixture was generated against; an unpinned load
    # would compare the recording to whatever code_eval the Hub serves today.
    generator = _generator_module()
    code_eval = evaluate.load(generator.CODE_EVAL, revision=generator.CODE_EVAL_REVISION)

    stale = []
    for record in benchmark:
        candidate = record["problem"] + record["solution"]
        live = float(
            code_eval.compute(references=[record["reference_answer"]], predictions=[[candidate]], k=[1])[0]["pass@1"]
        )
        if live != record["reference_grade"]:
            stale.append((record["task_id"], record["kind"], record["reference_grade"], live))
    assert stale == []


def test_passing_candidate_scores_one():
    assert humaneval.grade(PROMPT, "    return a + b\n", TESTS) == 1.0


def test_failing_candidate_scores_zero():
    assert humaneval.grade(PROMPT, "    return a * b\n", TESTS) == 0.0


def test_syntax_error_scores_zero():
    assert humaneval.grade(PROMPT, "    return (\n", TESTS) == 0.0


def test_candidate_is_joined_to_the_prompt():
    """The task's create_test filter concatenates prompt and completion."""
    assert humaneval.build_candidate("def f():\n", "    return 1\n") == "def f():\n    return 1\n"


def test_nonterminating_candidate_times_out():
    outcome = humaneval.check_correctness("while True:\n    pass\n", timeout=0.5)
    assert outcome == humaneval.TIMED_OUT


def test_large_failure_message_is_not_a_denial_of_service():
    """A failure message is candidate-controlled and must not outgrow the pipe.

    Past the pipe buffer the child would block in send(), the join would expire,
    and the parent would kill it mid-write and read a partial frame.
    """
    start = time.perf_counter()
    assert humaneval.grade(PROMPT, "    raise ValueError('x' * 70000)\n", TESTS) == 0.0
    assert time.perf_counter() - start < 1.0


def test_candidate_crash_does_not_take_down_the_grader():
    """Isolation is the point of the child process: a hard exit is just a failure."""
    assert humaneval.check_correctness("import os\nos._exit(1)\n", timeout=1.0) != humaneval.PASSED
    assert humaneval.grade(PROMPT, "    return a + b\n", TESTS) == 1.0


def test_destructive_calls_are_disabled_in_the_candidate():
    """reliability_guard runs before the candidate, so os.system is unavailable."""
    outcome = humaneval.check_correctness("import os\nos.system('true')\n", timeout=1.0)
    assert outcome.startswith("failed:")


def test_candidate_runs_outside_the_repository(tmp_path):
    """The candidate is chdir'd into a scratch directory, so stray writes land there."""
    marker = "sentinel_from_candidate.txt"
    outcome = humaneval.check_correctness(f"open({marker!r}, 'w').write('x')\n", timeout=1.0)
    assert outcome == humaneval.PASSED
    assert not (REPO / marker).exists()
    assert not (pathlib.Path.cwd() / marker).exists()
