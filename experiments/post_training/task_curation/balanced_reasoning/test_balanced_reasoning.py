# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the balanced-reasoning lane's pure modules."""


import pytest

from experiments.post_training.task_curation.balanced_reasoning.bands import (
    EVAL_OUTPUT_CAP,
    Band,
    BandBudget,
    band_for,
    budget_for,
)
from experiments.post_training.task_curation.balanced_reasoning.difficulty import (
    SOLVER_ALL,
    LedgerAttempt,
    TaskDifficulty,
    parse_ledger_row,
    tally_attempts,
)
from experiments.post_training.task_curation.balanced_reasoning.graceful import (
    BEST_EFFORT_MARKER,
    CLOSING_MARKERS,
    UNCERTAIN_MARKER,
    check_graceful_trace,
    graceful_convergence_prompt,
)
from experiments.post_training.task_curation.balanced_reasoning.selection import (
    CandidateTask,
    Quota,
    select_tasks,
    selection_digest,
)
from experiments.post_training.task_curation.balanced_reasoning.style import style_report


def test_band_edges() -> None:
    assert band_for(0.0) is Band.UNSOLVED
    assert band_for(0.29) is Band.HARD
    assert band_for(0.30) is Band.MEDIUM
    assert band_for(0.70) is Band.EASY
    assert band_for(1.0) is Band.EASY
    with pytest.raises(ValueError, match="within"):
        band_for(-0.1)


def test_budgets_respect_the_eval_cap() -> None:
    for budget in (budget_for(band) for band in Band):
        assert budget.cap_tokens <= EVAL_OUTPUT_CAP
        assert budget.attempts >= 1
    with pytest.raises(ValueError, match="exceeds the eval output cap"):
        BandBudget(Band.HARD, None, 0.30, (0, 1), EVAL_OUTPUT_CAP + 1, 1)


def test_parse_ledger_row_validates() -> None:
    row = {
        "source_id": "src",
        "task_sha256": "ab" * 32,
        "solver": "hosted",
        "outcome": "scored",
        "reward": 1.0,
        "attempt_id": "a1",
        "task_id": "t1",
    }
    attempt = parse_ledger_row(row)
    assert attempt.reward == 1.0
    with pytest.raises(ValueError, match="missing a usable"):
        parse_ledger_row({**row, "source_id": ""})
    with pytest.raises(ValueError, match="non-numeric reward"):
        parse_ledger_row({**row, "reward": "high"})


def test_tally_deduplicates_attempts_and_filters_solver() -> None:
    attempts = [
        LedgerAttempt("src", "ab" * 32, "hosted", "scored", 1.0, "a1", task_id="t1"),
        LedgerAttempt("src", "ab" * 32, "hosted", "scored", 0.0, "a1", task_id="t1"),  # replayed row
        LedgerAttempt("src", "ab" * 32, "hosted", "scored", 0.0, "a2", task_id="t1"),
        LedgerAttempt("src", "ab" * 32, "hosted", "infrastructure", None, "a3", task_id="t1"),
        LedgerAttempt("src", "ab" * 32, "small", "scored", 1.0, "a4", task_id="t1"),
        LedgerAttempt("src", "cd" * 32, "hosted", "scored", 1.0, "a5", task_id="t2"),
    ]
    tasks = tally_attempts(attempts)
    first = tasks[("src", "ab" * 32)]
    assert first.scored == 2
    assert first.solve_rate == 0.5
    assert first.band is Band.MEDIUM
    second = tasks[("src", "cd" * 32)]
    assert second.solve_rate == 1.0
    assert second.band is Band.EASY


def test_tally_skips_source_level_rows() -> None:
    """Measurement and cost receipts carry no task identity and never fold into rates."""
    attempts = [
        LedgerAttempt("src", None, "large", "cost", None, "r1"),
        LedgerAttempt("src", "ab" * 32, "hosted", "scored", 1.0, "a1", task_id="t1"),
    ]
    tasks = tally_attempts(attempts, solver=SOLVER_ALL)
    assert set(tasks) == {("src", "ab" * 32)}
    row = {
        "source_id": "src",
        "solver": "large",
        "outcome": "cost",
        "attempt_id": "r1",
        "task_sha256": None,
        "reward": None,
    }
    parsed = parse_ledger_row(row)
    assert parsed.task_sha256 is None


def test_style_battery_rejects_loops_and_rechecks() -> None:
    clean = style_report("Step one: add the numbers. Step two: verify the sum.\n12")
    assert not clean.degenerate
    loop_line = "Let's check the string GOWBXKCO"
    looped = style_report("\n".join([loop_line] * 10) + "\n12")
    assert looped.degenerate
    assert "verbatim loop" in (looped.reason() or "")
    recheck = style_report("wait wait wait " * 200)
    assert recheck.degenerate
    assert "recheck rate" in (recheck.reason() or "")


def test_graceful_form_contract() -> None:
    prompt = graceful_convergence_prompt()
    for marker in CLOSING_MARKERS:
        assert marker in prompt
    good = check_graceful_trace(
        "I could not finish the reduction.",
        f"{BEST_EFFORT_MARKER} 12 -- the partial sums point here.",
        "stop",
    )
    assert good.passed
    assert good.resolution == "graceful_best_effort"
    uncertain = check_graceful_trace(
        "The last case diverges.",
        f"{UNCERTAIN_MARKER} 13 -- the final case does not terminate.",
        "stop",
    )
    assert uncertain.passed
    assert uncertain.resolution == "graceful_uncertain"
    truncated = check_graceful_trace("reasoning", "partial thought", "length")
    assert not truncated.passed
    assert "did not commit" in (truncated.reason or "")
    missing = check_graceful_trace("reasoning", "I think it is 12", "stop")
    assert not missing.passed
    assert "lacks both contract closings" in (missing.reason or "")


def _candidate(task_id: str, sha: str, domain: str = "math") -> CandidateTask:
    return CandidateTask(
        task_id=task_id, source_id="src", task_sha256=sha, domain=domain, task_path=f"/tasks/{task_id}.json"
    )


def _difficulty(sha: str, rate: float) -> TaskDifficulty:
    return TaskDifficulty(
        source_id="src", task_sha256=sha, task_id="t", solve_rate=rate, scored=3, basis="ledger:hosted"
    )


def test_selection_quotas_sort_and_census() -> None:
    candidates = [
        _candidate("easy-1", "aa" * 32),
        _candidate("med-1", "bb" * 32),
        _candidate("hard-1", "cc" * 32),
        _candidate("unsolved-1", "dd" * 32),
        _candidate("no-estimate", "ee" * 32),
        _candidate("med-2", "ff" * 32, domain="puzzles"),
    ]
    rates = {
        "aa" * 32: 0.9,
        "bb" * 32: 0.5,
        "cc" * 32: 0.1,
        "dd" * 32: 0.0,
        "ff" * 32: 0.4,
    }
    difficulty = {("src", sha): _difficulty(sha, rate) for sha, rate in rates.items()}
    quota = Quota({Band.EASY: 1, Band.MEDIUM: 1, Band.HARD: 1, Band.UNSOLVED: 1})
    selected, census = select_tasks(candidates, difficulty, quota)
    bands = [entry.band for entry in selected]
    assert bands == [Band.EASY, Band.MEDIUM, Band.HARD, Band.UNSOLVED]
    assert census["skipped_without_estimate"] == 1
    assert census["easy"] == 1 and census["easy_target"] == 1
    assert census["medium"] == 1  # the math cell filled first; the puzzles candidate is spare
    assert selection_digest(selected) != selection_digest(selected[:3])
