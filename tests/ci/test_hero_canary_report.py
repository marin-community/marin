# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.ci.hero_canary_report import WARMUP_STEPS, incomplete_reasons, score_run, step_time_warning

START_STEP = 372000


def _row(step: int, duration: float, loss: float = 0.9) -> dict:
    return {
        "_step": step,
        "throughput/duration": duration,
        "throughput/mfu": 34.5 * 15.5 / duration,
        "throughput/tokens_per_second": 4_194_304 / duration,
        "memory/peak_gib": 127.9,
        "moe/drop_fraction": 1e-3,
        "train/loss": loss,
    }


def _rows(steady_durations: list[float]) -> list[dict]:
    warmup = [_row(START_STEP + i, d) for i, d in enumerate([96.0, 18.0, 16.0, 15.6][:WARMUP_STEPS])]
    steady = [_row(START_STEP + WARMUP_STEPS + i, d) for i, d in enumerate(steady_durations)]
    return warmup + steady


def test_score_ignores_warmup_and_duplicate_rows():
    rows = _rows([15.5, 15.6, 15.4])
    rows.insert(0, dict(rows[0]))  # W&B repeats the first logged step
    score = score_run("hero-canary-1", "abc", rows, {"eval_dropless/macro_loss": 2.0}, START_STEP)

    assert score.steps_logged == WARMUP_STEPS + 3
    assert score.step_time == 15.5
    assert score.eval_loss == 2.0


def test_incomplete_run_is_reported():
    score = score_run("hero-canary-1", "abc", _rows([15.5] * 5), {}, START_STEP)

    assert incomplete_reasons(score, expected_steps=35) == [
        f"logged {WARMUP_STEPS + 5} of 35 steps",
        "the dropless eval did not report",
    ]


def test_step_time_change_beyond_threshold_warns():
    previous = score_run("hero-canary-1", "aaa", _rows([15.5] * 31), {"eval_dropless/macro_loss": 2.0}, START_STEP)
    slower = score_run("hero-canary-2", "bbb", _rows([15.81] * 31), {"eval_dropless/macro_loss": 2.0}, START_STEP)
    within_noise = score_run("hero-canary-3", "ccc", _rows([15.58] * 31), {"eval_dropless/macro_loss": 2.0}, START_STEP)

    assert step_time_warning(slower, previous) == "+2.00% against hero-canary-1"
    assert step_time_warning(within_noise, previous) is None
    assert step_time_warning(slower, None) is None
