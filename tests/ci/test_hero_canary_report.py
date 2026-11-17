# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.ci.hero_canary_report import WARMUP_STEPS, baseline_score, score_run, step_time_regression

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


def test_one_regressed_run_does_not_become_the_baseline():
    def run(day: int, step_time: float):
        return score_run(f"hero-canary-{day}-1", "sha", _rows([step_time] * 31), {}, START_STEP)

    # Three healthy days, then a day that regressed by 2%.
    baseline = baseline_score([run(1, 15.5), run(2, 15.5), run(3, 15.5), run(4, 15.81)])

    assert step_time_regression(run(5, 15.81), baseline) is not None
    assert step_time_regression(run(5, 15.55), baseline) is None
