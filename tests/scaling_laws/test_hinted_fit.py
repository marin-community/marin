# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest
from marin.scaling_laws.hinted_fit import (
    ForecastKind,
    HintMethod,
    JointLawParams,
    LadderScores,
    ScoreCells,
    bootstrap_forecasts,
    fit_joint_law,
    fit_ladder,
    fit_two_stage_baseline,
)

# Cell accuracies from the paper's released rollouts (tatsu-lab/hinted-scaling-laws@c0490e53, data/results):
# GSM8K x Delphi (base models, seed means) and AIME x Qwen2.5-Instruct (chat models). Expected values below are
# printed by that commit's experiments/forecasting scripts.
DELPHI = json.loads((Path(__file__).parent / "delphi_gsm8k_cells.json").read_text())
AIME_QWEN25 = json.loads((Path(__file__).parent / "aime_qwen25_cells.json").read_text())
TABLE1_CELLS = {"gsm8k/delphi": DELPHI["seed_mean"], "aime/qwen25": AIME_QWEN25}

# table1.py, as printed: joint-law b1, b2, b0 (two decimals) and in-sample MAE in percentage points.
PAPER_TABLE1 = {
    ("gsm8k/delphi", HintMethod.MASKING): ((0.89, 4.20, -3.95), 4.11),
    ("gsm8k/delphi", HintMethod.TRUNCATION): ((1.73, 2.46, -6.53), 1.05),
    ("aime/qwen25", HintMethod.MASKING): ((1.89, 5.28, 32.66), 3.41),
    ("aime/qwen25", HintMethod.TRUNCATION): ((1.64, 7.26, 25.44), 5.67),
}
# plot.py forecasts: fitted rungs -> absolute error forecasting the 1e23 rung (sigmoid baseline, hinted ensemble).
PAPER_FORECAST_ERRORS = {
    7: (0.1931, 0.0735),
    8: (0.2081, 0.0355),
    9: (0.2184, 0.0171),
    10: (0.2210, 0.0606),
    11: (0.1165, 0.0189),
}
# two_stage.py without reference models: fitted rungs -> absolute error forecasting 1e23 on the original seed.
PAPER_TWO_STAGE_ERRORS = {9: 0.1628, 8: 0.2300, 7: 0.2304, 6: 0.2347, 5: 0.2364}


def _cells(record: dict) -> ScoreCells:
    # The fits use cell accuracies only, so a single column carrying each cell's mean is enough.
    accuracy = np.array(record["accuracy"])
    return ScoreCells(np.array(record["capability"]), np.array(record["hint"]), accuracy[:, None])


def _delphi_ladder() -> LadderScores:
    seed_mean = DELPHI["seed_mean"]
    unhinted = _cells(seed_mean["unhinted"])
    return LadderScores(
        unhinted=unhinted,
        hinted={method: _cells(seed_mean[method.value]) for method in HintMethod},
        # The paper releases no BPB for two of these rungs; the two-stage forecast is checked separately.
        bits_per_byte=None,
    )


@pytest.mark.parametrize(("pair", "method"), list(PAPER_TABLE1))
def test_joint_law_reproduces_paper_table1(pair: str, method: HintMethod):
    cells = TABLE1_CELLS[pair]
    c, h, y = (np.r_[cells[method.value][k], cells["unhinted"][k]] for k in ("capability", "hint", "accuracy"))

    fit = fit_joint_law(c, h, y)

    (b1, b2, b0), paper_mae_pp = PAPER_TABLE1[(pair, method)]
    assert fit.capability_slope == pytest.approx(b1, abs=0.005)
    assert fit.hint_slope == pytest.approx(b2, abs=0.005)
    assert fit.intercept == pytest.approx(b0, abs=0.005)
    # A, L and F are checked through the in-sample MAE: L and F trade off near zero accuracy (perturbing the inputs
    # by 1e-8 moves the Delphi truncation floor by 0.001) without changing the fitted curve.
    residuals = np.abs(fit.predict(c, h) - y)
    mae_pp = 100 * np.mean([residuals[c == model].mean() for model in np.unique(c)])
    assert mae_pp == pytest.approx(paper_mae_pp, abs=0.005)


def test_forecast_errors_reproduce_paper_on_delphi():
    ladder = _delphi_ladder()
    capability = ladder.unhinted.capability
    target, observed = capability[-1], ladder.unhinted.accuracy[-1]

    for fitted_rungs, (paper_baseline, paper_ensemble) in PAPER_FORECAST_ERRORS.items():
        forecasts = fit_ladder(ladder.up_to(capability[fitted_rungs - 1])).forecasts(target)
        assert abs(forecasts[ForecastKind.SIGMOID_BASELINE] - observed) == pytest.approx(paper_baseline, abs=5e-5)
        assert abs(forecasts[ForecastKind.HINTED_ENSEMBLE] - observed) == pytest.approx(paper_ensemble, abs=5e-5)


def test_two_stage_baseline_reproduces_paper_on_delphi():
    record = DELPHI["original_seed_ten_rungs"]
    capability = np.array(record["unhinted"]["capability"])
    accuracy = np.array(record["unhinted"]["accuracy"])
    bits_per_byte = np.array(record["bits_per_byte"])

    for fitted_rungs, paper_error in PAPER_TWO_STAGE_ERRORS.items():
        fit = fit_two_stage_baseline(capability[:fitted_rungs], bits_per_byte[:fitted_rungs], accuracy[:fitted_rungs])
        assert abs(float(fit.predict(capability[-1])) - accuracy[-1]) == pytest.approx(paper_error, abs=5e-5)


BOOTSTRAP_LAW = JointLawParams(capability_slope=1.2, hint_slope=4.0, intercept=-2.5, upper=0.9, lower=-0.05, floor=0.02)
BOOTSTRAP_CAPABILITIES = np.arange(18.0, 21.01, 0.5)
BOOTSTRAP_HINTS = np.array([0.2, 0.5, 0.8])
BOOTSTRAP_TARGET = 22.0


def _sampled_ladder(n_problems: int, rng: np.random.Generator) -> LadderScores:
    """One Bernoulli outcome per (cell, problem) from a known law, on a coarse hint grid to keep refits cheap."""

    def sample(c: np.ndarray, h: np.ndarray) -> ScoreCells:
        p = BOOTSTRAP_LAW.predict(c, h)[:, None]
        return ScoreCells(c, h, (rng.random((len(c), n_problems)) < p).astype(float))

    c, h = (a.ravel() for a in np.meshgrid(BOOTSTRAP_CAPABILITIES, BOOTSTRAP_HINTS, indexing="ij"))
    return LadderScores(
        unhinted=sample(BOOTSTRAP_CAPABILITIES, np.zeros_like(BOOTSTRAP_CAPABILITIES)),
        hinted={HintMethod.MASKING: sample(c, h)},
        bits_per_byte=np.linspace(1.0, 0.8, len(BOOTSTRAP_CAPABILITIES)),
    )


def test_bootstrap_intervals_cover_truth_and_narrow_with_more_problems():
    rng = np.random.default_rng(0)
    truth = float(BOOTSTRAP_LAW.predict(BOOTSTRAP_TARGET, 0.0))

    small = bootstrap_forecasts(_sampled_ladder(200, rng), BOOTSTRAP_TARGET, n_resamples=12, seed=1)
    large = bootstrap_forecasts(_sampled_ladder(3200, rng), BOOTSTRAP_TARGET, n_resamples=12, seed=1)

    small_lo, small_hi = small.interval(ForecastKind.MASKING, 0.95)
    large_lo, large_hi = large.interval(ForecastKind.MASKING, 0.95)
    assert small_lo <= truth <= small_hi
    assert large_lo <= truth <= large_hi
    # 16x the problems should shrink the interval about 4x; require at least 2x.
    assert (large_hi - large_lo) < 0.5 * (small_hi - small_lo)
