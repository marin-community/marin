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
# GSM8K x Delphi base models (seed means) and the Qwen chat-model families on AIME and HLE. Expected values below
# are printed by that commit's experiments/forecasting scripts.
DELPHI = json.loads((Path(__file__).parent / "delphi_gsm8k_cells.json").read_text())
QWEN = json.loads((Path(__file__).parent / "qwen_cells.json").read_text())
PAIR_CELLS = {"gsm8k/delphi": DELPHI["seed_mean"], **{pair: QWEN[pair] for pair in QWEN if pair != "source"}}

# table1.py, as printed: joint-law b1, b2, b0 (two decimals) and in-sample MAE in percentage points.
PAPER_TABLE1 = {
    ("gsm8k/delphi", HintMethod.MASKING): ((0.89, 4.20, -3.95), 4.11),
    ("gsm8k/delphi", HintMethod.TRUNCATION): ((1.73, 2.46, -6.53), 1.05),
    ("aime/qwen25", HintMethod.MASKING): ((1.89, 5.28, 32.66), 3.41),
    ("aime/qwen25", HintMethod.TRUNCATION): ((1.64, 7.26, 25.44), 5.67),
    ("hle/qwen3", HintMethod.MASKING): ((1.57, 5.99, 26.78), 1.69),
    ("hle/qwen3", HintMethod.TRUNCATION): ((1.28, 5.67, 20.55), 1.73),
    ("hle/qwen35", HintMethod.MASKING): ((2.52, 4.91, 45.62), 2.88),
    ("hle/qwen35", HintMethod.TRUNCATION): ((2.00, 3.46, 36.26), 2.85),
}
# plot.py forecasts: fitted models -> absolute error forecasting the pair's largest model
# (sigmoid baseline, hinted ensemble).
PAPER_FORECAST_ERRORS = {
    "gsm8k/delphi": {
        7: (0.1931, 0.0735),
        8: (0.2081, 0.0355),
        9: (0.2184, 0.0171),
        10: (0.2210, 0.0606),
        11: (0.1165, 0.0189),
    },
    "aime/qwen25": {2: (0.8815, 0.0295), 3: (0.6684, 0.0798), 4: (0.2482, 0.0574), 5: (0.1177, 0.0291)},
    "aime/qwen3": {2: (0.2195, 0.1240), 3: (0.2156, 0.1704), 4: (0.1705, 0.1173), 5: (0.1348, 0.0905)},
    "hle/qwen25": {2: (0.0799, 0.0525), 3: (0.0273, 0.0617), 4: (0.0109, 0.0456), 5: (0.0076, 0.0410)},
    "hle/qwen3": {2: (0.0801, 0.0181), 3: (0.0343, 0.0018), 4: (0.0590, 0.0244), 5: (0.0635, 0.0274)},
    "hle/qwen35": {2: (0.2865, 0.0938), 3: (0.2402, 0.0433), 4: (0.0858, 0.1540)},
}
# two_stage.py without reference models: fitted rungs -> absolute error forecasting 1e23 on the original seed.
PAPER_TWO_STAGE_ERRORS = {9: 0.1628, 8: 0.2300, 7: 0.2304, 6: 0.2347, 5: 0.2364}
# Half a unit of the four printed decimals, plus 1e-5 for least-squares results that differ across platforms:
# hle/qwen3 at 5 models gives 0.0274508 on Linux CI against the paper's printed 0.0274.
PRINTED_ERROR_TOLERANCE = 5e-5 + 1e-5


def _cells(record: dict) -> ScoreCells:
    # The fits use cell accuracies only, so a single column carrying each cell's mean is enough.
    accuracy = np.array(record["accuracy"])
    return ScoreCells(np.array(record["capability"]), np.array(record["hint"]), accuracy[:, None])


def _ladder(pair: str) -> LadderScores:
    cells = PAIR_CELLS[pair]
    return LadderScores(
        unhinted=_cells(cells["unhinted"]),
        hinted={method: _cells(cells[method.value]) for method in HintMethod},
        bits_per_byte=None,
    )


@pytest.mark.parametrize(("pair", "method"), list(PAPER_TABLE1))
def test_joint_law_reproduces_paper_table1(pair: str, method: HintMethod):
    cells = PAIR_CELLS[pair]
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


@pytest.mark.parametrize("pair", list(PAPER_FORECAST_ERRORS))
def test_forecast_errors_reproduce_paper(pair: str):
    ladder = _ladder(pair)
    capability = ladder.unhinted.capability
    target, observed = capability[-1], ladder.unhinted.accuracy[-1]

    for fitted_models, (paper_baseline, paper_ensemble) in PAPER_FORECAST_ERRORS[pair].items():
        forecasts = fit_ladder(ladder.up_to(capability[fitted_models - 1])).forecasts(target)
        assert abs(forecasts[ForecastKind.SIGMOID_BASELINE] - observed) == pytest.approx(
            paper_baseline, abs=PRINTED_ERROR_TOLERANCE
        )
        assert abs(forecasts[ForecastKind.HINTED_ENSEMBLE] - observed) == pytest.approx(
            paper_ensemble, abs=PRINTED_ERROR_TOLERANCE
        )


def test_two_stage_baseline_reproduces_paper_on_delphi():
    record = DELPHI["original_seed_ten_rungs"]
    capability = np.array(record["unhinted"]["capability"])
    accuracy = np.array(record["unhinted"]["accuracy"])
    bits_per_byte = np.array(record["bits_per_byte"])

    for fitted_rungs, paper_error in PAPER_TWO_STAGE_ERRORS.items():
        fit = fit_two_stage_baseline(capability[:fitted_rungs], bits_per_byte[:fitted_rungs], accuracy[:fitted_rungs])
        assert abs(float(fit.predict(capability[-1])) - accuracy[-1]) == pytest.approx(
            paper_error, abs=PRINTED_ERROR_TOLERANCE
        )


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
        bits_per_byte=None,
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
