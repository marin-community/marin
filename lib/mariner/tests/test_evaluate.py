# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mariner.data import MAIN_SWARM, objective_from_tasks, read_table, release_anchors, release_swarm, release_tasks
from mariner.evaluate import OUTER_FOLDS, cross_validate, heldout_metrics, outer_splits, selection_metrics
from mariner.fit import final_inner_fold_labels
from mariner.model import HeldoutBank, Swarm

# The paper's fold labels of the main swarm's fit rows, by observation id: final_inner, the final fit's inner fold;
# outer_r{r}, the outer fold that tests the run in repeat r; inner_r{r}_f{f}, the inner fold that validates it within
# outer fold f's training rows, empty when that fold tests it. -1 marks the calibration row, which no fold tests or
# validates.
FOLD_LABELS = Path(__file__).parent / "data" / "qwen3_fold_labels.csv"
PAPER_REPEATS = 5
PERTURBED_RUN = 7

# MARINER's row of the paper's surrogate benchmark (Table 10) in fits_surrogate_benchmark: the metrics of one
# repeat's pooled out-of-fold predictions. Uncheatable takes about 65 s; OlmoBaseEval Easy, about 8 minutes.
TABLE_10_MODEL = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
TABLE_10_OBJECTIVE = "uncheatable"
# The prediction tolerance: the metrics are computed from predictions that match the research fit's within it.
TABLE_10_TOLERANCE = 1e-6
TABLE_10_COLUMNS = {  # selection_metrics key -> fits_surrogate_benchmark column
    "rmse": "rmse",
    "spearman": "spearman",
    "regret_at_1": "regret_at_1",
    "optimism_at_pick": "selection_optimism",
}


def test_cross_validate_prediction_ignores_its_own_outcome(synthetic_problem):
    swarm, objective, anchors, _ = synthetic_problem
    outcomes = swarm.outcomes.copy()
    outcomes.loc[PERTURBED_RUN, "t0"] += 0.05
    run = swarm.runs[PERTURBED_RUN]
    before = cross_validate(swarm, objective, anchors).set_index("observation_id")
    after = cross_validate(dataclasses.replace(swarm, outcomes=outcomes), objective, anchors).set_index("observation_id")
    # Every run but the calibration row 0 is scored once; measured moves by 0.05 times t0's objective weight, 0.5.
    assert sorted(before.index) == sorted(swarm.runs[1:])
    assert after.loc[run, "measured"] == pytest.approx(before.loc[run, "measured"] + 0.025)
    assert after.loc[run, "predicted"] == before.loc[run, "predicted"]
    # The fits that train on the run see the change.
    assert (after["predicted"].drop(run) != before["predicted"].drop(run)).any()


def test_cross_validate_predicts_each_run_at_its_pool_fractions(synthetic_problem):
    swarm, objective, anchors, _ = synthetic_problem
    one_task = dataclasses.replace(objective, tasks=objective.tasks[:1], weights=objective.weights[:1])
    fractions = swarm.pool_fractions.copy()
    fractions[PERTURBED_RUN] = 0.5  # the run draws half of every pool, which doubles its exposures
    run = swarm.runs[PERTURBED_RUN]
    before = cross_validate(swarm, one_task, anchors).set_index("observation_id")
    one_run_halved = dataclasses.replace(swarm, pool_fractions=fractions)
    after = cross_validate(one_run_halved, one_task, anchors).set_index("observation_id")
    # The fit that predicts the run never trains on it, so in the run's fold only the run's own prediction moves.
    fold = before["fold"] == before.loc[run, "fold"]
    assert after.loc[run, "predicted"] != before.loc[run, "predicted"]
    others = fold & (before.index != run)
    pd.testing.assert_series_equal(after.loc[others, "predicted"], before.loc[others, "predicted"], check_exact=True)
    # The fits of the other folds train on the run at its new exposures.
    assert (after.loc[~fold, "predicted"] != before.loc[~fold, "predicted"]).any()


def test_outer_splits_never_validate_calibration_rows(synthetic_problem):
    swarm, _, _, _ = synthetic_problem
    splits = outer_splits(swarm, repeats=2)
    assert [(split.repeat, split.fold) for split in splits] == [(r, f) for r in range(2) for f in range(OUTER_FOLDS)]
    rows = np.arange(len(swarm.runs))
    for split in splits:
        np.testing.assert_array_equal(np.sort(np.concatenate([split.train, split.test])), rows)
        assert 0 in split.train and 0 not in split.test
        assert all(0 in train and 0 not in validation for train, validation in split.inner)
        # The inner folds validate each training run but row 0 exactly once, and train on the rest.
        validated = np.concatenate([validation for _, validation in split.inner])
        np.testing.assert_array_equal(np.sort(validated), split.train[split.train != 0])
        for train, validation in split.inner:
            np.testing.assert_array_equal(np.sort(np.concatenate([train, validation])), split.train)
    for repeat in range(2):
        tested = np.concatenate([split.test for split in splits if split.repeat == repeat])
        np.testing.assert_array_equal(np.sort(tested), rows[1:])


def test_outer_splits_reject_an_inner_fold_of_calibration_runs_only():
    # Fifteen calibration runs at one corner mixture, three runs near it and sixty near the centre: the corner's block
    # tests the three runs, and the split that tests them leaves the calibration runs an inner block of their own.
    rng = np.random.default_rng(0)
    corner = np.tile([0.97, 0.01, 0.01, 0.01], (15, 1))
    near = rng.dirichlet([291.0, 3.0, 3.0, 3.0], size=3)
    weights = np.vstack([corner, near, rng.dirichlet(np.full(4, 5.0), size=60)])
    swarm = Swarm(
        runs=tuple(f"run_{row:02d}" for row in range(len(weights))),
        buckets=("b0", "b1", "b2", "b3"),
        weights=weights,
        epochs_per_unit_weight=np.ones(4),
        pool_fractions=np.ones_like(weights),
        outcomes=pd.DataFrame({"t0": np.ones(len(weights))}),
        calibration=np.arange(len(weights)) < len(corner),
    )
    with pytest.raises(ValueError, match=r"inner fold \d of outer fold \d of repeat 0 validates no run"):
        outer_splits(swarm, repeats=1)


def test_selection_metrics_on_a_known_ranking():
    metrics = selection_metrics(np.array([0.3, 0.1, 0.2]), np.array([1.0, 1.2, 0.9]))
    # The pick, predicted 0.1, measures 1.2: the worst of three, 0.3 above the best.
    assert metrics == pytest.approx(
        {
            "rows": 3,
            "regret_at_1": 0.3,
            "selected_rank": 3,
            "optimism_at_pick": 1.1,
            "rmse": math.sqrt(0.73),
            "spearman": -0.5,
        }
    )


def test_heldout_metrics_score_each_exposure_by_its_measured_objective(synthetic_problem):
    _, objective, _, truth = synthetic_problem
    rng = np.random.default_rng(2)
    weights = rng.dirichlet(np.ones(len(truth.buckets)), size=12)
    fractions = np.where(rng.random(weights.shape) < 0.5, 0.5, 1.0)
    measured = pd.DataFrame(truth.predict_tasks(weights, fractions), columns=list(objective.tasks))
    measured.iloc[0] = math.nan  # an exposure whose runs recorded only the objective
    bank = HeldoutBank(
        exposure_ids=tuple(f"e{index}" for index in range(len(weights))),
        buckets=truth.buckets,
        weights=weights,
        pool_fractions=fractions,
        measured=measured,
        measured_objective=truth.predict(weights, fractions),
        runs=np.ones(len(weights), dtype=int),
    )
    # The generating model, evaluated at each exposure's weights and pool fractions, scores perfectly.
    assert heldout_metrics(truth, bank) == pytest.approx(
        {"rows": 12, "regret_at_1": 0.0, "selected_rank": 1, "optimism_at_pick": 0.0, "rmse": 0.0, "spearman": 1.0}
    )


@pytest.mark.data_integration
@pytest.mark.timeout(120)
def test_outer_fold_labels_match_paper_schedule(snapshot):
    swarm = release_swarm(snapshot, MAIN_SWARM, "obe_easy")
    oracle = pd.read_csv(FOLD_LABELS, comment="#").set_index("observation_id")
    assert sorted(oracle.index) == sorted(swarm.runs)
    runs = len(swarm.runs)
    labels = {"final_inner": final_inner_fold_labels(swarm).astype(float)}
    for split in outer_splits(swarm, repeats=PAPER_REPEATS):
        labels.setdefault(f"outer_r{split.repeat}", np.full(runs, -1.0))[split.test] = split.fold
        inner = np.full(runs, math.nan)
        inner[split.train] = -1.0
        for index, (_train, validation) in enumerate(split.inner):
            inner[validation] = index
        labels[f"inner_r{split.repeat}_f{split.fold}"] = inner
    assert set(labels) == set(oracle.columns)
    expected = oracle.loc[list(swarm.runs), list(labels)].to_numpy(float)
    np.testing.assert_array_equal(np.column_stack(list(labels.values())), expected)


@pytest.mark.data_integration
@pytest.mark.timeout(300)
def test_cross_validate_reproduces_table_10_mariner_row_on_uncheatable(snapshot):
    swarm = release_swarm(snapshot, MAIN_SWARM, TABLE_10_OBJECTIVE)
    objective = objective_from_tasks(release_tasks(snapshot), TABLE_10_OBJECTIVE)
    anchors = release_anchors(snapshot, MAIN_SWARM, TABLE_10_OBJECTIVE)
    table = cross_validate(swarm, objective, anchors, repeats=1)
    # The pooled out-of-fold predictions of every run but the calibration row.
    metrics = selection_metrics(table["predicted"].to_numpy(), table["measured"].to_numpy())
    assert metrics["rows"] == len(swarm.runs) - 1
    benchmark = read_table(snapshot, "fits_surrogate_benchmark")
    released = benchmark[
        (benchmark["model"] == TABLE_10_MODEL)
        & (benchmark["swarm"] == MAIN_SWARM)
        & (benchmark["objective"] == TABLE_10_OBJECTIVE)
    ]
    assert len(released) == 1
    gaps = {key: abs(metrics[key] - float(released[column].iloc[0])) for key, column in TABLE_10_COLUMNS.items()}
    assert all(gap <= TABLE_10_TOLERANCE for gap in gaps.values()), gaps
