# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mariner.cli import main
from mariner.data import (
    MAIN_SWARM,
    OLMIX_CAP,
    PAPER_TARGET_BUDGET,
    TRAINED_MARINER,
    TRAINED_OLMIX,
    mixture_weights,
    objective_from_tasks,
    pool_fractions,
    read_csv,
    read_table,
    release_partition,
    release_tasks,
    released_olmix_laws,
    trained_counts,
)
from mariner.evaluate import selection_metrics
from mariner.fit import fit_objective
from mariner.model import Anchor, Objective, ObjectiveFit, Swarm
from mariner.optimize import MIXTURE_BLOCK_SIZE, propose

# Pools of 3.2e9 down to 1e8 tokens give the synthetic swarm's 0.5 to 16 epochs per unit weight at this budget.
TARGET_BUDGET = 1.6e9
REPEAT_OFFSETS = (-1.0, 0.0, 1.0)  # in repeat SDs from each task's anchor: repeats with the anchor's mean and SD
# The tables' columns come back column-major, which moves the fit's last bit (measured 2.2e-16 in predictions).
FIT_TOLERANCE = 1e-12
FRESH_SEED = 7
FRESH_MIXTURES = 20
MIXTURE_COLUMNS = ["bucket", "continuous_weight", "runtime_count", "runtime_weight", "materialized_epochs"]
UNCHEATABLE_HELDOUT_EXPOSURES = 408
LAW_TOLERANCE = 1e-9  # agreed: refitted Olmix laws against fits_olmix_task_laws
BASELINE_TOLERANCE = 1e-15  # agreed: designed baselines against the released rows
BASELINES = ["baseline_proportional", "baseline_unimax", "baseline_stratified"]


def bucket_columns(prefix: str, buckets: Sequence[str], values: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{prefix}{bucket}": values[:, index] for index, bucket in enumerate(buckets)}


def write_own_swarm(directory: Path, swarm: Swarm, objective: Objective, anchors: dict[str, Anchor]) -> list[str]:
    """Write ``swarm`` as one's own run, partition and tasks tables and return the arguments that name them.

    The calibration run is ``baseline_proportional``; three ``repeat`` rows of the proportional mixture sit one repeat
    SD below, at and above each task's anchor.
    """
    fit_rows = pd.DataFrame(
        {
            "observation_id": swarm.runs,
            "run_label": [
                "baseline_proportional" if calibration else f"run_{row:05d}"
                for row, calibration in enumerate(swarm.calibration)
            ],
            "design_kind": np.where(swarm.calibration, "baseline", "sampled"),
            "fit_inclusion": True,
            **bucket_columns("w__", swarm.buckets, swarm.weights),
            **bucket_columns("pf__", swarm.buckets, swarm.pool_fractions),
            **{f"bpb__{task}": swarm.outcomes[task].to_numpy() for task in objective.tasks},
        }
    )
    offsets = np.asarray(REPEAT_OFFSETS)
    repeated = np.tile(swarm.weights[swarm.calibration][0], (len(offsets), 1))
    repeats = pd.DataFrame(
        {
            "observation_id": [f"repeat_{index}" for index in range(len(offsets))],
            "run_label": [f"proportional_repeat_{index:02d}" for index in range(len(offsets))],
            "design_kind": "repeat",
            "fit_inclusion": False,
            **bucket_columns("w__", swarm.buckets, repeated),
            **bucket_columns("pf__", swarm.buckets, np.ones_like(repeated)),
            **{
                f"bpb__{task}": anchors[task].proportional + anchors[task].repeat_sd * offsets
                for task in objective.tasks
            },
        }
    )
    paths = {name: directory / f"{name}.csv" for name in ("runs", "partition", "tasks")}
    pd.concat([fit_rows, repeats], ignore_index=True).to_csv(paths["runs"], index=False)
    pools = TARGET_BUDGET / swarm.epochs_per_unit_weight
    pd.DataFrame({"bucket_id": list(swarm.buckets), "pool_tokens": pools}).to_csv(paths["partition"], index=False)
    tasks = {"task_id": list(objective.tasks), "objective": objective.name, "aggregate_weight": objective.weights}
    pd.DataFrame(tasks).to_csv(paths["tasks"], index=False)
    named = [item for name, path in paths.items() for item in (f"--{name}", str(path))]
    return [*named, "--target-budget", str(TARGET_BUDGET)]


def test_cli_fit_predict_optimize_round_trip_on_own_swarm(tmp_path, synthetic_problem):
    swarm, objective, anchors, _ = synthetic_problem
    inputs = write_own_swarm(tmp_path, swarm, objective, anchors)
    fit_path = tmp_path / "fit.json"
    assert main(["fit", *inputs, "--objective", objective.name, "--out", str(fit_path)]) == 0
    fit = ObjectiveFit.from_json(json.loads(fit_path.read_text()))
    weights = np.random.default_rng(FRESH_SEED).dirichlet(np.ones(len(swarm.buckets)), size=FRESH_MIXTURES)
    # The fit of the tables is the fit of the swarm they hold, with the anchors their repeats give.
    reference = fit_objective(swarm, objective, anchors)
    np.testing.assert_allclose(fit.predict_tasks(weights), reference.predict_tasks(weights), rtol=0, atol=FIT_TOLERANCE)

    fractions = np.ones_like(weights)
    fractions[FRESH_MIXTURES // 2 :, -1] = 0.5  # the second half draws half of b5's pool
    mixtures = pd.DataFrame(
        {
            "mixture_id": [f"mix_{row:02d}" for row in range(FRESH_MIXTURES)],
            **bucket_columns("w__", swarm.buckets, weights),
            **bucket_columns("pf__", swarm.buckets, fractions),
        }
    )
    mixtures.to_csv(tmp_path / "mixtures.csv", index=False)
    predicted_path = tmp_path / "predicted.csv"
    command = ["predict", "--fit", str(fit_path), "--mixtures", str(tmp_path / "mixtures.csv")]
    assert main([*command, "--out", str(predicted_path)]) == 0
    predicted = read_csv(predicted_path)
    assert list(predicted.columns) == ["mixture_id", "predicted_synthetic"]
    assert list(predicted["mixture_id"]) == list(mixtures["mixture_id"])
    # The mixtures as the package reads them: a prediction's last bit depends on the array's memory layout, and the
    # file's columns come back column-major.
    written = read_csv(tmp_path / "mixtures.csv")
    expected = fit.predict(mixture_weights(written, swarm.buckets), pool_fractions(written, swarm.buckets))
    np.testing.assert_array_equal(predicted["predicted_synthetic"], expected)
    # Without pf__ columns, every mixture draws from whole pools.
    mixtures.filter(regex="^(mixture_id|w__)").to_csv(tmp_path / "whole_pools.csv", index=False)
    command = ["predict", "--fit", str(fit_path), "--mixtures", str(tmp_path / "whole_pools.csv")]
    assert main([*command, "--out", str(predicted_path)]) == 0
    whole_pools = fit.predict(mixture_weights(read_csv(tmp_path / "whole_pools.csv"), swarm.buckets))
    np.testing.assert_array_equal(read_csv(predicted_path)["predicted_synthetic"], whole_pools)

    mixture_path = tmp_path / "mixture.csv"
    assert main(["optimize", "--fit", str(fit_path), *inputs, "--out", str(mixture_path)]) == 0
    mixture = read_csv(mixture_path)
    assert list(mixture.columns) == MIXTURE_COLUMNS
    assert list(mixture["bucket"]) == list(swarm.buckets)
    assert mixture["runtime_count"].sum() == MIXTURE_BLOCK_SIZE
    np.testing.assert_array_equal(mixture["runtime_count"], propose(fit, swarm.weights).counts)
    runtime = mixture["runtime_count"].to_numpy() / MIXTURE_BLOCK_SIZE
    np.testing.assert_array_equal(mixture["runtime_weight"], runtime)
    # The target run repeats bucket i w_i D_tgt / P_i times.
    pools = read_csv(tmp_path / "partition.csv")["pool_tokens"].to_numpy()
    np.testing.assert_allclose(mixture["materialized_epochs"], runtime * TARGET_BUDGET / pools, rtol=1e-15, atol=0)
    diagnostics = json.loads((tmp_path / "mixture.diagnostics.json").read_text())
    assert diagnostics["summary"]["runtime_prediction"] == fit.predict(runtime)[0]


def test_cli_optimize_applies_cap_and_kl_penalty(tmp_path, synthetic_problem):
    swarm, objective, anchors, truth = synthetic_problem
    inputs = write_own_swarm(tmp_path, swarm, objective, anchors)
    fit_path, out = tmp_path / "fit.json", tmp_path / "mixture.csv"
    fit_path.write_text(json.dumps(truth.to_json()))
    # Unconstrained, the generating model's optimum puts more than 2 epochs on a bucket (test_optimize).
    assert main(["optimize", "--fit", str(fit_path), *inputs, "--cap", "2", "--kl", "0.01", "--out", str(out)]) == 0
    expected = propose(truth, swarm.weights, cap=2.0, kl=0.01)
    np.testing.assert_array_equal(read_csv(out)["runtime_count"], expected.counts)


def test_cli_predict_names_missing_buckets(tmp_path, synthetic_problem, capsys):
    swarm, _, _, truth = synthetic_problem
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(truth.to_json()))
    mixtures = tmp_path / "mixtures.csv"
    pd.DataFrame(bucket_columns("w__", swarm.buckets[:5], swarm.weights)).to_csv(mixtures, index=False)
    out = tmp_path / "predicted.csv"
    assert main(["predict", "--fit", str(fit_path), "--mixtures", str(mixtures), "--out", str(out)]) != 0
    error = capsys.readouterr().err
    assert "w__b5" in error and "w__b4" not in error
    assert not out.exists()


def test_cli_cross_validate_prints_selection_metrics_of_each_repeat(tmp_path, synthetic_problem, capsys):
    swarm, objective, anchors, _ = synthetic_problem
    one_task = dataclasses.replace(objective, tasks=objective.tasks[:1], weights=objective.weights[:1])
    inputs = write_own_swarm(tmp_path, swarm, one_task, anchors)
    out = tmp_path / "cv.csv"
    command = ["cross-validate", *inputs, "--objective", one_task.name, "--repeats", "2", "--out", str(out)]
    assert main(command) == 0
    table = read_csv(out)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [metrics.pop("repeat") for metrics in printed] == [0, 1]
    for repeat, metrics in enumerate(printed):
        rows = table[table["repeat"] == repeat]
        # Each repeat scores every run but the calibration run once, and only its own predictions.
        assert sorted(rows["observation_id"]) == sorted(swarm.runs[1:])
        assert metrics == selection_metrics(rows["predicted"].to_numpy(), rows["measured"].to_numpy())


@pytest.mark.data_integration
@pytest.mark.timeout(300)
def test_cli_fit_and_optimize_rederive_trained_mariner_mixture(snapshot, tmp_path, capsys):
    swarm = ["--data", str(snapshot), "--swarm", MAIN_SWARM]
    fit, mixture = tmp_path / "fit.json", tmp_path / "mixture.csv"
    assert main(["fit", *swarm, "--objective", "uncheatable", "--out", str(fit)]) == 0
    assert main(["optimize", "--fit", str(fit), *swarm, "--out", str(mixture)]) == 0
    trained = trained_counts(snapshot, TRAINED_MARINER["uncheatable"])
    np.testing.assert_array_equal(read_csv(mixture)["runtime_count"], trained)
    capsys.readouterr()
    assert main(["evaluate", "--fit", str(fit), "--data", str(snapshot)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == UNCHEATABLE_HELDOUT_EXPOSURES


@pytest.mark.data_integration
@pytest.mark.timeout(300)
def test_cli_olmix_refits_released_laws_and_rederives_trained_mixture(snapshot, tmp_path):
    trained = TRAINED_OLMIX["uncheatable"]
    out = tmp_path / "olmix.csv"
    command = ["olmix", "--data", str(snapshot), "--swarm", MAIN_SWARM, "--objective", "uncheatable"]
    assert main([*command, "--kl", str(trained.kl), "--cap", str(OLMIX_CAP), "--out", str(out)]) == 0
    np.testing.assert_array_equal(read_csv(out)["runtime_count"], trained_counts(snapshot, trained.mixture_id))
    laws = json.loads((tmp_path / "olmix.laws.json").read_text())["tasks"]
    assert [law["task"] for law in laws] == list(objective_from_tasks(release_tasks(snapshot), "uncheatable").tasks)
    refit = np.array([[law["log_c"], law["huber_loss"], *law["coefficients"]] for law in laws])
    released = released_olmix_laws(snapshot, "uncheatable")
    expected = np.array([[law.log_c, law.huber_loss, *law.coefficients] for law in released])
    np.testing.assert_allclose(refit, expected, rtol=0, atol=LAW_TOLERANCE)


@pytest.mark.data_integration
@pytest.mark.timeout(60)
def test_cli_design_baselines_and_removals_equal_released_swarm_rows(snapshot, tmp_path):
    partition, out = tmp_path / "partition.csv", tmp_path / "design.csv"
    read_table(snapshot, "partition").to_csv(partition, index=False)
    command = ["design", "--partition", str(partition), "--target-budget", str(PAPER_TARGET_BUDGET)]
    options = ["--candidates", "4", "--seed", "0", "--repeats", "2", "--unimax-budget-fraction", "0.8"]
    assert main([*command, *options, "--out", str(out)]) == 0
    design, released = read_csv(out), read_table(snapshot, f"swarm_{MAIN_SWARM}")
    columns = [f"w__{bucket}" for bucket in release_partition(snapshot).bucket_ids]
    np.testing.assert_allclose(
        design.set_index("run_label").loc[BASELINES, columns],
        released.set_index("run_label").loc[BASELINES, columns],
        rtol=0,
        atol=BASELINE_TOLERANCE,
    )
    # The released removals carry other labels, so each is matched by the bucket it zeroes.
    removals = [table.loc[table["design_kind"] == "deletion", columns].to_numpy() for table in (design, released)]
    designed, expected = (rows[np.argsort((rows == 0).argmax(axis=1))] for rows in removals)
    np.testing.assert_allclose(designed, expected, rtol=0, atol=BASELINE_TOLERANCE)
