# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mariner import olmix
from mariner.cli import INPUT_ERROR_STATUS, MIXTURES_TABLE, main
from mariner.data import (
    CALIBRATION_RUN,
    MAIN_SWARM,
    OLMIX_CAP,
    PAPER_TARGET_BUDGET,
    REPEAT,
    TRAINED_MARINER,
    TRAINED_OLMIX,
    mixture_weights,
    objective_from_tasks,
    pool_fractions,
    read_csv,
    read_partition,
    read_table,
    release_partition,
    release_runs,
    release_tasks,
    released_olmix_laws,
    swarm_from_runs,
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
# The refit repeats the released laws' fit on the same runs (measured gap 0); the margin is for floating-point
# differences between platforms.
LAW_TOLERANCE = 1e-9
BASELINE_TOLERANCE = 1e-15  # rounding: the designs repeat the released rows' arithmetic, 5.8e-16 apart at most
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
    swarm, objective, anchors = synthetic_problem.swarm, synthetic_problem.objective, synthetic_problem.anchors
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
    expected = fit.predict(
        mixture_weights(written, swarm.buckets, MIXTURES_TABLE), pool_fractions(written, swarm.buckets, MIXTURES_TABLE)
    )
    np.testing.assert_array_equal(predicted["predicted_synthetic"], expected)
    # Without pf__ columns, every mixture draws from whole pools.
    mixtures.filter(regex="^(mixture_id|w__)").to_csv(tmp_path / "whole_pools.csv", index=False)
    command = ["predict", "--fit", str(fit_path), "--mixtures", str(tmp_path / "whole_pools.csv")]
    assert main([*command, "--out", str(predicted_path)]) == 0
    whole_pools = fit.predict(mixture_weights(read_csv(tmp_path / "whole_pools.csv"), swarm.buckets, MIXTURES_TABLE))
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


def write_generating_fit(path: Path, swarm: Swarm, truth: ObjectiveFit) -> None:
    """Write the generating model as the fit JSON of ``swarm``'s runs, which optimize requires of its fit."""
    path.write_text(json.dumps(dataclasses.replace(truth, inner_fold_labels=dict.fromkeys(swarm.runs, 0)).to_json()))


def test_cli_optimize_applies_cap_and_kl_penalty(tmp_path, synthetic_problem):
    swarm, truth = synthetic_problem.swarm, synthetic_problem.truth
    inputs = write_own_swarm(tmp_path, swarm, synthetic_problem.objective, synthetic_problem.anchors)
    fit_path, out = tmp_path / "fit.json", tmp_path / "mixture.csv"
    write_generating_fit(fit_path, swarm, truth)
    # Unconstrained, the generating model's optimum puts more than 2 epochs on a bucket (test_optimize).
    assert main(["optimize", "--fit", str(fit_path), *inputs, "--cap", "2", "--kl", "0.01", "--out", str(out)]) == 0
    expected = propose(truth, swarm.weights, cap=2.0, kl=0.01)
    np.testing.assert_array_equal(read_csv(out)["runtime_count"], expected.counts)


def test_cli_optimize_reads_a_fit_json_whose_keys_were_sorted(tmp_path, synthetic_problem):
    swarm = synthetic_problem.swarm
    # Observation ids in the reverse of their sorted order, as a JSON tool that sorts object keys would reorder them.
    swarm = dataclasses.replace(swarm, runs=tuple(f"run_{len(swarm.runs) - row:03d}" for row in range(len(swarm.runs))))
    inputs = write_own_swarm(tmp_path, swarm, synthetic_problem.objective, synthetic_problem.anchors)
    original, resorted = tmp_path / "fit.json", tmp_path / "sorted.json"
    write_generating_fit(original, swarm, synthetic_problem.truth)
    resorted.write_text(json.dumps(json.loads(original.read_text()), indent=2, sort_keys=True))
    proposals = {}
    for path in (original, resorted):
        out = tmp_path / f"{path.stem}_mixture.csv"
        assert main(["optimize", "--fit", str(path), *inputs, "--out", str(out)]) == 0
        proposals[path] = read_csv(out)
    pd.testing.assert_frame_equal(proposals[resorted], proposals[original], check_exact=True)


def test_cli_optimize_refuses_tables_at_another_target_budget(tmp_path, synthetic_problem, capsys):
    swarm = synthetic_problem.swarm
    inputs = write_own_swarm(tmp_path, swarm, synthetic_problem.objective, synthetic_problem.anchors)
    fit_path, out = tmp_path / "fit.json", tmp_path / "mixture.csv"
    write_generating_fit(fit_path, swarm, synthetic_problem.truth)
    # The fit's tables at a thousand times its budget, where every bucket materializes a thousand times the epochs.
    other_budget = [*inputs[:-1], str(1000 * TARGET_BUDGET)]
    assert main(["optimize", "--fit", str(fit_path), *other_budget, "--out", str(out)]) == INPUT_ERROR_STATUS
    error = capsys.readouterr().err.splitlines()
    assert len(error) == 1 and error[0].startswith("mariner optimize: error:")
    assert not out.exists()


def test_cli_evaluate_scores_a_fit_only_at_the_main_swarm_s_target_budget(tmp_path, release_snapshot, capsys):
    # The miniature snapshot's main swarm as one's own tables: its pools of 400, 200 and 100 tokens give its 2, 4 and
    # 8 epochs per unit weight at a target budget of 800 tokens.
    data = release_snapshot / "data"
    own = ["--runs", str(data / f"swarm_{MAIN_SWARM}" / f"swarm_{MAIN_SWARM}.csv")]
    own += ["--partition", str(data / "partition" / "partition.csv"), "--tasks", str(data / "tasks" / "tasks.csv")]
    fits = {budget: tmp_path / f"fit_{budget}.json" for budget in (800, 8000)}
    for budget, path in fits.items():
        assert main(["fit", *own, "--target-budget", str(budget), "--objective", "obe_easy", "--out", str(path)]) == 0
    capsys.readouterr()
    assert main(["evaluate", "--fit", str(fits[800]), "--data", str(release_snapshot)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 2  # the held-out exposures e0 and e1
    # At ten times the budget, the fit predicts each mixture at ten times the epochs that the held-out runs trained.
    assert main(["evaluate", "--fit", str(fits[8000]), "--data", str(release_snapshot)]) == INPUT_ERROR_STATUS
    captured = capsys.readouterr()
    error = captured.err.splitlines()
    assert not captured.out and len(error) == 1 and error[0].startswith("mariner evaluate: error:")


def test_cli_predict_names_missing_buckets(tmp_path, synthetic_problem, capsys):
    swarm = synthetic_problem.swarm
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(synthetic_problem.truth.to_json()))
    mixtures = tmp_path / "mixtures.csv"
    pd.DataFrame(bucket_columns("w__", swarm.buckets[:5], swarm.weights)).to_csv(mixtures, index=False)
    out = tmp_path / "predicted.csv"
    assert main(["predict", "--fit", str(fit_path), "--mixtures", str(mixtures), "--out", str(out)]) != 0
    error = capsys.readouterr().err
    assert "w__b5" in error and "w__b4" not in error
    assert not out.exists()


def test_cli_cross_validate_prints_selection_metrics_of_each_repeat(tmp_path, synthetic_problem, capsys):
    swarm, objective = synthetic_problem.swarm, synthetic_problem.objective
    one_task = dataclasses.replace(objective, tasks=objective.tasks[:1], weights=objective.weights[:1])
    inputs = write_own_swarm(tmp_path, swarm, one_task, synthetic_problem.anchors)
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
        expected = selection_metrics(rows["predicted"].to_numpy(), rows["measured"].to_numpy())
        assert metrics == dataclasses.asdict(expected)


def test_cli_olmix_writes_the_laws_of_its_tables_and_their_proposal(tmp_path, synthetic_problem):
    swarm, objective = synthetic_problem.swarm, synthetic_problem.objective
    inputs = write_own_swarm(tmp_path, swarm, objective, synthetic_problem.anchors)
    out, kl = tmp_path / "olmix.csv", 0.05
    command = ["olmix", *inputs, "--objective", objective.name, "--kl", str(kl), "--cap", str(OLMIX_CAP)]
    assert main([*command, "--out", str(out)]) == 0
    # The laws that fit_laws gives for the swarm of the tables, read as the command reads them.
    partition = read_partition(tmp_path / "partition.csv", TARGET_BUDGET)
    tables = swarm_from_runs(read_csv(tmp_path / "runs.csv"), partition, objective.tasks)
    laws = olmix.fit_laws(tables, objective)
    written = json.loads((tmp_path / "olmix.laws.json").read_text())
    assert (written["objective"], written["buckets"]) == (objective.name, list(swarm.buckets))
    expected = [
        {"task": task, "log_c": law.log_c, "coefficients": law.coefficients.tolist(), "huber_loss": law.huber_loss}
        for task, law in zip(objective.tasks, laws, strict=True)
    ]
    assert written["tasks"] == expected
    proposal = olmix.propose(laws, objective, partition.epochs_per_unit_weight, OLMIX_CAP, kl)
    np.testing.assert_array_equal(read_csv(out)["runtime_count"], proposal.counts)


def test_cli_design_counts_data_seeds_up_from_seed_and_repeats_the_calibration_mixture(tmp_path):
    partition, out = tmp_path / "partition.csv", tmp_path / "design.csv"
    pd.DataFrame({"bucket_id": ["b0", "b1", "b2"], "pool_tokens": [400.0, 200.0, 100.0]}).to_csv(partition, index=False)
    command = ["design", "--partition", str(partition), "--target-budget", "500", "--candidates", "3", "--seed", "11"]
    assert main([*command, "--repeats", "2", "--unimax-budget-fraction", "0.8", "--out", str(out)]) == 0
    design = read_csv(out)
    np.testing.assert_array_equal(design["data_seed"], 11 + np.arange(len(design)))
    repeats = design[design["design_kind"] == REPEAT]
    assert list(repeats["run_label"]) == ["proportional_repeat_00", "proportional_repeat_01"]
    assert not repeats["fit_inclusion"].any()
    # The repeats train the calibration run's mixture, which the anchors require of them.
    mixture = design.columns[design.columns.str.match("^(w|pf)__")]
    calibration = design.loc[design["run_label"] == CALIBRATION_RUN, mixture].to_numpy()
    np.testing.assert_array_equal(repeats[mixture].to_numpy(), np.repeat(calibration, 2, axis=0))


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
    design, released = read_csv(out), release_runs(snapshot, MAIN_SWARM)
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
