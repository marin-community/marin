# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0", "matplotlib==3.10.8",
# ]
# ///
"""Assess matched phase gains, repeated optima and later branch-bank calibration."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import calibrate_delphi_phase_frontier_20260910 as audit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import write_json_atomic
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, t
from threadpoolctl import threadpool_limits

OUTPUT = audit.OUTPUT
REFERENCE = audit.REFERENCE
CELL_NAMES = {"19b0c3282a8c": "Uncheatable anchor", "17a03f4d49cd": "Table-9 anchor"}
MODELS = ("mariner", "hpr", "replacement")
SEED = 20260910


def frozen_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.copyfile(source, destination)


def matched_phase() -> None:
    pairs = pd.read_csv(OUTPUT / "inputs/mirror_pairs.csv")
    meta = pd.read_csv(OUTPUT / "inputs/frontier_metadata.csv").set_index("heldout_id")
    directions, contrasts, centers, cell_selections = [], [], [], []
    for context in ("canonical", "expanded0"):
        for objective in audit.TARGETS:
            frame = pd.read_csv(OUTPUT / "fits" / context / f"{objective}_predictions.csv")
            frame = frame[frame.cohort.eq("archive")].set_index("archive_id")
            for cell, group in pairs.groupby("cell_id"):
                example = json.loads(group.iloc[0].left_row_ids_json)[0]
                aggregate = frame.loc[example, "aggregate_id"]
                rows = frame[frame.aggregate_id.eq(aggregate)].copy()
                assert not rows.in_train.any()
                for model in MODELS:
                    population = rows.groupby("policy_id", as_index=False).agg(
                        measured=("measured", "mean"),
                        mariner=("mariner", "first"),
                        hpr=("hpr", "first"),
                        replacement=("replacement", "first"),
                        tied=("tied", "first"),
                    )
                    # Equal predictions within 1e-12 prefer tied rather than arbitrary phase order.
                    near = population[population[model] <= population[model].min() + 1e-12]
                    selected = near.sort_values(["tied", "policy_id"], ascending=[False, True]).iloc[0]
                    cell_selections.append(
                        dict(
                            context=context,
                            objective=objective,
                            cell=cell,
                            model=model,
                            measured=selected.measured,
                            predicted=selected[model],
                            optimism=selected.measured - selected[model],
                            tied=selected.tied,
                            regret=selected.measured - population.measured.min(),
                            policy_id=selected.policy_id,
                            within_cell_spearman=(
                                float(spearmanr(population.measured, population[model]).statistic)
                                if np.std(population[model]) > 1e-12
                                else np.nan
                            ),
                        )
                    )
                for source, block in rows.groupby("source"):
                    tied = block[block.tied]
                    if tied.empty:
                        continue
                    centers.append(
                        dict(
                            context=context,
                            objective=objective,
                            cell=cell,
                            source=source,
                            runs=len(tied),
                            mean=tied.measured.mean(),
                            sd=tied.measured.std(),
                            mariner=tied.mariner.iloc[0],
                            hpr=tied.hpr.iloc[0],
                        )
                    )
                rows["matched_seed"] = rows.index.map(meta.dry_run_data_seed)
                rows["matched_trainer"] = rows.index.map(meta.dry_run_trainer_seed)
                for row_id, row in rows[~rows.tied & rows.matched_seed.notna()].iterrows():
                    controls = rows[
                        rows.tied
                        & rows.source.eq(row.source)
                        & rows.matched_seed.eq(row.matched_seed)
                        & rows.matched_trainer.eq(row.matched_trainer)
                    ]
                    assert len(controls) == 1, (row_id, len(controls))
                    control = controls.iloc[0]
                    predicted = -row.phase_prediction
                    gain = control.measured - row.measured
                    contrasts.append(
                        dict(
                            context=context,
                            objective=objective,
                            cell=cell,
                            row_id=row_id,
                            source=row.source,
                            seed=row.matched_seed,
                            control_id=controls.index[0],
                            observed_gain=gain,
                            predicted_gain=predicted,
                            gain_optimism=predicted - gain,
                            regret=max(0.0, -gain) if predicted > 0 else max(0.0, gain),
                            prefer_phase=predicted > 0,
                        )
                    )
                for pair in group.itertuples():
                    left = frame.loc[json.loads(pair.left_row_ids_json)]
                    right = frame.loc[json.loads(pair.right_row_ids_json)]
                    assert len(left) == len(right) == 1
                    lp, rp = left.iloc[0].phase_prediction, right.iloc[0].phase_prediction
                    ly, ry = left.iloc[0].measured, right.iloc[0].measured
                    directions.append(
                        dict(
                            context=context,
                            objective=objective,
                            cell=cell,
                            left_id=left.index[0],
                            right_id=right.index[0],
                            predicted_difference=lp - rp,
                            measured_difference=ly - ry,
                            sign_correct=(lp - rp) * (ly - ry) > 0,
                            same_table9_source=left.iloc[0].table9_source == right.iloc[0].table9_source,
                            regret=ly - min(ly, ry) if lp < rp else ry - min(ly, ry),
                        )
                    )
    contrast = pd.DataFrame(contrasts)
    direction = pd.DataFrame(directions)
    contrast.to_csv(OUTPUT / "matched_seed_phase_gains.csv", index=False)
    direction.to_csv(OUTPUT / "mirror_predictions.csv", index=False)
    source_sensitivity = []
    for keys, group in direction.groupby(["context", "objective", "cell"]):
        for population, subset in [("all", group), ("same_metric_source", group[group.same_table9_source])]:
            source_sensitivity.append(
                dict(
                    context=keys[0],
                    objective=keys[1],
                    cell=keys[2],
                    population=population,
                    pairs=len(subset),
                    sign_accuracy=subset.sign_correct.mean(),
                    rmse=np.sqrt(np.mean((subset.predicted_difference - subset.measured_difference) ** 2)),
                    zero_rmse=np.sqrt(np.mean(subset.measured_difference**2)),
                )
            )
    pd.DataFrame(source_sensitivity).to_csv(OUTPUT / "mirror_source_sensitivity.csv", index=False)
    pd.DataFrame(centers).to_csv(OUTPUT / "frontier_center_calibration.csv", index=False)
    pd.DataFrame(cell_selections).to_csv(OUTPUT / "fixed_aggregate_selections.csv", index=False)
    summaries = []
    for keys, group in contrast.groupby(["context", "objective", "cell"]):
        source_groups = [list(block.groupby("seed")) for _, block in group.groupby("source")]
        rng = np.random.default_rng(SEED)
        optimism = []
        for _ in range(2000):
            sample = pd.concat(
                [blocks[i][1] for blocks in source_groups for i in rng.integers(0, len(blocks), len(blocks))]
            )
            optimism.append(sample.gain_optimism.mean())
        pair = direction[(direction.context.eq(keys[0])) & direction.objective.eq(keys[1]) & direction.cell.eq(keys[2])]
        observed, predicted = group.observed_gain.to_numpy(), group.predicted_gain.to_numpy()
        summaries.append(
            dict(
                context=keys[0],
                objective=keys[1],
                cell=keys[2],
                paired_runs=len(group),
                seed_blocks=sum(len(x) for x in source_groups),
                mean_observed_gain=observed.mean(),
                mean_predicted_gain=predicted.mean(),
                gain_rmse=np.sqrt(np.mean((observed - predicted) ** 2)),
                zero_gain_rmse=np.sqrt(np.mean(observed**2)),
                mean_gain_optimism=group.gain_optimism.mean(),
                gain_optimism_low=np.quantile(optimism, 0.025),
                gain_optimism_high=np.quantile(optimism, 0.975),
                mean_decision_regret=group.regret.mean(),
                gain_spearman=spearmanr(observed, predicted).statistic,
                mirror_pairs=len(pair),
                mirror_sign_accuracy=pair.sign_correct.mean(),
                mirror_rmse=np.sqrt(np.mean((pair.predicted_difference - pair.measured_difference) ** 2)),
                mirror_zero_rmse=np.sqrt(np.mean(pair.measured_difference**2)),
                mirror_decision_regret=pair.regret.mean(),
            )
        )
    pd.DataFrame(summaries).to_csv(OUTPUT / "phase_gain_calibration.csv", index=False)


def recent_optima() -> None:
    records = []
    for objective in audit.TARGETS:
        frame = pd.read_csv(OUTPUT / "fits/canonical" / f"{objective}_predictions.csv")
        frame = frame[frame.source.str.contains("frozen_procedure_validation|fairness_repeats")]
        for candidate, group in frame.groupby("candidate"):
            if len(group) != 3:
                raise ValueError((candidate, len(group)))
            prediction = group.mariner.iloc[0]
            mean, sd = group.measured.mean(), group.measured.std()
            half = t.ppf(0.975, len(group) - 1) * sd / np.sqrt(len(group))
            records.append(
                dict(
                    objective=objective,
                    candidate=candidate,
                    runs=len(group),
                    predicted=prediction,
                    mean=mean,
                    sd=sd,
                    optimism=mean - prediction,
                    mean_low=mean - half,
                    mean_high=mean + half,
                    target=group.target.iloc[0],
                )
            )
    pd.DataFrame(records).to_csv(OUTPUT / "repeated_policy_calibration.csv", index=False)


def later_branches() -> None:
    source = REFERENCE / "fixed_checkpoint_branch_wspu_20260907/data"
    inputs = OUTPUT / "branch_inputs"
    for name in ("rows.csv", "arrays.npz", "manifest.json", "validation.json", "README.md"):
        frozen_copy(source / name, inputs / name)
    frame, coordinates, module, _ = audit.data()
    rows = pd.read_csv(inputs / "rows.csv")
    with np.load(inputs / "arrays.npz") as payload:
        arrays = {k: payload[k] for k in payload.files}
    mariner = module.ObjectiveFit.from_json(json.loads((OUTPUT / "fits/mariner_uncheatable.json").read_text()))
    order = [list(arrays["bucket_names"]).index(b) for b in mariner.buckets]
    weights = np.stack([arrays["phase0_weight"][:, order], arrays["phase1_weight"][:, order]], axis=1)
    assert np.max(abs(weights.sum(axis=2) - 1)) < 1e-6
    weights /= weights.sum(axis=2, keepdims=True)
    aggregate = audit.ALPHA * weights[:, 0] + (1 - audit.ALPHA) * weights[:, 1]
    exposure_errors = [
        float(np.max(abs(arrays[f"phase{p}_epochs"][:, order] - fraction * weights[:, p] * mariner.inventory)))
        for p, fraction in [(0, audit.ALPHA), (1, 1 - audit.ALPHA)]
    ]
    relative_exposure_errors = [
        float(
            np.max(
                abs(arrays[f"phase{p}_epochs"][:, order] - fraction * weights[:, p] * mariner.inventory)
                / np.maximum(1.0, abs(arrays[f"phase{p}_epochs"][:, order]))
            )
        )
        for p, fraction in [(0, audit.ALPHA), (1, 1 - audit.ALPHA)]
    ]
    # The branch package used the six-decimal 0.905353 epoch anchor. MARINER
    # carries its unrounded value; the resulting relative difference is 5e-7.
    assert max(relative_exposure_errors) < 1e-6
    with (OUTPUT / "fits/canonical/uncheatable_hpr.pkl").open("rb") as handle:
        hpr = pickle.load(handle)
    prediction = hpr.predict(weights)
    tied = hpr.predict(np.repeat(aggregate[:, None, :], 2, axis=1))
    rows["mariner"] = mariner.predict(aggregate)
    rows["hpr"] = prediction
    rows["replacement"] = rows.mariner + prediction - tied
    rows["measured"] = arrays["target"]
    rows["policy_id"] = [audit.key(w) for w in weights]
    train = coordinates["weights"][frame.cohort.eq("canonical")]
    distance = cdist(weights.reshape(len(weights), -1), train.reshape(len(train), -1), "chebyshev").min(axis=1)
    rows["training_overlap"] = distance < 1e-9
    rows["physical_tied"] = np.max(abs(weights[:, 0] - weights[:, 1]), axis=1) < 1e-10
    records = []
    for (panel, state), group in rows[~rows.training_overlap & ~rows.is_tied_control].groupby(["panel", "state_id"]):
        group = group.groupby("policy_id", as_index=False).agg(
            measured=("measured", "mean"),
            mariner=("mariner", "first"),
            hpr=("hpr", "first"),
            replacement=("replacement", "first"),
        )
        for model in MODELS:
            for band, subset in [
                ("all", group),
                ("predicted_frontier20", group.nsmallest(max(1, int(np.ceil(0.2 * len(group)))), model)),
            ]:
                records.append(dict(panel=panel, state=state, model=model, band=band, **audit.summary(subset, model)))
    rows.to_csv(OUTPUT / "branch_predictions.csv", index=False)
    pd.DataFrame(records).to_csv(OUTPUT / "branch_calibration_metrics.csv", index=False)
    write_json_atomic(
        OUTPUT / "branch_checks.json",
        dict(
            rows=len(rows),
            unique_terminal_checkpoints=rows.terminal_checkpoint_uri.nunique(),
            overlapping_rows=int(rows.training_overlap.sum()),
            prefix_states=rows.state_id.nunique(),
            max_epoch_errors=exposure_errors,
            max_relative_epoch_errors=relative_exposure_errors,
            hashes={p.name: audit.sha(p) for p in inputs.iterdir() if p.is_file()},
        ),
    )


def checks() -> None:
    _, arrays, _, _ = audit.data()
    records = {}
    for context in ("canonical", "expanded0", "expanded1", "expanded2", "expanded3"):
        for target in audit.TARGETS:
            table = pd.read_csv(OUTPUT / "fits" / context / f"{target}_predictions.csv")
            train, test = table.in_train.to_numpy(), table.in_test.to_numpy()
            policy_distance = cdist(
                arrays["weights"][test].reshape(test.sum(), -1),
                arrays["weights"][train].reshape(train.sum(), -1),
                "chebyshev",
            ).min()
            assert policy_distance > 1e-9
            record = dict(policy_min_distance=float(policy_distance), train=int(train.sum()), test=int(test.sum()))
            if context != "canonical":
                agg_distance = cdist(arrays["aggregate"][test], arrays["aggregate"][train], "chebyshev").min()
                assert agg_distance > 1e-9
                record["aggregate_min_distance"] = float(agg_distance)
            records[f"{context}/{target}"] = record
    component_source = REFERENCE / "delphi_3e18_observed_components_20260724/observed_component_panel.csv"
    frozen_copy(component_source, OUTPUT / "component_checks/observed_component_panel.csv")
    components = pd.read_csv(OUTPUT / "component_checks/observed_component_panel.csv")
    definitions = pd.read_csv(OUTPUT / "inputs/objectives.csv")
    for target, column in audit.TARGETS.items():
        definition = definitions[definitions.objective.eq(target)]
        error = float(
            np.max(abs(components[definition.component].to_numpy() @ definition.weight.to_numpy() - components[column]))
        )
        assert error < 3e-6
        records[f"component_aggregation/{target}"] = dict(rows=len(components), max_abs_error=error)
    write_json_atomic(OUTPUT / "VALIDATION.json", records)


def plot() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    metrics = pd.read_csv(OUTPUT / "calibration_metrics.csv")
    comparisons = [
        ("MARINER / 1p", True, "mariner", "#0072B2"),
        ("HPR / 2p", False, "hpr", "#D55E00"),
        ("HPR + MARINER / 2p", False, "replacement", "#009E73"),
    ]
    for j, target in enumerate(audit.TARGETS):
        for name, tied, model, color in comparisons:
            m = metrics[
                (metrics.context.eq("canonical"))
                & metrics.scope.eq("all_sources")
                & metrics.objective.eq(target)
                & metrics.band.eq("predicted_frontier20")
                & metrics.tied.eq(tied)
                & metrics.model.eq(model)
            ].iloc[0]
            axes[0, j].barh(name, m.optimism, color=color)
            axes[0, j].annotate(
                f"{m.optimism:+.4f}",
                (m.optimism, name),
                xytext=(4 if m.optimism >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if m.optimism >= 0 else "right",
                fontsize=9,
            )
        axes[0, j].axvline(0, color="0.3", linewidth=0.8)
        axes[0, j].set_title(f"{target.title()}: predicted frontier")
        axes[0, j].set_xlabel("Mean optimism (observed - predicted BPB)")
        axes[0, j].margins(x=0.3)
        axes[0, j].xaxis.set_major_locator(MaxNLocator(nbins=4))
    gains = pd.read_csv(OUTPUT / "matched_seed_phase_gains.csv")
    for j, target in enumerate(audit.TARGETS):
        group = gains[gains.context.eq("canonical") & gains.objective.eq(target)]
        for cell, color in [("19b0c3282a8c", "#0072B2"), ("17a03f4d49cd", "#D55E00")]:
            x = group[group.cell.eq(cell)]
            axes[1, j].scatter(x.predicted_gain, x.observed_gain, s=13, alpha=0.5, color=color, label=CELL_NAMES[cell])
        values = np.r_[group.predicted_gain, group.observed_gain]
        limits = [float(values.min() - 0.002), float(values.max() + 0.002)]
        axes[1, j].plot(limits, limits, color="0.5", linestyle="--", linewidth=1)
        axes[1, j].axhline(0, color="0.8", linewidth=0.8)
        axes[1, j].axvline(0, color="0.8", linewidth=0.8)
        axes[1, j].set(
            xlabel="Predicted phase gain (BPB)",
            ylabel="Measured same-seed phase gain (BPB)",
            title=f"{target.title()}: fixed-aggregate contrasts",
        )
        axes[1, j].legend(frameon=False, fontsize=8)
    figure.suptitle("Delphi 3e18: frontier loss and paired phase-gain calibration", fontsize=13)
    figure.savefig(OUTPUT / "calibration_diagnostic.png", dpi=180)
    figure.savefig(OUTPUT / "calibration_diagnostic.svg")
    plt.close(figure)


def main() -> None:
    with threadpool_limits(limits=1):
        checks()
        recent_optima()
        later_branches()
        matched_phase()
        plot()


if __name__ == "__main__":
    main()
