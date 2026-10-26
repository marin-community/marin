# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib", "matplotlib"]
# ///
"""Read frozen full fits; audit sparse observations and conditional convexity."""

import dataclasses
import itertools
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_policy_banks_20260917 as validation,
)

LABELS = {"hs17_mariner": "MARINER", "hs17_log1": "Logarithmic", "hs17_bounded_exp_c1": "Bounded exponential"}
COLORS = {"hs17_mariner": "#009E73", "hs17_log1": "#0072B2", "hs17_bounded_exp_c1": "#D55E00"}


def complete_dose_predictions(output: Path) -> pd.DataFrame:
    panel = validation.bench.load_panel(validation.DOSE_PANEL)
    manifest, query = validation.dose_query()
    _, atomic_bank, runs = validation.bank_tables()
    results = pd.read_csv(validation.builder.EPOCH_DOSE_RESULTS)
    meta = manifest.merge(results, on=["run_order", "run_name"], validate="one_to_one")
    source = runs[runs.source.eq("conditional_epoch_dose_response")][["source_row_id", "coordinate_id"]]
    meta = meta.merge(source, left_on="run_name", right_on="source_row_id", how="left", validate="one_to_one")
    distance = np.max(np.abs(query.weights[:, None, :] - panel.features.weights[None, :, :]), axis=2).min(axis=1)
    meta["in_fit_panel"] = distance <= 1e-8
    assert meta.in_fit_panel.sum() == 40 and meta.coordinate_id.notna().sum() == 237
    frames = []
    anchor = int(np.flatnonzero(meta.run_name.eq("p000_proportional_anchor"))[0])
    for target in ("uncheatable", "table9"):
        group = panel.group(target)
        source_path = (
            validation.builder.EPOCH_DOSE_UNCHEATABLE_COMPONENTS
            if target == "uncheatable"
            else validation.builder.EPOCH_DOSE_TABLE9_COMPONENTS
        )
        source_atomic = (
            pd.read_csv(source_path)
            .pivot(index="run_name", columns="component", values="bpb")
            .reindex(index=meta.run_name, columns=group.components)
        )
        measured = source_atomic.to_numpy() @ group.aggregation_weights
        atomic = (
            atomic_bank[atomic_bank.panel.eq(panel.name) & atomic_bank.target.eq(target)]
            .pivot(index="coordinate_id", columns="component", values="bpb_mean")
            .reindex(columns=group.components)
        )
        bank_measured = atomic.reindex(meta.coordinate_id).to_numpy() @ group.aggregation_weights
        primary = np.isfinite(bank_measured)
        # Complete atomic bank rows retain their frozen, audited endpoint versions.
        measured[primary] = bank_measured[primary]
        for model in validation.MODEL_IDS:
            payloads = [
                validation.bench.load_shard(
                    validation.fit_path(output, model, panel.name, target, i).with_suffix(".npz")
                )
                for i in range(len(group.components))
            ]
            predicted = np.stack([item["dose_prediction"] for item in payloads], axis=1) @ group.aggregation_weights
            frame = meta[
                [
                    "run_name",
                    "coordinate_id",
                    "focal_domain",
                    "epoch_multiplier",
                    "target_simulated_epochs",
                    "focal_weight",
                    "complement_scale",
                    "in_fit_panel",
                    "trainer_seed",
                    "data_seed",
                    "training_wandb_run_id",
                ]
            ].copy()
            frame["model"], frame["target"] = model, target
            frame["measured"], frame["prediction"] = measured, predicted
            frame["observed_complete"] = np.isfinite(measured)
            frame["measured_delta"], frame["predicted_delta"] = (
                measured - measured[anchor],
                predicted - predicted[anchor],
            )
            for i, bucket in enumerate(panel.buckets):
                frame[f"weight::{bucket}"] = query.weights[:, i]
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "dose_predictions.csv", index=False)
    return result


def dose_metrics(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    records = []
    for (model, target, domain), own in frame[~frame.in_fit_panel & frame.observed_complete].groupby(
        ["model", "target", "focal_domain"]
    ):
        curve = own.groupby("target_simulated_epochs", as_index=False).agg(
            measured=("measured", "mean"),
            prediction=("prediction", "mean"),
            measured_delta=("measured_delta", "mean"),
            predicted_delta=("predicted_delta", "mean"),
        )
        x, y, p = (curve[key].to_numpy() for key in ("target_simulated_epochs", "measured", "prediction"))
        optimum, selected = np.argmin(y), np.argmin(p)
        tail = np.arange(len(x)) >= len(x) // 2
        records.append(
            {
                "model": model,
                "target": target,
                "focal_domain": domain,
                "n": len(x),
                "rmse": np.sqrt(np.mean((p - y) ** 2)),
                "spearman": validation.safe_rank(y, p),
                "change_rmse": np.sqrt(np.mean((np.diff(p) - np.diff(y)) ** 2)) if len(x) > 1 else np.nan,
                "adjacent_sign_agreement": np.mean(np.sign(np.diff(p)) == np.sign(np.diff(y))) if len(x) > 1 else np.nan,
                "relative_rmse": np.sqrt(np.mean((curve.predicted_delta - curve.measured_delta) ** 2)),
                "measured_min_epoch": x[optimum],
                "predicted_min_epoch": x[selected],
                "minimum_grid_distance": abs(int(optimum) - int(selected)),
                "minimum_in_one_neighbor": abs(int(optimum) - int(selected)) <= 1,
                "regret": y[selected] - y.min(),
                "tail_bias": np.mean(p[tail] - y[tail]),
                "tail_rmse": np.sqrt(np.mean((p[tail] - y[tail]) ** 2)),
                "measured_tail_change": y[-1] - y[optimum],
                "predicted_tail_change": p[-1] - p[optimum],
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(output / "dose_curve_metrics.csv", index=False)
    # Shape scores require at least three measured nonoverlapping epochs.
    result[result.n >= 3].groupby(["model", "target"]).agg(
        curves=("n", "size"),
        mean_epochs=("n", "mean"),
        rmse=("rmse", "mean"),
        change_rmse=("change_rmse", "mean"),
        relative_rmse=("relative_rmse", "mean"),
        adjacent_sign_agreement=("adjacent_sign_agreement", "mean"),
        regret=("regret", "mean"),
        tail_bias=("tail_bias", "mean"),
        tail_rmse=("tail_rmse", "mean"),
        minimum_in_one_neighbor=("minimum_in_one_neighbor", "mean"),
    ).reset_index().to_csv(output / "dose_summary.csv", index=False)
    return result


def affine_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the shared proportional control at its correct focal exposure."""
    panel = validation.bench.load_panel(validation.DOSE_PANEL)
    records = []
    weight_columns = [f"weight::{bucket}" for bucket in panel.buckets]
    for _, own in frame.groupby(["model", "target"]):
        anchor = own[own.run_name.eq("p000_proportional_anchor")].iloc[0]
        for domain, curve in own[own.focal_domain.notna()].groupby("focal_domain"):
            i = panel.buckets.index(domain)
            control = anchor.copy()
            control["focal_domain"] = domain
            control["focal_weight"] = anchor[f"weight::{domain}"]
            control["epoch_multiplier"] = 1.0
            control["target_simulated_epochs"] = control["focal_weight"] * panel.features.inventory[i]
            full = pd.concat([curve, pd.DataFrame([control])], ignore_index=True)
            x = full.focal_weight.to_numpy()
            design = np.column_stack([np.ones(len(x)), x])
            weights = full[weight_columns].to_numpy(float)
            fitted = design @ np.linalg.lstsq(design, weights, rcond=None)[0]
            affine_error = np.max(np.abs(fitted - weights))
            if affine_error > 1e-8:
                raise ValueError(f"Dose path is not affine: {domain}, residual {affine_error}")
            full["affine_max_abs_error"] = affine_error
            # Use the model's current inventory consistently, retaining the historical manifest value.
            # The archived inventory differs slightly from the canonical input on some buckets.
            full["manifest_target_simulated_epochs"] = full.target_simulated_epochs
            full["target_simulated_epochs"] = x * panel.features.inventory[i]
            full["manifest_epoch_difference"] = full.target_simulated_epochs - full.manifest_target_simulated_epochs
            records.append(full)
    return pd.concat(records, ignore_index=True)


def convexity_audit(curves: pd.DataFrame, output: Path) -> pd.DataFrame:
    panel = validation.bench.load_panel(validation.DOSE_PANEL)
    records, empirical_noise = [], []
    # Repeats in these curves are sparse; pool only same-coordinate residuals, without borrowing fitted error.
    for target, own in curves[curves.model.eq(validation.MODEL_IDS[0])].groupby("target"):
        unique = own.drop_duplicates("run_name")
        weight_cols = [column for column in unique if column.startswith("weight::")]
        unique = unique.assign(key=[tuple(np.round(row, 12)) for row in unique[weight_cols].to_numpy()])
        sum_squared, dof, repeated = 0.0, 0, 0
        for _, group in unique[unique.observed_complete].groupby("key"):
            if len(group) < 2:
                continue
            sum_squared += np.sum((group.measured - group.measured.mean()) ** 2)
            dof += len(group) - 1
            repeated += 1
        empirical_noise.append(
            {
                "target": target,
                "repeat_coordinates": repeated,
                "degrees_of_freedom": dof,
                "pooled_same_coordinate_sd": np.sqrt(sum_squared / dof) if dof else np.nan,
                "external_panel_repeat_sd": panel.repeat_sd[target],
            }
        )
    pd.DataFrame(empirical_noise).to_csv(output / "dose_repeat_noise.csv", index=False)
    for (model, target, domain), own in curves.groupby(["model", "target", "focal_domain"]):
        grouped = (
            own[own.observed_complete]
            .groupby("target_simulated_epochs", as_index=False)
            .agg(
                measured=("measured", "mean"),
                prediction=("prediction", "mean"),
                count=("measured", "size"),
                in_fit_panel=("in_fit_panel", "max"),
            )
        )
        x = grouped.target_simulated_epochs.to_numpy()
        for i, j, k in itertools.combinations(range(len(grouped)), 3):
            fraction = (x[j] - x[i]) / (x[k] - x[i])
            y, p = grouped.measured.to_numpy(), grouped.prediction.to_numpy()
            residual = y[j] - (1 - fraction) * y[i] - fraction * y[k]
            prediction_residual = p[j] - (1 - fraction) * p[i] - fraction * p[k]
            counts = grouped["count"].to_numpy()
            assumed_sd = panel.repeat_sd[target] * np.sqrt(
                1 / counts[j] + (1 - fraction) ** 2 / counts[i] + fraction**2 / counts[k]
            )
            records.append(
                {
                    "model": model,
                    "target": target,
                    "focal_domain": domain,
                    "x_left": x[i],
                    "x_middle": x[j],
                    "x_right": x[k],
                    "measured_convexity_violation": residual,
                    "prediction_convexity_violation": prediction_residual,
                    "adjacent": j == i + 1 and k == j + 1,
                    "all_heldout": not bool(grouped.in_fit_panel.iloc[[i, j, k]].any()),
                    "raw_secant_decrease": (y[j] - y[i]) / (x[j] - x[i]) - (y[k] - y[j]) / (x[k] - x[j]),
                    "assumed_violation_sd": assumed_sd,
                    "assumed_noise_z": residual / assumed_sd,
                    "positive_after_2sd": residual > 2 * assumed_sd,
                    "affine_max_abs_error": own.affine_max_abs_error.max(),
                }
            )
    result = pd.DataFrame(records)
    result.to_csv(output / "dose_convexity_triples.csv", index=False)
    for basis, selected in (
        ("adjacent", result[result.adjacent]),
        ("heldout_adjacent", result[result.adjacent & result.all_heldout]),
    ):
        summary = selected.groupby(["model", "target", "focal_domain"]).agg(
            triples=("x_left", "size"),
            max_measured_violation=("measured_convexity_violation", "max"),
            max_prediction_violation=("prediction_convexity_violation", "max"),
            positive_after_2sd=("positive_after_2sd", "sum"),
            max_assumed_z=("assumed_noise_z", "max"),
        )
        summary.reset_index().to_csv(output / f"dose_convexity_{basis}.csv", index=False)
    return result


def plot_curves(curves: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    own = curves[curves.target.eq("uncheatable")]
    domains = sorted(own.focal_domain.unique())
    fig, axes = plt.subplots(6, 7, figsize=(22, 16), constrained_layout=True)
    for ax, domain in zip(axes.flat, domains, strict=False):
        for model in validation.MODEL_IDS:
            curve = (
                own[own.focal_domain.eq(domain) & own.model.eq(model)]
                .groupby("target_simulated_epochs", as_index=False)
                .agg(
                    measured=("measured", "mean"),
                    prediction=("prediction", "mean"),
                    in_fit_panel=("in_fit_panel", "max"),
                )
            )
            ax.plot(
                curve.target_simulated_epochs, curve.prediction, color=COLORS[model], label=LABELS[model], linewidth=1.5
            )
            if model == validation.MODEL_IDS[0]:
                mask = curve.in_fit_panel
                ax.scatter(
                    curve.target_simulated_epochs[~mask],
                    curve.measured[~mask],
                    s=15,
                    color="#333333",
                    label="Held-out observations",
                    zorder=5,
                )
                ax.scatter(
                    curve.target_simulated_epochs[mask],
                    curve.measured[mask],
                    s=20,
                    facecolors="white",
                    edgecolors="#333333",
                    label="Fit controls",
                    zorder=5,
                )
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_title(domain.replace("dolma3_cc/", "CC/").replace("dolma3_", "").replace("dolmino_", ""), fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(domains) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_axis = axes[-1, 4]
    legend_axis.set_visible(True)
    legend_axis.axis("off")
    legend_axis.legend(handles, labels, loc="center left", frameon=False, fontsize=12)
    fig.supxlabel("Focal-bucket epochs (symlog); other mixture shares shrink proportionally", y=-0.025)
    fig.supylabel("Uncheatable BPB (current component weights)")
    fig.suptitle("Full-fit predictions on conditional repetition paths — dots are archived runs", fontsize=15)
    fig.savefig(output / "dose_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(output / "dose_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def audit_fits(output: Path) -> dict:
    fits = pd.read_csv(output / "fits.csv")
    checks = []
    for row in fits.itertuples(index=False):
        path = Path(row.fit_path)
        fitted = joblib.load(path)
        saved = validation.bench.load_shard(path.with_suffix(".npz"))
        panel = validation.bench.load_panel(row.panel)
        features = dataclasses.replace(panel.features, component=row.component)
        model = validation.make_model(row.model, features)
        _, query = validation.bank_query(panel, row.target)
        query = dataclasses.replace(query, component=row.component)
        design = model.base.design(query, fitted.shape)
        link = validation.candidates.base.LinkKind(str(fitted.diagnostics["link"]))
        spec = model._spec(fitted.shape, link, float(fitted.diagnostics["kappa"]))
        direct = validation.candidates.base.predict_head(fitted.head, design.values, spec)
        np.testing.assert_allclose(direct, saved["prediction"], rtol=1e-12, atol=1e-12)
        winner = np.unravel_index(np.argmin(fitted.cv_table), fitted.cv_table.shape)
        assert fitted.shape == model.base.shapes[winner[0]]
        assert fitted.ridge == model.base.ridge_grid[winner[1]]
        checks.append({"fit": str(path), "direct_prediction_reproduced": True, "shape_ridge_cv_argmin": True})
    pd.DataFrame(checks).to_csv(output / "fit_reconstruction_audit.csv", index=False)
    return {"fits": len(checks), "direct_prediction_reproduced": True, "shape_ridge_cv_argmin": True}


def main() -> None:
    output = validation.OUTPUT
    validation.verify_sources(output)
    prediction, checks = validation.report_banks(output)
    validation.report_run_noise(prediction, output)
    dose = complete_dose_predictions(output)
    metrics = dose_metrics(dose, output)
    curves = affine_curves(dose)
    curves.to_csv(output / "dose_affine_curves.csv", index=False)
    triples = convexity_audit(curves, output)
    plot_curves(curves, output)
    fit_check = audit_fits(output)
    validation.json_write(
        output / "verification.json",
        {
            "expected_fits": 522,
            "fit_audit": fit_check,
            "bank_inputs": checks,
            "source_identity": validation.verify_sources(output),
            "report_source_hash": validation.sha256(Path(__file__)),
            "complete_dose_predictions": len(dose),
            "observed_dose_counts": (
                dose.groupby(["model", "target", "in_fit_panel"])
                .observed_complete.sum()
                .reset_index()
                .to_dict("records")
            ),
            "dose_curve_metrics": len(metrics),
            "convexity_triples": len(triples),
            "complete": True,
        },
    )
    print(json.dumps({"complete": True, "fits": fit_check["fits"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
