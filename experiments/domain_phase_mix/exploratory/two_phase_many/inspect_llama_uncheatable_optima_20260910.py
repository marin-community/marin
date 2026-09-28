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
"""Optimize and inspect frozen Llama proposals without launching model training."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import audit_two_phase_link_softmax_solver_20260907 as softmax_solver
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optimize_two_phase_hpr_transfer_20260907 as transplant
import optimize_two_phase_link_transfer_20260907 as raw
import pandas as pd
import prepare_llama_uncheatable_optima_20260910 as preparation
from fit_two_phase_link_spines_20260907 import write_json_atomic
from threadpoolctl import threadpool_limits

OUTPUT = preparation.OUTPUT
REFERENCE = preparation.REFERENCE
REALIZED_ALPHA = {"60m": 3648 / 4577, "300m": 18304 / 22888}
SEED = 20260910


def load_single(scale: str) -> tuple[Any, Any, Any]:
    module, swarm, _, _, _ = preparation.single_inputs(scale)
    fit = module.ObjectiveFit.from_json(json.loads((OUTPUT / "fits" / scale / "mariner.json").read_text()))
    return module, swarm, fit


def one_phase(scale: str) -> None:
    path = OUTPUT / "proposals" / scale / "one_phase.json"
    if path.exists() and path.with_suffix(".csv").exists():
        print(f"reuse {scale} one-phase proposal", flush=True)
        return
    module, swarm, fit = load_single(scale)
    with threadpool_limits(limits=1):
        weights, starts = module.continuous_optimum(fit, swarm.weights)
        counts, diagnostics = module.runtime_policy(fit, weights)
    runtime = counts / module.MIXTURE_BLOCK_SIZE
    write_json_atomic(
        path,
        {
            "method": "authoritative MARINER; uncapped; KL=0",
            "weights": runtime.tolist(),
            "continuous_weights": weights.tolist(),
            "counts": counts.tolist(),
            "starts": starts,
            **diagnostics,
            "component_predictions": fit.predict_tasks(runtime)[0].tolist(),
            "weighted_floor": float(fit.weights @ np.asarray([t.head.floor for t in fit.tasks])),
            "source_sha256": preparation.sha(OUTPUT / "fits" / scale / "mariner.json"),
        },
    )
    pd.DataFrame(
        {"bucket": swarm.buckets, "count": counts, "weight": runtime, "epochs": runtime * swarm.inventory}
    ).to_csv(path.with_suffix(".csv"), index=False)
    print(f"{scale} MARINER: {diagnostics}", flush=True)


def hpr_model(scale: str) -> Any:
    cell = OUTPUT / "fits" / scale / "hpr"
    if scale == "300m" and not (cell / "model.pkl").exists():
        source = REFERENCE / "two_phase_link_transfer_20260907/controls/hierarchical_phase_replay/uncheatable/full"
        complete = json.loads((source / "complete.json").read_text())
        for name, digest in complete["sha256"].items():
            assert preparation.sha(source / name) == digest, name
        cell.mkdir(parents=True, exist_ok=True)
        for name in ("model.pkl", "fit.json", "complete.json"):
            shutil.copyfile(source / name, cell / name)
        write_json_atomic(
            cell / "reuse.json",
            {
                "source": str(source),
                "all_source_artifact_hashes_verified": True,
                "note": (
                    "Reuse the established 520-row Llama 200M HPR fit; "
                    "replace its aggregate prediction with current MARINER."
                ),
            },
        )
    with (cell / "model.pkl").open("rb") as handle:
        return pickle.load(handle)


def surface(scale: str) -> tuple[Any, Any, Any]:
    _, _, fit = load_single(scale)
    hpr = hpr_model(scale)
    response = transplant.hpr_response(hpr)
    assert tuple(hpr.dataset.domains) == fit.buckets
    assert np.max(abs(response.c0 + response.c1 - fit.inventory)) < 1e-9
    rate, power, threshold, intercept, coefficients, floor = fit._arrays()
    width = len(fit.buckets)
    model = raw.ObjectiveModel(
        fit.weights,
        rate,
        power,
        threshold,
        intercept,
        floor,
        coefficients[:, :width],
        coefficients[:, width:],
        np.zeros((len(fit.tasks), 2)),
        response.c0,
        response.c1,
    )
    return transplant.Surface(model, response, transplant.Arm.TRANSFER), hpr, fit


def batch_predict(model: Any, hpr: Any, fit: Any, weights: np.ndarray) -> np.ndarray:
    aggregate = model.phase_fraction * weights[:, 0] + (1 - model.phase_fraction) * weights[:, 1]
    return fit.predict(aggregate) + hpr.predict(weights) - hpr.predict(np.repeat(aggregate[:, None, :], 2, axis=1))


def verify_surface(model: Any, hpr: Any, fit: Any) -> dict:
    weights = hpr.dataset.weights
    reference = batch_predict(model, hpr, fit, weights)
    parity = float(np.max(abs(reference - np.asarray([model.value_gradient(w)[0] for w in weights]))))
    hpr_error = float(np.max(abs(hpr.predict(weights) - np.asarray([model.hpr.value_gradient(w)[0] for w in weights]))))
    rng = np.random.default_rng(SEED)
    derivative_error, tied_error = 0.0, 0.0
    for _ in range(6):
        policy = rng.dirichlet(np.full(model.buckets, 3.0), size=2)
        _, gradient, _ = model.value_gradient(policy)
        direction = rng.normal(size=policy.shape)
        direction -= direction.mean(axis=1, keepdims=True)
        direction /= np.linalg.norm(direction)
        step = 1e-6
        numerical = (
            model.value_gradient(policy + step * direction)[0] - model.value_gradient(policy - step * direction)[0]
        ) / (2 * step)
        derivative_error = max(derivative_error, abs(numerical - float(np.sum(gradient * direction))))
        tied_error = max(
            tied_error, abs(model.value_gradient(np.repeat(policy[:1], 2, axis=0))[0] - fit.predict(policy[:1])[0])
        )
    assert max(parity, hpr_error, tied_error) < 1e-8
    assert derivative_error < 2e-6
    return {
        "max_prediction_error": parity,
        "max_hpr_prediction_error": hpr_error,
        "max_tied_restriction_error": tied_error,
        "max_interior_tangent_gradient_error": derivative_error,
    }


def runtime_two(module: Any, model: Any, hpr: Any, fit: Any, weights: np.ndarray) -> tuple[np.ndarray, list]:
    maximum = np.full(model.buckets, module.MIXTURE_BLOCK_SIZE, dtype=int)
    counts = np.stack([module.constrained_counts(w, maximum) for w in weights])
    moves = []
    for cycle in range(50):
        total = 0
        for phase in range(2):

            def predict(rows: np.ndarray, phase: int = phase) -> np.ndarray:
                policies = np.repeat((counts / module.MIXTURE_BLOCK_SIZE)[None, :, :], len(rows), axis=0)
                policies[:, phase] = rows
                return batch_predict(model, hpr, fit, policies)

            counts[phase], steps = module.refine_counts(predict, counts[phase], maximum)
            total += steps
            moves.append({"cycle": cycle, "phase": phase, "moves": steps})
        if total == 0:
            assert np.all(counts.sum(axis=1) == module.MIXTURE_BLOCK_SIZE) and counts.min() >= 0
            return counts, moves
    raise RuntimeError("phase-wise exchange search did not converge")


def two_phase(scale: str) -> None:
    path = OUTPUT / "proposals" / scale / "two_phase.json"
    if path.exists() and path.with_suffix(".csv").exists():
        print(f"reuse {scale} two-phase proposal", flush=True)
        return
    module, swarm, fit = load_single(scale)
    model, hpr, _ = surface(scale)
    checks = verify_surface(model, hpr, fit)
    one = json.loads((path.parent / "one_phase.json").read_text())
    natural = 1 / fit.inventory
    natural /= natural.sum()
    rng = np.random.default_rng(SEED)
    rows = hpr.dataset.weights
    good = np.argsort(hpr.dataset.target)[:3]
    random = rng.choice(len(rows), 5, replace=False)
    starts = [
        ("one_phase_optimum", np.repeat(np.asarray(one["continuous_weights"])[None, :], 2, axis=0)),
        ("proportional", np.stack([natural, natural])),
    ]
    starts += [(f"observed_{i}", rows[i]) for i in good]
    starts += [(f"swarm_{i}", rows[i]) for i in random]
    starts += [(f"dirichlet_{i}", rng.dirichlet(np.full(model.buckets, 0.7), size=2)) for i in range(2)]
    records = []
    for name, start in starts:
        cache = path.parent / "solver" / f"slsqp_{name}.json"
        if cache.exists():
            record = json.loads(cache.read_text())
        else:
            _, record = raw.solve_start(model, start, False, name)
            write_json_atomic(cache, record)
        records.append({"solver": "slsqp", **record})
    candidates = sorted(records, key=lambda row: row["retained_bpb"])[:4]
    for index, candidate in enumerate(candidates):
        cache = path.parent / "solver" / f"softmax_{index}.json"
        if cache.exists():
            record = json.loads(cache.read_text())
        else:
            record = softmax_solver.optimize_start(model, np.asarray(candidate["weights"]), False, f"softmax_{index}")
            write_json_atomic(cache, record)
        records.append({"solver": "softmax", "retained_bpb": record["predicted_bpb"], **record})
    best = min(records, key=lambda row: row["retained_bpb"])
    continuous = np.asarray(best["weights"])
    counts, moves = runtime_two(module, model, hpr, fit, continuous)
    runtime = counts / module.MIXTURE_BLOCK_SIZE
    value, _, diagnostics = model.value_gradient(runtime)
    aggregate = model.phase_fraction * runtime[0] + (1 - model.phase_fraction) * runtime[1]
    epochs = aggregate * swarm.inventory
    nearest = np.min(
        0.8 * np.sum(abs(rows[:, 0] - runtime[0]), axis=1) / 2 + 0.2 * np.sum(abs(rows[:, 1] - runtime[1]), axis=1) / 2
    )
    actual_alpha = REALIZED_ALPHA[scale]
    actual_model = replace(
        model,
        aggregate=replace(model.aggregate, c0=actual_alpha * fit.inventory, c1=(1 - actual_alpha) * fit.inventory),
        hpr=replace(model.hpr, c0=actual_alpha * fit.inventory, c1=(1 - actual_alpha) * fit.inventory),
    )
    write_json_atomic(
        path,
        {
            "method": "HPR plus current MARINER aggregate replacement; uncapped; KL=0",
            "continuous_weights": continuous.tolist(),
            "weights": runtime.tolist(),
            "counts": counts.tolist(),
            "continuous_prediction": best["retained_bpb"],
            "runtime_prediction": value,
            "nominal_phase_fraction": 0.8,
            "realized_phase_fraction": actual_alpha,
            "prediction_at_realized_fraction": actual_model.value_gradient(runtime)[0],
            "phase_tv": float(np.abs(runtime[0] - runtime[1]).sum() / 2),
            "max_materialized_epoch": float(epochs.max()),
            "nearest_training_policy_tv": float(nearest),
            "exchange_moves": moves,
            "checks": checks,
            "best_solver": best["solver"],
            "successful_starts": sum(r["success"] for r in records),
            "starts": records,
            "fit_rows": len(rows),
            "predicted_gain_vs_1p_optimum": one["runtime_prediction"] - value,
            **diagnostics,
            "source_sha256": {
                "mariner": preparation.sha(OUTPUT / "fits" / scale / "mariner.json"),
                "hpr": preparation.sha(OUTPUT / "fits" / scale / "hpr/model.pkl"),
            },
        },
    )
    pd.DataFrame(
        {
            "bucket": swarm.buckets,
            "phase0_count": counts[0],
            "phase1_count": counts[1],
            "phase0_weight": runtime[0],
            "phase1_weight": runtime[1],
            "aggregate_weight": aggregate,
            "epochs": epochs,
            "realized_epochs": (actual_alpha * runtime[0] + (1 - actual_alpha) * runtime[1]) * swarm.inventory,
        }
    ).to_csv(path.with_suffix(".csv"), index=False)
    print(
        f"{scale} two-phase: {value:.6f}; aggregate {diagnostics['aggregate_bpb']:.6f}; "
        f"phase correction {diagnostics['hpr_contrast']:.6f}",
        flush=True,
    )


def summarize() -> None:
    frontier = REFERENCE / "delphi_phase_frontier_calibration_20260910/inputs"
    observed = pd.read_csv(frontier / "observations.csv")
    with np.load(frontier / "coordinates.npz") as arrays:
        coordinates = {k: arrays[k] for k in arrays.files}
    delphi_buckets = pd.read_csv(frontier / "buckets.csv").bucket.tolist()
    summaries, profiles, frontier_rows = [], [], []
    for phase in (1, 2):
        data = observed[observed.external & observed.uncheatable.notna() & observed.tied.eq(phase == 1)]
        groups = data.groupby("policy_id").agg(
            measured=("uncheatable", "mean"),
            repeats=("uncheatable", "count"),
            source=("source", "first"),
            candidate=("candidate", "first"),
        )
        top = groups.nsmallest(20, "measured")
        index = data.reset_index().drop_duplicates("policy_id").set_index("policy_id").loc[top.index, "index"].to_numpy()
        profile = coordinates["aggregate"][index].mean(axis=0)
        profiles.append(pd.Series(profile, index=delphi_buckets, name=f"delphi_top20_{phase}p"))
        frontier_rows.append(top.reset_index().assign(phase=phase))
    profiles = pd.concat(profiles, axis=1)
    branch_root = frontier.parent / "branch_inputs"
    branch = pd.read_csv(branch_root / "rows.csv")
    with np.load(branch_root / "arrays.npz") as archive:
        branch_arrays = {k: archive[k] for k in archive.files}
    branch_inventory = (
        pd.read_csv(frontier / "buckets.csv")
        .set_index("bucket")
        .loc[branch_arrays["bucket_names"], "epochs_per_unit_weight"]
        .to_numpy()
    )
    branch_aggregate = (branch_arrays["phase0_epochs"] + branch_arrays["phase1_epochs"]) / branch_inventory
    assert np.max(abs(branch_aggregate.sum(axis=1) - 1)) < 1e-6
    best_branch = (
        branch[~branch.is_tied_control]
        .groupby("coordinate_hash")
        .agg(measured=("target", "mean"), repeats=("target", "count"))
        .nsmallest(20, "measured")
    )
    branch_index = (
        branch.reset_index()
        .drop_duplicates("coordinate_hash")
        .set_index("coordinate_hash")
        .loc[best_branch.index, "index"]
        .to_numpy()
    )
    profiles["delphi_top20_fixed_prefix_branches"] = pd.Series(
        branch_aggregate[branch_index].mean(axis=0), index=branch_arrays["bucket_names"]
    )
    best_branch.to_csv(OUTPUT / "delphi_frontier_branch_policies.csv")
    pd.concat(frontier_rows, ignore_index=True).to_csv(OUTPUT / "delphi_frontier_policies.csv", index=False)
    profiles.to_csv(OUTPUT / "delphi_frontier_bucket_profiles.csv", index_label="bucket")
    tables = []
    for scale in preparation.SCALES:
        _, swarm, fit = load_single(scale)
        natural = 1 / swarm.inventory
        natural /= natural.sum()
        table = pd.DataFrame({"bucket": swarm.buckets, "proportional": natural})
        for phase in (1, 2):
            name = "one_phase" if phase == 1 else "two_phase"
            proposal = json.loads((OUTPUT / "proposals" / scale / f"{name}.json").read_text())
            w = np.asarray(proposal["weights"])
            a = w if phase == 1 else 0.8 * w[0] + 0.2 * w[1]
            table[f"{phase}p_aggregate"] = a
            table[f"{phase}p_epochs"] = a * swarm.inventory
            if phase == 2:
                table["2p_early"], table["2p_late"] = w
                table["2p_late_minus_early"] = w[1] - w[0]
            summaries.append(
                {
                    "scale": scale,
                    "phase": phase,
                    "predicted_bpb": proposal["runtime_prediction"],
                    "max_epochs": proposal["max_materialized_epoch"],
                    "active_buckets": int((a > 0).sum()),
                    "component_floor": float(fit.weights @ np.asarray([t.head.floor for t in fit.tasks])),
                    "aggregate_bpb": proposal.get("aggregate_bpb", proposal["runtime_prediction"]),
                    "phase_correction": proposal.get("hpr_contrast", 0),
                    "phase_tv": proposal.get("phase_tv", 0),
                }
            )
        table = table.join(profiles, on="bucket")
        assert table.notna().all().all()
        table["scale"] = scale
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(OUTPUT / "bucket_comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "proposal_summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


def hpr_terms(response: Any, policy: np.ndarray) -> dict[str, np.ndarray]:
    state = (
        np.exp(-response.forgetting * (1 - policy[1])) * response.c0 * policy[0]
        + response.late_multiplier * response.c1 * policy[1]
    )
    family = np.bincount(response.family_index, weights=state, minlength=len(response.family_count))
    return {
        "bucket_benefit": -response.bucket_benefit * response.power(state)[0],
        "family_benefit": -response.family_benefit * response.power(family)[0],
        "family_harm": response.family_harm * response.harm(family)[0],
        "member_harm": (
            response.member_harm[response.family_index]
            / response.family_count[response.family_index]
            * response.harm(state)[0]
        ),
        "phase_tv": np.asarray([response.tv * np.abs(policy[0] - policy[1]).sum() / 2]),
    }


def diagnostics() -> None:
    records = []
    checks = {}
    for scale in preparation.SCALES:
        model, hpr, fit = surface(scale)
        proposal = json.loads((OUTPUT / "proposals" / scale / "two_phase.json").read_text())
        policy = np.asarray(proposal["weights"])
        aggregate = 0.8 * policy[0] + 0.2 * policy[1]
        actual = hpr_terms(model.hpr, policy)
        tied = hpr_terms(model.hpr, np.stack([aggregate, aggregate]))
        for term in actual:
            if term in ("bucket_benefit", "member_harm"):
                names = fit.buckets
            elif term.startswith("family"):
                names = hpr.dataset.family_names
            else:
                names = ("all",)
            for name, value in zip(names, actual[term] - tied[term], strict=True):
                records.append({"scale": scale, "term": term, "bucket_or_family": name, "phase_correction_bpb": value})
        summed = sum(float((actual[k] - tied[k]).sum()) for k in actual)
        assert abs(summed - proposal["hpr_contrast"]) < 1e-9
        checks[scale] = {
            "phase_decomposition_error": abs(summed - proposal["hpr_contrast"]),
            "hpr_config": asdict(hpr.config),
            **proposal["checks"],
        }
    pd.DataFrame(records).to_csv(OUTPUT / "phase_correction_terms.csv", index=False)
    write_json_atomic(OUTPUT / "CHECKS.json", checks)
    table = pd.read_csv(OUTPUT / "bucket_comparison.csv")
    scores = table.groupby("bucket")[["1p_aggregate", "2p_early", "2p_late"]].max().max(axis=1)
    selected = scores.nlargest(17).index.tolist()
    labels = [
        b.replace("dolma3_cc/", "CC: ")
        .replace("science_math_and_technology", "science/math/tech")
        .replace("dolma3_", "")
        .replace("dolmino_", "Dolmino: ")
        .replace("_", " ")
        .strip()
        for b in selected
    ]
    labels.append("All remaining buckets")
    mpl.use("Agg")
    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(17, 11), sharey=True, layout="constrained")
    colors = ("#0072B2", "#E69F00", "#009E73")
    for ax, scale, title in zip(axes, preparation.SCALES, ("Llama 160M / 1.2B", "Llama 200M / 6B"), strict=True):
        frame = table[table.scale.eq(scale)].set_index("bucket")
        for index, (column, label, color) in enumerate(
            zip(
                ("1p_aggregate", "2p_early", "2p_late"),
                ("MARINER 1p", "HPR + MARINER 2p: early", "HPR + MARINER 2p: late"),
                colors,
                strict=True,
            )
        ):
            values = (
                np.r_[frame.loc[selected, column].to_numpy(), frame.loc[~frame.index.isin(selected), column].sum()] * 100
            )
            ax.barh(np.arange(len(labels)) + (index - 1) * 0.24, values, height=0.22, color=color, label=label)
        benchmark = (
            np.r_[
                frame.loc[selected, "delphi_top20_1p"].to_numpy(),
                frame.loc[~frame.index.isin(selected), "delphi_top20_1p"].sum(),
            ]
            * 100
        )
        ax.plot(
            benchmark,
            np.arange(len(labels)),
            "o",
            color="#444444",
            markersize=4,
            label="Delphi: mean of 20 best measured 1p policies",
        )
        ax.set_title(title, fontsize=15, pad=14)
        ax.set_xlim(0, 85)
        ax.set_xlabel("Mixture weight (%)")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Uncapped Uncheatable proposals: inspect composition before training", fontsize=17)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"proposal_composition.{suffix}", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(records).groupby(["scale", "term"]).phase_correction_bpb.sum().to_string())


def audit() -> None:
    validation = []
    for scale in preparation.SCALES:
        module, swarm, fit = load_single(scale)
        model, hpr, _ = surface(scale)
        for phase, name in ((1, "one_phase"), (2, "two_phase")):
            path = OUTPUT / "proposals" / scale / f"{name}.json"
            proposal = json.loads(path.read_text())
            table = pd.read_csv(path.with_suffix(".csv"))
            counts = np.asarray(proposal["counts"])
            weights = np.asarray(proposal["weights"])
            assert tuple(table.bucket) == swarm.buckets
            assert np.all(counts.sum(axis=-1) == module.MIXTURE_BLOCK_SIZE)
            assert counts.min() >= 0 and np.array_equal(counts / module.MIXTURE_BLOCK_SIZE, weights)
            if phase == 1:
                observed = table.weight.to_numpy()
                prediction = float(fit.predict(weights)[0])
            else:
                observed = table[["phase0_weight", "phase1_weight"]].to_numpy().T
                prediction = float(batch_predict(model, hpr, fit, weights[None, :, :])[0])
                assert proposal["exchange_moves"][-1]["moves"] == proposal["exchange_moves"][-2]["moves"] == 0
            assert np.max(abs(observed - weights)) < 1e-14
            assert abs(prediction - proposal["runtime_prediction"]) < 1e-10
            validation.append(
                {
                    "scale": scale,
                    "phase": phase,
                    "buckets": len(swarm.buckets),
                    "blocks_per_phase": module.MIXTURE_BLOCK_SIZE,
                    "reconstructed_prediction": prediction,
                    "prediction_error": abs(prediction - proposal["runtime_prediction"]),
                    "weights_and_counts_valid": True,
                }
            )
    # Save the exact 160M HPR inner table; the 200M fit retains its original frozen splits.
    small = hpr_model("60m").dataset
    groups = small.frame.phase_correspondence_key.to_numpy()
    aggregate = 0.8 * small.weights[:, 0] + 0.2 * small.weights[:, 1]
    folds = preparation.preparation.neighborhood_splits(
        groups, aggregate, groups == "baseline_proportional", np.arange(len(groups)), preparation.SEED
    )
    labels = np.full(len(groups), -1, dtype=int)
    for index, (_, test) in enumerate(folds):
        labels[test] = index
    small.frame.assign(inner_fold=labels).to_csv(OUTPUT / "fits/60m/hpr/folds.csv", index=False)
    old_splits = REFERENCE / "two_phase_link_transfer_20260907/inputs/splits.npz"
    shutil.copyfile(old_splits, OUTPUT / "fits/300m/hpr/original_splits.npz")
    write_json_atomic(OUTPUT / "VALIDATION.json", {"proposals": validation, "no_training_submitted": True})
    sources = {Path(__file__), Path(preparation.__file__)}
    for imported in tuple(sys.modules.values()):
        filename = getattr(imported, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(preparation.controls.REPO_ROOT):
                sources.add(path)
    manifest_path = OUTPUT / "MANIFEST.json"
    write_json_atomic(
        manifest_path,
        {
            "source_sha256": {str(p): preparation.sha(p) for p in sorted(sources)},
            "artifact_sha256": {
                str(p.relative_to(OUTPUT)): preparation.sha(p)
                for p in sorted(OUTPUT.rglob("*"))
                if p.is_file() and p != manifest_path and "__pycache__" not in p.parts
            },
            "delphi_comparison_source_sha256": {
                str(p): preparation.sha(p)
                for p in (
                    REFERENCE / "delphi_phase_frontier_calibration_20260910/inputs/observations.csv",
                    REFERENCE / "delphi_phase_frontier_calibration_20260910/inputs/coordinates.npz",
                    REFERENCE / "delphi_phase_frontier_calibration_20260910/branch_inputs/rows.csv",
                    REFERENCE / "delphi_phase_frontier_calibration_20260910/branch_inputs/arrays.npz",
                )
            },
        },
    )
    print("Four proposal reconstructions and block-count checks passed; saved exact folds and hashes.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("one", "two", "summary", "diagnostics", "audit"))
    parser.add_argument("--scale", choices=preparation.SCALES)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "summary":
            summarize()
            return
        if args.stage == "diagnostics":
            diagnostics()
            return
        if args.stage == "audit":
            audit()
            return
        assert args.scale is not None, "choose a scale"
        if args.stage == "one":
            one_phase(args.scale)
        else:
            two_phase(args.scale)


if __name__ == "__main__":
    main()
