# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered synthetic recovery of bounded and tangent temporal benefit.

Preparation uses frozen policy covariates and a previously saved synthetic head.
Actual TPP40 outcomes are never indexed. Every fitted cell and strength profile
is resumable under an explicit source, registration, and data identity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as joint
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mechanism_models_20261014 as mechanisms
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_models_20261015 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_optimization_20261015 as optimization
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_tangent_round5_20261015"
DATA = ROOT / "reference_outputs/tpp40_temporal_round_20261014/data"
ROUND3 = ROOT / "reference_outputs/tpp40_joint_estimation_round3_20261014"
TRUTH_SOURCE = ROUND3 / "contexts/aggregate_null_noise0_seed20261015/truth.json"
FAMILIES = ("aggregate_mariner", "bounded_benefit_odds", "tangent_benefit_odds")


@dataclass(frozen=True)
class Geometry:
    early: np.ndarray
    late: np.ndarray
    alpha: float
    tied: np.ndarray
    pool_tokens: np.ndarray
    phase0: np.ndarray
    phase1: np.ndarray
    scales: np.ndarray

    def policy_geometry(self) -> optimization.PolicyGeometry:
        return optimization.PolicyGeometry(self.phase0, self.phase1, self.scales, self.pool_tokens, self.alpha)


@dataclass(frozen=True)
class Context:
    spec: dict[str, Any]
    geometry: Geometry
    train: np.ndarray
    asymmetric: np.ndarray
    tied: np.ndarray
    inner_labels: np.ndarray
    clean: np.ndarray
    observed: np.ndarray
    clean_tied: np.ndarray
    truth: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.spec["name"])


@dataclass(frozen=True)
class SyntheticTruth:
    specification: dict[str, Any]
    alpha: float

    def predict(self, early: np.ndarray, late: np.ndarray) -> np.ndarray:
        return synthetic_prediction(self.specification, early, late, self.alpha)


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    if tuple(value["families"]) != FAMILIES:
        raise ValueError("Registered families differ from the three implemented comparisons")
    if tuple(value["fit_strengths"]) != models.STRENGTHS:
        raise ValueError("Registered strength grid differs from the model implementation")
    if len({item["name"] for item in value["contexts"]}) != len(value["contexts"]):
        raise ValueError("Synthetic context names must be unique")
    for item in value["contexts"]:
        if item["truth_family"] not in FAMILIES or item["truth_mode"] not in (
            "null",
            "fixed_strength",
            "train_gain_rms",
        ):
            raise ValueError(f"Unsupported synthetic truth specification: {item}")
        if item["noise_sd"] < 0:
            raise ValueError("Noise standard deviation must be nonnegative")
    return value


def load_geometry() -> Geometry:
    """Read only named policy covariates from the frozen mixed-content archive."""
    with np.load(DATA / "arrays.npz") as arrays:
        return Geometry(
            arrays["phase0_epochs"],
            arrays["phase1_epochs"],
            float(arrays["alpha"]),
            arrays["tied"],
            arrays["pool_tokens"],
            arrays["phase0"],
            arrays["phase1"],
            arrays["c0"] + arrays["c1"],
        )


def source_identity() -> dict[str, str]:
    """Pin explicit executable dependencies and frozen synthetic inputs."""
    paths = {
        Path(__file__).resolve(),
        Path(benchmark.__file__).resolve(),
        Path(models.__file__).resolve(),
        Path(optimization.__file__).resolve(),
        Path(joint.__file__).resolve(),
        Path(mechanisms.__file__).resolve(),
        Path(temporal.__file__).resolve(),
        Path(temporal.mariner.__file__).resolve(),
        ROOT / "two_phase_benchmark_data_20260921.py",
        OUTPUT / "registration.json",
        DATA / "manifest.json",
        DATA / "arrays.npz",
        DATA / "folds.csv",
        DATA / "rows.csv",
        TRUTH_SOURCE,
        TRUTH_SOURCE.parent / "complete.json",
        ROUND3 / "result_manifest.json",
    }
    return {str(path): benchmark.sha256(path) for path in sorted(paths)}


def execution_identity() -> dict[str, str]:
    expected = json.loads((OUTPUT / "execution_identity.json").read_text())
    if source_identity() != expected:
        raise ValueError("Round-five sources or inputs changed; preserve results and register a new version")
    return expected


def require_execution_approval() -> None:
    """Require a review receipt that names the exact prepared execution identity."""
    value = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "synthetic_fit_and_optimization",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Execution approval does not match the prepared synthetic protocol")


def completed(folder: Path, context: dict[str, Any], frozen: dict[str, str]) -> bool:
    path = folder / "complete.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record["identity"] != frozen or record["context"] != context:
        raise ValueError(f"Stale or mismatched result: {folder}")
    benchmark.verify_hashes(record["outputs"], folder)
    return True


def finish(folder: Path, context: dict[str, Any], files: tuple[str, ...], frozen: dict[str, str], **extra: Any) -> None:
    if source_identity() != frozen:
        raise ValueError("Sources changed during execution")
    benchmark.write_json(
        folder / "complete.json",
        {
            "identity": frozen,
            "context": context,
            "outputs": {name: benchmark.sha256(folder / name) for name in files},
            **extra,
        },
    )


def synthetic_prediction(truth: dict[str, Any], early: np.ndarray, late: np.ndarray, alpha: float) -> np.ndarray:
    matrix, offset = models.design_and_offset(early, late, alpha, truth["shape"], truth["model"], truth["parameters"])
    head = truth["head"]
    deficit = np.exp(float(head["intercept"]) + matrix @ np.asarray(head["coefficients"]) + offset)
    predicted = float(head["floor"]) + deficit
    if not np.isfinite(predicted).all():
        raise FloatingPointError("Synthetic truth produced nonfinite outcomes")
    return predicted


def make_truth(spec: dict[str, Any], geometry: Geometry, train: np.ndarray) -> dict[str, Any]:
    """Set temporal strength using preregistered values or training covariates."""
    source = json.loads(TRUTH_SOURCE.read_text())
    truth = {
        "model": spec["truth_family"],
        "shape": source["shape"],
        "head": source["head"],
        "parameters": {},
        "source": str(TRUTH_SOURCE),
        "source_sha256": benchmark.sha256(TRUTH_SOURCE),
        "amplitudes_intercept_and_floor_changed": False,
    }
    mode = spec["truth_mode"]
    if mode == "null":
        if truth["model"] != "aggregate_mariner":
            raise ValueError("The null world must use aggregate MARINER")
        truth["calibration"] = {"mode": mode, "strength": 0.0}
        return truth
    if truth["model"] == "aggregate_mariner":
        raise ValueError("Positive-effect worlds require a temporal law")
    if mode == "fixed_strength":
        strength = float(spec["truth_strength"])
        calibration = {"mode": mode, "strength": strength}
    else:
        config = registration()["truth_calibration"]
        lower, upper = map(float, config["strength_bounds"])
        target = float(spec["target_train_gain_rms"])
        if target <= 0 or lower != 0 or upper <= lower:
            raise ValueError("Weak-effect calibration requires positive target and a fixed nonnegative bracket")
        early, late = geometry.early[train], geometry.late[train]
        total = early + late
        null_truth = {**truth, "model": "aggregate_mariner", "parameters": {}}
        diagonal = synthetic_prediction(null_truth, geometry.alpha * total, (1 - geometry.alpha) * total, geometry.alpha)
        probes = []

        def difference(strength: float) -> float:
            candidate = {**truth, "parameters": {"kappa": float(strength)}}
            gain = diagonal - synthetic_prediction(candidate, early, late, geometry.alpha)
            rms = float(np.sqrt(np.mean(gain**2)))
            probes.append({"strength": float(strength), "train_gain_rms": rms})
            return rms - target

        strength, result = brentq(
            difference,
            lower,
            upper,
            xtol=float(config["xtol"]),
            rtol=float(config["rtol"]),
            maxiter=int(config["maxiter"]),
            full_output=True,
        )
        if not result.converged:
            raise ValueError("Training-covariate weak-effect calibration failed to converge")
        calibration = {
            "mode": mode,
            "strength": float(strength),
            "target_train_gain_rms": target,
            "train_rows": train.tolist(),
            "probes": probes,
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "config": config,
        }
    if strength < 0:
        raise ValueError("Synthetic temporal strength must be nonnegative")
    truth["parameters"] = {"kappa": float(strength)}
    truth["calibration"] = calibration
    return truth


def prepare() -> None:
    """Freeze exact source identity and materialize synthetic labels without fits."""
    config = registration()
    data_manifest = json.loads((DATA / "manifest.json").read_text())
    benchmark.verify_hashes(data_manifest["files"], DATA)
    truth_receipt = json.loads((TRUTH_SOURCE.parent / "complete.json").read_text())
    benchmark.verify_hashes(truth_receipt["outputs"], TRUTH_SOURCE.parent)
    geometry, folds = load_geometry(), pd.read_csv(DATA / "folds.csv")
    frozen = source_identity()
    identity_path = OUTPUT / "execution_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != frozen:
        raise ValueError("Existing execution identity differs; no automatic overwrite")
    if not identity_path.exists():
        benchmark.write_json(identity_path, frozen)
    for spec in config["contexts"]:
        folder = OUTPUT / "contexts" / spec["name"]
        if completed(folder, spec, frozen):
            continue
        outer = folds[folds.outer_fold.eq(spec["outer_fold"])]
        fit_rows = outer[outer.role.eq("train")]
        train, labels = fit_rows.row_index.to_numpy(int), fit_rows.inner_fold.to_numpy(int)
        query = outer.loc[outer.role.ne("train"), "row_index"].to_numpy(int)
        asymmetric = query[~geometry.tied[query]]
        tied = np.flatnonzero(geometry.tied)
        expected = config["expected_counts"]
        if (len(train), len(asymmetric), len(tied)) != (expected["train"], expected["asymmetric"], expected["tied"]):
            raise ValueError("Frozen split differs from preregistered synthetic geometry")
        if geometry.tied[train].any() or np.intersect1d(train, query).size:
            raise ValueError("Synthetic training must exclude tied and held-out rows")
        temporal.inner_splits(labels)
        truth = make_truth(spec, geometry, train)
        clean = synthetic_prediction(truth, geometry.early, geometry.late, geometry.alpha)
        total = geometry.early + geometry.late
        clean_tied = synthetic_prediction(truth, geometry.alpha * total, (1 - geometry.alpha) * total, geometry.alpha)
        noise = np.random.default_rng(spec["noise_seed"]).normal(0, spec["noise_sd"], len(clean))
        observed = clean.copy()
        observed[train] += noise[train]
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / "labels.npz",
            clean=clean,
            clean_tied=clean_tied,
            observed=observed,
            noise_vector=noise,
            train=train,
            asymmetric=asymmetric,
            tied=tied,
            inner_labels=labels,
        )
        roles = np.full(len(clean), "unused", dtype=object)
        roles[train], roles[asymmetric], roles[tied] = "train", "test", "tied_diagnostic"
        inner = np.full(len(clean), -1)
        inner[train] = labels
        pd.DataFrame(
            {
                "row_index": np.arange(len(clean)),
                "role": roles,
                "inner_fold": inner,
                "clean": clean,
                "observed": observed,
                "clean_tied": clean_tied,
                "clean_gain": clean_tied - clean,
            }
        ).to_csv(folder / "labels.csv", index=False)
        benchmark.write_json(folder / "truth.json", truth)
        finish(folder, spec, ("labels.npz", "labels.csv", "truth.json"), frozen)
        print(
            f"Prepared {spec['name']}: {len(train)} train / {len(asymmetric)} asymmetric / {len(tied)} tied", flush=True
        )


def load_context(spec: dict[str, Any], frozen: dict[str, str]) -> Context:
    folder = OUTPUT / "contexts" / spec["name"]
    if not completed(folder, spec, frozen):
        raise ValueError(f"Run prepare before fitting: {folder}")
    with np.load(folder / "labels.npz") as labels:
        return Context(
            spec,
            load_geometry(),
            labels["train"],
            labels["asymmetric"],
            labels["tied"],
            labels["inner_labels"],
            labels["clean"],
            labels["observed"],
            labels["clean_tied"],
            json.loads((folder / "truth.json").read_text()),
        )


def query_coordinates(context: Context) -> tuple[np.ndarray, np.ndarray]:
    query = np.r_[context.asymmetric, context.tied]
    geometry = context.geometry
    total = geometry.early[context.asymmetric] + geometry.late[context.asymmetric]
    return (
        np.vstack((geometry.early[query], geometry.alpha * total)),
        np.vstack((geometry.late[query], (1 - geometry.alpha) * total)),
    )


def prediction_scores(context: Context, prediction: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score held-out policies and their clean synthetic tied counterparts."""
    a, t = len(context.asymmetric), len(context.tied)
    if prediction.shape != (2 * a + t,) or not np.isfinite(prediction).all():
        raise ValueError("Prediction vector does not match the registered finite menu")
    clean = np.r_[context.clean[context.asymmetric], context.clean[context.tied], context.clean_tied[context.asymmetric]]
    roles = ["asymmetric"] * a + ["actual_tied"] * t + ["matched_tied"] * a
    rows = np.r_[context.asymmetric, context.tied, context.asymmetric]
    gain = prediction[a + t :] - prediction[:a]
    true_gain = context.clean_tied[context.asymmetric] - context.clean[context.asymmetric]
    tolerance = float(registration()["metrics"]["gain_sign_zero_tolerance"])
    informative = abs(true_gain) > tolerance
    gain_rank = None
    if np.ptp(true_gain) > tolerance and np.ptp(gain) > tolerance:
        gain_rank = float(spearmanr(true_gain, gain).statistic)
    shortlist = np.lexsort((rows, np.asarray(roles), prediction))
    selected = int(shortlist[0])
    frontier = shortlist[: math.ceil(0.2 * len(prediction))]
    asym_selected = int(np.lexsort((rows[:a], prediction[:a]))[0])
    scores = {
        "asymmetric_test_n": a,
        "actual_tied_diagnostic_n": t,
        "matched_tied_n": a,
        "clean_asymmetric_rmse": float(np.sqrt(np.mean((prediction[:a] - clean[:a]) ** 2))),
        "clean_actual_tied_rmse": float(np.sqrt(np.mean((prediction[a : a + t] - clean[a : a + t]) ** 2))),
        "clean_matched_tied_rmse": float(np.sqrt(np.mean((prediction[a + t :] - clean[a + t :]) ** 2))),
        "clean_gain_rmse": float(np.sqrt(np.mean((gain - true_gain) ** 2))),
        "clean_gain_rms": float(np.sqrt(np.mean(true_gain**2))),
        "predicted_gain_rms": float(np.sqrt(np.mean(gain**2))),
        "gain_sign_accuracy": (
            float(np.mean(np.sign(gain[informative]) == np.sign(true_gain[informative]))) if informative.any() else None
        ),
        "gain_sign_n": int(informative.sum()),
        "gain_spearman": gain_rank,
        "max_abs_predicted_gain": float(np.max(abs(gain))),
        "max_positive_predicted_gain": float(np.max(gain)),
        "min_predicted_gain": float(np.min(gain)),
        "clean_menu_regret_at1": float(clean[selected] - clean.min()),
        "clean_menu_regret_at3": float(clean[shortlist[:3]].min() - clean.min()),
        "clean_menu_regret_at5": float(clean[shortlist[:5]].min() - clean.min()),
        "predicted_frontier_fraction": 0.2,
        "predicted_frontier_n": len(frontier),
        "predicted_frontier_rmse": float(np.sqrt(np.mean((prediction[frontier] - clean[frontier]) ** 2))),
        "predicted_frontier_mean_optimism": float(np.mean(clean[frontier] - prediction[frontier])),
        "predicted_frontier_max_optimism": float(np.max(clean[frontier] - prediction[frontier])),
        "clean_menu_selected_optimism": float(clean[selected] - prediction[selected]),
        "menu_selected_role": roles[selected],
        "menu_selected_row": int(rows[selected]),
        "menu_selected_clean_rank": int(1 + np.sum(clean < clean[selected])),
        "clean_asymmetric_regret_at1": float(clean[asym_selected] - clean[:a].min()),
        "clean_asymmetric_selected_optimism": float(clean[asym_selected] - prediction[asym_selected]),
        "selected_asymmetric_predicted_gain": float(gain[asym_selected]),
        "selected_asymmetric_true_gain": float(true_gain[asym_selected]),
    }
    frame = pd.DataFrame(
        {"menu_index": np.arange(len(clean)), "row_index": rows, "role": roles, "clean": clean, "predicted": prediction}
    )
    frame["predicted_frontier"] = np.isin(np.arange(len(clean)), frontier)
    frame["clean_phase_gain"] = np.r_[true_gain, np.full(t, np.nan), true_gain]
    frame["predicted_phase_gain"] = np.r_[gain, np.zeros(t), gain]
    return scores, frame


def cell_spec(context: Context, family: str) -> dict[str, str]:
    return {"context": context.name, "family": family}


def load_cell(context: Context, family: str, frozen: dict[str, str]) -> models.JointFit:
    folder = OUTPUT / "contexts" / context.name / "cells" / family
    if not completed(folder, cell_spec(context, family), frozen):
        raise ValueError(f"Required predecessor is incomplete: {folder}")
    return models.JointFit.from_json(json.loads((folder / "model.json").read_text()))


def run_context(spec: dict[str, Any]) -> dict[str, Any]:
    frozen = execution_identity()
    require_execution_approval()
    context = load_context(spec, frozen)
    train, geometry = context.train, context.geometry
    query_early, query_late = query_coordinates(context)
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            folder = OUTPUT / "contexts" / context.name / "cells" / family
            cell = cell_spec(context, family)
            if completed(folder, cell, frozen):
                print(f"Cached {context.name} {family}", flush=True)
                continue
            print(f"Starting {context.name} {family}", flush=True)
            started = time.monotonic()
            null = None if family == "aggregate_mariner" else load_cell(context, "aggregate_mariner", frozen)
            fit = models.fit_joint(
                geometry.early[train],
                geometry.late[train],
                context.observed[train],
                context.inner_labels,
                geometry.alpha,
                family,
                matched_null_fit=null,
            )
            predictions = fit.predict(query_early, query_late)
            score, frame = prediction_scores(context, predictions)
            profiles = models.strength_profiles(
                fit,
                geometry.early[train],
                geometry.late[train],
                context.observed[train],
                context.inner_labels,
                query_early,
                query_late,
            )
            strengths, strength_predictions, inner_predictions = [], [], []
            for profile in profiles:
                strength = float(profile["strength"])
                strength_score, strength_frame = prediction_scores(context, np.asarray(profile["query_predictions"]))
                strengths.append({"strength": strength, **strength_score, "inner_rmse": profile["inner_rmse"]})
                strength_predictions.append(strength_frame.assign(strength=strength))
                inner_predictions.append(
                    pd.DataFrame(
                        {
                            "strength": strength,
                            "row_index": train,
                            "inner_fold": context.inner_labels,
                            "clean": context.clean[train],
                            "observed": context.observed[train],
                            "predicted": profile["inner_predictions"],
                        }
                    )
                )
            folder.mkdir(parents=True, exist_ok=True)
            benchmark.write_json(folder / "model.json", fit.to_json())
            benchmark.write_json(
                folder / "scores.json",
                {
                    **spec,
                    **cell,
                    **score,
                    "selected_strength": float(fit.parameters.get("kappa", 0.0)),
                    "selected_shape": fit.shape,
                    "selected_ridge": fit.ridge,
                    "selected_gamma": fit.kappa,
                    "selected_floor": fit.head.floor,
                    "inner_rmse": fit.inner_rmse,
                },
            )
            benchmark.write_json(folder / "strength_profiles.json", profiles)
            frame.to_csv(folder / "predictions.csv", index=False)
            pd.DataFrame(strengths).to_csv(folder / "strength_scores.csv", index=False)
            pd.concat(strength_predictions, ignore_index=True).to_csv(folder / "strength_predictions.csv", index=False)
            pd.concat(inner_predictions, ignore_index=True).to_csv(
                folder / "strength_inner_predictions.csv", index=False
            )
            finish(
                folder,
                cell,
                (
                    "model.json",
                    "scores.json",
                    "strength_profiles.json",
                    "predictions.csv",
                    "strength_scores.csv",
                    "strength_predictions.csv",
                    "strength_inner_predictions.csv",
                ),
                frozen,
                elapsed_seconds=time.monotonic() - started,
                context_receipt_sha256=benchmark.sha256(OUTPUT / "contexts" / context.name / "complete.json"),
            )
            print(
                f"Finished {context.name} {family}: {time.monotonic() - started:.1f}s; "
                f"asym {score['clean_asymmetric_rmse']:.6f}, gain {score['clean_gain_rmse']:.6f}",
                flush=True,
            )
    return {"context": context.name, "cells": len(FAMILIES)}


def summarize() -> None:
    frozen = execution_identity()
    rows, strengths, receipts = [], [], {}
    for spec in registration()["contexts"]:
        context = load_context(spec, frozen)
        for family in FAMILIES:
            folder = OUTPUT / "contexts" / context.name / "cells" / family
            if not completed(folder, cell_spec(context, family), frozen):
                raise ValueError(f"Cannot summarize an incomplete cell: {folder}")
            rows.append(json.loads((folder / "scores.json").read_text()))
            strengths.append(pd.read_csv(folder / "strength_scores.csv").assign(context=context.name, family=family))
            receipts[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
    pd.DataFrame(rows).to_csv(OUTPUT / "scores.csv", index=False)
    pd.concat(strengths, ignore_index=True).to_csv(OUTPUT / "strength_scores.csv", index=False)
    benchmark.write_json(
        OUTPUT / "summary_manifest.json",
        {
            "identity": frozen,
            "cell_receipts": receipts,
            "outputs": {name: benchmark.sha256(OUTPUT / name) for name in ("scores.csv", "strength_scores.csv")},
            "scope": "Synthetic diagnostic; original qualification, real-data and prospective gates remain unchanged.",
            "claim": "Metrics retain every preregistered context; this summary does not promote any model.",
        },
    )
    print(f"Summarized {len(rows)} cells", flush=True)


def optimize_context(spec: dict[str, Any]) -> dict[str, Any]:
    """Audit raw predicted policies against known synthetic truth without refits."""
    frozen = execution_identity()
    require_execution_approval()
    context = load_context(spec, frozen)
    geometry = context.geometry.policy_geometry()
    seed = int(registration()["optimization"]["seed"])
    truth = SyntheticTruth(context.truth, geometry.alpha)
    truth_key = {
        "model": context.truth["model"],
        "shape": context.truth["shape"],
        "head": context.truth["head"],
        "parameters": context.truth["parameters"],
        "alpha": geometry.alpha,
        "seed": seed,
    }
    # A fixed registered name permits reuse across paired noise seeds.
    truth_name = str(spec["truth_group"])
    truth_folder = OUTPUT / "truth_optima" / truth_name
    if not completed(truth_folder, truth_key, frozen):
        result = optimization.optimize_fit(
            truth,
            geometry,
            seed=seed,
            phase_blind=not any(context.truth["parameters"].values()),
        )
        truth_folder.mkdir(parents=True, exist_ok=True)
        benchmark.write_json(truth_folder / "optimization.json", result)
        finish(truth_folder, truth_key, ("optimization.json",), frozen)
    truth_optimum = json.loads((truth_folder / "optimization.json").read_text())
    for family in FAMILIES:
        fit = load_cell(context, family, frozen)
        folder = OUTPUT / "contexts" / context.name / "optimization" / family
        cell = {**cell_spec(context, family), "stage": "raw_optimization"}
        if completed(folder, cell, frozen):
            continue
        print(f"Optimizing {context.name} {family}", flush=True)
        started = time.monotonic()
        result = optimization.optimize_fit(
            fit,
            geometry,
            seed=seed,
            phase_blind=not any(fit.parameters.values()),
        )
        truth_scores = optimization.evaluate_selected_truth(result, truth, geometry, truth_optimum)
        folder.mkdir(parents=True, exist_ok=True)
        benchmark.write_json(folder / "optimization.json", result)
        benchmark.write_json(folder / "truth_scores.json", {**spec, **cell, **truth_scores})
        finish(
            folder,
            cell,
            ("optimization.json", "truth_scores.json"),
            frozen,
            elapsed_seconds=time.monotonic() - started,
            truth_optimum_receipt_sha256=benchmark.sha256(truth_folder / "complete.json"),
            model_receipt_sha256=benchmark.sha256(
                OUTPUT / "contexts" / context.name / "cells" / family / "complete.json"
            ),
        )
    return {"context": context.name, "optimization_cells": len(FAMILIES)}


def summarize_optimization() -> None:
    frozen = execution_identity()
    rows, receipts = [], {}
    for spec in registration()["contexts"]:
        context = load_context(spec, frozen)
        for family in FAMILIES:
            folder = OUTPUT / "contexts" / context.name / "optimization" / family
            cell = {**cell_spec(context, family), "stage": "raw_optimization"}
            if not completed(folder, cell, frozen):
                raise ValueError(f"Cannot summarize an incomplete raw optimization: {folder}")
            result = json.loads((folder / "optimization.json").read_text())
            scores = json.loads((folder / "truth_scores.json").read_text())
            rows.append(
                {
                    **scores,
                    "predicted_gain_same_aggregate": result["predicted_gain_same_aggregate"],
                    "predicted_gain_vs_static": result["predicted_gain_vs_static"],
                    "phase_tv": result["phase_tv"],
                    "max_total_epochs": result["max_total_epochs"],
                    "nearest_policy_sqrt_distance": result["nearest_policy_sqrt_distance"],
                    "selected_solver_success": result["best"]["success"],
                    "static_solver_success": result["best_static"]["success"],
                }
            )
            receipts[str(folder / "complete.json")] = benchmark.sha256(folder / "complete.json")
    pd.DataFrame(rows).to_csv(OUTPUT / "optimization_scores.csv", index=False)
    benchmark.write_json(
        OUTPUT / "optimization_manifest.json",
        {
            "identity": frozen,
            "cell_receipts": receipts,
            "outputs": {"optimization_scores.csv": benchmark.sha256(OUTPUT / "optimization_scores.csv")},
            "claim": "Synthetic truth audit at numerical incumbents; truth optimization is not globally certified.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize", "optimize", "summarize-optimization"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--context")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    if args.command == "summarize":
        summarize()
        return
    if args.command == "summarize-optimization":
        summarize_optimization()
        return
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    execution_identity()
    contexts = registration()["contexts"]
    if args.context is not None:
        contexts = [item for item in contexts if item["name"] == args.context]
        if not contexts:
            raise ValueError(f"Unknown registered context: {args.context}")
    worker = optimize_context if args.command == "optimize" else run_context
    # Truth optima are shared across noise seeds; create them once sequentially.
    if args.command == "optimize" and args.workers != 1:
        raise ValueError("Raw optimization runs sequentially to preserve shared truth receipts")
    if args.workers == 1:
        for spec in contexts:
            print(json.dumps(worker(spec), sort_keys=True), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, spec) for spec in contexts]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
