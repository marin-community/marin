# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "fsspec", "gcsfs", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Materialize a symmetric separate-heads comparison and the geometry frontier.

The separate-heads arm fits two policy classes independently:

* one phase: one exposure-response bowl over total per-bucket exposure;
* two phases: independent exposure-response bowls for phase 0 and phase 1.

Each policy class uses the same 289-row panel: 279 policy-matched swarm and
deletion rows plus 10 shared proportional repeats. Proportional observations are
grouped in
cross-validation, and an independently selected ridge penalty. The deployment
KL grid is identical for the two policy classes.

The geometry arm revisits the existing effective-exposure DSP plus nonnegative
phase-TV and aggregate-HHI terms. Unlike the earlier support-gated KL=5/10
panel, this panel sweeps the lower-KL frontier and includes the free two-phase
proposal, its exact aggregate-matched tied control, and an independently
optimized one-phase proposal under the fitted joint model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_best_phase_model_validation_panel_300m as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "symmetric_sepheads_geometry_frontier_panel_20260711"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_symmetric_sepheads_geometry_frontier_mixtures_20260711/mixtures"
)
SEPARATE_KL_VALUES = (0.05, 0.1, 0.2)
GEOMETRY_KL_VALUES = {
    "uncheatable": (0.2, 0.3, 0.5),
    "table9": (0.15, 0.2, 0.3),
}
L2_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0)
CV_SEEDS = (0, 1, 2)
N_SPLITS = 5
GEOMETRY_MAXITER = 16
GEOMETRY_COARSE_TOP_K = 2
TARGET_ABBR = {"uncheatable": "unch", "table9": "t9"}


@dataclass(frozen=True)
class SeparateModel:
    policy: str
    l2: float
    mus: tuple[np.ndarray, ...]
    intercept: float
    coefs: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class OptimizerResult:
    weights: np.ndarray
    regularized_objective: float
    successful_starts: int


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def kl_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def selected_mu(exposure: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    median = pooled.base_mu(exposure)
    best_rmse = np.inf
    best_mu = median
    for shift in pooled.MU_SHIFTS:
        mu = np.clip(median + shift, -2.0, 8.0)
        design = pooled.bowl_design(exposure, mu)
        intercept, coef = pooled.fit_raw_nnls(design, target, l2=l2)
        prediction = intercept + design @ coef
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_mu = mu
    return best_mu


def policy_exposures(dataset: pooled.Dataset, indices: np.ndarray, policy: str) -> tuple[np.ndarray, ...]:
    weights = dataset.weights[indices]
    if policy == "one_phase":
        if not np.allclose(weights[:, 0, :], weights[:, 1, :], atol=1e-10):
            raise ValueError("One-phase fit panel contains phase-varying weights")
        return (weights[:, 0, :] * (dataset.c0 + dataset.c1)[None, :],)
    if policy == "two_phase":
        return (
            weights[:, 0, :] * dataset.c0[None, :],
            weights[:, 1, :] * dataset.c1[None, :],
        )
    raise ValueError(f"Unknown policy {policy!r}")


def fit_separate_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    *,
    policy: str,
    l2: float,
) -> SeparateModel:
    target = dataset.y[indices]
    exposures = policy_exposures(dataset, indices, policy)
    mus = tuple(selected_mu(exposure, target, l2) for exposure in exposures)
    designs = [pooled.bowl_design(exposure, mu) for exposure, mu in zip(exposures, mus, strict=True)]
    design = np.hstack(designs)
    intercept, coef = pooled.fit_raw_nnls(design, target, l2=l2)
    width = designs[0].shape[1]
    coefs = tuple(np.asarray(coef[offset : offset + width], dtype=float) for offset in range(0, len(coef), width))
    return SeparateModel(policy=policy, l2=l2, mus=mus, intercept=intercept, coefs=coefs)


def predict_separate_model(
    model: SeparateModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
) -> np.ndarray:
    exposures = policy_exposures(dataset, indices, model.policy)
    prediction = np.full(len(indices), model.intercept, dtype=float)
    for exposure, mu, coef in zip(exposures, model.mus, model.coefs, strict=True):
        prediction += pooled.bowl_design(exposure, mu) @ coef
    return prediction


def select_l2(dataset: pooled.Dataset, policy: str, l2_values: tuple[float, ...]) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for l2 in l2_values:
        for seed in CV_SEEDS:
            folds = joint.grouped_folds(dataset.frame, seed, N_SPLITS)
            prediction = np.zeros(dataset.n, dtype=float)
            for train_indices, test_indices in folds:
                model = fit_separate_model(dataset, train_indices, policy=policy, l2=l2)
                prediction[test_indices] = predict_separate_model(model, dataset, test_indices)
            metric = pooled.metrics(dataset, f"separate_heads_{policy}_l2_{l2:g}", seed, prediction, folds)
            rows.append({"policy": policy, "l2": l2, **asdict(metric)})
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["policy", "l2"], as_index=False)
        .agg(
            oof_rmse_mean=("oof_rmse", "mean"),
            oof_spearman_mean=("oof_spearman", "mean"),
            fold_mean_regret_at_1_mean=("fold_mean_regret_at_1", "mean"),
            lower_tail_optimism_mean=("lower_tail_optimism", "mean"),
            low_tail_rmse_mean=("low_tail_rmse", "mean"),
        )
        .sort_values(["oof_rmse_mean", "fold_mean_regret_at_1_mean", "l2"])
        .reset_index(drop=True)
    )
    return float(summary.iloc[0]["l2"]), frame


def separate_predictor(model: SeparateModel, dataset: pooled.Dataset) -> Callable[[np.ndarray], float]:
    def predict(weights: np.ndarray) -> float:
        candidate = pooled.Dataset(
            name=dataset.name,
            frame=pd.DataFrame({"phase_correspondence_key": ["candidate"]}),
            y=np.zeros(1, dtype=float),
            weights=np.asarray(weights, dtype=float)[None, :, :],
            c0=dataset.c0,
            c1=dataset.c1,
            domain_names=dataset.domain_names,
        )
        return float(predict_separate_model(model, candidate, np.asarray([0], dtype=int))[0])

    return predict


def weighted_kl(weights: np.ndarray, natural: np.ndarray, alpha0: float, alpha1: float) -> float:
    return float(olmix.weighted_multiclass_kl(weights, natural, (alpha0, alpha1)))


def optimize_predictor(
    predictor: Callable[[np.ndarray], float],
    dataset: pooled.Dataset,
    natural: np.ndarray,
    kl_reg: float,
    *,
    one_phase: bool,
) -> OptimizerResult:
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    m = dataset.m

    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        if one_phase:
            values = softmax(logits)
            return np.stack([values, values])
        return np.stack([softmax(logits[:m]), softmax(logits[m:])])

    def logits_for_weights(weights: np.ndarray) -> np.ndarray:
        if one_phase:
            aggregate = alpha0 * weights[0] + alpha1 * weights[1]
            return np.log(np.clip(aggregate, 1e-12, 1.0))
        return np.log(np.clip(weights, 1e-12, 1.0)).reshape(-1)

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits)
        return predictor(weights) + kl_reg * weighted_kl(weights, natural, alpha0, alpha1)

    starts = [logits_for_weights(np.stack([natural, natural]))]
    starts.extend(logits_for_weights(dataset.weights[index]) for index in np.argsort(dataset.y)[:8])
    best_value = np.inf
    best_weights = None
    successful_starts = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-10, "maxls": 30},
        )
        if result.success:
            successful_starts += 1
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_weights = weights_from_logits(np.asarray(result.x, dtype=float))
    if best_weights is None:
        raise RuntimeError(f"No finite optimizer result for one_phase={one_phase}, KL={kl_reg}")
    return OptimizerResult(best_weights, best_value, successful_starts)


def one_phase_panel(frame: pd.DataFrame, objective: str) -> pooled.Dataset:
    target = joint.TARGET_COLUMNS[objective]
    single = frame.loc[frame["policy_family"].eq("single_phase")].copy()
    shared_repeats = frame.loc[
        frame["split"].eq("train") & frame["phase_pair_status"].eq("repeat_reference_to_baseline")
    ].copy()
    for domain in pooled.load_300m_dataset(objective).domain_names:
        shared_repeats[f"phase_0_{domain}"] = shared_repeats[f"phase_1_{domain}"] = float(
            shared_repeats.iloc[0][f"phase_0_{domain}"]
        )
    panel = pd.concat([single, shared_repeats], ignore_index=True)
    if len(panel) != 289:
        raise ValueError(f"Expected 289 one-phase rows, found {len(panel)}")
    return joint.dataset_from_frame(objective, panel, target)


def two_phase_panel(frame: pd.DataFrame, objective: str) -> pooled.Dataset:
    panel = frame.loc[frame["split"].eq("train") & ~frame["phase_pair_status"].eq("no_single_phase_counterpart")].copy()
    if len(panel) != 289:
        raise ValueError(f"Expected 289 two-phase rows, found {len(panel)}")
    return joint.dataset_from_frame(objective, panel, joint.TARGET_COLUMNS[objective])


def candidate_diagnostics(
    *,
    candidate: str,
    family: str,
    objective: str,
    policy: str,
    kl_reg: float,
    prediction: float,
    weights: np.ndarray,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: float,
    alpha0: float,
    alpha1: float,
    selected_l2: float | None,
    regularized_objective: float,
    optimizer_successful_starts: int | None,
) -> dict[str, object]:
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    epochs = olmix.simulated_epochs(weights, token_counts, target_budget=target_budget)
    return {
        "candidate": candidate,
        "family": family,
        "objective": objective,
        "policy": policy,
        "kl_reg": kl_reg,
        "selected_l2": selected_l2,
        "predicted_bpb_300m": prediction,
        "regularized_objective": regularized_objective,
        "weighted_kl_to_proportional": weighted_kl(weights, natural, alpha0, alpha1),
        "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - natural).sum()),
        "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "max_weight": float(weights.max()),
        "max_simulated_epoch": float(epochs.max()),
        "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
        "optimizer_successful_starts": optimizer_successful_starts,
    }


def write_candidate(
    output_dir: Path,
    gcs_output_dir: str,
    candidate: str,
    frame: pd.DataFrame,
    *,
    upload: bool,
) -> None:
    path = output_dir / f"{candidate}.csv"
    frame.to_csv(path, index=False)
    if upload:
        with fsspec.open(f"{gcs_output_dir.rstrip('/')}/{candidate}.csv", "wt") as handle:
            frame.to_csv(handle, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--l2-values", default=",".join(str(value) for value in L2_VALUES))
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = pooled.load_300m_dataset("table9")
    frame = joint.attach_single_phase_weights(pd.read_csv(args.packet), args.one_phase_source, reference.domain_names)
    l2_values = parse_float_list(args.l2_values)
    rows: list[dict[str, object]] = []
    cv_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []

    for objective in ("uncheatable", "table9"):
        print(f"=== {objective}: symmetric separate-heads ===", flush=True)
        one_dataset = one_phase_panel(frame, objective)
        two_dataset = two_phase_panel(frame, objective)
        _packet, domains, natural, token_counts, target_budget, _folds = bowl.load_objective(objective)
        for policy, dataset in (("1p", one_dataset), ("2p", two_dataset)):
            model_policy = "one_phase" if policy == "1p" else "two_phase"
            selected_l2, cv_frame = select_l2(dataset, model_policy, l2_values)
            cv_frame["objective"] = objective
            cv_frame["candidate_policy"] = policy
            cv_frames.append(cv_frame)
            model = fit_separate_model(
                dataset,
                np.arange(dataset.n),
                policy=model_policy,
                l2=selected_l2,
            )
            predictor = separate_predictor(model, dataset)
            fit_rows.append(
                {
                    "family": "separate_heads",
                    "objective": objective,
                    "policy": policy,
                    "fit_rows": dataset.n,
                    "selected_l2": selected_l2,
                    "nominal_parameter_count": (2 if policy == "1p" else 4) * dataset.m + (2 if policy == "1p" else 3),
                }
            )
            alpha0, alpha1 = coverage.phase_fractions(dataset)
            for kl_reg in SEPARATE_KL_VALUES:
                result = optimize_predictor(
                    predictor,
                    dataset,
                    natural,
                    kl_reg,
                    one_phase=policy == "1p",
                )
                candidate = f"symsep_{TARGET_ABBR[objective]}_{policy}_kl{kl_tag(kl_reg)}"
                mixture = per_component.mixture_frame(
                    domains=domains,
                    natural=natural,
                    weights=result.weights,
                    token_counts=token_counts,
                    target_budget=target_budget,
                )
                write_candidate(
                    args.output_dir,
                    args.gcs_output_dir,
                    candidate,
                    mixture,
                    upload=not args.skip_upload,
                )
                rows.append(
                    candidate_diagnostics(
                        candidate=candidate,
                        family="separate_heads",
                        objective=objective,
                        policy=policy,
                        kl_reg=kl_reg,
                        prediction=predictor(result.weights),
                        weights=result.weights,
                        natural=natural,
                        token_counts=token_counts,
                        target_budget=target_budget,
                        alpha0=alpha0,
                        alpha1=alpha1,
                        selected_l2=selected_l2,
                        regularized_objective=result.regularized_objective,
                        optimizer_successful_starts=result.successful_starts,
                    )
                )

        print(f"=== {objective}: effective-exposure + geometry frontier ===", flush=True)
        joint_dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            joint.TARGET_COLUMNS[objective],
        )
        alpha0, alpha1 = coverage.phase_fractions(joint_dataset)
        model = coverage.fit_model(
            joint_dataset,
            np.arange(joint_dataset.n),
            geometry.MODEL_CONFIG,
            linear_reg=coverage.dataset_linear_reg(joint_dataset),
            maxiter=GEOMETRY_MAXITER,
            coarse_top_k=GEOMETRY_COARSE_TOP_K,
        )
        for kl_reg in GEOMETRY_KL_VALUES[objective]:
            two_phase = geometry.optimize_candidate(
                joint_dataset,
                model,
                natural,
                kl_reg,
                one_phase=False,
                alpha0=alpha0,
                alpha1=alpha1,
            )
            aggregate = alpha0 * two_phase.weights[0] + alpha1 * two_phase.weights[1]
            candidates: tuple[tuple[str, np.ndarray, geometry.OptimizerResult | None], ...] = (
                ("2p", two_phase.weights, two_phase),
                ("tied", np.stack([aggregate, aggregate]), None),
                (
                    "1p",
                    geometry.optimize_candidate(
                        joint_dataset,
                        model,
                        natural,
                        kl_reg,
                        one_phase=True,
                        alpha0=alpha0,
                        alpha1=alpha1,
                    ).weights,
                    None,
                ),
            )
            for policy, weights, optimizer in candidates:
                candidate = f"geomfront_{TARGET_ABBR[objective]}_{policy}_kl{kl_tag(kl_reg)}"
                prediction = float(coverage.predict(model, weights[None, :, :], alpha0, alpha1)[0])
                mixture = per_component.mixture_frame(
                    domains=domains,
                    natural=natural,
                    weights=weights,
                    token_counts=token_counts,
                    target_budget=target_budget,
                )
                write_candidate(
                    args.output_dir,
                    args.gcs_output_dir,
                    candidate,
                    mixture,
                    upload=not args.skip_upload,
                )
                rows.append(
                    candidate_diagnostics(
                        candidate=candidate,
                        family="effective_exposure_geometry",
                        objective=objective,
                        policy=policy,
                        kl_reg=kl_reg,
                        prediction=prediction,
                        weights=weights,
                        natural=natural,
                        token_counts=token_counts,
                        target_budget=target_budget,
                        alpha0=alpha0,
                        alpha1=alpha1,
                        selected_l2=coverage.dataset_linear_reg(joint_dataset),
                        regularized_objective=(
                            optimizer.objective
                            if optimizer is not None
                            else prediction + kl_reg * weighted_kl(weights, natural, alpha0, alpha1)
                        ),
                        optimizer_successful_starts=(optimizer.successful_starts if optimizer is not None else None),
                    )
                )

    manifest = pd.DataFrame(rows).sort_values(["family", "objective", "kl_reg", "policy"])
    if len(manifest) != 30 or manifest["candidate"].duplicated().any():
        raise AssertionError("Expected exactly 30 unique candidates")
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    pd.concat(cv_frames, ignore_index=True).to_csv(args.output_dir / "separate_heads_cv_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.output_dir / "separate_heads_selected_models.csv", index=False)
    (args.output_dir / "panel_config.json").write_text(
        json.dumps(
            {
                "separate_kl_values": SEPARATE_KL_VALUES,
                "geometry_kl_values": GEOMETRY_KL_VALUES,
                "l2_values": l2_values,
                "cv_seeds": CV_SEEDS,
                "n_splits": N_SPLITS,
                "gcs_output_dir": args.gcs_output_dir,
                "candidate_count": len(manifest),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(manifest.to_string(index=False))
    print(f"Wrote {len(manifest)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
