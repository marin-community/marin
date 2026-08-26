# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Select runtime-exact KL paths of Delphi phase-0 prefixes.

An equal ensemble of shared-shape and bounded-shape DSPs is optimized under a
hard epoch cap and a forward-KL penalty away from the best observed
cap-admissible prefix. The cap and regularization ladder are explicit inputs.
The ensemble preserves measured model uncertainty:
bounded-shape is stronger on the exact Delphi boundary panel, while
shared-shape transfers better in several historical one-phase cells. Three
partition fits reduce dependence on one panel split. Runtime-materialized
incumbent and proportional controls are emitted beside the path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import benchmark_delphi_phase0_prefix_surrogates_20260824 as benchmark  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

DEFAULT_MODEL_PATH = benchmark.DEFAULT_OUTPUT_DIR / "full_models.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase0_prefix_candidates_20260824"
MIXTURE_BLOCK_SIZE = 2_048
DEFAULT_CAP_EPOCHS = benchmark.CAP_EPOCHS
LOCAL_EXCHANGE_TOLERANCE = 1e-12
DEFAULT_CANDIDATE_VARIANTS = ("shared_shape", "bounded_shape")
DEFAULT_KL_PENALTIES = (0.05, 0.2, 0.5)
ENSEMBLE_LABEL = "shared_bounded_ensemble"
FULL_FIT_SEEDS = (97, 98, 99)
MIN_RUNTIME_SUPPORT_FRACTION = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-variants", default=",".join(DEFAULT_CANDIDATE_VARIANTS))
    parser.add_argument("--kl-penalties", default=",".join(str(value) for value in DEFAULT_KL_PENALTIES))
    parser.add_argument("--cap-epochs", type=float, default=DEFAULT_CAP_EPOCHS)
    return parser.parse_args()


def epoch_scales(weights: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    scales = np.empty(exposure.shape[1], dtype=float)
    for column in range(exposure.shape[1]):
        nonzero = weights[:, column] > 1e-12
        ratios = exposure[nonzero, column] / weights[nonzero, column]
        if len(ratios) == 0 or not np.allclose(ratios, ratios[0], rtol=1e-9, atol=1e-9):
            raise ValueError(f"Could not recover a fixed epoch scale for bucket {column}")
        scales[column] = float(np.median(ratios))
    return scales


def load_models(path: Path) -> dict[str, benchmark.Fit]:
    payload = json.loads(path.read_text())
    expected = {variant.name for variant in benchmark.VARIANTS}
    if set(payload) != expected:
        raise ValueError(f"Model artifact variants changed: {sorted(payload)} != {sorted(expected)}")
    fits = {}
    for name, item in payload.items():
        variant = benchmark.Variant(**item["variant"])
        fits[name] = benchmark.Fit(
            variant=variant,
            shape=np.asarray(item["shape"], dtype=float),
            shrinkage=float(item["shrinkage"]),
            intercept=float(item["intercept"]),
            coefficients=np.asarray(item["coefficients"], dtype=float),
        )
    return fits


def runtime_counts(weights: np.ndarray) -> np.ndarray:
    target = np.asarray(weights, dtype=float) * MIXTURE_BLOCK_SIZE
    counts = np.floor(target).astype(np.int64)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    if remaining:
        order = np.argsort(-(target - counts), kind="stable")
        counts[order[:remaining]] += 1
    if int(counts.sum()) != MIXTURE_BLOCK_SIZE or int(counts.min()) < 0:
        raise ValueError("Invalid runtime mixture-block counts")
    return counts


def constrained_counts(weights: np.ndarray, maximum_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(weights, dtype=float) * MIXTURE_BLOCK_SIZE
    counts = np.minimum(np.floor(target).astype(np.int64), maximum_counts)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    while remaining:
        available = np.flatnonzero(counts < maximum_counts)
        if len(available) == 0:
            raise ValueError("The epoch cap cannot support a complete mixture block")
        deficit = target[available] - counts[available]
        chosen = int(available[int(np.argmax(deficit))])
        counts[chosen] += 1
        remaining -= 1
    return counts


def forward_kl(weights: np.ndarray, reference: np.ndarray) -> float:
    """Return KL(weights || reference) for a strictly positive reference."""
    if np.any(reference <= 0.0):
        raise ValueError("KL reference must be strictly positive")
    positive = weights > 0.0
    return float(np.sum(weights[positive] * np.log(weights[positive] / reference[positive])))


def hellinger_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.square(np.sqrt(first) - np.sqrt(second)).sum()))


def regularized_value(
    fits: tuple[benchmark.Fit, ...],
    scales: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray,
    kl_penalty: float,
) -> float:
    prediction = float(np.mean([benchmark.predict(fit, scales[None, :] * weights[None, :])[0] for fit in fits]))
    return prediction + kl_penalty * forward_kl(weights, reference)


def continuous_optimum(
    fits: tuple[benchmark.Fit, ...],
    scales: np.ndarray,
    starts: np.ndarray,
    reference: np.ndarray,
    kl_penalty: float,
    cap_epochs: float,
) -> tuple[np.ndarray, float]:
    upper = np.minimum(1.0, cap_epochs / scales)
    bounds = [(0.0, float(limit)) for limit in upper]
    constraint = {"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}
    best_weights = starts[0]
    best_value = regularized_value(fits, scales, best_weights, reference, kl_penalty)
    for start in starts:
        result = minimize(
            lambda value: regularized_value(fits, scales, value, reference, kl_penalty),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1e-12, "maxiter": 1_000},
        )
        if not result.success:
            continue
        value = float(result.fun)
        if value < best_value:
            best_weights = np.asarray(result.x, dtype=float)
            best_value = value
    if not np.isclose(best_weights.sum(), 1.0, atol=1e-9):
        raise ValueError("Continuous optimizer did not return a simplex point")
    return best_weights, best_value


def exchange_refine(
    fits: tuple[benchmark.Fit, ...],
    scales: np.ndarray,
    counts: np.ndarray,
    maximum_counts: np.ndarray,
    reference: np.ndarray,
    kl_penalty: float,
) -> np.ndarray:
    current = counts.copy()
    current_value = regularized_value(fits, scales, current / MIXTURE_BLOCK_SIZE, reference, kl_penalty)
    while True:
        donors = np.flatnonzero(current > 0)
        receivers = np.flatnonzero(current < maximum_counts)
        proposals = []
        moves = []
        for donor in donors:
            for receiver in receivers:
                if donor == receiver:
                    continue
                candidate = current.copy()
                candidate[donor] -= 1
                candidate[receiver] += 1
                proposals.append(candidate)
                moves.append((donor, receiver))
        if not proposals:
            break
        proposal_array = np.asarray(proposals, dtype=float) / MIXTURE_BLOCK_SIZE
        values = np.asarray(
            [regularized_value(fits, scales, proposal, reference, kl_penalty) for proposal in proposal_array]
        )
        chosen = int(np.argmin(values))
        if float(values[chosen]) >= current_value - LOCAL_EXCHANGE_TOLERANCE:
            break
        donor, receiver = moves[chosen]
        current[donor] -= 1
        current[receiver] += 1
        current_value = float(values[chosen])
    return current


def model_candidate(
    fits: tuple[benchmark.Fit, ...],
    scales: np.ndarray,
    starts: np.ndarray,
    maximum_counts: np.ndarray,
    reference: np.ndarray,
    kl_penalty: float,
    cap_epochs: float,
) -> tuple[np.ndarray, float, float]:
    continuous, continuous_value = continuous_optimum(fits, scales, starts, reference, kl_penalty, cap_epochs)
    counts = constrained_counts(continuous, maximum_counts)
    counts = exchange_refine(fits, scales, counts, maximum_counts, reference, kl_penalty)
    weights = counts / MIXTURE_BLOCK_SIZE
    if not np.array_equal(runtime_counts(weights), counts):
        raise ValueError("Candidate is not stable under runtime mixture realization")
    realized_value = regularized_value(fits, scales, weights, reference, kl_penalty)
    return weights, continuous_value, realized_value


def candidate_summary(
    candidate_id: str,
    source: str,
    weights: np.ndarray,
    scales: np.ndarray,
    fits: dict[str, benchmark.Fit],
    candidate_ensembles: dict[str, tuple[benchmark.Fit, ...]],
    frame: pd.DataFrame,
    incumbent: np.ndarray,
    proportional: np.ndarray,
    kl_penalty: float | None,
    model_variant: str | None,
) -> dict[str, float | int | str]:
    exposure = weights * scales
    distances = 0.5 * np.abs(frame.filter(like="phase_0_weight::").to_numpy(dtype=float) - weights).sum(axis=1)
    row: dict[str, float | int | str] = {
        "candidate_id": candidate_id,
        "source": source,
        "model_variant": model_variant or "control",
        "max_phase0_epoch": float(exposure.max()),
        "nearest_panel_run_order": int(frame.iloc[int(np.argmin(distances))]["run_order"]),
        "nearest_panel_tv": float(distances.min()),
        "kl_penalty": kl_penalty if kl_penalty is not None else np.nan,
        "kl_to_incumbent": forward_kl(weights, incumbent),
        "tv_to_incumbent": float(0.5 * np.abs(weights - incumbent).sum()),
        "hellinger_to_incumbent": hellinger_distance(weights, incumbent),
        "kl_to_proportional": forward_kl(weights, proportional),
        "tv_to_proportional": float(0.5 * np.abs(weights - proportional).sum()),
        "hellinger_to_proportional": hellinger_distance(weights, proportional),
    }
    for name, fit in fits.items():
        row[f"in_sample_predicted_uncheatable::{name}"] = float(benchmark.predict(fit, exposure[None, :])[0])
    for name, ensemble in candidate_ensembles.items():
        row[f"in_sample_predicted_uncheatable::{name}_partition_ensemble"] = float(
            np.mean([benchmark.predict(fit, exposure[None, :])[0] for fit in ensemble])
        )
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, buckets, weights, exposure = benchmark.load_panel()
    scales = epoch_scales(weights, exposure)
    if not np.isfinite(args.cap_epochs) or args.cap_epochs <= 0.0:
        raise ValueError("--cap-epochs must be finite and positive")
    maximum_counts = np.floor((args.cap_epochs / scales) * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
    if int(maximum_counts.sum()) < MIXTURE_BLOCK_SIZE:
        raise ValueError(f"The {args.cap_epochs:g}-epoch cap leaves no feasible runtime mixture")
    admissible = exposure.max(axis=1) <= args.cap_epochs + 1e-12
    if not np.any(admissible):
        raise ValueError(f"The panel has no rows admissible under the {args.cap_epochs:g}-epoch cap")
    panel_runtime_counts = np.vstack([runtime_counts(row) for row in weights])
    runtime_support_counts = (panel_runtime_counts > 0).sum(axis=0)
    minimum_support = int(np.ceil(MIN_RUNTIME_SUPPORT_FRACTION * len(frame)))
    if np.any(runtime_support_counts < minimum_support):
        raise ValueError(f"At least one bucket has runtime support below {minimum_support}/{len(frame)} panel rows")
    fits = load_models(args.model_path)
    candidate_variants = tuple(item.strip() for item in args.candidate_variants.split(",") if item.strip())
    if not candidate_variants or len(candidate_variants) != len(set(candidate_variants)):
        raise ValueError("Candidate variants must be a nonempty list without duplicates")
    unknown_variants = set(candidate_variants) - set(fits)
    if unknown_variants:
        raise ValueError(f"Unknown candidate variants: {sorted(unknown_variants)}")
    kl_penalties = tuple(float(item) for item in args.kl_penalties.split(",") if item.strip())
    if (
        not kl_penalties
        or len(kl_penalties) != len(set(kl_penalties))
        or any(not np.isfinite(value) or value < 0.0 for value in kl_penalties)
    ):
        raise ValueError("KL penalties must be a nonempty list of unique finite nonnegative values")

    primary_response = frame[benchmark.PRIMARY_TARGET].to_numpy(dtype=float)
    best_row = np.flatnonzero(admissible)[np.argmin(primary_response[admissible])]
    incumbent_counts = constrained_counts(weights[best_row], maximum_counts)
    incumbent = incumbent_counts / MIXTURE_BLOCK_SIZE
    proportional_continuous = (1.0 / scales) / np.sum(1.0 / scales)
    proportional_counts = constrained_counts(proportional_continuous, maximum_counts)
    proportional = proportional_counts / MIXTURE_BLOCK_SIZE
    starts = np.vstack([incumbent, proportional, weights[admissible]])

    candidates: dict[str, tuple[str, np.ndarray, float | None, str | None]] = {}
    optimization_rows = []
    primary_response = frame[benchmark.PRIMARY_TARGET].to_numpy(dtype=float)
    candidate_ensembles = {
        name: (
            fits[name],
            *(
                benchmark.fit_shape(
                    exposure,
                    primary_response,
                    fits[name].variant,
                    benchmark.mixture_blocks(weights, benchmark.OUTER_FOLDS, seed),
                    benchmark.quality_pairs(buckets),
                    fits[name].shrinkage,
                    seed,
                )
                for seed in FULL_FIT_SEEDS[1:]
            ),
        )
        for name in candidate_variants
    }
    deployment_ensemble = tuple(fit for name in candidate_variants for fit in candidate_ensembles[name])
    stability_rows = []
    for kl_penalty in kl_penalties:
        candidate, continuous_value, realized_value = model_candidate(
            deployment_ensemble,
            scales,
            starts,
            maximum_counts,
            incumbent,
            kl_penalty,
            args.cap_epochs,
        )
        penalty_label = f"{kl_penalty:g}".replace(".", "p")
        candidate_id = f"{ENSEMBLE_LABEL}_kl{penalty_label}"
        candidates[candidate_id] = (
            f"equal shared/bounded ensemble argmin with {kl_penalty:g} * KL(w || observed incumbent)",
            candidate,
            kl_penalty,
            ENSEMBLE_LABEL,
        )
        optimization_rows.append(
            {
                "candidate_id": candidate_id,
                "model_variant": ENSEMBLE_LABEL,
                "kl_penalty": kl_penalty,
                "continuous_regularized_objective": continuous_value,
                "runtime_regularized_objective": realized_value,
                "runtime_minus_continuous": realized_value - continuous_value,
                "runtime_unregularized_prediction": float(
                    np.mean(
                        [benchmark.predict(fit, scales[None, :] * candidate[None, :])[0] for fit in deployment_ensemble]
                    )
                ),
                "runtime_kl_to_incumbent": forward_kl(candidate, incumbent),
            }
        )
        for constituent_variant, ensemble in candidate_ensembles.items():
            for seed, constituent in zip(FULL_FIT_SEEDS, ensemble, strict=True):
                constituent_candidate, _, _ = model_candidate(
                    (constituent,),
                    scales,
                    starts,
                    maximum_counts,
                    incumbent,
                    kl_penalty,
                    args.cap_epochs,
                )
                stability_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "constituent_variant": constituent_variant,
                        "fit_seed": seed,
                        "tv_from_partition_ensemble_candidate": float(
                            0.5 * np.abs(constituent_candidate - candidate).sum()
                        ),
                        "hellinger_from_partition_ensemble_candidate": hellinger_distance(
                            constituent_candidate, candidate
                        ),
                    }
                )

    cap_label = f"{args.cap_epochs:g}".replace(".", "p")
    candidates[f"observed_cap{cap_label}_best"] = (
        f"runtime materialization of raw panel run_order={int(frame.iloc[best_row]['run_order'])}",
        incumbent,
        None,
        None,
    )
    candidates["proportional_control"] = ("runtime proportional control", proportional, None, None)

    summary_rows = []
    weight_rows = []
    for candidate_id, (source, candidate, kl_penalty, model_variant) in candidates.items():
        summary_rows.append(
            candidate_summary(
                candidate_id,
                source,
                candidate,
                scales,
                fits,
                candidate_ensembles,
                frame,
                incumbent,
                proportional,
                kl_penalty,
                model_variant,
            )
        )
        for bucket, weight, scale in zip(buckets, candidate, scales, strict=True):
            weight_rows.append(
                {
                    "candidate_id": candidate_id,
                    "bucket": bucket,
                    "phase_0_weight": weight,
                    "phase_0_count": round(weight * MIXTURE_BLOCK_SIZE),
                    "phase_0_materialized_epochs": weight * scale,
                }
            )

    summary = pd.DataFrame(summary_rows)
    long_weights = pd.DataFrame(weight_rows)
    matrix = long_weights.pivot(index="candidate_id", columns="bucket", values="phase_0_weight").loc[
        summary.candidate_id
    ]
    square_roots = np.sqrt(matrix.to_numpy())
    hellinger = np.sqrt(0.5 * ((square_roots[:, None] - square_roots[None, :]) ** 2).sum(axis=2))
    for index, _candidate_id in enumerate(summary.candidate_id):
        summary.loc[index, "minimum_candidate_hellinger"] = float(np.min(np.delete(hellinger[index], index)))

    candidate_summary_path = args.output_dir / "candidate_summary.csv"
    candidate_weights_path = args.output_dir / "candidate_weights.csv"
    optimization_audit_path = args.output_dir / "optimization_audit.csv"
    partition_stability_path = args.output_dir / "partition_stability.csv"
    summary.to_csv(candidate_summary_path, index=False)
    long_weights.to_csv(candidate_weights_path, index=False)
    pd.DataFrame(optimization_rows).to_csv(optimization_audit_path, index=False)
    pd.DataFrame(stability_rows).to_csv(partition_stability_path, index=False)
    manifest = {
        "panel_path": str(benchmark.PANEL_PATH.relative_to(REPO_ROOT)),
        "panel_sha256": hashlib.sha256(benchmark.PANEL_PATH.read_bytes()).hexdigest(),
        "model_path": str(args.model_path.relative_to(REPO_ROOT)),
        "model_sha256": hashlib.sha256(args.model_path.read_bytes()).hexdigest(),
        "selection_target": benchmark.PRIMARY_TARGET,
        "diagnostic_component": "exact-boundary github_cpp_bpb; included in uncheatable_bpb",
        "independent_transfer_guardrail": "historical one-phase Uncheatable panels",
        "phase_0_epoch_cap": args.cap_epochs,
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "candidate_variants": list(candidate_variants),
        "candidate_model_ensemble": ENSEMBLE_LABEL,
        "full_fit_partition_seeds": list(FULL_FIT_SEEDS),
        "minimum_runtime_bucket_support_rows": int(runtime_support_counts.min()),
        "required_runtime_bucket_support_rows": minimum_support,
        "kl_direction": "KL(candidate || runtime-materialized observed incumbent)",
        "kl_penalties": list(kl_penalties),
        "observed_incumbent_run_order": int(frame.iloc[best_row]["run_order"]),
        "observed_incumbent_materialization_tv": float(0.5 * np.abs(weights[best_row] - incumbent).sum()),
        "candidate_ids": summary.candidate_id.tolist(),
        "output_sha256": {
            "candidate_summary.csv": hashlib.sha256(candidate_summary_path.read_bytes()).hexdigest(),
            "candidate_weights.csv": hashlib.sha256(candidate_weights_path.read_bytes()).hexdigest(),
            "optimization_audit.csv": hashlib.sha256(optimization_audit_path.read_bytes()).hexdigest(),
            "partition_stability.csv": hashlib.sha256(partition_stability_path.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
