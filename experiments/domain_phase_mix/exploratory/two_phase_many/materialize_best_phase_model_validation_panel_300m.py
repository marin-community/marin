# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Materialize a paired 3e18 panel from the incumbent two-phase DSP model.

The incumbent is full effective-exposure DSP with nonnegative phase-TV and
aggregate-HHI terms. For Uncheatable BPB and Table-9 macro BPB, this script
fits the model on the joint 300M panel and optimizes two KL settings under:

* a free two-phase policy;
* the exact aggregate-matched tied control of that two-phase proposal; and
* an independently optimized one-phase policy.

The resulting 12 mixtures distinguish aggregate-mixture quality from phase
ordering while testing a supported and a more aggressive trust region. This
script only materializes candidates; it does not upload or submit jobs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_nested_coverage_dsp_optima as optimum,
)
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
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "best_phase_model_validation_panel_20260710"
DEFAULT_KL_VALUES = "5,10"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
MODEL_CONFIG = coverage.FitConfig(
    "full_effective_exposure_tv_hhi",
    True,
    "effective_exposure",
    (0, 1),
)


@dataclass(frozen=True)
class OptimizerResult:
    weights: np.ndarray
    objective: float
    successful_starts: int


def weights_from_logits(logits: np.ndarray, num_domains: int, one_phase: bool) -> np.ndarray:
    if one_phase:
        exponent = np.exp(logits - np.max(logits))
        weights = exponent / exponent.sum()
        return np.stack([weights, weights])
    return optimum.weights_from_logits(logits, num_domains)


def logits_for_weights(weights: np.ndarray, one_phase: bool, alpha0: float, alpha1: float) -> np.ndarray:
    if one_phase:
        aggregate = alpha0 * weights[0] + alpha1 * weights[1]
        return np.log(np.clip(aggregate, 1e-12, 1.0))
    return np.log(np.clip(weights, 1e-12, 1.0)).reshape(-1)


def unique_starts(starts: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for start in starts:
        if not any(np.allclose(start, previous) for previous in unique):
            unique.append(start)
    return unique


def optimize_candidate(
    dataset: pooled.Dataset,
    model: coverage.CoverageModel,
    natural: np.ndarray,
    kl_reg: float,
    *,
    one_phase: bool,
    alpha0: float,
    alpha1: float,
) -> OptimizerResult:
    num_domains = dataset.m

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits, num_domains, one_phase)
        prediction = float(coverage.predict(model, weights[None, :, :], alpha0, alpha1)[0])
        return prediction + kl_reg * optimum.weighted_kl(weights, natural, alpha0, alpha1)

    proportional = np.stack([natural, natural])
    starts = [logits_for_weights(proportional, one_phase, alpha0, alpha1)]
    starts.extend(
        logits_for_weights(dataset.weights[index], one_phase, alpha0, alpha1) for index in np.argsort(dataset.y)[:8]
    )
    best_value = np.inf
    best_weights: np.ndarray | None = None
    successful_starts = 0
    for start in unique_starts(starts):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 600, "ftol": 1e-10, "maxls": 40},
        )
        if result.success:
            successful_starts += 1
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_weights = weights_from_logits(np.asarray(result.x, dtype=float), num_domains, one_phase)
    if best_weights is None:
        raise RuntimeError(f"No finite optimizer result for one_phase={one_phase}, KL={kl_reg}")
    return OptimizerResult(best_weights, best_value, successful_starts)


def candidate_key(objective: str, policy: str, kl_reg: float) -> str:
    kl_name = f"{kl_reg:g}".replace(".", "p")
    return f"bestphase_{objective}_{policy}_kl{kl_name}"


def mean_phase_tv(weights: np.ndarray, reference: np.ndarray) -> float:
    return float(0.5 * np.abs(weights - reference).sum(axis=1).mean())


def validate_weights(weights: np.ndarray, key: str) -> None:
    if weights.shape[0] != 2:
        raise ValueError(f"{key}: expected two phase rows, got {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.min(weights) < -1e-12:
        raise ValueError(f"{key}: weights must be finite and nonnegative")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{key}: phase weights do not sum to one: {weights.sum(axis=1)}")


def append_candidate(
    *,
    output_dir: Path,
    objective: str,
    policy: str,
    pair_key: str,
    kl_reg: float,
    weights: np.ndarray,
    model: coverage.CoverageModel,
    dataset: pooled.Dataset,
    natural: np.ndarray,
    domains: list[str],
    token_counts: np.ndarray,
    target_budget: int,
    alpha0: float,
    alpha1: float,
    optimizer: OptimizerResult | None,
    rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
) -> None:
    key = candidate_key(objective, policy, kl_reg)
    validate_weights(weights, key)
    frame = per_component.mixture_frame(
        domains=domains,
        natural=natural,
        weights=weights,
        token_counts=token_counts,
        target_budget=target_budget,
    )
    mixture_dir = output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(mixture_dir / f"{key}.csv", index=False)

    proportional = np.stack([natural, natural])
    prediction = float(coverage.predict(model, weights[None, :, :], alpha0, alpha1)[0])
    weighted_kl = optimum.weighted_kl(weights, natural, alpha0, alpha1)
    simulated_epochs = olmix.simulated_epochs(weights, token_counts, target_budget=target_budget)
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    rows.append(
        {
            "candidate": key,
            "objective": objective,
            "policy": policy,
            "pair_key": pair_key,
            "kl_reg": kl_reg,
            "model": MODEL_CONFIG.name,
            "linear_reg": coverage.dataset_linear_reg(dataset),
            "predicted_bpb_300m": prediction,
            "weighted_kl_to_proportional": weighted_kl,
            "regularized_objective": prediction + kl_reg * weighted_kl,
            "tv_to_proportional": mean_phase_tv(weights, proportional),
            "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
            "aggregate_hhi": float(np.sum(aggregate**2)),
            "max_weight": float(np.max(weights)),
            "max_simulated_epoch": float(np.max(simulated_epochs)),
            "q95_simulated_epoch": float(np.quantile(simulated_epochs, 0.95)),
            "optimizer_successful_starts": optimizer.successful_starts if optimizer else None,
            "weights_csv": f"mixtures/{key}.csv",
        }
    )
    for phase in range(2):
        for domain, value in zip(domains, weights[phase], strict=True):
            weight_rows.append(
                {
                    "candidate": key,
                    "objective": objective,
                    "policy": policy,
                    "pair_key": pair_key,
                    "kl_reg": kl_reg,
                    "phase": phase,
                    "domain": domain,
                    "weight": float(value),
                }
            )


def write_plots(manifest: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("predicted_bpb_300m", "max_simulated_epoch", "tv_to_proportional"):
        figure = px.line(
            manifest,
            x="kl_reg",
            y=metric,
            color="policy",
            facet_col="objective",
            markers=True,
            color_discrete_sequence=["#d73027", "#fee08b", "#1a9850"],
            title=f"Incumbent phase-model validation panel: {metric}",
        )
        figure.write_html(
            output_dir / f"panel_{metric}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="uncheatable,table9")
    parser.add_argument("--kl-values", default=DEFAULT_KL_VALUES)
    parser.add_argument("--maxiter", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = list(pooled.load_300m_dataset("table9").domain_names)
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    objectives = [part.strip() for part in args.objectives.split(",") if part.strip()]
    unknown = sorted(set(objectives).difference(joint.TARGET_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    kl_values = pooled.parse_float_list(args.kl_values)
    if not kl_values or min(kl_values) <= 0.0:
        raise ValueError("KL values must be positive")

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for objective in objectives:
        target = joint.TARGET_COLUMNS[objective]
        fit_frame = frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy()
        dataset = joint.dataset_from_frame(objective, fit_frame, target)
        alpha0, alpha1 = coverage.phase_fractions(dataset)
        natural = optimum.natural_weights(dataset)
        _packet, loaded_domains, loaded_natural, token_counts, target_budget, _folds = bowl.load_objective(objective)
        if domains != list(loaded_domains):
            raise ValueError(f"{objective}: domain order differs between fit and materialization inputs")
        if not np.allclose(natural, loaded_natural):
            raise ValueError(f"{objective}: proportional weights differ between fit and materialization inputs")

        model = coverage.fit_model(
            dataset,
            np.arange(dataset.n),
            MODEL_CONFIG,
            linear_reg=coverage.dataset_linear_reg(dataset),
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        for kl_reg in sorted(kl_values):
            pair_key = f"{objective}_kl{kl_reg:g}".replace(".", "p")
            two_phase = optimize_candidate(
                dataset,
                model,
                natural,
                kl_reg,
                one_phase=False,
                alpha0=alpha0,
                alpha1=alpha1,
            )
            aggregate = alpha0 * two_phase.weights[0] + alpha1 * two_phase.weights[1]
            tied_weights = np.stack([aggregate, aggregate])
            one_phase = optimize_candidate(
                dataset,
                model,
                natural,
                kl_reg,
                one_phase=True,
                alpha0=alpha0,
                alpha1=alpha1,
            )
            for policy, weights, optimizer_result in (
                ("2p", two_phase.weights, two_phase),
                ("tied", tied_weights, None),
                ("1p", one_phase.weights, one_phase),
            ):
                append_candidate(
                    output_dir=args.output_dir,
                    objective=objective,
                    policy=policy,
                    pair_key=pair_key,
                    kl_reg=kl_reg,
                    weights=weights,
                    model=model,
                    dataset=dataset,
                    natural=natural,
                    domains=domains,
                    token_counts=np.asarray(token_counts, dtype=float),
                    target_budget=int(target_budget),
                    alpha0=alpha0,
                    alpha1=alpha1,
                    optimizer=optimizer_result,
                    rows=rows,
                    weight_rows=weight_rows,
                )
            if not np.allclose(
                alpha0 * two_phase.weights[0] + alpha1 * two_phase.weights[1],
                alpha0 * tied_weights[0] + alpha1 * tied_weights[1],
                atol=1e-10,
            ):
                raise AssertionError(f"{pair_key}: tied control does not preserve aggregate mixture")

    manifest = pd.DataFrame(rows).sort_values(["objective", "kl_reg", "policy"]).reset_index(drop=True)
    if len(manifest) != len(objectives) * len(kl_values) * 3:
        raise AssertionError(f"Expected {len(objectives) * len(kl_values) * 3} candidates, got {len(manifest)}")
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(args.output_dir / "candidate_weights_long.csv", index=False)
    write_plots(manifest, args.output_dir)
    print(manifest.to_string(index=False))
    print(f"Wrote {len(manifest)} validation candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
