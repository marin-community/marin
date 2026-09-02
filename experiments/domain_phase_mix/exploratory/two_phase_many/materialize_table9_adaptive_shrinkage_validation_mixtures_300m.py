# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Materialize adaptive-shrinkage Table-9 DSP validation mixtures.

This script loads the selected per-component effective-exposure DSP models and
the reliability weights from the 300M OLMoBaseEval Easy Table-9 diagnostics. It
then optimizes a small, hypothesis-driven panel of fixed and candidate-dependent
shrinkage objectives at the same KL trust-region used for the current best
3e18 validation candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_adaptive_shrinkage_300m as shrinkage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_l2_kl_sweep_deletion_augmented_300m as dsp_kl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_table9_dsp_validation_mixtures_300m as validation_materializer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_adaptive_shrinkage_validation_mixtures_300m_20260628"
DEFAULT_PER_COMPONENT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628"
DEFAULT_SHRINKAGE_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_adaptive_shrinkage_300m_20260628"
DEFAULT_DSP_VALIDATION_DIR = REFERENCE_OUTPUTS / "table9_dsp_validation_mixtures_300m_20260628"


@dataclass(frozen=True)
class ShrinkageSpec:
    key: str
    method: str
    reliability: str
    kl_reg: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class MaterializedShrinkageMixture:
    key: str
    method: str
    reliability: str
    kl_reg: float
    beta: float
    gamma: float
    source_csv: str
    predicted_bpb: float
    regularized_objective: float
    mean_phase_tv_to_proportional: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_weight: float
    optimizer_status: str


PANEL_SPECS: tuple[ShrinkageSpec, ...] = (
    ShrinkageSpec(
        key="dsp_shrink_fixed_spearman_kl0p2",
        method="fixed_component_shrinkage",
        reliability="oof_spearman_pos",
        kl_reg=0.2,
        beta=0.0,
        gamma=0.0,
    ),
    ShrinkageSpec(
        key="dsp_shrink_fixed_r2harm_kl0p2",
        method="fixed_component_shrinkage",
        reliability="oof_r2_x_harm_t",
        kl_reg=0.2,
        beta=0.0,
        gamma=0.0,
    ),
    ShrinkageSpec(
        key="dsp_shrink_tv_spearman_b0p5_kl0p2",
        method="tv_adaptive_shrinkage",
        reliability="oof_spearman_pos",
        kl_reg=0.2,
        beta=0.5,
        gamma=0.0,
    ),
    ShrinkageSpec(
        key="dsp_shrink_tv_r2harm_b1_kl0p2",
        method="tv_adaptive_shrinkage",
        reliability="oof_r2_x_harm_t",
        kl_reg=0.2,
        beta=1.0,
        gamma=0.0,
    ),
    ShrinkageSpec(
        key="dsp_shrink_delta_spearman_b0p5_kl0p2",
        method="delta_adaptive_shrinkage",
        reliability="oof_spearman_pos",
        kl_reg=0.2,
        beta=0.5,
        gamma=0.0,
    ),
    ShrinkageSpec(
        key="dsp_shrink_delta_r2harm_b1_kl0p2",
        method="delta_adaptive_shrinkage",
        reliability="oof_r2_x_harm_t",
        kl_reg=0.2,
        beta=1.0,
        gamma=0.0,
    ),
    ShrinkageSpec(
        key="dsp_shrink_unc_spearman_g0p25_kl0p2",
        method="tv_uncertainty_penalty",
        reliability="oof_spearman_pos",
        kl_reg=0.2,
        beta=0.0,
        gamma=0.25,
    ),
    ShrinkageSpec(
        key="dsp_shrink_unc_r2harm_g0p5_kl0p2",
        method="tv_uncertainty_penalty",
        reliability="oof_r2_x_harm_t",
        kl_reg=0.2,
        beta=0.0,
        gamma=0.5,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-component-dir", type=Path, default=DEFAULT_PER_COMPONENT_DIR)
    parser.add_argument("--shrinkage-dir", type=Path, default=DEFAULT_SHRINKAGE_DIR)
    parser.add_argument("--dsp-validation-dir", type=Path, default=DEFAULT_DSP_VALIDATION_DIR)
    parser.add_argument("--max-starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=160)
    return parser.parse_args()


def phase_tv_to_proportional(weights: np.ndarray, natural: np.ndarray) -> float:
    reference = np.stack([natural, natural], axis=0)
    return float(0.5 * np.abs(weights - reference).sum(axis=1).mean())


def load_reliability_arrays(path: Path) -> dict[str, np.ndarray]:
    quality = pd.read_csv(path)
    return shrinkage.reliability_arrays(quality)


def load_delta_scale(path: Path, components: list[str], selected: pd.DataFrame) -> tuple[np.ndarray, float]:
    oof = pd.read_csv(path)
    predicted = oof[[f"pred::{component}" for component in components]].to_numpy(dtype=float)
    prop_rows = oof["run_name"].eq("baseline_proportional")
    if int(prop_rows.sum()) != 1:
        raise ValueError("Expected one baseline_proportional row in selected component OOF predictions")
    prop_pred = predicted[int(np.flatnonzero(prop_rows.to_numpy())[0])]
    component_delta_scale = np.median(np.abs(predicted - prop_pred), axis=0) + 1e-12
    component_rmse = selected["selected_oof_rmse"].to_numpy(dtype=float)
    macro_sigma = float(np.sqrt(np.mean(component_rmse**2)) / np.sqrt(len(component_rmse)))
    return component_delta_scale, macro_sigma


def load_panel_tv_scale(path: Path) -> float:
    panel = pd.read_csv(path)
    prop_rows = panel["run_name"].eq("baseline_proportional")
    if int(prop_rows.sum()) != 1:
        raise ValueError("Expected one baseline_proportional row in fit panel")
    prop_idx = int(np.flatnonzero(prop_rows.to_numpy())[0])
    tv = shrinkage.phase_tv_to_proportional(panel, prop_idx)
    return float(np.median(tv[tv > 0.0]))


def existing_starts(natural: np.ndarray, dsp_validation_dir: Path) -> list[np.ndarray]:
    starts = [np.stack([natural, natural], axis=0)]
    for key in [
        "dsp_percomp_table9_kl0p2",
        "dsp_percomp_table9_kl0p25",
        "dsp_effexp_table9_kl0p2",
        "dsp_effexp_table9_kl0p25",
        "dsp_percomp_table9_kl0p3",
    ]:
        path = dsp_validation_dir / "mixtures" / f"{key}.csv"
        weights = validation_materializer.read_weight_matrix(path)
        if weights is not None:
            starts.append(weights)
    return starts


def optimize_spec(
    *,
    spec: ShrinkageSpec,
    models: list,
    reliability: dict[str, np.ndarray],
    natural: np.ndarray,
    starts: list[np.ndarray],
    tv_scale: float,
    component_delta_scale: np.ndarray,
    macro_sigma: float,
    max_starts: int,
    maxiter: int,
) -> tuple[np.ndarray, float, str, float]:
    if spec.reliability not in reliability:
        raise ValueError(f"Unknown reliability {spec.reliability}")
    r = reliability[spec.reliability]
    prop_pred = per_component.predict_component_matrix(models, np.stack([natural, natural], axis=0)[None, :, :])[0]
    m = len(natural)

    def objective(logits: np.ndarray) -> float:
        weights = dsp_kl.softmax_pair(logits, m)
        pred = per_component.predict_component_matrix(models, weights[None, :, :])[0]
        if spec.method == "fixed_component_shrinkage":
            adjusted = prop_pred + (pred - prop_pred) * r
        elif spec.method == "tv_adaptive_shrinkage":
            tv = phase_tv_to_proportional(weights, natural)
            adjusted_r = r / (1.0 + spec.beta * (tv / tv_scale))
            adjusted = prop_pred + (pred - prop_pred) * adjusted_r
        elif spec.method == "delta_adaptive_shrinkage":
            adjusted_r = r / (1.0 + spec.beta * np.abs(pred - prop_pred) / component_delta_scale)
            adjusted = prop_pred + (pred - prop_pred) * adjusted_r
        elif spec.method == "tv_uncertainty_penalty":
            tv = phase_tv_to_proportional(weights, natural)
            adjusted = prop_pred + (pred - prop_pred) * r
            return float(np.mean(adjusted)) + spec.gamma * macro_sigma * (1.0 + tv / tv_scale) + spec.kl_reg * base.weighted_multiclass_kl(
                weights,
                natural,
                base.PHASE_FRACTIONS,
            )
        else:
            raise ValueError(f"Unknown shrinkage method {spec.method}")
        return float(np.mean(adjusted)) + spec.kl_reg * base.weighted_multiclass_kl(
            weights,
            natural,
            base.PHASE_FRACTIONS,
        )

    weights, regularized, status = optimize_logits(
        objective=objective,
        starts=validation_materializer.capped_starts(starts, max_starts=max_starts),
        m=m,
        maxiter=maxiter,
    )
    raw_pred = float(np.mean(per_component.predict_component_matrix(models, weights[None, :, :])[0]))
    return weights, regularized, status, raw_pred


def optimize_logits(
    *,
    objective,
    starts: list[np.ndarray],
    m: int,
    maxiter: int,
) -> tuple[np.ndarray, float, str]:
    best = None
    for start in starts:
        result = minimize(
            objective,
            dsp_kl.weights_to_logits(start),
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxfun": maxiter * 400, "ftol": 1e-10, "maxls": 40},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Adaptive-shrinkage optimization failed")
    return dsp_kl.softmax_pair(np.asarray(best.x, dtype=float), m), float(best.fun), str(best.message)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _signal, _columns, domains, natural = base.load_raw_signal_panel()
    target_budget = base.load_target_budget()
    token_counts = base.load_domain_token_counts(domains)
    models = validation_materializer.load_per_component_models(args.per_component_dir)
    selected = pd.read_csv(args.per_component_dir / "selected_component_l2_summary.csv")
    components = selected["component"].astype(str).tolist()
    reliability = load_reliability_arrays(args.shrinkage_dir / "component_reliability_weights.csv")
    component_delta_scale, macro_sigma = load_delta_scale(
        args.per_component_dir / "selected_component_oof_predictions.csv",
        components,
        selected,
    )
    tv_scale = load_panel_tv_scale(args.per_component_dir / "fit_panel_table9_macro.csv")
    starts = existing_starts(natural, args.dsp_validation_dir)
    rows: list[MaterializedShrinkageMixture] = []
    for spec in PANEL_SPECS:
        print(f"Optimizing {spec.key}", flush=True)
        weights, regularized, status, raw_pred = optimize_spec(
            spec=spec,
            models=models,
            reliability=reliability,
            natural=natural,
            starts=starts,
            tv_scale=tv_scale,
            component_delta_scale=component_delta_scale,
            macro_sigma=macro_sigma,
            max_starts=int(args.max_starts),
            maxiter=int(args.maxiter),
        )
        starts.append(weights)
        csv_path = args.output_dir / "mixtures" / f"{spec.key}.csv"
        frame = validation_materializer.write_weight_csv(
            path=csv_path,
            domains=domains,
            natural=natural,
            weights=weights,
            token_counts=token_counts,
            target_budget=target_budget,
        )
        rows.append(
            MaterializedShrinkageMixture(
                key=spec.key,
                method=spec.method,
                reliability=spec.reliability,
                kl_reg=spec.kl_reg,
                beta=spec.beta,
                gamma=spec.gamma,
                source_csv=str(csv_path),
                predicted_bpb=raw_pred,
                regularized_objective=float(regularized),
                mean_phase_tv_to_proportional=phase_tv_to_proportional(weights, natural),
                max_simulated_epoch=float(frame["simulated_epochs"].max()),
                q95_simulated_epoch=float(frame["simulated_epochs"].quantile(0.95)),
                max_weight=float(max(frame["phase_0_weight"].max(), frame["phase_1_weight"].max())),
                optimizer_status=status,
            )
        )
    summary = pd.DataFrame([asdict(row) for row in rows]).sort_values("predicted_bpb")
    summary.to_csv(args.output_dir / "materialized_mixture_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
