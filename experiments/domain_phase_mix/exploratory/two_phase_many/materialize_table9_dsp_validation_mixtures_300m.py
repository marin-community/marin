# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Materialize Table-9 DSP validation mixtures from saved fitted models.

This script does not refit DSP. It loads the saved aggregate and per-component
effective-exposure DSP model JSON files, optimizes requested KL-to-proportional
proposal points, and writes source CSVs for the Delphi 3e18 validation launcher.
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
    fit_dsp_l2_kl_sweep_deletion_augmented_300m as dsp_kl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_dsp_validation_mixtures_300m_20260628"
DEFAULT_PER_COMPONENT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628"
DEFAULT_AGGREGATE_DIR = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_kl_sweep_linear_reg_0p0001"
)
DEFAULT_AGGREGATE_MODEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "dsp_effective_exposure"
    / "table9_macro_bpb"
    / "linear_reg_0.0001"
    / "model.json"
)
DEFAULT_OLMIX_WEIGHTS = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "two_phase_adapted_delta_0p01"
    / "proposed_mixture_weights.csv"
)


@dataclass(frozen=True)
class MaterializedMixture:
    family: str
    key: str
    kl_reg: float
    source_csv: str
    predicted_bpb: float
    regularized_objective: float
    mean_phase_tv_to_proportional: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    optimizer_status: str


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def kl_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def read_weight_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T


def write_weight_csv(
    *,
    path: Path,
    domains: list[str],
    natural: np.ndarray,
    weights: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
) -> pd.DataFrame:
    frame = per_component.mixture_frame(
        domains=domains,
        natural=natural,
        weights=weights,
        token_counts=token_counts,
        target_budget=target_budget,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_per_component_models(per_component_dir: Path) -> list[dsp.FittedDSPModel]:
    selected = pd.read_csv(per_component_dir / "selected_component_l2_summary.csv")
    models: list[dsp.FittedDSPModel] = []
    for row in selected.itertuples(index=False):
        component = str(row.component)
        linear_reg = float(row.selected_linear_reg)
        model_path = (
            per_component_dir
            / "component_models"
            / per_component.safe_name(component)
            / f"linear_reg_{linear_reg:g}"
            / "model.json"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing saved component model: {model_path}")
        models.append(dsp.model_from_json(json.loads(model_path.read_text())))
    return models


def existing_starts(
    *,
    natural: np.ndarray,
    aggregate_dir: Path,
    per_component_dir: Path,
    olmix_weights: Path,
) -> list[np.ndarray]:
    starts = [np.stack([natural, natural], axis=0)]
    weights = read_weight_matrix(olmix_weights)
    if weights is not None:
        starts.append(weights)
    for root in [aggregate_dir, per_component_dir]:
        for path in sorted(root.glob("kl_*/proposed_mixture_weights.csv")):
            weights = read_weight_matrix(path)
            if weights is not None:
                starts.append(weights)
    return starts


def capped_starts(starts: list[np.ndarray], *, max_starts: int) -> list[np.ndarray]:
    if max_starts <= 0:
        raise ValueError("--max-starts must be positive")
    return starts[:max_starts]


def materialize_aggregate(
    *,
    model: dsp.FittedDSPModel,
    kl_values: list[float],
    starts: list[np.ndarray],
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    output_dir: Path,
    max_starts: int,
    maxiter: int,
) -> list[MaterializedMixture]:
    rows: list[MaterializedMixture] = []
    running_starts = list(starts)
    for kl_reg in kl_values:
        print(f"Optimizing aggregate DSP KL={kl_reg:g}", flush=True)
        weights, regularized, status = optimize_aggregate_kl(
            model=model,
            natural=natural,
            kl_reg=float(kl_reg),
            starts=capped_starts(running_starts, max_starts=max_starts),
            maxiter=maxiter,
        )
        running_starts.append(weights)
        key = f"dsp_effexp_table9_kl{kl_slug(kl_reg)}"
        csv_path = output_dir / "mixtures" / f"{key}.csv"
        frame = write_weight_csv(
            path=csv_path,
            domains=domains,
            natural=natural,
            weights=weights,
            token_counts=token_counts,
            target_budget=target_budget,
        )
        rows.append(
            MaterializedMixture(
                family="aggregate_effective_exposure_dsp",
                key=key,
                kl_reg=float(kl_reg),
                source_csv=str(csv_path),
                predicted_bpb=float(dsp.predict(model, weights[None, :, :])[0]),
                regularized_objective=float(regularized),
                mean_phase_tv_to_proportional=float(0.5 * np.abs(weights - np.stack([natural, natural])).sum(axis=1).mean()),
                max_simulated_epoch=float(frame["simulated_epochs"].max()),
                q95_simulated_epoch=float(frame["simulated_epochs"].quantile(0.95)),
                optimizer_status=status,
            )
        )
    return rows


def materialize_per_component(
    *,
    models: list[dsp.FittedDSPModel],
    kl_values: list[float],
    starts: list[np.ndarray],
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    output_dir: Path,
    max_starts: int,
    maxiter: int,
) -> list[MaterializedMixture]:
    rows: list[MaterializedMixture] = []
    running_starts = list(starts)
    for kl_reg in kl_values:
        print(f"Optimizing per-component DSP KL={kl_reg:g}", flush=True)
        existing_path = output_dir / "mixtures" / f"dsp_percomp_table9_kl{kl_slug(kl_reg)}.csv"
        if existing_path.exists():
            print(f"  reusing existing {existing_path}", flush=True)
            frame = pd.read_csv(existing_path)
            weights = frame[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T
            regularized = per_component.per_component_objective(models, weights, natural, float(kl_reg))
            status = "reused_existing_csv"
        else:
            weights, regularized, status = optimize_per_component_kl(
                models=models,
                natural=natural,
                kl_reg=float(kl_reg),
                starts=capped_starts(running_starts, max_starts=max_starts),
                maxiter=maxiter,
            )
        running_starts.append(weights)
        key = f"dsp_percomp_table9_kl{kl_slug(kl_reg)}"
        csv_path = output_dir / "mixtures" / f"{key}.csv"
        frame = write_weight_csv(
            path=csv_path,
            domains=domains,
            natural=natural,
            weights=weights,
            token_counts=token_counts,
            target_budget=target_budget,
        )
        rows.append(
            MaterializedMixture(
                family="per_component_effective_exposure_dsp",
                key=key,
                kl_reg=float(kl_reg),
                source_csv=str(csv_path),
                predicted_bpb=float(np.mean(per_component.predict_component_matrix(models, weights[None, :, :]))),
                regularized_objective=float(regularized),
                mean_phase_tv_to_proportional=float(0.5 * np.abs(weights - np.stack([natural, natural])).sum(axis=1).mean()),
                max_simulated_epoch=float(frame["simulated_epochs"].max()),
                q95_simulated_epoch=float(frame["simulated_epochs"].quantile(0.95)),
                optimizer_status=status,
            )
        )
    return rows


def optimize_aggregate_kl(
    *,
    model: dsp.FittedDSPModel,
    natural: np.ndarray,
    kl_reg: float,
    starts: list[np.ndarray],
    maxiter: int,
) -> tuple[np.ndarray, float, str]:
    m = len(natural)

    def objective(logits: np.ndarray) -> float:
        weights = dsp_kl.softmax_pair(logits, m)
        return float(dsp.predict(model, weights[None, :, :])[0]) + kl_reg * base.weighted_multiclass_kl(
            weights,
            natural,
            base.PHASE_FRACTIONS,
        )

    return optimize_logits(objective=objective, starts=starts, m=m, maxiter=maxiter)


def optimize_per_component_kl(
    *,
    models: list[dsp.FittedDSPModel],
    natural: np.ndarray,
    kl_reg: float,
    starts: list[np.ndarray],
    maxiter: int,
) -> tuple[np.ndarray, float, str]:
    m = len(natural)

    def objective(logits: np.ndarray) -> float:
        weights = dsp_kl.softmax_pair(logits, m)
        return per_component.per_component_objective(models, weights, natural, kl_reg)

    return optimize_logits(objective=objective, starts=starts, m=m, maxiter=maxiter)


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
            options={"maxiter": maxiter, "ftol": 1e-9, "maxls": 20},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("KL optimization failed")
    return dsp_kl.softmax_pair(np.asarray(best.x, dtype=float), m), float(best.fun), str(best.message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-component-dir", type=Path, default=DEFAULT_PER_COMPONENT_DIR)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--aggregate-model", type=Path, default=DEFAULT_AGGREGATE_MODEL)
    parser.add_argument("--olmix-weights", type=Path, default=DEFAULT_OLMIX_WEIGHTS)
    parser.add_argument("--aggregate-kl-values", default="0.25,0.3,0.4")
    parser.add_argument("--per-component-kl-values", default="0.025,0.05,0.1,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--max-starts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _signal, _columns, domains, natural = base.load_raw_signal_panel()
    target_budget = base.load_target_budget()
    token_counts = base.load_domain_token_counts(domains)
    starts = existing_starts(
        natural=natural,
        aggregate_dir=args.aggregate_dir,
        per_component_dir=args.per_component_dir,
        olmix_weights=args.olmix_weights,
    )
    aggregate_model = dsp.model_from_json(json.loads(args.aggregate_model.read_text()))
    per_component_models = load_per_component_models(args.per_component_dir)
    rows = [
        *materialize_aggregate(
            model=aggregate_model,
            kl_values=parse_float_list(args.aggregate_kl_values),
            starts=starts,
            domains=domains,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            output_dir=args.output_dir,
            max_starts=int(args.max_starts),
            maxiter=int(args.maxiter),
        ),
        *materialize_per_component(
            models=per_component_models,
            kl_values=parse_float_list(args.per_component_kl_values),
            starts=starts,
            domains=domains,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            output_dir=args.output_dir,
            max_starts=int(args.max_starts),
            maxiter=int(args.maxiter),
        ),
    ]
    summary = pd.DataFrame([asdict(row) for row in rows]).sort_values(["family", "kl_reg"])
    summary.to_csv(args.output_dir / "materialized_mixture_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
