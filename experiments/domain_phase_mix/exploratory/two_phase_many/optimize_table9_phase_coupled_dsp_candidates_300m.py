# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Optimize Table-9 DSP candidates with an explicit phase-coupling penalty."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_olmix_reference_deletion_augmented_300m as base  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_phase_coupled_dsp_candidates_20260630"
DEFAULT_MODEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "dsp_effective_exposure"
    / "table9_macro_bpb"
    / "linear_reg_0.0001"
    / "model.json"
)
DEFAULT_PANEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_extra_300m_heldout_eval_20260630"
    / "expanded_300m_table9_diagnostic_panel.csv"
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TARGET_COL = "table9_macro_bpb"
PHASE_EPS = 1e-12


@dataclass(frozen=True)
class CandidateSummary:
    kl_reg: float
    phase_coupling_reg: float
    predicted_bpb: float
    regularized_objective: float
    proportional_kl: float
    phase_symmetric_kl: float
    phase_tv: float
    tv_to_proportional: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    nearest_observed_run_name: str
    nearest_observed_bpb: float
    nearest_observed_phase_tv_distance: float
    optimizer_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--kl-reg-values", default="0.1,0.2,0.3")
    parser.add_argument("--phase-coupling-values", default="0,0.001,0.003,0.01,0.03,0.1,0.3,1,3")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def softmax_pair(logits: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros((2, m), dtype=float)
    for phase_idx in range(2):
        phase_logits = logits[phase_idx * m : (phase_idx + 1) * m]
        exp = np.exp(phase_logits - np.max(phase_logits))
        out[phase_idx] = exp / exp.sum()
    return out


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    return np.log(np.clip(weights, PHASE_EPS, 1.0)).reshape(-1)


def multinomial_kl(p: np.ndarray, q: np.ndarray) -> float:
    p_clip = np.clip(p, PHASE_EPS, 1.0)
    q_clip = np.clip(q, PHASE_EPS, 1.0)
    return float(np.sum(p_clip * (np.log(p_clip) - np.log(q_clip))))


def proportional_kl(weights: np.ndarray, natural: np.ndarray) -> float:
    return float(sum(float(frac) * multinomial_kl(weights[idx], natural) for idx, frac in enumerate(base.PHASE_FRACTIONS)))


def phase_symmetric_kl(weights: np.ndarray) -> float:
    return 0.5 * (multinomial_kl(weights[0], weights[1]) + multinomial_kl(weights[1], weights[0]))


def load_candidate_csv(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path).set_index("domain")
    weights = np.stack(
        [
            frame.loc[domains, "phase_0_weight"].to_numpy(dtype=float),
            frame.loc[domains, "phase_1_weight"].to_numpy(dtype=float),
        ],
        axis=0,
    )
    return dsp.normalize_weights(weights[None, :, :])[0]


def start_weights(domains: list[str], natural: np.ndarray, panel: pd.DataFrame) -> list[np.ndarray]:
    starts = [np.stack([natural, natural], axis=0)]
    known_paths = [
        REFERENCE_OUTPUTS
        / "olmo_base_easy_one_phase_model_sweeps_300m_20260628"
        / "dsp_one_phase_effexp_linear_reg0p0001_kl0p1"
        / "proposed_mixture_weights.csv",
        REFERENCE_OUTPUTS
        / "table9_dsp_phase_functional_form_20260630"
        / "proposal_screen"
        / "mixtures"
        / "split_saturation_penalty_l2_0p01_kl_0p3.csv",
        REFERENCE_OUTPUTS
        / "table9_dsp_phase_functional_form_20260630"
        / "proposal_screen"
        / "mixtures"
        / "effective_exposure_l2_0p01_kl_0p5.csv",
    ]
    for path in known_paths:
        if path.exists():
            starts.append(load_candidate_csv(path, domains))
    phase_columns = [f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]
    best_rows = panel.sort_values(TARGET_COL).head(20)
    for _idx, row in best_rows.iterrows():
        starts.append(row[phase_columns].to_numpy(dtype=float).reshape(2, len(domains)))
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for weights in starts:
        normalized = dsp.normalize_weights(weights[None, :, :])[0]
        key = np.round(normalized, 12).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def load_panel_weights(panel: pd.DataFrame, domains: list[str]) -> np.ndarray:
    columns = [f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]
    return dsp.normalize_weights(panel[columns].to_numpy(dtype=float).reshape(len(panel), 2, len(domains)))


def objective(
    logits: np.ndarray,
    *,
    model: dsp.FittedDSPModel,
    m: int,
    natural: np.ndarray,
    kl_reg: float,
    phase_reg: float,
) -> float:
    weights = softmax_pair(logits, m)
    pred = float(dsp.predict(model, weights[None, :, :])[0])
    return pred + kl_reg * proportional_kl(weights, natural) + phase_reg * phase_symmetric_kl(weights)


def nearest_observed(weights: np.ndarray, panel: pd.DataFrame, panel_weights: np.ndarray) -> tuple[str, float, float]:
    distances = 0.5 * np.abs(panel_weights - weights[None, :, :]).sum(axis=2).mean(axis=1)
    idx = int(np.argmin(distances))
    return str(panel.iloc[idx]["run_name"]), float(panel.iloc[idx][TARGET_COL]), float(distances[idx])


def optimize_candidate(
    *,
    model: dsp.FittedDSPModel,
    starts: list[np.ndarray],
    natural: np.ndarray,
    kl_reg: float,
    phase_reg: float,
) -> tuple[np.ndarray, float, str]:
    m = len(natural)
    best: Any | None = None
    for weights in starts:
        def loss(logits: np.ndarray) -> float:
            return objective(logits, model=model, m=m, natural=natural, kl_reg=kl_reg, phase_reg=phase_reg)

        result = minimize(
            loss,
            weights_to_logits(weights),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("No optimization starts were evaluated")
    return softmax_pair(np.asarray(best.x, dtype=float), m), float(best.fun), str(best.message)


def write_weights(path: Path, domains: list[str], natural: np.ndarray, weights: np.ndarray, token_counts: np.ndarray, target_budget: int) -> None:
    sim_epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
    frame = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "aggregate_weight": base.aggregate_phase_weights(weights),
            "available_tokens": token_counts,
            "simulated_epochs": sim_epochs,
            "phase_0_epoch_multiplier": weights[0] / np.clip(natural, PHASE_EPS, None),
            "phase_1_epoch_multiplier": weights[1] / np.clip(natural, PHASE_EPS, None),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_plot(path: Path, summary: pd.DataFrame) -> None:
    fig = go.Figure()
    for kl_reg, group in summary.groupby("kl_reg", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["phase_coupling_reg"].map(lambda value: f"{float(value):g}"),
                y=group["predicted_bpb"],
                mode="lines+markers",
                name=f"KL={float(kl_reg):g}",
                customdata=np.stack([group["phase_tv"], group["max_simulated_epoch"], group["nearest_observed_bpb"]], axis=1),
                hovertemplate=(
                    "phase coupling=%{x}<br>pred=%{y:.5f}<br>phase TV=%{customdata[0]:.3f}"
                    "<br>max epoch=%{customdata[1]:.2f}<br>nearest observed=%{customdata[2]:.5f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Table-9 effective-exposure DSP with phase-coupling penalty",
        xaxis_title="Phase-coupling symmetric KL penalty",
        yaxis_title="Predicted Table-9 macro BPB",
        template="plotly_white",
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def safe_float_label(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = dsp.model_from_json(json.loads(args.model.read_text()))
    domains = list(model.domain_names)
    _signal, _columns, _domains, natural = base.load_raw_signal_panel()
    if _domains != domains:
        raise ValueError("Model domain order does not match raw signal domain order")
    panel = pd.read_csv(args.panel, low_memory=False)
    panel_weights = load_panel_weights(panel, domains)
    starts = start_weights(domains, natural, panel)
    token_counts = base.load_domain_token_counts(domains)
    target_budget = base.load_target_budget()
    rows: list[CandidateSummary] = []
    for kl_reg in parse_float_list(args.kl_reg_values):
        for phase_reg in parse_float_list(args.phase_coupling_values):
            print(f"Optimizing KL={kl_reg:g} phase_coupling={phase_reg:g}", flush=True)
            weights, regularized_objective, status = optimize_candidate(
                model=model,
                starts=starts,
                natural=natural,
                kl_reg=kl_reg,
                phase_reg=phase_reg,
            )
            nearest_name, nearest_bpb, nearest_distance = nearest_observed(weights, panel, panel_weights)
            sim_epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
            pred = float(dsp.predict(model, weights[None, :, :])[0])
            rows.append(
                CandidateSummary(
                    kl_reg=float(kl_reg),
                    phase_coupling_reg=float(phase_reg),
                    predicted_bpb=pred,
                    regularized_objective=float(regularized_objective),
                    proportional_kl=proportional_kl(weights, natural),
                    phase_symmetric_kl=phase_symmetric_kl(weights),
                    phase_tv=float(0.5 * np.abs(weights[0] - weights[1]).sum()),
                    tv_to_proportional=float(0.5 * np.abs(weights - np.stack([natural, natural])[None, :, :]).sum(axis=2).mean()),
                    max_simulated_epoch=float(np.max(sim_epochs)),
                    q95_simulated_epoch=float(np.quantile(sim_epochs, 0.95)),
                    nearest_observed_run_name=nearest_name,
                    nearest_observed_bpb=nearest_bpb,
                    nearest_observed_phase_tv_distance=nearest_distance,
                    optimizer_status=status,
                )
            )
            write_weights(
                args.output_dir
                / "mixtures"
                / f"effexp_kl_{safe_float_label(float(kl_reg))}_phase_{safe_float_label(float(phase_reg))}.csv",
                domains,
                natural,
                weights,
                token_counts,
                target_budget,
            )
    summary = pd.DataFrame.from_records([asdict(row) for row in rows])
    summary.to_csv(args.output_dir / "phase_coupled_candidate_summary.csv", index=False)
    write_plot(args.output_dir / "phase_coupled_candidate_sweep.html", summary)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
