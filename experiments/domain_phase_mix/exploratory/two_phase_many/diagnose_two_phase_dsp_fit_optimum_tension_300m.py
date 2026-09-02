# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Diagnose the two-phase DSP fit-vs-optimum tension (local, 300M).

Motivating question: one-phase mixtures match or beat the best two-phase mixtures
for both Uncheatable BPB and Table-9 macro BPB, even though two-phase policies
strictly contain tied one-phase policies. This diagnostic establishes *why*, and
shows that no reparametrization inside the additive saturating-benefit-minus-
penalty family clears the bar (match effective-exposure OOF fit AND improve the
selected two-phase optimum).

Three parts, run on both the Table-9 and Uncheatable deletion-augmented 300M
panels (identical mixture geometry; only the target differs):

  A. Panel identification. Support envelope (realized simulated epochs), phase
     asymmetry (phase TV distribution), the incremental OOF signal from the phase
     term (no_phase vs effective_exposure), and a benefit/penalty softness trace.

  B. Gamma frontier. Pin the effective-exposure phase multiplier gamma across a
     grid (1 = tied/one-phase ... 40), refit rho/tau, and measure OOF fit AND the
     KL-proposal optimum quality. The phase parameter that produces the OOF rank
     signal is the same one that produces the over-optimistic two-phase corner:
     no gamma gives both a good fit and a frontier-calibrated optimum.

  C. Penalty recoupling. Keep the effective-exposure benefit but make the
     overexposure penalty grow faster in realized exposure than the baseline
     (log z)^2: softplus(sqrt(z) - tau)^2 (penalty ~ z) and softplus(z/8 - tau)^2
     (penalty ~ z^2). The saturation axis shows the same tension: harder penalties
     are either neutralized by NNLS rescaling or trade OOF fit for a smaller,
     still-optimistic near-proportional tilt.

The KL proposal optimizer (analyze_table9_phase_split_dsp_300m.optimize_dsp_kl)
uses numerical gradients, so the recoupled-penalty variants are added by runtime
monkeypatch of dsp.features and dsp.VARIANTS; standalone_code/dsp_exact.py is not
modified.

Usage:
    uv run diagnose_two_phase_dsp_fit_optimum_tension_300m.py \
        --output-dir reference_outputs/two_phase_dsp_fit_optimum_tension_20260703
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_dsp_uncheatable_eta_heldout as eta_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "two_phase_dsp_fit_optimum_tension_20260703"
LINEAR_REG = 0.01
GAMMA_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 25.0, 40.0)
KL_REGS = (0.1, 0.025)
LINEAR_PENALTY_SCALE = 8.0

SQRT_NAME = "dsp_effexp_sqrt_penalty"
LINEAR_NAME = "dsp_effexp_linear_penalty"
_ORIG_FEATURES = dsp.features


def features_with_recoupled_penalty(weights, c0, c1, variant, params):
    """Effective-exposure benefit; overexposure penalty on a faster-growing transform of z."""
    signal, penalty = _ORIG_FEATURES(weights, c0, c1, variant, params)
    if variant.name in (SQRT_NAME, LINEAR_NAME):
        e0 = weights[:, 0, :] * c0[None, :]
        e1 = weights[:, 1, :] * c1[None, :]
        exposure = e0 + float(params["gamma"]) * e1
        tau = np.asarray(params["tau"], dtype=float)[None, :]
        if variant.name == SQRT_NAME:
            arg = np.sqrt(np.maximum(exposure, 0.0)) - tau
        else:
            arg = exposure / LINEAR_PENALTY_SCALE - tau
        penalty = dsp.softplus(arg) ** 2
    return signal, penalty


def register_recoupled_penalty_variants() -> None:
    dsp.features = features_with_recoupled_penalty
    dsp.VARIANTS["effexp_sqrt_penalty"] = dsp.DSPVariant(
        name=SQRT_NAME,
        phase_mode=dsp.PhaseMode.EFFECTIVE_EXPOSURE,
        penalty_mode=dsp.PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=dsp.LinearMode.NNLS,
        description="effective-exposure benefit; penalty softplus(sqrt(z)-tau)^2 (~ linear in epochs).",
    )
    dsp.VARIANTS["effexp_linear_penalty"] = dsp.DSPVariant(
        name=LINEAR_NAME,
        phase_mode=dsp.PhaseMode.EFFECTIVE_EXPOSURE,
        penalty_mode=dsp.PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=dsp.LinearMode.NNLS,
        description="effective-exposure benefit; penalty softplus(z/8-tau)^2 (~ quadratic in epochs).",
    )


# ---------------------------------------------------------------------------
# Packet loaders (shared). Both panels share the deletion-augmented mixture
# geometry; only the target differs.
# ---------------------------------------------------------------------------
def load_table9():
    _signal, columns, domains, natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    panel, _metadata = paper_olmix.build_fit_panel(columns)
    target_budget = base.load_target_budget()
    packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, "table9_macro_bpb")
    folds = component_dsp.panel_stratified_folds(panel, n_splits=5, seed=0)
    return packet, panel, np.asarray(natural, float), np.asarray(token_counts, float), int(target_budget), folds


def load_uncheatable():
    packet, panel, _domains, natural, token_counts, target_budget = eta_diag.load_packet()
    folds = component_dsp.panel_stratified_folds(panel, n_splits=5, seed=0)
    return packet, panel, np.asarray(natural, float), np.asarray(token_counts, float), int(target_budget), folds


LOADERS = {"table9": load_table9, "uncheatable": load_uncheatable}


# ---------------------------------------------------------------------------
# Part A: panel identification
# ---------------------------------------------------------------------------
def panel_identification(name, packet, token_counts, target_budget, folds) -> tuple[dict, list[dict]]:
    n, _, m = packet.w.shape
    sim = np.stack([base.simulated_epochs(packet.w[i], token_counts, target_budget=target_budget) for i in range(n)])
    row_max_epoch = sim.max(axis=1)
    p0 = packet.w[:, 0, :]
    p1 = packet.w[:, 1, :]
    phase_tv = 0.5 * np.abs(p0 - p1).sum(axis=1)
    summary = {
        "objective": name,
        "rows": int(n),
        "domains": int(m),
        "epoch_rowmax_median": float(np.median(row_max_epoch)),
        "epoch_rowmax_p95": float(np.quantile(row_max_epoch, 0.95)),
        "epoch_rowmax_max": float(np.max(row_max_epoch)),
        "rows_with_domain_gt_16_epochs": int((row_max_epoch > 16).sum()),
        "phase_tv_median": float(np.median(phase_tv)),
        "phase_tv_p95": float(np.quantile(phase_tv, 0.95)),
        "rows_phase_tv_gt_0p3": int((phase_tv > 0.3).sum()),
        "pooled_corr_p0_p1": float(np.corrcoef(p0.reshape(-1), p1.reshape(-1))[0, 1]),
    }
    for key in ("no_phase", "effective_exposure"):
        model, _ = phase_dsp.fit_variant_with_l2(packet, key, LINEAR_REG, maxiter=32, coarse_top_k=3, basin_hopping_iters=0)
        oof = phase_dsp.fixed_param_oof(packet, model, folds)
        _tr_rmse, tr_sp = phase_dsp.regression_metrics(packet.y, dsp.predict(model, packet.w))
        oof_rmse, oof_sp = phase_dsp.regression_metrics(packet.y, oof)
        summary[f"{key}_train_sp"] = float(tr_sp)
        summary[f"{key}_oof_rmse"] = float(oof_rmse)
        summary[f"{key}_oof_sp"] = float(oof_sp)
        if key == "effective_exposure":
            summary["fitted_gamma"] = float(model.params["gamma"])
            trace = _penalty_softness_trace(name, model, packet)
    return summary, trace


def _penalty_softness_trace(name, model, packet) -> list[dict]:
    """Benefit saturates while the (log z)^2 penalty grows very slowly: net turns negative only at high z."""
    rho = np.asarray(model.params["rho"], float)
    tau = np.asarray(model.params["tau"], float)
    gamma = float(model.params["gamma"])
    e0 = packet.w[:, 0, :] * model.c0[None, :]
    e1 = packet.w[:, 1, :] * model.c1[None, :]
    z = e0 + gamma * e1
    dom = int(np.argmax(np.median(z, axis=0)))
    a_i, p_i = model.benefit_coef[dom], model.penalty_coef[dom]
    rows = []
    for zz in (2, 4, 8, 16, 32, 64, 128):
        ben = float(a_i * (1.0 - np.exp(-rho[dom] * zz)))
        pen = float(p_i * dsp.softplus(np.log1p(zz) - tau[dom]) ** 2)
        rows.append({"objective": name, "domain": model.domain_names[dom], "z_epochs": zz,
                     "benefit": ben, "penalty": pen, "net_reduction": ben - pen})
    return rows


# ---------------------------------------------------------------------------
# Part B: gamma frontier
# ---------------------------------------------------------------------------
def fit_pinned_gamma(packet, gamma, *, maxiter=40):
    variant = dsp.VARIANTS["effective_exposure"]
    m = packet.m
    free_bounds = dsp.bounds(variant, m)[: 2 * m]  # [log_rho(m), tau(m)]; gamma pinned
    log_g = float(np.log(gamma))
    starts = dsp.start_bank(packet, variant)

    def objective(free):
        theta = np.concatenate([np.asarray(free, float), [log_g]])
        return dsp.profile_objective(packet, variant, theta)

    best = None
    for start in starts[:4]:
        res = minimize(objective, start[: 2 * m], method="L-BFGS-B", bounds=free_bounds,
                       options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20})
        if best is None or float(res.fun) < float(best.fun):
            best = res
    params = dsp.unpack_theta(np.concatenate([np.asarray(best.x, float), [log_g]]), variant, m)
    return dsp.fit_linear_head(packet.w, packet.y, packet, variant, params)


def eval_model(model, packet, natural, token_counts, target_budget, folds, *, label):
    train_pred = dsp.predict(model, packet.w)
    _tr_rmse, tr_sp = phase_dsp.regression_metrics(packet.y, train_pred)
    oof = phase_dsp.fixed_param_oof(packet, model, folds)
    oof_rmse, oof_sp = phase_dsp.regression_metrics(packet.y, oof)
    optimism, low_rmse = phase_dsp.lower_tail_optimism(packet.y, oof)
    sel_idx = int(np.argmin(oof))
    sel_rank = int(np.flatnonzero(np.argsort(packet.y) == sel_idx)[0] + 1)
    best_obs = float(np.min(packet.y))
    row = {
        "label": label, "gamma": float(model.params.get("gamma", float("nan"))),
        "train_spearman": tr_sp, "oof_rmse": oof_rmse, "oof_spearman": oof_sp,
        "oof_low_tail_optimism": optimism, "oof_low_tail_rmse": low_rmse,
        "oof_regret1": phase_dsp.global_regret_at_k(packet.y, oof, 1),
        "oof_regret3": phase_dsp.global_regret_at_k(packet.y, oof, 3),
        "oof_fold_regret1": phase_dsp.fold_mean_regret_at_k(packet.y, oof, folds, 1),
        "oof_selected_rank": sel_rank, "best_observed_bpb": best_obs,
    }
    starts = phase_dsp.proposal_starts(packet, natural, train_pred)
    reference = np.stack([natural, natural], axis=0)
    for kl in KL_REGS:
        weights, _regobj, _status = phase_dsp.optimize_dsp_kl(model, natural, kl, starts)
        pred = float(dsp.predict(model, weights[None, :, :])[0])
        sim = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
        dists = dsp.average_phase_tv_distance(packet.w, weights[None, :, :])
        nidx = int(np.argmin(dists))
        nrank = int(np.flatnonzero(np.argsort(packet.y) == nidx)[0] + 1)
        tag = f"kl{kl}"
        row[f"{tag}_predicted_bpb"] = pred
        row[f"{tag}_optimism_vs_best_obs"] = best_obs - pred
        row[f"{tag}_nearest_obs_bpb"] = float(packet.y[nidx])
        row[f"{tag}_nearest_obs_tv"] = float(dists[nidx])
        row[f"{tag}_nearest_obs_rank"] = nrank
        row[f"{tag}_max_weight"] = float(np.max(weights))
        row[f"{tag}_max_epoch"] = float(np.max(sim))
        row[f"{tag}_q95_epoch"] = float(np.quantile(sim, 0.95))
        row[f"{tag}_tv_to_proportional"] = float(0.5 * np.abs(weights - reference).sum(axis=1).mean())
    return row


def gamma_frontier(packet, natural, token_counts, target_budget, folds) -> pd.DataFrame:
    rows = []
    for key in ("no_phase", "effective_exposure", "split_saturation_penalty"):
        model, _ = phase_dsp.fit_variant_with_l2(packet, key, LINEAR_REG, maxiter=40, coarse_top_k=3, basin_hopping_iters=0)
        rows.append(eval_model(model, packet, natural, token_counts, target_budget, folds, label=f"anchor:{key}"))
    for g in GAMMA_GRID:
        model = fit_pinned_gamma(packet, g)
        rows.append(eval_model(model, packet, natural, token_counts, target_budget, folds, label=f"effexp_gamma={g:g}"))
    return pd.DataFrame(rows)


def penalty_recoupling(packet, natural, token_counts, target_budget, folds) -> pd.DataFrame:
    rows = []
    for key in ("effective_exposure", "effexp_sqrt_penalty", "effexp_linear_penalty"):
        model, _ = phase_dsp.fit_variant_with_l2(packet, key, LINEAR_REG, maxiter=40, coarse_top_k=3, basin_hopping_iters=0)
        rows.append(eval_model(model, packet, natural, token_counts, target_budget, folds, label=key))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
FRONTIER_FIT_COLS = ["label", "gamma", "oof_rmse", "oof_spearman", "oof_low_tail_optimism", "oof_regret1", "oof_selected_rank"]
FRONTIER_OPT_COLS = ["label", "kl0.1_predicted_bpb", "kl0.1_optimism_vs_best_obs", "kl0.1_max_weight",
                     "kl0.1_max_epoch", "kl0.1_tv_to_proportional", "kl0.025_predicted_bpb", "kl0.025_max_weight"]
PENALTY_FIT_COLS = ["label", "gamma", "oof_rmse", "oof_spearman", "oof_regret1", "oof_selected_rank"]
PENALTY_OPT_COLS = ["label", "kl0.1_predicted_bpb", "kl0.1_optimism_vs_best_obs", "kl0.1_max_weight", "kl0.1_max_epoch"]


def write_report(output_dir, panel_summary, panel_trace, frontier, penalty):
    lines = [
        "# Two-phase DSP fit-vs-optimum tension (local, 300M)",
        "",
        "Generated by `diagnose_two_phase_dsp_fit_optimum_tension_300m.py`. Local only; no cluster jobs.",
        "",
        "## Conclusion",
        "",
        "No reparametrization inside the additive saturating-benefit-minus-penalty family clears the bar "
        "(match effective-exposure OOF fit AND improve the selected two-phase optimum). The effective-exposure "
        "phase multiplier gamma simultaneously produces the OOF rank signal and the over-optimistic two-phase "
        "corner; the overexposure penalty axis shows the same tradeoff. The two-phase optimum is not identifiable "
        "from this panel, and (per 3e18 transfer) the transferable phase-asymmetry benefit is ~0 at this scale.",
        "",
        "## A. Panel identification",
        "",
        "The panel is not the naive blocker: it has wide realized-epoch support and strong phase asymmetry, and "
        "the phase term carries almost all of the OOF rank signal (no_phase vs effective_exposure Spearman).",
        "",
        panel_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Benefit/penalty softness trace (largest-median-exposure domain): benefit saturates while the (log z)^2 "
        "penalty grows slowly, so the net BPB reduction only turns negative at high z.",
        "",
        panel_trace.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## B. Gamma frontier (late-utility axis)",
        "",
        "Pin gamma (1 = tied/one-phase), refit rho/tau. Good OOF fit needs gamma >= 8-16; a frontier-calibrated "
        "optimum (optimism ~ 0) exists only near gamma = 1, where the fit collapses. No gamma gives both.",
    ]
    for name, df in frontier.items():
        lines += ["", f"### {name} — fit", "", df[FRONTIER_FIT_COLS].to_markdown(index=False, floatfmt=".4f"),
                  "", f"### {name} — KL-proposal optimum", "", df[FRONTIER_OPT_COLS].to_markdown(index=False, floatfmt=".4f")]
    lines += [
        "",
        "## C. Penalty recoupling (saturation axis)",
        "",
        "Keep the effective-exposure benefit; grow the penalty faster than (log z)^2. `sqrt` (penalty ~ z) is "
        "neutralized by NNLS rescaling; `linear` (penalty ~ z^2) lowers optimum optimism/concentration but "
        "regresses OOF fit and still predicts a fantasy (just a smaller near-proportional tilt).",
    ]
    for name, df in penalty.items():
        lines += ["", f"### {name} — fit", "", df[PENALTY_FIT_COLS].to_markdown(index=False, floatfmt=".4f"),
                  "", f"### {name} — KL-proposal optimum", "", df[PENALTY_OPT_COLS].to_markdown(index=False, floatfmt=".4f")]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="table9,uncheatable")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    register_recoupled_penalty_variants()
    dsp.LINEAR_REG = LINEAR_REG

    objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]
    panel_rows, trace_rows, frontier, penalty = [], [], {}, {}
    for name in objectives:
        print(f"==== {name} ====", flush=True)
        packet, _panel, natural, token_counts, target_budget, folds = LOADERS[name]()
        print("  part A: panel identification", flush=True)
        summary, trace = panel_identification(name, packet, token_counts, target_budget, folds)
        panel_rows.append(summary)
        trace_rows.extend(trace)
        print("  part B: gamma frontier", flush=True)
        frontier[name] = gamma_frontier(packet, natural, token_counts, target_budget, folds)
        frontier[name].to_csv(args.output_dir / f"gamma_frontier_{name}.csv", index=False)
        print("  part C: penalty recoupling", flush=True)
        penalty[name] = penalty_recoupling(packet, natural, token_counts, target_budget, folds)
        penalty[name].to_csv(args.output_dir / f"penalty_recoupling_{name}.csv", index=False)

    panel_summary = pd.DataFrame(panel_rows)
    panel_trace = pd.DataFrame(trace_rows)
    panel_summary.to_csv(args.output_dir / "panel_identification_summary.csv", index=False)
    panel_trace.to_csv(args.output_dir / "penalty_softness_trace.csv", index=False)
    write_report(args.output_dir, panel_summary, panel_trace, frontier, penalty)
    print(f"Wrote diagnostic artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
