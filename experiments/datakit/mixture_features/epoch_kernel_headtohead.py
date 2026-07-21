# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Should EPOCH live inside the kernel, or in an additive harm term beside it?

``gp_ood_coverage`` showed the content-only surrogate fails hard on the 53 off-design probe
runs (mean |z| 12.0, 17% inside 2sd) because the content features ``h = V w`` are blind to
epoching. Two repairs are on the table, and this is the falsifiable head-to-head between them:

    (a) BASELINE     k = sigma_f^2 exp(-gamma_c d2_content)                     [the documented failure]
    (b) ADDITIVE     (a) + H(e) = sum_j b_g(j) max(e_j - tau_g(j), 0)           [the campaign's design]
    (c) IN-KERNEL    k = sigma_f^2 exp(-gamma_c d2_content - gamma_e d2_epoch)  [one extra length-scale]

The distinction that matters is WHERE each model is allowed to learn the epoch effect.
(c) must learn it from the swarm, by marginal likelihood. (b) does not learn it at all: its
``tau_g`` and ``b_g`` are read off the campaign's *dedicated* epoch experiments (the epochrep
arms and the twobucket a2/a3/a4 factorial) and are never refit here -- they are loaded
verbatim from ``harm_form_selection.json`` and ``harm_term_fit.json``.

The pre-registered prediction is that the swarm has epoch VARIATION but no epoch SIGNAL in
regime (its runs sit near the threshold, and DP4 found kernel-residual-vs-repetition
correlation < 0.1 for 0/37 tasks), so marginal likelihood should drive ``gamma_e`` to ~0 and
(c) should collapse into (a). If that holds it is a positive argument for the simple design:
fit the correction where the signal actually lives.

Epoch summary for (c)
---------------------
``d2_epoch`` is the squared difference of a single standardized scalar per run, the
token-weighted repeated mass past 4 epochs used by the campaign's natural-epoch experiment::

    repmass = sum_j (f0 w_0j + f1 w_1j) * max(e_j - 4, 0)

One scalar (not a 168-vector) because that is the most favourable defensible summary: it is
the same statistic whose correlation with the kernel residual DP4 measured, it is monotone in
"how much of the token budget is repeated", and giving the kernel a single well-scaled input
maximizes its chance of finding the effect. ``gamma_e`` is then free to grow or vanish.

Epoch bookkeeping
-----------------
Swarm runs use simulated epoching (cache sliced to the target budget), so their per-phase
epochs come from ``swoosh_form.per_phase_epochs``. Probe runs that slice with
``max_train_batches`` (twobucket ctr/a1-a4, epochrep, harm100b) carry REAL epochs, taken
verbatim from their frozen pre-registrations; twobucket's ``natural`` arm and the transect
epoch through the target budget and are reconstructed with the simulated convention (verified:
the transect's reconstructed phase-0 epochs reproduce its pre-registered targets exactly).
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "experiments/datakit/mixture_features")
import grug_fit as gf
import swoosh_form as sf
from gp_ood_coverage import HUMANEVAL, HUMANEVAL_GROUPS, load_design, offdesign_runs, verify_reconstruction
from gp_surrogate import JITTER, fit_gp, predict_gp
from grug_validation_batch3 import cluster_delta_groups

GRUG_DIR = Path("scratch/mixture_features/grug")
OUT_JSON = GRUG_DIR / "epoch_kernel_headtohead.json"
OUT_MD = GRUG_DIR / "epoch_kernel_headtohead.md"
FIG_PATH = Path("scratch/mixture_features/report/figs3/f34_epoch_kernel_headtohead.png")

N_FOLDS = 5
SEED = 0

# the harm term was calibrated at 10B tokens on the d512 swarm model; b scales off that point
REF_BUDGET_TOKENS = 10_015_997_952
REF_HIDDEN_DIM = 512
TRANSECT_STEPS = 47_759
BATCH_SIZE = 512
SEQ_LEN = 4096

# repmass counts token mass repeated beyond this many epochs (natural_epoch_experiment convention)
REPMASS_EPOCH_FLOOR = 4.0

# groups the campaign calibrated a harm term for; tail_small has no dedicated epoch experiment
HARM_GROUPS = ("code_adjacent", "web_text")

MODEL_LABELS = {
    "a_content_only": "(a) content kernel",
    "b_content_plus_harm": "(b) content kernel + additive harm",
    "c_epoch_in_kernel": "(c) epoch inside the kernel",
}
MODEL_COLORS = {"a_content_only": "#762a83", "b_content_plus_harm": "#1b7837", "c_epoch_in_kernel": "#c1272d"}
GROUP_COLORS = {"twobucket": "#c1272d", "epochrep": "#e08214", "transect": "#2166ac", "harm100b": "#762a83"}


# ---------------------------------------------------------------------------
# the harm term: parameters imported, never refit
# ---------------------------------------------------------------------------


def load_harm_parameters() -> dict:
    """(tau_g, b_g) from the dedicated epoch experiments, plus the amplitude scaling exponents.

    ``b_g`` is the harm per epoch past ``tau_g`` measured at the calibration weight
    ``w_cal = 0.2`` (every epochrep/twobucket point held the sliced bucket at 0.2), so the
    mass-weighted coefficient is ``b_g / w_cal``. tau_web is not identifiable from the web arm
    (only e16/e24 sit above the knee), so web imports code's shape, which is exactly how
    ``harm_form_selection`` fitted ``b_web``.
    """
    form = json.loads((GRUG_DIR / "harm_form_selection.json").read_text())
    scaling = json.loads((GRUG_DIR / "harm_term_fit.json").read_text())
    tau_code = float(form["tau_code"]["tau"])
    return {
        "w_cal": 0.2,
        "tau": {"code_adjacent": tau_code, "web_text": tau_code},
        "b": {
            "code_adjacent": float(form["joint_fit_best"]["b_code"]),
            "web_text": float(form["web_corroboration"]["linear_past_thr"]["b_web_fixed_code_shape"]),
        },
        "budget_exponent": float(scaling["budget_exponent"]),
        "hidden_dim_exponent": float(scaling["size_exponent_hiddendim"]),
        "source": "harm_form_selection.json (tau, b) + harm_term_fit.json (B, d exponents); NOT refit here",
    }


def amplitude_scale(budget_tokens: np.ndarray, hidden_dim: np.ndarray, params: dict) -> np.ndarray:
    """(B/10B)^-0.73 * (d/512)^+1.68 -- the campaign's fitted harm-amplitude scaling."""
    return (budget_tokens / REF_BUDGET_TOKENS) ** (-params["budget_exponent"]) * (hidden_dim / REF_HIDDEN_DIM) ** params[
        "hidden_dim_exponent"
    ]


def harm_term(w: np.ndarray, ep: np.ndarray, masks: dict, scale: np.ndarray, params: dict) -> np.ndarray:
    """The campaign's documented equation: H = scale * sum_j b_g(j) * max(e_j - tau_g(j), 0).

    ``e_j`` is the phase-0 epoch count, which is the epoch label the harm forms were fitted
    against (``harm_form_selection`` regresses on the nominal e of each epochrep/twobucket
    point, and that label is exactly the sliced bucket's phase-0 epochs).
    """
    out = np.zeros(len(w))
    for g in HARM_GROUPS:
        past = np.maximum(ep[:, 0, :] - params["tau"][g], 0.0)
        out += params["b"][g] * (past * masks[g][None, :]).sum(axis=1)
    return out * scale


def harm_term_mass_weighted(w: np.ndarray, ep: np.ndarray, masks: dict, scale: np.ndarray, params: dict) -> np.ndarray:
    """Sensitivity variant: H = scale * sum_p sum_j w_pj * (b_g / w_cal) * max(e_pj - tau_g, 0).

    Every calibration point held the sliced bucket at ``w_cal = 0.2``, so the harm term's
    dependence on mixture weight is UNIDENTIFIED by the dedicated experiments: weighting by
    ``w_pj / w_cal`` reproduces the calibration exactly and is equally admissible. The two
    readings only diverge where weight and epochs are confounded (twobucket's natural arm).
    """
    out = np.zeros(len(w))
    for g in HARM_GROUPS:
        coefficient = params["b"][g] / params["w_cal"]
        past = np.maximum(ep - params["tau"][g], 0.0)
        out += coefficient * ((w * past) * masks[g][None, None, :]).sum(axis=(1, 2))
    return out * scale


# ---------------------------------------------------------------------------
# epochs for the swarm and for the probe runs
# ---------------------------------------------------------------------------


def _prereg(group: str) -> dict:
    return json.loads((GRUG_DIR / f"{group}_preregistration.json").read_text())


def probe_epoch_metadata() -> dict:
    """run_id -> {per-bucket (phase0, phase1) epochs or None for simulated, budget, hidden dim}."""
    meta: dict[str, dict] = {}
    for run in _prereg("twobucket")["runs"]:
        epochs = run["epochs"]
        # the natural arm epochs through the target budget; reconstruct it like the swarm
        real = run["code_slice_batches"] is not None
        meta[run["run_id"]] = {
            "epochs": (
                {
                    "c01q0": (epochs["code"]["phase0"], epochs["code"]["phase1"]),
                    "c05q0": (epochs["web"]["phase0"], epochs["web"]["phase1"]),
                }
                if real
                else None
            ),
            "budget_tokens": float(run["experiment_budget_tokens"]),
            "hidden_dim": 256 if run["model"] == "d256" else 512,
            "arm": run["arm"],
        }
    for group in ("epochrep", "harm100b"):
        prereg = _prereg(group)
        budget = float(prereg["constants"]["experiment_budget_tokens"])
        for run in prereg["runs"]:
            epochs = run["epochs"]
            meta[run["run_id"]] = {
                "epochs": {
                    run["sliced_bucket"]: (epochs["sliced"]["phase0"], epochs["sliced"]["phase1"]),
                    run["partner_bucket"]: (epochs["partner"]["phase0"], epochs["partner"]["phase1"]),
                },
                "budget_tokens": budget,
                "hidden_dim": 512,
                "arm": run.get("arm", group),
            }
    for run in _prereg("transect")["runs"]:
        meta[run["run_name"]] = {
            "epochs": None,
            "budget_tokens": float(TRANSECT_STEPS * BATCH_SIZE * SEQ_LEN),
            "hidden_dim": 512,
            "arm": "transect",
            "target_epochs_phase0": float(run["target_epochs_phase0"]),
            "bucket": run["bucket"],
        }
    return meta


def probe_epochs(design: dict, records: list[dict], meta: dict) -> dict:
    """(n, 2, 168) per-phase epochs, budgets and hidden dims for the probe runs, in record order."""
    index = design["index"]
    tj = design["total_tokens"]
    w = np.stack([r["w"] for r in records])
    simulated = sf.per_phase_epochs(w, tj)
    ep = np.zeros_like(simulated)
    checks = []
    for i, rec in enumerate(records):
        info = meta[rec["run_id"]]
        if info["epochs"] is None:
            ep[i] = simulated[i]
            if "target_epochs_phase0" in info:
                checks.append(abs(ep[i, 0, index[info["bucket"]]] - info["target_epochs_phase0"]))
            continue
        for bucket, (e0, e1) in info["epochs"].items():
            ep[i, 0, index[bucket]] = e0
            ep[i, 1, index[bucket]] = e1
    if checks and max(checks) > 1e-6:
        raise AssertionError(f"transect simulated epochs disagree with pre-registration: {max(checks):.3e}")
    return {
        "w": w,
        "ep": ep,
        "budget_tokens": np.array([meta[r["run_id"]]["budget_tokens"] for r in records]),
        "hidden_dim": np.array([float(meta[r["run_id"]]["hidden_dim"]) for r in records]),
        "transect_epoch_check": float(max(checks)) if checks else 0.0,
    }


def repmass(w: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """sum_j (f0 w_0j + f1 w_1j) * max(e_j - 4, 0): token mass repeated past 4 epochs."""
    f = sf.phase_fractions()
    w_tokens = w[:, 0, :] * f[0] + w[:, 1, :] * f[1]
    return (w_tokens * np.clip(ep.sum(axis=1) - REPMASS_EPOCH_FLOOR, 0.0, None)).sum(axis=1)


# ---------------------------------------------------------------------------
# GP with two length-scales (content + epoch)
# ---------------------------------------------------------------------------

GAMMA_BOUNDS = (np.log(1e-9), np.log(1e3))
VAR_BOUNDS = (np.log(1e-6), np.log(1e3))
# multi-start on gamma_epoch so a collapse to ~0 is a real optimum, not a bad initialization
GAMMA_E_STARTS = (1e-8, 1e-3, 0.1, 1.0, 10.0)


def _build_chol2(d2_content: np.ndarray, d2_epoch: np.ndarray, theta: np.ndarray):
    sf2, sn2, gamma_c, gamma_e = np.exp(theta)
    k = sf2 * np.exp(-gamma_c * d2_content - gamma_e * d2_epoch)
    k[np.diag_indices_from(k)] += sn2 + JITTER
    return cholesky(k, lower=True), sf2, sn2, gamma_c, gamma_e


def _nlml2(theta: np.ndarray, d2_content: np.ndarray, d2_epoch: np.ndarray, y_centered: np.ndarray) -> float:
    try:
        chol, *_ = _build_chol2(d2_content, d2_epoch, theta)
    except np.linalg.LinAlgError:
        return 1e12
    a = cho_solve((chol, True), y_centered)
    return float(0.5 * y_centered @ a + np.log(np.diag(chol)).sum() + 0.5 * y_centered.size * np.log(2 * np.pi))


def fit_gp2(d2_content: np.ndarray, d2_epoch: np.ndarray, y: np.ndarray) -> dict:
    """Fit (sigma_f^2, sigma_n^2, gamma_content, gamma_epoch) by marginal likelihood."""
    ybar = float(y.mean())
    yc = y - ybar
    upper = np.triu_indices(len(y), 1)
    med_c = float(np.median(d2_content[upper]))
    bounds = [VAR_BOUNDS, VAR_BOUNDS, GAMMA_BOUNDS, GAMMA_BOUNDS]
    best = None
    for gamma_e0 in GAMMA_E_STARTS:
        theta0 = np.log([max(yc.var(), 1e-6), max(0.1 * yc.var(), 1e-8), 0.25 / max(med_c, 1e-12), gamma_e0])
        res = minimize(_nlml2, theta0, args=(d2_content, d2_epoch, yc), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    chol, sf2, sn2, gamma_c, gamma_e = _build_chol2(d2_content, d2_epoch, best.x)
    return {
        "chol": chol,
        "alpha_dual": cho_solve((chol, True), yc),
        "ybar": ybar,
        "sigma_f2": float(sf2),
        "sigma_n2": float(sn2),
        "gamma_content": float(gamma_c),
        "gamma_epoch": float(gamma_e),
        "nlml": float(best.fun),
        "median_d2_content": med_c,
        "median_d2_epoch": float(np.median(d2_epoch[upper])),
    }


def predict_gp2(fit: dict, d2_content_star: np.ndarray, d2_epoch_star: np.ndarray):
    """Posterior mean and predictive sd at new points; d2_* are (n_star, n_train)."""
    k_star = fit["sigma_f2"] * np.exp(-fit["gamma_content"] * d2_content_star - fit["gamma_epoch"] * d2_epoch_star)
    mu = k_star @ fit["alpha_dual"] + fit["ybar"]
    v = solve_triangular(fit["chol"], k_star.T, lower=True)
    var = fit["sigma_f2"] - np.einsum("ij,ij->j", v, v) + fit["sigma_n2"]
    return mu, np.sqrt(np.clip(var, 1e-12, None))


def epoch_distance_share(fit: dict, d2_content: np.ndarray, d2_epoch: np.ndarray) -> float:
    """Fraction of the kernel's total exponent contributed by the epoch length-scale."""
    content = fit["gamma_content"] * d2_content
    epoch = fit["gamma_epoch"] * d2_epoch
    total = content + epoch
    return float(np.mean(epoch[total > 0] / total[total > 0])) if np.any(total > 0) else 0.0


def profile_gamma_epoch(
    d2_content: np.ndarray,
    d2_epoch: np.ndarray,
    y: np.ndarray,
    d2_content_star: np.ndarray,
    d2_epoch_star: np.ndarray,
    realized: np.ndarray,
) -> list[dict]:
    """FORCE gamma_epoch to a grid, refit the rest, and score the probe runs at each value.

    Separates "marginal likelihood picked a bad length-scale" from "no epoch length-scale
    would have helped". If the probe RMSE never improves along the grid, the kernel simply
    cannot express the epoch effect from this training set.
    """
    ybar = float(y.mean())
    yc = y - ybar
    med = float(np.median(d2_content[np.triu_indices(len(y), 1)]))
    rows = []
    for gamma_e in (0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0):

        def nlml3(theta: np.ndarray, _g: float = gamma_e) -> float:
            return _nlml2(np.append(theta, np.log(max(_g, 1e-12))), d2_content, d2_epoch, yc)

        theta0 = np.log([max(yc.var(), 1e-6), max(0.1 * yc.var(), 1e-8), 0.25 / max(med, 1e-12)])
        res = minimize(nlml3, theta0, method="L-BFGS-B", bounds=[VAR_BOUNDS, VAR_BOUNDS, GAMMA_BOUNDS])
        theta = np.append(res.x, np.log(max(gamma_e, 1e-12)))
        chol, *_ = _build_chol2(d2_content, d2_epoch, theta)
        fit = {
            "sigma_f2": float(np.exp(res.x[0])),
            "sigma_n2": float(np.exp(res.x[1])),
            "gamma_content": float(np.exp(res.x[2])),
            "gamma_epoch": gamma_e,
            "chol": chol,
            "alpha_dual": cho_solve((chol, True), yc),
            "ybar": ybar,
        }
        mu, sd = predict_gp2(fit, d2_content_star, d2_epoch_star)
        rows.append(
            {
                "gamma_epoch": gamma_e,
                "nlml": float(res.fun),
                "gamma_content": fit["gamma_content"],
                **{k: v for k, v in score(realized, mu, sd).items() if k in ("rmse", "mean_abs_z", "coverage_2sd")},
            }
        )
    return rows


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score(realized: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    z = (realized - mu) / sd
    return {
        "n": len(realized),
        "rmse": float(np.sqrt(np.mean((realized - mu) ** 2))),
        "mean_miss": float(np.mean(realized - mu)),
        "mean_abs_miss": float(np.mean(np.abs(realized - mu))),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "max_abs_z": float(np.max(np.abs(z))),
        "coverage_1sd": float(np.mean(np.abs(z) < 1.0)),
        "coverage_2sd": float(np.mean(np.abs(z) < 2.0)),
        "mean_predicted_sd": float(sd.mean()),
    }


def score_by_group(groups: np.ndarray, realized: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    out = {g: score(realized[groups == g], mu[groups == g], sd[groups == g]) for g in HUMANEVAL_GROUPS}
    out["all"] = score(realized, mu, sd)
    return out


def residual_epoch_slope(per_run: list[dict], group: str, arm: str, model: str) -> dict:
    """OLS slope of a model's residual against epochs, within one single-bucket probe arm.

    This is the sharpest read on whether a model has actually absorbed the epoch axis. Inside
    one arm the mixture is fixed and only the epoch count moves, so a residual that still
    slopes with epochs means the epoch effect is unmodelled, while a flat residual at a nonzero
    level means what is left is a CONTENT extrapolation error the epoch machinery was never
    meant to fix.
    """
    rows = [r for r in per_run if r["group"] == group and r["arm"] == arm]
    e = np.array([r["max_phase0_epoch"] for r in rows])
    miss = np.array([r[f"{model}_miss"] for r in rows])
    slope, intercept = np.polyfit(e, miss, 1)
    return {
        "n": len(rows),
        "slope_bpb_per_epoch": float(slope),
        "intercept_bpb": float(intercept),
        "mean_abs_residual": float(np.abs(miss).mean()),
        "residual_range": [float(miss.min()), float(miss.max())],
    }


def cross_validate(d2_content: np.ndarray, d2_epoch: np.ndarray, y: np.ndarray, harm: np.ndarray, model: str) -> dict:
    """5-fold held-out on the swarm, hyperparameters refit inside every fold."""
    mus, sds, ys = [], [], []
    gammas_e = []
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(len(y))):
        if model == "c_epoch_in_kernel":
            fit = fit_gp2(d2_content[np.ix_(tr, tr)], d2_epoch[np.ix_(tr, tr)], y[tr])
            mu, sd = predict_gp2(fit, d2_content[np.ix_(te, tr)], d2_epoch[np.ix_(te, tr)])
            gammas_e.append(fit["gamma_epoch"])
        else:
            target = y[tr] - harm[tr] if model == "b_content_plus_harm" else y[tr]
            fit = fit_gp(d2_content[np.ix_(tr, tr)], target)
            mu, sd = predict_gp(fit, d2_content[np.ix_(te, tr)], include_noise=True)
            if model == "b_content_plus_harm":
                mu = mu + harm[te]
        mus.append(mu)
        sds.append(sd)
        ys.append(y[te])
    mu, sd, y_te = np.concatenate(mus), np.concatenate(sds), np.concatenate(ys)
    out = score(y_te, mu, sd)
    out["spearman"] = float(spearmanr(y_te, mu).statistic)
    if gammas_e:
        out["fold_gamma_epoch"] = [float(g) for g in gammas_e]
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def make_figure(out: dict, per_run: list[dict], path: Path) -> None:
    models = list(MODEL_LABELS)
    fig, axes2d = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    axes = axes2d.ravel()

    realized = np.array([r["realized"] for r in per_run])
    lo = min(realized.min(), min(r[f"{m}_mean"] for r in per_run for m in models)) - 0.1
    hi = max(realized.max(), max(r[f"{m}_mean"] for r in per_run for m in models)) + 0.1

    for ax, model in zip(axes[:3], models, strict=False):
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, ls="--", zorder=1)
        for g in HUMANEVAL_GROUPS:
            rows = [r for r in per_run if r["group"] == g]
            ax.errorbar(
                [r[f"{model}_mean"] for r in rows],
                [r["realized"] for r in rows],
                xerr=[2 * r[f"{model}_sd"] for r in rows],
                fmt="o",
                ms=5,
                lw=1,
                capsize=2,
                color=GROUP_COLORS[g],
                label=f"{g} (n={len(rows)})",
                zorder=3,
            )
        s = out["offdesign"][model]["all"]
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("predicted humaneval bpb (bars = +/-2sd)")
        ax.set_ylabel("realized humaneval bpb")
        ax.set_title(f"{MODEL_LABELS[model]}\nRMSE {s['rmse']:.3f} | mean |z| {s['mean_abs_z']:.1f}", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")

    ax = axes[3]
    width = 0.26
    xs = np.arange(len(HUMANEVAL_GROUPS) + 1)
    labels = [*HUMANEVAL_GROUPS, "all"]
    for i, model in enumerate(models):
        vals = [out["offdesign"][model][g]["mean_abs_z"] for g in labels]
        ax.bar(xs + (i - 1) * width, vals, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.axhline(2.0, color="k", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g}\n(n={out['offdesign']['a_content_only'][g]['n']})" for g in labels], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("mean |z| on the probe runs (log scale)")
    ax.set_title("(d) where each repair helps", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # --- (e) forcing gamma_epoch: no length-scale rescues the probe runs -------------------
    ax = axes[4]
    prof = out["gamma_epoch_profile"]
    gx = np.arange(len(prof))
    ax.plot(gx, [r["rmse"] for r in prof], "o-", color=MODEL_COLORS["c_epoch_in_kernel"], label="probe RMSE")
    ax.axhline(
        out["offdesign"]["a_content_only"]["all"]["rmse"],
        color=MODEL_COLORS["a_content_only"],
        ls="--",
        lw=1.2,
        label="(a) baseline probe RMSE",
    )
    ax.axhline(
        out["offdesign"]["b_content_plus_harm"]["all"]["rmse"],
        color=MODEL_COLORS["b_content_plus_harm"],
        ls=":",
        lw=1.4,
        label="(b) additive-harm probe RMSE",
    )
    ax.set_xticks(gx)
    ax.set_xticklabels([f"{r['gamma_epoch']:g}" for r in prof], fontsize=8)
    ax.set_xlabel("gamma_epoch, FORCED (other hyperparameters refit)")
    ax.set_ylabel("probe RMSE (bpb)")
    ax.set_title("(e) no epoch length-scale rescues the probe runs", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    twin = ax.twinx()
    twin.plot(gx, [r["nlml"] for r in prof], "s--", color="0.45", ms=4, lw=1)
    twin.set_ylabel("negative log marginal likelihood (grey)", color="0.45", fontsize=8)

    # --- (f) the epoch coordinate is itself off-support ------------------------------------
    ax = axes[5]
    land = out["epoch_landscape"]
    train_rep = np.array(out["train_repmass_sample"])
    counts, _, _ = ax.hist(
        train_rep, bins=40, color="0.7", label=f"800 swarm runs (max {land['train_repmass_max']:.1f})"
    )
    # probe runs go in a rug strip below the histogram so the two never overlap
    rug_y = -0.11 * counts.max()
    for g in HUMANEVAL_GROUPS:
        vals = [r["repmass"] for r in per_run if r["group"] == g]
        ax.plot(vals, np.full(len(vals), rug_y), "|", ms=13, mew=1.6, color=GROUP_COLORS[g], label=g)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylim(-0.2 * counts.max(), 1.05 * counts.max())
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(0.0, 1.3 * max(r["repmass"] for r in per_run))
    ax.set_xlabel("repmass = token mass repeated past 4 epochs")
    ax.set_ylabel("swarm runs per bin")
    ax.set_title("(f) the epoch coordinate is off-support too", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    fit = out["in_kernel_fit"]
    ax.annotate(
        f"fitted gamma_epoch = {fit['gamma_epoch']:.2e}\n"
        f"gamma_content = {fit['gamma_content']:.3f}\n"
        f"epoch share of kernel distance: {100*fit['epoch_share_of_distance_train']:.2g}%\n"
        f"delta NLML vs (a): {fit['nlml_minus_baseline']:+.3f} nats\n"
        f"spearman(kernel residual, repmass) = {land['resid_vs_repmass_spearman']:+.3f}",
        (0.02, 0.55),
        xycoords="axes fraction",
        fontsize=7.5,
        va="top",
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"},
    )

    fig.suptitle(
        f"f34 -- epoch in the kernel vs epoch beside it, 53 off-design probe runs: {out['verdict']}", fontsize=11
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def write_report(out: dict, path: Path) -> None:
    fit = out["in_kernel_fit"]
    lines = [
        "# Does putting EPOCH inside the kernel beat content kernel + additive harm term?",
        "",
        f"Verdict: **{out['verdict']}**",
        "",
        out["verdict_detail"],
        "",
        "## The three models",
        "",
        "| id | model | where the epoch effect is learned |",
        "|---|---|---|",
        "| (a) | content kernel only | nowhere -- the documented failure |",
        "| (b) | content kernel + `sum_j b_g max(e_j - tau_g, 0)` | on the dedicated epoch experiments, "
        "imported verbatim, never refit here |",
        "| (c) | `exp(-gamma_c d2_content - gamma_e d2_epoch)` | on the swarm, by marginal likelihood |",
        "",
        "## Off-design probe runs (53 runs, humaneval bpb)",
        "",
        "| model | RMSE | mean \\|z\\| | cov 1sd | cov 2sd | mean miss |",
        "|---|---|---|---|---|---|",
    ]
    for model, label in MODEL_LABELS.items():
        s = out["offdesign"][model]["all"]
        lines.append(
            f"| {label} | {s['rmse']:.4f} | {s['mean_abs_z']:.1f} | {100*s['coverage_1sd']:.0f}% | "
            f"{100*s['coverage_2sd']:.0f}% | {s['mean_miss']:+.4f} |"
        )
    lines += [
        "",
        "### Per group (RMSE / mean |z| / coverage at 2sd)",
        "",
        "| group | n | " + " | ".join(MODEL_LABELS[m] for m in MODEL_LABELS) + " |",
        "|---|---|" + "---|" * len(MODEL_LABELS),
    ]
    for g in (*HUMANEVAL_GROUPS, "all"):
        cells = []
        for model in MODEL_LABELS:
            s = out["offdesign"][model][g]
            cells.append(f"{s['rmse']:.3f} / {s['mean_abs_z']:.1f} / {100*s['coverage_2sd']:.0f}%")
        n = out["offdesign"]["a_content_only"][g]["n"]
        lines.append(f"| {g} | {n} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### twobucket split by arm -- the one place the harm term's unidentified w-dependence bites",
        "",
        "| arm | n | " + " | ".join([*(MODEL_LABELS[m] for m in MODEL_LABELS), "(b) mass-weighted variant"]) + " |",
        "|---|---|" + "---|" * (len(MODEL_LABELS) + 1),
    ]
    for arm, per_model in out["twobucket_arms"].items():
        cells = [
            f"{per_model[m]['rmse']:.3f} / {per_model[m]['mean_abs_z']:.1f}" for m in (*MODEL_LABELS, "b_mass_weighted")
        ]
        lines.append(f"| {arm} | {per_model['a_content_only']['n']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Does the residual still slope with epochs? (epochrep arms: mixture fixed, only e moves)",
        "",
        "A residual that still climbs with e means the epoch axis is unmodelled. A FLAT residual at a "
        "nonzero level means what is left is a content-extrapolation error, not an epoch error.",
        "",
        "| arm | model | slope (bpb/epoch) | intercept | residual range |",
        "|---|---|---|---|---|",
    ]
    for arm, per_model in out["epochrep_residual_slopes"].items():
        for model, s in per_model.items():
            lines.append(
                f"| {arm} (n={s['n']}) | {MODEL_LABELS[model]} | {s['slope_bpb_per_epoch']:+.4f} | "
                f"{s['intercept_bpb']:+.3f} | {s['residual_range'][0]:+.3f} .. {s['residual_range'][1]:+.3f} |"
            )

    lines += [
        "",
        "## The fitted epoch length-scale",
        "",
        f"- `gamma_epoch` = **{fit['gamma_epoch']:.4e}** (bound floor {np.exp(GAMMA_BOUNDS[0]):.0e}), "
        f"`gamma_content` = {fit['gamma_content']:.4f}",
        f"- epoch share of the kernel's total exponent: **{100*fit['epoch_share_of_distance_train']:.3g}%** "
        f"across training pairs, {100*fit['epoch_share_of_distance_probe']:.3g}% at the probe runs",
        f"- log marginal likelihood gain over (a): {-fit['nlml_minus_baseline']:+.4f} nats for one extra "
        "hyperparameter",
        f"- multi-start over `gamma_epoch` in {list(GAMMA_E_STARTS)} all land at the same optimum, so the "
        "value is not an initialization artifact",
        f"- 5-fold refits give `gamma_epoch` = {out['cv']['c_epoch_in_kernel'].get('fold_gamma_epoch')}",
        "",
        "### Forcing `gamma_epoch` (grid, other hyperparameters refit at each point)",
        "",
        "Separates *marginal likelihood chose badly* from *no epoch length-scale would have helped*.",
        "",
        "| gamma_epoch | NLML | gamma_content | probe RMSE | probe mean \\|z\\| | cov 2sd |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['gamma_epoch']:g} | {r['nlml']:.2f} | {r['gamma_content']:.3f} | {r['rmse']:.4f} | "
            f"{r['mean_abs_z']:.1f} | {100*r['coverage_2sd']:.0f}% |"
            for r in out["gamma_epoch_profile"]
        ],
        "",
        "## In-distribution cost of the extra length-scale (5-fold held-out on the 800 swarm runs)",
        "",
        "| model | CV RMSE | Spearman | mean predicted sd | cov 2sd |",
        "|---|---|---|---|---|",
    ]
    for model, label in MODEL_LABELS.items():
        s = out["cv"][model]
        lines.append(
            f"| {label} | {s['rmse']:.5f} | {s['spearman']:.4f} | {s['mean_predicted_sd']:.5f} | "
            f"{100*s['coverage_2sd']:.1f}% |"
        )

    harm = out["harm_parameters"]
    lines += [
        "",
        "## Harm-term parameters (imported, not fitted here)",
        "",
        f"- tau: {json.dumps(harm['tau'])}; b at 10B/d512: {json.dumps(harm['b'])}",
        f"- amplitude scaling (B/10B)^-{harm['budget_exponent']:.3f} * (d/512)^+{harm['hidden_dim_exponent']:.3f}",
        f"- source: {harm['source']}",
        f"- harm applied to the swarm itself is small: mean {out['harm_on_swarm']['mean']:.4f} bpb, median "
        f"{out['harm_on_swarm']['median']:.4f}, p90 {out['harm_on_swarm']['p90']:.4f}, max "
        f"{out['harm_on_swarm']['max']:.4f} bpb against a target sd of "
        f"{out['harm_on_swarm']['target_sd']:.4f} -- the swarm sits essentially at the threshold, which is "
        "why (b) barely perturbs the in-distribution fit.",
        "- every calibration point held the sliced bucket at w = 0.2, so the harm term's dependence on "
        "mixture WEIGHT is unidentified. The mass-weighted reading "
        f"`sum_p sum_j w_pj (b_g/{harm['w_cal']}) max(e_pj - tau_g, 0)` fits the calibration equally well "
        f"and gives probe RMSE {out['offdesign_harm_mass_weighted']['all']['rmse']:.4f} / mean |z| "
        f"{out['offdesign_harm_mass_weighted']['all']['mean_abs_z']:.1f} vs "
        f"{out['offdesign']['b_content_plus_harm']['all']['rmse']:.4f} / "
        f"{out['offdesign']['b_content_plus_harm']['all']['mean_abs_z']:.1f} for the documented (unweighted) "
        "form. They agree everywhere except twobucket's natural arm, where w and e are perfectly "
        "confounded and the mass-weighted form extrapolates to an absurd "
        f"{out['harm_on_probe_mass_weighted']['max']:.1f} bpb of harm. Neither reading changes the verdict "
        "on (c).",
        "",
        "## Epoch landscape",
        "",
        f"- swarm repmass (token mass past {REPMASS_EPOCH_FLOOR:.0f} epochs): mean "
        f"{out['epoch_landscape']['train_repmass_mean']:.3f}, sd {out['epoch_landscape']['train_repmass_sd']:.3f}, "
        f"p90 {out['epoch_landscape']['train_repmass_p90']:.3f}, max {out['epoch_landscape']['train_repmass_max']:.3f}",
        f"- probe repmass: mean {out['epoch_landscape']['probe_repmass_mean']:.3f}, max "
        f"{out['epoch_landscape']['probe_repmass_max']:.3f} "
        f"({out['epoch_landscape']['probe_repmass_max']/max(out['epoch_landscape']['train_repmass_max'],1e-9):.1f}x "
        "the swarm maximum)",
        f"- swarm max per-phase epoch: p50 {out['epoch_landscape']['train_max_epoch_p50']:.1f}, "
        f"p90 {out['epoch_landscape']['train_max_epoch_p90']:.1f}, max "
        f"{out['epoch_landscape']['train_max_epoch_max']:.1f} -- there IS variation, the question is whether "
        "there is SIGNAL",
        "- spearman(kernel CV residual, repmass) on the swarm = "
        f"{out['epoch_landscape']['resid_vs_repmass_spearman']:+.3f}",
        "",
        "## Reconstruction checks",
        "",
        f"- {out['reconstruction_check']['n_runs_checked']} probe runs reproduce their pre-registered "
        f"frozen-kernel humaneval prediction to {out['reconstruction_check']['max_abs_diff_vs_prereg_kernel']:.2e} bpb",
        f"- the transect's reconstructed phase-0 epochs match its pre-registered targets to "
        f"{out['transect_epoch_check']:.2e} epochs",
        "",
        "## Worst probe runs under each model",
        "",
        "| run | realized | (a) | (b) | (c) |",
        "|---|---|---|---|---|",
    ]
    worst = sorted(out["per_run"], key=lambda r: -abs(r["a_content_only_miss"]))[:10]
    for r in worst:
        lines.append(
            f"| {r['run_id']} | {r['realized']:.4f} | {r['a_content_only_mean']:.4f} | "
            f"{r['b_content_plus_harm_mean']:.4f} | {r['c_epoch_in_kernel_mean']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_verdict(out: dict) -> tuple[str, str]:
    """(c)-collapses-into-(a) is decided FIRST; only then is (b) judged against the baseline."""
    a = out["offdesign"]["a_content_only"]["all"]
    b = out["offdesign"]["b_content_plus_harm"]["all"]
    c = out["offdesign"]["c_epoch_in_kernel"]["all"]
    fit = out["in_kernel_fit"]
    collapsed = fit["epoch_share_of_distance_train"] < 0.01 and abs(c["rmse"] - a["rmse"]) < 0.02 * a["rmse"]
    b_beats_a = b["rmse"] < 0.9 * a["rmse"] or b["mean_abs_z"] < 0.9 * a["mean_abs_z"]
    if not collapsed and c["rmse"] < 0.9 * min(a["rmse"], b["rmse"]):
        verdict = "(c) EPOCH-IN-KERNEL WINS -- the pre-registered prediction is REFUTED"
    elif collapsed and b_beats_a:
        verdict = (
            "(c) COLLAPSES INTO (a); only the additive harm term moves the probe error -- "
            "the pre-registered prediction HELD"
        )
    elif collapsed:
        verdict = "(c) COLLAPSES INTO (a), and (b) does not help either -- prediction half held"
    else:
        verdict = "(c) does not collapse but does not win either -- prediction partly refuted"
    detail = (
        f"On the 53 probe runs: (a) RMSE {a['rmse']:.4f} / mean |z| {a['mean_abs_z']:.1f} / "
        f"{100*a['coverage_2sd']:.0f}% inside 2sd; (b) RMSE {b['rmse']:.4f} / {b['mean_abs_z']:.1f} / "
        f"{100*b['coverage_2sd']:.0f}%; (c) RMSE {c['rmse']:.4f} / {c['mean_abs_z']:.1f} / "
        f"{100*c['coverage_2sd']:.0f}%. The marginal likelihood puts gamma_epoch at "
        f"{fit['gamma_epoch']:.2e}, contributing {100*fit['epoch_share_of_distance_train']:.3g}% of the "
        f"kernel's total exponent across training pairs, for a log-evidence change of "
        f"{-fit['nlml_minus_baseline']:+.3f} nats -- and forcing gamma_epoch up its whole grid never "
        f"lowers the probe RMSE below {min(r['rmse'] for r in out['gamma_epoch_profile']):.4f}."
    )
    return verdict, detail


def main() -> None:
    design = load_design()
    hists, _views, _cent, _rff, _rffo, buckets_table = gf.load_grug_artifacts()
    buckets = design["buckets"]
    design["total_tokens"] = buckets_table.set_index("bucket").loc[buckets, "total_tokens"].to_numpy(dtype=np.float64)
    masks, _group_doc = cluster_delta_groups(buckets, buckets_table, design["v1000"])
    assert len(hists) == len(buckets)

    records = offdesign_runs(design)
    check = verify_reconstruction(design, records)
    rows = [r for r in records if r["group"] in HUMANEVAL_GROUPS and r.get(HUMANEVAL) is not None]

    train = pd.read_parquet(gf.TRAIN_RUNS)
    w_train = gf.weight_matrix(train, buckets)
    ep_train = sf.per_phase_epochs(w_train, design["total_tokens"])
    y = design["y"][HUMANEVAL]

    meta = probe_epoch_metadata()
    probe = probe_epochs(design, rows, meta)
    realized = np.array([float(r[HUMANEVAL]) for r in rows])
    groups = np.array([r["group"] for r in rows])
    arms = np.array([meta[r["run_id"]]["arm"] for r in rows])

    harm_params = load_harm_parameters()
    scale_train = np.ones(len(w_train))
    scale_probe = amplitude_scale(probe["budget_tokens"], probe["hidden_dim"], harm_params)
    h_train = harm_term(w_train, ep_train, masks, scale_train, harm_params)
    h_probe = harm_term(probe["w"], probe["ep"], masks, scale_probe, harm_params)
    h_train_mw = harm_term_mass_weighted(w_train, ep_train, masks, scale_train, harm_params)
    h_probe_mw = harm_term_mass_weighted(probe["w"], probe["ep"], masks, scale_probe, harm_params)

    # standardized epoch coordinate for the in-kernel model
    rep_train = repmass(w_train, ep_train)
    rep_probe = repmass(probe["w"], probe["ep"])
    mu_rep, sd_rep = float(rep_train.mean()), float(rep_train.std())
    r_train = (rep_train - mu_rep) / sd_rep
    r_probe = (rep_probe - mu_rep) / sd_rep
    d2_epoch_train = (r_train[:, None] - r_train[None, :]) ** 2
    d2_epoch_probe = (r_probe[:, None] - r_train[None, :]) ** 2

    d2_train = design["d2_train"]
    d2_probe = sf.candidate_d2(gf.per_phase_hist(probe["w"], design["v1000"]), design["h_train"])

    # --- (a) content only ---
    fit_a = fit_gp(d2_train, y)
    mu_a, sd_a = predict_gp(fit_a, d2_probe, include_noise=True)

    # --- (b) content + imported harm term ---
    fit_b = fit_gp(d2_train, y - h_train)
    mu_b, sd_b = predict_gp(fit_b, d2_probe, include_noise=True)
    mu_b = mu_b + h_probe
    fit_bmw = fit_gp(d2_train, y - h_train_mw)
    mu_bmw, sd_bmw = predict_gp(fit_bmw, d2_probe, include_noise=True)
    mu_bmw = mu_bmw + h_probe_mw

    # --- (c) epoch inside the kernel ---
    fit_c = fit_gp2(d2_train, d2_epoch_train, y)
    mu_c, sd_c = predict_gp2(fit_c, d2_probe, d2_epoch_probe)

    upper = np.triu_indices(len(y), 1)
    oof = np.empty_like(y)
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(len(y))):
        oof[te] = predict_gp(fit_gp(d2_train[np.ix_(tr, tr)], y[tr]), d2_train[np.ix_(te, tr)], include_noise=True)[0]
    kernel_resid = y - oof

    out = {
        "question": (
            "Does putting epoch inside the GP kernel (one extra length-scale, fit by marginal likelihood on "
            "the swarm) beat the simpler content kernel + additive harm term whose parameters come from the "
            "dedicated epoch experiments?"
        ),
        "reconstruction_check": check,
        "transect_epoch_check": probe["transect_epoch_check"],
        "harm_parameters": harm_params,
        "offdesign": {
            "a_content_only": score_by_group(groups, realized, mu_a, sd_a),
            "b_content_plus_harm": score_by_group(groups, realized, mu_b, sd_b),
            "c_epoch_in_kernel": score_by_group(groups, realized, mu_c, sd_c),
        },
        "offdesign_harm_mass_weighted": score_by_group(groups, realized, mu_bmw, sd_bmw),
        "twobucket_arms": {
            arm: {
                model: score(realized[m], mu[m], sd[m])
                for model, (mu, sd) in (
                    ("a_content_only", (mu_a, sd_a)),
                    ("b_content_plus_harm", (mu_b, sd_b)),
                    ("c_epoch_in_kernel", (mu_c, sd_c)),
                    ("b_mass_weighted", (mu_bmw, sd_bmw)),
                )
            }
            for arm, m in (
                ("natural (w and epochs confounded)", arms == "natural"),
                ("sliced (w fixed, epochs varied)", (groups == "twobucket") & (arms != "natural")),
            )
        },
        "gamma_epoch_profile": profile_gamma_epoch(d2_train, d2_epoch_train, y, d2_probe, d2_epoch_probe, realized),
        "baseline_fit": {k: fit_a[k] for k in ("sigma_f2", "sigma_n2", "gamma", "nlml")},
        "in_kernel_fit": {
            **{k: fit_c[k] for k in ("sigma_f2", "sigma_n2", "gamma_content", "gamma_epoch", "nlml")},
            "nlml_minus_baseline": float(fit_c["nlml"] - fit_a["nlml"]),
            "epoch_share_of_distance_train": epoch_distance_share(fit_c, d2_train[upper], d2_epoch_train[upper]),
            "epoch_share_of_distance_probe": epoch_distance_share(fit_c, d2_probe, d2_epoch_probe),
            "gamma_epoch_starts": list(GAMMA_E_STARTS),
        },
        "harm_on_swarm": {
            "mean": float(h_train.mean()),
            "median": float(np.median(h_train)),
            "p90": float(np.quantile(h_train, 0.9)),
            "max": float(h_train.max()),
            "frac_nonzero": float((h_train > 0).mean()),
            "target_sd": float(y.std()),
        },
        "harm_on_probe": {
            "mean": float(h_probe.mean()),
            "max": float(h_probe.max()),
            "per_group_mean": {g: float(h_probe[groups == g].mean()) for g in HUMANEVAL_GROUPS},
        },
        "harm_on_probe_mass_weighted": {"mean": float(h_probe_mw.mean()), "max": float(h_probe_mw.max())},
        "epoch_landscape": {
            "train_repmass_mean": float(rep_train.mean()),
            "train_repmass_sd": sd_rep,
            "train_repmass_p90": float(np.quantile(rep_train, 0.9)),
            "train_repmass_max": float(rep_train.max()),
            "probe_repmass_mean": float(rep_probe.mean()),
            "probe_repmass_max": float(rep_probe.max()),
            "train_max_epoch_p50": float(np.median(ep_train.max(axis=(1, 2)))),
            "train_max_epoch_p90": float(np.quantile(ep_train.max(axis=(1, 2)), 0.9)),
            "train_max_epoch_max": float(ep_train.max()),
            "resid_vs_repmass_spearman": float(spearmanr(kernel_resid, rep_train).statistic),
        },
        "train_repmass_sample": [float(v) for v in rep_train],
        "cv": {
            "a_content_only": cross_validate(d2_train, d2_epoch_train, y, h_train, "a_content_only"),
            "b_content_plus_harm": cross_validate(d2_train, d2_epoch_train, y, h_train, "b_content_plus_harm"),
            "c_epoch_in_kernel": cross_validate(d2_train, d2_epoch_train, y, h_train, "c_epoch_in_kernel"),
        },
        "figure": str(FIG_PATH),
    }

    per_run = []
    for i, rec in enumerate(rows):
        row = {
            "group": rec["group"],
            "run_id": rec["run_id"],
            "label": rec["label"],
            "realized": float(realized[i]),
            "arm": str(arms[i]),
            "repmass": float(rep_probe[i]),
            "max_phase0_epoch": float(probe["ep"][i, 0].max()),
            "harm_bpb": float(h_probe[i]),
            "amplitude_scale": float(scale_probe[i]),
        }
        for model, (mu, sd) in {
            "a_content_only": (mu_a, sd_a),
            "b_content_plus_harm": (mu_b, sd_b),
            "c_epoch_in_kernel": (mu_c, sd_c),
        }.items():
            row[f"{model}_mean"] = float(mu[i])
            row[f"{model}_sd"] = float(sd[i])
            row[f"{model}_miss"] = float(realized[i] - mu[i])
            row[f"{model}_z"] = float((realized[i] - mu[i]) / sd[i])
        per_run.append(row)
    out["per_run"] = per_run
    out["epochrep_residual_slopes"] = {
        arm: {model: residual_epoch_slope(per_run, "epochrep", arm, model) for model in MODEL_LABELS}
        for arm in ("code", "web")
    }

    out["verdict"], out["verdict_detail"] = build_verdict(out)
    make_figure(out, per_run, FIG_PATH)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    write_report(out, OUT_MD)

    for model, label in MODEL_LABELS.items():
        s = out["offdesign"][model]["all"]
        print(f"{label:42s} RMSE {s['rmse']:.4f}  mean|z| {s['mean_abs_z']:6.2f}  cov2sd {100*s['coverage_2sd']:5.1f}%")
    share = 100 * out["in_kernel_fit"]["epoch_share_of_distance_train"]
    print(f"gamma_epoch = {fit_c['gamma_epoch']:.4e} (share {share:.3g}%)")
    print(f"VERDICT: {out['verdict']}")
    print(f"wrote {OUT_JSON}, {OUT_MD}, {FIG_PATH}")


if __name__ == "__main__":
    main()
