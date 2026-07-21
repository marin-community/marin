# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Are the GP surrogate's error bars trustworthy OFF-SUPPORT?

``gp_surrogate_validate`` established that the credible intervals are calibrated
*in*-distribution (RMS z = 1.014, 95% coverage 96.2%) -- but the predicted sd is nearly flat
there (range 1.37x) because it is dominated by the irreducible seed noise. That is a weak
test: everything in the swarm is well covered, so a nearly constant sd can look calibrated.

The decisive test is the one the campaign can actually run: the deliberately extreme probe
runs (two-bucket, epoch-repetition, transect, math-generality, 100B-harm) sit far outside the
Dirichlet swarm's support, and the kernel is KNOWN to miss them badly (up to ~+0.4 bpb on
humaneval on the two-bucket axis-1 sweep). So:

    does the posterior sd inflate enough off-support to COVER those known misses,
    or is the GP most overconfident exactly where it is most wrong?

Method: fit the GP on the 800 swarm training runs with target = humaneval bpb, reconstruct
every probe run's per-phase bucket weights from its frozen pre-registration, map them through
the SAME K=1000 content basis (h = V w), and score z = (realized - posterior mean) / sd.

Note the content features are blind to epoching by construction: probe runs that differ only
in how many epochs the sliced bucket is repeated get IDENTICAL h, hence identical posterior
mean AND sd. Their realized humaneval bpb differs by more than a bpb. That is the sharpest
form of the question.

humaneval is the target (not zmacro) because it is complete for the swarm and for every probe
group that measured it. mathgen/mathgen2 evaluated gsm8k + sat_math only, so they are scored
in a secondary gsm8k-target pass.
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "experiments/datakit/mixture_features")
import featurize
import grug_fit as gf
import swoosh_form as sf
from gp_surrogate import fit_gp, predict_gp
from retrodiction import _sq_hellinger

GRUG_DIR = Path("scratch/mixture_features/grug")
OUT_JSON = GRUG_DIR / "gp_ood_coverage.json"
OUT_MD = GRUG_DIR / "gp_ood_coverage.md"
FIG_PATH = Path("scratch/mixture_features/report/figs3/f33_gp_ood_coverage.png")

HUMANEVAL = "logprob_humaneval_10shot"
GSM8K = "logprob_gsm8k_5shot"

CORRECTED_PHASE1_START = 38144
CORRECTED_TOTAL_STEPS = 47759

N_FOLDS = 5
SEED = 0

# groups scored against humaneval, in the order they are reported
HUMANEVAL_GROUPS = ("twobucket", "epochrep", "transect", "harm100b")
GSM8K_GROUPS = ("mathgen", "mathgen2")

GROUP_COLORS = {
    "twobucket": "#c1272d",
    "epochrep": "#e08214",
    "transect": "#2166ac",
    "harm100b": "#762a83",
    "mathgen": "#1b7837",
    "mathgen2": "#7fbc41",
}


# ---------------------------------------------------------------------------
# design: the 800-run swarm, its content features, and the K=1000 basis
# ---------------------------------------------------------------------------


def load_design() -> dict:
    """Swarm weights -> per-phase h (K=1000) -> train/train squared-Hellinger d2, plus targets."""
    gf.phase_step_split = lambda: (CORRECTED_PHASE1_START, CORRECTED_TOTAL_STEPS - CORRECTED_PHASE1_START)
    gf.TOTAL_STEPS = CORRECTED_TOTAL_STEPS
    hists, views, _cent, _rff, _rffo, _bt = gf.load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, order = featurize.composition_matrix(hists, k=1000, views=views)
    if order != buckets:
        raise AssertionError("composition_matrix column order does not match bucket order")
    v1000 = np.asarray(v1000, dtype=np.float64)

    train = pd.read_parquet(gf.TRAIN_RUNS)
    w_train = gf.weight_matrix(train, buckets)
    h_train = gf.per_phase_hist(w_train, v1000)
    evals = [json.loads(e) for e in train["evals"]]
    return {
        "buckets": buckets,
        "index": {b: j for j, b in enumerate(buckets)},
        "v1000": v1000,
        "h_train": h_train,
        "d2_train": _sq_hellinger(h_train),
        "y": {task: np.array([e[task]["bpb"] for e in evals], dtype=np.float64) for task in (HUMANEVAL, GSM8K)},
    }


# ---------------------------------------------------------------------------
# off-design runs: reconstruct per-phase weights from the frozen pre-registrations
# ---------------------------------------------------------------------------


def _prereg(group: str) -> dict:
    return json.loads((GRUG_DIR / f"{group}_preregistration.json").read_text())


def _two_bucket_weights(index: dict[str, int], sliced: str, partner: str, w: float) -> np.ndarray:
    """(2, n_buckets): the probe launchers all set the same two-bucket weights in BOTH phases."""
    arr = np.zeros((2, len(index)))
    arr[:, index[sliced]] = w
    arr[:, index[partner]] = 1.0 - w
    return arr


def _combined_bpb(group: str, task: str) -> dict[str, float]:
    """realized bpb per run_id from the group's combined eval-results dump."""
    runs = json.loads((GRUG_DIR / f"{group}_eval_results_combined.json").read_text())["runs"]
    out = {}
    for run_id, tasks in runs.items():
        block = tasks.get(task)
        if block is not None:
            out[run_id] = float(block[task]["bpb,none"])
    return out


def offdesign_runs(design: dict) -> list[dict]:
    """Every probe run: group, run_id, reconstructed (2, n_buckets) weights, realized bpb."""
    index = design["index"]
    n_buckets = len(index)
    records: list[dict] = []

    # --- twobucket: code c01q0 at w_code, web c05q0 at 1 - w_code ---
    hev = _combined_bpb("twobucket", HUMANEVAL)
    for run in _prereg("twobucket")["runs"]:
        records.append(
            {
                "group": "twobucket",
                "run_id": run["run_id"],
                "label": run["point"],
                "w": _two_bucket_weights(index, "c01q0", "c05q0", float(run["w_code"])),
                HUMANEVAL: hev.get(run["run_id"]),
            }
        )

    # --- epochrep / harm100b / mathgen / mathgen2: sliced bucket at w_target, partner at 1 - w ---
    for group, task in (("epochrep", HUMANEVAL), ("harm100b", HUMANEVAL), ("mathgen2", GSM8K)):
        realized = _combined_bpb(group, task)
        for run in _prereg(group)["runs"]:
            records.append(
                {
                    "group": group,
                    "run_id": run["run_id"],
                    "label": run["point"],
                    "w": _two_bucket_weights(index, run["sliced_bucket"], run["partner_bucket"], float(run["w_target"])),
                    task: realized.get(run["run_id"]),
                }
            )

    # mathgen (v1) predates the combined dump; its readout carries the per-point gsm8k bpb
    mathgen_readout = json.loads((GRUG_DIR / "mathgen_readout.json").read_text())["per_point"]
    for run in _prereg("mathgen")["runs"]:
        point = mathgen_readout.get(run["point"], {})
        records.append(
            {
                "group": "mathgen",
                "run_id": run["run_id"],
                "label": run["point"],
                "w": _two_bucket_weights(index, run["sliced_bucket"], run["partner_bucket"], float(run["w_target"])),
                GSM8K: point.get("gsm8k_bpb"),
            }
        )

    # --- transect: full phase-0 mixture from the prereg, phase 1 unchanged at the anchor ---
    transect = _prereg("transect")
    anchor = sf.load_anchor(design["buckets"])
    hev = _combined_bpb("transect", HUMANEVAL)
    for run in transect["runs"]:
        w = np.zeros((2, n_buckets))
        for bucket, val in transect["mixtures"][run["run_name"]]["phase0"].items():
            w[0, index[bucket]] = val
        w[1, :] = anchor[1]
        records.append(
            {
                "group": "transect",
                "run_id": run["run_name"],
                "label": run["run_name"].rsplit("_", 1)[-1],
                "w": w,
                HUMANEVAL: hev.get(run["run_name"]),
            }
        )

    for rec in records:
        if abs(rec["w"].sum(axis=1) - 1.0).max() > 1e-9:
            raise AssertionError(f"{rec['run_id']}: reconstructed phase weights do not sum to 1")
    return records


def verify_reconstruction(design: dict, records: list[dict]) -> dict:
    """The reconstructed weights must reproduce the pre-registered kernel predictions exactly.

    epochrep and transect committed ``pred_kernel_humaneval`` under the campaign's frozen
    kernel (gamma, alpha from ``frozen_model_hyperparams.json``). Reproducing those values is
    an end-to-end check that the weight reconstruction, the K=1000 basis and the distance are
    all the ones the campaign actually used.
    """
    frozen = json.loads((GRUG_DIR / "frozen_model_hyperparams.json").read_text())["models"]["4_hellinger_kernel_k1000"]
    gamma, alpha = float(frozen["gamma"]), float(frozen["alpha"])
    dual, ymean = sf.kernel_dual(design["d2_train"], design["y"][HUMANEVAL], gamma, alpha)

    by_run = {r["run_id"]: r for r in records}
    committed: dict[str, float] = {}
    for run in _prereg("epochrep")["runs"]:
        committed[run["run_id"]] = float(run["pred_kernel_humaneval"])
    for run in _prereg("transect")["runs"]:
        committed[run["run_name"]] = float(run["pred_kernel_humaneval"])

    run_ids = [r for r in committed if r in by_run]
    h = gf.per_phase_hist(np.stack([by_run[r]["w"] for r in run_ids]), design["v1000"])
    pred = np.exp(-gamma * sf.candidate_d2(h, design["h_train"])) @ dual + ymean
    err = np.abs(pred - np.array([committed[r] for r in run_ids]))
    if err.max() > 1e-9:
        worst = run_ids[int(err.argmax())]
        raise AssertionError(f"weight reconstruction mismatch at {worst}: {err.max():.3e}")
    return {"n_runs_checked": len(run_ids), "max_abs_diff_vs_prereg_kernel": float(err.max())}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def indistribution_reference(d2: np.ndarray, y: np.ndarray) -> dict:
    """Held-out (5-fold, hyperparameters refit inside each fold) calibration on the swarm."""
    mus, sds, ys, nn = [], [], [], []
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(len(y))):
        fold = fit_gp(d2[np.ix_(tr, tr)], y[tr])
        mu, sd = predict_gp(fold, d2[np.ix_(te, tr)], include_noise=True)
        mus.append(mu)
        sds.append(sd)
        ys.append(y[te])
        nn.append(np.sqrt(d2[np.ix_(te, tr)].min(axis=1)))
    mu, sd, y_te, nn = np.concatenate(mus), np.concatenate(sds), np.concatenate(ys), np.concatenate(nn)
    z = (y_te - mu) / sd
    return {
        "n": len(y),
        "cv_rmse": float(np.sqrt(np.mean((y_te - mu) ** 2))),
        "mean_predicted_sd": float(sd.mean()),
        "predicted_sd_range": [float(sd.min()), float(sd.max())],
        "rms_z": float(np.sqrt((z**2).mean())),
        "mean_abs_z": float(np.abs(z).mean()),
        "coverage_1sd": float(np.mean(np.abs(z) < 1.0)),
        "coverage_2sd": float(np.mean(np.abs(z) < 2.0)),
        "median_nn_hellinger": float(np.median(nn)),
        "median_nn_d2": float(np.median(nn**2)),
        "max_nn_hellinger": float(nn.max()),
        "_sd": sd,
        "_z": z,
        "_nn": nn,
    }


def score_group(rows: list[dict]) -> dict:
    miss = np.array([r["miss"] for r in rows])
    sd = np.array([r["gp_sd"] for r in rows])
    z = np.array([r["z"] for r in rows])
    nn = np.array([r["nn_hellinger"] for r in rows])
    return {
        "n": len(rows),
        "mean_miss": float(miss.mean()),
        "mean_abs_miss": float(np.abs(miss).mean()),
        "max_abs_miss": float(np.abs(miss).max()),
        "mean_predicted_sd": float(sd.mean()),
        "predicted_sd_range": [float(sd.min()), float(sd.max())],
        "mean_abs_z": float(np.abs(z).mean()),
        "max_abs_z": float(np.abs(z).max()),
        "coverage_1sd": float(np.mean(np.abs(z) < 1.0)),
        "coverage_2sd": float(np.mean(np.abs(z) < 2.0)),
        "mean_nn_hellinger": float(nn.mean()),
        "mean_nn_d2": float((nn**2).mean()),
        "min_nn_hellinger": float(nn.min()),
        "frac_miss_positive": float(np.mean(miss > 0)),
    }


def epoching_blindness(per_run: list[dict]) -> dict:
    """Probe runs that share a mixture but differ in epochs get the IDENTICAL posterior.

    The content features are h = V w, so two runs with the same weights are the same point to
    the GP no matter how many times the sliced bucket is repeated. Grouping runs by their
    (mean, sd) exposes how much realized bpb varies across a single credible interval.
    """
    clusters: dict[tuple[float, float], list[dict]] = {}
    for rec in per_run:
        clusters.setdefault((round(rec["gp_mean"], 9), round(rec["gp_sd"], 9)), []).append(rec)
    tied = [
        {
            "gp_mean": key[0],
            "gp_sd": key[1],
            "n_runs": len(rows),
            "realized_min": min(r["realized"] for r in rows),
            "realized_max": max(r["realized"] for r in rows),
            "realized_spread": max(r["realized"] for r in rows) - min(r["realized"] for r in rows),
            "spread_in_sd_units": (max(r["realized"] for r in rows) - min(r["realized"] for r in rows)) / key[1],
            "run_ids": [r["run_id"] for r in rows],
        }
        for key, rows in clusters.items()
        if len(rows) > 1
    ]
    tied.sort(key=lambda c: -c["realized_spread"])
    return {
        "n_distinct_posteriors": len(clusters),
        "n_runs": len(per_run),
        "worst_tied_cluster": tied[0] if tied else None,
        "tied_clusters": tied,
    }


def score_offdesign(design: dict, records: list[dict], task: str, groups: tuple[str, ...]) -> dict:
    """Fit the GP on all 800 swarm runs for ``task``, then score every probe run in ``groups``."""
    y = design["y"][task]
    fit = fit_gp(design["d2_train"], y)
    rows_all = [r for r in records if r["group"] in groups]
    missing = [r["run_id"] for r in rows_all if r.get(task) is None]
    rows = [r for r in rows_all if r.get(task) is not None]

    h = gf.per_phase_hist(np.stack([r["w"] for r in rows]), design["v1000"])
    d2_star = sf.candidate_d2(h, design["h_train"])
    mu, sd = predict_gp(fit, d2_star, include_noise=True)
    nn = np.sqrt(d2_star.min(axis=1))

    per_run = []
    for i, rec in enumerate(rows):
        realized = float(rec[task])
        per_run.append(
            {
                "group": rec["group"],
                "run_id": rec["run_id"],
                "label": rec["label"],
                "realized": realized,
                "gp_mean": float(mu[i]),
                "gp_sd": float(sd[i]),
                "miss": realized - float(mu[i]),
                "z": (realized - float(mu[i])) / float(sd[i]),
                "nn_hellinger": float(nn[i]),
            }
        )

    # in-sample sd at the 800 training locations under the SAME fit: the floor the GP reports
    # where it has data, and the fair denominator for "how much does sd inflate off-support?"
    _, sd_insample = predict_gp(fit, design["d2_train"], include_noise=True)
    z_all = np.array([r["z"] for r in per_run])
    miss_all = np.abs([r["miss"] for r in per_run])
    return {
        "epoching_blindness": epoching_blindness(per_run),
        "spearman_sd_abs_miss": float(spearmanr(np.array([r["gp_sd"] for r in per_run]), miss_all).statistic),
        # how much larger every sd would have to be before the 2sd interval covered 95% of these runs
        "sd_scale_needed_for_95pct_coverage": float(np.quantile(np.abs(z_all), 0.95) / 2.0),
        "task": task,
        "hyperparameters": {k: fit[k] for k in ("sigma_f2", "sigma_n2", "gamma", "ridge_equivalent_alpha", "nlml")},
        "noise_sd": float(np.sqrt(fit["sigma_n2"])),
        "train_target_mean": float(y.mean()),
        "train_target_sd": float(y.std()),
        "in_sample_mean_sd": float(sd_insample.mean()),
        "missing_runs": missing,
        "per_group": {g: score_group([r for r in per_run if r["group"] == g]) for g in groups},
        "overall": score_group(per_run),
        "coverage_1sd_all": float(np.mean(np.abs(z_all) < 1.0)),
        "coverage_2sd_all": float(np.mean(np.abs(z_all) < 2.0)),
        "per_run": per_run,
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def make_figure(hev: dict, ref: dict, path: Path) -> None:
    per_run = hev["per_run"]
    groups = [g for g in HUMANEVAL_GROUPS if any(r["group"] == g for r in per_run)]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    # --- (a) predicted +/- 2sd vs realized -------------------------------------------------
    ax = axes[0]
    lo = min(min(r["gp_mean"] for r in per_run), min(r["realized"] for r in per_run)) - 0.1
    hi = max(max(r["gp_mean"] for r in per_run), max(r["realized"] for r in per_run)) + 0.1
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, ls="--", zorder=1)
    ax.fill_between(
        [lo, hi],
        [lo - 2 * ref["mean_predicted_sd"], hi - 2 * ref["mean_predicted_sd"]],
        [lo + 2 * ref["mean_predicted_sd"], hi + 2 * ref["mean_predicted_sd"]],
        color="0.85",
        zorder=0,
        label=f"in-distribution 2sd band (+/-{2*ref['mean_predicted_sd']:.3f})",
    )
    for g in groups:
        rows = [r for r in per_run if r["group"] == g]
        ax.errorbar(
            [r["gp_mean"] for r in rows],
            [r["realized"] for r in rows],
            xerr=[2 * r["gp_sd"] for r in rows],
            fmt="o",
            ms=5,
            lw=1,
            capsize=2,
            color=GROUP_COLORS[g],
            label=f"{g} (n={len(rows)})",
            zorder=3,
        )
    ax.set_xlabel("GP posterior mean, humaneval bpb (bars = +/-2sd)")
    ax.set_ylabel("realized humaneval bpb")
    ax.set_title("(a) off-design runs vs the GP's credible interval")
    ax.legend(fontsize=7, loc="upper left")

    # --- (b) predicted sd vs distance to the training support --------------------------------
    ax = axes[1]
    ax.scatter(
        ref["_nn"] ** 2, ref["_sd"], s=6, color="0.65", label=f"in-distribution held-out (n={ref['n']})", zorder=2
    )
    for g in groups:
        rows = [r for r in per_run if r["group"] == g]
        ax.scatter(
            [r["nn_hellinger"] ** 2 for r in rows],
            [r["gp_sd"] for r in rows],
            s=34,
            color=GROUP_COLORS[g],
            edgecolor="white",
            lw=0.5,
            label=g,
            zorder=3,
        )
    ax.axhline(hev["noise_sd"], color="k", ls=":", lw=1, label=f"seed-noise floor {hev['noise_sd']:.3f}")
    ax.axvline(ref["median_nn_d2"], color="0.4", ls="--", lw=1)
    ylo, yhi = ax.get_ylim()
    ax.annotate(
        f"in-dist median\nNN d2 {ref['median_nn_d2']:.4f}",
        (ref["median_nn_d2"], ylo + 0.62 * (yhi - ylo)),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=7,
        color="0.3",
    )
    ax.set_xlabel("squared Hellinger distance to nearest training run")
    ax.set_ylabel("GP posterior sd (bpb)")
    ax.set_title("(b) the sd barely moves off-support")
    ax.legend(fontsize=7, loc="upper left")

    # --- (c) |z| per run ---------------------------------------------------------------------
    ax = axes[2]
    x = 0
    ticks, labels = [], []
    for g in groups:
        rows = sorted((r for r in per_run if r["group"] == g), key=lambda r: abs(r["z"]))
        xs = np.arange(x, x + len(rows))
        ax.bar(xs, [abs(r["z"]) for r in rows], color=GROUP_COLORS[g], width=0.85)
        ticks.append(float(xs.mean()))
        labels.append(f"{g}\n(n={len(rows)})")
        x += len(rows) + 5
    ax.axhline(2.0, color="k", ls="--", lw=1, label="|z| = 2 (95% credible interval)")
    ax.axhline(np.sqrt((ref["_z"] ** 2).mean()), color="0.5", ls=":", lw=1.2, label="in-distribution RMS z (~1)")
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("|z| = |realized - mean| / sd   (log scale)")
    ax.set_title("(c) how badly the interval is violated")
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "f33 -- GP credible intervals off-support: calibrated inside the swarm, "
        "catastrophically overconfident outside it",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _group_table(res: dict, groups: tuple[str, ...]) -> list[str]:
    lines = [
        "| group | n | mean miss (bpb) | mean predicted sd | mean \\|z\\| | max \\|z\\| | cov 1sd | cov 2sd "
        "| mean NN dist |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in groups:
        s = res["per_group"][g]
        lines.append(
            f"| {g} | {s['n']} | {s['mean_miss']:+.3f} | {s['mean_predicted_sd']:.4f} | {s['mean_abs_z']:.1f} | "
            f"{s['max_abs_z']:.1f} | {100*s['coverage_1sd']:.0f}% | {100*s['coverage_2sd']:.0f}% | "
            f"{s['mean_nn_hellinger']:.3f} |"
        )
    o = res["overall"]
    lines.append(
        f"| **all** | {o['n']} | {o['mean_miss']:+.3f} | {o['mean_predicted_sd']:.4f} | {o['mean_abs_z']:.1f} | "
        f"{o['max_abs_z']:.1f} | {100*o['coverage_1sd']:.0f}% | {100*o['coverage_2sd']:.0f}% | "
        f"{o['mean_nn_hellinger']:.3f} |"
    )
    return lines


def _blindness_line(res: dict) -> str:
    worst = res["epoching_blindness"]["worst_tied_cluster"]
    if worst is None:
        return "- no two probe runs share a posterior."
    return (
        f"- worst case: {worst['n_runs']} runs share the posterior "
        f"{worst['gp_mean']:.3f} +/- {worst['gp_sd']:.3f} bpb while their realized bpb spans "
        f"{worst['realized_min']:.3f} to {worst['realized_max']:.3f} -- a "
        f"{worst['realized_spread']:.3f} bpb spread, {worst['spread_in_sd_units']:.0f} sd wide "
        f"({', '.join(worst['run_ids'][:4])}{', ...' if len(worst['run_ids']) > 4 else ''})."
    )


def write_report(out: dict, path: Path) -> None:
    hev, gsm, ref = out["humaneval"], out["gsm8k"], out["in_distribution_humaneval"]
    inflation = hev["overall"]["mean_predicted_sd"] / ref["mean_predicted_sd"]
    lines = [
        "# Are the GP surrogate's error bars trustworthy off-support?",
        "",
        f"Verdict: **{out['verdict']}**",
        "",
        f"{out['verdict_detail']}",
        "",
        "## In-distribution reference (5-fold held-out on the 800 swarm runs, humaneval bpb)",
        "",
        f"- CV RMSE {ref['cv_rmse']:.4f} bpb, mean predicted sd {ref['mean_predicted_sd']:.4f} bpb",
        f"- RMS z {ref['rms_z']:.3f}, coverage 1sd {100*ref['coverage_1sd']:.1f}%, "
        f"2sd {100*ref['coverage_2sd']:.1f}%",
        f"- median distance to nearest training run: {ref['median_nn_d2']:.4f} squared Hellinger "
        f"(= {ref['median_nn_hellinger']:.4f} Hellinger; max {ref['max_nn_hellinger']:.4f} Hellinger). "
        "The campaign quotes this in both conventions -- 0.0945 is the squared-Hellinger figure.",
        f"- fitted seed-noise sd {hev['noise_sd']:.4f} bpb; in-sample mean sd " f"{hev['in_sample_mean_sd']:.4f} bpb",
        "",
        "## Off-design runs, humaneval bpb",
        "",
        *_group_table(hev, HUMANEVAL_GROUPS),
        "",
        f"- predicted sd off-support / in-distribution held-out = **{inflation:.2f}x**",
        f"- predicted sd off-support / in-sample (at the training runs) = "
        f"**{hev['overall']['mean_predicted_sd']/hev['in_sample_mean_sd']:.2f}x**",
        f"- distance to nearest training run: off-design mean {hev['overall']['mean_nn_d2']:.4f} "
        f"squared Hellinger vs in-distribution median {ref['median_nn_d2']:.4f} "
        f"(**{hev['overall']['mean_nn_d2']/ref['median_nn_d2']:.1f}x** further out; "
        f"{hev['overall']['mean_nn_hellinger']:.3f} vs {ref['median_nn_hellinger']:.3f} in "
        f"Hellinger units). These runs really are off-support.",
        f"- spearman(predicted sd, |miss|) across the off-design runs = "
        f"{hev['spearman_sd_abs_miss']:+.3f}: the sd does carry some rank signal off-support "
        "(the extreme two-bucket weights are both furthest out and worst), but the SCALE is "
        "hopeless -- see the next line.",
        f"- the sd would have to be scaled by **{hev['sd_scale_needed_for_95pct_coverage']:.0f}x** for "
        "the 2sd interval to actually cover 95% of these runs.",
        f"- and widening would be the wrong fix anyway: {100*hev['overall']['frac_miss_positive']:.0f}% of the "
        "misses are POSITIVE (the surrogate is optimistic every time). This is a one-sided BIAS the "
        "kernel cannot represent, not extra variance.",
        "",
        "### The sd cannot see epoching at all",
        "",
        f"- the {hev['epoching_blindness']['n_runs']} probe runs collapse to only "
        f"{hev['epoching_blindness']['n_distinct_posteriors']} distinct posteriors: runs that share a "
        "mixture but repeat the sliced bucket a different number of times are the SAME point to the "
        "content kernel, so they receive an identical mean and an identical sd.",
        _blindness_line(hev),
        "",
        "## Off-design runs, gsm8k bpb (mathgen groups; they have no humaneval eval)",
        "",
        *_group_table(gsm, GSM8K_GROUPS),
        "",
        "## Where the intervals DO hold",
        "",
        "- transect (mean |z| 1.3, 86% inside 2sd) and harm100b (mean |z| 1.0, 67%) are the two groups "
        "whose realized humaneval bpb stays near the anchor: transect perturbs only phase 0 around the "
        "anchor mixture, and harm100b runs at 100B where the epoch harm has vanished. So the intervals "
        "are not vacuous -- they hold wherever the kernel's own content assumption holds.",
        "- the failure is specific: it is the epoch-repetition axis (and extreme single-bucket weights), "
        "exactly the physics the content features do not encode.",
        "",
        "## Caveats",
        "",
        "- twobucket's a3 arm varies the token budget (2.5B/40B) and its a4 arm uses a d256 model, so part "
        "of those runs' miss is a budget/size effect rather than a mixture-surrogate error. This does not "
        "rescue the verdict: epochrep is single-budget, single-architecture, matched to the swarm, and is "
        "the WORST group (mean |z| 16.6, 0/18 covered).",
        "- the in-distribution reference refits hyperparameters inside each of 5 folds (640 training runs), "
        "while the off-design predictions come from a single fit on all 800; the in-sample sd at the 800 "
        "training locations is reported alongside as the same-fit comparison.",
        "",
        "## Reconstruction check",
        "",
        f"- {out['reconstruction_check']['n_runs_checked']} probe runs reproduce their "
        f"pre-registered frozen-kernel humaneval prediction to "
        f"{out['reconstruction_check']['max_abs_diff_vs_prereg_kernel']:.2e} bpb",
        "",
        "## Worst individual runs (humaneval)",
        "",
        "| run | realized | GP mean | sd | z | NN dist |",
        "|---|---|---|---|---|---|",
    ]
    worst = sorted(hev["per_run"], key=lambda r: -abs(r["z"]))[:10]
    for r in worst:
        lines.append(
            f"| {r['run_id']} | {r['realized']:.4f} | {r['gp_mean']:.4f} | {r['gp_sd']:.4f} | "
            f"{r['z']:+.1f} | {r['nn_hellinger']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def build_verdict(hev: dict, ref: dict) -> tuple[str, str]:
    o = hev["overall"]
    inflation = o["mean_predicted_sd"] / ref["mean_predicted_sd"]
    covered = o["coverage_2sd"] >= 0.8 and o["mean_abs_z"] < 2.0
    if covered:
        verdict = "COVERED -- the posterior sd inflates enough off-support to contain the known misses"
    else:
        verdict = "OVERCONFIDENT -- the posterior sd does NOT inflate enough to cover the known misses"
    worst = max(hev["per_group"].items(), key=lambda kv: kv[1]["mean_abs_z"])
    best = min(hev["per_group"].items(), key=lambda kv: kv[1]["mean_abs_z"])
    detail = (
        f"Off-design mean |z| = {o['mean_abs_z']:.1f} (max {o['max_abs_z']:.1f}); "
        f"{100*o['coverage_2sd']:.0f}% of probe runs fall inside their 2sd credible interval "
        f"(nominal 95%). The predicted sd grows only {inflation:.2f}x going from the swarm's "
        f"support out to runs {o['mean_nn_d2']/ref['median_nn_d2']:.1f}x further "
        f"from the nearest training run (squared Hellinger), while the mean absolute error grows to "
        f"{o['mean_abs_miss']:.3f} bpb (vs an in-distribution CV RMSE of {ref['cv_rmse']:.4f}). "
        f"The sd is also uninformative about WHICH off-support run fails: {worst[0]} "
        f"(mean |z| {worst[1]['mean_abs_z']:.1f}) and {best[0]} (mean |z| {best[1]['mean_abs_z']:.1f}) "
        f"sit at nearly the same distance from the training support "
        f"({worst[1]['mean_nn_hellinger']:.3f} vs {best[1]['mean_nn_hellinger']:.3f} Hellinger) and get "
        f"nearly the same sd ({worst[1]['mean_predicted_sd']:.4f} vs "
        f"{best[1]['mean_predicted_sd']:.4f} bpb)."
    )
    return verdict, detail


def main() -> None:
    design = load_design()
    records = offdesign_runs(design)
    check = verify_reconstruction(design, records)

    ref = indistribution_reference(design["d2_train"], design["y"][HUMANEVAL])
    hev = score_offdesign(design, records, HUMANEVAL, HUMANEVAL_GROUPS)
    gsm = score_offdesign(design, records, GSM8K, GSM8K_GROUPS)
    verdict, detail = build_verdict(hev, ref)

    make_figure(hev, ref, FIG_PATH)
    ref_public = {k: v for k, v in ref.items() if not k.startswith("_")}
    out = {
        "question": (
            "Do the GP surrogate's credible intervals cover its KNOWN off-design misses, or is it "
            "overconfident exactly where it is most wrong?"
        ),
        "verdict": verdict,
        "verdict_detail": detail,
        "reconstruction_check": check,
        "in_distribution_humaneval": ref_public,
        "sd_inflation_ood_over_indist": hev["overall"]["mean_predicted_sd"] / ref["mean_predicted_sd"],
        "sd_inflation_ood_over_insample": hev["overall"]["mean_predicted_sd"] / hev["in_sample_mean_sd"],
        "distance_ratio_ood_over_indist_median_d2": hev["overall"]["mean_nn_d2"] / ref["median_nn_d2"],
        "distance_ratio_ood_over_indist_median_hellinger": (
            hev["overall"]["mean_nn_hellinger"] / ref["median_nn_hellinger"]
        ),
        "humaneval": hev,
        "gsm8k": gsm,
        "figure": str(FIG_PATH),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    write_report(out, OUT_MD)

    print(f"reconstruction check: {check}")
    print(f"in-distribution: mean sd {ref['mean_predicted_sd']:.4f}, RMS z {ref['rms_z']:.3f}")
    for g in HUMANEVAL_GROUPS:
        s = hev["per_group"][g]
        print(
            f"  {g:10s} n={s['n']:2d} miss {s['mean_miss']:+.3f} sd {s['mean_predicted_sd']:.4f} "
            f"|z| {s['mean_abs_z']:6.1f} cov2sd {100*s['coverage_2sd']:5.1f}% nn {s['mean_nn_hellinger']:.3f}"
        )
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT_JSON}, {OUT_MD}, {FIG_PATH}")


if __name__ == "__main__":
    main()
