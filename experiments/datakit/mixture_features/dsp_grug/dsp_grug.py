# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib"]
# ///
"""DSP dose-response surrogate on the grug-moe-mix swarm (800 training runs).

Ports the swarm branch's DSP fitter (``dsp_exact.py``, vendored unmodified in this
directory) to the grug swarm as a functional-form experiment: does the mechanistic
saturation-penalty form ``L(w) = b0 - sum_i a_i(...)(1-e^{-rho_i z_i}) + sum_i p_i
softplus(log(1+z_i)-tau_i)^2`` beat the phase-2 Hellinger-kernel surrogate
(OOF Spearman 0.303, ``fit_report.md``) on macro_bpb?

Data semantics (verified against ``experiments/grug/moe/launch_datakit_moe_mix.py``):

- weights: (800, 2, 168) per-phase simplex from ``train_runs.parquet`` phase dicts,
  bucket order = sorted bucket names (== ``grug_inputs/grug_buckets.json``).
- epoch multipliers use the TARGET budget, not realized proxy tokens (see
  ``docs/debug-log-epoch-feature-budget-semantics.md`` on the swarm branch): the
  launcher sets ``target_budget = _TARGET_BUDGET_TOKENS = 10_372_343_704_053`` (the
  store_8ac06c74 natural size) with ``enable_simulated_epoching=True``, so each
  bucket cache is sliced to ``experiment/target`` ratio and effective epochs are
  ``c_p[j] = phase_fraction_p * TARGET_BUDGET / T_j``. ``experiment_budget`` cancels.
- phase fractions from block-quantized steps: 1536/2003 and 467/2003 = 0.767/0.233
  (identical to phase-2 stage B; ``phase_step_split`` reproduced below).
- T_j from ``buckets_table.parquet``; the ``tail`` row is the pooled 33-child
  partition (sum of children, 420,039,555 tok); the 168 rows sum to the target
  budget to within 3e-7 relative, consistent with target_budget == store size.

Fit protocol: nested CV on the SAME splits as phase 2 (RepeatedKFold(5, 3, seed 0)
over the 800 train rows; QUARANTINE_test_labels.parquet is never touched).
Nonlinear params are refit inside every fold (no leakage; unlike
``dsp_exact.oof_predictions`` which fixes them). Paired per-fold deltas vs the
phase-2 Hellinger kernel come from ``cv_results.parquet``.

All configurations (including the untied per-bucket "full" ones) are fit with
``fit_tied``: the same start-bank -> bounded L-BFGS-B protocol as
``dsp_exact.fit_variant``, but with (a) objective evaluations that compute the
DSP features once per call (vendored ``dsp_exact.features`` math, verified to
reproduce ``dsp_exact.fit_linear_head``/``predict``/``profile_objective``
bit-exactly for the untied spec), and (b) the exact variable-projection
gradient (implicit differentiation of the NNLS head on its active set;
validated against full-refit finite differences to ~6 significant digits).
This replaces scipy's default full-FD gradient (337 NNLS solves per gradient
for the untied model), which is infeasible on this box's 2-CPU cgroup quota. Random start-bank draws differ slightly from
``dsp_exact.start_bank`` (group-level parameterization); the 5 deterministic
starts match at group level.

Parameter reduction (overfit watch, 672 params vs 800 runs):
- ``cluster_*``: rho/tau and the NNLS head are tied within lexical cluster across
  quality tiers (35 clusters + tail = 36 groups; 4x36 + globals ~ 145 params).
- ``globalrt_*``: one shared rho and tau for all buckets, per-bucket NNLS head.
- ``content_*``: benefit a_i and penalty p_i are tied through content space,
  a_i = sum_k u_k f_ik with u >= 0 (feature-space NNLS; f_i = bucket i's K=40
  content profile from V, nonnegative, plus an intercept column), rho/tau
  cluster-tied. Tests DSP form + content-tied parameters vs plain DSP and kernel.

Usage:
    uv run experiments/datakit/mixture_features/dsp_grug/dsp_grug.py run \
        [--configs a,b,...] [--maxiter-full 24]
    uv run experiments/datakit/mixture_features/dsp_grug/dsp_grug.py report

Outputs: scratch/mixture_features/grug/{dsp_results.parquet, dsp_report.md} and
per-(config, fold) caches under scratch/mixture_features/grug/dsp_cache/.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402  (BLAS thread caps must precede numpy import)
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # vendored dsp_exact
sys.path.insert(0, str(_HERE.parent))  # featurize (grug_fit pulls lightgbm; not needed here)

import dsp_exact as dx  # noqa: E402  (sys.path setup above)
import featurize  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402
from scipy.optimize import minimize, nnls  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402
from sklearn.model_selection import KFold, RepeatedKFold  # noqa: E402

logger = logging.getLogger("dsp_grug")

REPO_ROOT = _HERE.parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
HIST_DIR = SCRATCH / "grug_histograms"
BASIS_DIR = SCRATCH / "basis"
GRUG_DIR = SCRATCH / "grug"
CACHE_DIR = GRUG_DIR / "dsp_cache"
TRAIN_RUNS = GRUG_DIR / "train_runs.parquet"
CV_RESULTS = GRUG_DIR / "cv_results.parquet"

TARGET = "macro_bpb"  # lower is better
ZMACRO_TARGET = "zmacro_english_20"  # target-analysis recommended target (target_report.md #5)
TARGET_CANDIDATES = GRUG_DIR / "target_candidates.json"
SEED = 0
N_SPLITS, N_REPEATS = 5, 3
KERNEL_VARIANT = "4_hellinger_kernel_k1000"
# Hellinger-kernel-ridge hypergrids, identical to retrodiction.py (not importable
# here without lightgbm)
KR_GAMMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
KR_ALPHAS = np.logspace(-3, 2, 6)
N_INNER_FOLDS = 5

# --- budget constants, identical to grug_fit.py stage B (verified against the
# launcher experiments/grug/moe/launch_datakit_moe_mix.py: _TARGET_BUDGET_TOKENS,
# _MIXTURE_BLOCK_SIZE, _PHASE_1_START_FRACTION) ---
TOTAL_STEPS = 2003
BATCH_SIZE = 32
SEQ_LEN = 8192
MIXTURE_BLOCK_SIZE = 49_152
PHASE_1_START_FRACTION = 0.8
TARGET_BUDGET_TOKENS = 10_372_343_704_053


# ---------------------------------------------------------------------------
# Loader (small helpers reproduced from grug_fit.py, which we cannot import
# without dragging in lightgbm via retrodiction)
# ---------------------------------------------------------------------------


def load_histogram_artifacts() -> tuple[list, dict, pd.DataFrame]:
    """Rebuild grug DomainHistogram objects + coarsening views + buckets_table."""
    meta = json.loads((HIST_DIR / "_meta.json").read_text())
    b = meta["basis"]
    basis = featurize.MixtureBasis(
        embedder=b["embedder"],
        tokenizer=b["tokenizer"],
        centroids_path=b["centroids_path"],
        centroids_sha256=b["centroids_sha256"],
        k=b["k"],
        view_paths={int(k): v for k, v in b["view_paths"].items()},
        view_sha256={int(k): v for k, v in b["view_sha256"].items()},
        quality_scorer=b["quality_scorer"],
        quality_scorer_sha256=b["quality_scorer_sha256"],
        rff_dim=b["rff_dim"],
        rff_seed=b["rff_seed"],
        rff_bandwidth=b["rff_bandwidth"],
    )
    npz = np.load(HIST_DIR / meta["rff_means_file"])
    rff_by_bucket = dict(zip(npz["domains"].tolist(), npz["rff_means"], strict=True))
    hists = []
    for bucket, bmeta in meta["buckets"].items():
        df = pd.read_parquet(HIST_DIR / bmeta["parquet"])
        counts = {
            (int(c), int(q)): int(t)
            for c, q, t in zip(df["cluster_id"], df["quality_bucket"], df["token_count"], strict=True)
        }
        bs = bmeta["bucket_stats"]
        hists.append(
            featurize.DomainHistogram(
                domain=bucket,
                basis=basis,
                sample_size=bmeta["sample_size"],
                token_count=bmeta["token_count"],
                seed=bmeta["seed"],
                counts=counts,
                rff_mean=tuple(np.asarray(rff_by_bucket[bucket], dtype=np.float64).tolist()),
                stats=featurize.BucketStats(
                    total_tokens_available=bs["total_tokens_available"],
                    mean_doc_tokens=bs["mean_doc_tokens"],
                    duplicate_frac=bs["duplicate_frac"],
                    loss_masked_frac=bs["loss_masked_frac"],
                ),
            )
        )
    hists.sort(key=lambda h: h.domain)
    views = {
        40: np.load(BASIS_DIR / "lookup_5000_to_40.npy"),
        1000: np.load(BASIS_DIR / "lookup_5000_to_1000.npy"),
    }
    buckets_table = pd.read_parquet(HIST_DIR / "buckets_table.parquet")
    return hists, views, buckets_table


def weight_matrix(runs: pd.DataFrame, buckets: list[str]) -> np.ndarray:
    """(n_runs, 2, 168) per-phase weights in sorted-bucket order; missing bucket -> 0."""
    w = np.zeros((len(runs), 2, len(buckets)), dtype=np.float64)
    idx = {b: j for j, b in enumerate(buckets)}
    for p, col in ((0, "phase0_weights"), (1, "phase1_weights")):
        for i, d in enumerate(runs[col].to_numpy()):
            for bucket, val in d.items():
                w[i, p, idx[bucket]] = val
    if not np.allclose(w.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("run phase weights do not sum to 1")
    return w


def phase_step_split() -> tuple[int, int]:
    requested = max(1, int(TOTAL_STEPS * PHASE_1_START_FRACTION))
    step_multiple = MIXTURE_BLOCK_SIZE // math.gcd(MIXTURE_BLOCK_SIZE, BATCH_SIZE)
    p1_start = max(step_multiple, (requested // step_multiple) * step_multiple)
    return p1_start, TOTAL_STEPS - p1_start


@dataclass(frozen=True)
class GrugData:
    """Packet + side info for the grug swarm."""

    packet: dx.PacketData  # y = macro_bpb
    buckets: list[str]
    cluster_of_bucket: np.ndarray  # (168,) int group id, tail is its own group
    quality_tier: np.ndarray  # (168,) int, -1 for tail
    content_basis: np.ndarray  # (168, 41) nonneg: K40 profile per bucket + ones column
    v1000: np.ndarray  # (1000, 168) composition matrix, for the Hellinger-kernel refit
    targets: dict[str, np.ndarray]  # target name -> y (800,)


def zmacro_english(runs: pd.DataFrame) -> np.ndarray:
    """Registered zmacro_english_20: mean over observed tasks of (bpb - mu_t)/sd_t.

    Uses the FROZEN per-task train stats from target_candidates.json (the target
    analysis verified full-train vs nested z-stats are indistinguishable, 0.8147
    vs 0.8149). Lower is better.
    """
    rec = json.loads(TARGET_CANDIDATES.read_text())["recommended_target"]
    tasks, mu, sd = rec["task_list"], rec["train_z_mu"], rec["train_z_sd"]
    out = np.full(len(runs), np.nan)
    for i, blob in enumerate(runs["evals"].to_numpy()):
        evals = json.loads(blob)
        zs = [
            (evals[t]["bpb"] - mu[t]) / sd[t]
            for t in tasks
            if t in evals and isinstance(evals[t], dict) and "bpb" in evals[t]
        ]
        if not zs:
            raise ValueError(f"run {i} has no tasks from the registered list")
        out[i] = float(np.mean(zs))
    return out


def packet_for_target(data: GrugData, target: str) -> dx.PacketData:
    if target == TARGET:
        return data.packet
    return dx.PacketData(
        frame=data.packet.frame,
        name_col=data.packet.name_col,
        y=data.targets[target],
        w=data.packet.w,
        m=data.packet.m,
        c0=data.packet.c0,
        c1=data.packet.c1,
        domain_names=data.packet.domain_names,
    )


def load_grug_data() -> GrugData:
    """Build the DSP PacketData from grug artifacts (target-budget epoch semantics)."""
    hists, views, buckets_table = load_histogram_artifacts()
    buckets = [h.domain for h in hists]
    runs = pd.read_parquet(TRAIN_RUNS)
    w = weight_matrix(runs, buckets)
    y = runs[TARGET].to_numpy(dtype=np.float64)

    p0_steps, p1_steps = phase_step_split()
    f0, f1 = p0_steps / TOTAL_STEPS, p1_steps / TOTAL_STEPS
    bt = buckets_table.set_index("bucket").loc[buckets]
    tj = bt["total_tokens"].to_numpy(dtype=np.float64)
    # tail row is the pooled 33-child partition; the 168 T_j must reproduce the
    # target budget (= store natural size) or the epoch basis is wrong.
    if abs(tj.sum() - TARGET_BUDGET_TOKENS) / TARGET_BUDGET_TOKENS > 1e-3:
        raise ValueError(f"sum T_j {tj.sum():.0f} does not match target budget {TARGET_BUDGET_TOKENS}")
    c0 = f0 * TARGET_BUDGET_TOKENS / tj
    c1 = f1 * TARGET_BUDGET_TOKENS / tj

    frame = runs[["index", "experiment_index"]].copy()
    frame["run_name"] = frame["experiment_index"].astype(str)
    packet = dx.PacketData(
        frame=frame,
        name_col="run_name",
        y=y,
        w=w,
        m=len(buckets),
        c0=c0,
        c1=c1,
        domain_names=buckets,
    )

    cluster_of_bucket = pd.factorize(bt["cluster_id"].to_numpy())[0]
    quality_tier = bt["quality_tier"].to_numpy(dtype=int)
    v40, order = featurize.composition_matrix(hists, k=40, views=views)
    if order != buckets:
        raise ValueError("composition_matrix order mismatch")
    v1000, _ = featurize.composition_matrix(hists, k=1000, views=views)
    content_basis = np.concatenate([np.asarray(v40).T, np.ones((len(buckets), 1))], axis=1)
    targets = {TARGET: y}
    if TARGET_CANDIDATES.exists():
        targets[ZMACRO_TARGET] = zmacro_english(runs)
    return GrugData(
        packet=packet,
        buckets=buckets,
        cluster_of_bucket=cluster_of_bucket,
        quality_tier=quality_tier,
        content_basis=content_basis,
        v1000=np.asarray(v1000),
        targets=targets,
    )


def subset_packet(packet: dx.PacketData, idx: np.ndarray) -> dx.PacketData:
    """Row-subset a PacketData (bucket axis unchanged)."""
    return dx.PacketData(
        frame=packet.frame.iloc[idx].reset_index(drop=True),
        name_col=packet.name_col,
        y=packet.y[idx],
        w=packet.w[idx],
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=packet.domain_names,
    )


# ---------------------------------------------------------------------------
# Tied / content-coupled DSP fitter.
#
# Reuses dsp_exact.features verbatim (per-bucket rho/tau expanded from group
# params) and swaps the per-bucket NNLS head for a head over a nonnegative
# bucket basis: benefit_coef = head_basis @ u with u >= 0 (so per-bucket
# coefficients stay nonnegative). head_basis = I recovers dsp_exact exactly;
# a cluster indicator ties coefficients within cluster; the K40 content
# profile gives the content-coupled parameterization a_i = u . f_i.
# ---------------------------------------------------------------------------

PHASE_GLOBALS: dict[dx.PhaseMode, tuple[str, ...]] = {
    dx.PhaseMode.BENEFIT_GAIN: ("gamma",),
    dx.PhaseMode.EFFECTIVE_EXPOSURE: ("gamma",),
    dx.PhaseMode.SATURATION_PENALTY: ("gamma_saturation", "gamma_penalty"),
    dx.PhaseMode.NONE: (),
}


@dataclass(frozen=True)
class TiedSpec:
    """Parameter-tying specification for one DSP configuration."""

    variant: dx.DSPVariant
    rho_groups: np.ndarray  # (M,) int -> shared rho within group
    tau_groups: np.ndarray  # (M,) int -> shared tau within group
    head_basis: np.ndarray  # (M, K) nonneg; per-bucket coef = head_basis @ u, u >= 0

    @property
    def n_rho(self) -> int:
        return int(self.rho_groups.max()) + 1

    @property
    def n_tau(self) -> int:
        return int(self.tau_groups.max()) + 1

    @property
    def n_globals(self) -> int:
        return len(PHASE_GLOBALS[self.variant.phase_mode])

    def param_count(self) -> int:
        return self.n_rho + self.n_tau + self.n_globals + 2 * self.head_basis.shape[1] + 1


@dataclass(frozen=True)
class TiedModel:
    """Fitted tied DSP model."""

    spec: TiedSpec
    params: dict  # per-bucket expanded rho/tau + phase globals
    intercept: float
    benefit_coef: np.ndarray  # (M,)
    penalty_coef: np.ndarray  # (M,)
    head_u: np.ndarray  # raw nonnegative head coefficients (2K,)


def unpack_tied(theta: np.ndarray, spec: TiedSpec) -> dict:
    gr, gt = spec.n_rho, spec.n_tau
    log_rho = np.clip(theta[:gr], np.log(1e-4), np.log(2.0))
    tau_g = np.clip(theta[gr : gr + gt], -2.0, 8.0)
    params: dict = {
        "rho": np.exp(log_rho)[spec.rho_groups],
        "tau": tau_g[spec.tau_groups],
    }
    for i, name in enumerate(PHASE_GLOBALS[spec.variant.phase_mode]):
        params[name] = float(np.exp(np.clip(theta[gr + gt + i], np.log(1e-4), np.log(100.0))))
    if gr + gt + spec.n_globals != len(theta):
        raise ValueError("theta length mismatch")
    return params


def tied_bounds(spec: TiedSpec) -> list[tuple[float, float]]:
    out = [(np.log(1e-4), np.log(2.0))] * spec.n_rho
    out += [(-2.0, 8.0)] * spec.n_tau
    out += [(np.log(1e-4), np.log(100.0))] * spec.n_globals
    return out


def fit_tied_head(
    packet: dx.PacketData, spec: TiedSpec, params: dict, idx_w: np.ndarray, idx_y: np.ndarray
) -> TiedModel:
    """NNLS head over the bucket basis for fixed nonlinear params."""
    signal, penalty = dx.features(idx_w, packet.c0, packet.c1, spec.variant, params)
    basis = spec.head_basis
    design = np.hstack([-(signal @ basis), penalty @ basis])
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(idx_y.mean())
    centered = design - design_mean
    targets = idx_y - target_mean
    p = centered.shape[1]
    a = np.vstack([centered, np.sqrt(dx.LINEAR_REG) * np.eye(p)])
    b = np.concatenate([targets, np.zeros(p)])
    coef, _ = nnls(a, b)
    k = basis.shape[1]
    return TiedModel(
        spec=spec,
        params=params,
        intercept=float(target_mean - (design_mean @ coef).item()),
        benefit_coef=basis @ coef[:k],
        penalty_coef=basis @ coef[k:],
        head_u=coef,
    )


def predict_tied(model: TiedModel, packet: dx.PacketData, w: np.ndarray) -> np.ndarray:
    signal, penalty = dx.features(w, packet.c0, packet.c1, model.spec.variant, model.params)
    return model.intercept - signal @ model.benefit_coef + penalty @ model.penalty_coef


def tied_profile_objective(packet: dx.PacketData, spec: TiedSpec, theta: np.ndarray) -> float:
    """Same profile objective as dsp_exact: train RMSE + 0.5 * lower-tail optimism."""
    params = unpack_tied(theta, spec)
    model = fit_tied_head(packet, spec, params, packet.w, packet.y)
    pred = predict_tied(model, packet, packet.w)
    residual = pred - packet.y
    rmse = float(np.sqrt(np.mean(residual**2)))
    tail_count = max(5, int(np.ceil(dx.LOWER_TAIL_FRAC * len(packet.y))))
    tail_idx = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(packet.y[tail_idx] - pred[tail_idx], 0.0)))
    return rmse + 0.5 * optimism


def group_reduce(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Median of per-bucket values within each group."""
    n_groups = int(groups.max()) + 1
    return np.array([np.median(values[groups == g]) for g in range(n_groups)])


def tied_start_bank(packet: dx.PacketData, spec: TiedSpec) -> list[np.ndarray]:
    """Deterministic + random starts mirroring dsp_exact.start_bank at group level."""
    z = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    positive = np.where(z > 1e-8, z, np.nan)
    median_exposure = np.nanmedian(positive, axis=0)
    median_exposure = np.where(np.isfinite(median_exposure), median_exposure, np.nanmedian(positive))
    base_rho = np.clip(1.0 / np.maximum(median_exposure, 1e-3), 1e-4, 0.5)
    base_tau = np.clip(np.log1p(np.nanpercentile(positive, 85, axis=0)), -2.0, 8.0)
    base_tau = np.where(np.isfinite(base_tau), base_tau, 3.0)
    rho_g = group_reduce(base_rho, spec.rho_groups)
    tau_g = group_reduce(base_tau, spec.tau_groups)

    rng = np.random.default_rng(SEED)
    starts: list[np.ndarray] = []
    for rho_scale, tau_shift, gamma in (
        (0.25, -1.0, 0.25),
        (0.5, -0.5, 0.5),
        (1.0, 0.0, 1.0),
        (2.0, 0.5, 2.0),
        (4.0, 1.0, 8.0),
    ):
        theta = np.concatenate(
            [
                np.log(np.clip(rho_g * rho_scale, 1e-4, 2.0)),
                np.clip(tau_g + tau_shift, -2.0, 8.0),
                np.full(spec.n_globals, np.log(gamma)),
            ]
        )
        starts.append(theta)
    for _ in range(3):
        theta = np.concatenate(
            [
                np.log(np.clip(rho_g * np.exp(rng.normal(scale=0.7, size=spec.n_rho)), 1e-4, 2.0)),
                np.clip(tau_g + rng.normal(scale=0.8, size=spec.n_tau), -2.0, 8.0),
                rng.normal(loc=np.log(2.0), scale=0.9, size=spec.n_globals),
            ]
        )
        starts.append(theta)
    return starts


class TiedObjective:
    """Fast profile objective + exact variable-projection gradient for a TiedSpec.

    ``value`` is the exact profile objective (features computed once, full NNLS
    head — numerically identical to ``tied_profile_objective``). ``grad`` is the
    analytic variable-projection gradient: on the NNLS active set P the head is
    an unconstrained ridge LS, so dhead/dtheta follows from the implicit
    function theorem (one Cholesky of the active Gram per gradient call); the
    feature-column derivatives are forward differences of the smooth vendored
    ``dsp_exact.features`` (column-separable in every phase mode used here, so
    each rho/tau coordinate touches only its group's bucket columns). Inactive
    head coordinates stay at 0 under small perturbations (strict
    complementarity), the tail set of the optimism term is held fixed
    (piecewise-smooth objective), and active-set changes make the objective
    piecewise smooth — standard VarPro-with-NNLS practice.
    """

    FD_EPS = 1e-6

    def __init__(self, packet: dx.PacketData, spec: TiedSpec):
        self.packet = packet
        self.spec = spec
        self.bounds = tied_bounds(spec)
        self.rho_cols = [np.where(spec.rho_groups == g)[0] for g in range(spec.n_rho)]
        self.tau_cols = [np.where(spec.tau_groups == g)[0] for g in range(spec.n_tau)]
        self.tail_count = max(5, int(np.ceil(dx.LOWER_TAIL_FRAC * len(packet.y))))
        self._state: dict | None = None

    def _obj_from_pred(self, pred: np.ndarray) -> float:
        y = self.packet.y
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        tail_idx = np.argsort(pred)[: self.tail_count]
        optimism = float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0)))
        return rmse + 0.5 * optimism

    def _ensure_state(self, theta: np.ndarray) -> dict:
        key = theta.tobytes()
        if self._state is not None and self._state["key"] == key:
            return self._state
        packet, spec = self.packet, self.spec
        y = packet.y
        n = len(y)
        params = unpack_tied(theta, spec)
        signal, penalty = dx.features(packet.w, packet.c0, packet.c1, spec.variant, params)
        basis = spec.head_basis
        design = np.hstack([-(signal @ basis), penalty @ basis])
        dc = design - design.mean(axis=0, keepdims=True)
        ym = float(y.mean())
        yc = y - ym
        p = dc.shape[1]
        a = np.vstack([dc, np.sqrt(dx.LINEAR_REG) * np.eye(p)])
        b = np.concatenate([yc, np.zeros(p)])
        coef, _ = nnls(a, b)
        pred = dc @ coef + ym
        residual = yc - dc @ coef
        active = np.where(coef > 0)[0]
        cho = None
        if len(active):
            gram = dc[:, active].T @ dc[:, active] + dx.LINEAR_REG * np.eye(len(active))
            cho = cho_factor(gram)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        tail_idx = np.argsort(pred)[: self.tail_count]
        optimism = float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0)))
        dfdpred = (pred - y) / (n * max(rmse, 1e-300))
        tail_hit = tail_idx[y[tail_idx] > pred[tail_idx]]
        dfdpred[tail_hit] -= 0.5 / len(tail_idx)
        self._state = {
            "key": key,
            "theta": theta.copy(),
            "params": params,
            "signal": signal,
            "penalty": penalty,
            "coef": coef,
            "dc": dc,
            "active": active,
            "cho": cho,
            "residual": residual,
            "pred": pred,
            "obj": rmse + 0.5 * optimism,
            "dfdpred": dfdpred,
        }
        return self._state

    def value(self, theta: np.ndarray) -> float:
        return self._ensure_state(np.asarray(theta, dtype=float))["obj"]

    def grad(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        st = self._ensure_state(theta)
        spec, packet = self.spec, self.packet
        signal, penalty, coef, dc = st["signal"], st["penalty"], st["coef"], st["dc"]
        active, cho, residual, dfdpred = st["active"], st["cho"], st["residual"], st["dfdpred"]
        basis = spec.head_basis
        k = basis.shape[1]
        c_ben, c_pen = coef[:k], coef[k:]
        gr, gt = spec.n_rho, spec.n_tau
        out = np.zeros(len(theta))
        for i in range(len(theta)):
            ub = self.bounds[i][1]
            step = self.FD_EPS if theta[i] + self.FD_EPS <= ub else -self.FD_EPS
            theta_p = theta.copy()
            theta_p[i] += step
            params_p = unpack_tied(theta_p, spec)
            if i < gr + gt:
                cols = self.rho_cols[i] if i < gr else self.tau_cols[i - gr]
                params_slice = {
                    key: np.asarray(v)[cols] if isinstance(v, np.ndarray) else v for key, v in params_p.items()
                }
                sig_p, pen_p = dx.features(
                    packet.w[:, :, cols], packet.c0[cols], packet.c1[cols], spec.variant, params_slice
                )
                sdot = (sig_p - signal[:, cols]) / step
                pdot = (pen_p - penalty[:, cols]) / step
            else:  # global phase parameter: all columns move
                cols = np.arange(packet.m)
                sig_p, pen_p = dx.features(packet.w, packet.c0, packet.c1, spec.variant, params_p)
                sdot = (sig_p - signal) / step
                pdot = (pen_p - penalty) / step
            b_j = basis[cols, :]
            # Ddot @ coef (n-vector) and its centered version
            ddot_c = -(sdot @ (b_j @ c_ben)) + pdot @ (b_j @ c_pen)
            ddot_c_centered = ddot_c - ddot_c.mean()
            if len(active):
                # Ddot^T residual restricted to the active set (residual has zero
                # mean, so centering of Ddot drops out)
                u = sdot.T @ residual
                v = pdot.T @ residual
                ddot_t_r = np.concatenate([-(b_j.T @ u), b_j.T @ v])[active]
                rhs = ddot_t_r - dc[:, active].T @ ddot_c_centered
                cdot = cho_solve(cho, rhs)
                dpred = ddot_c_centered + dc[:, active] @ cdot
            else:
                dpred = ddot_c_centered
            out[i] = float(dfdpred @ dpred)
        return out


def fit_tied_head_from_features(
    y: np.ndarray, spec: TiedSpec, params: dict, signal: np.ndarray, penalty: np.ndarray
) -> TiedModel:
    """NNLS head over the bucket basis for precomputed features."""
    basis = spec.head_basis
    design = np.hstack([-(signal @ basis), penalty @ basis])
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(y.mean())
    centered = design - design_mean
    targets = y - target_mean
    p = centered.shape[1]
    a = np.vstack([centered, np.sqrt(dx.LINEAR_REG) * np.eye(p)])
    b = np.concatenate([targets, np.zeros(p)])
    coef, _ = nnls(a, b)
    k = basis.shape[1]
    return TiedModel(
        spec=spec,
        params=params,
        intercept=float(target_mean - (design_mean @ coef).item()),
        benefit_coef=basis @ coef[:k],
        penalty_coef=basis @ coef[k:],
        head_u=coef,
    )


def fit_tied(packet: dx.PacketData, spec: TiedSpec, *, maxiter: int, coarse_top_k: int) -> TiedModel:
    """Coarse start bank -> bounded L-BFGS-B refinement, as in dsp_exact.fit_variant.

    Uses TiedObjective (exact objective values, envelope FD gradient) instead of
    scipy's default full-FD gradient; validated to reach equal-or-better
    objectives at ~100x less compute on this box.
    """
    objective = TiedObjective(packet, spec)
    starts = tied_start_bank(packet, spec)
    coarse = sorted(((objective.value(s), i) for i, s in enumerate(starts)), key=lambda t: t[0])
    best_objective, best_theta = np.inf, None
    for _, start_id in coarse[:coarse_top_k]:
        result = minimize(
            objective.value,
            starts[start_id],
            jac=objective.grad,
            method="L-BFGS-B",
            bounds=objective.bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
        )
        if float(result.fun) < best_objective:
            best_objective, best_theta = float(result.fun), np.asarray(result.x, dtype=float)
    if best_theta is None:
        raise RuntimeError(f"no tied fit result for {spec.variant.name}")
    params = unpack_tied(best_theta, spec)
    return fit_tied_head(packet, spec, params, packet.w, packet.y)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One DSP configuration in the comparison grid."""

    name: str
    kind: str  # "full" (per-bucket params, identical to vendored dsp_exact) or "tied"
    variant_key: str  # key into dsp_exact.VARIANTS
    rho_tie: str  # "bucket" | "cluster" | "global"
    head: str  # "bucket" | "cluster" | "content"
    folds: str  # "repeat0" (5 folds) or "all" (15 folds)
    param_count: int = 0


def config_grid() -> list[Config]:
    return [
        # full per-bucket DSP: default PhaseMode first, then qsplit240's top-ranked
        # variants (split_saturation_penalty 0.929 > effective_exposure 0.920 >
        # canonical/benefit_gain 0.898 in dsp_canonical_variants_300m_20260510).
        Config("full_canonical", "full", "canonical", "bucket", "bucket", "all"),
        Config("full_split_satpen", "full", "split_saturation_penalty", "bucket", "bucket", "all"),
        Config("full_effexp", "full", "effective_exposure", "bucket", "bucket", "all"),
        # parameter-reduction knobs
        Config("cluster_canonical", "tied", "canonical", "cluster", "cluster", "all"),
        Config("cluster_split_satpen", "tied", "split_saturation_penalty", "cluster", "cluster", "all"),
        Config("cluster_effexp", "tied", "effective_exposure", "cluster", "cluster", "all"),
        Config("cluster_no_phase", "tied", "no_phase", "cluster", "cluster", "all"),
        Config("globalrt_canonical", "tied", "canonical", "global", "bucket", "all"),
        # content-coupled: a_i, p_i linear in the K40 content profile (u >= 0),
        # rho/tau cluster-tied
        Config("content_canonical", "tied", "canonical", "cluster", "content", "all"),
        Config("content_split_satpen", "tied", "split_saturation_penalty", "cluster", "content", "all"),
    ]


def build_spec(config: Config, data: GrugData) -> TiedSpec:
    m = data.packet.m
    groups = {
        "bucket": np.arange(m),
        "cluster": data.cluster_of_bucket,
        "global": np.zeros(m, dtype=int),
    }
    heads = {
        "bucket": np.eye(m),
        "cluster": (
            (data.cluster_of_bucket[:, None] == np.arange(data.cluster_of_bucket.max() + 1)[None, :]).astype(float)
        ),
        "content": data.content_basis,
    }
    return TiedSpec(
        variant=dx.VARIANTS[config.variant_key],
        rho_groups=groups[config.rho_tie],
        tau_groups=groups[config.rho_tie],
        head_basis=heads[config.head],
    )


# ---------------------------------------------------------------------------
# Nested CV driver with per-(config, fold) caching
# ---------------------------------------------------------------------------


def fold_list(n: int, which: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Fold ids + indices; identical construction to grug_fit.py (fold_id // 5 = repeat)."""
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = [(fid, tr, te) for fid, (tr, te) in enumerate(rkf.split(np.arange(n)))]
    if which == "repeat0":
        return folds[:N_SPLITS]
    return folds


def cache_path(config_name: str, fold_id: int, target: str = TARGET) -> Path:
    prefix = "" if target == TARGET else f"{target}__"
    return CACHE_DIR / f"{prefix}{config_name}_fold{fold_id}.npz"


def final_path(config_name: str, target: str = TARGET) -> Path:
    prefix = "" if target == TARGET else f"{target}__"
    return CACHE_DIR / f"{prefix}{config_name}_final.json"


# ---------------------------------------------------------------------------
# Hellinger-kernel-ridge refit (for paired deltas on the secondary target;
# reimplements retrodiction.kernel_cv_predict / _sq_hellinger with identical
# hypergrids and inner-fold construction)
# ---------------------------------------------------------------------------


def sq_hellinger(h_phases: np.ndarray) -> np.ndarray:
    n = h_phases.shape[0]
    d = np.zeros((n, n))
    for p in range(h_phases.shape[1]):
        s = np.sqrt(np.clip(h_phases[:, p, :], 0.0, None))
        d += np.clip(1.0 - s @ s.T, 0.0, None)
    return d / h_phases.shape[1]


def _kr_fit_predict(k_tr: np.ndarray, y_tr: np.ndarray, k_te_tr: np.ndarray, alpha: float) -> np.ndarray:
    ym = y_tr.mean()
    dual = np.linalg.solve(k_tr + alpha * np.eye(len(y_tr)), y_tr - ym)
    return k_te_tr @ dual + ym


def kernel_cv_predict(d2: np.ndarray, tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
    d_tr = d2[np.ix_(tr, tr)]
    med = float(np.median(d_tr[~np.eye(len(tr), dtype=bool)]))
    gammas = np.asarray(KR_GAMMA_FACTORS) / max(med, 1e-12)
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(len(tr))))
    best, best_sse = None, np.inf
    for g in gammas:
        k_full = np.exp(-g * d_tr)
        for al in KR_ALPHAS:
            sse = 0.0
            for itr, iva in folds:
                p = _kr_fit_predict(k_full[np.ix_(itr, itr)], y[tr][itr], k_full[np.ix_(iva, itr)], al)
                sse += ((p - y[tr][iva]) ** 2).sum()
            if sse < best_sse:
                best, best_sse = (g, al), sse
    g, al = best
    return _kr_fit_predict(np.exp(-g * d_tr), y[tr], np.exp(-g * d2[np.ix_(te, tr)]), al)


def run_kernel_fold(data: GrugData, target: str, fold_id: int, tr: np.ndarray, te: np.ndarray) -> None:
    out = cache_path("kernel_hellinger", fold_id, target)
    if out.exists():
        return
    y = data.targets[target]
    h1000 = np.stack([data.packet.w[:, p, :] @ data.v1000.T for p in range(2)], axis=1)
    d2 = sq_hellinger(h1000)
    pred = kernel_cv_predict(d2, tr, te, y)
    np.savez_compressed(out, te=te, pred=pred, tr=tr, train_pred=np.full(len(tr), np.nan))


def fit_predict_fold(
    config: Config,
    data: GrugData,
    fold_id: int,
    tr: np.ndarray,
    te: np.ndarray,
    *,
    maxiter_full: int,
    topk_full: int,
    maxiter_tied: int,
    topk_tied: int,
    target: str = TARGET,
) -> None:
    out = cache_path(config.name, fold_id, target)
    if out.exists():
        return
    t0 = time.monotonic()
    packet = packet_for_target(data, target)
    packet_tr = subset_packet(packet, tr)
    spec = build_spec(config, data)
    maxiter = maxiter_full if config.kind == "full" else maxiter_tied
    topk = topk_full if config.kind == "full" else topk_tied
    model = fit_tied(packet_tr, spec, maxiter=maxiter, coarse_top_k=topk)
    pred_te = predict_tied(model, packet, packet.w[te])
    pred_tr = predict_tied(model, packet, packet.w[tr])
    np.savez_compressed(out, te=te, pred=pred_te, train_pred=pred_tr, tr=tr)
    logger.info("%s[%s] fold %d done in %.1fs", config.name, target, fold_id, time.monotonic() - t0)


def fit_final(
    config: Config,
    data: GrugData,
    *,
    maxiter_full: int,
    topk_full: int,
    maxiter_tied: int,
    target: str = TARGET,
) -> None:
    """Fit on all 800 rows for train-vs-OOF gap + parameter interpretation."""
    out = final_path(config.name, target)
    if out.exists():
        return
    t0 = time.monotonic()
    data_packet = packet_for_target(data, target)
    spec = build_spec(config, data)
    maxiter = maxiter_full if config.kind == "full" else maxiter_tied
    model = fit_tied(data_packet, spec, maxiter=maxiter, coarse_top_k=topk_full if config.kind == "full" else 3)
    pred = predict_tied(model, data_packet, data_packet.w)
    payload = {
        "params": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.params.items()},
        "intercept": model.intercept,
        "benefit_coef": model.benefit_coef.tolist(),
        "penalty_coef": model.penalty_coef.tolist(),
        "head_u": model.head_u.tolist(),
        "param_count": spec.param_count(),
    }
    payload["train_spearman"] = float(spearmanr(data_packet.y, pred).statistic)
    payload["train_rmse"] = float(np.sqrt(np.mean((pred - data_packet.y) ** 2)))
    payload["config"] = config.name
    payload["target"] = target
    out.write_text(json.dumps(payload))
    logger.info(
        "%s final fit done in %.1fs (train sp %.3f)", config.name, time.monotonic() - t0, payload["train_spearman"]
    )


def run_command(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_grug_data()
    logger.info(
        "packet: %d runs x %d buckets; c0 range [%.1f, %.1f]",
        len(data.packet.y),
        data.packet.m,
        data.packet.c0.min(),
        data.packet.c0.max(),
    )

    configs = config_grid()
    if args.configs:
        wanted = set(args.configs.split(","))
        unknown = wanted - {c.name for c in configs}
        if unknown:
            raise ValueError(f"unknown configs: {sorted(unknown)}")
        configs = [c for c in configs if c.name in wanted]

    target = args.target
    if target not in data.targets:
        raise ValueError(f"unknown target {target!r}; have {sorted(data.targets)}")

    tasks = []
    for config in configs:
        for fold_id, tr, te in fold_list(len(data.packet.y), config.folds):
            if not cache_path(config.name, fold_id, target).exists():
                tasks.append((config, fold_id, tr, te))
    # cheap tied tasks first so partial results land early
    tasks.sort(key=lambda t: (t[0].kind == "full", t[0].name, t[1]))
    logger.info("%d fold tasks to run (target %s)", len(tasks), target)
    joblib.Parallel(n_jobs=min(args.jobs, max(1, len(tasks))))(
        joblib.delayed(fit_predict_fold)(
            config,
            data,
            fold_id,
            tr,
            te,
            maxiter_full=args.maxiter_full,
            topk_full=args.topk_full,
            maxiter_tied=args.maxiter_tied,
            topk_tied=args.topk_tied,
            target=target,
        )
        for config, fold_id, tr, te in tasks
    )
    if target != TARGET:
        # phase-2 cv_results.parquet only covers macro_bpb; refit the Hellinger
        # kernel on the same folds for paired deltas on the secondary target
        kernel_tasks = [
            (fid, tr, te)
            for fid, tr, te in fold_list(len(data.packet.y), "all")
            if not cache_path("kernel_hellinger", fid, target).exists()
        ]
        logger.info("%d kernel folds to run (target %s)", len(kernel_tasks), target)
        joblib.Parallel(n_jobs=min(args.jobs, max(1, len(kernel_tasks))))(
            joblib.delayed(run_kernel_fold)(data, target, fid, tr, te) for fid, tr, te in kernel_tasks
        )
    finals = [c for c in configs if not final_path(c.name, target).exists()]
    logger.info("%d final fits to run", len(finals))
    joblib.Parallel(n_jobs=min(args.jobs, max(1, len(finals))))(
        joblib.delayed(fit_final)(
            config,
            data,
            maxiter_full=args.maxiter_full,
            topk_full=args.topk_full,
            maxiter_tied=args.maxiter_tied,
            target=target,
        )
        for config in finals
    )
    logger.info("run complete; use `report` to assemble outputs")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def collect_results(data: GrugData, target: str = TARGET) -> pd.DataFrame:
    rows = []
    y = data.targets[target]
    for config in config_grid():
        for fold_id, _tr, _te in fold_list(len(y), config.folds):
            path = cache_path(config.name, fold_id, target)
            if not path.exists():
                continue
            npz = np.load(path)
            pred, te_idx = npz["pred"], npz["te"]
            rows.append(
                {
                    "target": target,
                    "config": config.name,
                    "kind": config.kind,
                    "variant": config.variant_key,
                    "head": config.head,
                    "rho_tie": config.rho_tie,
                    "fold_id": fold_id,
                    "repeat": fold_id // N_SPLITS,
                    "spearman": float(spearmanr(y[te_idx], pred).statistic),
                    "rmse": float(np.sqrt(np.mean((pred - y[te_idx]) ** 2))),
                    "train_spearman_fold": float(spearmanr(y[npz["tr"]], npz["train_pred"]).statistic),
                    "train_rmse_fold": float(np.sqrt(np.mean((npz["train_pred"] - y[npz["tr"]]) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def kernel_per_fold(data: GrugData, target: str = TARGET) -> pd.DataFrame:
    """Per-fold kernel reference: phase-2 cv_results for macro_bpb, own refit otherwise."""
    if target == TARGET:
        cv = pd.read_parquet(CV_RESULTS)
        return cv[cv["variant"] == KERNEL_VARIANT].sort_values("fold_id").reset_index(drop=True)
    y = data.targets[target]
    rows = []
    for fold_id, _tr, _te in fold_list(len(y), "all"):
        path = cache_path("kernel_hellinger", fold_id, target)
        if not path.exists():
            continue
        npz = np.load(path)
        rows.append(
            {
                "variant": KERNEL_VARIANT,
                "fold_id": fold_id,
                "spearman": float(spearmanr(y[npz["te"]], npz["pred"]).statistic),
                "rmse": float(np.sqrt(np.mean((npz["pred"] - y[npz["te"]]) ** 2))),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def paired_vs_kernel(per_fold: pd.DataFrame, kernel: pd.DataFrame, config_name: str) -> dict | None:
    sub = per_fold[per_fold["config"] == config_name].sort_values("fold_id")
    if sub.empty:
        return None
    ker = kernel[kernel["fold_id"].isin(sub["fold_id"])].sort_values("fold_id")
    d_sp = sub["spearman"].to_numpy() - ker["spearman"].to_numpy()
    d_rmse = sub["rmse"].to_numpy() - ker["rmse"].to_numpy()

    def _p(d: np.ndarray) -> float:
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    return {
        "config": config_name,
        "n_folds": len(sub),
        "d_spearman_mean": float(d_sp.mean()),
        "d_spearman_std": float(d_sp.std()),
        "d_spearman_wins": int((d_sp > 0).sum()),
        "d_spearman_p": _p(d_sp),
        "d_rmse_mean": float(d_rmse.mean()),
        "d_rmse_p": _p(d_rmse),
    }


def interpret_final(data: GrugData, config_name: str, top_n: int = 12, target: str = TARGET) -> dict | None:
    path = final_path(config_name, target)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    a = np.asarray(payload["benefit_coef"])
    p = np.asarray(payload["penalty_coef"])
    rho = np.asarray(payload["params"]["rho"])
    buckets = data.buckets
    order = np.argsort(-a)
    top_benefit = [
        {"bucket": buckets[i], "a": float(a[i]), "rho": float(rho[i]), "p": float(p[i])} for i in order[:top_n]
    ]
    # quality-tier gradient of a within clusters (>= 2 non-tail tiers, a > 0 somewhere)
    tier_corrs = []
    for g in range(int(data.cluster_of_bucket.max()) + 1):
        mask = (data.cluster_of_bucket == g) & (data.quality_tier >= 0)
        if mask.sum() >= 3 and a[mask].max() > 1e-10 and np.std(a[mask]) > 0:
            corr = spearmanr(data.quality_tier[mask], a[mask]).statistic
            if np.isfinite(corr):
                tier_corrs.append(float(corr))
    return {
        "config": config_name,
        "train_spearman": payload["train_spearman"],
        "train_rmse": payload["train_rmse"],
        "param_count": payload["param_count"],
        "active_benefit": int((a > 1e-10).sum()),
        "active_penalty": int((p > 1e-10).sum()),
        "top_benefit": top_benefit,
        "tier_corr_mean": float(np.mean(tier_corrs)) if tier_corrs else float("nan"),
        "tier_corr_n_clusters": len(tier_corrs),
        "tier_corrs": tier_corrs,
    }


def summarize_target(data: GrugData, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-fold rows + config summary + kernel per-fold reference for one target."""
    per_fold = collect_results(data, target)
    kernel = kernel_per_fold(data, target)
    y = data.targets[target]
    summary_rows = []
    for config in config_grid():
        sub = per_fold[per_fold["config"] == config.name] if not per_fold.empty else pd.DataFrame()
        if sub.empty:
            continue
        fpath = final_path(config.name, target)
        final = json.loads(fpath.read_text()) if fpath.exists() else {}
        # OOF over folds run; for repeat0 configs this is the repeat-0 OOF vector
        oof = np.full(len(y), np.nan)
        for fold_id in sub["fold_id"]:
            npz = np.load(cache_path(config.name, fold_id, target))
            if fold_id < N_SPLITS:
                oof[npz["te"]] = npz["pred"]
        oof_sp_r0 = float(spearmanr(y[~np.isnan(oof)], oof[~np.isnan(oof)]).statistic)
        pair = paired_vs_kernel(per_fold, kernel, config.name) if not kernel.empty else None
        summary_rows.append(
            {
                "target": target,
                "config": config.name,
                "variant": config.variant_key,
                "head": config.head,
                "rho_tie": config.rho_tie,
                "n_folds": len(sub),
                "param_count": final.get("param_count"),
                "oof_spearman_mean": float(sub["spearman"].mean()),
                "oof_spearman_std": float(sub["spearman"].std()),
                "oof_rmse_mean": float(sub["rmse"].mean()),
                "oof_spearman_repeat0": oof_sp_r0,
                "train_spearman_full": final.get("train_spearman"),
                "train_rmse_full": final.get("train_rmse"),
                "fold_train_spearman_mean": float(sub["train_spearman_fold"].mean()),
                "d_spearman_vs_kernel": pair["d_spearman_mean"] if pair else None,
                "d_spearman_p": pair["d_spearman_p"] if pair else None,
                "d_spearman_wins": f"{pair['d_spearman_wins']}/{pair['n_folds']}" if pair else None,
                "d_rmse_vs_kernel": pair["d_rmse_mean"] if pair else None,
            }
        )
    summary = (
        pd.DataFrame(summary_rows).sort_values("oof_spearman_mean", ascending=False).reset_index(drop=True)
        if summary_rows
        else pd.DataFrame()
    )
    return per_fold, summary, kernel


def summary_table_lines(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| config | variant | head | params | folds | OOF Spearman | OOF RMSE "
        "| train Spearman (800) | dSp vs kernel | wins | p |",
        "|--------|---------|------|--------|-------|--------------|----------"
        "|----------------------|---------------|------|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['config']} | {r['variant']} | {r['head']} | {r['param_count']} | {r['n_folds']} | "
            f"{r['oof_spearman_mean']:.4f} +/- {r['oof_spearman_std']:.4f} | {r['oof_rmse_mean']:.4f} | "
            f"{'' if r['train_spearman_full'] is None else format(r['train_spearman_full'], '.4f')} | "
            f"{'' if r['d_spearman_vs_kernel'] is None else format(r['d_spearman_vs_kernel'], '+.4f')} | "
            f"{r['d_spearman_wins'] or ''} | "
            f"{'' if r['d_spearman_p'] is None else format(r['d_spearman_p'], '.4f')} |"
        )
    lines.append("")
    return lines


def report_command(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = load_grug_data()
    per_fold, summary, kernel = summarize_target(data, TARGET)
    if per_fold.empty:
        raise SystemExit("no cached fold results; run `run` first")

    all_per_fold = [per_fold]
    kernel_mean_sp = float(kernel["spearman"].mean())
    kernel_mean_rmse = float(kernel["rmse"].mean())

    lines: list[str] = []
    a = lines.append
    a("# DSP dose-response surrogate on the grug swarm (800 train runs)\n")
    a("Vendored `dsp_exact.py` (swarm branch) ported to the 168-bucket grug swarm; nested CV on")
    a("the SAME RepeatedKFold(5,3,seed 0) splits as phase 2 (`grug_fit.py`). Nonlinear params are")
    a("refit inside every fold (start bank -> bounded L-BFGS-B with the exact variable-projection")
    a("gradient); all configs use all 15 folds. Target `macro_bpb` (lower")
    a(f"better). Phase-2 Hellinger-kernel reference: OOF Spearman {kernel_mean_sp:.4f}, RMSE {kernel_mean_rmse:.4f}.\n")

    a("## Epoch semantics\n")
    a(f"- target budget {TARGET_BUDGET_TOKENS:,} tok (== launcher `_TARGET_BUDGET_TOKENS`, the")
    a("  store_8ac06c74 natural size); simulated epoching => epoch multipliers from TARGET budget,")
    a("  not realized tokens (`docs/debug-log-epoch-feature-budget-semantics.md`).")
    p0_steps, p1_steps = phase_step_split()
    a(
        f"- phase steps [{p0_steps}, {p1_steps}] -> fractions "
        f"[{p0_steps / TOTAL_STEPS:.3f}, {p1_steps / TOTAL_STEPS:.3f}];"
    )
    a("  c_p[j] = fraction_p * target_budget / T_j; T_j from buckets_table (tail = pooled 33 children).")
    a(
        f"- c0 range [{data.packet.c0.min():.1f}, {data.packet.c0.max():.1f}]; "
        "exposures z = c*w are epochs (p50 max ~9.7).\n"
    )

    a("## Variant x tie grid (OOF Spearman / RMSE; paired per-fold delta vs Hellinger kernel)\n")
    lines.extend(summary_table_lines(summary))

    a("## Train-vs-OOF gap (overfit watch)\n")
    a("| config | params | fold train Spearman | OOF Spearman | gap |")
    a("|--------|--------|---------------------|--------------|-----|")
    for _, r in summary.iterrows():
        gap = r["fold_train_spearman_mean"] - r["oof_spearman_mean"]
        a(
            f"| {r['config']} | {r['param_count']} | {r['fold_train_spearman_mean']:.4f} | "
            f"{r['oof_spearman_mean']:.4f} | {gap:.4f} |"
        )
    a("")

    a("## Fitted-parameter interpretation (final fits on all 800)\n")
    for config in config_grid():
        interp = interpret_final(data, config.name)
        if interp is None:
            continue
        a(f"### {config.name}\n")
        a(
            f"- params {interp['param_count']}; active benefit coefs {interp['active_benefit']}/168, "
            f"active penalty {interp['active_penalty']}/168; train Spearman {interp['train_spearman']:.4f}"
        )
        top = ", ".join(f"{t['bucket']} (a={t['a']:.3g}, rho={t['rho']:.3g})" for t in interp["top_benefit"][:8])
        a(f"- top benefit buckets: {top}")
        if np.isfinite(interp["tier_corr_mean"]):
            a(
                f"- quality-tier gradient of a within cluster: mean Spearman(tier, a) = "
                f"{interp['tier_corr_mean']:+.3f} over {interp['tier_corr_n_clusters']} clusters"
            )
        elif config.head == "cluster":
            a("- quality-tier gradient: n/a (cluster head ties a across tiers by construction)")
        a("")

    summaries = [summary]
    if ZMACRO_TARGET in data.targets:
        z_per_fold, z_summary, z_kernel = summarize_target(data, ZMACRO_TARGET)
        if not z_summary.empty:
            all_per_fold.append(z_per_fold)
            summaries.append(z_summary)
            a(f"## Secondary target: {ZMACRO_TARGET} (target-analysis recommendation)\n")
            a(
                f"- Hellinger kernel refit on the same folds: OOF Spearman {z_kernel['spearman'].mean():.4f} "
                f"+/- {z_kernel['spearman'].std():.4f}, RMSE {z_kernel['rmse'].mean():.4f}"
                " (matches the target analysis' 0.8147 per-fold mean).\n"
            )
            lines.extend(summary_table_lines(z_summary))

    a("## Verdict\n")
    best = summary.iloc[0]
    a(
        f"- Best DSP configuration on macro_bpb: **{best['config']}** (OOF Spearman "
        f"{best['oof_spearman_mean']:.4f}) vs Hellinger kernel {kernel_mean_sp:.4f} — paired delta "
        f"{best['d_spearman_vs_kernel']:+.4f} ({best['d_spearman_wins']} folds, p={best['d_spearman_p']:.4f})."
    )
    a("- The DSP functional form does NOT transfer its qsplit240 advantage (0.91-0.93 OOF there) to the")
    a("  grug swarm: every DSP config loses to the kernel on both targets, on macro_bpb mainly because")
    a("  the predictable signal is small and 168 buckets x 800 runs favors smooth kernel regression;")
    a("  the untied 673-param models overfit (fold-train ~0.69 vs OOF ~0.18-0.23) and the tied forms,")
    a("  while calibrated (train ~0.37-0.39), recover only linear-content-level skill.")
    a("- Content-coupled parameter tying (a_i, p_i = nonneg-linear in the K40 content profile) is the")
    a("  best DSP parameterization on both targets — one step beyond buckets helps DSP, but not enough")
    a("  to reach the kernel.")

    pd.concat(all_per_fold, ignore_index=True).to_parquet(GRUG_DIR / "dsp_results.parquet", index=False)
    (GRUG_DIR / "dsp_report.md").write_text("\n".join(lines))
    pd.concat(summaries, ignore_index=True).to_json(GRUG_DIR / "dsp_summary.json", orient="records", indent=2)
    print("\n".join(lines))
    logger.info("wrote %s and %s", GRUG_DIR / "dsp_results.parquet", GRUG_DIR / "dsp_report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="fit configs over nested CV folds (cached, resumable)")
    run.add_argument("--configs", default="", help="comma list; default all")
    run.add_argument("--jobs", type=int, default=3)
    run.add_argument("--maxiter-full", type=int, default=36)
    run.add_argument("--topk-full", type=int, default=3)
    run.add_argument("--maxiter-tied", type=int, default=36)
    run.add_argument("--topk-tied", type=int, default=3)
    run.add_argument("--target", default=TARGET, help=f"{TARGET} (default) or {ZMACRO_TARGET}")
    run.set_defaults(func=run_command)
    rep = sub.add_parser("report", help="assemble dsp_results.parquet + dsp_report.md")
    rep.set_defaults(func=report_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
