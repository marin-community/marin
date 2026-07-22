# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Head-to-head: our Hellinger-kernel surrogate vs the parametric mixture forms proposed on
marin#2403, vs a linear RegMix baseline -- with honest uncertainty on the DIFFERENCES.

Every earlier comparison in this campaign reported point estimates (a Spearman here, an
RMSE there) and left the reader to guess whether a 0.02 gap is a result or a coin flip.
The whole question this module exists to answer is whether the gaps between methods are
real, so every headline number is a PAIRED difference with a bootstrap confidence interval
attached, and the verdict for each pair is literally "CI excludes zero" or "CI covers zero".

METHODS
-------
``kernel``
    The campaign's frozen Hellinger kernel ridge over the K=1000 content basis
    (gamma_factor 0.25, alpha 0.1; gamma re-derived from each training fold's own median
    pairwise distance). Our model.

``p3_epoch`` / ``p3_flat``  -- THE PRIMARY COMPETITOR
    The P3 functional form from marin#2403 (Calvin-Xu's 2026-05-05 comment, attributed to
    Will), the closest published competitor to our model: like ours it is built for the
    two-phase, epoching-aware problem, so this is parametric-form vs kernel-plus-harm on
    the same two jobs rather than against a content-blind baseline. Exactly as written
    there,

        yhat(w0, w1) = b + sum_d beta_d (w0_d + eta w1_d)^a
                         - gamma0 sum_d (c0_d w0_d)^p
                         - gamma1 sum_d (c1_d w1_d)^p

    Three nonlinear coefficients only -- that is the point of the form:
      eta  phase-1 multiplier: how much a cooldown-phase token counts against a phase-0 one
      a    global nonlinear exposure power (a < 1 = saturating returns to dose)
      p    overexposure / concentration penalty power
    plus the linear head: intercept b, per-domain beta_d, and penalty scales gamma0/gamma1.

    WHAT ARE c0_d / c1_d? The issue does not define them, and the reading changes the
    model, so we resolve the ambiguity explicitly and run BOTH readings:

      p3_epoch  (primary) c_{p,d} = f_p * B / T_d, so that c_{p,d} * w_{p,d} is exactly the
                campaign's per-phase EPOCH count for bucket d -- identical to
                ``swoosh_form.per_phase_epochs`` and ``grug_fit.stage_b_epochs``
                (f = phase token fractions, B = 10.372e12 target budget, T_d = bucket token
                total). This makes terms 3 and 4 an overexposure/repetition penalty, which
                is the natural reading of #2403's "extended & generalized to handle
                epoching".
      p3_flat   c_{p,d} = 1, so the terms are a pure concentration penalty with no data-
                volume scaling. Reported alongside because the ambiguity is real and the
                two readings are different models.

    Fitting exploits the form's structure: given (eta, a, p) the model is LINEAR in
    (b, beta, gamma0, gamma1), so the linear head is profiled out in closed form (ridge
    least squares with gamma0, gamma1 constrained non-negative, since they are penalties)
    and only the 3 nonlinear coefficients go to a derivative-free search. The ridge on the
    head is selected by inner CV and its grid includes 0, so the unregularized published
    form is always available to be chosen.

``linear_ridge``
    RidgeCV on the raw 2x168 per-phase bucket proportions -- the linear stand-in for
    RegMix. RegMix proper fits LightGBM on the same features; that cannot be run here (see
    LIGHTGBM_NOTE), so this is a linear lower bound on the RegMix family, not RegMix.

``law_grouped`` / ``law_pooled`` / ``law_grouped3`` / ``law_full``  -- SECONDARY
    The published parametric data-mixing law of Ye et al. 2024, "Data Mixing Laws:
    Optimizing Data Mixtures by Predicting Language Modeling Performance"
    (arXiv:2403.16952). Their Eq. 5 exponential form is, for validation domain i and
    training-domain proportions r_1..r_M,

        L_i(r_1..r_M) = c_i + k_i * exp( sum_j t_ij * r_j )

    We fit the scalar-target version, L(w) = c + k * exp(beta^T w), by penalized nonlinear
    least squares. Kept as a secondary baseline. Variants differ only in what plays the
    role of the domain-proportion vector w:

      law_grouped   the campaign's content GROUPS, per phase (2 x n_groups features). The
                    form's intended regime -- Ye et al. work with 5 domains up to ~20.
      law_pooled    the same groups pooled across phases into one budget-weighted overall
                    proportion per group. The cheap phase-blind variant.
      law_grouped3  the 3-domain structure ``function_contours.py`` slices on.
      law_full      beta over all 2 x 168 raw bucket proportions: the apples-to-apples
                    feature set, reported WITH its identifiability damage, not instead of it.

``GRP`` -- NOT RUN. See GRP_NOTE. #2403 calls GRP the thoroughly-tested incumbent, but the
    only implementation available here is hardcoded to a different domain taxonomy, and
    re-targeting it at our 168 buckets would mean inventing a domain-family partition and a
    quality-pair structure by hand. That would be a reconstruction wearing GRP's name, so
    it is omitted rather than approximated.

GROUPING (law_grouped / law_pooled)
-----------------------------------
Built from the campaign's existing structure, not invented here: the validation-batch-3
``code_adjacent`` group (clusters c01/c02/c06, 15 buckets, 24.4% of tokens) is kept whole,
the ``N_SOLO_CLUSTERS`` largest remaining clusters by token share enter as their own
domains, and everything else pools into a single ``remainder`` domain. With the default
this yields 10 domains -- inside the 5-20 range the published law was validated on.

IDENTIFIABILITY
---------------
Reported, not hidden. For the mixing law there are two degeneracies:

  1. GAUGE. Per-phase proportions sum to 1, so adding a constant to every beta entry within
     one phase multiplies exp(beta^T w) by a constant, which k absorbs exactly. beta
     therefore has one exact null direction PER PHASE in every variant, grouped included.
     The ridge penalty fixes the gauge (minimum-norm beta); we report beta per-phase-centered.
  2. SCALE. When the exponent range is small, c + k*exp(z) ~= (c + k) + k*z: the law
     collapses onto the linear model and (c, k) are only jointly identified. We report the
     realized exponent span so the reader can see how much curvature the fit actually uses.

``law_full`` additionally carries 2 + 2*168 = 338 free parameters against 800 runs. P3's
linear head carries 1 + 168 + 2 = 171. For both we report the eigenspectrum and effective
rank of the normal matrix at the optimum, so "ill-posed" is a measurement, not an adjective.

EVALUATION
----------
Nested out-of-fold over the 800 train runs on ONE shared fold assignment, so every method
sees identical training and test rows and the comparison is PAIRED. Every hyperparameter
(ridge alpha, law penalty, P3's nonlinear coefficients and head ridge) is chosen by an
inner CV inside the training fold only.

  - per method: out-of-fold Spearman, Pearson, RMSE, plus each score as a fraction of the
    measured noise ceiling;
  - per method PAIR: paired bootstrap over runs (``N_BOOT`` resamples, the SAME resampled
    row set for both members of the pair) giving 95% CIs on delta-Spearman and delta-RMSE,
    and an explicit excludes-zero verdict;
  - fold-assignment sensitivity: the protocol re-run over ``FOLD_SEEDS`` fold assignments,
    so a delta that only exists for one fold split is visible as such.

NOISE CEILING
-------------
Measured, not assumed: the 10-seed anchor panel gives sigma(zmacro) = 0.2127 z against a
signal spread of 0.5103 z, i.e. reliability 0.826 and an implied maximum Spearman of 0.909;
for humaneval bpb the same panel gives sigma = 0.005651 against signal 0.063597, i.e.
reliability 0.992 and ceiling 0.996. Every Spearman is also reported as a fraction of its
target's ceiling.

PREDICTIVE UNCERTAINTY, AND WHETHER THE METHODS DISAGREE MORE THAN THEY KNOW
---------------------------------------------------------------------------
Two different readings of "uncertainty between methods", and this module answers both. The
first is the paired bootstrap above: is the SCORE gap real. The second is whether the
methods' per-run PREDICTIONS are distinguishable given each model's own error bars. For
that we attach to every method a nested split-conformal interval (fit on a proper-train
subset of each training fold, absolute-residual quantile on a held-out calibration subset)
and, for the kernel, its GP posterior sd as well. Two methods' predictions for a run are
RESOLVABLE when their conformal intervals are disjoint, i.e. |mu_A - mu_B| > q_A + q_B; we
report that fraction per pair alongside the intervals' realized coverage.

Writes report/figs3/f37..f43 and grug/method_comparison.{json,md}.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import grug_fit as gf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from function_contours import (
    ALPHA,
    BIN_GRID,
    BUCKETS_TABLE,
    CODE_CLUSTERS,
    CORR_F0,
    CORR_F1,
    GAMMA_FACTOR,
    MIN_BIN_N,
    N_GRID,
    RIDGE_ALPHAS,
    Slice,
    bin_empirical,
    build_slices,
    d2_to_train,
    flat_weights,
    load_data,
    mixture_sqrt_hists,
)
from gp_surrogate import fit_gp, krr_predict, predict_gp
from joblib import Parallel, delayed
from scipy.optimize import least_squares, minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
GRUG = SCRATCH / "grug"
FIGS = SCRATCH / "report" / "figs3"

LIGHTGBM_NOTE = (
    "RegMix proper (LightGBM on the raw bucket proportions) was NOT run: this host has no libgomp, so "
    "`import lightgbm` fails outright. linear_ridge is therefore a LINEAR stand-in and a lower bound on "
    "the RegMix family, not RegMix itself. No substitute gradient-boosting library was swapped in."
)
GRP_NOTE = (
    "GRP was NOT run, and is not approximated. #2403 calls GRP the thoroughly-tested incumbent (60M and 300M), so "
    "we looked for a faithful implementation and did not find one that can be re-targeted. (1) STRUCTURE IS "
    "HARDCODED, NOT DATA: the surrogate takes a packet argument, but the packet is manufactured by string-matching "
    "dolma3/dolmino domain names -- families come from a module constant "
    "GENERIC_FAMILY_NAMES = ('broad_text','tech_code','reasoning') with the parameter vector spelled out per family "
    "(a_broad_text, tau_tech_code, ...), and quality pairs from `domain[:-5] + '_low'` on dolma3_cc/ prefixes. Our "
    "buckets are embedding-derived clusters c01..c36 x 5 quality tiers q0..q4: that pairing rule yields zero pairs "
    "and 168 singletons, silently deleting GRP's cross-quality term rather than erroring, and the constructor "
    "hard-fails unless every domain lands in exactly one of the three named families. (2) GRP's quality model is one "
    "scalar discounting the low half of a HIGH/LOW pair; our design has a 5-tier ladder, which the form cannot "
    "express without a new parameterization. (3) DATA MISSING: two_phase_many_epoch_metadata.csv and both best-param "
    "CSVs are absent filesystem-wide, and the shipped nonlinear parameters are pre-fitted to eval/uncheatable_eval/bpb "
    "on the 39-domain 60M/1.2B panel -- three of them are named after that panel's families, so they have no referent "
    "here. Re-targeting GRP would mean inventing a family partition and a quality-pair structure by hand, which is "
    "exactly the open problem #2403 describes and would decide 7 of its 10 nonlinear parameters. That is a "
    "reconstruction borrowing GRP's functional forms, not GRP, so it is omitted rather than mislabelled."
)

# Parallelism. The per-(fold, method) OOF fits and the full-data fits are independent, and
# the expensive ones (P3's nonlinear-least-squares refit ~30-40s, the full-bucket law ~55s)
# dominate, so they are dispatched across cores. The `multiprocessing` (fork) backend is
# REQUIRED, not incidental: the workers must inherit the parent's already-imported modules --
# including the `lightgbm` stub the throwaway runner installs before import -- because
# `retrodiction` imports lightgbm at module top and this host has no libgomp. A spawn backend
# (loky) re-imports the module in a fresh interpreter without the stub and crashes on import.
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
PARALLEL_BACKEND = "multiprocessing"
# Read-only inputs shared with fork workers via copy-on-write, set just before each dispatch;
# only tiny per-task index arrays are pickled, never the 800x800 distance matrix.
_SHARED: dict = {}

N_SOLO_CLUSTERS = 8  # largest non-code clusters entering the grouping as their own domains
N_OUTER, N_INNER = 5, 3  # outer OOF folds; inner folds for every hyperparameter choice
PRIMARY_SEED = 0
FOLD_SEEDS = (0, 1, 2)  # fold-assignment sensitivity check
N_BOOT = 4000  # paired bootstrap resamples (brief asks >= 2000)
CI_LO, CI_HI = 2.5, 97.5
CONFORMAL_LEVEL = 0.90  # nominal coverage of the split-conformal intervals
CALIB_FRAC = 0.25  # share of each training fold reserved for conformal calibration

# --- P3 (marin#2403) search settings -------------------------------------------------
P3_BOUNDS = ((0.02, 20.0), (0.05, 2.0), (0.25, 6.0))  # eta, a, p
P3_STARTS = ((1.0, 1.0, 2.0), (0.30, 0.50, 2.0))  # linear-exposure start, saturating start
# Head ridge grid. 0 keeps the unregularized published form reachable; the top end is well
# past where the inner CV settles, so the selected value is interior rather than clipped.
P3_RIDGES = (0.0, 1e-1, 1.0, 1e1, 1e2, 1e3)
P3_MAXFEV = 120

# --- Data Mixing Laws (secondary) ----------------------------------------------------
LAW_PENALTIES = np.concatenate([[0.0], np.logspace(-6.0, 3.0, 7)])  # 0 keeps the published form reachable
LAW_INNER = 2  # inner CV folds for the law's penalty selection (P3 keeps the richer N_INNER)
EXP_CLIP = 30.0  # exponent clamp; the fits never approach it, but NLS excursions can
# Iteration cap on each nonlinear least-squares fit. The grouped law converges in <20
# iterations so the cap never binds there; it binds only on the full-bucket law's
# heavily-regularized (beta->0) fits, whose prediction is already flat by then -- so the cap
# bounds the cost of the known-ill-posed variant without changing what it says.
MAX_NFEV = 120

# Measured seed-noise floors and the reliability ceiling they imply (seedpanel_readout.json).
NOISE = {
    "zmacro": {"sigma": 0.2127, "signal_sd": 0.5103273932971658, "ceiling_spearman": 0.9089911402423266},
    "humaneval": {"sigma": 0.005651438386197603, "signal_sd": 0.0635967242126479, "ceiling_spearman": 0.9960443},
}

# Every method that gets scored. PAIR_METHODS is the subset entering the pairwise bootstrap
# grid: the law's extra parameterizations are informative as scores but would triple the
# number of pairs without adding a comparison anyone asked for.
METHODS = ("kernel", "p3_epoch", "p3_flat", "linear_ridge", "law_grouped", "law_full", "law_pooled", "law_grouped3")
PAIR_METHODS = ("kernel", "p3_epoch", "p3_flat", "linear_ridge", "law_grouped", "law_full")
HEADLINE = ("kernel", "p3_epoch", "linear_ridge", "law_grouped")  # the four drawn as surfaces
PAIRS = tuple((a, b) for i, a in enumerate(PAIR_METHODS) for b in PAIR_METHODS[i + 1 :])
P3_METHODS = ("p3_epoch", "p3_flat")
LAW_METHODS = ("law_grouped", "law_pooled", "law_grouped3", "law_full")

LABEL = {
    "kernel": "Hellinger kernel ridge\n(ours, frozen)",
    "p3_epoch": "P3 form, epoch-scaled penalty\n(#2403; c·w = epochs)",
    "p3_flat": "P3 form, flat penalty\n(#2403; c = 1)",
    "linear_ridge": "linear ridge on proportions\n(RegMix stand-in)",
    "law_grouped": "Data Mixing Law, grouped\n(Ye+2024, 10 domains)",
    "law_full": "Data Mixing Law, full-bucket\n(336 proportions)",
    "law_pooled": "Data Mixing Law, phase-pooled",
    "law_grouped3": "Data Mixing Law, 3 domains",
}
SHORT = {
    "kernel": "kernel",
    "p3_epoch": "P3-epoch",
    "p3_flat": "P3-flat",
    "linear_ridge": "linear",
    "law_grouped": "law-grouped",
    "law_full": "law-full",
    "law_pooled": "law-pooled",
    "law_grouped3": "law-3dom",
}
COLOR = {
    "kernel": "#1b5e6b",
    "p3_epoch": "#b5651d",
    "p3_flat": "#d8a13a",
    "linear_ridge": "#4a5a86",
    "law_grouped": "#6f7f4a",
    "law_full": "#8c3b4a",
    "law_pooled": "#8a6f3f",
    "law_grouped3": "#7a6a8a",
}

CMAP = plt.get_cmap("RdYlGn_r")  # green = lower = better (campaign convention)
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#d9d7d2"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.labelcolor": INK,
        "text.color": INK,
    }
)


# ---------------------------------------------------------------------------
# Grouping (used by the Data Mixing Law variants and for reading P3's beta back out)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grouping:
    """A partition of the 168 buckets into named content domains."""

    name: str
    names: list[str]
    membership: np.ndarray  # (n_buckets,) index into `names`

    @property
    def n_domains(self) -> int:
        return len(self.names)

    def matrix(self) -> np.ndarray:
        """(n_buckets, n_domains) 0/1 aggregation matrix: w @ matrix = per-domain shares."""
        m = np.zeros((len(self.membership), self.n_domains))
        m[np.arange(len(self.membership)), self.membership] = 1.0
        return m


def build_grouping(cluster_id: np.ndarray, p_tok: np.ndarray, n_solo: int) -> Grouping:
    """code_adjacent whole + the `n_solo` largest remaining clusters solo + pooled remainder."""
    code = np.isin(cluster_id, CODE_CLUSTERS)
    rest_clusters = sorted(set(cluster_id[~code].tolist()))
    share = {c: float(p_tok[cluster_id == c].sum()) for c in rest_clusters}
    solo = sorted(share, key=lambda c: -share[c])[:n_solo]
    names = ["code_adjacent"] + [f"c{c:02d}" for c in solo] + ["remainder"]
    membership = np.full(len(cluster_id), len(names) - 1, dtype=int)
    membership[code] = 0
    for i, c in enumerate(solo, start=1):
        membership[cluster_id == c] = i
    return Grouping(f"code_adjacent+top{n_solo}+remainder", names, membership)


def build_grouping3(cluster_id: np.ndarray) -> Grouping:
    """The 3-domain structure function_contours.py slices on: code_adjacent / c05 / rest."""
    membership = np.full(len(cluster_id), 2, dtype=int)
    membership[np.isin(cluster_id, CODE_CLUSTERS)] = 0
    membership[cluster_id == 5] = 1
    return Grouping("code_adjacent/c05/rest", ["code_adjacent", "c05", "rest"], membership)


def group_features(w: np.ndarray, grouping: Grouping) -> np.ndarray:
    """(n, 2 * n_domains) per-phase domain proportions; each phase block sums to 1."""
    m = grouping.matrix()
    return np.concatenate([w[:, 0, :] @ m, w[:, 1, :] @ m], axis=1)


def pooled_features(w: np.ndarray, grouping: Grouping) -> np.ndarray:
    """(n, n_domains) budget-weighted overall domain proportions; sums to 1."""
    m = grouping.matrix()
    return (CORR_F0 * w[:, 0, :] + CORR_F1 * w[:, 1, :]) @ m


# ---------------------------------------------------------------------------
# P3 (marin#2403):  yhat = b + sum_d beta_d (w0_d + eta w1_d)^a
#                          - gamma0 sum_d (c0_d w0_d)^p - gamma1 sum_d (c1_d w1_d)^p
# ---------------------------------------------------------------------------


def epoch_constants(t_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(c0, c1) such that c_{p,d} * w_{p,d} is bucket d's per-phase EPOCH count.

    Identical to ``swoosh_form.per_phase_epochs`` / ``grug_fit.stage_b_epochs``:
    e_{p,d} = w_{p,d} * f_p * B / T_d with B the target budget and f_p the phase token
    fractions, so c_{p,d} = f_p * B / T_d.
    """
    p0, p1 = gf.phase_step_split()
    f = np.array([p0, p1], dtype=np.float64) / gf.TOTAL_STEPS
    assert abs(f[0] - CORR_F0) < 1e-3 and abs(f[1] - CORR_F1) < 1e-3, "phase fractions drifted from the corrected pair"
    return f[0] * gf.TARGET_BUDGET_TOKENS / t_tokens, f[1] * gf.TARGET_BUDGET_TOKENS / t_tokens


@dataclass(frozen=True)
class LinearHead:
    """Ridge least-squares head with the trailing coefficients constrained non-negative.

    Columns are standardized internally, because the exponents a and p move the raw column
    scales by orders of magnitude and a fixed ridge would otherwise mean something different
    at every point of the nonlinear search.
    """

    mu: np.ndarray
    sd: np.ndarray
    coef: np.ndarray  # in standardized space
    ybar: float
    ridge: float
    objective: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ybar + ((x - self.mu) / self.sd) @ self.coef

    def original_scale(self) -> tuple[float, np.ndarray]:
        """(intercept, coefficients) of the model written against the RAW columns."""
        coef = self.coef / self.sd
        return float(self.ybar - self.mu @ coef), coef


def _solve_faces(
    gram_full: np.ndarray, rhs_full: np.ndarray, yy: float, ridge: float, n_nonneg_tail: int
) -> tuple[np.ndarray, float]:
    """Exact bounded ridge solve from precomputed cross-products.

    The objective is convex and the only constraints are lower bounds on the last
    `n_nonneg_tail` coefficients (gamma0, gamma1 -- penalty scales, so a negative value would
    turn the penalty into a bonus). The constrained optimum is the feasible face-minimizer
    with the lowest objective. The UNCONSTRAINED solution is tried first because whenever it
    is already feasible it is also the constrained optimum, and on this data it usually is --
    which turns the common case into one linear solve instead of four.
    """
    n_col = gram_full.shape[0]
    head_n = n_col - n_nonneg_tail
    faces = [np.arange(n_col)] + [
        np.array([i for i in range(n_col) if i not in drop])
        for drop in ([n_col - 1], [n_col - 2], [n_col - 2, n_col - 1])
    ]
    best: tuple[np.ndarray, float] | None = None
    for rank, free in enumerate(faces):
        sub = gram_full[np.ix_(free, free)]
        try:
            cf = np.linalg.solve(sub + ridge * np.eye(len(free)), rhs_full[free])
        except np.linalg.LinAlgError:
            continue
        if np.any(cf[head_n:] < 0.0):
            continue  # infeasible on this face
        coef = np.zeros(n_col)
        coef[free] = cf
        # ||z cf - yc||^2 + ridge ||cf||^2, expanded so we never form the n-vector residual
        obj = float(cf @ (sub @ cf) - 2.0 * cf @ rhs_full[free] + yy + ridge * (cf @ cf))
        if rank == 0:
            return coef, obj  # unconstrained optimum is feasible => it is the constrained one
        if best is None or obj < best[1]:
            best = (coef, obj)
    assert best is not None, "no feasible active set: the ridge solve failed on every face"
    return best


def solve_head_path(x: np.ndarray, y: np.ndarray, ridges, n_nonneg_tail: int = 2) -> list[LinearHead]:
    """Bounded ridge heads for a whole ridge grid, sharing one standardization and one Gram.

    The Gram is what costs; adding `ridge * I` to it is free. Evaluating the entire ridge
    grid per design matrix is therefore nearly the price of one ridge, which is what makes
    it affordable to fold the ridge choice into P3's nonlinear search.
    """
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-300, sd, 1.0)
    z = (x - mu) / sd
    ybar = float(y.mean())
    yc = y - ybar
    gram_full = z.T @ z
    rhs_full = z.T @ yc
    yy = float(yc @ yc)
    heads = []
    for ridge in ridges:
        coef, obj = _solve_faces(gram_full, rhs_full, yy, float(ridge), n_nonneg_tail)
        heads.append(LinearHead(mu=mu, sd=sd, coef=coef, ybar=ybar, ridge=float(ridge), objective=obj))
    return heads


def solve_head(x: np.ndarray, y: np.ndarray, ridge: float, n_nonneg_tail: int = 2) -> LinearHead:
    """Bounded ridge least-squares head at a single ridge value."""
    return solve_head_path(x, y, [ridge], n_nonneg_tail)[0]


def p3_design(w: np.ndarray, c0: np.ndarray, c1: np.ndarray, eta: float, a: float, p: float) -> np.ndarray:
    """(n, n_buckets + 2) design: exposure columns then the two (negated) penalty columns.

    The penalty columns carry the minus sign of the equation, so gamma0/gamma1 are just the
    trailing non-negative coefficients of a plain linear head.
    """
    exposure = np.clip(w[:, 0, :] + eta * w[:, 1, :], 0.0, None) ** a
    pen0 = np.sum(np.clip(c0[None, :] * w[:, 0, :], 0.0, None) ** p, axis=1)
    pen1 = np.sum(np.clip(c1[None, :] * w[:, 1, :], 0.0, None) ** p, axis=1)
    return np.column_stack([exposure, -pen0, -pen1])


@dataclass(frozen=True)
class P3Fit:
    """A fitted P3 model: 3 nonlinear coefficients plus a profiled linear head."""

    eta: float
    a: float
    p: float
    head: LinearHead
    c0: np.ndarray
    c1: np.ndarray
    ridge: float
    cv_sse: float
    penalty_curve: dict  # inner-CV RMSE across P3_RIDGES at the selected (eta, a, p)

    def predict(self, w: np.ndarray) -> np.ndarray:
        return self.head.predict(p3_design(w, self.c0, self.c1, self.eta, self.a, self.p))

    def parameters(self) -> dict:
        b, coef = self.head.original_scale()
        return {
            "eta": self.eta,
            "a": self.a,
            "p": self.p,
            "b": b,
            "beta": coef[:-2],
            "gamma0": float(coef[-2]),
            "gamma1": float(coef[-1]),
            "ridge": self.ridge,
        }


def _p3_cv_sse_per_ridge(nl: np.ndarray, w, y, c0, c1, folds) -> np.ndarray:
    """Inner-CV SSE at one (eta, a, p) for EVERY ridge on the grid, sharing the design."""
    eta, a, p = np.exp(nl)
    x = p3_design(w, c0, c1, eta, a, p)
    if not np.all(np.isfinite(x)):
        return np.full(len(P3_RIDGES), 1e18)
    total = np.zeros(len(P3_RIDGES))
    for tr, te in folds:
        for i, head in enumerate(solve_head_path(x[tr], y[tr], P3_RIDGES)):
            total[i] += float(np.sum((head.predict(x[te]) - y[te]) ** 2))
    return total


def _p3_objective(nl: np.ndarray, w, y, c0, c1, folds) -> float:
    """Profile the head ridge out of the search: the ridge grid is nearly free per design."""
    return float(_p3_cv_sse_per_ridge(nl, w, y, c0, c1, folds).min())


def fit_p3(w: np.ndarray, y: np.ndarray, c0: np.ndarray, c1: np.ndarray, seed: int) -> P3Fit:
    """Fit P3: multi-start bounded Powell on (log eta, log a, log p) against inner-CV SSE.

    The head ridge is selected jointly rather than by an outer loop: at any (eta, a, p) the
    whole ridge grid costs one extra triangular solve apiece on an already-computed Gram, so
    the search minimizes over it and the winner is read off at the optimum.
    """
    folds = list(KFold(n_splits=N_INNER, shuffle=True, random_state=seed).split(np.arange(len(y))))
    log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in P3_BOUNDS]
    best = None
    for start in P3_STARTS:
        res = minimize(
            _p3_objective,
            np.log(start),
            args=(w, y, c0, c1, folds),
            method="Powell",
            bounds=log_bounds,
            options={"maxfev": P3_MAXFEV, "xtol": 1e-3, "ftol": 1e-4},
        )
        if best is None or res.fun < best[0]:
            best = (float(res.fun), np.exp(res.x))
    cv_sse, (eta, a, p) = best
    per_ridge = _p3_cv_sse_per_ridge(np.log([eta, a, p]), w, y, c0, c1, folds)
    ridge = float(P3_RIDGES[int(np.argmin(per_ridge))])
    head = solve_head(p3_design(w, c0, c1, eta, a, p), y, ridge)
    curve = {"penalties": list(P3_RIDGES), "cv_rmse": [float(np.sqrt(v / len(y))) for v in per_ridge]}
    return P3Fit(
        eta=float(eta),
        a=float(a),
        p=float(p),
        head=head,
        c0=c0,
        c1=c1,
        ridge=ridge,
        cv_sse=cv_sse,
        penalty_curve=curve,
    )


# ---------------------------------------------------------------------------
# Data Mixing Law (secondary):  L(w) = c + k * exp(beta^T w)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LawFit:
    """Fitted parameters of L(w) = c + k * exp(beta^T w), plus what the fit cost."""

    c: float
    k: float
    beta: np.ndarray
    penalty: float
    cost: float
    n_fev: int
    exponent_range: tuple[float, float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.c + self.k * np.exp(np.clip(x @ self.beta, -EXP_CLIP, EXP_CLIP))


def _law_residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    c, k, beta = params[0], params[1], params[2:]
    z = np.clip(x @ beta, -EXP_CLIP, EXP_CLIP)
    data = (c + k * np.exp(z) - y) / np.sqrt(len(y))
    return np.concatenate([data, np.sqrt(penalty) * beta])


def _law_jacobian(params: np.ndarray, x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    k, beta = params[1], params[2:]
    n, p = x.shape
    e = np.exp(np.clip(x @ beta, -EXP_CLIP, EXP_CLIP))
    top = np.empty((n, p + 2))
    top[:, 0] = 1.0
    top[:, 1] = e
    top[:, 2:] = (k * e)[:, None] * x
    top /= np.sqrt(n)
    bottom = np.zeros((p, p + 2))
    bottom[:, 2:] = np.sqrt(penalty) * np.eye(p)
    return np.vstack([top, bottom])


def _law_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linearize the law about zero curvature: c + k*exp(z) ~= (c + k) + k*z near z = 0."""
    lin = Ridge(alpha=1e-6).fit(x, y)
    k0 = float(np.std(y)) or 1.0
    beta0 = lin.coef_ / k0
    z0 = float(np.mean(x @ beta0))
    c0 = float(np.mean(y)) - k0 * float(np.exp(np.clip(z0, -EXP_CLIP, EXP_CLIP)))
    return np.concatenate([[c0, k0], beta0])


def fit_law(x: np.ndarray, y: np.ndarray, penalty: float, start: np.ndarray | None = None) -> LawFit:
    """Penalized nonlinear least squares for L(w) = c + k*exp(beta^T w).

    The penalty is a ridge on beta only (c and k unpenalized). It doubles as the gauge fix
    for the exact per-phase constant-shift null direction described in the module docstring.
    """
    p0 = _law_init(x, y) if start is None else start
    res = least_squares(_law_residuals, p0, jac=_law_jacobian, args=(x, y, penalty), method="trf", max_nfev=MAX_NFEV)
    beta = res.x[2:]
    z = x @ beta
    return LawFit(
        c=float(res.x[0]),
        k=float(res.x[1]),
        beta=beta,
        penalty=penalty,
        cost=float(res.cost),
        n_fev=int(res.nfev),
        exponent_range=(float(z.min()), float(z.max())),
    )


def fit_law_path(x: np.ndarray, y: np.ndarray, penalties: np.ndarray) -> dict[float, LawFit]:
    """Fit the whole penalty path from most to least regularized, warm-starting downwards."""
    fits: dict[float, LawFit] = {}
    start = None
    for pen in sorted(penalties, reverse=True):
        fit = fit_law(x, y, pen, start)
        fits[float(pen)] = fit
        start = np.concatenate([[fit.c, fit.k], fit.beta])
    return fits


def select_law_penalty(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, dict]:
    """Inner-CV selection of the ridge penalty. Returns the choice and the whole curve."""
    inner = list(KFold(n_splits=LAW_INNER, shuffle=True, random_state=seed).split(np.arange(len(y))))
    sse = {float(p): 0.0 for p in LAW_PENALTIES}
    for tr, te in inner:
        for pen, fit in fit_law_path(x[tr], y[tr], LAW_PENALTIES).items():
            sse[pen] += float(np.sum((fit.predict(x[te]) - y[te]) ** 2))
    best = min(sse, key=lambda p: sse[p])
    curve = {"penalties": sorted(sse), "cv_rmse": [float(np.sqrt(sse[p] / len(y))) for p in sorted(sse)]}
    return best, curve


# ---------------------------------------------------------------------------
# Conditioning diagnostics
# ---------------------------------------------------------------------------


def spectrum_report(jac: np.ndarray, n_params: int, n_runs: int, penalty: float) -> dict:
    """Eigen-conditioning of a Gauss-Newton / design normal matrix, with the ridge trace."""
    eig = np.clip(np.linalg.eigvalsh(jac.T @ jac)[::-1], 0.0, None)
    nonzero = eig[eig > eig[0] * 1e-14]
    eff = float(np.sum(eig / (eig + penalty))) if penalty > 0 else float(np.sum(eig > eig[0] * 1e-12))
    return {
        "n_params": int(n_params),
        "n_runs": int(n_runs),
        "condition_number_nonzero": float(eig[0] / max(nonzero.min(), 1e-300)) if nonzero.size else float("inf"),
        "rank_numeric": int(np.sum(eig > eig[0] * 1e-12)),
        "effective_n_params": eff,
        "selected_penalty": float(penalty),
        "spectrum": [float(v) for v in eig],
    }


def law_identifiability(fit: LawFit, x: np.ndarray) -> dict:
    """Conditioning of the UNPENALIZED data fit at the law's optimum."""
    jac = _law_jacobian(np.concatenate([[fit.c, fit.k], fit.beta]), x, np.zeros(len(x)), 0.0)[: len(x)]
    return spectrum_report(jac * np.sqrt(len(x)), len(fit.beta) + 2, len(x), fit.penalty)


def p3_identifiability(fit: P3Fit, w: np.ndarray) -> dict:
    """Conditioning of P3's profiled linear head at the selected nonlinear coefficients."""
    x = p3_design(w, fit.c0, fit.c1, fit.eta, fit.a, fit.p)
    z = (x - fit.head.mu) / fit.head.sd
    return spectrum_report(z, x.shape[1] + 1, len(w), fit.ridge)


# ---------------------------------------------------------------------------
# Method fits (all share one signature so the OOF loop stays flat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Features:
    """Everything the methods need about one set of runs, precomputed once."""

    d2: np.ndarray  # (n, n) pairwise squared Hellinger
    w: np.ndarray  # (n, 2, n_buckets)
    c0: np.ndarray  # epoch constants, (n_buckets,)
    c1: np.ndarray
    grouped: np.ndarray
    pooled: np.ndarray
    grouped3: np.ndarray

    def law_x(self, method: str) -> np.ndarray:
        return {
            "law_grouped": self.grouped,
            "law_pooled": self.pooled,
            "law_grouped3": self.grouped3,
            "law_full": flat_weights(self.w),
        }[method]

    def p3_c(self, method: str) -> tuple[np.ndarray, np.ndarray]:
        if method == "p3_epoch":
            return self.c0, self.c1
        return np.ones_like(self.c0), np.ones_like(self.c1)


def fit_predict(method: str, feats: Features, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> np.ndarray:
    """Fit `method` on rows `tr` and predict rows `te`. Hyperparameters chosen inside `tr` only."""
    if method == "kernel":
        d_tr = feats.d2[np.ix_(tr, tr)]
        gamma = GAMMA_FACTOR / float(np.median(d_tr[~np.eye(len(tr), dtype=bool)]))
        return krr_predict(d_tr, feats.d2[np.ix_(te, tr)], y[tr], gamma, ALPHA)
    if method == "linear_ridge":
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(flat_weights(feats.w[tr]), y[tr])
        return model.predict(flat_weights(feats.w[te]))
    if method in P3_METHODS:
        c0, c1 = feats.p3_c(method)
        return fit_p3(feats.w[tr], y[tr], c0, c1, seed).predict(feats.w[te])
    x = feats.law_x(method)
    pen, _ = select_law_penalty(x[tr], y[tr], seed)
    return fit_law(x[tr], y[tr], pen).predict(x[te])  # one fit at the selected penalty, not the whole path


def fit_full(method: str, feats: Features, y: np.ndarray, seed: int):
    """Fit `method` on ALL runs -- for the contour surfaces and the identifiability report."""
    if method == "linear_ridge":
        return RidgeCV(alphas=RIDGE_ALPHAS).fit(flat_weights(feats.w), y)
    if method in P3_METHODS:
        c0, c1 = feats.p3_c(method)
        return fit_p3(feats.w, y, c0, c1, seed)
    x = feats.law_x(method)
    pen, curve = select_law_penalty(x, y, seed)
    return {"fit": fit_law(x, y, pen), "penalty_curve": curve}


def fit_full_batch(methods: tuple[str, ...], feats: Features, y: np.ndarray, seed: int) -> dict:
    """Fit several methods on all runs in parallel (independent optimizations)."""
    _SHARED["feats"], _SHARED["y"] = feats, y
    fits = Parallel(n_jobs=min(N_JOBS, len(methods)), backend=PARALLEL_BACKEND, batch_size=1)(
        delayed(_full_worker)(m, seed) for m in methods
    )
    return dict(zip(methods, fits, strict=True))


def _full_worker(method: str, seed: int):
    return fit_full(method, _SHARED["feats"], _SHARED["y"], seed)


# ---------------------------------------------------------------------------
# Out-of-fold protocol (paired: one fold assignment shared by every method)
# ---------------------------------------------------------------------------


def conformal_halfwidth(residuals: np.ndarray, level: float) -> float:
    """Split-conformal absolute-residual quantile with the finite-sample (n+1)/n correction."""
    n = len(residuals)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(np.abs(residuals), q))


def _oof_cell(fold: dict, method: str, seed: int, with_conformal: bool) -> dict:
    """One (fold, method) cell: point prediction on `te`, and a conformal half-width from a
    proper-train/calibration split inside the training fold. Runs in a fork worker."""
    feats, y = _SHARED["feats"], _SHARED["y"]
    pred = fit_predict(method, feats, y, fold["tr"], fold["te"], seed)
    half = np.nan
    if with_conformal:
        resid = fit_predict(method, feats, y, fold["proper"], fold["cal"], seed) - y[fold["cal"]]
        half = conformal_halfwidth(resid, CONFORMAL_LEVEL)
    return {"te": fold["te"], "method": method, "pred": pred, "half": float(half)}


def _gp_cell(fold: dict) -> dict:
    """GP posterior sd on `te` from a fit on the training fold. Runs in a fork worker."""
    feats, y = _SHARED["feats"], _SHARED["y"]
    gp = fit_gp(feats.d2[np.ix_(fold["tr"], fold["tr"])], y[fold["tr"]])
    _, sd = predict_gp(gp, feats.d2[np.ix_(fold["te"], fold["tr"])], include_noise=False)
    return {"te": fold["te"], "sd": sd}


def run_oof(feats: Features, y: np.ndarray, seed: int, methods: tuple[str, ...], with_conformal: bool = True) -> dict:
    """Nested OOF predictions, conformal half-widths and GP sd on one shared fold assignment.

    The (fold, method) cells are independent and dispatched across cores; the GP sd (needed
    only for the resolvability section) is computed per fold and skipped when conformal
    intervals are off. ``with_conformal`` doubles the model fits (each method is refit on a
    proper-train subset to calibrate its interval), so the fold-assignment sensitivity sweep
    -- which only needs the point predictions -- turns it off.
    """
    n = len(y)
    splits = list(KFold(n_splits=N_OUTER, shuffle=True, random_state=seed).split(np.arange(n)))
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in splits:
        perm = rng.permutation(len(tr))
        n_cal = round(CALIB_FRAC * len(tr))
        folds.append({"tr": tr, "te": te, "cal": tr[perm[:n_cal]], "proper": tr[perm[n_cal:]]})

    _SHARED["feats"], _SHARED["y"] = feats, y
    t0 = time.time()
    # batch_size=1: cell costs span 0.5s to ~40s (law_full), so hand out one at a time and let
    # a freed worker grab the next, rather than letting joblib clump long cells onto one worker.
    cells = Parallel(n_jobs=N_JOBS, backend=PARALLEL_BACKEND, batch_size=1)(
        delayed(_oof_cell)(fold, m, seed, with_conformal) for fold in folds for m in methods
    )
    pred = {m: np.full(n, np.nan) for m in methods}
    half = {m: np.full(n, np.nan) for m in methods}
    for c in cells:
        pred[c["method"]][c["te"]] = c["pred"]
        half[c["method"]][c["te"]] = c["half"]

    gp_sd = np.full(n, np.nan)
    if with_conformal:
        for g in Parallel(n_jobs=min(N_JOBS, len(folds)), backend=PARALLEL_BACKEND)(
            delayed(_gp_cell)(fold) for fold in folds
        ):
            gp_sd[g["te"]] = g["sd"]
    print(
        f"    [seed {seed}] OOF over {len(methods)} methods x {len(folds)} folds "
        f"(conformal={with_conformal}) in {time.time() - t0:.1f}s on {N_JOBS} cores",
        flush=True,
    )
    assert all(np.isfinite(v).all() for v in pred.values()), "an out-of-fold prediction is missing"
    return {"pred": pred, "conformal_half": half, "gp_sd": gp_sd, "n_folds": len(folds)}


def score_method(p: np.ndarray, y: np.ndarray, ceiling: float) -> dict:
    rho = float(spearmanr(p, y).statistic)
    return {
        "spearman": rho,
        "pearson": float(pearsonr(p, y).statistic),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "spearman_frac_of_ceiling": rho / ceiling,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap on the DIFFERENCES -- the point of the exercise
# ---------------------------------------------------------------------------


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    return r


def _fast_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson on ordinal ranks, ties broken by position rather than averaged.

    Bootstrap resamples DO contain ties -- only ~530 of 800 rows are unique per replicate --
    but a duplicated row is duplicated in both vectors at the same positions, so the tie
    structure is shared and identical across methods, and it cancels out of a paired
    difference. Checked numerically against scipy's average-rank spearmanr: over 400
    replicates the delta-Spearman agreed to 2.2e-6 and the 95% CI endpoints to 6 decimals.
    """
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def paired_bootstrap(pred: dict[str, np.ndarray], y: np.ndarray, pairs, seed: int = 12345) -> dict:
    """Bootstrap over RUNS with a shared resample index per replicate, so pairs stay paired."""
    rng = np.random.default_rng(seed)
    n = len(y)
    methods = sorted({m for pair in pairs for m in pair})
    rho = {m: np.empty(N_BOOT) for m in methods}
    rmse = {m: np.empty(N_BOOT) for m in methods}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)  # ONE index set, reused by every method: this is the pairing
        yb = y[idx]
        for m in methods:
            pb = pred[m][idx]
            rho[m][b] = _fast_spearman(pb, yb)
            rmse[m][b] = np.sqrt(np.mean((pb - yb) ** 2))
    out = {}
    for a, b in pairs:
        entry = {}
        for stat, draws, point in (
            ("delta_spearman", rho, {m: _fast_spearman(pred[m], y) for m in (a, b)}),
            ("delta_rmse", rmse, {m: float(np.sqrt(np.mean((pred[m] - y) ** 2))) for m in (a, b)}),
        ):
            d = draws[a] - draws[b]
            lo, hi = (float(v) for v in np.percentile(d, [CI_LO, CI_HI]))
            entry[stat] = {
                "point": point[a] - point[b],
                "ci95": [lo, hi],
                "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "boot_frac_positive": float(np.mean(d > 0.0)),
                "boot_sd": float(np.std(d, ddof=1)),
            }
        out[f"{a}__vs__{b}"] = entry
    per_method = {
        m: {
            "spearman_ci95": [float(v) for v in np.percentile(rho[m], [CI_LO, CI_HI])],
            "rmse_ci95": [float(v) for v in np.percentile(rmse[m], [CI_LO, CI_HI])],
        }
        for m in methods
    }
    return {"pairs": out, "per_method_ci": per_method, "n_boot": N_BOOT}


# ---------------------------------------------------------------------------
# Second reading of "uncertainty between methods": are predictions resolvable?
# ---------------------------------------------------------------------------


def resolvability(oof: dict, y: np.ndarray, methods: tuple[str, ...], pairs) -> dict:
    """Do the methods disagree by more than their own predictive error bars?"""
    pred, half = oof["pred"], oof["conformal_half"]
    intervals = {
        m: {
            "mean_halfwidth": float(np.mean(half[m])),
            "coverage": float(np.mean(np.abs(pred[m] - y) <= half[m])),
            "nominal_coverage": CONFORMAL_LEVEL,
        }
        for m in methods
    }
    intervals["kernel"]["gp_mean_posterior_sd"] = float(np.mean(oof["gp_sd"]))
    intervals["kernel"]["gp_coverage_2sd"] = float(np.mean(np.abs(pred["kernel"] - y) <= 2.0 * oof["gp_sd"]))
    out = {}
    for a, b in pairs:
        gap = np.abs(pred[a] - pred[b])
        combined = half[a] + half[b]
        out[f"{a}__vs__{b}"] = {
            "rms_prediction_gap": float(np.sqrt(np.mean(gap**2))),
            "median_prediction_gap": float(np.median(gap)),
            "frac_intervals_disjoint": float(np.mean(gap > combined)),
            "gap_over_mean_combined_halfwidth": float(np.mean(gap) / float(np.mean(combined))),
            "prediction_correlation": float(pearsonr(pred[a], pred[b]).statistic),
        }
    return {"intervals": intervals, "pairwise": out}


# ---------------------------------------------------------------------------
# Contour surfaces on slice A
# ---------------------------------------------------------------------------


def method_surfaces(sl: Slice, data: dict, feats: Features, y: np.ndarray, full_fits: dict) -> dict:
    """Predict the headline methods over slice A's grid, plus the support mask.

    ``full_fits`` are the all-runs fits already computed once for the shape/identifiability
    tables, so the surfaces reuse them instead of refitting.
    """
    ax_x = np.linspace(0.0, sl.xmax, N_GRID)
    ax_y = np.linspace(0.0, sl.ymax, N_GRID)
    xx, yy = np.meshgrid(ax_x, ax_y)
    w_grid = sl.weights(data["p_tok"], xx.ravel(), yy.ravel())
    feasible = np.ones_like(xx, dtype=bool) if sl.per_phase else (xx + yy <= 1.0)
    assert np.allclose(w_grid[feasible.ravel()].sum(axis=2), 1.0, atol=1e-12), "grid mixtures do not sum to 1"
    d2_star = d2_to_train(mixture_sqrt_hists(w_grid, data["v"]), data["sq_train"])
    nn = np.sqrt(d2_star.min(axis=1)).reshape(xx.shape)

    surf = {"kernel": krr_predict(feats.d2, d2_star, y, data["gamma"], ALPHA).reshape(xx.shape)}
    for m in HEADLINE:
        if m == "kernel":
            continue
        f = full_fits[m]
        if m in P3_METHODS:
            surf[m] = f.predict(w_grid).reshape(xx.shape)
        elif m in LAW_METHODS:
            surf[m] = f["fit"].predict(_law_grid_x(m, w_grid, data)).reshape(xx.shape)
        else:
            surf[m] = f.predict(flat_weights(w_grid)).reshape(xx.shape)
    return {
        "ax_x": ax_x,
        "ax_y": ax_y,
        "surfaces": surf,
        "in_support": feasible & (nn <= data["support_radius"]),
        "nn": nn,
    }


def _law_grid_x(method: str, w_grid: np.ndarray, data: dict) -> np.ndarray:
    if method == "law_pooled":
        return pooled_features(w_grid, data["grouping"])
    if method == "law_grouped3":
        return group_features(w_grid, data["grouping3"])
    if method == "law_full":
        return flat_weights(w_grid)
    return group_features(w_grid, data["grouping"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _spines(ax) -> None:
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED)


def figure_contours(sl: Slice, surf: dict, target_label: str, emp: dict, out: Path, caption: str) -> None:
    """f37/f38: one BIG panel per method on the slice-A axes, one shared color scale."""
    mask = surf["in_support"]
    vals = np.concatenate([surf["surfaces"][m][mask] for m in HEADLINE])
    vmin, vmax = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.4))
    xx, yy = np.meshgrid(surf["ax_x"], surf["ax_y"])
    for ax, m in zip(axes.ravel(), HEADLINE, strict=True):
        z = np.where(mask, surf["surfaces"][m], np.nan)
        im = ax.contourf(xx, yy, z, levels=22, cmap=CMAP, vmin=vmin, vmax=vmax, extend="both")
        ax.contour(xx, yy, z, levels=10, colors="#00000055", linewidths=0.7)
        ax.contourf(xx, yy, np.where(mask, np.nan, 1.0), levels=[0.5, 1.5], colors=["#e8e6e2"], zorder=2)
        ax.scatter(sl.run_x, sl.run_y, s=8, c="#111111", alpha=0.30, linewidths=0, zorder=3, label="800 train runs")
        ax.scatter(
            [sl.anchor[0]],
            [sl.anchor[1]],
            marker="X",
            s=200,
            facecolors="white",
            edgecolors=INK,
            linewidths=1.8,
            zorder=4,
            label="token-proportional anchor",
        )
        ax.set_xlim(0, sl.xmax)
        ax.set_ylim(0, sl.ymax)
        ax.set_title(LABEL[m], fontsize=15, pad=10)
        _spines(ax)
    for ax in axes[1, :]:
        ax.set_xlabel(sl.xlabel, fontsize=12.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(sl.ylabel, fontsize=12.5)
    axes[0, 0].legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=11.5)
    fig.subplots_adjust(left=0.065, right=0.885, top=0.885, bottom=0.135, wspace=0.14, hspace=0.20)
    cax = fig.add_axes((0.905, 0.135, 0.019, 0.75))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"predicted {target_label}  (green = lower = better)", fontsize=12.5)
    cb.ax.tick_params(labelsize=11.5)
    fig.suptitle(
        f"f37/f38  Four surrogates on the SAME slice: code_adjacent phase-0 vs phase-1 share   [{target_label}]",
        fontsize=17,
        y=0.965,
    )
    fig.text(
        0.03,
        0.012,
        caption + f"  Empirical reference: {int(emp['count'].sum())} runs fall in this window; "
        f"{int((emp['count'] >= MIN_BIN_N).sum())} of {BIN_GRID * BIN_GRID} bins hold >= {MIN_BIN_N} runs.",
        fontsize=10.5,
        color=MUTED,
        wrap=True,
        va="bottom",
    )
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def figure_paired_deltas(boot: dict, target_label: str, unit: str, out: Path, caption: str) -> None:
    """f39/f40: THE uncertainty-between-methods figure. Paired deltas with bootstrap 95% CIs."""
    keys = list(boot["pairs"])
    labels = [f"{SHORT[k.split('__vs__')[0]]}  -  {SHORT[k.split('__vs__')[1]]}" for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 1.4 + 0.56 * len(keys) + 3.4))
    specs = (
        ("delta_spearman", r"$\Delta$ out-of-fold Spearman", "> 0 : LEFT method ranks better", ""),
        ("delta_rmse", r"$\Delta$ out-of-fold RMSE", "< 0 : LEFT method predicts better", unit),
    )
    ypos = np.arange(len(keys))[::-1]
    for ax, (stat, title, sense, uu) in zip(axes, specs, strict=True):
        for yy_, key in zip(ypos, keys, strict=True):
            e = boot["pairs"][key][stat]
            lo, hi = e["ci95"]
            real = e["excludes_zero"]
            col = "#1b5e6b" if real else "#8a8781"
            ax.plot([lo, hi], [yy_, yy_], color=col, lw=3.6 if real else 2.1, solid_capstyle="round", zorder=2)
            ax.scatter(
                [e["point"]], [yy_], s=125 if real else 80, color=col, zorder=3, edgecolors="white", linewidths=1.1
            )
        ax.axvline(0.0, color="#b23b3b", lw=1.7, ls="--", zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=12.5)
        ax.set_title(f"{title}\n{sense}", fontsize=15, pad=10)
        ax.set_xlabel(f"difference{f' [{uu}]' if uu else ''}, 95% paired-bootstrap CI", fontsize=13)
        ax.grid(axis="x", color=LINE, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        _spines(ax)
    axes[1].tick_params(labelleft=False)
    fig.suptitle(
        f"f39/f40  Are the method gaps REAL?  Paired differences with 95% bootstrap CIs   [{target_label}]",
        fontsize=17,
        y=0.975,
    )
    fig.subplots_adjust(left=0.155, right=0.985, top=0.79, bottom=0.20, wspace=0.06)
    fig.text(0.02, 0.012, caption, fontsize=10.8, color=MUTED, wrap=True, va="bottom")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def figure_scatter(oof: dict, y: np.ndarray, scores: dict, target_label: str, sigma: float, out: Path, cap: str):
    """f41/f42: predicted vs realized per method, with the measured seed-noise band."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 13.0))
    lo = float(min(y.min(), min(oof["pred"][m].min() for m in HEADLINE)))
    hi = float(max(y.max(), max(oof["pred"][m].max() for m in HEADLINE)))
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    line = np.array([lo, hi])
    for ax, m in zip(axes.ravel(), HEADLINE, strict=True):
        ax.fill_between(
            line,
            line - 2 * sigma,
            line + 2 * sigma,
            color="#c9d8dc",
            alpha=0.55,
            zorder=1,
            label=rf"$\pm 2\sigma$ seed noise ($\sigma$={sigma:.4g})",
        )
        ax.plot(line, line, color=INK, lw=1.4, ls="--", zorder=2, label="perfect prediction")
        ax.scatter(oof["pred"][m], y, s=16, color=COLOR[m], alpha=0.5, linewidths=0, zorder=3)
        s = scores[m]
        ax.text(
            0.035,
            0.965,
            f"Spearman {s['spearman']:+.3f}  ({s['spearman_frac_of_ceiling']:.0%} of ceiling)\n"
            f"Pearson  {s['pearson']:+.3f}\nRMSE     {s['rmse']:.4g}  ({s['rmse'] / sigma:.2f}x sigma)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            family="DejaVu Sans Mono",
            bbox={"facecolor": "white", "edgecolor": LINE, "boxstyle": "round,pad=0.45"},
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(LABEL[m], fontsize=14.5, pad=10)
        _spines(ax)
    for ax in axes[1, :]:
        ax.set_xlabel(f"out-of-fold prediction [{target_label}]", fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"realized [{target_label}]", fontsize=13)
    axes[0, 0].legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=11.5)
    fig.suptitle(
        f"f41/f42  Out-of-fold predicted vs realized, against the measured noise floor   [{target_label}]",
        fontsize=17,
        y=0.965,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.125, wspace=0.14, hspace=0.20)
    fig.text(0.02, 0.012, cap, fontsize=10.8, color=MUTED, wrap=True, va="bottom")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def figure_identifiability(ident: dict, curves: dict, out: Path, caption: str) -> None:
    """f43: how well each parametric form is pinned down by 800 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8))
    ax = axes[0]
    for m, s in ident.items():
        spec = np.array(s["spectrum"])
        spec = spec / spec[0]
        ax.semilogy(
            np.arange(1, len(spec) + 1) / len(spec),
            np.clip(spec, 1e-20, None),
            color=COLOR[m],
            lw=2.7,
            label=f"{SHORT[m]}  ({s['n_params']} params, eff. {s['effective_n_params']:.0f})",
        )
    ax.axhline(1e-12, color="#b23b3b", lw=1.4, ls=":", label="numerical-rank cut (1e-12)")
    ax.set_xlabel("eigenvalue index / number of parameters", fontsize=13)
    ax.set_ylabel("normalized eigenvalue of the normal matrix", fontsize=13)
    ax.set_title("Conditioning at the optimum\n(a cliff = directions the 800 runs do not constrain)", fontsize=14.5)
    ax.legend(loc="lower left", fontsize=11.5)
    ax.grid(color=LINE, lw=0.8)
    ax.set_axisbelow(True)
    _spines(ax)

    ax = axes[1]
    for m, c in curves.items():
        pens = np.array(c["penalties"], dtype=float)
        rm = np.array(c["cv_rmse"], dtype=float)
        floor = pens[pens > 0].min() / 10.0 if np.any(pens > 0) else 1e-12
        x = np.where(pens <= 0, floor, pens)
        ax.semilogx(x, rm / rm.min(), color=COLOR[m], lw=2.7, marker="o", ms=5.5, label=SHORT[m])
        j = int(np.argmin(rm))
        ax.scatter([x[j]], [1.0], s=190, facecolors="none", edgecolors=COLOR[m], lw=2.5, zorder=4)
    ax.set_xlabel("regularization strength  (leftmost point = 0, drawn off-scale)", fontsize=13)
    ax.set_ylabel("inner-CV RMSE / its own minimum", fontsize=13)
    ax.set_title("How much does each form lean on regularization?\n(ring = selected)", fontsize=14.5)
    ax.legend(fontsize=11.5)
    ax.grid(color=LINE, lw=0.8)
    ax.set_axisbelow(True)
    _spines(ax)

    fig.suptitle("f43  Are the parametric forms identified by 800 runs?", fontsize=17, y=0.965)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.795, bottom=0.235, wspace=0.22)
    fig.text(0.02, 0.012, caption, fontsize=10.8, color=MUTED, wrap=True, va="bottom")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

CONTOUR_CAPTION = (
    "All four panels share ONE color scale, so a level difference between panels is a real level difference. The "
    "surface is a CONDITIONAL cut: the code_adjacent group's per-phase total is pinned to (x, y), within-group "
    "proportions stay at token-proportional anchor ratios and the other 153 buckets renormalize proportionally. "
    "Grey cells are off-support (nearest train run farther than the train p95 nearest-neighbour Hellinger distance) "
    "and are excluded from the color scale. Dots are the 800 real runs at their own group doses -- the fields are "
    "MODEL OUTPUT, not measurements, and the swarm holds no run at anchor within-group ratios, so the binned "
    "empirical cannot adjudicate a level offset between the surfaces."
)

DELTA_CAPTION = (
    "Each row is one method PAIR (left minus right). Point = the out-of-fold difference over all 800 runs; bar = the "
    "95% percentile interval from a paired bootstrap over runs (one resampled row set per replicate, reused by BOTH "
    "methods, so the pairing is preserved and the CI is on the DIFFERENCE, not on two independent scores). Teal = "
    "the interval excludes zero, i.e. the gap survives run-level resampling; grey = it does not, and the two methods "
    "are not distinguishable on this evidence. The bootstrap resamples runs only: it does not propagate per-run seed "
    "noise, so it answers 'would another draw of 800 mixtures reverse this', not 'would another seed'."
)

SCATTER_CAPTION = (
    "Out-of-fold predictions (nested: every hyperparameter chosen inside the training fold) against realized values, "
    "on the shared fold assignment. The band is +-2 sigma of the MEASURED 10-seed anchor-panel noise -- a point "
    "inside it is predicted as well as a rerun of the same mixture would be. RMSE is also printed in units of that "
    "sigma."
)

IDENT_CAPTION = (
    "Left: eigenvalue spectrum of each form's normal matrix at its optimum, normalized by the top eigenvalue -- for "
    "the Data Mixing Law this is the Gauss-Newton matrix of the nonlinear fit, for P3 the design matrix of its "
    "profiled linear head. Directions below the dotted line are not constrained by the 800 runs at all; the "
    "regularizer, not the data, decides them. Every Data Mixing Law variant carries two exactly-null directions by "
    "construction (per-phase proportions sum to 1, so a constant shift of beta within a phase is absorbed by k). "
    "Right: each form's inner-CV curve normalized by its own minimum -- the penalty scales are NOT comparable across "
    "forms, so read the DEPTH of each optimum (how much the fit degrades unregularized), not the x-position."
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def target_block(feats: Features, name: str, y: np.ndarray, grouping: Grouping) -> dict:
    noise = NOISE[name]
    oof = run_oof(feats, y, PRIMARY_SEED, METHODS)
    scores = {m: score_method(oof["pred"][m], y, noise["ceiling_spearman"]) for m in METHODS}
    boot = paired_bootstrap(oof["pred"], y, PAIRS)
    resolve = resolvability(oof, y, METHODS, PAIRS)

    seeds = {}
    for s in FOLD_SEEDS:
        o = oof if s == PRIMARY_SEED else run_oof(feats, y, s, PAIR_METHODS, with_conformal=False)
        seeds[s] = {m: score_method(o["pred"][m], y, noise["ceiling_spearman"]) for m in PAIR_METHODS}
    across = {
        m: {
            "spearman_mean": float(np.mean([seeds[s][m]["spearman"] for s in FOLD_SEEDS])),
            "spearman_min": float(np.min([seeds[s][m]["spearman"] for s in FOLD_SEEDS])),
            "spearman_max": float(np.max([seeds[s][m]["spearman"] for s in FOLD_SEEDS])),
        }
        for m in PAIR_METHODS
    }
    delta_across = {
        f"{a}__vs__{b}": {
            "delta_spearman_per_fold_seed": [seeds[s][a]["spearman"] - seeds[s][b]["spearman"] for s in FOLD_SEEDS],
            "sign_stable": bool(
                len({np.sign(seeds[s][a]["spearman"] - seeds[s][b]["spearman"]) for s in FOLD_SEEDS}) == 1
            ),
        }
        for a, b in PAIRS
    }

    # One parallel batch of all-runs fits, reused for shape, identifiability AND the surfaces.
    full = fit_full_batch((*P3_METHODS, *LAW_METHODS), feats, y, PRIMARY_SEED)
    ident = {
        "p3_epoch": p3_identifiability(full["p3_epoch"], feats.w),
        "law_grouped": law_identifiability(full["law_grouped"]["fit"], feats.law_x("law_grouped")),
        "law_full": law_identifiability(full["law_full"]["fit"], feats.law_x("law_full")),
    }
    curves = {m: full[m]["penalty_curve"] for m in ("law_grouped", "law_full")}
    curves["p3_epoch"] = full["p3_epoch"].penalty_curve

    p3_shape = {m: _p3_shape(full[m], feats, grouping) for m in P3_METHODS}
    law_shape = {}
    for m in LAW_METHODS:
        f = full[m]["fit"]
        span = f.exponent_range[1] - f.exponent_range[0]
        law_shape[m] = {
            "c": f.c,
            "k": f.k,
            "penalty": f.penalty,
            "exponent_range": list(f.exponent_range),
            "exponent_span": float(span),
            "curvature_ratio": float(np.exp(span)),
            "beta_norm": float(np.linalg.norm(f.beta)),
            "n_beta": len(f.beta),
        }
    law_shape["law_grouped"]["beta_per_domain"] = _grouped_beta(full["law_grouped"]["fit"], grouping)

    return {
        "target": name,
        "noise": noise,
        "scores": scores,
        "bootstrap": boot,
        "resolvability": resolve,
        "fold_seed_sensitivity": {"per_method": across, "per_pair": delta_across, "seeds": list(FOLD_SEEDS)},
        "identifiability": ident,
        "penalty_curves": curves,
        "p3_shape": p3_shape,
        "law_shape": law_shape,
        "_oof": oof,
        "_full_fits": {**full, "linear_ridge": fit_full("linear_ridge", feats, y, PRIMARY_SEED)},
    }


def _p3_shape(fit: P3Fit, feats: Features, grouping: Grouping) -> dict:
    """P3's fitted coefficients, with beta read back at the group level for interpretability."""
    par = fit.parameters()
    beta = par["beta"]
    exposure = np.clip(feats.w[:, 0, :] + fit.eta * feats.w[:, 1, :], 0.0, None) ** fit.a
    pen0 = np.sum(np.clip(fit.c0[None, :] * feats.w[:, 0, :], 0.0, None) ** fit.p, axis=1)
    pen1 = np.sum(np.clip(fit.c1[None, :] * feats.w[:, 1, :], 0.0, None) ** fit.p, axis=1)
    benefit = exposure @ beta
    return {
        "eta": par["eta"],
        "a": par["a"],
        "p": par["p"],
        "b": par["b"],
        "gamma0": par["gamma0"],
        "gamma1": par["gamma1"],
        "ridge": par["ridge"],
        "beta_norm": float(np.linalg.norm(beta)),
        "beta_token_weighted_by_domain": {
            n: float(beta[grouping.membership == i].mean()) for i, n in enumerate(grouping.names)
        },
        "term_sd_over_runs": {
            "exposure_term": float(np.std(benefit)),
            "penalty0_term": float(np.std(par["gamma0"] * pen0)),
            "penalty1_term": float(np.std(par["gamma1"] * pen1)),
        },
        "penalty_share_of_prediction_variance": float(
            np.var(par["gamma0"] * pen0 + par["gamma1"] * pen1) / max(np.var(fit.predict(feats.w)), 1e-30)
        ),
    }


def _grouped_beta(fit: LawFit, grouping: Grouping) -> dict:
    """beta of the grouped law, per-phase mean-centered to fix the sum-to-one gauge."""
    n = grouping.n_domains
    out = {}
    for p in range(2):
        b = fit.beta[p * n : (p + 1) * n]
        out[f"phase{p}"] = {name: float(v) for name, v in zip(grouping.names, b - b.mean(), strict=True)}
    return out


def main() -> None:
    data = load_data()
    grouping = build_grouping(data["cluster_id"], data["p_tok"], N_SOLO_CLUSTERS)
    grouping3 = build_grouping3(data["cluster_id"])
    data["grouping"], data["grouping3"] = grouping, grouping3
    bt = pd.read_parquet(BUCKETS_TABLE).set_index("bucket")
    t_tokens = bt.loc[sorted(bt.index), "total_tokens"].to_numpy(dtype=np.float64)
    assert len(t_tokens) == data["w"].shape[2], "bucket token totals do not align with the weight matrix"
    c0, c1 = epoch_constants(t_tokens)
    feats = Features(
        d2=data["d2"],
        w=data["w"],
        c0=c0,
        c1=c1,
        grouped=group_features(data["w"], grouping),
        pooled=pooled_features(data["w"], grouping),
        grouped3=group_features(data["w"], grouping3),
    )
    epochs = c0[None, :] * data["w"][:, 0, :] + c1[None, :] * data["w"][:, 1, :]
    peak = epochs.max(axis=1)
    print(f"epoch check: per-run peak epochs -- median {np.median(peak):.2f}, max {peak.max():.2f}")
    slice_a = build_slices(data["p_tok"], data["cluster_id"], data["w"])[0]

    results = {}
    for name, label, unit, figs in (
        ("zmacro", "zmacro (z-scored bpb)", "z", ("f37", "f39", "f41")),
        ("humaneval", "humaneval 10-shot bpb", "bpb", ("f38", "f40", "f42")),
    ):
        y = data["targets"][name]
        print(f"\n=== {label}: n={len(y)} runs, {N_OUTER}-fold nested OOF, {len(METHODS)} methods ===")
        block = target_block(feats, name, y, grouping)
        for m in METHODS:
            s = block["scores"][m]
            print(
                f"  {SHORT[m]:>12}  rho {s['spearman']:+.4f} ({s['spearman_frac_of_ceiling']:.1%} of ceiling)"
                f"  r {s['pearson']:+.4f}  RMSE {s['rmse']:.5f}"
            )
        surf = method_surfaces(slice_a, data, feats, y, block["_full_fits"])
        emp = bin_empirical(slice_a, y)
        figure_contours(slice_a, surf, label, emp, FIGS / f"{figs[0]}_method_contours_{name}.png", CONTOUR_CAPTION)
        figure_paired_deltas(
            block["bootstrap"], label, unit, FIGS / f"{figs[1]}_paired_deltas_{name}.png", DELTA_CAPTION
        )
        figure_scatter(
            block["_oof"],
            y,
            block["scores"],
            label,
            block["noise"]["sigma"],
            FIGS / f"{figs[2]}_pred_vs_realized_{name}.png",
            SCATTER_CAPTION,
        )
        block["slice_a_surface_agreement"] = _surface_agreement(surf)
        del block["_oof"], block["_full_fits"]
        results[name] = block

    figure_identifiability(
        results["zmacro"]["identifiability"],
        results["zmacro"]["penalty_curves"],
        FIGS / "f43_form_identifiability.png",
        IDENT_CAPTION,
    )

    payload = {
        "protocol": {
            "n_runs": len(data["targets"]["zmacro"]),
            "methods": list(METHODS),
            "pair_methods": list(PAIR_METHODS),
            "p3_form": (
                "yhat(w0,w1) = b + sum_d beta_d (w0_d + eta w1_d)^a - gamma0 sum_d (c0_d w0_d)^p "
                "- gamma1 sum_d (c1_d w1_d)^p   [marin#2403, Calvin-Xu 2026-05-05, attributed to Will]"
            ),
            "p3_c_constants": {
                "p3_epoch": (
                    "c_{p,d} = f_p * B / T_d so that c_{p,d} * w_{p,d} IS the per-phase epoch count "
                    "(f = phase token fractions, B = target budget 10.372e12, T_d = bucket token total); identical to "
                    "swoosh_form.per_phase_epochs / grug_fit.stage_b_epochs"
                ),
                "p3_flat": "c_{p,d} = 1: pure concentration penalty with no data-volume scaling",
                "ambiguity": "#2403 does not define c; both readings are fitted and reported separately",
            },
            "p3_fit": (
                "linear head (b, beta, gamma0>=0, gamma1>=0) profiled out in closed form by bounded ridge "
                "least squares on standardized columns; the 3 nonlinear coefficients (eta, a, p) by multi-start "
                "bounded Powell on the inner-CV SSE; head ridge grid includes 0 so the unregularized form is reachable"
            ),
            "p3_bounds": {"eta": list(P3_BOUNDS[0]), "a": list(P3_BOUNDS[1]), "p": list(P3_BOUNDS[2])},
            "p3_ridge_grid": list(P3_RIDGES),
            "law_form": "L(w) = c + k * exp(beta^T w)  [Ye et al. 2024, arXiv:2403.16952, Eq. 5, scalar target]",
            "law_fit": (
                "penalized nonlinear least squares (scipy least_squares, trf, analytic Jacobian); ridge "
                "penalty on beta only, selected by inner CV; the penalty also fixes the sum-to-one gauge"
            ),
            "grouping": {
                "name": grouping.name,
                "domains": grouping.names,
                "n_domains": grouping.n_domains,
                "buckets_per_domain": {n: int((grouping.membership == i).sum()) for i, n in enumerate(grouping.names)},
                "token_share_per_domain": {
                    n: float(data["p_tok"][grouping.membership == i].sum()) for i, n in enumerate(grouping.names)
                },
            },
            "grouping3": {"name": grouping3.name, "domains": grouping3.names},
            "kernel": {"gamma_factor": GAMMA_FACTOR, "alpha": ALPHA, "gamma_full_data": data["gamma"]},
            "linear_ridge": "RidgeCV(logspace(-3,3,25)) on the raw 2x168 per-phase bucket proportions",
            "regmix_gbdt": LIGHTGBM_NOTE,
            "grp": GRP_NOTE,
            "cv": {
                "outer_folds": N_OUTER,
                "inner_folds": N_INNER,
                "primary_fold_seed": PRIMARY_SEED,
                "fold_seeds_checked": list(FOLD_SEEDS),
                "paired": (
                    "one fold assignment shared by every method; bootstrap reuses one resample index per "
                    "replicate across methods"
                ),
            },
            "bootstrap": {"n_boot": N_BOOT, "ci": [CI_LO, CI_HI]},
            "conformal": {"level": CONFORMAL_LEVEL, "calibration_fraction": CALIB_FRAC},
            "parallelism": (
                f"per-(fold, method) OOF fits and the all-runs fits dispatched over {N_JOBS} cores via "
                f"joblib backend '{PARALLEL_BACKEND}' (fork; workers inherit the lightgbm import stub); the paired "
                "bootstrap re-scores precomputed OOF predictions and runs single-threaded"
            ),
        },
        "results": results,
    }
    (GRUG / "method_comparison.json").write_text(json.dumps(payload, indent=1) + "\n")
    (GRUG / "method_comparison.md").write_text(render_markdown(payload))
    print("\nwrote", GRUG / "method_comparison.json", "and", GRUG / "method_comparison.md")
    register_manifest()


def _surface_agreement(surf: dict) -> dict:
    mask = surf["in_support"]
    out = {}
    for i, a in enumerate(HEADLINE):
        for b in HEADLINE[i + 1 :]:
            u, v = surf["surfaces"][a][mask], surf["surfaces"][b][mask]
            out[f"{a}__vs__{b}"] = {
                "pearson": float(pearsonr(u, v).statistic),
                "spearman": float(spearmanr(u, v).statistic),
                "rms_difference": float(np.sqrt(np.mean((u - v) ** 2))),
            }
    for m in HEADLINE:
        out[f"{m}__range_in_support"] = [
            float(surf["surfaces"][m][mask].min()),
            float(surf["surfaces"][m][mask].max()),
        ]
    out["frac_grid_off_support"] = float(1.0 - mask.mean())
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(payload: dict) -> str:
    p = payload["protocol"]
    g = p["grouping"]
    out = [
        "# Head-to-head: Hellinger kernel vs the #2403 P3 form vs a linear RegMix baseline",
        "",
        f"- {p['n_runs']} train runs, {p['cv']['outer_folds']}-fold nested out-of-fold "
        f"(inner {p['cv']['inner_folds']}-fold for every hyperparameter), one fold assignment shared by all methods.",
        f"- **P3 form (primary competitor, marin#2403):** `{p['p3_form']}`.",
        f"  - `p3_epoch`: {p['p3_c_constants']['p3_epoch']}.",
        f"  - `p3_flat`: {p['p3_c_constants']['p3_flat']}.",
        f"  - {p['p3_c_constants']['ambiguity']}.",
        f"  - fit: {p['p3_fit']}.",
        f"- Data Mixing Law (secondary): `{p['law_form']}`; {p['law_fit']}.",
        f"- law grouping ({g['n_domains']} domains, `{g['name']}`): "
        + ", ".join(
            f"{n} ({g['buckets_per_domain'][n]} buckets, {g['token_share_per_domain'][n]:.1%})" for n in g["domains"]
        )
        + ".",
        f"- **{p['regmix_gbdt']}**",
        f"- **{p['grp']}**",
        f"- paired bootstrap: {p['bootstrap']['n_boot']} resamples over runs, one resample index per replicate "
        "reused by every method.",
        "",
    ]
    for name, res in payload["results"].items():
        n = res["noise"]
        out += [
            f"## {name}",
            "",
            f"Measured seed noise sigma = {n['sigma']:.4g} against signal spread {n['signal_sd']:.4g}"
            f" => reliability {1 - (n['sigma'] / n['signal_sd']) ** 2:.3f}, ceiling Spearman"
            f" **{n['ceiling_spearman']:.4f}**.",
            "",
            "### out-of-fold scores",
            "",
            "| method | Spearman | % of ceiling | Pearson | RMSE | RMSE / sigma | Spearman 95% CI |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in p["methods"]:
            s = res["scores"][m]
            ci = res["bootstrap"]["per_method_ci"].get(m)
            ci_txt = f"[{ci['spearman_ci95'][0]:+.4f}, {ci['spearman_ci95'][1]:+.4f}]" if ci else "-"
            out.append(
                f"| {SHORT[m]} | {s['spearman']:+.4f} | {s['spearman_frac_of_ceiling']:.1%} | {s['pearson']:+.4f} "
                f"| {s['rmse']:.5f} | {s['rmse'] / n['sigma']:.2f} | {ci_txt} |"
            )
        out += [
            "",
            "### paired differences (the point of this table)",
            "",
            "`left - right`. A CI that excludes zero means the gap survives resampling the 800 runs.",
            "",
            "| pair | dSpearman | 95% CI | real? | dRMSE | 95% CI | real? | sign stable over fold seeds |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key, e in res["bootstrap"]["pairs"].items():
            a, b = key.split("__vs__")
            ds, dr = e["delta_spearman"], e["delta_rmse"]
            stable = res["fold_seed_sensitivity"]["per_pair"][key]["sign_stable"]
            out.append(
                f"| {SHORT[a]} - {SHORT[b]} | {ds['point']:+.4f} | [{ds['ci95'][0]:+.4f}, {ds['ci95'][1]:+.4f}] "
                f"| {'**YES**' if ds['excludes_zero'] else 'no'} | {dr['point']:+.5f} "
                f"| [{dr['ci95'][0]:+.5f}, {dr['ci95'][1]:+.5f}] | {'**YES**' if dr['excludes_zero'] else 'no'} "
                f"| {'yes' if stable else 'NO'} |"
            )
        out += [
            "",
            "### P3's fitted shape",
            "",
            "| variant | eta | a | p | gamma0 | gamma1 | penalty share of pred. var | head ridge |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m, s in res["p3_shape"].items():
            out.append(
                f"| {SHORT[m]} | {s['eta']:.4g} | {s['a']:.4g} | {s['p']:.4g} | {s['gamma0']:.4g} | "
                f"{s['gamma1']:.4g} | {s['penalty_share_of_prediction_variance']:.1%} | {s['ridge']:.3g} |"
            )
        out += ["", "### predictive uncertainty and whether the methods are distinguishable per run", ""]
        iv = res["resolvability"]["intervals"]
        out += [
            f"| method | conformal half-width ({p['conformal']['level']:.0%}) | realized coverage |",
            "|---|---|---|",
        ]
        for m in p["methods"]:
            out.append(f"| {SHORT[m]} | {iv[m]['mean_halfwidth']:.5f} | {iv[m]['coverage']:.3f} |")
        out += [
            "",
            f"The kernel additionally has a GP posterior sd: mean {iv['kernel']['gp_mean_posterior_sd']:.5f}, "
            f"realized +-2sd coverage {iv['kernel']['gp_coverage_2sd']:.3f}.",
            "",
            "| pair | RMS prediction gap | gap / combined half-width | conformal intervals disjoint |",
            "|---|---|---|---|",
        ]
        for key, e in res["resolvability"]["pairwise"].items():
            a, b = key.split("__vs__")
            out.append(
                f"| {SHORT[a]} - {SHORT[b]} | {e['rms_prediction_gap']:.5f} | "
                f"{e['gap_over_mean_combined_halfwidth']:.3f} | {e['frac_intervals_disjoint']:.1%} |"
            )
        out += [
            "",
            "### Data Mixing Law shape (secondary)",
            "",
            "| variant | c | k | exponent span | exp(span) | \\|\\|beta\\|\\| | n_beta |",
            "|---|---|---|---|---|---|---|",
        ]
        for m, s in res["law_shape"].items():
            out.append(
                f"| {SHORT[m]} | {s['c']:+.4g} | {s['k']:+.4g} | {s['exponent_span']:.4g} | "
                f"{s['curvature_ratio']:.3g} | {s['beta_norm']:.4g} | {s['n_beta']} |"
            )
        out += [
            "",
            "### identifiability",
            "",
            "| form | params | numeric rank | effective params | cond. number | selected regularization |",
            "|---|---|---|---|---|---|",
        ]
        for m, s in res["identifiability"].items():
            out.append(
                f"| {SHORT[m]} | {s['n_params']} | {s['rank_numeric']} | {s['effective_n_params']:.1f} | "
                f"{s['condition_number_nonzero']:.3g} | {s['selected_penalty']:.3g} |"
            )
        out += ["", "### slice-A surface agreement (the f37/f38 axes)", ""]
        sa = res["slice_a_surface_agreement"]
        out.append(f"- {sa['frac_grid_off_support']:.1%} of the slice-A grid is off-support and excluded.")
        for key in [k for k in sa if k.endswith("__range_in_support")]:
            m = key.replace("__range_in_support", "")
            out.append(f"- {SHORT[m]} in-support range: {sa[key][0]:.4f} .. {sa[key][1]:.4f}.")
        for key in [k for k in sa if "__vs__" in k]:
            a, b = key.split("__vs__")
            e = sa[key]
            out.append(
                f"- {SHORT[a]} vs {SHORT[b]}: Pearson {e['pearson']:+.3f}, Spearman {e['spearman']:+.3f}, "
                f"RMS level difference {e['rms_difference']:.4f}."
            )
        out += ["", *findings(res, p), ""]
    return "\n".join(out) + "\n"


def _pair_name(key: str) -> str:
    a, b = key.split("__vs__")
    return f"{SHORT[a]}-{SHORT[b]}"


def findings(res: dict, p: dict) -> list[str]:
    """Read the tables back as prose, every number pulled from the results."""
    sc, boot = res["scores"], res["bootstrap"]
    ceiling = res["noise"]["ceiling_spearman"]
    order = sorted(p["methods"], key=lambda m: -sc[m]["spearman"])
    lines = ["### what the numbers say", ""]
    lines.append(
        "- ranking by out-of-fold Spearman: "
        + " > ".join(f"{SHORT[m]} {sc[m]['spearman']:+.3f}" for m in order)
        + f" (ceiling {ceiling:.3f})."
    )
    real = [k for k, e in boot["pairs"].items() if e["delta_spearman"]["excludes_zero"]]
    notreal = [k for k, e in boot["pairs"].items() if not e["delta_spearman"]["excludes_zero"]]
    lines.append(
        f"- of {len(boot['pairs'])} method pairs, {len(real)} have a delta-Spearman CI that EXCLUDES zero "
        f"({', '.join(_pair_name(k) for k in real) or 'none'}) and {len(notreal)} do not "
        f"({', '.join(_pair_name(k) for k in notreal) or 'none'})."
    )
    for key, headline in (
        ("kernel__vs__p3_epoch", "kernel vs the #2403 P3 form (epoch-scaled penalty)"),
        ("kernel__vs__p3_flat", "kernel vs P3 with a flat (c=1) penalty"),
        ("p3_epoch__vs__linear_ridge", "P3 vs the linear RegMix stand-in"),
        ("kernel__vs__law_grouped", "kernel vs the Data Mixing Law (grouped)"),
    ):
        e = boot["pairs"].get(key)
        if not e:
            continue
        a, b = key.split("__vs__")
        ds = e["delta_spearman"]
        verdict = "REAL (CI excludes zero)" if ds["excludes_zero"] else "NOT resolvable (CI covers zero)"
        lines.append(
            f"- {headline}: dSpearman {ds['point']:+.4f} [{ds['ci95'][0]:+.4f}, {ds['ci95'][1]:+.4f}] -- {verdict}. "
            f"{SHORT[a]} reaches {sc[a]['spearman_frac_of_ceiling']:.0%} of the noise ceiling, "
            f"{SHORT[b]} {sc[b]['spearman_frac_of_ceiling']:.0%}."
        )
    amb = boot["pairs"].get("p3_epoch__vs__p3_flat", {}).get("delta_spearman")
    if amb:
        lines.append(
            f"- the c_d ambiguity in P3 matters {'MATERIALLY' if amb['excludes_zero'] else 'little'}: "
            f"epoch-scaled minus flat dSpearman {amb['point']:+.4f} [{amb['ci95'][0]:+.4f}, {amb['ci95'][1]:+.4f}]."
        )
    for m, s in res["p3_shape"].items():
        lines.append(
            f"- {SHORT[m]} fitted shape: eta {s['eta']:.3g} (phase-1 token counts {s['eta']:.2g}x a phase-0 one), "
            f"a {s['a']:.3g} ({'saturating' if s['a'] < 1 else 'expansive'} returns to dose), p {s['p']:.3g}; the two "
            f"penalty terms carry {s['penalty_share_of_prediction_variance']:.1%} of the prediction variance "
            f"(gamma0 {s['gamma0']:.3g}, gamma1 {s['gamma1']:.3g})."
        )
    ident = res["identifiability"]["law_full"]
    lines.append(
        f"- full-bucket Data Mixing Law identifiability: {ident['n_params']} parameters against {ident['n_runs']} "
        f"runs; the data-fit normal matrix has numeric rank {ident['rank_numeric']} and the selected penalty "
        f"{ident['selected_penalty']:.3g} leaves only {ident['effective_n_params']:.1f} effective parameters -- beta "
        "is decided by the penalty, not the runs."
    )
    shape = res["law_shape"]["law_grouped"]
    lines.append(
        f"- how much curvature the Data Mixing Law actually uses: its exponent spans {shape['exponent_span']:.3g} "
        f"over the 800 runs, i.e. exp() bends by a factor {shape['curvature_ratio']:.3g} across the whole design. "
        "Near a span of zero the law degenerates to the linear model, so a small span is itself the explanation for "
        "any law-vs-linear tie."
    )
    worst = max(res["resolvability"]["pairwise"].items(), key=lambda kv: kv[1]["frac_intervals_disjoint"])
    lines.append(
        f"- per-run resolvability: the most-separated pair is {_pair_name(worst[0])}, whose conformal intervals are "
        f"disjoint on {worst[1]['frac_intervals_disjoint']:.1%} of runs. Where that fraction is small the methods "
        "disagree by less than their own error bars, so no single run can adjudicate between them."
    )
    return lines


def register_manifest() -> None:
    path = FIGS / "manifest3.json"
    manifest = json.loads(path.read_text())
    source = (
        "train_runs.parquet (800) + buckets_table.parquet + the frozen K=1000 content basis; builder "
        "experiments/datakit/mixture_features/method_comparison.py"
    )
    common = (
        "Head-to-head of the frozen Hellinger kernel ridge, the marin#2403 P3 parametric form "
        "(b + sum beta_d (w0_d + eta w1_d)^a - gamma_p sum (c_pd w_pd)^p, fitted in both the epoch-scaled and "
        "flat readings of c), a linear ridge on the raw bucket proportions (RegMix stand-in; RegMix's LightGBM "
        "could not be run -- no libgomp on this host) and the Ye et al. 2024 Data Mixing Law as a secondary "
        "baseline. GRP was not run: no faithful implementation exists for this bucket taxonomy. "
    )
    entries = {
        "f37_method_contours_zmacro.png": (
            common
            + "Four BIG contour panels on the slice-A axes (code_adjacent phase-0 vs phase-1 share), one per method, "
            "one shared color scale, off-support cells greyed. Colors are model predictions, not measurements. "
            "Target: zmacro."
        ),
        "f38_method_contours_humaneval.png": (
            common + "Same four-panel slice-A contour comparison against humaneval 10-shot bpb."
        ),
        "f39_paired_deltas_zmacro.png": (
            common
            + "THE uncertainty-between-methods figure for zmacro: per method PAIR, the out-of-fold delta-Spearman and "
            "delta-RMSE with 95% paired-bootstrap CIs (4000 resamples over runs, one shared resample index per "
            "replicate). Teal = CI excludes zero (real gap), grey = it does not."
        ),
        "f40_paired_deltas_humaneval.png": common + "Same paired-difference CI figure against humaneval 10-shot bpb.",
        "f41_pred_vs_realized_zmacro.png": (
            common
            + "Out-of-fold predicted vs realized per method for zmacro, with the measured +-2 sigma seed-noise band "
            "(sigma = 0.2127 z from the 10-seed anchor panel) and each method's Spearman as a fraction of the 0.909 "
            "reliability ceiling."
        ),
        "f42_pred_vs_realized_humaneval.png": (
            common
            + "Same predicted-vs-realized comparison against humaneval 10-shot bpb (seed sigma 0.00565 bpb, ceiling "
            "0.996)."
        ),
        "f43_form_identifiability.png": (
            "Are the parametric forms identified by 800 runs? Eigenvalue spectrum of each form's normal matrix at "
            "its optimum (P3's profiled linear head; the Data Mixing Law's Gauss-Newton matrix, grouped and "
            "full-bucket) plus each form's inner-CV regularization curve normalized by its own minimum. The "
            "full-bucket law carries 338 parameters against 800 runs and every law variant has two exactly-null "
            "directions by construction (per-phase proportions sum to 1, so a constant shift of beta within a "
            "phase is absorbed by k)."
        ),
    }
    for fname, message in entries.items():
        manifest[fname] = {"message": message, "data_source": source}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("updated", path)


if __name__ == "__main__":
    main()
