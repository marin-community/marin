# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained inference + sampling for the mixing-via-embeddings surrogate.

Given a candidate data mixture (per-phase weights over the 168 grug buckets), this predicts its
humaneval bits-per-byte from the mixture's *content* -- reusing the frozen 800-run swarm -- so you
can score and rank mixtures and propose new ones to sweep WITHOUT training anything. Depends only on
numpy/scipy; the frozen model lives in ``model/`` (~1.7 MB).

The model (settled on marin#7067; see README):

    yhat(w) = kernel(h = V.w) + epoch_harm(w)

where ``h = V.w`` is the mixture's per-phase content histogram over a frozen K=1000 codebook and the
predictor is Hellinger-kernel ridge over those histograms. ``epoch_harm`` is an ALWAYS-ON additive
bits-per-byte penalty for over-repeating a source; it is self-gating at the real training budget
(identically 0 over the whole in-support swarm, so it never perturbs in-distribution ranking, and it
fires only for genuine over-repetition -- e.g. heavy weight on a small bucket). Lower yhat is better.

Everything is frozen to the deployed configuration (gamma_factor 0.25, alpha 0.1); the frozen path
reproduces the full research pipeline's out-of-fold Spearman exactly (humaneval 0.938).
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize_scalar

ART = Path(__file__).parent / "model"
JITTER = 1e-10


# --------------------------------------------------------------------------- featurization
def content(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-phase content histogram h = V.w.  w:(..., 2, n_buckets), V:(k, n_buckets) -> (..., 2, k).

    Each phase's h is a probability distribution over the K codebook cells (sums to 1 because w
    sums to 1 per phase and every V column sums to 1)."""
    return np.einsum("...pb,kb->...pk", w, v)


def sq_hellinger(ha: np.ndarray, hb: np.ndarray) -> np.ndarray:
    """Mean-over-phases squared Hellinger distance between two sets of content histograms.

    ha:(na, 2, k), hb:(nb, 2, k) -> (na, nb).  This is the exact distance the kernel uses."""
    d = np.zeros((ha.shape[0], hb.shape[0]))
    for p in range(ha.shape[1]):
        sa, sb = np.sqrt(np.clip(ha[:, p, :], 0.0, None)), np.sqrt(np.clip(hb[:, p, :], 0.0, None))
        d += np.clip(1.0 - sa @ sb.T, 0.0, None)
    return d / ha.shape[1]


# --------------------------------------------------------------------------- the surrogate
class MixtureSurrogate:
    """Frozen content surrogate for humaneval bpb. Load once, then ``predict``/``propose``.

    Args:
        budget_tokens: the REAL token budget the proposed runs will train at. This is a required,
            first-class parameter because it governs the epoch penalty: a source's repetition is
            ``e = w * f_phase * budget / bucket_tokens``, and the penalty's amplitude decays as
            ``(budget/10B)^-0.73``. At the 10B calibration budget the penalty is strongest; by >=100B
            it is dormant. Over the in-support swarm the penalty is identically 0 for any budget in
            1e10..1e12, so this choice never changes in-distribution ranking -- it only sets how hard
            genuine over-repetition (heavy weight on a small bucket) is penalised.
        hidden_dim: model width the runs will train at; the penalty grows as ``(d/512)^1.68``.
        target: which frozen target to predict. Only ``"humaneval_bpb"`` is baked in; pass ``target_y``
            to score any other per-run target you supply.
        target_y: optional (n_train,) array of an external per-run target aligned to the frozen swarm
            (same run order as ``train_w``). Use this to fit the surrogate to a target the package does
            not ship (e.g. macro bpb, or a corpus-perplexity eval computed on the 800 checkpoints).
        with_uncertainty: also fit the Gaussian-process posterior variance (calibrated in-distribution;
            see README for its limits off-support). Adds a one-off Cholesky at load; the mean is
            identical either way.
    """

    def __init__(
        self,
        target: str = "humaneval_bpb",
        *,
        budget_tokens: float,
        hidden_dim: int = 512,
        target_y: np.ndarray | None = None,
        with_uncertainty: bool = True,
        art_dir: Path = ART,
    ):
        self.meta = json.loads((art_dir / "meta.json").read_text())
        self.target = target
        self.budget_tokens = float(budget_tokens)
        self.hidden_dim = int(hidden_dim)
        self.buckets: list[str] = json.loads((art_dir / "buckets.json").read_text())
        self.v = np.load(art_dir / "V.npy").astype(np.float64)  # (k, n_buckets)
        self._train_w = np.load(art_dir / "train_w.npy").astype(np.float64)  # (n, 2, n_buckets)

        if target_y is not None:
            y = np.asarray(target_y, dtype=np.float64)
            if y.shape != (self._train_w.shape[0],):
                raise ValueError(f"target_y must be ({self._train_w.shape[0]},) aligned to train_w, got {y.shape}")
        elif target == "humaneval_bpb":
            y = np.load(art_dir / "train_y_humaneval.npy").astype(np.float64)
        else:
            raise ValueError(f"target {target!r} is not baked in; pass target_y=<per-run array> to use it")

        # epoch-harm primitives: per-bucket token supply and the code/web group masks
        self._tj = np.load(art_dir / "tj.npy").astype(np.float64)  # (n_buckets,) unique tokens per bucket
        self._mask_code = np.load(art_dir / "mask_code.npy").astype(bool)
        self._mask_web = np.load(art_dir / "mask_web.npy").astype(bool)
        self._f0 = float(self.meta["phase_fractions"][0])
        self._hp = self.meta["epoch_harm"]

        ok = np.isfinite(y)
        self._train_w, y = self._train_w[ok], y[ok]
        self._train_h = content(self._train_w, self.v)
        d2 = sq_hellinger(self._train_h, self._train_h)
        self.alpha = float(self.meta["alpha"])
        med = float(np.median(d2[np.triu_indices(len(y), 1)]))
        self.gamma = float(self.meta["gamma_factor"]) / med
        self._p95_nn = float(np.quantile(np.sqrt(np.sort(d2, 1)[:, 1]), 0.95))  # off-support threshold

        # kernel-ridge dual solve on the raw target -- the epoch harm is additive on top, NOT folded
        # into the fit (folding it in wrecks in-distribution ranking, since the swarm has epoch
        # variation but no epoch signal; see README / epoch_kernel_headtohead).
        k = np.exp(-self.gamma * d2)
        k[np.diag_indices_from(k)] += self.alpha
        self._ybar = float(y.mean())
        self._a = np.linalg.solve(k, y - self._ybar)
        self._y = y

        # optional Gaussian-process posterior variance (calibrated in-distribution)
        self._gp = _fit_gp_variance(d2, y) if with_uncertainty else None

    # -- epoch harm -------------------------------------------------------------------------
    def epoch_harm(self, w: np.ndarray) -> np.ndarray:
        """Additive epoch-repetition penalty (bpb) for mixture(s), at this surrogate's budget/width.

        H(w) = amp * [ b_code * sum_{j in code} max(e0_j - tau, 0) + b_web * sum_{j in web} max(e0_j - tau, 0) ]
        with phase-0 epochs e0_j = w0_j * f0 * budget / bucket_tokens and amp = (B/refB)^-0.73 (d/refD)^1.68.
        Returns (n,) for a batch or a scalar for a single (2, n_buckets) mixture. Always applied inside
        ``predict``."""
        w = np.asarray(w, float)
        single = w.ndim == 2
        if single:
            w = w[None]
        hp = self._hp
        e0 = w[:, 0, :] * (self._f0 * self.budget_tokens) / self._tj[None, :]  # (n, n_buckets)
        amp = (self.budget_tokens / hp["ref_budget_tokens"]) ** (-hp["budget_exponent"]) * (
            self.hidden_dim / hp["ref_hidden_dim"]
        ) ** hp["hidden_dim_exponent"]
        past_code = np.maximum(e0 - hp["tau_code"], 0.0) * self._mask_code[None, :]
        past_web = np.maximum(e0 - hp["tau_web"], 0.0) * self._mask_web[None, :]
        harm = (hp["b_code"] * past_code.sum(1) + hp["b_web"] * past_web.sum(1)) * amp
        return float(harm[0]) if single else harm

    # -- prediction -------------------------------------------------------------------------
    def predict(self, w: np.ndarray) -> dict:
        """Predict humaneval bpb for candidate mixture(s), epoch penalty included.

        w: (2, n_buckets) for one mixture or (n, 2, n_buckets) for a batch; each phase must sum to 1.
        Returns a dict of arrays: ``mean`` (predicted bpb = content kernel + epoch harm, lower=better),
        ``content_mean`` (kernel term alone), ``epoch_harm`` (the additive penalty, >0 only for
        over-repetition), ``nn_distance`` (Hellinger distance to the nearest swarm run -- the off-support
        signal), ``in_support`` (nn below the training p95), and ``sd`` (GP posterior sd or None).
        """
        w = np.asarray(w, float)
        single = w.ndim == 2
        if single:
            w = w[None]
        _check_weights(w, len(self.buckets))
        h = content(w, self.v)
        d2s = sq_hellinger(h, self._train_h)  # (n_cand, n_train)
        content_mean = np.exp(-self.gamma * d2s) @ self._a + self._ybar
        harm = self.epoch_harm(w)
        nn = np.sqrt(d2s.min(axis=1))
        out = {
            "mean": content_mean + harm,
            "content_mean": content_mean,
            "epoch_harm": harm,
            "nn_distance": nn,
            "in_support": nn <= self._p95_nn,
            "sd": _gp_sd(self._gp, d2s) if self._gp is not None else None,
        }
        return {k: v[0] if (single and v is not None) else v for k, v in out.items()}

    # -- proposal / sampling ----------------------------------------------------------------
    def anchor(self) -> np.ndarray:
        """The mean training mixture (design centre), (2, n_buckets) -- a sane sampling anchor."""
        return self._train_w.mean(axis=0)

    def sample_candidates(self, n: int, *, concentration: float = 200.0, anchor=None, seed: int = 0) -> np.ndarray:
        """Draw ``n`` candidate mixtures from a Dirichlet around an anchor, per phase.

        Larger ``concentration`` = candidates closer to the anchor (the explored region); smaller =
        more exploratory but more likely off-support. Returns (n, 2, n_buckets)."""
        rng = np.random.default_rng(seed)
        a = self.anchor() if anchor is None else np.asarray(anchor, float)
        nb = len(self.buckets)
        w = np.empty((n, 2, nb))
        for p in range(2):
            w[:, p, :] = rng.dirichlet(concentration * np.maximum(a[p], 1e-6), size=n)
        return w

    def propose(
        self,
        *,
        n: int = 20000,
        concentration: float = 200.0,
        top_k: int = 50,
        in_support_only: bool = True,
        seed: int = 0,
    ) -> dict:
        """Sample a candidate bank, score it (epoch penalty included), and return the best ``top_k``.

        Returns a dict with ``weights`` (top_k, 2, n_buckets), ``mean``, ``content_mean``, ``epoch_harm``,
        ``nn_distance``, ``in_support`` and ``sd`` -- ranked best (lowest predicted bpb) first. Set
        ``in_support_only`` False to allow proposals outside the training p95 (higher, poorly-calibrated
        risk)."""
        w = self.sample_candidates(n, concentration=concentration, seed=seed)
        p = self.predict(w)
        keep = p["in_support"] if in_support_only else np.ones(n, bool)
        idx = np.where(keep)[0]
        idx = idx[np.argsort(p["mean"][idx])[:top_k]]
        return {
            "weights": w[idx],
            "mean": p["mean"][idx],
            "content_mean": p["content_mean"][idx],
            "epoch_harm": p["epoch_harm"][idx],
            "nn_distance": p["nn_distance"][idx],
            "in_support": p["in_support"][idx],
            "sd": None if p["sd"] is None else p["sd"][idx],
            "buckets": self.buckets,
            "target": self.target,
            "budget_tokens": self.budget_tokens,
            "n_sampled": n,
            "n_in_support": int(keep.sum()),
        }


# --------------------------------------------------------------------------- helpers
def _check_weights(w: np.ndarray, nb: int):
    if w.shape[-2:] != (2, nb):
        raise ValueError(f"weights must end in (2, {nb}) = (phases, buckets), got {w.shape}")
    s = w.sum(axis=-1)
    if not np.allclose(s, 1.0, atol=1e-4):
        raise ValueError(f"each phase's weights must sum to 1 (got sums in [{s.min():.4f}, {s.max():.4f}])")
    if (w < -1e-9).any():
        raise ValueError("weights must be non-negative")


def _fit_gp_variance(d2: np.ndarray, y: np.ndarray) -> dict:
    """Fit (sigma_f^2, sigma_n^2) by marginal likelihood at the frozen gamma; returns the posterior."""
    ybar = float(y.mean())
    yc = y - ybar
    med = float(np.median(d2[np.triu_indices(len(y), 1)]))
    gamma = 0.25 / med
    base = np.exp(-gamma * d2)

    def nlml(log_sn2):
        sf2 = max(yc.var(), 1e-6)
        k = sf2 * base + (np.exp(log_sn2) + JITTER) * np.eye(len(y))
        try:
            c = cholesky(k, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        a = cho_solve((c, True), yc)
        return 0.5 * yc @ a + np.log(np.diag(c)).sum()

    res = minimize_scalar(nlml, bounds=(np.log(1e-6), np.log(10.0)), method="bounded")
    sf2, sn2 = max(yc.var(), 1e-6), float(np.exp(res.x))
    k = sf2 * base + (sn2 + JITTER) * np.eye(len(y))
    return {"chol": cholesky(k, lower=True), "sf2": sf2, "sn2": sn2, "gamma": gamma}


def _gp_sd(gp: dict, d2s: np.ndarray) -> np.ndarray:
    k_star = gp["sf2"] * np.exp(-gp["gamma"] * d2s)  # (n_cand, n_train)
    v = solve_triangular(gp["chol"], k_star.T, lower=True)
    var = gp["sf2"] - np.einsum("ij,ij->j", v, v) + gp["sn2"]
    return np.sqrt(np.clip(var, 1e-12, None))
