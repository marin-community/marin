# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SNR probe: would weighting each direction by its gradient SNR beat Muon's equal weights?

ANVIL III (Hyperstition, "Cutting pretraining costs by 62%") replaces Muon's all-ones singular weights with an
Adam-style signal-to-noise ratio per direction, ``EMA(eᵢᵀGeⱼ) / sqrt(EMA((eᵢᵀGeⱼ)²))``. Under MuonH the hyperball
fixes each step's Frobenius size, so such a weighting only moves step budget from noisy directions to consistent
ones. This probe measures whether that pays off, before any optimizer is built.

At one training step N (``snr_probe_step``):

- **Window.** For the ``WINDOW`` steps before N, the gradient of the captured matrices (``grad_capture``) on a fresh
  probe batch at the run's current parameters: the stream an optimizer would see, one step at a time.
- **Candidates** built from that window, each a descent direction ``D`` for every captured matrix
  (``candidate_directions``): momentum (the bias-corrected EMA ``M``), Muon (Newton-Schulz of ``M``), elementwise
  SNR, SNR in the eigenbasis of the window's Kronecker factors (SOAP-like), and SNR in the singular basis of ``M``
  (full, and diagonal only). Every EMA uses the same ``BETA`` in numerator and denominator, so each SNR is at most 1.
- **Held-out payoff.** ``⟨G_b, D/‖D‖⟩`` on ``HELDOUT_BATCHES`` unseen batches at step N (first order, per unit
  Frobenius step).
- **Loss.** The real held-out loss after moving one family's matrices by ``-k θ ‖W‖ D/‖D‖`` and projecting back
  onto ``‖W‖`` (the MuonH step), where ``θ`` is each matrix's own relative step at N, on ``EVAL_BATCHES`` batches.
  This keeps the curvature that a first-order score leaves out (which would otherwise favour plain momentum).

Probe batches come from past the run's last step, so the run's data and trajectory are unchanged. Results go to
``snr_probe_step<N>.npz``.
"""

import functools
import io
import logging
import re
from collections.abc import Callable, Iterator

import fsspec
import jax
import numpy as np

from experiments.grug.fast_track.stiefel import MSIGN_STEPS, _msign

logger = logging.getLogger(__name__)

WINDOW = 48
BETA = 0.95
HELDOUT_BATCHES = 64
EVAL_BATCHES = 16
STEP_SCALES = (1.0, 4.0)
PROBE_FILE = "snr_probe_step{step}.npz"
# Momentum/gradient agreement: Muon's direction plus a multiple of ``P sym(Pᵀ g)``, with ``P = NS(M)`` and ``g`` the
# newest gradient. For ``P = U Vᵀ`` that term is ``U sym(Uᵀ g V) Vᵀ``: it boosts the momentum's singular directions
# the current gradient confirms and damps those it contradicts, with no SVD. Scaled to ``alpha ‖P‖``.
AGREEMENT_ALPHAS = (0.25, 0.5, 1.0)
CANDIDATES = (
    "momentum",
    "muon",
    "adam_snr",
    "eig_snr",
    "svd_snr",
    "svd_snr_diag",
    *(f"muon_agree{alpha:g}" for alpha in AGREEMENT_ALPHAS),
)
# Families whose matrices train with MuonH in the probed recipe (the routers train with Adam).
FAMILIES = {
    "kda": r"\.kda\.",
    "mla": r"\.mla\.",
    "latent": r"\.latent\.",
    "shared": r"\.shared\.",
    "expert": r"\.expert\d",
}
_EPS = 1e-30


def ema_weights(length: int, beta: float) -> np.ndarray:
    """Bias-corrected EMA weights over ``length`` steps, oldest first, summing to 1."""
    w = (1.0 - beta) * beta ** np.arange(length - 1, -1, -1, dtype=np.float64)
    return w / w.sum()


def _snr(stream: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``EMA(x) / sqrt(EMA(x²))`` along the leading (time) axis."""
    mean = np.tensordot(w, stream, axes=1)
    second = np.tensordot(w, np.square(stream), axes=1)
    return mean / np.sqrt(second + _EPS)


def candidate_directions(
    grads: np.ndarray, beta: float, ns: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    """Each candidate's descent direction for one matrix from its gradient window ``grads`` ``[T, m, n]`` (oldest
    first). ``ns`` is Muon's orthogonalization."""
    g = grads.astype(np.float64)
    w = ema_weights(len(g), beta)
    momentum = np.tensordot(w, g, axes=1)
    weighted = g * w[:, None, None]
    left = np.tensordot(weighted, g, axes=([0, 2], [0, 2]))
    right = np.tensordot(weighted, g, axes=([0, 1], [0, 1]))
    q_left = np.linalg.eigh(left)[1]
    q_right = np.linalg.eigh(right)[1]
    eig_snr = q_left @ _snr(q_left.T @ g @ q_right, w) @ q_right.T
    u, _, vt = np.linalg.svd(momentum, full_matrices=False)
    projected = u.T @ g @ vt.T
    snr = _snr(projected, w)
    muon = ns(momentum)
    agree = muon_agreement_term(muon, g[-1])
    return {
        **{f"muon_agree{alpha:g}": muon + alpha * agree for alpha in AGREEMENT_ALPHAS},
        "momentum": momentum,
        "muon": muon,
        "adam_snr": _snr(g, w),
        "eig_snr": eig_snr,
        "svd_snr": u @ snr @ vt,
        "svd_snr_diag": (u * np.diag(snr)) @ vt,
    }


def muon_agreement_term(muon: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """``P sym(Pᵀ g)`` rescaled to ``‖P‖`` (zero when ``g`` is), for Muon's direction ``P`` and a gradient ``g``."""
    pg = muon.T @ grad
    term = muon @ (0.5 * (pg + pg.T))
    norm = np.linalg.norm(term)
    return term * (np.linalg.norm(muon) / norm) if norm > _EPS else term


def sphere_step(param: np.ndarray, direction: np.ndarray, relative_step: float) -> np.ndarray:
    """The MuonH delta: move ``relative_step ‖W‖`` against ``direction``, then project back onto ``‖W‖``."""
    p = param.astype(np.float64)
    norm = np.linalg.norm(p)
    moved = p - relative_step * norm * direction / max(np.linalg.norm(direction), _EPS)
    return moved * norm / max(np.linalg.norm(moved), _EPS) - p


def family_sites(names, family: str) -> list[str]:
    return [n for n in names if re.search(FAMILIES[family], n)]


@functools.partial(jax.jit, static_argnames=("steps",))
def _msign_device(x: jax.Array, steps: int) -> jax.Array:
    return _msign(x, steps)


class SnrProbe:
    """Runs the probe around ``step`` in the train loop. ``before_step`` gets ``gradients(params, batch, step)``
    (the captured matrices' gradients and values) and ``losses(params, batch, deltas)`` (the held-out loss after
    adding ``deltas`` to the captured matrices); ``batches`` yields probe batches disjoint from the run's own."""

    def __init__(
        self,
        step: int,
        path: str,
        captured_params: Callable,
        batches: Iterator,
        ns_steps: int = MSIGN_STEPS,
    ):
        if step < WINDOW:
            raise ValueError(f"snr_probe_step must be at least the window ({WINDOW}), got {step}")
        self.step, self.path = step, path.rstrip("/")
        self._captured_params, self._batches, self._ns_steps = captured_params, batches, ns_steps
        self._window: list[dict[str, np.ndarray]] = []
        self._params_before: dict[str, np.ndarray] | None = None
        self._shardings: dict = {}

    def before_step(self, current_step: int, params, step_array, gradients: Callable, losses: Callable) -> None:
        if self.step - WINDOW <= current_step < self.step:
            grads, _ = gradients(params, next(self._batches), step_array)
            self._window.append(jax.device_get(grads))
            if current_step == self.step - 1:
                self._params_before = jax.device_get(self._captured_params(params))
        elif current_step == self.step:
            self._run(params, step_array, gradients, losses)

    def _on_device(self, name: str, x: np.ndarray) -> jax.Array:
        """A replicated global copy (every process computes the same host value), laid out like the site's capture."""
        return jax.device_put(np.asarray(x, np.float32), self._shardings[name])

    def _run(self, params, step_array, gradients: Callable, losses: Callable) -> None:
        captured = self._captured_params(params)
        self._shardings = {name: value.sharding for name, value in captured.items()}
        params_now = jax.device_get(captured)

        assert self._params_before is not None and len(self._window) == WINDOW
        names = [n for n in params_now if any(re.search(p, n) for p in FAMILIES.values())]
        out: dict[str, np.ndarray] = {"step": np.asarray(self.step), "candidates": np.asarray(CANDIDATES)}
        directions: dict[str, dict[str, np.ndarray]] = {}
        relative_steps: dict[str, float] = {}
        for name in names:
            window = np.stack([g[name] for g in self._window])
            cands = candidate_directions(
                window,
                BETA,
                lambda x, n=name: np.asarray(jax.device_get(_msign_device(self._on_device(n, x), self._ns_steps))),
            )
            directions[name] = {c: d / max(np.linalg.norm(d), _EPS) for c, d in cands.items()}
            now = params_now[name].astype(np.float64)
            relative_steps[name] = float(np.linalg.norm(now - self._params_before[name]) / np.linalg.norm(now))
            out[f"relative_step/{name}"] = np.asarray(relative_steps[name])
        self._window = []

        payoffs = {name: np.zeros((len(CANDIDATES), HELDOUT_BATCHES)) for name in names}
        for b in range(HELDOUT_BATCHES):
            grads, _ = gradients(params, next(self._batches), step_array)
            grads = jax.device_get(grads)
            for name in names:
                g = grads[name].astype(np.float64)
                for c, cand in enumerate(CANDIDATES):
                    payoffs[name][c, b] = np.sum(g * directions[name][cand])
        for name in names:
            out[f"payoff/{name}"] = payoffs[name]

        eval_batches = [next(self._batches) for _ in range(EVAL_BATCHES)]
        zeros = {n: self._on_device(n, np.zeros(params_now[n].shape)) for n in names}
        out["loss/base"] = np.asarray([float(losses(params, batch, zeros)) for batch in eval_batches])
        for family in FAMILIES:
            sites = family_sites(names, family)
            for cand in CANDIDATES:
                for scale in STEP_SCALES:
                    deltas = dict(zeros)
                    for name in sites:
                        delta = sphere_step(params_now[name], directions[name][cand], scale * relative_steps[name])
                        deltas[name] = self._on_device(name, delta)
                    out[f"loss/{family}/{cand}/x{scale:g}"] = np.asarray(
                        [float(losses(params, batch, deltas)) for batch in eval_batches]
                    )
        buffer = io.BytesIO()
        np.savez(buffer, **out)
        path = f"{self.path}/{PROBE_FILE.format(step=self.step)}"
        if jax.process_index() == 0:
            with fsspec.open(path, "wb") as f:
                f.write(buffer.getvalue())
        logger.info("snr probe: wrote %s", path)
