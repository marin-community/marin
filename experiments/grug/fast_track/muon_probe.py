# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon probe: where does upweighting a gradient's small singular directions pay off?

At one training step θ₀ (``muon_probe_step``), for each captured matrix (``grad_capture.capture_matrices``):

- **Basis.** ``G_A = U Σ Vᵀ``, the gradient averaged over ``BASIS_BATCHES`` batches. Muon steps along
  ``-η Σ u_i v_iᵀ`` (equal weight per direction); SGD weights direction i by its singular value. A second basis
  comes from the previous step's applied update, whose singular vectors are those of MuonH's (Nesterov,
  pre-normed) momentum, since the polar factor keeps them.
- **Payoff per direction.** On a batch b, the step's first-order loss change is ``-η Σ_i s_i(b)`` with
  ``s_i(b) = u_iᵀ G_b v_i``.
- **Data reliability.** ``s_i`` on ``HELDOUT_BATCHES`` batches the basis never saw.
- **Position robustness.** ``s_i(θ_k)`` on a fixed set of ``EVAL_BATCHES`` batches at each of the next
  ``TRAJECTORY_STEPS`` real training steps, so only the parameters change.

The probe batches come from a separate iterator past the run's last step, so the run's own data and trajectory are
unchanged. Results go to ``muon_probe_step<N>.npz``; ``analyze_muon_probe.py`` reduces and plots them.
"""

import io
import logging
from collections.abc import Callable, Iterator

import fsspec
import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.stiefel import _msign

logger = logging.getLogger(__name__)

BASIS_BATCHES = 8
HELDOUT_BATCHES = 64
EVAL_BATCHES = 4
TRAJECTORY_STEPS = 32
PROBE_FILE = "muon_probe_step{step}.npz"


def _svd_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    return u.astype(np.float32), s.astype(np.float32), vt.T.astype(np.float32)


@jax.jit
def _project(grads: dict[str, jax.Array], bases: dict[str, tuple[jax.Array, jax.Array]]) -> dict[str, jax.Array]:
    """``s_i = u_iᵀ G v_i`` for every basis direction of every captured matrix."""
    return {name: jnp.einsum("mi,mn,ni->i", u, grads[name].astype(jnp.float32), v) for name, (u, v) in bases.items()}


@jax.jit
def _ns_spectra(grads: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Singular values of Muon's quintic Newton-Schulz output for each gradient."""
    return {name: jnp.linalg.svd(_msign(g.astype(jnp.float32)), compute_uv=False) for name, g in grads.items()}


class MuonProbe:
    """Runs the probe around ``step`` in the train loop. Each ``before_step`` call gets ``gradients(params, batch,
    step)``, returning the captured matrices' gradients and values under the current step's settings; ``batches``
    yields probe batches disjoint from the run's own."""

    def __init__(self, step: int, path: str, captured_params: Callable, batches: Iterator):
        if step < 1:
            raise ValueError("muon_probe_step must be at least 1 (the momentum basis needs the previous update)")
        self.step, self.path = step, path.rstrip("/")
        self._captured_params, self._batches = captured_params, batches
        self._gradients: Callable | None = None
        self._params_before: dict[str, np.ndarray] | None = None
        self._bases: dict[str, dict[str, tuple[jax.Array, jax.Array]]] = {}
        self._eval_batches: list = []
        self._out: dict[str, np.ndarray] = {}
        self._trajectory: dict[str, list[dict[str, np.ndarray]]] = {"grad": [], "momentum": []}

    def _mean_grads(self, params, batches, step) -> dict[str, jax.Array]:
        total = None
        for batch in batches:
            grads, _ = self._gradients(params, batch, step)
            total = grads if total is None else jax.tree.map(jnp.add, total, grads)
        return jax.tree.map(lambda g: g / len(batches), total)

    def _projections(self, params, batches, step) -> dict[str, dict[str, np.ndarray]]:
        grads = self._mean_grads(params, batches, step)
        return {kind: jax.device_get(_project(grads, bases)) for kind, bases in self._bases.items()}

    def before_step(self, current_step: int, params, step_array, gradients: Callable) -> None:
        self._gradients = gradients
        if current_step == self.step - 1:
            self._params_before = jax.device_get(self._captured_params(params))
        elif current_step == self.step:
            self._start(params, step_array)
        elif self.step < current_step <= self.step + TRAJECTORY_STEPS:
            for kind, s in self._projections(params, self._eval_batches, step_array).items():
                self._trajectory[kind].append(s)
            if current_step == self.step + TRAJECTORY_STEPS:
                self._finish()

    def _start(self, params, step_array) -> None:
        assert self._params_before is not None, "the probe must see the step before muon_probe_step"
        params_now = jax.device_get(self._captured_params(params))
        basis_batches = [next(self._batches) for _ in range(BASIS_BATCHES)]
        g_a = self._mean_grads(params, basis_batches, step_array)
        g_a_host = jax.device_get(g_a)
        grad_bases, momentum_bases = {}, {}
        for name, g in g_a_host.items():
            u, s, v = _svd_basis(g)
            place = g_a[name].sharding
            grad_bases[name] = (jax.device_put(u, place), jax.device_put(v, place))
            self._out[f"sigma_grad/{name}"] = s
            um, sm, vm = _svd_basis(params_now[name] - self._params_before[name])
            momentum_bases[name] = (jax.device_put(um, place), jax.device_put(vm, place))
            self._out[f"sigma_update/{name}"] = sm
        self._bases = {"grad": grad_bases, "momentum": momentum_bases}
        for name, s in jax.device_get(_ns_spectra(g_a)).items():
            self._out[f"sigma_ns/{name}"] = s
        heldout: dict[str, list[dict[str, np.ndarray]]] = {"grad": [], "momentum": []}
        for _ in range(HELDOUT_BATCHES):
            for kind, s in self._projections(params, [next(self._batches)], step_array).items():
                heldout[kind].append(s)
        for kind, rows in heldout.items():
            for name in rows[0]:
                self._out[f"heldout_{kind}/{name}"] = np.stack([r[name] for r in rows])
        self._eval_batches = [next(self._batches) for _ in range(EVAL_BATCHES)]
        for kind, s in self._projections(params, self._eval_batches, step_array).items():
            self._trajectory[kind].append(s)
        logger.info("muon probe: basis and %d held-out batches done at step %d", HELDOUT_BATCHES, self.step)

    def _finish(self) -> None:
        for kind, rows in self._trajectory.items():
            for name in rows[0]:
                self._out[f"trajectory_{kind}/{name}"] = np.stack([r[name] for r in rows])
        self._out["step"] = np.asarray(self.step)
        buffer = io.BytesIO()
        np.savez(buffer, **self._out)
        path = f"{self.path}/{PROBE_FILE.format(step=self.step)}"
        if jax.process_index() == 0:
            with fsspec.open(path, "wb") as f:
                f.write(buffer.getvalue())
        logger.info("muon probe: wrote %s", path)
