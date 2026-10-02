# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight attribution probe: which parameter updates move one in-context prediction, step by step.

For a probe ``spot`` (a (row, position) of the fact-probe rows) and a step window, every step records, per
parameter tensor and per layer of a stacked tensor:

- ``Gd``: ``G . (theta_t - theta_{t-1})``, ``G`` the gradient of log p(target) at ``theta_t``. Summed over all
  tensors it is the first-order change in log p(target) over the step; per tensor it attributes that change.
- ``Gdp``, ``ddp``, ``dd``: ``G`` against the previous update, and the current update against the previous one
  (``ddp < 0`` with ``|d|`` steady is a tensor stepping back and forth).
- For MuonH tensors with Bi-Maxwell rails, ``G`` against the fast rail, the previous fast rail and the slow rail.
  The fresh gradient fed to the rails is ``g_t = fast_{t-1} + (fast_t - fast_{t-1}) / fast_rate``, and the
  pre-Newton-Schulz direction is ``g + m (M - g)`` with ``M = (1 - w) fast + w slow``.

Tensors matching ``exclude`` (the big row-sparse ``token_embed2`` by default) are left out to bound memory: the
probe keeps a previous-step copy of the parameters, of the update and of the fast rails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fsspec
import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import MuonMomentumState

ATTRIBUTION_FILE = "weight_attribution_{index:04d}.npz"
DOT_NAMES = ("Gd", "Gdp", "ddp", "dd", "GG", "Gf", "Gfp", "Gs")
_STACKED = re.compile(r"(^|\.)(stacked_blocks|stacked_blocks_tail|kda_blocks)\.stacked\.")


def leaf_name(path) -> str:
    return jax.tree_util.keystr(path, simple=True, separator=".")


def per_layer_sum(name: str, x: jax.Array) -> jax.Array:
    """Sum over every axis but the leading layer axis of a stacked tensor; a [1] total otherwise."""
    x = x.astype(jnp.float32)
    if _STACKED.search(name) and x.ndim >= 2:
        return jnp.sum(x, axis=tuple(range(1, x.ndim)))
    return jnp.sum(x)[None]


def rails_by_name(opt_state) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """The Bi-Maxwell fast and slow rails of every MuonH tensor, keyed by the tensor's parameter path."""
    fast: dict[str, jax.Array] = {}
    slow: dict[str, jax.Array] = {}
    nodes = jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, MuonMomentumState))
    for node in nodes:
        if not isinstance(node, MuonMomentumState) or node.fast is None:
            continue
        for path, leaf in jax.tree_util.tree_leaves_with_path(node.fast):
            if isinstance(leaf, jax.Array):
                fast[leaf_name(path)] = leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(node.slow):
            if isinstance(leaf, jax.Array):
                slow[leaf_name(path)] = leaf
    return fast, slow


@dataclass
class AttributionWriter:
    """Buffers per-step dot products and writes chunks of ``chunk_size`` steps (process 0)."""

    directory: str
    chunk_size: int = 50

    def __post_init__(self):
        self.rows: list[tuple[int, dict[str, np.ndarray]]] = []
        self.index = 0

    def add(self, step: int, logp: float, dots: dict[str, dict[str, np.ndarray]]) -> None:
        self.rows.append((step, {"__logp__": logp, **dots}))
        if len(self.rows) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        steps = np.asarray([s for s, _ in self.rows], np.int64)
        logp = np.asarray([d["__logp__"] for _, d in self.rows], np.float64)
        names = sorted(n for n in self.rows[0][1] if n != "__logp__")
        arrays = {
            f"{kind}/{name}": np.stack([d[name][kind] for _, d in self.rows]) for name in names for kind in DOT_NAMES
        }
        path = f"{self.directory.rstrip('/')}/{ATTRIBUTION_FILE.format(index=self.index)}"
        with fsspec.open(path, "wb") as f:
            np.savez(f, steps=steps, logp=logp, **arrays)
        self.index += 1
        self.rows = []
