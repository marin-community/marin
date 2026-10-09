# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient noise scale per model component (McCandlish et al., "An Empirical Model of Large-Batch Training").

At each probe step the run's current parameters stay fixed while, for every batch size ``k`` in ``BATCH_SIZES``
(sequences), ``BATCHES_PER_SIZE`` independent size-``k`` gradients are taken on probe batches past the run's own
data. A size-``k`` batch is ``k`` sequences of a loaded training batch (the rest get zero loss weight, so one
program serves every ``k`` up to the training batch); larger ``k`` average several loaded batches. With
``g = G + e / sqrt(k)``, two independent gradients satisfy ``E[g_1 . g_2] = |G|^2`` and
``E|g|^2 = |G|^2 + tr(Sigma) / k``, so per component the probe records

- ``dot``: the mean pairwise dot product (from the running sum ``S`` and ``sum |g_i|^2``:
  ``(|S|^2 - sum |g_i|^2) / (n (n - 1))``), an unbiased ``|G|^2``;
- ``sq``: the mean ``|g|^2``;

and the same for each matrix's orthogonalized gradient (its polar factor, Muon's direction) on the MuonH-trained
leaves. ``B_noise = k (sq - dot) / dot`` and the expected cosine ``dot / sq`` follow offline.

Components: every parameter leaf, split per layer for the stacked blocks, per expert for the routed expert banks,
and into id buckets (``EMBED_ROW_BUCKETS``) for the token embedding and lm_head. Results go to
``noise_scale_step<N>.npz``.
"""

import io
import itertools
import logging
import re
from collections.abc import Callable, Iterator

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.fast_track.stiefel import MSIGN_STEPS, _msign

logger = logging.getLogger(__name__)

BATCH_SIZES = (8, 16, 32, 64, 128, 256)
BATCHES_PER_SIZE = 16
NOISE_FILE = "noise_scale_step{step}.npz"
# Row-id buckets of the vocabulary matrices (BPE ids run roughly from frequent to rare).
EMBED_ROW_BUCKETS = (0, 1024, 4096)
_VOCAB_LEAVES = re.compile(r"(^|\.)(token_embed|output_proj)$")
# Optimizer groups that orthogonalize the update (MuonH and its variants).
_ORTHO_GROUPS = re.compile(r"^(muon\w*|eig|okls|upper_qk|stiefel)$")


def _path_names(tree) -> list[str]:
    return [str(p) for p in jax.tree.leaves(leaf_key_paths(tree))]


def _group_sqnorm(name: str, x: jax.Array) -> jax.Array:
    """Per-component squared norm of one leaf: ``[L, E]`` for stacked expert banks, ``[L]`` for stacked leaves,
    ``[buckets]`` for the vocabulary matrices, else a scalar."""
    x = x.astype(jnp.float32)
    sq = jnp.square(x)
    if ".stacked." in f".{name}." and "expert_mlp" in name and x.ndim >= 4:
        return jnp.sum(sq, axis=tuple(range(2, x.ndim)))
    if ".stacked." in f".{name}." and x.ndim >= 2:
        return jnp.sum(sq, axis=tuple(range(1, x.ndim)))
    if _VOCAB_LEAVES.search(name):
        rows = jnp.sum(sq, axis=1) if name.endswith("token_embed") else jnp.sum(sq, axis=0)
        edges = (*EMBED_ROW_BUCKETS, rows.shape[0])
        return jnp.stack([jnp.sum(rows[lo:hi]) for lo, hi in itertools.pairwise(edges)])
    return jnp.sum(sq)


def _orthogonalize(x: jax.Array, steps: int) -> jax.Array:
    """Polar factor of each trailing ``[m, n]`` matrix (Newton-Schulz on the wide orientation). The matrix axes are
    gathered first (leading stack / expert axes keep their sharding) and the result is laid out like ``x``."""
    spec = jax.typeof(x).sharding.spec
    spec = tuple(spec) + (None,) * (x.ndim - len(spec))
    y = reshard(x.astype(jnp.float32), P(*spec[:-2], None, None))
    if y.shape[-2] > y.shape[-1]:
        y = jnp.swapaxes(_msign(jnp.swapaxes(y, -1, -2), steps), -1, -2)
    else:
        y = _msign(y, steps)
    return reshard(y, P(*spec))


class NoiseScaleProbe:
    """Runs at ``step`` in the train loop: ``run(params, step_array, gradients)`` with
    ``gradients(params, batch, row_weight, step)`` the training gradient of ``batch`` with each sequence's loss
    weight scaled by ``row_weight`` ``[B]``. ``batches`` yields probe batches disjoint from the run's own; ``labels``
    maps each parameter path to its optimizer group."""

    def __init__(self, step: int, path: str, batches: Iterator, labels: dict[str, str], ns_steps: int = MSIGN_STEPS):
        self.step, self.path = step, path.rstrip("/")
        self._batches = batches
        self._ortho = {name for name, group in labels.items() if _ORTHO_GROUPS.match(group)}
        self._ns_steps = ns_steps
        self._fns: dict[str, Callable] | None = None
        self._slot: list | None = None  # [loaded batch, k, next subset index] for k below the training batch

    def _build(self, grads_like) -> dict[str, Callable]:
        names = _path_names(grads_like)
        ortho = [
            name in self._ortho and leaf.ndim >= 2 for name, leaf in zip(names, jax.tree.leaves(grads_like), strict=True)
        ]
        steps = self._ns_steps

        def orth_tree(g):
            leaves, treedef = jax.tree.flatten(g)
            return jax.tree.unflatten(
                treedef,
                [
                    _orthogonalize(x, steps) if o else jnp.zeros((), jnp.float32)
                    for x, o in zip(leaves, ortho, strict=True)
                ],
            )

        def sqnorms(g, only_ortho: bool):
            leaves = jax.tree.leaves(g)
            return {
                n: _group_sqnorm(n, x) for n, x, o in zip(names, leaves, ortho, strict=True) if (o or not only_ortho)
            }

        @jax.jit
        def add(total, total_orth, g):
            o = orth_tree(g)
            new_total = jax.tree.map(lambda a, b: a + b.astype(jnp.float32), total, g)
            new_orth = jax.tree.map(lambda a, b: a + b, total_orth, o)
            return new_total, new_orth, sqnorms(g, False), sqnorms(o, True)

        @jax.jit
        def finish(total, total_orth):
            return sqnorms(total, False), sqnorms(total_orth, True)

        @jax.jit
        def zeros_like(g):
            # Only the orthogonalized tree's shapes are used, so XLA drops its Newton-Schulz.
            zero = lambda x: jnp.zeros_like(x, dtype=jnp.float32)  # noqa: E731
            return jax.tree.map(zero, g), jax.tree.map(zero, orth_tree(g))

        @jax.jit
        def scale(g, factor):
            return jax.tree.map(lambda x: x * factor, g)

        return {"add": add, "finish": finish, "zeros": zeros_like, "scale": scale}

    def run(self, params, step_array, gradients: Callable) -> None:
        logger.info("noise_scale probe at step %d", self.step)
        loaded = next(self._batches)
        seqs = loaded.tokens.shape[0]
        first = gradients(params, loaded, np.ones(seqs, np.float32), step_array)
        if self._fns is None:
            self._fns = self._build(first)
        fns = self._fns
        zeros = fns["zeros"](first)
        del first
        results: dict[str, list] = {}
        pending = [loaded]

        def next_batch():
            return pending.pop() if pending else next(self._batches)

        for k in BATCH_SIZES:
            total, total_orth = zeros
            sq_sum: dict[str, jax.Array] = {}
            sq_sum_orth: dict[str, jax.Array] = {}
            for _ in range(BATCHES_PER_SIZE):
                g = self._sample(k, seqs, params, step_array, gradients, next_batch)
                total, total_orth, sq, sq_o = fns["add"](total, total_orth, g)
                sq_sum = sq if not sq_sum else jax.tree.map(jnp.add, sq_sum, sq)
                sq_sum_orth = sq_o if not sq_sum_orth else jax.tree.map(jnp.add, sq_sum_orth, sq_o)
            total_sq, total_sq_orth = fns["finish"](total, total_orth)
            n = BATCHES_PER_SIZE
            for prefix, tot, sums in (("", total_sq, sq_sum), ("orth:", total_sq_orth, sq_sum_orth)):
                for name in sums:
                    s_tot, s_ind = np.asarray(tot[name], np.float64), np.asarray(sums[name], np.float64)
                    results.setdefault(f"{prefix}{name}|dot", []).append((s_tot - s_ind) / (n * (n - 1)))
                    results.setdefault(f"{prefix}{name}|sq", []).append(s_ind / n)
            logger.info("noise_scale probe step %d: batch size %d done", self.step, k)
        if jax.process_index() == 0:
            buf = io.BytesIO()
            np.savez(
                buf,
                batch_sizes=np.asarray(BATCH_SIZES),
                batches_per_size=np.asarray(BATCHES_PER_SIZE),
                seq_len=np.asarray(loaded.tokens.shape[1]),
                train_batch=np.asarray(seqs),
                **{key: np.stack(vals) for key, vals in results.items()},
            )
            with fsspec.open(f"{self.path}/{NOISE_FILE.format(step=self.step)}", "wb") as f:
                f.write(buf.getvalue())
            logger.info("noise_scale probe step %d written", self.step)

    def _sample(self, k: int, seqs: int, params, step_array, gradients: Callable, next_batch: Callable):
        """One independent size-``k`` gradient: ``k`` sequences of a loaded batch, or the mean over ``k / seqs``
        loaded batches."""
        assert self._fns is not None
        if k > seqs:
            if k % seqs:
                raise ValueError(f"batch size {k} must be a multiple of the training batch {seqs}")
            reps = k // seqs
            total = None
            for _ in range(reps):
                g = gradients(params, next_batch(), np.ones(seqs, np.float32), step_array)
                total = g if total is None else jax.tree.map(jnp.add, total, g)
            return self._fns["scale"](total, 1.0 / reps)
        if seqs % k:
            raise ValueError(f"training batch {seqs} must be a multiple of batch size {k}")
        if self._slot is None or self._slot[1] != k or self._slot[2] >= seqs // k:
            self._slot = [next_batch(), k, 0]
        batch, _, index = self._slot
        self._slot[2] += 1
        weight = (np.arange(seqs) // k == index).astype(np.float32)
        return gradients(params, batch, weight, step_array)
