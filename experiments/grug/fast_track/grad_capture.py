# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture per-step gradients and applied updates of a fixed set of matrices, for offline optimizer diagnostics.

During each capture window the train loop recomputes the step's gradient on the same batch, reads the captured
matrices before and after the optimizer step, and writes ``grad_capture_step<N>.npz`` with ``grad/<site>`` (the raw
minibatch gradient the optimizer receives) and ``update/<site>`` (the change the step applied to the weight). The
first step of each window also stores ``param/<site>``. Sites cover every matrix family once or twice: the KDA and
MLA projections, shared experts, the LatentMoE projections, routers, and a few routed experts.
"""

import concurrent.futures
import io
import logging

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

logger = logging.getLogger(__name__)

GRAD_CAPTURE_FILE = "grad_capture_step{step}.npz"
# Layer layout of the KMA recipe: KDA at layers 0, 1, 2, 4 and MLA at 3, 5 (index into each stack).
_KDA_STACK_INDICES = {0: 0, 2: 2}
_MLA_STACK_INDICES = {3: 0, 5: 1}
_KDA_PROJECTIONS = ("w_q", "w_k", "w_v", "w_o", "w_g")
_MLA_PROJECTIONS = ("w_q", "w_dkv", "w_uk", "w_uv", "w_o")
# Routed experts are captured by index; which ones are busy is read from their gradient norms.
CAPTURE_EXPERTS = (0, 1, 2, 3)


def _take(value: jax.Array, index: int) -> jax.Array:
    """Layer ``index`` of a stacked leaf, replicated. Replicating before any further indexing matters for the
    expert banks: one expert cannot be sliced out of an expert-sharded axis."""
    layer = value[index]
    return reshard(layer, P(*(None,) * layer.ndim))


def _mlp_sites(layer: int, stack, index: int) -> dict:
    mlp = stack.mlp
    sites = {
        f"L{layer}.shared.w_up": _take(stack.shared[0].w_up, index),
        f"L{layer}.shared.w_down": _take(stack.shared[0].w_down, index),
        f"L{layer}.latent.w_down": _take(mlp.w_latent_down, index),
        f"L{layer}.latent.w_up": _take(mlp.w_latent_up, index),
        f"L{layer}.router": _take(mlp.router, index),
    }
    w_up = _take(mlp.expert_mlp.w_up, index)
    w_down = _take(mlp.expert_mlp.w_down, index)
    for expert in CAPTURE_EXPERTS:
        sites[f"L{layer}.expert{expert}.w_up"] = w_up[expert]
        sites[f"L{layer}.expert{expert}.w_down"] = w_down[expert]
    return sites


def capture_matrices(tree) -> dict[str, jax.Array]:
    """The captured matrices of a Transformer-shaped tree (parameters, gradients or updates), each replicated so
    every process holds a full copy. Call under ``jax.jit``."""
    kda = tree.kda_blocks.stacked
    mla = tree.stacked_blocks.stacked
    sites: dict[str, jax.Array] = {}
    for layer, index in _KDA_STACK_INDICES.items():
        sites.update({f"L{layer}.kda.{name}": _take(getattr(kda.attn, name), index) for name in _KDA_PROJECTIONS})
    for layer, index in _MLA_STACK_INDICES.items():
        sites.update({f"L{layer}.mla.{name}": _take(getattr(mla.attn, name), index) for name in _MLA_PROJECTIONS})
    sites.update(_mlp_sites(0, kda, _KDA_STACK_INDICES[0]))
    sites.update(_mlp_sites(5, mla, _MLA_STACK_INDICES[5]))
    return sites


def _add_at(value: jax.Array, delta: jax.Array, *index: int) -> jax.Array:
    """``value`` with ``delta`` added at the leading ``index`` (a one-hot mask, so the stacked leaf keeps its
    sharding: a scatter into an expert-sharded axis would need an explicit output sharding)."""
    mask = jnp.ones((), value.dtype)
    for axis, i in enumerate(index):
        shape = [1] * value.ndim
        shape[axis] = value.shape[axis]
        mask = mask * (jnp.arange(value.shape[axis]) == i).astype(value.dtype).reshape(shape)
    return value + mask * delta.astype(value.dtype)


def add_to_captured(tree, deltas: dict[str, jax.Array]):
    """The inverse of ``capture_matrices``: ``tree`` with ``deltas[site]`` added to each named site's matrix. Sites
    missing from ``deltas`` are unchanged. Call under ``jax.jit``."""
    unknown = set(deltas) - set(jax.eval_shape(capture_matrices, tree))
    if unknown:
        raise ValueError(f"unknown capture sites: {sorted(unknown)}")
    for layer, index in _KDA_STACK_INDICES.items():
        for name in _KDA_PROJECTIONS:
            site = f"L{layer}.kda.{name}"
            if site in deltas:
                tree = eqx.tree_at(
                    lambda t, n=name: getattr(t.kda_blocks.stacked.attn, n),
                    tree,
                    replace_fn=lambda v, d=deltas[site], i=index: _add_at(v, d, i),
                )
    for layer, index in _MLA_STACK_INDICES.items():
        for name in _MLA_PROJECTIONS:
            site = f"L{layer}.mla.{name}"
            if site in deltas:
                tree = eqx.tree_at(
                    lambda t, n=name: getattr(t.stacked_blocks.stacked.attn, n),
                    tree,
                    replace_fn=lambda v, d=deltas[site], i=index: _add_at(v, d, i),
                )
    for layer, stack_of, index in (
        (0, lambda t: t.kda_blocks.stacked, _KDA_STACK_INDICES[0]),
        (5, lambda t: t.stacked_blocks.stacked, _MLA_STACK_INDICES[5]),
    ):
        leaves = {
            f"L{layer}.shared.w_up": lambda t, s=stack_of: s(t).shared[0].w_up,
            f"L{layer}.shared.w_down": lambda t, s=stack_of: s(t).shared[0].w_down,
            f"L{layer}.latent.w_down": lambda t, s=stack_of: s(t).mlp.w_latent_down,
            f"L{layer}.latent.w_up": lambda t, s=stack_of: s(t).mlp.w_latent_up,
            f"L{layer}.router": lambda t, s=stack_of: s(t).mlp.router,
        }
        for site, where in leaves.items():
            if site in deltas:
                tree = eqx.tree_at(where, tree, replace_fn=lambda v, d=deltas[site], i=index: _add_at(v, d, i))
        for expert in CAPTURE_EXPERTS:
            for name in ("w_up", "w_down"):
                site = f"L{layer}.expert{expert}.{name}"
                if site in deltas:
                    tree = eqx.tree_at(
                        lambda t, s=stack_of, n=name: getattr(s(t).mlp.expert_mlp, n),
                        tree,
                        replace_fn=lambda v, d=deltas[site], i=index, e=expert: _add_at(v, d, i, e),
                    )
    return tree


def capture_steps(starts: tuple[int, ...], length: int) -> frozenset[int]:
    """The steps (``state.step`` before the train step) captured by windows of ``length`` steps at ``starts``."""
    if length <= 0:
        raise ValueError(f"grad_capture_len must be positive, got {length}")
    return frozenset(step for start in starts for step in range(start, start + length))


def write_capture(
    path: str,
    step: int,
    grads: dict[str, np.ndarray],
    updates: dict[str, np.ndarray],
    params: dict[str, np.ndarray] | None,
) -> None:
    """Write one step's capture to an fsspec ``path`` (uncompressed: the payload is dense floats)."""
    arrays = {"step": np.asarray(step)}
    arrays.update({f"grad/{k}": v for k, v in grads.items()})
    arrays.update({f"update/{k}": v for k, v in updates.items()})
    if params is not None:
        arrays.update({f"param/{k}": v for k, v in params.items()})
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with fsspec.open(path, "wb") as f:
        f.write(buffer.getvalue())


class CaptureWriter:
    """Writes captures from a background thread so the train loop only pays for the device-to-host copy."""

    def __init__(self, root: str):
        self._root = root.rstrip("/")
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="grad-capture")
        self._pending: list[concurrent.futures.Future] = []

    def submit(self, step, grads, updates, params) -> None:
        path = f"{self._root}/{GRAD_CAPTURE_FILE.format(step=step)}"
        self._pending.append(self._pool.submit(write_capture, path, step, grads, updates, params))

    def close(self) -> None:
        for future in self._pending:
            future.result()
        self._pool.shutdown()
        logger.info("grad capture: wrote %d steps to %s", len(self._pending), self._root)
