# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""All-position log-probs of a Levanter Snowball model, in the #298 ``positions.pt`` schema, for task #310.

``logit_check.py`` then compares them with the stored vLLM reference, unchanged. The model is scored in the view the
trainer's loss sees: ``levanter.compute_dtype.cast_model_for_compute(model, mp, compute_dtype_overrides)``, the helper
``levanter.trainer.WrappedLossFunction`` uses. With no overrides every parameter (including the FP32 router biases) is
cast to the compute dtype; a reviewed override (e.g. ``router_bias: float32``) keeps that field in its dtype.
Activations come from ``model.activations``; the LM head is an einsum with ``preferred_element_type=float32`` (FP32
output), and log-softmax is in FP32. That is meant to match the fused next-token loss with ``loss_dtype=float32``; the
fused kernel's own head arithmetic is not checked here.

Inputs are the stored token ids of the pinned positions.pt (``logit_check.stored_sequences``), never a fresh
tokenization. ``batch`` real sequences (the mesh's batch axes, e.g. data 2 x expert 4 = 8) run per step, sorted by
length; a short last step is filled with copies of its first sequence. Each row is right-padded with ``pad_id`` to
a multiple of ``pad_multiple`` (a multiple of the context axis), and the padded positions are discarded; under the
causal mask they cannot change a real position. There is no packing: rows never share attention.

Entry ``i`` of each record is the distribution over token ``i`` given tokens ``0..i-1`` (vLLM's layout):
``{"tokens", "actual" [T] (nan at 0), "top_ids" [T, 50] (-1 at 0), "top_lp" [T, 50] (nan at 0)}``.

With ``routes=True`` each record also stores the selected experts per token and layer, for the cross-stack routing
row (``routing.compare_routes_cross_stack``). These are Levanter compute-view routes; they are never served-vLLM routes.

This is a log-probability comparison at matched positions, not raw-logit parity.
"""

import io
import logging
import time

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from haliax import Axis
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.compute_dtype import cast_model_for_compute
from rigging.filesystem.buckets import filesystem_for

from experiments.interp.snowball_identity.sequences import Sequence

logger = logging.getLogger(__name__)

TOP_K = 50
PAD_ID = 128001
"""Snowball v0.1's native EOS/PAD token; right padding only, and padded positions are discarded."""


def _scores(model, tokens: jax.Array, routes: bool = False) -> tuple:
    """Per row of a [batch, S] token grid: next-token log-probs of the following token [B, S-1] and the top-k
    [B, S-1, k], for positions 1..S-1; with ``routes``, also every layer's selected experts [L, B, S, K]."""
    Batch, Pos = Axis("batch", tokens.shape[0]), model.Pos.resize(tokens.shape[1])
    selected = None
    if routes:
        hidden, selected = model.activations_with_routes(hax.named(tokens, (Batch, Pos)))
    else:
        hidden = model.activations(hax.named(tokens, (Batch, Pos)))
    logits = jnp.einsum("bsd,dv->bsv", hidden.array, model.get_lm_head().array, preferred_element_type=jnp.float32)
    actuals, top_ids, top_lps = [], [], []
    for row in range(tokens.shape[0]):
        # The head shards vocab over the model axis; gather and top-k need one row whole.
        logprobs = jax.nn.log_softmax(reshard(logits[row], P(None, None))[:-1], axis=-1)
        actuals.append(jnp.take_along_axis(logprobs, tokens[row, 1:, None], axis=-1)[:, 0])
        top_lp, top_id = jax.lax.top_k(logprobs, TOP_K)
        top_lps.append(top_lp)
        top_ids.append(top_id)
    # Replicated, so every process can read the outputs of a multi-host mesh.
    outputs = (jnp.stack(actuals), jnp.stack(top_ids), jnp.stack(top_lps))
    if selected is not None:
        outputs += (selected,)
    return tuple(reshard(x, P()) for x in outputs) + ((None,) if selected is None else ())


def positions_records(
    model, sequences: list[Sequence], batch: int = 1, pad_multiple: int = 64, pad_id: int = PAD_ID, routes: bool = False
) -> dict[str, dict]:
    """``model`` is the compute view (already cast). One record per sequence, in the #298 schema, in input order.

    With ``routes``, each record also holds ``routes`` [T, L, K] (int16): the experts the forward selected for
    token ``i`` at each layer, from the model's own route observer (not recomputed)."""
    scores = jax.jit(_scores, static_argnums=2)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i].tokens))
    records: dict[int, dict] = {}
    started = time.time()
    for step in range(0, len(order), batch):
        indices = order[step : step + batch]
        rows = [sequences[i].tokens for i in indices]
        rows += [rows[0]] * (batch - len(rows))
        length = -(-max(len(r) for r in rows) // pad_multiple) * pad_multiple
        grid = np.full((batch, length), pad_id, dtype=np.int32)
        for row, row_tokens in enumerate(rows):
            grid[row, : len(row_tokens)] = row_tokens
        actual, top_ids, top_lp, selected = (
            None if x is None else np.asarray(jax.device_get(x)) for x in scores(model, jnp.asarray(grid), routes)
        )
        for row, index in enumerate(indices):
            n = len(sequences[index].tokens)
            records[index] = {
                "tokens": list(sequences[index].tokens),
                "actual": torch.from_numpy(np.concatenate([[np.nan], actual[row, : n - 1]]).astype(np.float32)),
                "top_ids": torch.from_numpy(
                    np.concatenate([np.full((1, TOP_K), -1), top_ids[row, : n - 1]]).astype(np.int32)
                ),
                "top_lp": torch.from_numpy(
                    np.concatenate([np.full((1, TOP_K), np.nan), top_lp[row, : n - 1]]).astype(np.float32)
                ),
            }
            if selected is not None:
                records[index]["routes"] = torch.from_numpy(selected[:, row, :n].transpose(1, 0, 2).astype(np.int16))
        logger.info(
            "levanter positions %d/%d (length %d, %.0fs)", len(records), len(sequences), length, time.time() - started
        )
    return {sequences[i].id: records[i] for i in range(len(sequences))}


def _dot_generals(jaxpr, found: list) -> list:
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            found.append(eqn)
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    _dot_generals(inner, found)
    return found


def router_dot_dtypes(view, compute_dtype) -> list[dict]:
    """The dtypes of the router dot in the first block of ``view``, read from its traced forward (the dot whose
    right operand has the router's shape). This is the source witness for the FP32 router dot exception."""
    mlp = next(iter(view.transformer.blocks)).mlp
    x = jnp.zeros((1, 8, mlp.router.shape[0]), compute_dtype)
    jaxpr = jax.make_jaxpr(lambda m, x: m(x))(mlp, x).jaxpr
    return [
        {"lhs": str(e.invars[0].aval.dtype), "rhs": str(e.invars[1].aval.dtype), "out": str(e.outvars[0].aval.dtype)}
        for e in _dot_generals(jaxpr, [])
        if tuple(e.invars[1].aval.shape) == tuple(mlp.router.shape)
    ]


def router_dot_is_fp32(dtypes: list[dict]) -> bool:
    return bool(dtypes) and all(set(d.values()) == {"float32"} for d in dtypes)


def write_positions(
    model,
    mp,
    sequences: list[Sequence],
    path: str,
    batch: int = 1,
    pad_multiple: int = 64,
    pad_id: int = PAD_ID,
    axis_mapping=None,
    routes: bool = False,
    compute_dtype_overrides=None,
) -> list[dict]:
    """Score the trainer's compute view of ``model`` and save ``positions.pt`` at ``path`` (s3:// or local); return
    the view's ``router_dot_dtypes``.

    ``model`` is the trainer-loaded model (e.g. ``state.model`` after ``Trainer.initial_state``), ``mp`` the
    trainer's policy (``trainer.mp``), ``axis_mapping`` its compute mapping (``trainer.compute_axis_mapping``) and
    ``compute_dtype_overrides`` its ``trainer.config.compute_dtype_overrides``. The view is built by the same helper
    ``WrappedLossFunction`` uses (``levanter.compute_dtype.cast_model_for_compute``), so a reviewed exception such
    as ``{"router_bias": "float32"}`` is scored as trained. Call it inside the trainer's mesh, on every process (the
    forward is a collective); only process 0 writes the file."""
    with hax.axis_mapping(axis_mapping or {}):
        view = cast_model_for_compute(model, mp, compute_dtype_overrides)
        records = positions_records(view, sequences, batch, pad_multiple, pad_id, routes)
        dot_dtypes = router_dot_dtypes(view, mp.compute_dtype)
    if jax.process_index() != 0:
        return dot_dtypes
    buffer = io.BytesIO()
    torch.save(records, buffer)
    fs, local = filesystem_for(path)
    fs.pipe_file(local, buffer.getvalue())
    return dot_dtypes
