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

The forward is ``model.activations_with_stats`` with a token-validity grid built from the stored lengths: a real
row's tokens are valid, its padding and every filler row are not, so neither takes MoE dispatch capacity. The same
forward counts the valid assignments each layer dropped for capacity (sender and receiver), summed over all steps.

Entry ``i`` of each record is the distribution over token ``i`` given tokens ``0..i-1`` (vLLM's layout):
``{"tokens", "actual" [T] (nan at 0), "top_ids" [T, 50] (-1 at 0), "top_lp" [T, 50] (nan at 0)}``.

With ``reference_ids`` (``logit_check.reference_ids``: vLLM's stored top-20 ids per position), each record also holds
``reference_ids`` [T, 20] and ``reference_lp`` [T, 20]: the log-prob at exactly those ids, gathered from the same
log-softmax as the top 50 (-1 / nan where vLLM stored no id, and at 0). Every vLLM token is then measured, wherever it
falls in Levanter's ranking.

With ``routes=True`` each record also stores the selected experts per token and layer, for the cross-stack routing
row (``routing.compare_routes_cross_stack``). These are Levanter compute-view routes; they are never served-vLLM routes.

This is a log-probability comparison at matched positions, not raw-logit parity.
"""

import io
import logging
import math
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
from levanter.models.snowball import _BATCH_AXES as SNOWBALL_BATCH_AXES
from rigging.filesystem.buckets import filesystem_for

from experiments.interp.snowball_identity.multihost import gather_global
from experiments.interp.snowball_identity.sequences import Sequence

logger = logging.getLogger(__name__)

TOP_K = 50
PAD_ID = 128001
"""Snowball v0.1's native EOS/PAD token; right padding only, and padded positions are discarded."""


def _scores(
    model, tokens: jax.Array, valid: jax.Array, targets: jax.Array, routes: bool = False, reference=None
) -> tuple:
    """For every position of a [batch, S] token grid: the log-prob of ``targets`` (the next token) [B, S] and the
    top-k [B, S, k]; with ``routes``, also every layer's selected experts [L, B, S, K]; with ``reference`` ([B, S, R]
    token ids, -1 for none), the log-prob at those ids [B, S, R]; and every layer's sender and receiver drops of
    ``valid`` tokens' assignments [L]. Position ``S - 1`` has no next token and is dropped by the caller.

    The whole batch is scored at once: batch and position stay sharded as the model left them (on a multi-host mesh
    they span hosts), and only the vocab axis is reduced. No single row or shifted slice of a sharded axis is taken,
    and the full logits are never replicated; only the small reduced outputs are."""
    Batch, Pos = Axis("batch", tokens.shape[0]), model.Pos.resize(tokens.shape[1])
    hidden, stats = model.activations_with_stats(hax.named(tokens, (Batch, Pos)), hax.named(valid, (Batch, Pos)))
    selected = stats.routes if routes else None
    logits = jnp.einsum("bsd,dv->bsv", hidden.array, model.get_lm_head().array, preferred_element_type=jnp.float32)
    # The head shards vocab over the model axis; log-softmax and top-k need vocab whole, batch and position may stay.
    batch_spec, position_spec = jax.typeof(logits).sharding.spec[:2]
    logprobs = jax.nn.log_softmax(reshard(logits, P(batch_spec, position_spec, None)), axis=-1)
    targets = reshard(targets, P(batch_spec, position_spec))
    actual = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]
    top_lp, top_id = jax.lax.top_k(logprobs, TOP_K)
    measured = None
    if reference is not None:
        reference = reshard(reference, P(batch_spec, position_spec, None))
        measured = jnp.take_along_axis(logprobs, jnp.maximum(reference, 0), axis=-1)
        measured = jnp.where(reference >= 0, measured, jnp.nan)
    # Replicated, so every process can read the outputs of a multi-host mesh.
    outputs = (actual, top_id, top_lp, selected, measured, stats.sender_dropped, stats.receiver_dropped)
    return tuple(None if x is None else reshard(x, P()) for x in outputs)


def positions_records(
    model,
    sequences: list[Sequence],
    batch: int = 1,
    pad_multiple: int = 64,
    pad_id: int = PAD_ID,
    routes: bool = False,
    reference_ids: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, dict], dict]:
    """``model`` is the compute view (already cast). One record per sequence, in the #298 schema, in input order,
    and the dispatch counts: per layer, the valid assignments dropped by senders and by receivers, and their sum,
    over every step (all zero when the MoE is dropless), plus the number of valid tokens.

    With ``routes``, each record also holds ``routes`` [T, L, K] (int16): the experts the forward selected for
    token ``i`` at each layer, from the model's own route observer (not recomputed).

    ``reference_ids`` maps each sequence id to [T, R] token ids in the record layout (row ``i`` holds ids for token
    ``i``; -1 for none); each record then holds ``reference_ids`` and the gathered ``reference_lp``."""
    scores = jax.jit(_scores, static_argnames="routes")
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i].tokens))
    records: dict[int, dict] = {}
    sender = receiver = 0
    valid_tokens = 0
    started = time.time()
    for step in range(0, len(order), batch):
        indices = order[step : step + batch]
        rows = [sequences[i].tokens for i in indices]
        rows += [rows[0]] * (batch - len(rows))
        length = -(-max(len(r) for r in rows) // pad_multiple) * pad_multiple
        grid = np.full((batch, length), pad_id, dtype=np.int32)
        for row, row_tokens in enumerate(rows):
            grid[row, : len(row_tokens)] = row_tokens
        # Filler rows (copies past the real ones) and padding are not valid: they take no dispatch capacity.
        valid = np.zeros((batch, length), dtype=bool)
        for row, index in enumerate(indices):
            valid[row, : len(sequences[index].tokens)] = True
        valid_tokens += int(valid.sum())
        # The next token of each position, shifted on the host; the last position's target is padding (dropped).
        targets = np.full_like(grid, pad_id)
        targets[:, :-1] = grid[:, 1:]
        reference = None
        if reference_ids is not None:
            width = next(iter(reference_ids.values())).shape[1]
            reference = np.full((batch, length, width), -1, dtype=np.int32)
            for row, index in enumerate(indices):
                # Grid position p predicts token p + 1: record row p + 1.
                ids = reference_ids[sequences[index].id]
                reference[row, : len(ids) - 1] = ids[1:]
            reference = jnp.asarray(reference)
        outputs = scores(
            model, jnp.asarray(grid), jnp.asarray(valid), jnp.asarray(targets), routes=routes, reference=reference
        )
        actual, top_ids, top_lp, selected, measured, sent, received = (
            None if x is None else gather_global(x) for x in outputs
        )
        sender = sender + sent.astype(np.int64)
        receiver = receiver + received.astype(np.int64)
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
            if measured is not None:
                width = measured.shape[-1]
                records[index]["reference_ids"] = torch.from_numpy(
                    np.concatenate([np.full((1, width), -1), reference_ids[sequences[index].id][1:n]]).astype(np.int32)
                )
                records[index]["reference_lp"] = torch.from_numpy(
                    np.concatenate([np.full((1, width), np.nan), measured[row, : n - 1]]).astype(np.float32)
                )
            if selected is not None:
                records[index]["routes"] = torch.from_numpy(selected[:, row, :n].transpose(1, 0, 2).astype(np.int16))
        logger.info(
            "levanter positions %d/%d (length %d, %.0fs)", len(records), len(sequences), length, time.time() - started
        )
    dispatch = {
        "sender_dropped": [int(n) for n in sender],
        "receiver_dropped": [int(n) for n in receiver],
        "dropped": [int(n) for n in sender + receiver],
        "valid_tokens": valid_tokens,
    }
    return {sequences[i].id: records[i] for i in range(len(sequences))}, dispatch


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
    right operand has the router's shape). This is the source witness for the FP32 router dot exception.

    The trace is abstract (no arrays are allocated), under the current mesh, with an input the mesh can shard:
    batch = the product of Snowball's batch axes, sequence = 8 x the context axis."""
    mlp = next(iter(view.transformer.blocks)).mlp
    mesh = jax.sharding.get_abstract_mesh()
    sizes = {} if mesh.empty else dict(mesh.shape)
    shape = (math.prod(sizes.get(axis, 1) for axis in SNOWBALL_BATCH_AXES), 8 * sizes.get("context", 1))
    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype) if isinstance(a, jax.Array) else a, mlp)
    x = jax.ShapeDtypeStruct((*shape, mlp.router.shape[0]), compute_dtype)
    jaxpr = jax.make_jaxpr(lambda m, x: m(x))(abstract, x).jaxpr
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
    reference_ids: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], dict]:
    """Score the trainer's compute view of ``model`` and save ``positions.pt`` at ``path`` (s3:// or local); return
    the view's ``router_dot_dtypes`` and the dispatch counts of ``positions_records``.

    ``model`` is the trainer-loaded model (e.g. ``state.model`` after ``Trainer.initial_state``), ``mp`` the
    trainer's policy (``trainer.mp``), ``axis_mapping`` its compute mapping (``trainer.compute_axis_mapping``) and
    ``compute_dtype_overrides`` its ``trainer.config.compute_dtype_overrides``. The view is built by the same helper
    ``WrappedLossFunction`` uses (``levanter.compute_dtype.cast_model_for_compute``), so a reviewed exception such
    as ``{"router_bias": "float32"}`` is scored as trained. Call it inside the trainer's mesh, on every process (the
    forward is a collective); only process 0 writes the file."""
    with hax.axis_mapping(axis_mapping or {}):
        view = cast_model_for_compute(model, mp, compute_dtype_overrides)
        records, dispatch = positions_records(view, sequences, batch, pad_multiple, pad_id, routes, reference_ids)
        dot_dtypes = router_dot_dtypes(view, mp.compute_dtype)
    if jax.process_index() == 0:
        save_positions(records, path)
    return dot_dtypes, dispatch


def save_positions(records: dict[str, dict], path: str) -> None:
    buffer = io.BytesIO()
    torch.save(records, buffer)
    fs, local = filesystem_for(path)
    fs.pipe_file(local, buffer.getvalue())
