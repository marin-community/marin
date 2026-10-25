# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""All-position log-probs of a Levanter Snowball model, in the #298 ``positions.pt`` schema, for task #310.

``logit_check.py`` then compares them with the stored vLLM reference, unchanged. The model is scored in the view the
trainer's loss sees: ``mp.cast_to_compute(model)``, as ``levanter.trainer.WrappedLossFunction`` does, so every
parameter (including the FP32 router biases) is cast to the compute dtype. Activations come from
``model.activations``; the LM head is applied with FP32 accumulation and FP32 output, as the fused next-token loss
does with ``loss_dtype=float32``; log-softmax is in FP32.

Inputs are the stored token ids of the pinned positions.pt (``logit_check.stored_sequences``), never a fresh
tokenization. ``batch`` real sequences (the mesh's batch axes, e.g. data 2 x expert 4 = 8) run per step, sorted by
length; a short last step is filled with copies of its first sequence. Each row is right-padded with ``pad_id`` to
a multiple of ``pad_multiple`` (a multiple of the context axis), and the padded positions are discarded; under the
causal mask they cannot change a real position. There is no packing: rows never share attention.

Entry ``i`` of each record is the distribution over token ``i`` given tokens ``0..i-1`` (vLLM's layout):
``{"tokens", "actual" [T] (nan at 0), "top_ids" [T, 50] (-1 at 0), "top_lp" [T, 50] (nan at 0)}``.

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
from rigging.filesystem.buckets import filesystem_for

from experiments.interp.snowball_identity.sequences import Sequence

logger = logging.getLogger(__name__)

TOP_K = 50


def _scores(model, tokens: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per row of a [batch, S] token grid: next-token log-probs of the following token [B, S-1] and the top-k
    [B, S-1, k], for positions 1..S-1."""
    Batch, Pos = Axis("batch", tokens.shape[0]), model.Pos.resize(tokens.shape[1])
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
    return jnp.stack(actuals), jnp.stack(top_ids), jnp.stack(top_lps)


def positions_records(
    model, sequences: list[Sequence], batch: int = 1, pad_multiple: int = 64, pad_id: int = 0
) -> dict[str, dict]:
    """``model`` is the compute view (already cast). One record per sequence, in the #298 schema, in input order."""
    scores = jax.jit(_scores)
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
        actual, top_ids, top_lp = (np.asarray(jax.device_get(x)) for x in scores(model, jnp.asarray(grid)))
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
        logger.info(
            "levanter positions %d/%d (length %d, %.0fs)", len(records), len(sequences), length, time.time() - started
        )
    return {sequences[i].id: records[i] for i in range(len(sequences))}


def write_positions(
    model,
    mp,
    sequences: list[Sequence],
    path: str,
    batch: int = 1,
    pad_multiple: int = 64,
    pad_id: int = 0,
    axis_mapping=None,
) -> None:
    """Score ``mp.cast_to_compute(model)`` and save ``positions.pt`` at ``path`` (s3:// or local).

    ``model`` is the trainer-loaded model (e.g. ``state.model`` after ``Trainer.initial_state``), ``mp`` the
    trainer's policy (``trainer.config.mp``) and ``axis_mapping`` its compute mapping, as ``WrappedLossFunction``
    uses. Call it inside the trainer's mesh."""
    with hax.axis_mapping(axis_mapping or {}):
        records = positions_records(mp.cast_to_compute(model), sequences, batch, pad_multiple, pad_id)
    buffer = io.BytesIO()
    torch.save(records, buffer)
    fs, local = filesystem_for(path)
    fs.pipe_file(local, buffer.getvalue())
