# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capacity drops of the training dispatch on realistic batches.

The run trained with ``moe_implementation=ragged_all_to_all`` on 8 H100s (expert axis 8, 48 local experts per GPU,
``moe_ragged_chunks=2``) at ``capacity_factor=1.05``, 16 sequences of 4096 tokens per GPU. Each receiving GPU
accepts ``ceil(1.05 * all assignments / 8)`` rows (split over its 2 expert chunks), earlier local experts first and
earlier senders first within an expert; the rest are dropped (the slot just loses its weight). This replays that
with the real levanter helpers on whole 128-sequence training batches: layer by layer (a drop changes the inputs of
later layers, so each layer's drops are computed from a forward that already applies the earlier layers' drops),
then scores the batch with all drops applied against the dropless forward.
"""

import argparse
import logging
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug._moe.common import _scaled_capacity
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes
from levanter.grug._moe.ep_ragged_all_to_all import _accepted_assignments

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

SHARDS = 8
SEQS_PER_SHARD = 16
CAPACITY_FACTOR = 1.05
EXPERT_CHUNKS = 2
L, E, K = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.TOP_K


@jax.jit
def accepted_assignments(selected: jax.Array) -> jax.Array:
    """``[SHARDS, T, K]`` bool: which of each shard's assignments the ragged all-to-all dispatch accepts."""
    shards, t, k = selected.shape
    local = E // SHARDS
    chunk_experts = local // EXPERT_CHUNKS
    physical = max(local, math.ceil(CAPACITY_FACTOR * t * k))
    flat = selected.reshape(shards, t * k)
    group_sizes = jax.vmap(lambda f: jnp.bincount(f, length=E).astype(jnp.int32))(flat)  # [S, E]
    logical = _scaled_capacity(
        jnp.sum(group_sizes, dtype=jnp.int32),
        capacity_factor=CAPACITY_FACTOR,
        divisor=SHARDS,
        minimum=local,
        maximum=physical,
    )
    chunk_capacity = jnp.maximum((logical + EXPERT_CHUNKS - 1) // EXPERT_CHUNKS, chunk_experts)
    chunk_of_expert = (jnp.arange(E, dtype=jnp.int32) % local) // chunk_experts
    accepted_sizes = sum(
        _clip_receiver_group_sizes(
            jnp.where(chunk_of_expert[None, :] == c, group_sizes, 0),
            local_expert_size=local,
            receiver_capacity=chunk_capacity,
        )
        for c in range(EXPERT_CHUNKS)
    )  # [S, E]
    keep = jax.vmap(lambda f, gs, acc: _accepted_assignments(f, jnp.argsort(f), gs, acc))(
        flat, group_sizes, accepted_sizes
    )
    return keep.reshape(shards, t, k)


def make_forward():
    @jax.jit
    def run(model, tokens, seg, combine_mult):
        base = forward.no_override(tokens.shape[0] * tokens.shape[1])
        tr = forward.trace_forward(
            forward.replicate(model), tokens, seg, forward.no_ablation(), base._replace(combine_mult=combine_mult)
        )
        return (
            tr["loss"],
            jnp.stack([lay["selected"] for lay in tr["layers"]]),
            jnp.stack([lay["combine"] for lay in tr["layers"]]),
            jnp.stack([lay["weight_rank"] for lay in tr["layers"]]),
        )

    return run


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    t0 = time.time()
    run = make_forward()
    per_batch = SHARDS * SEQS_PER_SHARD
    seqs = common.train_batches(SEQS_PER_SHARD, args.batches * per_batch, seed=args.seed)
    t = SEQS_PER_SHARD * common.SEQ_LEN
    drops = {k: [] for k in ("batch", "layer", "shard", "token", "expert", "rank", "weight", "position", "prev")}
    loss_rows, assign_rows, load_rows = [], [], []
    with set_mesh(mesh):
        model = common.compute_model(params)
        for b in range(args.batches):
            shards = seqs[b * SHARDS : (b + 1) * SHARDS]
            data = [common.device_batch(s, mesh) for s in shards]
            mult = np.ones((SHARDS, L, t, K), np.float32)
            dropless = None
            for layer in range(L):
                outs = [run(model, tok, seg, jnp.asarray(mult[s])) for s, (tok, seg, _) in enumerate(data)]
                if dropless is None:
                    dropless = [np.asarray(o[0]) for o in outs]
                selected = jnp.stack([o[1][layer] for o in outs])  # [S, T, K]
                keep = np.asarray(accepted_assignments(selected))
                mult[:, layer] = keep
                sel = np.asarray(selected)
                comb = np.stack([np.asarray(o[2][layer]) for o in outs])
                rank = np.stack([np.asarray(o[3][layer]) for o in outs])
                s_idx, t_idx, k_idx = np.nonzero(~keep)
                flat_tok = np.stack([sh.tokens.reshape(-1) for sh in shards])
                drops["batch"].append(np.full(len(s_idx), b))
                drops["layer"].append(np.full(len(s_idx), layer))
                drops["shard"].append(s_idx)
                drops["token"].append(flat_tok[s_idx, t_idx])
                drops["expert"].append(sel[s_idx, t_idx, k_idx])
                drops["rank"].append(rank[s_idx, t_idx, k_idx])
                drops["weight"].append(comb[s_idx, t_idx, k_idx])
                drops["position"].append(t_idx)
                drops["prev"].append(np.where(t_idx % common.SEQ_LEN > 0, flat_tok[s_idx, np.maximum(t_idx - 1, 0)], -1))
                counts = np.stack([np.bincount(sel[s].reshape(-1), minlength=E) for s in range(SHARDS)])  # [S, E]
                recv = counts.reshape(SHARDS, SHARDS, E // SHARDS).sum(axis=(0, 2))  # per receiving shard
                load_rows.append(recv / recv.mean())
                assign_rows.append((b, layer, keep.size, int((~keep).sum()), int(np.any(~keep, axis=-1).sum())))
                logger.info(
                    "batch %d layer %d: %d of %d assignments dropped (%.0fs)",
                    b,
                    layer,
                    (~keep).sum(),
                    keep.size,
                    time.time() - t0,
                )
            dropped = [np.asarray(run(model, tok, seg, jnp.asarray(mult[s]))[0]) for s, (tok, seg, _) in enumerate(data)]
            weight = np.stack([sh.loss_weight for sh in shards])
            loss_rows.append(
                (
                    float((np.stack(dropless) * weight).sum() / weight.sum()),
                    float((np.stack(dropped) * weight).sum() / weight.sum()),
                )
            )
            logger.info("batch %d loss dropless %.5f with drops %.5f", b, *loss_rows[-1])
    out = {f"drop/{k}": np.concatenate(v) for k, v in drops.items()}
    out["assignments"] = np.asarray(assign_rows, np.int64)
    out["receiver_load"] = np.stack(load_rows).astype(np.float32)
    out["loss"] = np.asarray(loss_rows)
    common.save_npz(args.tag, "capacity", **out)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
