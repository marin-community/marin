# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Document-level retrieval index for the document study: which units each held-out document uses.

Over unseen training-cache sequences and Paloma, every document (a same-segment run of at least ``MIN_DOC`` scored
tokens) gets each unit's summed positive attribution: activation x gradient of the position's own next-token
log-prob, from one backward per batch in ``forward.trace_forward(local_grad=True)`` (each position's gradient along
its own residual stream; see there). Units are numbered ``unit_id``: routed ``(l, e, i)`` first, then the shared
units. Kept per document: its ``TOP_UNITS`` routed units by that mass and its ``TOP_SHARED`` shared units (which run
on every token and would otherwise fill every list), the totals, loss and location. Corpus-wide: each unit's
summed positive and absolute attribution. Writes ``docs_index.npz`` and the analysed tokens ``docs_tokens.npz``.
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

BATCH = 4
MIN_DOC = 64
MAX_DOCS = 128  # per batch; any further documents of a batch go to the overflow slot
TOP_UNITS = 512
TOP_SHARED = 64
L, E, I, K = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.TOP_K
NUM_ROUTED = L * E * I
NUM_UNITS = NUM_ROUTED + L * forward.SHARED_WIDTH
SOURCES = ("train", *common.PALOMA_SETS)


def unit_id(layer: int, expert: int, unit: int) -> int:
    return (layer * E + expert) * I + unit


def shared_unit_id(layer: int, unit: int) -> int:
    return NUM_ROUTED + layer * forward.SHARED_WIDTH + unit


def documents(batch: common.TokenBatch) -> list[tuple[int, int, int]]:
    """``(row, start, end)`` of each scored same-segment run of at least ``MIN_DOC`` tokens."""
    docs = []
    for b in range(batch.tokens.shape[0]):
        seg, valid = batch.segment_ids[b], batch.loss_weight[b] > 0
        cuts = np.flatnonzero(np.diff(seg) != 0) + 1
        for start, end in zip(np.r_[0, cuts], np.r_[cuts, seg.shape[0]], strict=True):
            n = int(valid[start:end].sum())
            if n >= MIN_DOC:
                docs.append((b, int(start), int(end)))
    return docs


def doc_slots(shape: tuple[int, int], docs: list[tuple[int, int, int]]) -> np.ndarray:
    """``[B, S]`` each position's document slot in the batch (``MAX_DOCS``: none or overflow)."""
    slots = np.full(shape, MAX_DOCS, np.int32)
    for j, (b, start, end) in enumerate(docs[:MAX_DOCS]):
        slots[b, start:end] = j
    return slots


def target_mask(seg: jax.Array, weight: jax.Array) -> jax.Array:
    """Positions whose next token is a scored token of the same document."""
    same = jnp.pad(seg[:, 1:] == seg[:, :-1], ((0, 0), (0, 1)))
    nxt = jnp.pad(weight[:, 1:] > 0, ((0, 0), (0, 1)))
    return (same & nxt & (weight > 0)).astype(jnp.float32)


@jax.jit
def index_step(model, tokens, seg, weight, slots):
    model = forward.replicate(model)
    b, s = tokens.shape
    t = b * s
    mask = target_mask(seg, weight)
    taps = (jnp.zeros((L, t * K, I), jnp.float32), jnp.zeros((L, t, forward.SHARED_WIDTH), jnp.float32))

    def objective(taps):
        edit = forward.UnitEdit(routed_tap=taps[0], shared_tap=taps[1])
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation(), edit=edit, local_grad=True)
        return -jnp.sum(tr["loss"] * mask), tr

    (g_routed, g_shared), tr = jax.grad(objective, has_aux=True)(taps)
    tr = forward.replicate(tr)
    flat_slot = slots.reshape(-1)
    nd = MAX_DOCS + 1
    routed, shared, routed_abs, shared_abs = [], [], [], []
    for li, lay in enumerate(tr["layers"]):
        attr = lay["expert_act"] * g_routed[li]  # [TK, I]
        seg_ids = flat_slot[lay["token_sorted"]] * E + lay["expert_sorted"]
        routed.append(jax.ops.segment_sum(jnp.maximum(attr, 0.0), seg_ids, num_segments=nd * E).reshape(nd, E * I))
        routed_abs.append(jax.ops.segment_sum(jnp.abs(attr), lay["expert_sorted"], num_segments=E).reshape(E * I))
        s_attr = lay["shared_act"] * g_shared[li]  # [T, I]
        shared.append(jax.ops.segment_sum(jnp.maximum(s_attr, 0.0), flat_slot, num_segments=nd))
        shared_abs.append(jnp.sum(jnp.abs(s_attr), axis=0))
    per_routed = jnp.concatenate(routed, axis=1)[:MAX_DOCS]  # [MAX_DOCS, NUM_ROUTED]
    per_shared = jnp.concatenate(shared, axis=1)[:MAX_DOCS]
    top_val, top_id = jax.lax.top_k(per_routed, TOP_UNITS)
    s_val, s_id = jax.lax.top_k(per_shared, TOP_SHARED)
    loss = tr["loss"].reshape(-1) * mask.reshape(-1)
    return dict(
        top_val=top_val,
        top_id=top_id,
        s_top_val=s_val,
        s_top_id=s_id + NUM_ROUTED,
        total=jnp.sum(per_routed, axis=1),
        s_total=jnp.sum(per_shared, axis=1),
        loss=jax.ops.segment_sum(loss, flat_slot, num_segments=nd)[:MAX_DOCS],
        scored=jax.ops.segment_sum(mask.reshape(-1), flat_slot, num_segments=nd)[:MAX_DOCS],
        unit_pos=jnp.concatenate([jnp.sum(per_routed, axis=0), jnp.sum(per_shared, axis=0)]),
        unit_abs=jnp.concatenate([*routed_abs, *shared_abs]),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="tn-d512-ckpt1-rno")
    parser.add_argument("--train-sequences", type=int, default=6144)
    parser.add_argument("--paloma-sequences", type=int, default=96, help="per Paloma set")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", default="docs_index")
    args = parser.parse_args()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    t0 = time.time()
    batches = [
        *common.paloma_batches(BATCH, max_sequences=args.paloma_sequences),
        *common.train_batches(BATCH, args.train_sequences, seed=args.seed),
    ]
    logger.info("%d batches loaded in %.0fs", len(batches), time.time() - t0)
    cols = {k: [] for k in ("top_val", "top_id", "s_top_val", "s_top_id", "total", "s_total", "loss", "scored")}
    meta = {k: [] for k in ("source", "seq", "start", "end")}
    seq_tokens, seq_segs, seq_valid, seq_sources = [], [], [], []
    unit_pos = np.zeros(NUM_UNITS, np.float64)
    unit_abs = np.zeros(NUM_UNITS, np.float64)
    with set_mesh(mesh):
        model = common.compute_model(params)
        for n, batch in enumerate(batches):
            docs = documents(batch)[:MAX_DOCS]
            slots = doc_slots(batch.tokens.shape, docs)
            tokens, seg, w = common.device_batch(batch, mesh)
            out = jax.device_get(index_step(model, tokens, seg, w, jnp.asarray(slots)))
            nd = len(docs)
            for k in cols:
                cols[k].append(np.asarray(out[k])[:nd])
            for b, start, end in docs:
                meta["source"].append(SOURCES.index(batch.source))
                meta["seq"].append(n * BATCH + b)
                meta["start"].append(start)
                meta["end"].append(end)
            unit_pos += out["unit_pos"]
            unit_abs += out["unit_abs"]
            seq_tokens.append(batch.tokens.astype(np.uint16))
            seq_segs.append(batch.segment_ids.astype(np.int32))
            seq_valid.append((batch.loss_weight > 0).astype(np.uint8))
            seq_sources += [SOURCES.index(batch.source)] * BATCH
            if n % 100 == 0:
                logger.info("batch %d/%d (%s) %d docs, %.0fs", n, len(batches), batch.source, nd, time.time() - t0)
    common.save_npz(
        args.tag,
        args.name,
        doc_units=np.concatenate(cols["top_id"]).astype(np.int32),
        doc_mass=np.concatenate(cols["top_val"]).astype(np.float16),
        doc_total=np.concatenate(cols["total"]).astype(np.float32),
        doc_shared_units=np.concatenate(cols["s_top_id"]).astype(np.int32),
        doc_shared_mass=np.concatenate(cols["s_top_val"]).astype(np.float16),
        doc_shared_total=np.concatenate(cols["s_total"]).astype(np.float32),
        doc_loss=np.concatenate(cols["loss"]).astype(np.float32),
        doc_scored=np.concatenate(cols["scored"]).astype(np.int32),
        **{f"doc_{k}": np.asarray(v, np.int32) for k, v in meta.items()},
        unit_pos=unit_pos.astype(np.float32),
        unit_abs=unit_abs.astype(np.float32),
        sources=np.asarray(SOURCES),
    )
    common.save_npz(
        args.tag,
        f"{args.name}_tokens",
        tokens=np.concatenate(seq_tokens),
        segment_ids=np.concatenate(seq_segs),
        valid=np.concatenate(seq_valid),
        sources=np.asarray(seq_sources, np.int8),
    )
    logger.info("done in %.0fs, %d documents", time.time() - t0, len(meta["seq"]))


if __name__ == "__main__":
    main()
