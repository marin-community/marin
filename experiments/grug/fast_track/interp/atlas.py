# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron and expert atlas pass: domain-tagged unit statistics over ~25M tokens (unseen training-cache sequences
plus every Paloma subset).

Each routed unit's "high" activations are those above ``HIGH_FRACTION`` of its mean top-16 activation in the earlier
``stats`` pass. Accumulated on the device, per layer:

- routed units ``[E, I]``: firing count, activation sum / square sum / max, high count and high mass; high counts
  per source domain ``[E, I, D]`` and per document-position bucket ``[E, I, P]``; high mass per expert-top token
  ``[E, TOP_TOKENS + 1, I]`` (the expert's ``TOP_TOKENS`` most-routed tokens in ``stats``, then "other"); the
  top-``TOP_CONTEXTS`` activations as the top-1 of each ``BATCH``-sequence window, with their global positions;
- shared units ``[I]``: the same except the per-token mass;
- experts: routed tokens per (expert, token id) and per (expert, domain), combine-weight sums, bias-driven slots.

Weights-only: each unit's logit lens (``stats.neuron_logit_lens``) and the token embedding table. Writes
``atlas.npz`` and the analysed tokens ``atlas_tokens.npz``.
"""

import argparse
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import common, forward, stats
from experiments.grug.fast_track.model import _small_top_k

logger = logging.getLogger(__name__)

BATCH = 4
TOP_CONTEXTS = 32
TOP_TOKENS = 64
HIGH_FRACTION = 0.3
SOURCES = ("train", *common.PALOMA_SETS)
# Coarse domain groups of the sources (the training mix is its own group).
GROUPS = ("train", "web", "code", "social", "science", "wiki", "news")
SOURCE_GROUP = {
    "train": "train",
    "4chan": "social",
    "c4_100_domains": "web",
    "c4_en": "web",
    "dolma-v1_5": "web",
    "dolma_100_programing_languages": "code",
    "dolma_100_subreddits": "social",
    "falcon-refinedweb": "web",
    "gab": "social",
    "m2d2_s2orc_unsplit": "science",
    "m2d2_wikipedia_unsplit": "wiki",
    "manosphere_meta_sep": "social",
    "mc4": "web",
    "ptb": "news",
    "redpajama": "web",
    "twitterAAE_HELM_fixed": "social",
    "wikitext_103": "wiki",
}
POSITION_EDGES = (1, 4, 16, 64, 512)  # buckets [0], [1, 4), [4, 16), [16, 64), [64, 512), [512, inf)
NUM_POS = len(POSITION_EDGES) + 1
STATS_TAG = "tn-d512-ckpt1-rno"


def load_stats(tag: str) -> dict[str, np.ndarray]:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/stats.npz", "rb") as f:
        z = np.load(f)
        return {k: z[k] for k in ("top_val", "s_top_val", "expert_token_ids")}


def tables(prior: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """High thresholds per unit and each expert's top-token slot table ``[L, E, V]`` (``TOP_TOKENS`` = other)."""
    L, E, V = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.VOCAB
    top = np.where(prior["top_val"] > 0, prior["top_val"], np.nan)
    thr = HIGH_FRACTION * np.nan_to_num(np.nanmean(top, axis=-1), nan=1e9)
    s_top = np.where(prior["s_top_val"] > 0, prior["s_top_val"], np.nan)
    s_thr = HIGH_FRACTION * np.nan_to_num(np.nanmean(s_top, axis=-1), nan=1e9)
    counts = prior["expert_token_ids"]  # [L, E, V]
    top_ids = np.argsort(-counts, axis=-1)[..., :TOP_TOKENS]  # [L, E, TOP_TOKENS]
    slot = np.full((L, E, V), TOP_TOKENS, np.int8)
    li, ei = np.meshgrid(np.arange(L), np.arange(E), indexing="ij")
    for j in range(TOP_TOKENS):
        have = np.take_along_axis(counts, top_ids[..., j : j + 1], axis=-1)[..., 0] > 0
        slot[li[have], ei[have], top_ids[..., j][have]] = j
    return dict(thr=thr.astype(np.float32), s_thr=s_thr.astype(np.float32), slot=slot, top_ids=top_ids)


def init_state() -> dict:
    L, E, I, V, D = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.VOCAB, len(SOURCES)
    z = jnp.zeros
    return dict(
        fire=z((L, E, I)),
        act_sum=z((L, E, I)),
        act_sq=z((L, E, I)),
        act_max=z((L, E, I)),
        high=z((L, E, I)),
        high_mass=z((L, E, I)),
        high_dom=z((L, E, I, D)),
        high_pos=z((L, E, NUM_POS, I)),
        tok_mass=z((L, E, TOP_TOKENS + 1, I)),
        top_val=jnp.full((L, E, I, TOP_CONTEXTS), -1.0),
        top_pos=jnp.zeros((L, E, I, TOP_CONTEXTS), jnp.int32),
        s_fire=z((L, I)),
        s_act_sum=z((L, I)),
        s_act_max=z((L, I)),
        s_high=z((L, I)),
        s_high_dom=z((L, I, D)),
        s_high_pos=z((L, NUM_POS, I)),
        s_top_val=jnp.full((L, I, TOP_CONTEXTS), -1.0),
        s_top_pos=jnp.zeros((L, I, TOP_CONTEXTS), jnp.int32),
        expert_token_ids=jnp.zeros((L, E, V), jnp.int32),
        expert_dom=z((L, E, D)),
        expert_weight=z((L, E)),
        expert_bias_driven=z((L, E)),
        dom_tokens=z((D,)),
        dom_loss=z((D,)),
        pos_tokens=z((NUM_POS,)),
    )


def _merge(vals, ids, new_val, new_id):
    v = jnp.concatenate([vals, new_val[..., None]], axis=-1)
    i = jnp.concatenate([ids, new_id[..., None]], axis=-1)
    top, pick = jax.lax.top_k(v, TOP_CONTEXTS)
    return top, jnp.take_along_axis(i, pick, axis=-1)


def make_step():
    E, I, K = forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.TOP_K

    @jax.jit
    def step(state, model, tokens, seg, weight, domain, offset, tab):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        flat_tok = tokens.reshape(-1)
        valid = (weight.reshape(-1) > 0).astype(jnp.float32)
        pos = forward.position_in_document(seg).reshape(-1)
        pbucket = jnp.searchsorted(jnp.asarray(POSITION_EDGES), pos, side="right").astype(jnp.int32)
        new = dict(state)
        acc = {k: [] for k in state}
        for i, lay in enumerate(tr["layers"]):
            act = lay["expert_act"]  # [TK, I]
            ex = lay["expert_sorted"]
            ts = lay["token_sorted"]
            okf = valid[ts][:, None]
            high = (act > tab["thr"][i][ex]) * okf
            acc["fire"].append(jax.ops.segment_sum((act > 0) * okf, ex, num_segments=E))
            acc["act_sum"].append(jax.ops.segment_sum(act * okf, ex, num_segments=E))
            acc["act_sq"].append(jax.ops.segment_sum(jnp.square(act) * okf, ex, num_segments=E))
            win = jax.ops.segment_max(jnp.where(okf > 0, act, -1.0), ex, num_segments=E)  # [E, I]
            acc["act_max"].append(jnp.maximum(win, 0.0))
            acc["high"].append(jax.ops.segment_sum(high, ex, num_segments=E))
            acc["high_mass"].append(jax.ops.segment_sum(high * act, ex, num_segments=E))
            acc["high_pos"].append(
                jax.ops.segment_sum(high, ex * NUM_POS + pbucket[ts], num_segments=E * NUM_POS).reshape(E, NUM_POS, I)
            )
            slot = tab["slot"][i][ex, flat_tok[ts]].astype(jnp.int32)
            acc["tok_mass"].append(
                jax.ops.segment_sum(high * act, ex * (TOP_TOKENS + 1) + slot, num_segments=E * (TOP_TOKENS + 1)).reshape(
                    E, TOP_TOKENS + 1, I
                )
            )
            is_max = (act == win[ex]) & (okf > 0)
            arg = jax.ops.segment_max(jnp.where(is_max, ts[:, None], -1), ex, num_segments=E)  # [E, I]
            acc["top"] = [*acc.get("top", []), (win, jnp.maximum(arg, 0) + offset)]

            s_act = lay["shared_act"]  # [T, I]
            vt = valid[:, None]
            s_high = (s_act > tab["s_thr"][i][None, :]) * vt
            acc["s_fire"].append(jnp.sum((s_act > 0) * vt, axis=0))
            acc["s_act_sum"].append(jnp.sum(s_act * vt, axis=0))
            s_win = jnp.max(jnp.where(vt > 0, s_act, -1.0), axis=0)
            acc["s_act_max"].append(jnp.maximum(s_win, 0.0))
            acc["s_high"].append(jnp.sum(s_high, axis=0))
            acc["s_high_pos"].append(jax.ops.segment_sum(s_high, pbucket, num_segments=NUM_POS))
            acc["s_top"] = [
                *acc.get("s_top", []),
                (s_win, jnp.argmax(jnp.where(vt > 0, s_act, -1.0), axis=0).astype(jnp.int32) + offset),
            ]

            sel = lay["selected"]
            vflat = jnp.broadcast_to(valid[:, None], sel.shape).reshape(-1)
            acc["expert_token_ids"].append(
                jnp.zeros((E, forward.VOCAB), jnp.int32)
                .at[sel.reshape(-1), jnp.repeat(flat_tok, K)]
                .add(vflat.astype(jnp.int32))
            )
            acc["expert_weight"].append(
                jnp.zeros((E,), jnp.float32).at[sel.reshape(-1)].add((lay["combine"].reshape(-1)) * vflat)
            )
            _, unbiased = _small_top_k(lay["router_logits"], K)
            bias_only = ~jnp.any(sel[:, :, None] == unbiased[:, None, :], axis=-1)
            acc["expert_bias_driven"].append(
                jnp.zeros((E,), jnp.float32).at[sel.reshape(-1)].add(bias_only.reshape(-1) * vflat)
            )
            acc["expert_dom"].append(jnp.zeros((E,), jnp.float32).at[sel.reshape(-1)].add(vflat))
            acc["high_dom"].append(acc["high"][-1])
            acc["s_high_dom"].append(acc["s_high"][-1])
        for key in ("fire", "act_sum", "act_sq", "high", "high_mass", "high_pos", "tok_mass", "expert_token_ids"):
            new[key] = state[key] + jnp.stack(acc[key])
        for key in ("s_fire", "s_act_sum", "s_high", "s_high_pos", "expert_weight", "expert_bias_driven"):
            new[key] = state[key] + jnp.stack(acc[key])
        new["act_max"] = jnp.maximum(state["act_max"], jnp.stack(acc["act_max"]))
        new["s_act_max"] = jnp.maximum(state["s_act_max"], jnp.stack(acc["s_act_max"]))
        new["high_dom"] = state["high_dom"].at[..., domain].add(jnp.stack(acc["high_dom"]))
        new["s_high_dom"] = state["s_high_dom"].at[..., domain].add(jnp.stack(acc["s_high_dom"]))
        new["expert_dom"] = state["expert_dom"].at[..., domain].add(jnp.stack(acc["expert_dom"]))
        new["top_val"], new["top_pos"] = _merge(
            state["top_val"],
            state["top_pos"],
            jnp.stack([v for v, _ in acc["top"]]),
            jnp.stack([p for _, p in acc["top"]]),
        )
        new["s_top_val"], new["s_top_pos"] = _merge(
            state["s_top_val"],
            state["s_top_pos"],
            jnp.stack([v for v, _ in acc["s_top"]]),
            jnp.stack([p for _, p in acc["s_top"]]),
        )
        new["dom_tokens"] = state["dom_tokens"].at[domain].add(jnp.sum(valid))
        new["dom_loss"] = state["dom_loss"].at[domain].add(jnp.sum(tr["loss"].reshape(-1) * valid))
        new["pos_tokens"] = state["pos_tokens"] + jax.ops.segment_sum(valid, pbucket, num_segments=NUM_POS)
        return new

    return step


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-sequences", type=int, default=4096)
    parser.add_argument("--paloma-sequences", type=int, default=128, help="per Paloma set")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    tab_host = tables(load_stats(args.tag))
    t0 = time.time()
    with set_mesh(mesh):
        model = common.compute_model(params)
        tab = {k: jnp.asarray(tab_host[k]) for k in ("thr", "s_thr", "slot")}
        state = init_state()
        step = make_step()
        seq_tokens, seq_segs, seq_valid, sources = [], [], [], []
        batches = [
            *common.paloma_batches(BATCH, max_sequences=args.paloma_sequences),
            *common.train_batches(BATCH, args.train_sequences, seed=args.seed),
        ]
        logger.info("%d batches loaded in %.0fs", len(batches), time.time() - t0)
        for n, batch in enumerate(batches):
            tokens, seg, w = common.device_batch(batch, mesh)
            offset = jnp.asarray(n * BATCH * common.SEQ_LEN, jnp.int32)
            domain = jnp.asarray(SOURCES.index(batch.source), jnp.int32)
            state = step(state, model, tokens, seg, w, domain, offset, tab)
            seq_tokens.append(batch.tokens.astype(np.uint16))
            seq_segs.append(batch.segment_ids.astype(np.int32))
            seq_valid.append((batch.loss_weight > 0).astype(np.uint8))
            sources += [SOURCES.index(batch.source)] * BATCH
            if n % 200 == 0:
                logger.info("batch %d/%d (%s) %.0fs", n, len(batches), batch.source, time.time() - t0)
        host = {k: np.asarray(v) for k, v in state.items()}
        host.update(stats.neuron_logit_lens(model))
        host["token_embed"] = np.asarray(params.token_embed, np.float16)
        host["router_bias"] = np.stack([np.asarray(layer.mlp.router_bias, np.float32) for layer in params.layers()])
    for key in ("act_sq", "high_mass", "tok_mass", "high_pos", "high_dom"):
        host[key] = host[key].astype(np.float32)
    host["thr"] = tab_host["thr"]
    host["s_thr"] = tab_host["s_thr"]
    host["expert_top_ids"] = tab_host["top_ids"].astype(np.int16)
    host["sources"] = np.asarray(SOURCES)
    common.save_npz(args.tag, "atlas", **host)
    common.save_npz(
        args.tag,
        "atlas_tokens",
        tokens=np.concatenate(seq_tokens),
        segment_ids=np.concatenate(seq_segs),
        valid=np.concatenate(seq_valid),
        sources=np.asarray(sources, np.int8),
    )
    logger.info("done in %.0fs, %d tokens", time.time() - t0, host["dom_tokens"].sum())


if __name__ == "__main__":
    main()
