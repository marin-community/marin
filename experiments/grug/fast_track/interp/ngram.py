# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""N-gram neuron search, scoring pass: for every routed and shared ReLU^2 unit, how well its candidate context
``(a, b)`` (previous, current token) explains its activation beyond the current token ``b`` alone, and how
specifically it promotes the candidate's next token ``c``.

Candidates come from ``stats.npz`` (top-16 activating positions per unit over the stats sample) and that sample's
tokens: each unit's modal ``(a, b)`` by top-context activation mass, the modal two-token prefix ``(z, a, b)``, and
``c``, the most frequent next token after ``(a, b)`` in the sample. The scoring pass runs on fresh unseen training
sequences (a different draw from the stats sample) and accumulates, per unit, count / sum / square sum of its
activation in four groups of the tokens it is computed on (routed: tokens routed to its expert; shared: all tokens):
``(a, b)``, ``b`` after another token, ``(z, a, b)``, and everything; plus the activation mass above a per-unit
threshold, routing counts, and the unit's lag-0 direct logit effect on ``c`` at its ``(a, b)`` occurrences.
Weights only: each unit's centred unembedding logit for ``c`` (z-score and rank over the vocabulary).

Writes ``ngram_score.npz`` (and the scored tokens, for the data counts the hypothesis test uses).
"""

import argparse
import io
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

BATCH = 8
L, E, I, V = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.VOCAB
S = common.SEQ_LEN
THRESHOLD_FRACTION = 0.3  # "high" activations: above this fraction of the unit's mean top-16 activation
SCORE_SEED = 1  # the stats sample used seed 0
NUM_GROUPS = 4  # (a, b), b after another token, (z, a, b), all
NONE = -1


def load_npz(tag: str, name: str) -> dict[str, np.ndarray]:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/{name}.npz", "rb") as f:
        data = np.load(io.BytesIO(f.read()))
        return {k: data[k] for k in data.files}


def context_arrays(tokens: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[B, S]`` tokens -> the previous, second-previous and next token within the same document (``NONE`` across a
    document or row boundary)."""
    tok = tokens.astype(np.int64)
    same = np.zeros_like(tok, bool)
    same[:, 1:] = seg[:, 1:] == seg[:, :-1]
    prev = np.full_like(tok, NONE)
    prev[:, 1:] = np.where(same[:, 1:], tok[:, :-1], NONE)
    prev2 = np.full_like(tok, NONE)
    prev2[:, 2:] = np.where(same[:, 2:] & same[:, 1:-1], tok[:, :-2], NONE)
    nxt = np.full_like(tok, NONE)
    nxt[:, :-1] = np.where(same[:, 1:], tok[:, 1:], NONE)
    return prev, prev2, nxt


def _modal(key: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``[..., k]`` keys: the key with the largest summed weight and its share of the row's weight."""
    eq = key[..., :, None] == key[..., None, :]
    mass = np.einsum("...ij,...j->...i", eq, weight)
    j = np.argmax(mass, axis=-1)[..., None]
    total = weight.sum(-1) + 1e-9
    return np.take_along_axis(key, j, -1)[..., 0], np.take_along_axis(mass, j, -1)[..., 0] / total


def next_token_table(tokens: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sample's bigram keys ``a * V + b`` (sorted, unique) with each one's count and most frequent next token."""
    prev, _, nxt = context_arrays(tokens, seg)
    tok = tokens.astype(np.int64)
    ok = (prev >= 0) & (nxt >= 0)
    ab = (prev * V + tok)[ok]
    tri, tri_n = np.unique(ab * V + nxt[ok], return_counts=True)
    tri_ab = tri // V
    # Most frequent continuation per bigram: sort by (bigram, -count).
    order = np.lexsort((-tri_n, tri_ab))
    tri_ab, tri, tri_n = tri_ab[order], tri[order], tri_n[order]
    first = np.concatenate([[True], tri_ab[1:] != tri_ab[:-1]])
    keys = tri_ab[first]
    counts = np.add.reduceat(tri_n, np.flatnonzero(first))
    return keys, counts, (tri[first] % V)


def candidates(stats: dict, tokens: np.ndarray, seg: np.ndarray) -> dict[str, np.ndarray]:
    """Each unit's candidate ``(z, a, b)`` and ``c`` from its top-16 contexts (routed ``[L, E, I]``, shared
    ``[L, I]``), the top-context shares of the modal ``b``, ``(a, b)`` and ``(z, a, b)``, and the high-activation
    threshold."""
    prev, prev2, _ = context_arrays(tokens, seg)
    flat_tok, flat_prev, flat_prev2 = (x.reshape(-1) for x in (tokens.astype(np.int64), prev, prev2))
    keys, _, next_tok = next_token_table(tokens, seg)
    out = {}
    for prefix, pos_key, val_key in (("", "top_pos", "top_val"), ("s_", "s_top_pos", "s_top_val")):
        pos, val = stats[pos_key], stats[val_key]
        w = np.maximum(val, 0.0)
        b, a, z = flat_tok[pos], flat_prev[pos], flat_prev2[pos]
        mb, share_b = _modal(b, w)
        mab, share_ab = _modal(a * V + b, w)
        # The modal two-token prefix among the contexts that are the modal bigram.
        on_ab = (a * V + b) == mab[..., None]
        mzab, share_zab = _modal(np.where(on_ab, z * V * V + a * V + b, -2), w * on_ab)
        ca, cb = mab // V, mab % V
        cz = np.where(mzab >= 0, mzab // (V * V), NONE)
        idx = np.clip(np.searchsorted(keys, mab), 0, len(keys) - 1)
        found = (keys[idx] == mab) & (ca >= 0)
        out[prefix + "cand_a"] = np.where(ca >= 0, ca, NONE).astype(np.int32)
        out[prefix + "cand_b"] = cb.astype(np.int32)
        out[prefix + "cand_z"] = cz.astype(np.int32)
        out[prefix + "cand_c"] = np.where(found, next_tok[idx], NONE).astype(np.int32)
        out[prefix + "top_share_b"] = share_b.astype(np.float32)
        out[prefix + "top_modal_b"] = mb.astype(np.int32)
        out[prefix + "top_share_ab"] = share_ab.astype(np.float32)
        out[prefix + "top_share_zab"] = (share_zab * share_ab).astype(np.float32)
        out[prefix + "threshold"] = (THRESHOLD_FRACTION * w.mean(-1)).astype(np.float32)
    return out


def _group_masks(ca, cb, cz, tok, prev, prev2):
    """Masks ``[N, I]`` of (a, b), b-after-another-token and (z, a, b), from the rows' candidates ``[N or 1, I]``."""
    is_b = cb == tok[:, None]
    is_ab = is_b & (ca == prev[:, None]) & (ca >= 0)
    is_zab = is_ab & (cz == prev2[:, None]) & (cz >= 0)
    return is_ab, is_b & ~is_ab, is_zab


def _accumulate(act, ok, masks, high_thr, reduce):
    """Per-row ``[N, I]`` activations -> ``[G, 3, ...]`` (count, sum, square sum) for every group plus the
    high-activation mass of (a, b) and of all ``[2, ...]``, each reduced over rows by ``reduce``."""
    okf = ok[:, None].astype(jnp.float32)
    stacked = []
    for g in [*masks, None]:
        gf = jnp.broadcast_to(okf, act.shape) if g is None else g.astype(jnp.float32) * okf
        stacked.append(jnp.stack([reduce(gf), reduce(gf * act), reduce(gf * act * act)]))
    high = jnp.where(act > high_thr, act, 0.0) * okf
    return jnp.stack(stacked), jnp.stack([reduce(high * masks[0]), reduce(high)])


def unit_directions(model) -> tuple[jax.Array, jax.Array]:
    """Each routed unit's lag-0 output direction in the residual stream through the final gain ``[L, E, I, D]`` and
    the shared units' ``[L, I, D]``."""
    m = forward.replicate(model)
    gain = forward.final_gain(m)
    routed, shared = [], []
    for lay in m.layers():
        lag0 = lay.sconv_mlp.weight[0].astype(jnp.float32) * gain
        up = lay.mlp.w_latent_up.astype(jnp.float32)
        routed.append(jnp.einsum("eir,rd->eid", lay.mlp.expert_mlp.w_down.astype(jnp.float32), up) * lag0)
        shared.append(lay.shared[0].w_down.astype(jnp.float32) * lag0)
    return jnp.stack(routed), jnp.stack(shared)


@jax.jit
def output_test(model, cand_c: jax.Array, s_cand_c: jax.Array) -> dict:
    """Weights only: each unit's centred unembedding logit for its ``c``, as a z-score over the vocabulary and a
    rank (0 = the unit's most promoted token), and the unit's top promoted token."""
    m = forward.replicate(model)
    w_out = m.output_proj.astype(jnp.float32)
    routed, shared = unit_directions(model)

    def one(dirs, c):  # [N, D], [N]
        logits = dirs @ w_out
        logits = logits - jnp.mean(logits, axis=-1, keepdims=True)
        std = jnp.std(logits, axis=-1)
        lc = jnp.take_along_axis(logits, jnp.maximum(c, 0)[:, None], axis=-1)[:, 0]
        rank = jnp.sum(logits > lc[:, None], axis=-1)
        return lc / std, rank, jnp.argmax(logits, axis=-1), lc

    out = jax.lax.map(lambda xs: one(*xs), (routed.reshape(L * E, I, -1), cand_c.reshape(L * E, I)))
    s_out = jax.lax.map(lambda xs: one(*xs), (shared, s_cand_c))
    names = ("c_z", "c_rank", "top_promoted", "c_logit")
    res = {n: v.reshape(L, E, I) for n, v in zip(names, out, strict=True)}
    res.update({"s_" + n: v for n, v in zip(names, s_out, strict=True)})
    return res


def make_step():
    @jax.jit
    def step(state, model, tokens, seg, weight, prev, prev2, nxt, cand, c_logit, s_c_logit):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        flat_tok, flat_prev, flat_prev2, flat_next, inv_rms = (
            reshard(x.reshape(-1), P(None)) for x in (tokens, prev, prev2, nxt, tr["inv_rms"])
        )
        valid = reshard(weight.reshape(-1), P(None)) > 0
        new = dict(state)
        for li, lay in enumerate(tr["layers"]):
            experts, toks = lay["expert_sorted"], lay["token_sorted"]
            act = lay["expert_act"]
            masks = _group_masks(
                *(cand[k][li][experts] for k in "abz"), flat_tok[toks], flat_prev[toks], flat_prev2[toks]
            )

            def by_expert(x, experts=experts):
                return jax.ops.segment_sum(x, experts, num_segments=E, indices_are_sorted=True)

            groups, high = _accumulate(act, valid[toks], masks, cand["thr"][li][experts], by_expert)
            new["groups"] = new["groups"].at[li].add(groups)
            new["high"] = new["high"].at[li].add(high)
            # Lag-0 direct logit effect on c at the (a, b) occurrences.
            w = lay["used_combine"][toks, lay["slot_sorted"]]
            c = cand["c"][li][experts]  # [N, I]
            dla = act * (w * inv_rms[toks])[:, None] * c_logit[li][experts] * masks[0] * valid[toks][:, None]
            new["dla_ab"] = new["dla_ab"].at[li].add(by_expert(dla))
            hit = masks[0] & (flat_next[toks][:, None] == c) & valid[toks][:, None]
            new["ab_next_c"] = (
                new["ab_next_c"].at[li].add(jax.ops.segment_sum(hit.astype(jnp.float32), experts, num_segments=E))
            )

            s_act = lay["shared_act"]
            s_masks = _group_masks(*(cand["s_" + k][li][None, :] for k in "abz"), flat_tok, flat_prev, flat_prev2)
            s_groups, s_high = _accumulate(
                s_act, valid, s_masks, cand["s_thr"][li][None, :], lambda x: jnp.sum(x, axis=0)
            )
            new["s_groups"] = new["s_groups"].at[li].add(s_groups)
            new["s_high"] = new["s_high"].at[li].add(s_high)
            s_dla = s_act * inv_rms[:, None] * s_c_logit[li][None, :] * s_masks[0] * valid[:, None]
            new["s_dla_ab"] = new["s_dla_ab"].at[li].add(jnp.sum(s_dla, axis=0))
        new["tokens"] = state["tokens"] + jnp.sum(valid)
        return new

    return step


def _candidate_device_tables(cand: dict) -> dict:
    tables = {k: jnp.asarray(cand[f"cand_{k}"]) for k in ("a", "b", "z", "c")}
    tables.update({f"s_{k}": jnp.asarray(cand[f"s_cand_{k}"]) for k in ("a", "b", "z", "c")})
    tables["thr"] = jnp.asarray(cand["threshold"])
    tables["s_thr"] = jnp.asarray(cand["s_threshold"])
    return tables


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-sequences", type=int, default=3072)
    args = parser.parse_args()
    t0 = time.time()
    stats = load_npz(args.tag, "stats")
    sample = load_npz(args.tag, "tokens")
    cand = candidates(stats, sample["tokens"], sample["segment_ids"])
    logger.info("candidates (%.0fs)", time.time() - t0)
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    with set_mesh(mesh):
        model = common.compute_model(params)
        tables = _candidate_device_tables(cand)
        state = dict(
            groups=jnp.zeros((L, NUM_GROUPS, 3, E, I)),
            high=jnp.zeros((L, 2, E, I)),
            dla_ab=jnp.zeros((L, E, I)),
            ab_next_c=jnp.zeros((L, E, I)),
            s_groups=jnp.zeros((L, NUM_GROUPS, 3, I)),
            s_high=jnp.zeros((L, 2, I)),
            s_dla_ab=jnp.zeros((L, I)),
            tokens=jnp.zeros(()),
        )
        step = make_step()
        ot = output_test(model, tables["c"], tables["s_c"])
        seq_tokens, seq_segs, seq_valid = [], [], []
        for n, batch in enumerate(common.train_batches(BATCH, args.train_sequences, seed=SCORE_SEED)):
            prev, prev2, nxt = context_arrays(batch.tokens, batch.segment_ids)
            tokens, seg, w = common.device_batch(batch, mesh)
            state = step(
                state,
                model,
                tokens,
                seg,
                w,
                jnp.asarray(prev, jnp.int32),
                jnp.asarray(prev2, jnp.int32),
                jnp.asarray(nxt, jnp.int32),
                tables,
                ot["c_logit"],
                ot["s_c_logit"],
            )
            seq_tokens.append(batch.tokens.astype(np.uint16))
            seq_segs.append(batch.segment_ids.astype(np.int32))
            seq_valid.append((batch.loss_weight > 0).astype(np.uint8))
            if n % 50 == 0:
                logger.info("batch %d (%.0fs)", n, time.time() - t0)
        host = {k: np.asarray(v) for k, v in state.items()}
        host.update({f"ot_{k}": np.asarray(v) for k, v in ot.items()})
    host.update(cand)
    common.save_npz(args.tag, "ngram_score", **host)
    common.save_npz(
        args.tag,
        "ngram_tokens",
        tokens=np.concatenate(seq_tokens),
        segment_ids=np.concatenate(seq_segs),
        valid=np.concatenate(seq_valid),
    )
    logger.info("done in %.0fs, %d tokens", time.time() - t0, host["tokens"])


if __name__ == "__main__":
    main()
