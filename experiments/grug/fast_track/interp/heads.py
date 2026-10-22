# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What every attention head does.

MLA heads (layers 3, 5): the full softmax rows ``q . k / sqrt(d) + Inkling bias`` (causal, same document), with the
logit split into the key-offset channels (the first 64 key channels hold the *previous* token's key), the current
channels and the bias. Per head: entropy, attention distance, mass at the document start, on the "induction" keys
``s`` with ``tokens[s - 1] == tokens[t]`` (the offset channels can match the query's token against the key's
predecessor, so attending there retrieves the token that followed it) and on same-token keys; the mean Inkling bias by
distance; the expected logit parts under the attention.

KDA heads (layers 0, 1, 2, 4): the delta rule ``S_t = (I - beta_t k_t k_t^T) diag(exp g_t) S_(t-1) + beta_t k_t v_t^T``,
``o_t = S_t^T q_t`` is a linear token mixer ``o_t = sum_s A[t, s] v_s`` with ``A[t, s] = beta_s k_s . (M_(s+1)^T ...
M_t^T q_t)``; ``A`` is computed exactly by a backward recurrence (checked against the kernel output) and summarized like
the MLA rows (by ``|A|``), plus each channel's decay half-life and the write strength ``beta``.

Induction test: random-token segments repeated twice; the loss on the first vs second copy, each head's induction
score, and every head's zero-ablation effect on the second copy's loss.

Writes ``heads.npz``.
"""

import argparse
import logging
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.attention._inkling_relpos import band_to_compact, dense_rel_bias

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

BATCH = 4
DIST_EDGES = (0, 1, 2, 5, 17, 65, 257, 1025, 4097)  # distance bins [e_i, e_(i+1))
EXCERPT = 96  # tokens of the example patterns
INDUCTION_LEN = 128  # tokens per random segment (repeated once: 256-token documents)
REL_EXTENT = 1024
HALF = forward.HEAD_DIM // 2


def _dist_bins(s: int) -> jax.Array:
    dist = jnp.arange(s)[:, None] - jnp.arange(s)[None, :]
    return jnp.sum(dist[..., None] >= jnp.asarray(DIST_EDGES[1:-1]), axis=-1)  # [S, S] bin index


def row_stats(a: jax.Array, tokens: jax.Array, seg: jax.Array, valid_q: jax.Array) -> dict:
    """Summaries of ``[H, S, S]`` nonnegative attention rows (rows sum to 1) for one sequence."""
    s = tokens.shape[0]
    bins = _dist_bins(s)
    nb = len(DIST_EDGES) - 1
    onehot = jax.nn.one_hot(bins, nb)  # [S, S, nb]
    by_dist = jnp.einsum("hts,tsb->htb", a, onehot)
    dist = (jnp.arange(s)[:, None] - jnp.arange(s)[None, :]).astype(jnp.float32)
    mean_dist = jnp.sum(a * dist, axis=-1)
    entropy = -jnp.sum(a * jnp.log(jnp.maximum(a, 1e-30)), axis=-1)
    same_doc = seg[:, None] == seg[None, :]
    causal = jnp.arange(s)[None, :] <= jnp.arange(s)[:, None]
    ok = same_doc & causal
    first = jnp.argmax(seg[None, :] == seg[:, None], axis=-1)  # each query's document start
    doc_start = jnp.take_along_axis(a, jnp.broadcast_to(first[None, :, None], (a.shape[0], s, 1)), axis=-1)[..., 0]
    prev_tok = jnp.pad(tokens[:-1], (1, 0), constant_values=-1)
    induct = (
        ok
        & (prev_tok[None, :] == tokens[:, None])
        & (jnp.arange(s)[None, :] >= 1)
        & (first[:, None] < jnp.arange(s)[None, :])
    )
    same_tok = ok & (tokens[None, :] == tokens[:, None]) & (jnp.arange(s)[None, :] < jnp.arange(s)[:, None])
    has_ind = jnp.any(induct, axis=-1) & valid_q
    has_same = jnp.any(same_tok, axis=-1) & valid_q
    vq = valid_q.astype(jnp.float32)
    return dict(
        n=jnp.sum(vq),
        by_dist=jnp.einsum("htb,t->hb", by_dist, vq),
        mean_dist=jnp.einsum("ht,t->h", mean_dist, vq),
        entropy=jnp.einsum("ht,t->h", entropy, vq),
        doc_start=jnp.einsum("ht,t->h", doc_start, vq),
        induction=jnp.einsum("ht,t->h", jnp.sum(a * induct, axis=-1), has_ind.astype(jnp.float32)),
        n_induction=jnp.sum(has_ind.astype(jnp.float32)),
        same_token=jnp.einsum("ht,t->h", jnp.sum(a * same_tok, axis=-1), has_same.astype(jnp.float32)),
        n_same_token=jnp.sum(has_same.astype(jnp.float32)),
        self_mass=jnp.einsum("ht,t->h", jnp.diagonal(a, axis1=1, axis2=2), vq),
        prev_mass=jnp.einsum("ht,t->h", jnp.pad(jnp.diagonal(a, offset=-1, axis1=1, axis2=2), ((0, 0), (1, 0))), vq),
    )


def mla_rows(lay: dict, seg: jax.Array):
    """Softmax rows ``[H, S, S]`` of one sequence and their logit parts (offset channels, current channels, bias)."""
    q = lay["q"].astype(jnp.float32)  # [S, H, D]
    k = lay["k"].astype(jnp.float32)
    s = q.shape[0]
    scale = 1.0 / math.sqrt(forward.HEAD_DIM)
    prev = jnp.einsum("thd,shd->hts", q[..., :HALF], k[..., :HALF]) * scale
    cur = jnp.einsum("thd,shd->hts", q[..., HALF:], k[..., HALF:]) * scale
    bias = dense_rel_bias(lay["rel_bias"][None].astype(jnp.float32), s, s)[0]
    ok = (seg[:, None] == seg[None, :]) & (jnp.arange(s)[None, :] <= jnp.arange(s)[:, None])
    logits = jnp.where(ok[None], prev + cur + bias, -jnp.inf)
    return jax.nn.softmax(logits, axis=-1), prev, cur, bias


def kda_matrix(lay: dict, seg: jax.Array, h: int) -> jax.Array:
    """Exact KDA mixing matrix ``A [S, S]`` (``o_t = sum_s A[t, s] v_s``) of head ``h`` of one sequence."""
    q = lay["q"][:, h].astype(jnp.float32)
    k = lay["k"][:, h].astype(jnp.float32)
    eps = 1e-6
    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + eps) / math.sqrt(forward.HEAD_DIM)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + eps)
    g = lay["log_decay"][:, h].astype(jnp.float32)  # [S, D]
    beta = lay["beta"][:, h].astype(jnp.float32)  # [S]
    s = q.shape[0]
    keep = jnp.concatenate([jnp.ones((1,)), (seg[1:] == seg[:-1]).astype(jnp.float32)])

    def step(w, r):
        started = jnp.arange(s) >= r  # queries t >= r
        kr, br = k[r], beta[r]
        dots = w @ kr  # [S]
        col = jnp.where(started, br * dots, 0.0)
        w_new = (w - br * dots[:, None] * kr[None, :]) * (jnp.exp(g[r]) * keep[r])[None, :]
        return jnp.where(started[:, None], w_new, w), col

    _, cols = jax.lax.scan(step, q, jnp.arange(s - 1, -1, -1))
    return cols[::-1].T  # [t, s]


def make_fns(capacity: int):
    keep = ("q", "k", "rel_bias", "v", "log_decay", "beta", "kda_o")

    @jax.jit
    def trace(model, tokens, seg):
        tr = forward.trace_forward(forward.replicate(model), tokens, seg, forward.no_ablation(), capacity=capacity)
        return forward.replicate([{k: lay[k] for k in keep if k in lay} for lay in tr["layers"]])

    @jax.jit
    def mla_seq_stats(lay, tokens_b, seg_b, valid_b):
        a, prev, cur, bias = mla_rows(lay, seg_b)
        st = row_stats(a, tokens_b, seg_b, valid_b)
        vq = valid_b.astype(jnp.float32)
        st["e_prev"] = jnp.einsum("hts,t->h", jnp.where(a > 0, a * prev, 0.0), vq)
        st["e_cur"] = jnp.einsum("hts,t->h", jnp.where(a > 0, a * cur, 0.0), vq)
        st["e_bias"] = jnp.einsum("hts,t->h", jnp.where(a > 0, a * bias, 0.0), vq)
        compact = band_to_compact(lay["rel_bias"][None].astype(jnp.float32))[0]  # [H, S, extent]
        st["bias_by_dist"] = jnp.einsum("htd,t->hd", compact, vq)
        st["bias_by_dist_sq"] = jnp.einsum("htd,t->hd", jnp.square(compact), vq)
        return st, a[:, :EXCERPT, :EXCERPT]

    @jax.jit
    def kda_seq_stats(lay, tokens_b, seg_b, valid_b):
        mats = jnp.stack([kda_matrix(lay, seg_b, h) for h in range(forward.NUM_HEADS)])  # [H, S, S]
        v = lay["v"].astype(jnp.float32)  # [S, H, D]
        recon = jnp.einsum("hts,shd->thd", mats, v)
        o = lay["kda_o"].astype(jnp.float32)
        rel_err = jnp.linalg.norm(recon - o) / jnp.linalg.norm(o)
        mag = jnp.abs(mats)
        norm = mag / jnp.maximum(jnp.sum(mag, axis=-1, keepdims=True), 1e-30)
        st = row_stats(norm, tokens_b, seg_b, valid_b)
        vq = valid_b.astype(jnp.float32)
        st["signed_sum"] = jnp.einsum("hts,t->h", mats, vq)
        st["abs_sum"] = jnp.einsum("hts,t->h", mag, vq)
        st["neg_frac"] = jnp.einsum(
            "hts,t->h", jnp.where(mats < 0, mag, 0.0) / jnp.maximum(jnp.sum(mag, -1, keepdims=True), 1e-30), vq
        )
        g = lay["log_decay"]  # [S, H, D]
        st["log_decay_sum"] = jnp.einsum("shd,s->hd", g, vq)
        st["beta_sum"] = jnp.einsum("sh,s->h", lay["beta"], vq)
        st["beta_hist"] = jnp.stack(
            [
                jnp.zeros((50,)).at[jnp.clip((lay["beta"][:, h] * 50).astype(jnp.int32), 0, 49)].add(vq)
                for h in range(forward.NUM_HEADS)
            ]
        )
        return st, mats[:, :EXCERPT, :EXCERPT], rel_err

    @jax.jit
    def head_ablation_loss(model, tokens, seg, mult):
        abl = forward.no_ablation()._replace(head_mult=mult)
        tr = forward.trace_forward(forward.replicate(model), tokens, seg, abl, capacity=capacity)
        return tr["loss"]

    return trace, mla_seq_stats, kda_seq_stats, head_ablation_loss


def induction_batch(n_seq: int, seed: int) -> common.TokenBatch:
    """Documents of ``INDUCTION_LEN`` random tokens followed by the same tokens again."""
    rng = np.random.default_rng(seed)
    per = common.SEQ_LEN // (2 * INDUCTION_LEN)
    first = rng.integers(500, forward.VOCAB - 100, size=(n_seq, per, INDUCTION_LEN))
    tokens = np.concatenate([first, first], axis=-1).reshape(n_seq, common.SEQ_LEN).astype(np.int32)
    seg = np.repeat(np.arange(per, dtype=np.int32), 2 * INDUCTION_LEN)[None].repeat(n_seq, 0)
    return common.TokenBatch("induction", tokens, seg, np.ones(tokens.shape, np.float32))


def _acc(acc: dict, key: str, st: dict):
    d = acc.setdefault(key, {})
    for k, v in st.items():
        d[k] = d.get(k, 0.0) + np.asarray(v, np.float64)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--paloma-sequences", type=int, default=1, help="per Paloma set")
    parser.add_argument("--induction-sequences", type=int, default=4)
    args = parser.parse_args()
    checkpoint = args.checkpoint or common.CHECKPOINTS[args.tag]
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, checkpoint)
    capacity = forward.default_capacity(BATCH * common.SEQ_LEN)
    t0 = time.time()
    natural = [
        *common.paloma_batches(BATCH, max_sequences=args.paloma_sequences),
        *common.train_batches(BATCH, args.train_sequences, seed=3),
    ]
    induction = [induction_batch(BATCH, seed) for seed in range(args.induction_sequences // BATCH)]
    results: dict[str, np.ndarray] = {}
    acc: dict[str, dict] = {}
    examples: dict[str, list] = {}
    kda_errs = []
    with set_mesh(mesh):
        model = common.compute_model(params)
        trace, mla_fn, kda_fn, ablate_fn = make_fns(capacity)
        for kind, batches in (("natural", natural), ("induction", induction)):
            for bi, batch in enumerate(batches):
                tokens, seg, w = common.device_batch(batch, mesh)
                layers = trace(model, tokens, seg)
                tokens, seg, w = forward.replicate((tokens, seg, w))
                for b in range(BATCH):
                    if batch.loss_weight[b].sum() == 0:
                        continue
                    tb, sb, vb = tokens[b], seg[b], w[b] > 0
                    for i, lay in enumerate(layers):
                        if i in forward.MLA_LAYERS:
                            sub = {k: lay[k][b] for k in ("q", "k", "rel_bias")}
                            st, ex = mla_fn(sub, tb, sb, vb)
                        else:
                            sub = {k: lay[k][b] for k in ("q", "k", "v", "log_decay", "beta", "kda_o")}
                            st, ex, err = kda_fn(sub, tb, sb, vb)
                            kda_errs.append(float(err))
                        _acc(acc, f"{kind}/L{i}", st)
                        if bi == 0 and b == 0:
                            examples.setdefault(f"{kind}/L{i}", []).append(np.asarray(ex))
                    if bi == 0 and b == 0:
                        examples.setdefault(f"{kind}/tokens", []).append(np.asarray(batch.tokens[b, :EXCERPT]))
                logger.info(
                    "%s batch %d done (%.0fs), kda recon err max %.2e", kind, bi, time.time() - t0, max(kda_errs or [0])
                )
        # Induction: loss on the first vs the second copy, and each head's zero-ablation effect on it.
        second = np.tile((np.arange(2 * INDUCTION_LEN) >= INDUCTION_LEN), common.SEQ_LEN // (2 * INDUCTION_LEN))
        first_mask = ~second
        # Position t predicts t+1: drop the last token of each document (it predicts the next document).
        last = np.tile(np.arange(2 * INDUCTION_LEN) == 2 * INDUCTION_LEN - 1, common.SEQ_LEN // (2 * INDUCTION_LEN))
        second = second & ~last
        first_mask = first_mask & ~(
            np.tile(np.arange(2 * INDUCTION_LEN) == INDUCTION_LEN - 1, common.SEQ_LEN // (2 * INDUCTION_LEN))
        )
        ind_loss = {}
        configs = [("none", None)] + [
            (f"L{i}.h{h}", (i, h)) for i in range(forward.NUM_LAYERS) for h in range(forward.NUM_HEADS)
        ]
        configs += [(f"L{i}.all", (i, None)) for i in range(forward.NUM_LAYERS)]
        for name, target in configs:
            mult = jnp.ones((forward.NUM_LAYERS, forward.NUM_HEADS))
            if target is not None:
                i, h = target
                mult = mult.at[i].set(0.0) if h is None else mult.at[i, h].set(0.0)
            f_sum = s_sum = 0.0
            pos_loss = np.zeros(2 * INDUCTION_LEN)
            for batch in induction:
                tokens, seg, _ = common.device_batch(batch, mesh)
                loss = np.asarray(ablate_fn(model, tokens, seg, mult))
                f_sum += float(loss[:, first_mask].mean())
                s_sum += float(loss[:, second].mean())
                pos_loss += loss.reshape(BATCH, -1, 2 * INDUCTION_LEN).mean(axis=(0, 1))
            ind_loss[name] = (f_sum / len(induction), s_sum / len(induction), pos_loss / len(induction))
            logger.info("induction %s: first %.3f second %.3f", name, *ind_loss[name][:2])
    for key, d in acc.items():
        for k, v in d.items():
            results[f"{key}/{k}"] = v
    for key, exs in examples.items():
        results[f"example/{key}"] = np.stack(exs)
    results["induction_names"] = np.asarray([n for n, _ in configs])
    results["induction_first"] = np.asarray([ind_loss[n][0] for n, _ in configs])
    results["induction_second"] = np.asarray([ind_loss[n][1] for n, _ in configs])
    results["induction_pos_loss"] = np.stack([ind_loss[n][2] for n, _ in configs])
    results["kda_recon_err"] = np.asarray(kda_errs)
    results["dist_edges"] = np.asarray(DIST_EDGES)
    common.save_npz(args.tag, "heads", **results)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
