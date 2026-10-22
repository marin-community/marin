# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two deep dives: the MLA value-embedding (VE) copy path, and whether the KDA heads use a sink-like mechanism.

VE (layers 3 and 5, ``v = lambda1 v_proj + sigmoid(x W_ve) * VE[token]``):

- weights: each token's VE row per head through ``w_o``, the branch SConv's lag-0 tap and the final norm gain onto the
  unembedding: the rank and margin of the token itself, of its case/space variants and of its most frequent next token;
  cosines with the token's lm_head column and input embedding row; row norms; token frequency;
- natural text: per-head VE direct logit attribution by category, the gate ``w`` and ``|w VE| / |lambda1 v_proj|`` per
  head, worked examples where the L5 VE term moves the correct logit by more than 2 nats (attended keys, gate, logit
  parts, loss without it), failure cases where it promotes a wrong top-1;
- ablations by category and on the copy task: L3 / L5 / both VE zeroed or mean-replaced, and each L5 head's VE alone;
- path patching into L5: an upstream component is mean-ablated only in the input of L5's queries or only in its keys;
  the effect on L5's induction attention and on the L5 VE direct logit for the target.

KDA sink: from the exact mixing matrices, the share of each head's ``|A|`` (and signed ``A``) on the document's first 1,
4 and 16 tokens by query position, the early writes' share of the output, persistent channels, "no-op" routes (beta,
output gate, read norm), and the loss when the early tokens' writes are zeroed or mean-replaced.

Writes ``deepdive.npz``.
"""

import argparse
import io
import json
import logging
import math
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.attention._inkling_relpos import dense_rel_bias

from experiments.grug.fast_track.interp import categories, common, forward, heads

logger = logging.getLogger(__name__)

BATCH = 4
L, H, HD = forward.NUM_LAYERS, forward.NUM_HEADS, forward.HEAD_DIM
VE_LAYERS = forward.MLA_LAYERS
EARLY = (1, 4, 16)
POS_EDGES = (0, 16, 64, 256, 1024, 4096)  # query position-in-document bins
EXAMPLE_THRESHOLD = 2.0
MAX_EXAMPLES = 40
EXAMPLE_LEFT = 24
TOP_KEYS = 3
TOP_TOKENS = 6


def _normalized(s: str) -> str:
    return s.strip().lower()


def weight_analysis(model, vocab: list[str], next_token: np.ndarray, freq: np.ndarray) -> dict:
    """VE rows through each head's ``w_o`` (and all heads summed) onto the unembedding."""
    model = forward.replicate(model)
    unembed = forward.unembed_direction(model)  # [D, V]
    mean_dir = jnp.mean(unembed, axis=1, keepdims=True)
    centered = unembed - mean_dir
    lm_cols = model.output_proj.astype(jnp.float32).T  # [V, D]
    embed = model.token_embed.astype(jnp.float32)  # [V, D]
    variants: dict[str, list[int]] = {}
    for i, s in enumerate(vocab):
        variants.setdefault(_normalized(s), []).append(i)
    variant_lists = [[j for j in variants[_normalized(s)] if j != i] for i, s in enumerate(vocab)]
    out = {}
    layers = model.layers()
    for li in VE_LAYERS:
        attn = layers[li].attn
        lag0 = layers[li].sconv_attn.weight[0].astype(jnp.float32)
        ve = attn.value_embed.astype(jnp.float32).reshape(-1, H, HD)  # [V, H, hd]
        w_o = attn.w_o.astype(jnp.float32).reshape(H, HD, -1)  # [H, hd, D]

        @jax.jit
        def block(rows, ids, w_o, lag0):
            # rows [C, H, hd] -> per head and summed: residual direction [C, H+1, D] and its vocab logits.
            dirs = jnp.einsum("chk,hkd->chd", rows, w_o) * lag0
            dirs = jnp.concatenate([dirs, jnp.sum(dirs, axis=1, keepdims=True)], axis=1)
            logits = jnp.einsum("chd,dv->chv", dirs, centered)
            self_logit = jnp.take_along_axis(logits, ids[:, None, None], axis=-1)[..., 0]
            rank = jnp.sum(logits > self_logit[..., None], axis=-1)
            other = jnp.where(jax.nn.one_hot(ids, logits.shape[-1], dtype=bool)[:, None, :], -jnp.inf, logits)
            margin = self_logit - jnp.max(other, axis=-1)
            nxt = jnp.take_along_axis(logits, jnp.asarray(next_token)[ids][:, None, None], axis=-1)[..., 0]
            next_rank = jnp.sum(logits > nxt[..., None], axis=-1)
            norm = jnp.sqrt(jnp.sum(jnp.square(dirs), axis=-1))
            cos_lm = jnp.einsum("chd,cd->ch", dirs, lm_cols[ids]) / (
                norm * jnp.linalg.norm(lm_cols[ids], axis=-1)[:, None] + 1e-9
            )
            cos_emb = jnp.einsum("chd,cd->ch", dirs, embed[ids]) / (
                norm * jnp.linalg.norm(embed[ids], axis=-1)[:, None] + 1e-9
            )
            return rank, margin, next_rank, norm, cos_lm, cos_emb, logits

        res = {k: [] for k in ("rank", "margin", "next_rank", "norm", "cos_lm", "cos_emb", "variant_rank")}
        chunk = 256
        for start in range(0, len(vocab), chunk):
            ids = jnp.arange(start, min(len(vocab), start + chunk))
            rank, margin, next_rank, norm, cos_lm, cos_emb, logits = block(ve[ids], ids, w_o, lag0)
            logits = np.asarray(logits)
            vr = np.full((len(ids), H + 1), -1, np.int64)
            for r, tok in enumerate(np.asarray(ids)):
                if variant_lists[tok]:
                    best = logits[r][:, variant_lists[tok]].max(axis=-1)  # [H+1]
                    vr[r] = (logits[r] > best[:, None]).sum(-1)
            for k, v in zip(
                ("rank", "margin", "next_rank", "norm", "cos_lm", "cos_emb"),
                (rank, margin, next_rank, norm, cos_lm, cos_emb),
                strict=True,
            ):
                res[k].append(np.asarray(v))
            res["variant_rank"].append(vr)
        for k, v in res.items():
            out[f"w/L{li}/{k}"] = np.concatenate(v)
        out[f"w/L{li}/lambda"] = np.asarray(attn.ve_lambda, np.float32)
    out["w/freq"] = freq
    out["w/has_variant"] = np.asarray([bool(v) for v in variant_lists])
    return out


def make_fns():
    @jax.jit
    def trace_small(model, tokens, seg, abl):
        tr = forward.replicate(forward.trace_forward(forward.replicate(model), tokens, seg, abl))
        keep_layer = (
            "q",
            "k",
            "rel_bias",
            "v_proj",
            "ve",
            "ve_weight",
            "head_gate",
            "ve_head_out",
            "attn_in",
            "log_decay",
            "beta",
            "v",
            "kda_o",
            "head_out",
        )
        return dict(
            layers=[{k: lay[k] for k in keep_layer if k in lay} for lay in tr["layers"]],
            w_final=tr["w_final"],
            inv_rms=tr["inv_rms"],
            loss=tr["loss"],
            labels=tr["labels"],
            top_ids=tr["top_ids"],
        )

    @jax.jit
    def loss_fn(model, tokens, seg, abl):
        return forward.trace_forward(forward.replicate(model), tokens, seg, abl)["loss"]

    @jax.jit
    def ve_natural(model, tr, tokens, cats):
        """Per-head VE DLA to the target and the top-1 by category, gate and magnitude statistics."""
        m = forward.replicate(model)
        unembed = forward.unembed_direction(m)
        mean_dir = jnp.mean(unembed, axis=1)
        labels, top1 = tr["labels"], tr["top_ids"][..., 0]
        u_y = unembed.T[labels] - mean_dir
        u_t = unembed.T[top1] - mean_dir
        inv = tr["inv_rms"][..., None]
        out = {}
        for li in VE_LAYERS:
            lay = tr["layers"][li]
            w = tr["w_final"][li + 1][..., None, None]  # [B, S, 1, 1]
            v = lay["ve_head_out"].astype(jnp.float32) * w * inv[..., None]  # [B, S, H, D]
            dla_y = jnp.einsum("bshd,bsd->bsh", v, u_y)
            dla_t = jnp.einsum("bshd,bsd->bsh", v, u_t)
            hout = lay["head_out"].astype(jnp.float32) * w * inv[..., None]
            out[f"L{li}/dla_ve_y"] = jnp.einsum("cbs,bsh->ch", cats, dla_y)
            out[f"L{li}/dla_ve_top"] = jnp.einsum("cbs,bsh->ch", cats, dla_t)
            out[f"L{li}/dla_head_y"] = jnp.einsum("cbs,bsh->ch", cats, jnp.einsum("bshd,bsd->bsh", hout, u_y))
            gate = lay["ve_weight"].astype(jnp.float32)  # [B, S, H]
            out[f"L{li}/gate_mean"] = jnp.einsum("cbs,bsh->ch", cats, gate)
            out[f"L{li}/gate_hist"] = jnp.stack(
                [
                    jnp.zeros((20,))
                    .at[jnp.clip((gate[..., h] * 20).astype(jnp.int32), 0, 19).reshape(-1)]
                    .add(cats[0].reshape(-1))
                    for h in range(H)
                ]
            )
            ve_n = jnp.sqrt(jnp.sum(jnp.square(lay["ve"].astype(jnp.float32)), -1))
            vp_n = jnp.sqrt(jnp.sum(jnp.square(lay["v_proj"].astype(jnp.float32)), -1))
            ratio = jnp.log10(jnp.maximum(ve_n, 1e-9) / jnp.maximum(vp_n, 1e-9))
            out[f"L{li}/ratio_mean"] = jnp.einsum("cbs,bsh->ch", cats, ratio)
            out[f"L{li}/ratio_hist"] = jnp.stack(
                [
                    jnp.zeros((40,))
                    .at[jnp.clip(((ratio[..., h] + 2) * 10).astype(jnp.int32), 0, 39).reshape(-1)]
                    .add(cats[0].reshape(-1))
                    for h in range(H)
                ]
            )
            out[f"L{li}/dla_ve_y_tok"] = dla_y
            out[f"L{li}/dla_ve_top_tok"] = dla_t
        return out

    @jax.jit
    def head_row(lay, b, t, h, seg_b):
        b, t, h = jnp.asarray(b), jnp.asarray(t), jnp.asarray(h)
        """One MLA query's softmax row and logit parts ``[S]`` (head ``h``)."""
        q = lay["q"][b, t, h].astype(jnp.float32)
        k = lay["k"][b, :, h].astype(jnp.float32)
        s = k.shape[0]
        scale = 1.0 / math.sqrt(HD)
        prev = (k[:, : HD // 2] @ q[: HD // 2]) * scale
        cur = (k[:, HD // 2 :] @ q[HD // 2 :]) * scale
        band = jax.lax.dynamic_slice_in_dim(jax.lax.dynamic_slice_in_dim(lay["rel_bias"], b, 1, 0), h, 1, 1)
        bias = dense_rel_bias(band.astype(jnp.float32), s, s)[0, 0, t]
        ok = (seg_b == seg_b[t]) & (jnp.arange(s) <= t)
        p = jax.nn.softmax(jnp.where(ok, prev + cur + bias, -jnp.inf))
        return p, prev, cur, bias

    @jax.jit
    def ve_ov(model, lay_ve, lay_gate, lay_vfull, b, t, h, keys, probs):
        """Vocab directions of the top keys' VE terms (``a_j P_perp(ve_j)``) through gate, w_o, SConv lag 0 and gain."""
        m = forward.replicate(model)
        layer = m.layers()[5]
        unembed = forward.unembed_direction(m)
        centered = unembed - jnp.mean(unembed, axis=1, keepdims=True)
        w_o = jnp.take(layer.attn.w_o.astype(jnp.float32).reshape(H, HD, -1), h, axis=0)
        lag0 = layer.sconv_attn.weight[0].astype(jnp.float32)
        v_t = lay_vfull[b, t, h].astype(jnp.float32)
        ve_j = lay_ve[b, keys, h].astype(jnp.float32)  # [K, hd]
        y = probs[:, None] * ve_j
        y = y - (y @ v_t)[:, None] * v_t[None] / (v_t @ v_t + 1e-6)
        d = (lay_gate[b, t, h] * y) @ w_o * lag0  # [K, D]
        logits = d @ centered
        top_v, top_i = jax.lax.top_k(logits, TOP_TOKENS)
        return top_i, top_v

    return trace_small, loss_fn, ve_natural, head_row, ve_ov


def kda_sink_stats(lay, tok, seg, valid) -> dict:
    """Doc-start mass of each KDA head's exact mixing matrix, early writes' share of the output, no-op routes."""
    mats = jnp.stack([heads.kda_matrix(lay, seg, h) for h in range(H)])  # [H, S, S]
    pos = forward.position_in_document(seg[None])[0]
    key_pos = pos[None, :]
    same = seg[:, None] == seg[None, :]
    mag = jnp.abs(mats)
    tot_abs = jnp.maximum(jnp.sum(mag, -1), 1e-30)
    tot_signed = jnp.sum(mats, -1)
    qbin = jnp.sum(pos[:, None] >= jnp.asarray(POS_EDGES[1:-1]), axis=-1)
    onehot = jax.nn.one_hot(qbin, len(POS_EDGES) - 1) * valid[:, None]
    out = {"n_by_pos": jnp.sum(onehot, 0)}
    v = lay["v"].astype(jnp.float32)
    o = jnp.einsum("hts,shd->thd", mats, v)
    o_n = jnp.linalg.norm(o, axis=-1)  # [S, H]
    for k in EARLY:
        early = same & (key_pos < k)
        frac_abs = jnp.sum(mag * early, -1) / tot_abs  # [H, S]
        frac_signed = jnp.sum(mats * early, -1) / jnp.where(jnp.abs(tot_signed) > 1e-6, tot_signed, 1e-6)
        out[f"early{k}_abs"] = jnp.einsum("hs,sb->hb", frac_abs, onehot)
        out[f"early{k}_signed"] = jnp.einsum("hs,sb->hb", jnp.clip(frac_signed, -5, 5), onehot)
        contrib = jnp.einsum("hts,shd->thd", mats * early, v)
        share = jnp.linalg.norm(contrib, axis=-1) / jnp.maximum(o_n, 1e-9)
        out[f"early{k}_out_share"] = jnp.einsum("sh,sb->hb", share, onehot)
    gate = lay["head_gate"].astype(jnp.float32)  # [S, H]
    beta = lay["beta"].astype(jnp.float32)
    med = jnp.median(jnp.where(valid[:, None], o_n, jnp.nan), axis=0)
    vq = valid.astype(jnp.float32)
    out["beta_small"] = jnp.einsum("sh,s->h", (beta < 0.05).astype(jnp.float32), vq)
    out["gate_small"] = jnp.einsum("sh,s->h", (gate < 0.1).astype(jnp.float32), vq)
    out["read_small"] = jnp.einsum("sh,s->h", (o_n < 0.1 * med[None]).astype(jnp.float32), vq)
    out["gate_mean"] = jnp.einsum("sh,s->h", gate, vq)
    g = lay["log_decay"].astype(jnp.float32)  # [S, H, D]
    out["log_decay_sum"] = jnp.einsum("shd,s->hd", g, vq)
    out["n"] = jnp.sum(vq)
    return out


def mla_doc_start(lay, seg, valid) -> dict:
    """The same doc-start shares for the MLA heads' softmax rows."""
    a, _, _, _ = heads.mla_rows(lay, seg)  # [H, S, S]
    pos = forward.position_in_document(seg[None])[0]
    same = seg[:, None] == seg[None, :]
    qbin = jnp.sum(pos[:, None] >= jnp.asarray(POS_EDGES[1:-1]), axis=-1)
    onehot = jax.nn.one_hot(qbin, len(POS_EDGES) - 1) * valid[:, None]
    out = {}
    for k in EARLY:
        out[f"early{k}_abs"] = jnp.einsum("hs,sb->hb", jnp.sum(a * (same & (pos[None, :] < k)), -1), onehot)
    out["n_by_pos"] = jnp.sum(onehot, 0)
    return out


def patched_l5(model, clean, patched, tokens, seg) -> dict:
    """L5 attention with queries from one run and keys from another: induction mass per head, and the L5 VE direct
    logit for the target (clean values, gate and final weights)."""
    m = forward.replicate(model)
    layer = m.layers()[5]
    attn = layer.attn
    x_c, x_p = clean["layers"][5]["attn_in"], patched["layers"][5]["attn_in"]
    ve_mult, ve_mean = jnp.ones((H,)), jnp.zeros((H, HD))
    qc, kc, *_ = forward.mla_qkv(attn, x_c, seg, tokens, ve_mult, ve_mean)
    qp, kp, *_ = forward.mla_qkv(attn, x_p, seg, tokens, ve_mult, ve_mean)
    rel_c = attn.rel_pos(x_c)
    rel_p = attn.rel_pos(x_p)
    qc, kc, qp, kp, rel_c, rel_p = forward.replicate((qc, kc, qp, kp, rel_c, rel_p))
    lay = clean["layers"][5]
    unembed = forward.unembed_direction(m)
    u_y = unembed.T[clean["labels"]] - jnp.mean(unembed, axis=1)
    w_o = attn.w_o.astype(jnp.float32).reshape(H, HD, -1)
    out = {}
    s = tokens.shape[1]
    scale = 1.0 / math.sqrt(HD)
    for name, (q, k, rel) in {
        "clean": (qc, kc, rel_c),
        "q_patched": (qp, kc, rel_p),
        "k_patched": (qc, kp, rel_c),
        "both": (qp, kp, rel_p),
    }.items():
        ind_sum = jnp.zeros((H,))
        ind_n = jnp.zeros(())
        dla = jnp.zeros((H,))
        for b in range(tokens.shape[0]):
            logits = jnp.einsum("thd,shd->hts", q[b].astype(jnp.float32), k[b].astype(jnp.float32)) * scale
            logits = logits + dense_rel_bias(rel[b : b + 1].astype(jnp.float32), s, s)[0]
            ok = (seg[b][:, None] == seg[b][None, :]) & (jnp.arange(s)[None, :] <= jnp.arange(s)[:, None])
            a = jax.nn.softmax(jnp.where(ok[None], logits, -jnp.inf), axis=-1)
            prev_tok = jnp.pad(tokens[b, :-1], (1, 0), constant_values=-1)
            induct = ok & (prev_tok[None, :] == tokens[b][:, None]) & (jnp.arange(s)[None, :] >= 1)
            has = jnp.any(induct, -1)
            ind_sum = ind_sum + jnp.einsum("hts,t->h", a * induct, has.astype(jnp.float32))
            ind_n = ind_n + jnp.sum(has)
            ve = lay["ve"][b].astype(jnp.float32)  # [S, H, hd]
            v_full = (lay["v_proj"][b] + lay["ve"][b]).astype(jnp.float32)
            y = jnp.einsum("hts,shd->thd", a, ve)
            y = (
                y
                - jnp.sum(y * v_full, -1, keepdims=True) / (jnp.sum(v_full * v_full, -1, keepdims=True) + 1e-6) * v_full
            )
            c = jnp.einsum("thd,hde->the", lay["head_gate"][b][..., None].astype(jnp.float32) * y, w_o)
            c = c * layer.sconv_attn.weight[0].astype(
                jnp.float32
            )  # lag-0 tap (the SConv's other taps read earlier tokens)
            c = c * (clean["w_final"][6][b][:, None, None] * clean["inv_rms"][b][:, None, None])
            dla = dla + jnp.einsum("the,te->h", c, u_y[b])
        out[f"{name}/induction"] = ind_sum
        out[f"{name}/n_induction"] = ind_n
        out[f"{name}/ve_dla"] = dla
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--natural-train", type=int, default=16)
    parser.add_argument("--natural-paloma", type=int, default=1)
    parser.add_argument("--ablation-batches", type=int, default=12)
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/attrib.npz", "rb") as f:
        attrib = np.load(io.BytesIO(f.read()))
    head_mean, sub_mean = jnp.asarray(attrib["head_mean"]), jnp.asarray(attrib["sub_mean"])
    freq_rank = attrib["freq_rank"]
    vocab = common.vocab_strings()
    feats = categories.token_features(vocab)
    natural = [
        *common.paloma_batches(BATCH, max_sequences=args.natural_paloma),
        *common.train_batches(BATCH, args.natural_train, seed=7),
    ]
    abl_batches = [*common.paloma_batches(BATCH, max_sequences=2), *common.train_batches(BATCH, 16, seed=8)]
    pick = np.linspace(0, len(abl_batches) - 1, args.ablation_batches).round().astype(int)
    abl_batches = [abl_batches[j] for j in pick]
    induction = [heads.induction_batch(BATCH, 200 + s) for s in range(2)]
    fit = list(common.train_batches(BATCH, 64, seed=1))
    counts = np.bincount(np.concatenate([b.tokens.reshape(-1) for b in fit]), minlength=forward.VOCAB)
    bigram = np.zeros((forward.VOCAB,), np.int64)
    pairs = np.concatenate([np.stack([b.tokens[:, :-1].reshape(-1), b.tokens[:, 1:].reshape(-1)], 1) for b in fit])
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
    best = {}
    for (a, b), c in zip(uniq, cnt, strict=True):
        if c > best.get(a, (0, -1))[0]:
            best[a] = (c, b)
    for a, (_, b) in best.items():
        bigram[a] = b
    results: dict[str, np.ndarray] = {}
    base = forward.no_ablation()
    with set_mesh(mesh):
        model = common.compute_model(params)
        trace_small, loss_fn, ve_natural, head_row, ve_ov = make_fns()
        kda_sink_fn, mla_doc_fn = jax.jit(kda_sink_stats), jax.jit(mla_doc_start)
        results.update(weight_analysis(model, vocab, bigram, counts.astype(np.float64)))
        logger.info("weights done (%.0fs)", time.time() - t0)
        # Mean gated VE per (layer, head) from the fit batches, for mean ablation.
        ve_sum = np.zeros((L, H, HD))
        n_tok = 0
        for b in fit[:8]:
            tokens, seg, w = common.device_batch(b, mesh)
            tr = trace_small(model, tokens, seg, base)
            for li in VE_LAYERS:
                ve_sum[li] += np.asarray(jnp.einsum("bshd,bs->hd", tr["layers"][li]["ve"].astype(jnp.float32), w))
            n_tok += float(np.asarray(w).sum())
        ve_mean = jnp.asarray(ve_sum / n_tok, jnp.float32)
        # Natural text: VE statistics, examples, failures, KDA sink, MLA doc-start shares.
        acc: dict[str, np.ndarray] = {}
        examples, failures = [], []
        sink: dict[str, np.ndarray] = {}
        for bi, b in enumerate(natural):
            tokens, seg, w = common.device_batch(b, mesh)
            cats = categories.category_masks(b.tokens, b.segment_ids, b.loss_weight, b.source, feats, freq_rank)
            tr = trace_small(model, tokens, seg, base)
            no_ve = base._replace(sub_mult=base.sub_mult.at[5, forward.SUB_VE].set(0.0))
            loss_nove = np.asarray(loss_fn(model, tokens, seg, no_ve))
            st = ve_natural(model, tr, tokens, jnp.asarray(cats))
            for k, v in st.items():
                if k.endswith("_tok"):
                    continue
                acc[k] = acc.get(k, 0.0) + np.asarray(v, np.float64)
            acc["n"] = acc.get("n", 0.0) + cats.sum(axis=(1, 2))
            dla_y = np.asarray(st["L5/dla_ve_y_tok"]).sum(-1)  # [B, S]
            dla_t = np.asarray(st["L5/dla_ve_top_tok"]).sum(-1)
            per_head = np.asarray(st["L5/dla_ve_y_tok"])
            loss = np.asarray(tr["loss"])
            labels = np.asarray(tr["labels"])
            top1 = np.asarray(tr["top_ids"])[..., 0]
            scored = cats[0] > 0
            cand = np.argwhere(scored & (dla_y > EXAMPLE_THRESHOLD))
            fail = np.argwhere(scored & (top1 != labels) & (dla_t > EXAMPLE_THRESHOLD) & (loss_nove < loss))
            acc["n_examples"] = acc.get("n_examples", 0.0) + len(cand)
            acc["n_failures"] = acc.get("n_failures", 0.0) + len(fail)
            acc["n_scored"] = acc.get("n_scored", 0.0) + scored.sum()
            rng = np.random.default_rng(bi)
            lay5 = tr["layers"][5]
            for kind, pool, store in (("example", cand, examples), ("failure", fail, failures)):
                if len(pool) == 0 or len(store) >= MAX_EXAMPLES:
                    continue
                for bb, t in pool[rng.choice(len(pool), size=min(3, len(pool)), replace=False)]:
                    h = int(
                        np.argmax(per_head[bb, t] if kind == "example" else np.asarray(st["L5/dla_ve_top_tok"])[bb, t])
                    )
                    p, prev, cur, bias = head_row(lay5, int(bb), int(t), h, seg[bb])
                    p = np.asarray(p)
                    keys = np.argsort(-p)[:TOP_KEYS]
                    top_i, _ = ve_ov(
                        model,
                        lay5["ve"],
                        lay5["head_gate"],
                        lay5["v_proj"] + lay5["ve"],
                        int(bb),
                        int(t),
                        h,
                        jnp.asarray(keys),
                        jnp.asarray(p[keys]),
                    )
                    row = b.tokens[bb]
                    store.append(
                        dict(
                            source=b.source,
                            left="".join(vocab[x] for x in row[max(0, t - EXAMPLE_LEFT) : t + 1]),
                            target=vocab[labels[bb, t]],
                            top1=vocab[top1[bb, t]],
                            head=h,
                            ve_dla=float(dla_y[bb, t] if kind == "example" else dla_t[bb, t]),
                            ve_dla_heads=per_head[bb, t].tolist(),
                            gate=float(np.asarray(lay5["ve_weight"])[bb, t, h]),
                            loss=float(loss[bb, t]),
                            loss_without_l5_ve=float(loss_nove[bb, t]),
                            keys=[
                                dict(
                                    pos=int(k),
                                    dist=int(t - k),
                                    prob=float(p[k]),
                                    token=vocab[row[k]],
                                    prev_token=vocab[row[k - 1]] if k > 0 else "",
                                    logit_offset=float(np.asarray(prev)[k]),
                                    logit_current=float(np.asarray(cur)[k]),
                                    logit_bias=float(np.asarray(bias)[k]),
                                    promotes=[vocab[int(x)] for x in np.asarray(top_i)[j]],
                                )
                                for j, k in enumerate(keys)
                            ],
                        )
                    )
            # KDA sink and MLA doc-start shares, per sequence.
            tok_r, seg_r, w_r = forward.replicate((tokens, seg, w))
            for bb in range(BATCH):
                if b.loss_weight[bb].sum() == 0:
                    continue
                for li, lay in enumerate(tr["layers"]):
                    if li in forward.KDA_LAYERS:
                        sub = {k: lay[k][bb] for k in ("q", "k", "v", "log_decay", "beta", "head_gate")}
                        stt = kda_sink_fn(sub, tok_r[bb], seg_r[bb], w_r[bb] > 0)
                    else:
                        sub = {k: lay[k][bb] for k in ("q", "k", "rel_bias")}
                        stt = mla_doc_fn(sub, seg_r[bb], w_r[bb] > 0)
                    for k, v in stt.items():
                        key = f"sink/L{li}/{k}"
                        sink[key] = sink.get(key, 0.0) + np.asarray(v, np.float64)
            logger.info(
                "natural batch %d/%d (%.0fs) examples %d failures %d",
                bi,
                len(natural),
                time.time() - t0,
                len(examples),
                len(failures),
            )
        results.update({f"nat/{k}": v for k, v in acc.items()})
        results.update(sink)
        # Ablations by category and on the copy task.
        configs = [("none", base)]
        for li in VE_LAYERS:
            configs.append((f"L{li}.ve:zero", base._replace(sub_mult=base.sub_mult.at[li, forward.SUB_VE].set(0.0))))
            configs.append(
                (
                    f"L{li}.ve:mean",
                    base._replace(sub_mult=base.sub_mult.at[li, forward.SUB_VE].set(0.0), ve_mean=ve_mean),
                )
            )
        both = base.sub_mult.at[3, forward.SUB_VE].set(0.0).at[5, forward.SUB_VE].set(0.0)
        configs.append(("L3+L5.ve:zero", base._replace(sub_mult=both)))
        for h in range(H):
            configs.append((f"L5.h{h}.ve:zero", base._replace(ve_head_mult=base.ve_head_mult.at[5, h].set(0.0))))
        for k in EARLY:
            for mode, name in ((1, "zero"), (2, "mean")):
                configs.append(
                    (
                        f"kda_early{k}:{name}",
                        base._replace(kda_early_tokens=jnp.asarray(k), kda_early_mode=jnp.asarray(mode)),
                    )
                )
        abl_cats = [
            categories.category_masks(b.tokens, b.segment_ids, b.loss_weight, b.source, feats, freq_rank)
            for b in abl_batches
        ]
        period = 2 * heads.INDUCTION_LEN
        pos = np.arange(period)
        reps = common.SEQ_LEN // period
        second = np.tile((pos >= heads.INDUCTION_LEN) & (pos < period - 1), reps)
        names, nat_loss, copy_loss = [], [], []
        n_cat = np.stack([c.sum(axis=(1, 2)) for c in abl_cats]).sum(0)
        for name, abl in configs:
            tot = np.zeros(len(categories.CATEGORIES))
            for b, c in zip(abl_batches, abl_cats, strict=True):
                tokens, seg, _ = common.device_batch(b, mesh)
                loss = np.asarray(loss_fn(model, tokens, seg, abl))
                tot += np.einsum("cbs,bs->c", c, loss)
            cl = 0.0
            for b in induction:
                tokens, seg, _ = common.device_batch(b, mesh)
                cl += float(np.asarray(loss_fn(model, tokens, seg, abl))[:, second].mean())
            names.append(name)
            nat_loss.append(tot / n_cat)
            copy_loss.append(cl / len(induction))
            logger.info(
                "ablation %s: natural %.4f copy %.4f (%.0fs)", name, nat_loss[-1][0], copy_loss[-1], time.time() - t0
            )
        results["abl/names"] = np.asarray(names)
        results["abl/natural"] = np.stack(nat_loss)
        results["abl/copy"] = np.asarray(copy_loss)
        # Path patching into L5 queries / keys.
        sources = [
            (f"L3.h{h}", base._replace(head_mult=base.head_mult.at[3, h].set(0.0), head_mean=head_mean))
            for h in range(H)
        ]
        sources += [
            (f"L{li}.attn", base._replace(sub_mult=base.sub_mult.at[li, forward.SUB_ATTN].set(0.0), sub_mean=sub_mean))
            for li in (0, 1, 2, 3, 4)
        ]
        sources += [
            (
                f"L{li}.mlp",
                base._replace(
                    sub_mult=base.sub_mult.at[li, forward.SUB_ROUTED].set(0.0).at[li, forward.SUB_SHARED].set(0.0),
                    sub_mean=sub_mean,
                ),
            )
            for li in (3, 4)
        ]
        patch_fn = jax.jit(patched_l5)
        patch = {}
        for kind, batches in (("induction", induction), ("natural", natural[:2])):
            clean_tr = [trace_small(model, *common.device_batch(b, mesh)[:2], base) for b in batches]
            for name, abl in sources:
                for b, ctr in zip(batches, clean_tr, strict=True):
                    tokens, seg, _ = common.device_batch(b, mesh)
                    ptr = trace_small(model, tokens, seg, abl)
                    tok_r, seg_r = forward.replicate((tokens, seg))
                    out = patch_fn(model, ctr, ptr, tok_r, seg_r)
                    for k, v in out.items():
                        key = f"patch/{kind}/{name}/{k}"
                        patch[key] = patch.get(key, 0.0) + np.asarray(v, np.float64)
                logger.info("patch %s %s (%.0fs)", kind, name, time.time() - t0)
        results.update(patch)
        results["patch/sources"] = np.asarray([n for n, _ in sources])
    results["examples_json"] = np.asarray(json.dumps(examples))
    results["failures_json"] = np.asarray(json.dumps(failures))
    results["categories"] = np.asarray(categories.CATEGORIES)
    results["pos_edges"] = np.asarray(POS_EDGES)
    common.save_npz(args.tag, "deepdive", **results)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
