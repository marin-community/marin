# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron, expert and routing statistics over a large token sample (unseen training-cache sequences plus Paloma).

Per layer, accumulated on the device over every batch:

- routed ReLU^2 hidden units ``[E, I]``: firing count (pre-activation > 0), activation sum / square sum / max, the
  top-``TOP_CONTEXTS`` activations with their global token position, and a log10 activation histogram; the same for
  the shared expert's units ``[I]``;
- experts: tokens routed per expert, per (expert, current token id) counts ``[E, V]``;
- routing: the combine weight histogram by weight rank, the effective number of experts ``exp(H(w / sum w))``,
  low-weight ("passenger") slots, slots selected only because of the QB bias (in the biased top-k, not the unbiased
  one), each slot's weighted output norm ``||w_k y_k W_up||``, and per-expert mean router logits.

Weights-only: each neuron's output direction through the MLP SConv's lag-0 tap and the final norm gain, projected on
the unembedding (top promoted / suppressed tokens). Writes ``stats.npz`` and the analysed tokens ``tokens.npz``.
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import common, forward
from experiments.grug.fast_track.model import _small_top_k

logger = logging.getLogger(__name__)

BATCH = 8
TOP_CONTEXTS = 16
LOG_BINS = 90  # log10(activation) in [-6, 3), 0.1 wide
LOG_LO = -6.0
WEIGHT_BINS = 125  # combine weight in [0, 2.5)
NORM_BINS = 80  # log10(weighted output norm) in [-4, 4)
EFF_BINS = 100  # effective experts in [1, 11)
LENS_TOP = 12
PASSENGER_FRACTIONS = (0.01, 0.02, 0.05)  # of the token's weight sum (uniform = 0.1)


def _log_hist(x: jax.Array, positive: jax.Array, bins: int, lo: float) -> jax.Array:
    idx = jnp.clip(jnp.floor((jnp.log10(jnp.maximum(x, 1e-30)) - lo) * 10).astype(jnp.int32), 0, bins - 1)
    return jnp.zeros((bins,), jnp.float32).at[idx.reshape(-1)].add(positive.reshape(-1).astype(jnp.float32))


def _lin_hist(x: jax.Array, bins: int, lo: float, hi: float, weight: jax.Array | None = None) -> jax.Array:
    idx = jnp.clip(jnp.floor((x - lo) / (hi - lo) * bins).astype(jnp.int32), 0, bins - 1)
    w = jnp.ones_like(x) if weight is None else weight.astype(jnp.float32)
    return jnp.zeros((bins,), jnp.float32).at[idx.reshape(-1)].add(w.reshape(-1))


def _merge_top(vals, ids, new_vals, new_ids):
    v = jnp.concatenate([vals, new_vals], axis=-1)
    i = jnp.concatenate([ids, new_ids], axis=-1)
    top, pick = jax.lax.top_k(v, TOP_CONTEXTS)
    return top, jnp.take_along_axis(i, pick, axis=-1)


def init_state() -> dict:
    L, E, I, V, K = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.VOCAB, forward.TOP_K
    z = jnp.zeros
    return dict(
        fire=z((L, E, I)),
        act_sum=z((L, E, I)),
        act_sq=z((L, E, I)),
        act_max=z((L, E, I)),
        top_val=jnp.full((L, E, I, TOP_CONTEXTS), -1.0),
        top_pos=jnp.zeros((L, E, I, TOP_CONTEXTS), jnp.int32),
        act_hist=z((L, LOG_BINS)),
        s_fire=z((L, I)),
        s_act_sum=z((L, I)),
        s_act_sq=z((L, I)),
        s_act_max=z((L, I)),
        s_top_val=jnp.full((L, I, TOP_CONTEXTS), -1.0),
        s_top_pos=jnp.zeros((L, I, TOP_CONTEXTS), jnp.int32),
        s_act_hist=z((L, LOG_BINS)),
        expert_tokens=z((L, E)),
        expert_token_ids=jnp.zeros((L, E, V), jnp.int32),
        token_counts=z((V,)),
        weight_hist=z((L, K, WEIGHT_BINS)),
        eff_hist=z((L, EFF_BINS)),
        norm_hist=z((L, K, NORM_BINS)),
        passengers=z((L, len(PASSENGER_FRACTIONS), E)),
        passenger_norm=z((L, len(PASSENGER_FRACTIONS), E)),
        bias_driven=z((L, E)),
        bias_driven_hist=z((L, 2, WEIGHT_BINS)),
        bias_driven_rank=z((L, K)),
        slot_weight_sum=z((L, E)),
        slot_norm_sum=z((L, E)),
        logit_sum=z((L, E)),
        selected_logit_sum=z((L, E)),
        frac_norm_sum=z((L, K)),
        routed_norm_sum=z((L,)),
        shared_norm_sum=z((L,)),
        attn_norm_sum=z((L,)),
        loss_sum=z(()),
        tokens=z(()),
    )


def make_step():
    @jax.jit
    def step(state, model, tokens, seg, weight, offset):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        flat_tok = tokens.reshape(-1)
        valid_tok = (weight.reshape(-1) > 0).astype(jnp.float32)
        new = dict(state)
        upd = {k: [] for k in state}
        for i, lay in enumerate(tr["layers"]):
            act = lay["expert_act"]  # [TK, I], assignments in expert order
            experts = lay["expert_sorted"]
            tok_sorted = lay["token_sorted"]
            ok = valid_tok[tok_sorted]
            okf = ok[:, None]
            pos = act > 0
            E = forward.NUM_EXPERTS
            upd["fire"].append(jax.ops.segment_sum(pos * okf, experts, num_segments=E))
            upd["act_sum"].append(jax.ops.segment_sum(act * okf, experts, num_segments=E))
            upd["act_sq"].append(jax.ops.segment_sum(jnp.square(act) * okf, experts, num_segments=E))
            upd["act_max"].append(jnp.maximum(jax.ops.segment_max(act * okf, experts, num_segments=E), 0.0))
            # Each expert's top activations per unit: sort every column by (expert, -activation), then read the first
            # TOP_CONTEXTS rows of each expert's segment.
            masked = jnp.where(okf > 0, act, -1.0)
            _, neg, toks = jax.lax.sort(
                (
                    jnp.broadcast_to(experts[:, None], act.shape),
                    -masked,
                    jnp.broadcast_to(tok_sorted[:, None], act.shape),
                ),
                dimension=0,
                num_keys=2,
            )
            j = jnp.arange(TOP_CONTEXTS)
            rows = jnp.minimum(lay["expert_starts"][:, None] + j[None, :], act.shape[0] - 1)  # [E, k]
            have = j[None, :] < lay["expert_counts"][:, None]
            tv = jnp.where(have[..., None], -neg[rows], -1.0)  # [E, k, I]
            tp = toks[rows]
            upd["top"] = [*upd.get("top", []), (jnp.swapaxes(tv, 1, 2), jnp.swapaxes(tp, 1, 2) + offset)]
            upd["act_hist"].append(_log_hist(act, pos & (okf > 0), LOG_BINS, LOG_LO))

            s_act = lay["shared_act"] * valid_tok[:, None]
            s_pos = (lay["shared_act"] > 0) & (valid_tok[:, None] > 0)
            upd["s_fire"].append(jnp.sum(s_pos, axis=0).astype(jnp.float32))
            upd["s_act_sum"].append(jnp.sum(s_act, axis=0))
            upd["s_act_sq"].append(jnp.sum(jnp.square(s_act), axis=0))
            upd["s_act_max"].append(jnp.max(s_act, axis=0))
            stv, sti = jax.lax.top_k(jnp.where(valid_tok[None, :] > 0, lay["shared_act"].T, -1.0), TOP_CONTEXTS)
            upd["s_top"] = [*upd.get("s_top", []), (stv, sti + offset)]
            upd["s_act_hist"].append(_log_hist(lay["shared_act"], s_pos, LOG_BINS, LOG_LO))

            sel = lay["selected"]  # [T, K]
            w = lay["combine"]  # [T, K]
            vt = valid_tok[:, None]
            onehot_counts = (
                jnp.zeros((forward.NUM_EXPERTS,), jnp.float32)
                .at[sel.reshape(-1)]
                .add(jnp.broadcast_to(vt, sel.shape).reshape(-1))
            )
            upd["expert_tokens"].append(onehot_counts)
            upd["expert_token_ids"].append(
                jnp.zeros((forward.NUM_EXPERTS, forward.VOCAB), jnp.int32)
                .at[sel.reshape(-1), jnp.repeat(flat_tok, forward.TOP_K)]
                .add(jnp.broadcast_to(vt, sel.shape).reshape(-1).astype(jnp.int32))
            )
            w_sorted = -jnp.sort(-w, axis=-1)
            upd["weight_hist"].append(
                jax.vmap(lambda col, wt: _lin_hist(col, WEIGHT_BINS, 0.0, 2.5, wt), in_axes=(1, None))(
                    w_sorted, valid_tok
                )
            )
            p = w / jnp.sum(w, axis=-1, keepdims=True)
            eff = jnp.exp(-jnp.sum(p * jnp.log(jnp.maximum(p, 1e-30)), axis=-1))
            upd["eff_hist"].append(_lin_hist(eff, EFF_BINS, 1.0, 11.0, valid_tok))
            # Weighted output norm of each slot in the residual stream (before the MLP SConv).
            slot_vec = jnp.einsum(
                "tko,od->tkd",
                (w[..., None] * lay["slot_latent_out"]).astype(model.output_proj.dtype),
                model.layers()[i].mlp.w_latent_up,
                preferred_element_type=jnp.float32,
            )
            norms = jnp.sqrt(jnp.sum(jnp.square(slot_vec), axis=-1))  # [T, K]
            routed_norm = jnp.sqrt(jnp.sum(jnp.square(jnp.sum(slot_vec, axis=1)), axis=-1))  # [T]
            rank = lay["weight_rank"]
            onehot_rank = jax.nn.one_hot(rank, forward.TOP_K)  # [T, K(slot), K(rank)]
            norm_by_rank = jnp.einsum("tk,tkr->tr", norms, onehot_rank)
            upd["norm_hist"].append(
                jax.vmap(
                    lambda col, wt: _log_hist(col, wt > 0, NORM_BINS, -4.0),
                    in_axes=(1, None),
                )(norm_by_rank, valid_tok)
            )
            upd["frac_norm_sum"].append(jnp.sum(norm_by_rank / jnp.maximum(routed_norm, 1e-9)[:, None] * vt, axis=0))
            flat_sel = sel.reshape(-1)
            vflat = jnp.broadcast_to(vt, sel.shape).reshape(-1)
            pas, pas_n = [], []
            for frac in PASSENGER_FRACTIONS:
                low = (p < frac).reshape(-1) * vflat
                pas.append(jnp.zeros((forward.NUM_EXPERTS,), jnp.float32).at[flat_sel].add(low))
                pas_n.append(jnp.zeros((forward.NUM_EXPERTS,), jnp.float32).at[flat_sel].add(low * norms.reshape(-1)))
            upd["passengers"].append(jnp.stack(pas))
            upd["passenger_norm"].append(jnp.stack(pas_n))
            logits = lay["router_logits"]
            _, unbiased_sel = _small_top_k(logits, forward.TOP_K)
            in_unbiased = jnp.any(sel[:, :, None] == unbiased_sel[:, None, :], axis=-1)  # [T, K]
            bias_only = (~in_unbiased).astype(jnp.float32) * vt
            upd["bias_driven"].append(
                jnp.zeros((forward.NUM_EXPERTS,), jnp.float32).at[flat_sel].add(bias_only.reshape(-1))
            )
            upd["bias_driven_hist"].append(
                jnp.stack(
                    [
                        _lin_hist(w, WEIGHT_BINS, 0.0, 2.5, (1.0 - bias_only / jnp.maximum(vt, 1e-9)) * vt),
                        _lin_hist(w, WEIGHT_BINS, 0.0, 2.5, bias_only),
                    ]
                )
            )
            upd["bias_driven_rank"].append(jnp.einsum("tk,tkr->r", bias_only, onehot_rank))
            upd["slot_weight_sum"].append(
                jnp.zeros((forward.NUM_EXPERTS,), jnp.float32).at[flat_sel].add((w * vt).reshape(-1))
            )
            upd["slot_norm_sum"].append(
                jnp.zeros((forward.NUM_EXPERTS,), jnp.float32).at[flat_sel].add((norms * vt).reshape(-1))
            )
            upd["logit_sum"].append(jnp.sum(logits * vt, axis=0))
            upd["selected_logit_sum"].append(
                jnp.zeros((forward.NUM_EXPERTS,), jnp.float32)
                .at[flat_sel]
                .add((jnp.take_along_axis(logits, sel, axis=-1) * vt).reshape(-1))
            )
            vb = valid_tok.reshape(tokens.shape)[..., None]

            def mean_norm(x, vb=vb):
                return jnp.sum(jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)) * vb)

            upd["routed_norm_sum"].append(mean_norm(lay["routed_out"]))
            upd["shared_norm_sum"].append(mean_norm(lay["shared_out"]))
            upd["attn_norm_sum"].append(mean_norm(lay["attn_out"]))
        for key in state:
            if key in ("top_val", "top_pos", "s_top_val", "s_top_pos", "act_max", "s_act_max", "loss_sum", "tokens"):
                continue
            if key == "token_counts":
                continue
            new[key] = state[key] + jnp.stack(upd[key])
        new["act_max"] = jnp.maximum(state["act_max"], jnp.stack(upd["act_max"]))
        new["s_act_max"] = jnp.maximum(state["s_act_max"], jnp.stack(upd["s_act_max"]))
        tv = jnp.stack([v for v, _ in upd["top"]])
        tp = jnp.stack([p_ for _, p_ in upd["top"]])
        new["top_val"], new["top_pos"] = _merge_top(state["top_val"], state["top_pos"], tv, tp)
        stv = jnp.stack([v for v, _ in upd["s_top"]])
        stp = jnp.stack([p_ for _, p_ in upd["s_top"]])
        new["s_top_val"], new["s_top_pos"] = _merge_top(state["s_top_val"], state["s_top_pos"], stv, stp)
        new["token_counts"] = state["token_counts"].at[flat_tok].add(valid_tok)
        new["loss_sum"] = state["loss_sum"] + jnp.sum(tr["loss"].reshape(-1) * valid_tok)
        new["tokens"] = state["tokens"] + jnp.sum(valid_tok)
        return new

    return step


def neuron_logit_lens(model) -> dict:
    """Each neuron's direct unembedding effect: its output direction (routed: ``w_down[e, i] @ w_latent_up``) times
    the MLP SConv's lag-0 tap and the final norm gain, on ``W_out``, centred over the vocabulary."""
    model = forward.replicate(model)
    unembed = forward.unembed_direction(model)  # [D, V]
    out = {}

    @jax.jit
    def project(dirs):
        logits = dirs @ unembed
        logits = logits - jnp.mean(logits, axis=-1, keepdims=True)
        top_v, top_i = jax.lax.top_k(logits, LENS_TOP)
        bot_v, bot_i = jax.lax.top_k(-logits, LENS_TOP)
        return top_v, top_i, -bot_v, bot_i, jnp.std(logits, axis=-1), jnp.sqrt(jnp.sum(jnp.square(dirs), axis=-1))

    routed, shared = [], []
    for layer in model.layers():
        lag0 = layer.sconv_mlp.weight[0].astype(jnp.float32)
        w_down = layer.mlp.expert_mlp.w_down.astype(jnp.float32)  # [E, I, R]
        up = layer.mlp.w_latent_up.astype(jnp.float32)  # [R, D]
        per_expert = [project(jnp.einsum("ir,rd->id", w_down[e], up) * lag0) for e in range(forward.NUM_EXPERTS)]
        routed.append([np.stack([np.asarray(x[j]) for x in per_expert]) for j in range(6)])
        sd = layer.shared[0].w_down.astype(jnp.float32) * lag0
        shared.append([np.asarray(x) for x in project(sd)])
    names = ("top_val", "top_id", "bot_val", "bot_id", "logit_std", "dir_norm")
    for j, name in enumerate(names):
        out[f"lens_{name}"] = np.stack([r[j] for r in routed])
        out[f"s_lens_{name}"] = np.stack([s[j] for s in shared])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-sequences", type=int, default=2048)
    parser.add_argument("--paloma-sequences", type=int, default=64, help="per Paloma set")
    args = parser.parse_args()
    checkpoint = args.checkpoint or common.CHECKPOINTS[args.tag]
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, checkpoint)
    t0 = time.time()
    with set_mesh(mesh):
        model = common.compute_model(params)
        state = init_state()
        step = make_step()
        sources, seq_tokens, seq_segs, seq_weights = [], [], [], []
        batches = [
            *common.paloma_batches(BATCH, max_sequences=args.paloma_sequences),
            *common.train_batches(BATCH, args.train_sequences),
        ]
        for n, batch in enumerate(batches):
            tokens, seg, w = common.device_batch(batch, mesh)
            offset = jnp.asarray(n * BATCH * common.SEQ_LEN, jnp.int32)
            state = step(state, model, tokens, seg, w, offset)
            seq_tokens.append(batch.tokens.astype(np.uint16))
            seq_segs.append(batch.segment_ids.astype(np.int32))
            seq_weights.append((batch.loss_weight > 0).astype(np.uint8))
            sources += [batch.source] * BATCH
            if n % 50 == 0:
                logger.info("batch %d/%d (%s) %.0fs", n, len(batches), batch.source, time.time() - t0)
        host = {k: np.asarray(v) for k, v in state.items()}
        host.update(neuron_logit_lens(model))
        host["router_bias"] = np.stack([np.asarray(layer.mlp.router_bias, np.float32) for layer in params.layers()])
    common.save_npz(args.tag, "stats", **host)
    common.save_npz(
        args.tag,
        "tokens",
        tokens=np.concatenate(seq_tokens),
        segment_ids=np.concatenate(seq_segs),
        valid=np.concatenate(seq_weights),
        sources=np.asarray(sources),
    )
    logger.info(
        "done in %.0fs, %d tokens, mean loss %.4f", time.time() - t0, host["tokens"], host["loss_sum"] / host["tokens"]
    )


if __name__ == "__main__":
    main()
