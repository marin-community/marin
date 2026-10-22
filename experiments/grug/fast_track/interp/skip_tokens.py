# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Which tokens need the second half (L3-L5) of the d512 model: per-token losses of early exits against the full model.

An exit after stage ``s`` reads the final AttnRes gate's query over the sources available at ``s`` (``attrib`` stages
``block2``, ``attn3``, ``block3``, ``block4``), RMS-normed, through one of three readouts:

- ``tuned``: the ridge-regression affine translator to the final hidden state (as ``attrib``), frozen lm_head;
- ``aff``: an affine map (initialized from ``tuned``) trained on next-token cross entropy, frozen lm_head;
- ``full``: an affine map plus a trainable lm_head and vocabulary bias (initialized from the model's), trained on CE.

The heads train on unseen training-cache sequences (seed 3); the evaluation uses others (seed 2) and a slice of every
Paloma set. Per evaluated token it records the full model's loss, entropy and max prob; each exit's loss, KL(full ||
exit), top-1 agreement, entropy and max prob; the direct logit attribution of every L3-L5 term (and the first half's
sum); the loss under mean ablations of second-half sublayers and unit groups; and a first-half proxy of the L3 router
(L3's router on the MLP gate's mix of ``[embed, L0, L1, L2]``) with its top-10 overlap with the real selection. A
subset of sequences also keeps the block2 stream and the proxy router logits for probe training.

Writes ``skip_tokens.npz`` (per token), ``skip_probe.npz`` (probe subset) and ``skip_heads.npz`` (head training curve).
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import attrib, common, forward, ngram

logger = logging.getLogger(__name__)

EPS = 1e-5
STAGES = ("block2", "attn3", "block3", "block4")
STAGE_INDEX = tuple(attrib.STAGES.index(s) for s in STAGES)
HEADS = ("tuned", "aff", "full")
TRAINED = ("aff", "full")
SECOND_HALF_TERMS = tuple(n for n in attrib.TERMS if n.split(".")[0] in ("L3", "L4", "L5"))
FIRST_HALF_TERMS = tuple(n for n in attrib.TERMS if n not in SECOND_HALF_TERMS and not n.endswith(".ve"))
UNIT_LAYERS = (3, 4, 5)
FIT_BATCH = 8
EVAL_BATCH = 4


def stage_features(model_r, tr) -> list[jax.Array]:
    """RMS-normed final-gate mixes at each exit stage, float32 ``[B, S, D]``."""
    sources = attrib._stage_sources(tr)
    out = []
    for k in STAGE_INDEX:
        _, mixed = forward._gate(
            sources[k], model_r.attn_res_query_final, EPS, plain_sum=model_r.config.attn_res_plain_sum
        )
        out.append(forward.rms_norm(mixed.astype(jnp.float32), EPS))
    return out


def _affine(x: jax.Array, a: jax.Array) -> jax.Array:
    return x @ a[:-1] + a[-1]


def head_logp(kind: str, k: int, x: jax.Array, params: dict, tuned: jax.Array, out_proj: jax.Array) -> jax.Array:
    """Exit ``kind``'s next-token log-probs at stage ``k`` from the normed mix ``x``."""
    if kind == "full":
        h = _affine(x, params["full_a"][k]).astype(jnp.bfloat16)
        z = jnp.einsum("bsd,dv->bsv", h, params["full_w"][k].astype(jnp.bfloat16), preferred_element_type=jnp.float32)
        z = z + params["full_b"][k]
    else:
        a = tuned[k] if kind == "tuned" else params["aff"][k]
        h = _affine(x, a).astype(jnp.bfloat16)
        z = jnp.einsum("bsd,dv->bsv", h, out_proj, preferred_element_type=jnp.float32)
    return jax.nn.log_softmax(forward.soft_cap(z), axis=-1)


def scored_weight(tokens: jax.Array, seg: jax.Array, weight: jax.Array) -> jax.Array:
    same_doc = jnp.pad(seg[:, 1:] == seg[:, :-1], ((0, 0), (0, 1)))
    return weight * same_doc


def make_fns():
    def run(model, tokens, seg, abl):
        model_r = forward.replicate(model)
        return model_r, forward.replicate(forward.trace_forward(model_r, tokens, seg, abl))

    @jax.jit
    def features(model, tokens, seg):
        model_r, tr = run(model, tokens, seg, forward.no_ablation())
        return jnp.stack(stage_features(model_r, tr)), tr["labels"]

    def train_loss(params, feats, labels, w, out_proj):
        losses = []
        for k in range(len(STAGES)):
            for kind in TRAINED:
                logp = head_logp(kind, k, feats[k], params, None, out_proj)
                ce = -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
                losses.append(jnp.sum(ce * w) / jnp.maximum(jnp.sum(w), 1.0))
        losses = jnp.stack(losses)
        return jnp.sum(losses), losses

    @jax.jit
    def ablated_loss(model, tokens, seg, abl):
        _, tr = run(model, tokens, seg, abl)
        return tr["loss"]

    @jax.jit
    def evaluate(model, tokens, seg, tuned, params):
        model_r, tr = run(model, tokens, seg, forward.no_ablation())
        out_proj = model_r.output_proj
        labels = tr["labels"]
        full_logp = jax.nn.log_softmax(forward.soft_cap(forward._final_logits(model_r, tr["hidden"])), axis=-1)
        p_full = jnp.exp(full_logp)
        top1 = jnp.argmax(full_logp, axis=-1)
        res = dict(
            loss_full=tr["loss"],
            entropy_full=tr["entropy"],
            maxp_full=jnp.max(p_full, axis=-1),
            top1_full=top1,
            correct_full=(top1 == labels),
        )
        feats = stage_features(model_r, tr)
        fields = {f: [] for f in ("loss", "kl", "agree", "entropy", "maxp", "correct")}
        for k in range(len(STAGES)):
            for kind in HEADS:
                logp = head_logp(kind, k, feats[k], params, tuned, out_proj)
                p = jnp.exp(logp)
                exit_top1 = jnp.argmax(logp, axis=-1)
                fields["loss"].append(-jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0])
                fields["kl"].append(jnp.sum(p_full * (full_logp - logp), axis=-1))
                fields["agree"].append(exit_top1 == top1)
                fields["entropy"].append(-jnp.sum(p * logp, axis=-1))
                fields["maxp"].append(jnp.max(p, axis=-1))
                fields["correct"].append(exit_top1 == labels)
        for f, vals in fields.items():
            res[f"exit_{f}"] = jnp.stack(vals).reshape(len(STAGES), len(HEADS), *labels.shape)
        # Direct logit attribution of the target (centred over the vocabulary), as attrib.
        unembed = forward.unembed_direction(model_r)
        mean_dir = jnp.mean(unembed, axis=1)
        u_y = (unembed.T[labels] - mean_dir).astype(jnp.float32)
        inv = tr["inv_rms"][..., None]
        wf = tr["w_final"]

        def dla(vec, src):
            return jnp.sum(vec.astype(jnp.float32) * wf[src][..., None] * inv * u_y, axis=-1)

        terms = {"embed": dla(tr["embed"], 0)}
        for i, lay in enumerate(tr["layers"]):
            for h in range(forward.NUM_HEADS):
                terms[f"L{i}.h{h}"] = dla(lay["head_out"][:, :, h], i + 1)
            terms[f"L{i}.ve"] = dla(lay["ve_out"], i + 1)
            terms[f"L{i}.shared"] = dla(lay["shared_out"], i + 1)
            terms[f"L{i}.routed"] = dla(lay["routed_out"], i + 1)
        first = sum(terms[n] for n in FIRST_HALF_TERMS)
        res["dla"] = jnp.stack([first, *[terms[n] for n in SECOND_HALF_TERMS]])
        # First-half proxy of the L3 router: L3's MLP gate without the L3 attention source.
        layer3 = list(model_r.layers())[3]
        _, h2 = forward._gate(
            tr["blocks"][:4], layer3.attn_res_query_mlp, EPS, plain_sum=model_r.config.attn_res_plain_sum
        )
        proxy = jnp.einsum("bsd,de->bse", layer3._mlp_input(h2), layer3.mlp.router).astype(jnp.float32)
        bias = layer3.mlp.router_bias.astype(jnp.float32)
        _, proxy_sel = jax.lax.top_k(proxy + bias, forward.TOP_K)
        real_sel = tr["layers"][3]["selected"].reshape(*labels.shape, forward.TOP_K)
        overlap = jnp.sum(proxy_sel[..., :, None] == real_sel[..., None, :], axis=(-1, -2))
        res["router3_overlap"] = overlap
        res["router3_proxy_max"] = jnp.max(proxy, axis=-1)
        res["router3_proxy_entropy"] = -jnp.sum(jax.nn.softmax(proxy, -1) * jax.nn.log_softmax(proxy, -1), axis=-1)
        res["probe_x"] = feats[0].astype(jnp.bfloat16)
        res["probe_router3"] = proxy.astype(jnp.bfloat16)
        return res

    return features, train_loss, ablated_loss, evaluate


def init_heads(tuned: np.ndarray, out_proj: jax.Array) -> dict:
    n = len(STAGES)
    w = jnp.asarray(out_proj, jnp.float32)
    return dict(
        aff=jnp.asarray(tuned),
        full_a=jnp.asarray(tuned),
        full_w=jnp.broadcast_to(w, (n, *w.shape)),
        full_b=jnp.zeros((n, w.shape[1]), jnp.float32),
    )


def optimizer(steps: int, lr_affine: float, lr_unembed: float):
    def sched(lr):
        return optax.warmup_cosine_decay_schedule(0.0, lr, min(50, steps // 10), steps, end_value=0.0)

    labels = dict(aff="a", full_a="a", full_w="w", full_b="w")
    return optax.multi_transform({"a": optax.adam(sched(lr_affine)), "w": optax.adam(sched(lr_unembed))}, labels)


def ablation_set(head_mean, sub_mean, routed_mean, masks: dict | None) -> list[tuple[str, forward.Ablation]]:
    base = forward.no_ablation()
    out = []

    def subs(name, pairs):
        mult = base.sub_mult
        for i, s in pairs:
            mult = mult.at[i, s].set(0.0)
        out.append((name, base._replace(sub_mult=mult, sub_mean=sub_mean)))

    moe = (forward.SUB_SHARED, forward.SUB_ROUTED)
    for i in UNIT_LAYERS:
        subs(f"attn:L{i}", [(i, forward.SUB_ATTN)])
        subs(f"moe:L{i}", [(i, s) for s in moe])
    subs("attn:L345", [(i, forward.SUB_ATTN) for i in UNIT_LAYERS])
    subs("moe:L345", [(i, s) for i in UNIT_LAYERS for s in moe])
    subs("layer:L345", [(i, s) for i in UNIT_LAYERS for s in (forward.SUB_ATTN, *moe)])
    out.append(("ve:L5:zero", base._replace(sub_mult=base.sub_mult.at[5, forward.SUB_VE].set(0.0))))
    out.append(("head:L3.h0", base._replace(head_mult=base.head_mult.at[3, 0].set(0.0), head_mean=head_mean)))
    if masks is None:
        return out
    layer_ok = np.zeros((forward.NUM_LAYERS, 1, 1), bool)
    layer_ok[list(UNIT_LAYERS)] = True
    for group in ("ngram", "output"):
        mult = jnp.asarray(np.where(masks[group] & layer_ok, 0.0, 1.0), jnp.float32)
        out.append((f"units:{group}", base._replace(neuron_mult=mult, neuron_mean=routed_mean)))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="tn-d512-ckpt1-rno")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fit-sequences", type=int, default=128, help="train sequences for the tuned lens and means")
    parser.add_argument("--head-steps", type=int, default=1500)
    parser.add_argument("--head-batch", type=int, default=4)
    parser.add_argument("--lr-affine", type=float, default=1e-3)
    parser.add_argument("--lr-unembed", type=float, default=3e-4)
    parser.add_argument("--eval-train-sequences", type=int, default=256)
    parser.add_argument("--eval-paloma-sequences", type=int, default=4, help="per Paloma set")
    parser.add_argument("--probe-sequences", type=int, default=48, help="train eval sequences kept for probes")
    parser.add_argument(
        "--masks-tag", default="tn-d512-ckpt1-rno", help="tag holding skip_unit_masks ('': no unit ablations)"
    )
    args = parser.parse_args()
    t0 = time.time()
    checkpoint = args.checkpoint or common.CHECKPOINTS[args.tag]
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, checkpoint)
    masks = ngram.load_npz(args.masks_tag, "skip_unit_masks") if args.masks_tag else None
    stats = ngram.load_npz(args.tag, "stats")
    routed_mean = jnp.asarray(stats["act_sum"] / np.maximum(stats["expert_tokens"][..., None], 1.0), jnp.float32)
    fit = list(common.train_batches(FIT_BATCH, args.fit_sequences, seed=1))
    head_data = list(common.train_batches(args.head_batch, args.head_steps * args.head_batch, seed=3))
    evals = [
        *common.train_batches(EVAL_BATCH, args.eval_train_sequences, seed=2),
        *common.paloma_batches(EVAL_BATCH, max_sequences=args.eval_paloma_sequences),
    ]
    logger.info("data loaded (%.0fs)", time.time() - t0)
    with set_mesh(mesh):
        model = common.compute_model(params)
        _, means_fn, tuned_fn, _ = attrib.make_fns()
        features_fn, train_loss, ablated_loss, evaluate = make_fns()
        head_sum = sub_sum = xtx = xty = None
        n_sum = 0.0
        for b in fit:
            tokens, seg, w = common.device_batch(b, mesh)
            h, s, n = means_fn(model, tokens, seg, w)
            a, c = tuned_fn(model, tokens, seg, w)
            head_sum = h if head_sum is None else head_sum + h
            sub_sum = s if sub_sum is None else sub_sum + s
            xtx = a if xtx is None else xtx + a
            xty = c if xty is None else xty + c
            n_sum += float(n)
        head_mean, sub_mean = head_sum / n_sum, sub_sum / n_sum
        xtx_h, xty_h = np.asarray(xtx, np.float64), np.asarray(xty, np.float64)
        tuned = np.stack(
            [
                np.linalg.solve(xtx_h[k] + attrib.TUNED_RIDGE * n_sum * np.eye(xtx_h.shape[1]), xty_h[k])
                for k in STAGE_INDEX
            ]
        ).astype(np.float32)
        logger.info("tuned lens fit on %d tokens (%.0fs)", n_sum, time.time() - t0)

        out_proj = forward.replicate(model).output_proj
        heads = init_heads(tuned, out_proj)
        opt = optimizer(args.head_steps, args.lr_affine, args.lr_unembed)
        opt_state = opt.init(heads)

        @jax.jit
        def step(heads, opt_state, model, tokens, seg, w):
            feats, labels = features_fn(model, tokens, seg)
            feats = jax.lax.stop_gradient(feats)
            ws = scored_weight(tokens, seg, w)
            (_, losses), grads = jax.value_and_grad(train_loss, has_aux=True)(heads, feats, labels, ws, out_proj)
            updates, opt_state = opt.update(grads, opt_state, heads)
            return optax.apply_updates(heads, updates), opt_state, losses

        curve = []
        for i, b in enumerate(head_data):
            tokens, seg, w = common.device_batch(b, mesh)
            heads, opt_state, losses = step(heads, opt_state, model, tokens, seg, w)
            if i % 50 == 0 or i == len(head_data) - 1:
                losses = np.asarray(losses)
                curve.append(np.concatenate([[i], losses]))
                logger.info("head step %d losses %s (%.0fs)", i, np.round(losses, 3).tolist(), time.time() - t0)
        common.save_npz(args.tag, "skip_heads", curve=np.stack(curve), tuned=tuned)

        abls = ablation_set(head_mean, sub_mean, routed_mean, masks)
        tuned_dev = jnp.asarray(tuned)
        per_token: dict[str, list[np.ndarray]] = {}
        probe: dict[str, list[np.ndarray]] = {"x": [], "router3": []}
        sources = []
        for j, b in enumerate(evals):
            tokens, seg, w = common.device_batch(b, mesh)
            res = evaluate(model, tokens, seg, tuned_dev, heads)
            px, pr = res.pop("probe_x"), res.pop("probe_router3")
            if j * EVAL_BATCH < args.probe_sequences:
                probe["x"].append(np.asarray(px.astype(jnp.float16)))
                probe["router3"].append(np.asarray(pr.astype(jnp.float16)))
            for key, v in res.items():
                v = np.asarray(v)
                v = v.astype(np.float16) if v.dtype.kind == "f" else v
                per_token.setdefault(key, []).append(v)
            abl_losses = [np.asarray(ablated_loss(model, tokens, seg, abl), np.float16) for _, abl in abls]
            per_token.setdefault("abl_loss", []).append(np.stack(abl_losses))
            per_token.setdefault("tokens", []).append(b.tokens)
            per_token.setdefault("segment_ids", []).append(b.segment_ids)
            per_token.setdefault("weight", []).append(np.asarray(scored_weight(tokens, seg, w)) > 0)
            sources += [b.source] * b.tokens.shape[0]
            if j % 10 == 0:
                logger.info("eval batch %d/%d (%.0fs)", j, len(evals), time.time() - t0)
        # Batch axis: leading for [B, S] fields, after the stage/head (or term/ablation) axes otherwise.
        arrays = {}
        for key, vals in per_token.items():
            axis = vals[0].ndim - 2
            arrays[key] = np.concatenate(vals, axis=axis)
        arrays["sources"] = np.asarray(sources)
        arrays["stages"] = np.asarray(STAGES)
        arrays["heads"] = np.asarray(HEADS)
        arrays["dla_terms"] = np.asarray(["first_half", *SECOND_HALF_TERMS])
        arrays["ablations"] = np.asarray([name for name, _ in abls])
        common.save_npz(args.tag, "skip_tokens", **arrays)
        common.save_npz(args.tag, "skip_probe", **{k: np.concatenate(v) for k, v in probe.items()})
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
