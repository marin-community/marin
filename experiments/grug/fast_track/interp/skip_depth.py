# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Depth comparison for the second-half skip study, on any Ckpt 1 ladder rung through the model's own forward.

An exit at layer ``k`` runs the model with ``LayerSkip`` dropping layers ``k .. L-1`` for every token: the final
AttnRes gate then mixes only the earlier sources (a zero source scales the mix uniformly, which the final RMSNorm
removes, so this is the final gate's lens over ``[embed, L0, ..., L(k-1)]``). Each exit gets a trained head (an affine
map on the normed hidden state plus a trainable lm_head and vocabulary bias, initialized to the identity and the
model's lm_head, trained on next-token CE as ``skip_tokens``' ``full`` head). Per evaluated token: the full model's
loss, entropy and max prob, and per exit the lens loss, the head's loss, KL(full || head), top-1 agreement, entropy
and max prob.

Writes ``skip_depth.npz``.
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.interp import common, forward, skip_tokens
from experiments.grug.fast_track.model import LayerSkip

logger = logging.getLogger(__name__)

EVAL_BATCH = 4


def exit_hiddens(model, tokens, seg, exits: tuple[int, ...]) -> tuple[jax.Array, list[jax.Array]]:
    """The full model's final hidden state and, per exit ``k``, the one with layers ``k ..`` skipped."""
    mask = AttentionMask.causal().with_segment_ids(seg)
    full, _ = model(tokens, mask=mask)
    layers = model.config.num_layers
    outs = []
    for k in exits:
        skip = LayerSkip(tuple(range(k, layers)), jnp.zeros(tokens.shape, bool))
        h, _ = model(tokens, mask=mask, layer_skip=skip)
        outs.append(h)
    return full, outs


def head_logp(params: dict, k: int, h: jax.Array) -> jax.Array:
    x = h.astype(jnp.float32)
    z = (x @ params["a"][k] + params["c"][k]).astype(jnp.bfloat16)
    logits = jnp.einsum("bsd,dv->bsv", z, params["w"][k].astype(jnp.bfloat16), preferred_element_type=jnp.float32)
    return jax.nn.log_softmax(forward.soft_cap(logits + params["b"][k]), axis=-1)


def lens_logp(model, h: jax.Array) -> jax.Array:
    logits = jnp.einsum("bsd,dv->bsv", h, model.output_proj, preferred_element_type=jnp.float32)
    return jax.nn.log_softmax(forward.soft_cap(logits), axis=-1)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--exits", default=None, help="comma-separated exit layers (default: L/2 and 3L/4)")
    parser.add_argument("--head-steps", type=int, default=3000)
    parser.add_argument("--head-batch", type=int, default=4)
    parser.add_argument("--lr-affine", type=float, default=1e-3)
    parser.add_argument("--lr-unembed", type=float, default=3e-4)
    parser.add_argument("--eval-sequences", type=int, default=256)
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config(args.tag)
    layers = cfg.num_layers
    exits = tuple(int(x) for x in args.exits.split(",")) if args.exits else (layers // 2, (3 * layers) // 4)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    head_data = list(common.train_batches(args.head_batch, args.head_steps * args.head_batch, seed=3))
    evals = list(common.train_batches(EVAL_BATCH, args.eval_sequences, seed=2))
    logger.info("%s: %d layers, exits %s; data loaded (%.0fs)", args.tag, layers, exits, time.time() - t0)
    with set_mesh(mesh):
        model = common.compute_model(params)
        d = cfg.hidden_dim
        w = jnp.asarray(model.output_proj, jnp.float32)
        n = len(exits)
        heads = dict(
            a=jnp.broadcast_to(jnp.eye(d, dtype=jnp.float32), (n, d, d)),
            c=jnp.zeros((n, d), jnp.float32),
            w=jnp.broadcast_to(w, (n, *w.shape)),
            b=jnp.zeros((n, w.shape[1]), jnp.float32),
        )
        labels_map = dict(a="a", c="a", w="w", b="w")
        sched = [
            optax.warmup_cosine_decay_schedule(0.0, lr, min(50, args.head_steps // 10), args.head_steps)
            for lr in (args.lr_affine, args.lr_unembed)
        ]
        opt = optax.multi_transform({"a": optax.adam(sched[0]), "w": optax.adam(sched[1])}, labels_map)
        opt_state = opt.init(heads)

        @jax.jit
        def step(heads, opt_state, model, tokens, seg, wt):
            _, hs = exit_hiddens(model, tokens, seg, exits)
            hs = [jax.lax.stop_gradient(h) for h in hs]
            labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
            ws = skip_tokens.scored_weight(tokens, seg, wt)

            def loss(p):
                out = []
                for k, h in enumerate(hs):
                    ce = -jnp.take_along_axis(head_logp(p, k, h), labels[..., None], axis=-1)[..., 0]
                    out.append(jnp.sum(ce * ws) / jnp.maximum(jnp.sum(ws), 1.0))
                out = jnp.stack(out)
                return jnp.sum(out), out

            (_, losses), grads = jax.value_and_grad(loss, has_aux=True)(heads)
            updates, opt_state = opt.update(grads, opt_state, heads)
            return optax.apply_updates(heads, updates), opt_state, losses

        @jax.jit
        def evaluate(model, heads, tokens, seg):
            full, hs = exit_hiddens(model, tokens, seg, exits)
            labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
            full_logp = lens_logp(model, full)
            p_full = jnp.exp(full_logp)
            top1 = jnp.argmax(full_logp, axis=-1)

            def nll(logp):
                return -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]

            res = dict(
                loss_full=nll(full_logp),
                entropy_full=-jnp.sum(p_full * full_logp, axis=-1),
                maxp_full=jnp.max(p_full, axis=-1),
            )
            fields = {f: [] for f in ("lens_loss", "loss", "kl", "agree", "entropy", "maxp")}
            for k, h in enumerate(hs):
                logp = head_logp(heads, k, h)
                p = jnp.exp(logp)
                fields["lens_loss"].append(nll(lens_logp(model, h)))
                fields["loss"].append(nll(logp))
                fields["kl"].append(jnp.sum(p_full * (full_logp - logp), axis=-1))
                fields["agree"].append(jnp.argmax(logp, axis=-1) == top1)
                fields["entropy"].append(-jnp.sum(p * logp, axis=-1))
                fields["maxp"].append(jnp.max(p, axis=-1))
            res.update({f"exit_{f}": jnp.stack(v) for f, v in fields.items()})
            return res

        curve = []
        for i, b in enumerate(head_data):
            tokens, seg, wt = common.device_batch(b, mesh)
            heads, opt_state, losses = step(heads, opt_state, model, tokens, seg, wt)
            if i % 100 == 0 or i == len(head_data) - 1:
                curve.append(np.concatenate([[i], np.asarray(losses)]))
                logger.info("head step %d losses %s (%.0fs)", i, np.round(curve[-1][1:], 3).tolist(), time.time() - t0)
        per_token: dict[str, list[np.ndarray]] = {}
        for b in evals:
            tokens, seg, wt = common.device_batch(b, mesh)
            for key, v in evaluate(model, heads, tokens, seg).items():
                v = np.asarray(v)
                per_token.setdefault(key, []).append(v.astype(np.float16) if v.dtype.kind == "f" else v)
            per_token.setdefault("tokens", []).append(b.tokens)
            per_token.setdefault("segment_ids", []).append(b.segment_ids)
            per_token.setdefault("weight", []).append(np.asarray(skip_tokens.scored_weight(tokens, seg, wt)) > 0)
        arrays = {k: np.concatenate(v, axis=v[0].ndim - 2) for k, v in per_token.items()}
        w_all = arrays["weight"]
        logger.info(
            "full loss %.4f; exit head losses %s",
            float(arrays["loss_full"][w_all].astype(np.float32).mean()),
            [float(arrays["exit_loss"][k][w_all].astype(np.float32).mean()) for k in range(n)],
        )
        common.save_npz(
            args.tag,
            "skip_depth",
            **arrays,
            exits=np.asarray(exits),
            num_layers=np.asarray(layers),
            curve=np.stack(curve),
        )
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
