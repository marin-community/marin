# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How predictions are made, per next-token category (``categories.CATEGORIES``):

- direct logit attribution: the final hidden state is the final AttnRes gate's mix of ``[embed, L0, ..., L5]`` and each
  layer block is ``sum_h head_h + sconv(shared) + sconv(routed)``, so with the final RMSNorm's per-token scale held
  fixed the (pre-cap) logits split exactly into terms (``TERMS``); each term's logit for the target, centred over the
  vocabulary;
- lenses at every sublayer boundary: the final gate's query over the sources available so far (``gate`` lens), and a
  ridge-regression affine translator from that mix to the final hidden state (``tuned`` lens);
- AttnRes source weights of every gate, routing concentration (effective experts, top weight);
- ablations: every head (zero / mean), every sublayer (zero / mean), the MLA value embeddings, and dropping each token's
  lowest-weight routed experts; the change in loss per category.

Writes ``attrib.npz``.
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import categories, common, forward

logger = logging.getLogger(__name__)

BATCH = 8
NC = len(categories.CATEGORIES)
L, H = forward.NUM_LAYERS, forward.NUM_HEADS
# Stages of the lenses: the embedding, then after each attention sublayer and each layer block.
STAGES = ("embed", *[f"{kind}{i}" for i in range(L) for kind in ("attn", "block")])
NUM_GATES = 2 * L + 1
MAX_SOURCES = L + 1
TUNED_RIDGE = 1e-2


def term_names() -> list[str]:
    names = ["embed"]
    for i in range(L):
        names += [f"L{i}.h{h}" for h in range(H)] + [f"L{i}.ve", f"L{i}.shared", f"L{i}.routed"]
    return names


TERMS = term_names()


def _stage_sources(tr) -> list[list]:
    """The final gate's candidate sources at each stage."""
    blocks = tr["blocks"]
    stages = [[blocks[0]]]
    for i, lay in enumerate(tr["layers"]):
        stages.append([*blocks[: i + 1], lay["attn_out"]])
        stages.append(blocks[: i + 2])
    return stages


def _labels_logp(model, hidden):
    logits = forward.soft_cap(jnp.einsum("bsd,dv->bsv", hidden, model.output_proj, preferred_element_type=jnp.float32))
    return jax.nn.log_softmax(logits, axis=-1)


def make_fns():
    eps = 1e-5

    def run(model, tokens, seg, abl):
        return forward.trace_forward(forward.replicate(model), tokens, seg, abl)

    @jax.jit
    def loss_by_category(model, tokens, seg, cats, abl):
        tr = run(model, tokens, seg, abl)
        return jnp.einsum("cbs,bs->c", cats, tr["loss"]), jnp.sum(cats, axis=(1, 2))

    @jax.jit
    def means(model, tokens, seg, weight):
        tr = run(model, tokens, seg, forward.no_ablation())
        w = weight[..., None]
        n = jnp.sum(weight)
        head = jnp.stack(
            [jnp.einsum("bshd,bs->hd", lay["head_contrib"].astype(jnp.float32), weight) for lay in tr["layers"]]
        )
        sub = jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.sum(lay["attn_out"].astype(jnp.float32) * w, axis=(0, 1)),
                        jnp.sum(lay["shared_pre_conv"].astype(jnp.float32) * w, axis=(0, 1)),
                        jnp.sum(lay["routed_pre_conv"].astype(jnp.float32) * w, axis=(0, 1)),
                        jnp.zeros((forward.HIDDEN,)),
                    ]
                )
                for lay in tr["layers"]
            ]
        )
        return head, sub, n

    @jax.jit
    def tuned_fit_stats(model, tokens, seg, weight):
        tr = run(model, tokens, seg, forward.no_ablation())
        final_q = forward.replicate(model).attn_res_query_final
        target = tr["hidden"].astype(jnp.float32).reshape(-1, forward.HIDDEN)
        w = weight.reshape(-1, 1)
        xtx, xty = [], []
        for srcs in _stage_sources(tr):
            _, mixed = forward._gate(srcs, final_q, eps)
            x = jnp.concatenate(
                [forward.rms_norm(mixed.astype(jnp.float32), eps).reshape(-1, forward.HIDDEN), jnp.ones_like(w)], axis=1
            )
            xtx.append((x * w).T @ x)
            xty.append((x * w).T @ target)
        return jnp.stack(xtx), jnp.stack(xty)

    @jax.jit
    def analyse(model, tokens, seg, cats, tuned):
        model_r = forward.replicate(model)
        tr = forward.trace_forward(model_r, tokens, seg, forward.no_ablation())
        labels = tr["labels"]
        top1 = tr["top_ids"][..., 0]
        unembed = forward.unembed_direction(model_r)  # [D, V]
        mean_dir = jnp.mean(unembed, axis=1)
        u_y = (unembed.T[labels] - mean_dir).astype(jnp.float32)  # [B, S, D]
        u_top = (unembed.T[top1] - mean_dir).astype(jnp.float32)
        inv = tr["inv_rms"][..., None]
        wf = tr["w_final"]

        def dla(vec, src):
            v = vec.astype(jnp.float32) * wf[src][..., None] * inv
            return jnp.sum(v * u_y, axis=-1), jnp.sum(v * u_top, axis=-1)

        terms = [dla(tr["embed"], 0)]
        for i, lay in enumerate(tr["layers"]):
            for h in range(H):
                terms.append(dla(lay["head_out"][:, :, h], i + 1))
            terms.append(dla(lay["ve_out"], i + 1))
            terms.append(dla(lay["shared_out"], i + 1))
            terms.append(dla(lay["routed_out"], i + 1))
        dla_y = jnp.stack([t[0] for t in terms])  # [NT, B, S]
        dla_top = jnp.stack([t[1] for t in terms])
        is_ve = jnp.asarray([n.endswith(".ve") for n in TERMS])
        full_y = jnp.sum(jnp.where(is_ve[:, None, None], 0.0, dla_y), axis=0)
        # Check: the terms add up to the centred target logit (pre-cap).
        raw = jnp.einsum("bsd,bsd->bs", tr["hidden"].astype(jnp.float32), u_y / forward.final_gain(model_r))
        recon_err = jnp.max(jnp.abs(full_y - raw) * cats[0])

        out = dict(
            n=jnp.sum(cats, axis=(1, 2)),
            loss=jnp.einsum("cbs,bs->c", cats, tr["loss"]),
            top1_correct=jnp.einsum("cbs,bs->c", cats, (top1 == labels).astype(jnp.float32)),
            entropy=jnp.einsum("cbs,bs->c", cats, tr["entropy"]),
            dla_y=jnp.einsum("cbs,tbs->ct", cats, dla_y),
            dla_y_sq=jnp.einsum("cbs,tbs->ct", cats, jnp.square(dla_y)),
            dla_top=jnp.einsum("cbs,tbs->ct", cats, dla_top),
            dla_y_pos=jnp.einsum("cbs,tbs->ct", cats, (dla_y > 0).astype(jnp.float32)),
            recon_err=recon_err,
        )
        # Lenses.
        final_q = model_r.attn_res_query_final
        gate_loss, gate_agree, tuned_loss, tuned_agree, gate_kl = [], [], [], [], []
        final_logp = _labels_logp(model_r, tr["hidden"])
        for k, srcs in enumerate(_stage_sources(tr)):
            _, mixed = forward._gate(srcs, final_q, eps)
            for kind, hidden in (
                ("gate", model_r.final_norm(mixed)),
                (
                    "tuned",
                    (
                        jnp.concatenate(
                            [forward.rms_norm(mixed.astype(jnp.float32), eps), jnp.ones((*mixed.shape[:2], 1))], axis=-1
                        )
                        @ tuned[k]
                    ).astype(mixed.dtype),
                ),
            ):
                logp = _labels_logp(model_r, hidden)
                loss = -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
                agree = (jnp.argmax(logp, axis=-1) == top1).astype(jnp.float32)
                kl = jnp.sum(jnp.exp(final_logp) * (final_logp - logp), axis=-1)
                if kind == "gate":
                    gate_loss.append(loss)
                    gate_agree.append(agree)
                    gate_kl.append(kl)
                else:
                    tuned_loss.append(loss)
                    tuned_agree.append(agree)
        for name, arr in (
            ("lens_gate_loss", gate_loss),
            ("lens_gate_agree", gate_agree),
            ("lens_gate_kl", gate_kl),
            ("lens_tuned_loss", tuned_loss),
            ("lens_tuned_agree", tuned_agree),
        ):
            out[name] = jnp.einsum("cbs,kbs->ck", cats, jnp.stack(arr))
        # AttnRes gate weights: [gate, source] sums per category.
        gate_w = []
        for lay in tr["layers"]:
            for key in ("w_attn", "w_mlp"):
                w = lay[key]
                gate_w.append(jnp.pad(w, ((0, MAX_SOURCES - w.shape[0]), (0, 0), (0, 0))))
        gate_w.append(wf)
        gate_w = jnp.stack(gate_w)  # [G, N, B, S]
        out["gate_w"] = jnp.einsum("cbs,gnbs->cgn", cats, gate_w)
        out["gate_entropy"] = jnp.einsum(
            "cbs,gbs->cg", cats, -jnp.sum(gate_w * jnp.log(jnp.maximum(gate_w, 1e-30)), axis=1)
        )
        # Routing concentration.
        eff, top_w = [], []
        for lay in tr["layers"]:
            w = lay["combine"].reshape((*tokens.shape, forward.TOP_K))
            p = w / jnp.sum(w, axis=-1, keepdims=True)
            eff.append(jnp.exp(-jnp.sum(p * jnp.log(jnp.maximum(p, 1e-30)), axis=-1)))
            top_w.append(jnp.max(p, axis=-1))
        out["eff_experts"] = jnp.einsum("cbs,lbs->cl", cats, jnp.stack(eff))
        out["top_weight"] = jnp.einsum("cbs,lbs->cl", cats, jnp.stack(top_w))
        # Per-position output norms of each sublayer and the final gate's weight, for scale.
        out["term_norm"] = jnp.stack(
            [
                jnp.einsum("cbs,bs->c", cats, jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1)))
                for x in [tr["embed"]] + [
                    lay[k] for lay in tr["layers"] for k in ("attn_out", "shared_out", "routed_out")
                ]
            ],
            axis=1,
        )
        return out

    return loss_by_category, means, tuned_fit_stats, analyse


def ablations(head_mean, sub_mean) -> list[tuple[str, forward.Ablation]]:
    base = forward.no_ablation()
    out = []
    for i in range(L):
        for h in range(H):
            for kind in ("zero", "mean"):
                mult = base.head_mult.at[i, h].set(0.0)
                mean = head_mean if kind == "mean" else jnp.zeros_like(head_mean)
                out.append((f"head:L{i}.h{h}:{kind}", base._replace(head_mult=mult, head_mean=mean)))
    for i in range(L):
        for sub, name in ((forward.SUB_ATTN, "attn"), (forward.SUB_SHARED, "shared"), (forward.SUB_ROUTED, "routed")):
            for kind in ("zero", "mean"):
                mult = base.sub_mult.at[i, sub].set(0.0)
                mean = sub_mean if kind == "mean" else jnp.zeros_like(sub_mean)
                out.append((f"sub:L{i}.{name}:{kind}", base._replace(sub_mult=mult, sub_mean=mean)))
    for i in forward.MLA_LAYERS:
        out.append((f"sub:L{i}.ve:zero", base._replace(sub_mult=base.sub_mult.at[i, forward.SUB_VE].set(0.0))))
    for drop in (1, 2, 3, 5):
        keep = jnp.ones((L, forward.TOP_K)).at[:, forward.TOP_K - drop :].set(0.0)
        for renorm in (0.0, 1.0):
            name = f"slots:drop{drop}:{'renorm' if renorm else 'raw'}"
            out.append((name, base._replace(slot_mult=keep, slot_renorm=jnp.asarray(renorm))))
    for i in range(L):
        keep = jnp.ones((L, forward.TOP_K)).at[i, forward.TOP_K - 2 :].set(0.0)
        out.append((f"slots:L{i}.drop2:renorm", base._replace(slot_mult=keep, slot_renorm=jnp.asarray(1.0))))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fit-sequences", type=int, default=128, help="train sequences for means and the tuned lens")
    parser.add_argument("--eval-train-sequences", type=int, default=64)
    parser.add_argument("--eval-paloma-sequences", type=int, default=8, help="per Paloma set")
    parser.add_argument("--ablation-batches", type=int, default=12)
    args = parser.parse_args()
    checkpoint = args.checkpoint or common.CHECKPOINTS[args.tag]
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, checkpoint)
    vocab = common.vocab_strings()
    feats = categories.token_features(vocab)
    t0 = time.time()
    fit = list(common.train_batches(BATCH, args.fit_sequences, seed=1))
    counts = np.bincount(np.concatenate([b.tokens.reshape(-1) for b in fit]), minlength=forward.VOCAB)
    freq_rank = np.empty(forward.VOCAB, np.int64)
    freq_rank[np.argsort(-counts, kind="stable")] = np.arange(forward.VOCAB)
    evals = [
        *common.paloma_batches(BATCH, max_sequences=args.eval_paloma_sequences),
        *common.train_batches(BATCH, args.eval_train_sequences, seed=2),
    ]
    cats_all = [
        categories.category_masks(b.tokens, b.segment_ids, b.loss_weight, b.source, feats, freq_rank) for b in evals
    ]
    results: dict[str, np.ndarray] = {}
    with set_mesh(mesh):
        model = common.compute_model(params)
        loss_by_category, means_fn, tuned_fn, analyse_fn = make_fns()
        head_sum = sub_sum = None
        n_sum = 0.0
        xtx = xty = None
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
                np.linalg.solve(xtx_h[k] + TUNED_RIDGE * n_sum * np.eye(xtx_h.shape[1]), xty_h[k])
                for k in range(len(STAGES))
            ]
        ).astype(np.float32)
        logger.info("means and tuned lens fit on %d tokens (%.0fs)", n_sum, time.time() - t0)
        tuned_dev = jnp.asarray(tuned)
        acc: dict[str, np.ndarray] = {}
        for b, cats in zip(evals, cats_all, strict=True):
            tokens, seg, _ = common.device_batch(b, mesh)
            out = analyse_fn(model, tokens, seg, jnp.asarray(cats), tuned_dev)
            for k, v in out.items():
                v = np.asarray(v, np.float64)
                if k == "recon_err":
                    acc[k] = np.maximum(acc.get(k, 0.0), v)
                else:
                    acc[k] = acc.get(k, 0.0) + v
        results.update({k: np.asarray(v) for k, v in acc.items()})
        logger.info("analysis done (%.0fs), recon err %.4f", time.time() - t0, float(acc["recon_err"]))
        # Ablations on a fixed subset of the eval batches spread over the sources.
        pick = np.linspace(0, len(evals) - 1, args.ablation_batches).round().astype(int)
        abl_names, abl_loss = [], []
        base_loss = np.zeros(NC)
        base_n = np.zeros(NC)
        for j in pick:
            tokens, seg, _ = common.device_batch(evals[j], mesh)
            lsum, n = loss_by_category(model, tokens, seg, jnp.asarray(cats_all[j]), forward.no_ablation())
            base_loss += np.asarray(lsum)
            base_n += np.asarray(n)
        for name, abl in ablations(head_mean, sub_mean):
            total = np.zeros(NC)
            for j in pick:
                tokens, seg, _ = common.device_batch(evals[j], mesh)
                lsum, _ = loss_by_category(model, tokens, seg, jnp.asarray(cats_all[j]), abl)
                total += np.asarray(lsum)
            abl_names.append(name)
            abl_loss.append(total)
            logger.info(
                "ablation %s: dloss(all) %+.4f (%.0fs)", name, (total[0] - base_loss[0]) / base_n[0], time.time() - t0
            )
        results["ablation_names"] = np.asarray(abl_names)
        results["ablation_loss"] = np.stack(abl_loss)
        results["ablation_base_loss"] = base_loss
        results["ablation_n"] = base_n
        results["head_mean"] = np.asarray(head_mean)
        results["sub_mean"] = np.asarray(sub_mean)
        results["tuned"] = tuned
    results["categories"] = np.asarray(categories.CATEGORIES)
    results["terms"] = np.asarray(TERMS)
    results["stages"] = np.asarray(STAGES)
    results["freq_rank"] = freq_rank
    common.save_npz(args.tag, "attrib", **results)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
