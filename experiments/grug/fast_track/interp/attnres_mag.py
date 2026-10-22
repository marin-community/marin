# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AttnRes weights are not shares: every gate mixes the *raw* sources (only its logits read them RMS-normalized), so a
source's contribution is ``w_i * src_i``. Per gate (the 12 sublayer gates and the final readout): each source's RMS, its
magnitude share ``|w_i src_i| / sum_j |w_j src_j|``, and its projection share ``(w_i src_i . mix) / |mix|^2`` (shares sum
to 1, cancellation shows as negative). At the final gate: each source's direct logit attribution to the target, the loss
with one source's term zeroed or replaced by its mean, and the loss with the gate replaced by a plain sum rescaled to the
mix's RMS. Writes ``attnres_mag.npz``.
"""

import argparse
import logging

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import categories, common, forward

logger = logging.getLogger(__name__)

BATCH = 8
EPS = 1e-5
NSRC = forward.NUM_LAYERS + 1


def gate_sources(tr) -> list[tuple[list, jax.Array]]:
    """``(sources, weights)`` of every gate in order: attn0, mlp0, ..., attn5, mlp5, final."""
    blocks = tr["blocks"]
    out = []
    for i, lay in enumerate(tr["layers"]):
        out.append((blocks[: i + 1], lay["w_attn"]))
        out.append(([*blocks[: i + 1], lay["attn_out"]], lay["w_mlp"]))
    out.append((blocks, tr["w_final"]))
    return out


def make_fn():
    @jax.jit
    def run(model, tokens, seg, cats, src_mean):
        m = forward.replicate(model)
        tr = forward.replicate(forward.trace_forward(m, tokens, seg, forward.no_ablation()))
        w = cats[0]  # scored positions
        n = jnp.sum(w)
        out = {}
        rms, mag, proj = [], [], []
        for srcs, wts in gate_sources(tr):
            terms = [wts[j][..., None] * s.astype(jnp.float32) for j, s in enumerate(srcs)]
            mix = sum(terms)
            norms = jnp.stack([jnp.linalg.norm(t, axis=-1) for t in terms])  # [N, B, S]
            pr = jnp.stack([jnp.sum(t * mix, -1) for t in terms]) / jnp.maximum(jnp.sum(mix * mix, -1), 1e-9)
            r = jnp.stack([jnp.sqrt(jnp.mean(jnp.square(s.astype(jnp.float32)), -1)) for s in srcs])
            pad = ((0, NSRC - len(srcs)), (0, 0), (0, 0))
            rms.append(jnp.pad(jnp.einsum("nbs,bs->n", r, w)[:, None, None], pad)[:, 0, 0] / n)
            mag.append(
                jnp.pad(jnp.einsum("nbs,bs->n", norms / jnp.sum(norms, 0, keepdims=True), w)[:, None, None], pad)[
                    :, 0, 0
                ]
                / n
            )
            proj.append(jnp.pad(jnp.einsum("nbs,bs->n", pr, w)[:, None, None], pad)[:, 0, 0] / n)
        out["rms"], out["mag_share"], out["proj_share"] = jnp.stack(rms), jnp.stack(mag), jnp.stack(proj)
        # Final gate: per-source DLA, single-source ablations, plain-sum counterfactual.
        srcs, wts = gate_sources(tr)[-1]
        terms = [wts[j][..., None] * s.astype(jnp.float32) for j, s in enumerate(srcs)]
        mix = sum(terms)
        labels = tr["labels"]
        unembed = forward.unembed_direction(m)
        u_y = unembed.T[labels] - jnp.mean(unembed, axis=1)
        inv = tr["inv_rms"][..., None]
        out["dla"] = jnp.stack([jnp.einsum("cbs,bs->c", cats, jnp.sum(t * inv * u_y, -1)) for t in terms])  # [N, C]

        def loss_of(mixed):
            hidden = m.final_norm(mixed.astype(tr["mixed"].dtype))
            logits = forward.soft_cap(
                jnp.einsum("bsd,dv->bsv", hidden, m.output_proj, preferred_element_type=jnp.float32)
            )
            logp = jax.nn.log_softmax(logits, -1)
            return jnp.einsum("cbs,bs->c", cats, -jnp.take_along_axis(logp, labels[..., None], -1)[..., 0])

        out["base"] = loss_of(mix)
        zero, mean = [], []
        for j, t in enumerate(terms):
            zero.append(loss_of(mix - t))
            mean.append(loss_of(mix - t + wts[j][..., None] * src_mean[j]))
        out["zero"], out["mean"] = jnp.stack(zero), jnp.stack(mean)
        plain = sum(s.astype(jnp.float32) for s in srcs)
        scale = jnp.sqrt(
            jnp.mean(mix * mix, -1, keepdims=True) / jnp.maximum(jnp.mean(plain * plain, -1, keepdims=True), 1e-9)
        )
        out["plain_sum"] = loss_of(plain * scale)
        uniform = plain / len(srcs)
        out["uniform"] = loss_of(uniform)
        out["n"] = jnp.sum(cats, axis=(1, 2))
        out["src_sum"] = jnp.stack([jnp.einsum("bsd,bs->d", s.astype(jnp.float32), w) for s in srcs])
        return out

    return run


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-sequences", type=int, default=64)
    parser.add_argument("--paloma-sequences", type=int, default=4)
    args = parser.parse_args()
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    vocab = common.vocab_strings()
    feats = categories.token_features(vocab)
    batches = [
        *common.paloma_batches(BATCH, max_sequences=args.paloma_sequences),
        *common.train_batches(BATCH, args.train_sequences, seed=11),
    ]
    counts = np.bincount(np.concatenate([b.tokens.reshape(-1) for b in batches]), minlength=forward.VOCAB)
    freq_rank = np.empty(forward.VOCAB, np.int64)
    freq_rank[np.argsort(-counts, kind="stable")] = np.arange(forward.VOCAB)
    acc: dict[str, np.ndarray] = {}
    with set_mesh(mesh):
        model = common.compute_model(params)
        run = make_fn()
        src_mean = jnp.zeros((NSRC, forward.HIDDEN))
        # Pass 1: the sources' mean vectors (for mean ablation); pass 2: everything with them.
        for pass_index in range(2):
            acc = {}
            for b in batches:
                tokens, seg, _ = common.device_batch(b, mesh)
                cats = categories.category_masks(b.tokens, b.segment_ids, b.loss_weight, b.source, feats, freq_rank)
                out = run(model, tokens, seg, jnp.asarray(cats), src_mean)
                for k, v in out.items():
                    acc[k] = acc.get(k, 0.0) + np.asarray(v, np.float64)
            src_mean = jnp.asarray(acc["src_sum"] / acc["n"][0], jnp.float32)
            logger.info("pass %d: base loss %.4f", pass_index, acc["base"][0] / acc["n"][0])
    n_batches = len(batches)
    for k in ("rms", "mag_share", "proj_share"):
        acc[k] = acc[k] / n_batches
    acc["categories"] = np.asarray(categories.CATEGORIES)
    common.save_npz(args.tag, "attnres_mag", **acc)


if __name__ == "__main__":
    main()
