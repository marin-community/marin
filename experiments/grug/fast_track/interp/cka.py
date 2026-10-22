# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Linear CKA between two checkpoints' per-layer representations on the same tokens (the layer onset study).

Both models run ``forward.trace_forward`` on identical batches (unseen training-cache sequences plus Paloma). For each
representation family -- ``block`` (what each layer writes: the embedding, then each layer's block output), ``read``
(each layer's attention-gate mix, what it reads), ``attn`` / ``mlp`` (each sublayer's output) and ``final`` (the final
gate's mix) -- and every pair of layers ``(i in tag, j in ref)``, the centred cross-covariance is accumulated over all
valid tokens, so CKA is exact over the sample: ``CKA(X, Y) = ||C_xy||_F^2 / (||C_xx||_F ||C_yy||_F)``. Writes
``cka_<ref>.npz`` with one ``[N_tag, N_ref]`` matrix per family, and ``self_<family>`` (the tag model against itself).
"""

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

# Full float32 products: raw second moments are centred later, so TF32 rounding would not cancel out.
HIGHEST = jax.lax.Precision.HIGHEST

BATCH = 8
FAMILIES = ("block", "read", "attn", "mlp", "final")


def representations(tr: dict) -> dict[str, list[jax.Array]]:
    """Each family's ``[B, S, D]`` representations, in layer order."""
    layers = tr["layers"]
    return {
        "block": tr["blocks"],
        "read": [lay["attn_gate_in"] for lay in layers],
        "attn": [lay["attn_out"] for lay in layers],
        "mlp": [lay["mlp_out"] for lay in layers],
        "final": [tr["mixed"]],
    }


def make_fn():
    @jax.jit
    def run(model_a, model_b, tokens, seg, w):
        abl = forward.no_ablation()
        ra = representations(forward.trace_forward(model_a, tokens, seg, abl))
        rb = representations(forward.trace_forward(model_b, tokens, seg, abl))
        m = reshard((w > 0).astype(jnp.float32).reshape(-1), P(None))
        out = {"n": jnp.sum(m)}

        def flat(x):
            # Replicated (one GPU): the token contraction is then unambiguous under explicit sharding.
            return reshard(x.astype(jnp.float32).reshape(-1, x.shape[-1]), P(None, None)) * m[:, None]

        for fam in FAMILIES:
            xa = jnp.stack([flat(x) for x in ra[fam]])  # [Na, T, D], zero rows off the mask
            xb = jnp.stack([flat(x) for x in rb[fam]])
            out[f"{fam}/sum_a"] = jnp.sum(xa, axis=1)
            out[f"{fam}/sum_b"] = jnp.sum(xb, axis=1)
            out[f"{fam}/ab"] = jnp.einsum("itd,jte->ijde", xa, xb, precision=HIGHEST)
            out[f"{fam}/aa"] = jnp.einsum("itd,jte->ijde", xa, xa, precision=HIGHEST)
            out[f"{fam}/bb"] = jnp.einsum("itd,ite->ide", xb, xb, precision=HIGHEST)
        return out

    return run


def cka_matrices(acc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact linear CKA from the accumulated sums (float64)."""
    n = acc["n"]
    result = {}
    for fam in FAMILIES:
        mu_a, mu_b = acc[f"{fam}/sum_a"] / n, acc[f"{fam}/sum_b"] / n
        c_ab = acc[f"{fam}/ab"] / n - np.einsum("id,je->ijde", mu_a, mu_b)
        c_aa = acc[f"{fam}/aa"] / n - np.einsum("id,je->ijde", mu_a, mu_a)
        c_bb = acc[f"{fam}/bb"] / n - np.einsum("id,ie->ide", mu_b, mu_b)
        norm_a = np.sqrt(np.einsum("iide->i", c_aa**2))
        norm_b = np.sqrt(np.einsum("ide->i", c_bb**2))
        hsic = np.einsum("ijde->ij", c_ab**2)
        result[fam] = hsic / (norm_a[:, None] * norm_b[None, :])
        result[f"self_{fam}"] = np.einsum("ijde->ij", c_aa**2) / (norm_a[:, None] * norm_a[None, :])
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ref", required=True, help="the reference checkpoint's tag (e.g. the control)")
    parser.add_argument("--train-sequences", type=int, default=64)
    parser.add_argument("--paloma-sequences", type=int, default=2, help="per Paloma set")
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params_a = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    params_b = common.load_params(cfg, mesh, common.CHECKPOINTS[args.ref])
    batches = [
        *common.paloma_batches(BATCH, max_sequences=args.paloma_sequences),
        *common.train_batches(BATCH, args.train_sequences, seed=21),
    ]
    acc: dict[str, np.ndarray] = {}
    with set_mesh(mesh):
        model_a, model_b = common.compute_model(params_a), common.compute_model(params_b)
        run = make_fn()
        for k, b in enumerate(batches):
            tokens, seg, w = common.device_batch(b, mesh)
            for name, v in run(model_a, model_b, tokens, seg, w).items():
                acc[name] = acc.get(name, 0.0) + np.asarray(v, np.float64)
            if k % 10 == 0:
                logger.info("batch %d/%d (%.0fs)", k, len(batches), time.time() - t0)
    result = cka_matrices(acc)
    for fam in FAMILIES:
        logger.info("%s diag CKA vs %s: %s", fam, args.ref, np.round(np.diagonal(result[fam]), 3).tolist())
    common.save_npz(args.tag, f"cka_{args.ref}", n=np.asarray(acc["n"]), **result)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
