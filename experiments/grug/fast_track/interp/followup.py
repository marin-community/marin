# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Follow-ups to the first results: the copy task under *mean* ablation of each head (zero ablation also removes a
head's constant output, which dominated L3.h2's natural-text effect), and with the MLA value embeddings removed.

Reads the head and sublayer means from ``attrib.npz``; writes ``followup.npz``.
"""

import argparse
import io
import logging

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import common, forward, heads

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--induction-sequences", type=int, default=8)
    args = parser.parse_args()
    checkpoint = common.CHECKPOINTS[args.tag]
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, checkpoint)
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/attrib.npz", "rb") as f:
        attrib = np.load(io.BytesIO(f.read()))
    head_mean, sub_mean = jnp.asarray(attrib["head_mean"]), jnp.asarray(attrib["sub_mean"])
    batches = [heads.induction_batch(heads.BATCH, 100 + s) for s in range(args.induction_sequences // heads.BATCH)]
    period = 2 * heads.INDUCTION_LEN
    reps = common.SEQ_LEN // period
    pos = np.arange(period)
    second = np.tile((pos >= heads.INDUCTION_LEN) & (pos < period - 1), reps)
    first = np.tile(pos < heads.INDUCTION_LEN - 1, reps)
    base = forward.no_ablation()
    configs: list[tuple[str, forward.Ablation]] = [("none", base)]
    for i in range(forward.NUM_LAYERS):
        for h in range(forward.NUM_HEADS):
            configs.append(
                (f"L{i}.h{h}:mean", base._replace(head_mult=base.head_mult.at[i, h].set(0.0), head_mean=head_mean))
            )
        configs.append((f"L{i}.all:mean", base._replace(head_mult=base.head_mult.at[i].set(0.0), head_mean=head_mean)))
        for sub, name in ((forward.SUB_SHARED, "shared"), (forward.SUB_ROUTED, "routed")):
            configs.append(
                (f"L{i}.{name}:mean", base._replace(sub_mult=base.sub_mult.at[i, sub].set(0.0), sub_mean=sub_mean))
            )
    for i in forward.MLA_LAYERS:
        configs.append((f"L{i}.ve:zero", base._replace(sub_mult=base.sub_mult.at[i, forward.SUB_VE].set(0.0))))
    results = {"names": [], "first": [], "second": []}
    with set_mesh(mesh):
        model = common.compute_model(params)

        @jax.jit
        def loss_fn(model, tokens, seg, abl):
            return forward.trace_forward(forward.replicate(model), tokens, seg, abl)["loss"]

        for name, abl in configs:
            f_sum = s_sum = 0.0
            for b in batches:
                tokens, seg, _ = common.device_batch(b, mesh)
                loss = np.asarray(loss_fn(model, tokens, seg, abl))
                f_sum += float(loss[:, first].mean())
                s_sum += float(loss[:, second].mean())
            results["names"].append(name)
            results["first"].append(f_sum / len(batches))
            results["second"].append(s_sum / len(batches))
            logger.info("%s: first %.3f second %.3f", name, results["first"][-1], results["second"][-1])
    common.save_npz(args.tag, "followup", **{k: np.asarray(v) for k, v in results.items()})


if __name__ == "__main__":
    main()
