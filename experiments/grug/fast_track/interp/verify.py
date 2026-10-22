# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the d512 checkpoint and the instrumented forward.

1. Chunk completeness: every array in the checkpoint's OCDBT store has every chunk its grid needs (the TensorStore
   0.1.84 bug fixed in 0dbf62af35 silently dropped whole chunks).
2. No routed expert of any layer has all-zero weights.
3. Paloma eval through ``Transformer.next_token_loss`` (dropless local MoE, the run's eval path): per-set and macro
   loss, to compare with the run's logged final eval.
4. ``forward.trace_forward`` against it on the first batches of every set.

Writes ``verify.json`` and the vocabulary (``vocab.json``) under ``common.OUTPUT_ROOT``.
"""

import argparse
import asyncio
import collections
import json
import logging
import math
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.attention import AttentionMask
from levanter.tensorstore_serialization import _list_ocdbt_keys

from experiments.grug.fast_track.interp import common, forward
from experiments.grug.fast_track.train import _to_dropless_local, dropless_moe_implementation

logger = logging.getLogger(__name__)

EVAL_BATCH = 16
TRACE_BATCH = 8
TRACE_BATCHES_PER_SET = 2


def chunk_report(root: str) -> dict:
    fs, path = fsspec.core.url_to_fs(root)
    manifest = json.loads(fs.cat(path + "/manifest.json"))
    keys = [k for k in asyncio.run(_list_ocdbt_keys(root)) if not k.endswith("zarr.json")]

    def array_of(key: str) -> str:
        parts = key.split("/")
        i = len(parts) - 1
        while i > 0 and parts[i] != "c":
            i -= 1
        return "/".join(parts[:i])

    per = collections.Counter(array_of(k) for k in keys)
    short = []
    for a in manifest["arrays"]:
        expected = (
            math.prod(math.ceil(s / c) for s, c in zip(a["shape"], a["chunk_shape"], strict=True)) if a["shape"] else 1
        )
        got = per.get(a["path"], 0)
        if got != expected:
            short.append(dict(path=a["path"], got=got, expected=expected, shape=a["shape"], chunk=a["chunk_shape"]))
    return dict(arrays=len(manifest["arrays"]), short=short)


def zero_expert_report(params) -> dict:
    out = {}
    for stack, indices in zip(params.layer_stacks(), params.stack_layer_indices(), strict=True):
        em = stack.stacked.mlp.expert_mlp
        for name in ("w_up", "w_down"):
            w = np.asarray(jax.device_get(getattr(em, name)))
            zero = np.all(w == 0, axis=tuple(range(2, w.ndim)))  # [layers, experts]
            for row, layer in enumerate(indices):
                out[f"L{layer}/{name}"] = int(zero[row].sum())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", required=True, help="study tag (a key of common.CHECKPOINTS, or any name with --checkpoint)"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--logged-paloma-macro", type=float, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint or common.CHECKPOINTS[args.tag]
    cfg = common.model_config()
    forward.check_config(cfg)
    report: dict = {"config": {k: str(v) for k, v in vars(cfg).items()}}
    t0 = time.time()
    report["checkpoint"] = checkpoint
    report["chunks"] = chunk_report(checkpoint)
    logger.info(
        "chunk check: %d arrays, %d short: %s",
        report["chunks"]["arrays"],
        len(report["chunks"]["short"]),
        report["chunks"]["short"],
    )
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, checkpoint)
    report["zero_experts"] = zero_expert_report(params)
    logger.info("zero experts: %s", report["zero_experts"])
    vocab = common.vocab_strings()
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/vocab.json", "w") as f:
        json.dump(vocab, f)

    with set_mesh(mesh):
        model = common.compute_model(params)
        eval_model = _to_dropless_local(model, implementation=dropless_moe_implementation(cfg))

        @jax.jit
        def model_loss(m, tokens, seg):
            mask = AttentionMask.causal().with_segment_ids(seg)
            return m.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32), mask=mask, reduction="none")

        @jax.jit
        def trace_loss(m, tokens, seg):
            tr = forward.trace_forward(m, tokens, seg, forward.no_ablation())
            return tr["loss"]

        sums = collections.defaultdict(float)
        weights = collections.defaultdict(float)
        trace_cmp = collections.defaultdict(list)
        set_batches = collections.Counter()
        for batch in common.paloma_batches(EVAL_BATCH):
            tokens, seg, _ = common.device_batch(batch, mesh)
            loss = np.asarray(model_loss(eval_model, tokens, seg))
            wn = np.asarray(batch.loss_weight)
            sums[batch.source] += float(np.sum(loss * wn))
            weights[batch.source] += float(np.sum(wn))
            if set_batches[batch.source] < TRACE_BATCHES_PER_SET:
                for half in range(EVAL_BATCH // TRACE_BATCH):
                    sl = slice(half * TRACE_BATCH, (half + 1) * TRACE_BATCH)
                    tl = trace_loss(model, tokens[sl], seg[sl])
                    tl = np.asarray(tl)
                    ww = wn[sl]
                    trace_cmp[batch.source].append(
                        dict(
                            model=float(np.sum(loss[sl] * ww) / max(ww.sum(), 1)),
                            trace=float(np.sum(tl * ww) / max(ww.sum(), 1)),
                            max_abs=float(np.max(np.abs(tl - loss[sl]) * (ww > 0))),
                        )
                    )
            set_batches[batch.source] += 1
        per_set = {name: sums[name] / weights[name] for name in sums}
        report["paloma_loss"] = per_set
        report["paloma_tokens"] = dict(weights)
        report["paloma_macro"] = float(np.mean(list(per_set.values())))
        report["logged_paloma_macro"] = args.logged_paloma_macro
        report["trace_vs_model"] = trace_cmp
    report["seconds"] = time.time() - t0
    logger.info("paloma macro %.6f (logged %.6f)", report["paloma_macro"], args.logged_paloma_macro)
    common.save_json(args.tag, "verify", report)


if __name__ == "__main__":
    main()
