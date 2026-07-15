# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward-only device-ceiling microbenchmark for the pooled fast-transformer.

Measures the *device upper bound*: pre-tokenized ``[B, T]`` ids resident on the TPU,
warm (compiled) forward, several async launches queued before a single
``block_until_ready``. This is deliberately NOT the achievable pipeline throughput
(the real pipeline is host-bound by tokenization) -- it answers "how little of the
v6e does the forward pass actually use?" and bounds the device side of the scaling
projection.

Weights are random and the vocab is a placeholder: forward FLOPs and array shapes are
config-dependent only, so throughput is weight-independent. Run on a v6e via an Iris
job; prints one JSON line per (config, batch) measurement to stdout.
"""

import argparse
import json
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.inference import (
    _predict_batch,
    data_parallel_shardings,
)
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastTransformer,
    FastTransformerConfig,
)

logger = logging.getLogger(__name__)

# Peak bf16 FLOP/s per v6e chip (fray/device_flops.py).
V6E_BF16_PEAK_FLOPS = 918e12

# The deployed scorer's architecture (train.py DEPLOY_CONFIG + MAX_TOKENS).
DEPLOYED = dict(embed_dim=256, hidden_dim=256, num_layers=2, num_heads=4, pool_window=64, pool_kind="meanmaxmin")
# A 4x-larger variant: still under the 1M FLOPs/token budget (~420K), to show device headroom.
BIG = dict(embed_dim=512, hidden_dim=512, num_layers=4, num_heads=8, pool_window=64, pool_kind="meanmaxmin")


def make_model(cfg_kwargs: dict, vocab_size: int, max_tokens: int) -> tuple[FastTransformer, FastTransformerConfig]:
    config = FastTransformerConfig(vocab_size=vocab_size, max_tokens=max_tokens, final_pool="mean", **cfg_kwargs)
    model = FastTransformer(config, key=jax.random.PRNGKey(0))
    return model, config


def bench_one(
    model: FastTransformer,
    config: FastTransformerConfig,
    *,
    batch: int,
    launches: int,
    repeats: int,
) -> dict:
    """Time ``launches`` async forwards on a resident ``[batch, T]`` batch, ``repeats`` times.

    Returns the best (max-throughput) repeat to reject scheduler noise. The batch is
    sharded data-parallel across all chips exactly as ``predict`` shards it.
    """
    t = config.max_tokens
    ndev, replicated, batch_shard = data_parallel_shardings()
    batch = max(ndev, (batch // ndev) * ndev)
    model = jax.device_put(model, replicated)
    # Resident ids: real token range so the embedding gather is representative.
    ids_host = np.random.default_rng(0).integers(1, config.vocab_size, size=(batch, t), dtype=np.int32)
    ids = jax.device_put(jnp.asarray(ids_host), batch_shard)

    # Warmup: force compile for this shape, then block.
    jax.block_until_ready(_predict_batch(model, ids))

    best_tok_s = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        outs = [_predict_batch(model, ids) for _ in range(launches)]  # queued async
        jax.block_until_ready(outs)
        dt = time.perf_counter() - t0
        tok_s = batch * t * launches / dt
        best_tok_s = max(best_tok_s, tok_s)

    flops_per_token = config.flops_per_token()
    mxu_util = best_tok_s * flops_per_token / (ndev * V6E_BF16_PEAK_FLOPS)
    return {
        "kind": "microbench",
        "n_chips": ndev,
        "batch_global": batch,
        "max_tokens": t,
        "launches": launches,
        "flops_per_token": round(flops_per_token),
        "tokens_per_s": round(best_tok_s),
        "tokens_per_s_per_chip": round(best_tok_s / ndev),
        "docs_per_s_approx": round(best_tok_s / t),  # 1 window == 1 "doc" here
        "mxu_util_frac": round(mxu_util, 5),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["deployed", "big"], default="deployed")
    p.add_argument("--vocab-size", type=int, default=48000)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--batches", type=int, nargs="+", default=[512, 2048, 8192, 32768, 65536])
    p.add_argument("--launches", type=int, default=32, help="async forwards queued before one sync")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg_kwargs = DEPLOYED if args.model == "deployed" else BIG
    logger.info("devices: %s", jax.devices())
    for batch in args.batches:
        model, config = make_model(cfg_kwargs, args.vocab_size, args.max_tokens)
        try:
            row = bench_one(model, config, batch=batch, launches=args.launches, repeats=args.repeats)
        except Exception as e:  # OOM at large batch: record and continue the sweep
            row = {"kind": "microbench", "batch_global": batch, "max_tokens": args.max_tokens, "error": repr(e)[:200]}
        row["model"] = args.model
        print("BENCH " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
