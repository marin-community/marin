# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU microbench: fused ragged ReLU^2 expert MLP vs haliax ``ragged_dot`` + elementwise ReLU^2.

Shape is one ragged-all-to-all MoE chunk on one GPU at the d512 recipe (see ``bench_ragged_vs_pooled``):
384 experts over 8 GPUs = 48 local experts in 2 chunks of G=24; receiver buffer M = ceil(1.15 * 65536 * 8 / 2)
= 301466 rows, of which ~262144 are active (QB-balanced routing); K = 256 (latent), N = 384 (intermediate).

1. Checks fused vs haliax value + grads (bf16).
2. Sweeps the row-grouped GEMM tiles (up, down, d pre, dx) and the weight-gradient tiles separately.
3. Times forward and forward+backward: haliax path, fused defaults, fused with the swept winners.

    python -m scratch_lc1.bench_relu2_ragged_kernel
"""

import argparse
import dataclasses
import itertools
import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn.ragged_dot import ragged_dot
from levanter.kernels.pallas.relu2_ragged_mlp import BlockSizes, GmmBlockSizes, TgmmBlockSizes, relu2_ragged_mlp
from levanter.kernels.pallas.relu2_ragged_mlp.pallas_gpu import Epilogue, gmm, tgmm

TOKENS_PER_GPU = 65536
GPUS = 8
TOPK = 8
NUM_EXPERTS = 384
CHUNKS = 2
CAPACITY_FACTOR = 1.15
K = 256
N = 384

GMM_CANDIDATES = [
    GmmBlockSizes(bm=bm, bn=bn, bk=bk, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [
        (64, 64, 64, 4, 2),
        (64, 64, 64, 4, 3),
        (64, 128, 64, 4, 3),
        (128, 64, 64, 4, 3),
        (128, 128, 32, 4, 3),
        (128, 128, 64, 4, 3),
        (128, 128, 64, 8, 3),
        (128, 128, 64, 8, 4),
        (128, 64, 128, 4, 2),
        (64, 128, 128, 4, 2),
        (128, 128, 128, 8, 2),
        (256, 128, 64, 8, 3),
    ]
]
TGMM_CANDIDATES = [
    TgmmBlockSizes(bm=bm, bn=bn, bk=bk, splits=sp, num_warps=w, num_stages=s)
    for (bm, bn, bk, w, s), sp in itertools.product(
        [(64, 128, 64, 4, 2), (64, 64, 64, 4, 3), (128, 128, 32, 4, 3), (128, 128, 64, 8, 3), (128, 128, 64, 4, 3)],
        [1, 2, 4, 8],
    )
]


def _inputs(seed: int):
    local_experts = NUM_EXPERTS // GPUS
    groups = local_experts // CHUNKS
    rows = math.ceil(math.ceil(CAPACITY_FACTOR * TOKENS_PER_GPU * TOPK) / CHUNKS)
    active = TOKENS_PER_GPU * TOPK * GPUS // NUM_EXPERTS * groups
    rng = np.random.default_rng(seed)
    sizes = rng.multinomial(active, np.full(groups, 1.0 / groups)).astype(np.int32)
    physical = sizes.copy()
    physical[-1] += rows - sizes.sum()
    keys = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(keys[0], (rows, K), jnp.bfloat16)
    x = x.at[int(sizes.sum()) :].set(0)  # trailing capacity rows are the dispatch buffer's zeros
    w_up = (jax.random.normal(keys[1], (groups, K, N)) * K**-0.5).astype(jnp.bfloat16)
    w_down = (jax.random.normal(keys[2], (groups, N, K)) * N**-0.5).astype(jnp.bfloat16)
    g = jax.random.normal(keys[3], (rows, K), jnp.bfloat16)
    return x, w_up, w_down, jnp.asarray(sizes), jnp.asarray(physical), g


def _haliax_mlp(x, w_up, w_down, physical_sizes):
    """The ragged backend's generic path: two haliax ``ragged_dot`` over the physical sizes, elementwise ReLU^2."""
    return ragged_dot(jnp.square(jax.nn.relu(ragged_dot(x, w_up, physical_sizes))), w_down, physical_sizes)


def _time_ms(fn, *args, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _fwd_bwd(mlp, g):
    def loss(x, w_up, w_down):
        return jnp.sum((mlp(x, w_up, w_down) * g).astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))


def _rel_err(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()
    print(f"{jax.devices()[0].device_kind}; jax {jax.__version__}")

    x, w_up, w_down, sizes, physical, g = _inputs(args.seed)
    rows, groups = x.shape[0], w_up.shape[0]
    print(f"M={rows} (active {int(sizes.sum())}) G={groups} K={K} N={N}")
    timing = dict(iters=args.iters, warmup=args.warmup)

    def fused(bs: BlockSizes):
        return lambda a, b, c: relu2_ragged_mlp(a, b, c, sizes, implementation="pallas_gpu", block_sizes=bs)

    haliax_fb = _fwd_bwd(lambda a, b, c: _haliax_mlp(a, b, c, physical), g)
    fused_fb = _fwd_bwd(fused(BlockSizes()), g)
    (want, want_grads), (got, got_grads) = haliax_fb(x, w_up, w_down), fused_fb(x, w_up, w_down)
    print(f"parity vs haliax: loss rel {abs(float(got) - float(want)) / abs(float(want)):.2e}", end="")
    for name, a, b in zip(["dx", "dw_up", "dw_down"], got_grads, want_grads, strict=True):
        print(f"  {name} max rel {_rel_err(a, b):.2e}", end="")
    print()

    best = BlockSizes()
    if not args.no_sweep:
        post = gmm(x, w_up, sizes, trans_b=False, epilogue=Epilogue.RELU2, block_sizes=GmmBlockSizes())
        dpre = post  # same shape and dtype as d pre; timing only

        def gmm_calls(bs: GmmBlockSizes):
            @jax.jit
            def run(x, w_up, w_down, post, g, dpre):
                call = lambda *a, **k: gmm(*a, block_sizes=bs, **k)  # noqa: E731
                return (
                    call(x, w_up, sizes, trans_b=False, epilogue=Epilogue.RELU2),
                    call(post, w_down, sizes, trans_b=False, epilogue=Epilogue.NONE),
                    call(g, w_down, sizes, post, trans_b=True, epilogue=Epilogue.RELU2_DPRE),
                    call(dpre, w_up, sizes, trans_b=True, epilogue=Epilogue.NONE),
                )

            return run

        def tgmm_calls(bs: TgmmBlockSizes):
            @jax.jit
            def run(x, post, g, dpre):
                return tgmm(post, g, sizes, block_sizes=bs), tgmm(x, dpre, sizes, block_sizes=bs)

            return run

        print("\ngmm (up + down + dpre + dx) ms:")
        gmm_results = []
        for bs in GMM_CANDIDATES:
            try:
                ms = _time_ms(gmm_calls(bs), x, w_up, w_down, post, g, dpre, **timing)
            except Exception as exc:  # a tile that fails to compile (e.g. shared memory) is a sweep result
                print(f"  {bs}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
                continue
            gmm_results.append((ms, bs))
            print(f"  {bs}: {ms:.3f}")
        print("\ntgmm (d w_down + d w_up) ms:")
        tgmm_results = []
        for bs in TGMM_CANDIDATES:
            try:
                ms = _time_ms(tgmm_calls(bs), x, post, g, dpre, **timing)
            except Exception as exc:
                print(f"  {bs}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
                continue
            tgmm_results.append((ms, bs))
            print(f"  {bs}: {ms:.3f}")
        best = BlockSizes(gmm=min(gmm_results, key=lambda r: r[0])[1], tgmm=min(tgmm_results, key=lambda r: r[0])[1])
        print(f"\nbest: {best}")

    print("\nvariant                    fwd ms   fwd+bwd ms")
    for name, mlp in [
        ("haliax ragged_dot+relu2", lambda a, b, c: _haliax_mlp(a, b, c, physical)),
        ("fused (defaults)", fused(BlockSizes())),
        ("fused (swept best)", fused(best)),
    ]:
        fwd_ms = _time_ms(jax.jit(mlp), x, w_up, w_down, **timing)
        fb_ms = _time_ms(_fwd_bwd(mlp, g), x, w_up, w_down, **timing)
        print(f"{name:<26} {fwd_ms:7.3f}   {fb_ms:8.3f}")
    print(f"(x{CHUNKS} chunks per MoE layer; best as code: {dataclasses.asdict(best)})")


if __name__ == "__main__":
    main()
