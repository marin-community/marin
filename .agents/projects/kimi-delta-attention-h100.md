# Kimi Delta Attention (KDA) on H100 — MFU / throughput

Target: make the KDA linear-attention token mixer (`experiments/grug/moe/kda.py`)
run fast on **H100 (Hopper, SM90)**. Prior note `kimi-delta-attention-hybrid.md`
targeted GB200/SM100; this is the Hopper follow-up. Companion microbenchmark:
`experiments/grug/moe/bench_kda.py`.

## What the current kernel is

`chunk_kda` is already the **chunked-parallel** delta-rule form (not a token scan):

- Per chunk C: cumsum the per-channel log-decay `g`, fold it into q/k by
  inflate/deflate so every intra-chunk interaction is a plain GEMM; build the
  strictly-lower delta-correction matrix `A` (C×C), invert `(I−A)` with a
  **log-depth Neumann product** (`_unit_lower_triangular_inverse`), form the
  pseudo-values `U` and decayed-key summary `k_cumdecay`.
- A single `lax.scan` over the `L/C` chunks carries the fp32 state
  `S ∈ R^{d_k×d_v}`; each step does the intra-chunk attention plus three
  state-touching GEMMs (`v_prime`, `inter`, state update `add`).

All math was fp32. Shapes benchmarked: B=4, H=8, L=8192, d_k=d_v=128, C=64.

## H100 measurements (cw-us-east-02a, 1×H100, jax 0.11.1)

FLOPs from an analytic GEMM model — **`cost_analysis` undercounts `lax.scan`
bodies** (returned 12G for C=64 vs ~137G analytic), so it is not used for MFU.

Shipped kernel (L=8192, B=4, H=8, d=128), the two viable chunk sizes:

| config | fwd ms | fwd tok/s | fwd+bwd ms | fwd+bwd tok/s |
|---|---|---|---|---|
| fp32 (TF32), C=64 (old baseline) | 5.95 | 44.1M | 19.79 | 13.3M |
| fp32 true (`precision=highest`), C=64 | 7.75 | 33.8M | 23.80 | 11.0M |
| bf16, C=64 | 5.52 | 47.5M | 17.46 | 15.0M |
| fp32 (TF32), C=128 | 6.32 | 41.5M | 19.61 | 13.4M |
| **bf16, C=128 (new default)** | **4.89** | **53.6M** | **14.72** | **17.8M** |

Two independent levers stack:
- **bf16 intra-chunk GEMMs** alone (at C=64): +7.8% fwd, +13.3% fwd+bwd.
- **bf16 shifts the optimal chunk 64→128** (larger intra-chunk GEMMs are cheap on
  tensor cores, and C=128 halves the sequential scan steps to L/128=64). In fp32
  C=64 was best; in bf16 C=128 wins.
- **Combined (bf16 + C=128) vs the old fp32/C=64 baseline: +21.6% fwd, +34.3%
  fwd+bwd** tokens/s. bf16-C=128 accuracy vs the recurrence is ~5.8e-3, same as C=64.

## Bottleneck (the headline)

The kernel achieves only ~11–12 GEMM-TFLOP/s = **~2.4% of the H100 TF32 peak
(494 TF/s), ~1.2% of bf16 peak (989 TF/s)**. It is **not compute-bound**. Two
observations pin it down:

1. `precision=highest` (true fp32) is 31% slower than default → the default
   already uses **TF32 tensor cores**, i.e. precision was never the main lever.
2. Even in TF32 the kernel sits at ~2.4% of peak → it is **latency / kernel-launch
   / memory-bandwidth bound** by the sequential inter-chunk `lax.scan`: L/C = 128
   sequential steps, each launching several *tiny* GEMMs (C=64, d=128). The
   matmuls are far too small to saturate Hopper tensor cores, and the scan
   serializes them.

## Change landed

Two defaults changed in `chunk_kda`:

- `matmul_dtype=jnp.bfloat16`: the **intra-chunk** GEMMs (delta-correction `A`, its
  Neumann inverse, `U`/`k_cumdecay`, and the intra-chunk attention) run in bf16 on
  tensor cores; the **cross-chunk fp32 state recurrence and all decay/cumsum math
  stay fp32** (stability). Pass `matmul_dtype=jnp.float32` for the exact-fp32 oracle.
- `chunk_size=128` (was 64): the H100 optimum under bf16. C=64 remains a
  lower-memory fallback (the reverse pass saves ~2x less C×C state), so the model's
  `kda_chunk_size` can drop it back for very long context if memory-bound.

Combined win over the old fp32/C=64 kernel: **+21.6% fwd, +34.3% fwd+bwd** tokens/s.

Correctness: `test_kda.py` — exact-fp32 parity pinned via `matmul_dtype=float32`
(rtol 1e-4 vs `recurrent_kda`; scalar-limit exactly matches levanter's
HF-validated `recurrent_gated_delta_rule`); bf16 tests at rtol 2e-2 for C=32/64/128;
gradients and strong-decay finiteness on the bf16 default. 14 tests pass.

## Remaining bottleneck & next steps (highest leverage first)

1. **Fuse the inter-chunk scan into one Pallas/Mosaic-GPU (Hopper) kernel.** The
   ~128 sequential scan steps × several tiny GEMMs are the dominant cost. A single
   kernel that keeps `S` in registers/SMEM across chunks and issues wgmma for the
   intra-chunk blocks (the fla `chunk_kda`/`chunk_gated_delta_rule` Triton kernels
   do exactly this) would move utilization from ~2% toward tens of %. Largest win,
   largest effort. Use the repo `add-pallas-kernel` skill.
2. **Grow the per-op work so tensor cores fill.** Increase the batch folded into
   each GEMM (process more (B·H) per step) and/or raise d — but d is fixed by the
   architecture (128). Cheap to try: confirm the scan GEMMs are emitted as batched
   `dot_general` over B·H (they are) and that XLA is not padding C=64 poorly.
3. **Cut scan-carried state traffic.** `S` is d_k×d_v=128×128 fp32 per (B,H); the
   `add`/`inter`/`v_prime` GEMMs re-read it every step. A bf16 *shadow* of `S`
   for the read-side GEMMs (keeping the fp32 accumulator) is a candidate — needs a
   stability check before enabling.
4. Re-confirm C=64 once (1)–(3) change the compute/latency balance.

## How to run the benchmark

```bash
# from a worktree off origin/main containing kda.py + bench_kda.py
NO_PROXY="*" uv run iris --cluster marin job run --no-wait \
  --enable-extra-resources --target-cluster cw-us-east-02a \
  --gpu H100x1 --memory 64GB --cpu 8 --extra gpu \
  -e KDA_FULL 1 -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.92 \
  --job-name kda-bench-h100 -- python -m experiments.grug.moe.bench_kda
```

Retrieval note: finelog does not serve federated peer-cluster (cw-us-east-02a)
logs from the marin controller after termination, so the bench prints its results
table then `sys.exit(3)`; read it back from the captured stdout tail via
`iris --cluster marin job summary /<user>/<job> --json` (`error` / `tasks[].error`
field). Set `--memory` ≥ 48GB (the numpy inputs + XLA host compile OOM the 1GB
default).
