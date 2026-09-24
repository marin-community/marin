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

| variant (L=8192, C=64) | fwd ms | fwd tok/s | fwd+bwd ms | fwd+bwd tok/s |
|---|---|---|---|---|
| fp32, true fp32 (`precision=highest`) | 7.80 | 33.6M | 23.79 | 11.0M |
| fp32, default (**TF32**) | 5.98 | 43.8M | 19.74 | 13.3M |
| **bf16 intra-chunk GEMMs (fp32 state)** | **5.56** | **47.2M** | **17.49** | **15.0M** |

Chunk-size sweep (fwd, fp32-default, B=8 earlier run): C=32 → 11.4ms, **C=64 →
9.87ms (best)**, C=128 → 11.5ms. C=64 is the H100 sweet spot.

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

`chunk_kda(..., matmul_dtype=jnp.bfloat16)` (new default): the **intra-chunk**
GEMMs (delta-correction `A`, its Neumann inverse, `U`/`k_cumdecay`, and the
intra-chunk attention) run in bf16 on tensor cores; the **cross-chunk fp32 state
recurrence and all decay/cumsum math stay fp32** (stability). Result: **+7.6% fwd,
+12.9% fwd+bwd** tokens/s, at ~0.5% max relative error vs the fp32 recurrence.
This is a real but modest win — consistent with the kernel being memory/latency
bound, where bf16 mainly cuts operand traffic rather than compute time.

Correctness: `test_kda.py` — exact-fp32 parity pinned via `matmul_dtype=float32`
(rtol 1e-4 vs `recurrent_kda`; scalar-limit exactly matches levanter's
HF-validated `recurrent_gated_delta_rule`); new bf16 test at rtol 2e-2; gradients
and strong-decay finiteness on the bf16 default. 13 tests pass.

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
