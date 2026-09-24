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

## Phase 2 — chunk-parallel inter-chunk recurrence (associative scan)

The decomposition (KDA_DECOMPOSE) showed the Neumann inverse is only ~25% of the
fwd time (1.22ms of 4.89ms at C=128, running at ~10% of bf16 peak); the rest is the
**sequential inter-chunk scan** (L/C steps) running at <1% of peak. `lax.scan(unroll=)`
barely helped fwd (~6%) — confirming it is *depth*, not per-step launch, that hurts.

The cross-chunk update is **linear in S**, so it was replaced with a log-depth
`lax.associative_scan` over the affine transforms `(M_n, C_n)`,
`M_n = Diag(decay_tail_n) − Kw_nᵀ k_cumdecay_n`, `C_n = Kw_nᵀ v_pseudo_n`; depth
drops from L/C to log2(L/C) and the per-chunk outputs then compute in one parallel
batched pass (`_parallel_chunk_recurrence`, default `scan_impl="parallel"`; the old
serial path stays as `scan_impl="sequential"`). All state math is fp32 (arXiv 2406.06484).

Measured (H100, L=8192, bf16, parallel vs sequential):

| | seq fwd | par fwd | seq fwd+bwd | par fwd+bwd |
|---|---|---|---|---|
| B=4 C=64 | 5.55ms | 4.80ms (+16%) | 17.39ms | 11.27ms (+54%) |
| B=4 C=128 | 4.90ms | 4.23ms (+16%) | 14.66ms | 10.35ms (+42%) |
| B=2 C=128 | 3.31ms | 2.23ms (+48%) | 10.07ms | 5.39ms (+87%) |
| B=4 C=256 | 5.88ms | 5.11ms (+15%) | — | — |

Hardware utilization rose from ~2.4% (original fp32) to **~4–5% at the wall-time
optimum (C=128) and 11.7% at C=256 fwd** (116 GEMM-TFLOP/s). Cumulative vs the
original fp32/seq/C=64 kernel: **fwd 5.95→4.23ms (1.41×)**, **fwd+bwd 19.79→10.35ms
(1.91×)** at the shipped default (C=128, parallel, bf16). Correctness:
parallel==sequential to ~3e-8 fp32, ==recurrence to ~1e-7 fp32 / 5.9e-3 bf16
(`test_parallel_scan_matches_sequential`, 18 tests pass).

**Memory tradeoff:** the associative scan materializes the d_k×d_k transition
matrices, so its footprint scales O(L/C · d_k²) (vs the serial scan's O(d_k²)).
C=128 fwd+bwd fits fine at B=4 on a bare 80GB H100 (10.35ms) when run alone — an
earlier "OOM" was a benchmark artifact (several chunk sizes compiled in one process
without freeing GPU memory). For very long context or large per-device batch, drop
to C=64 or `scan_impl="sequential"` (constant-memory), or wrap the layer in the
model's usual gradient checkpointing.

## Remaining bottleneck & next steps (highest leverage first)

1. **Fuse into one Pallas/Mosaic-GPU (Hopper) kernel** — still the ceiling-breaker.
   Even after the parallel scan the kernel is a chain of ~20 separate XLA ops with
   HBM round-trips between them, capping util at single-digit % on the wall-time
   optimum. A fused kernel that keeps chunk state resident and issues wgmma for the
   intra-chunk blocks (fla's Triton `chunk_kda`) is the path to tens-of-%.
2. **Cut the redundant Neumann FLOPs.** The log-depth inverse does 4·C³·log2(C) —
   ~log2(C)× more matmul work than a blocked triangular solve. At C=128 it is ~25%
   of fwd; a blocked inverse would reduce it (though it is more sequential).
3. **`jax.checkpoint` the parallel recurrence** to make C=128 fwd+bwd fit at larger
   batch (memory tradeoff above).
4. **bf16 shadow of S** for the read-side state GEMMs (fp32 accumulator) — stability
   check needed.

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
