# Issue #7279: latent MoE expert-parallel throughput

Append-only research log for the single-rack GB200 baseline and latent-MoE
matrix requested in [#7279](https://github.com/marin-community/marin/issues/7279).

## 2026-07-20 15:15 PDT - Matrix selected

- Reference: [#7201 comment 5016389653](https://github.com/marin-community/marin/issues/7201#issuecomment-5016389653).
- Fixed architecture: 64 GB200 GPUs, replica axis 2, batch 1,024, sequence
  length 4,096, D=6,144, 48 layers, 128 routed experts, top-4, routed
  intermediate 3,072, shared intermediate 6,144, 48 query heads, 8 KV heads,
  sliding window 512, and one global layer every six layers.
- Baselines: `sonic_cute` EP1 diagnostic, `ring_cute` EP4, `ring_cute` EP8,
  and `ragged_all_to_all_cute` EP8.
- Matched-work latent treatment: L=3,072, routed intermediate 6,144, latent
  RMSNorm, router and shared expert unchanged at D=6,144. This halves the
  dispatched activation width while preserving routed-expert parameters and
  active routed-expert GEMM work.
- Deferred efficient arm: L=3,072, 256 experts, top-4, intermediate 3,072 on
  the fastest backend after the matched-work matrix.

## 2026-07-20 16:05 PDT - Standalone harness ready for accelerator smoke

- Added explicit query/KV head, expert/shared intermediate, sliding-window,
  global-layer-frequency, and latent-MoE flags.
- Ported the validated shared D-to-L projection, latent RMSNorm, routed expert
  path, and L-to-D projection. The no-latent RNG split remains unchanged.
- Analytic FLOPs now replace the baseline routed-expert term with the latent
  expert term and add both projections. Each metrics row reports GB200 MFU using
  2.5 PFLOP/s/GPU and retains the historical B200-convention field.
- Removed two unused optimizer-choice registrations from the standalone module.
  They collided with production optimizer registration when the harness and
  Grug contract tests shared a process; the standalone constructs its optimizer
  directly and does not deserialize through the registry.
- Verification: 19 passed, 1 GPU-only skipped, 1 unrelated existing TensorStore
  checkpoint test deselected. That test reproducibly times out at 60 seconds in
  `test_grug_variant_contracts.py` alone while awaiting async checkpoint
  serialization.
- Required lint wrapper: Markdown, AST, conflict, whitespace, license, Levanter
  Ruff, and formatting checks pass after autoformat. Branch-pre-existing failures
  remain: 72 Ruff findings in the generated/inlined standalone source and seven
  Pyrefly missing-import errors for Blackwell-only `cutlass`/`quack` packages.

## Attempts

| Arm | Iris job | Placement | Commit | State | Median tok/s | Median step | GB200 MFU | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| Smoke baseline, ring_cute EP4 | `/mwittmann/b200lmoe-smoke-baseline` | 1 x GB200x4 | `50fa034cd` | succeeded | 71,557 | 0.0804 s | 0.307% | Three-step reduced-model compile/execute smoke; cold step took 518 s. |
| Smoke latent, ring_cute EP4 | `/mwittmann/b200lmoe-smoke-latent` | 1 x GB200x4 | `50fa034cd` | succeeded | 71,337 | 0.0816 s | 0.308% | Matched-work reduced model with L=256 and latent RMSNorm; cold step took 512 s. |
| Initial D6144 matrix, all eight arms | `/mwittmann/b200lmoe-001-a` through `002-d` | 16 x GB200x4, replica axis 2 | `50fa034cd` | failed | — | — | — | Deterministic first-step OOM. XLA planned 240–571 GiB/GPU; this incorrectly reused the d5120 #7279 replica-2 topology and duplicated the 360B model. Historical #7201 logs and the later `mx7201-b64g2` reproduction confirm D6144 used replica axis 1. |

## 2026-07-20 16:47 PDT - Corrected #7201 topology

- The #7201 D6144 run ID's `r1` denotes replica axis 1. It shards one model over
  all 64 GPUs while retaining 16 sequences per GPU because the product of the
  replica, data, and expert batch-sharding axes remains 64.
- Historical #7201 XLA logs planned 148.90 GiB/GPU and completed all 30 train
  steps. The incorrect replica-2 standalone arms planned 239.83 GiB/GPU at
  ring EP4 and 571.05 GiB/GPU at EP1, then OOMed exactly as predicted.
- Retry policy: validate the corrected full-size ring EP4 baseline/latent pair
  first, then launch the remaining six corrected arms if both fit.

## 2026-07-20 17:19 PDT - Corrected baseline and latent feasibility

| Arm | Iris job | Placement | Commit | State | Median tok/s | Median step | GB200 MFU | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| Baseline, ring_cute EP4 | `/mwittmann/b200lmoe-001-b-r1` | 16 x GB200x4, replica axis 1 | `50fa034cd` | succeeded | 220,648 | 19.0117 s | 19.686% | XLA plan 150.82 GiB/GPU. Historical B200-denominator MFU: 21.873%. |
| Matched-work latent L3072/I6144, ring_cute EP4 | `/mwittmann/b200lmoe-002-b-r1` | 16 x GB200x4, replica axis 1 | `50fa034cd` | failed | — | — | — | XLA plan 181.34 GiB/GPU; first execution failed with CUDA/NCCL OOM. |
| Iso-work latent L2048/I9216, ring_cute EP4 | `/mwittmann/b200lmoe-002-b-l2048-r1` | 16 x GB200x4, replica axis 1 | `50fa034cd` | failed | — | — | — | XLA plan 185.95 GiB/GPU; narrowing L while widening I increased rectangular expert temporaries and worsened HBM. |
| Baseline, ragged_all_to_all_cute EP8 | `/mwittmann/b200lmoe-001-d-r1` | 16 x GB200x4, replica axis 1 | `50fa034cd` | failed | — | — | — | XLA plan 154.75 GiB/GPU, then `CUDA_ERROR_INVALID_VALUE` while loading the in-memory CUBIN. This is code generation/loading, not HBM or the one-shot all-to-all flag. |

The exact matched-work construction is not viable for this 360B model under
the current optimizer: it retains the full baseline routed-expert parameter
budget and adds two dense projections per layer. The narrower L2048 variant
does not solve that because preserving active work requires I9216. The next
feasibility arm therefore keeps L3072 and I3072 with 128 experts. It halves the
EP payload, routed-expert parameters, and active routed-expert work; the
architecture-aware MFU calculation accounts for the reduced work. Its ring EP4
compiler plan is 134.62 GiB/GPU, leaving about 16.2 GiB more margin than the
successful baseline.

## 2026-07-20 17:25 PDT - Corrected baseline readout

All steady medians exclude eight warmup steps and use the architecture-aware
47,582,281,728 forward FLOPs/token. GB200 MFU uses 2.5 PFLOP/s/GPU.

| Backend | EP | Iris job | State | Median tok/s | Median step | GB200 MFU | Historical B200 MFU |
|---|---:|---|---|---:|---:|---:|---:|
| sonic_cute | 1 | `/mwittmann/b200lmoe-001-a-r1` | succeeded | 188,998 | 22.2032 s | 16.862% | 18.735% |
| ring_cute | 4 | `/mwittmann/b200lmoe-001-b-r1` | succeeded | **220,648** | **19.0117 s** | **19.686%** | **21.873%** |
| ring_cute | 8 | `/mwittmann/b200lmoe-001-c-r1` | succeeded | 204,117 | 20.5513 s | 18.211% | 20.234% |
| ragged_all_to_all_cute | 8 | `/mwittmann/b200lmoe-001-d-r1` | CUBIN load failure | — | — | — | — |

Ring EP4 is the new best baseline: +16.7% throughput over EP1 and +8.1% over
ring EP8. The initial all-to-all EP8 attempt is being retried once because the
CUBIN loader fault is intermittent and independent of the backend result.

## 2026-07-20 17:36 PDT - Efficient latent EP1 result

The viable arm uses L3072, I3072, 128 routed experts, top-4, latent RMSNorm,
and the unchanged D6144 router/shared expert. It halves dispatched width,
routed-expert parameters, and active routed-expert work relative to the #7201
baseline.

| Backend | EP | Iris job | State | Median tok/s | Median step | GB200 MFU | FLOPs/token |
|---|---:|---|---|---:|---:|---:|---:|
| sonic_cute | 1 | `/mwittmann/b200lmoe-003-a-e128-r1` | succeeded | **269,630** | **15.6722 s** | **20.391%** | 40,334,524,416 |

Against the EP1 baseline this is +42.7% throughput, -29.4% step time, and
+3.53 percentage points of architecture-aware GB200 MFU. Analytic work/token
falls only 15.2%, so the remaining gain reflects substantially better execution
efficiency rather than FLOP removal alone.

The first ring EP4 and both first EP8 latent launches reached first execution
with viable XLA plans but failed in the intermittent #7421 CUBIN loader. Ring
EP4 planned 134.62 GiB/GPU, ring EP8 147.49 GiB/GPU, and all-to-all EP8
140.07 GiB/GPU. One clean retry of each EP>1 arm is the terminal retry policy.

## 2026-07-20 17:49 PDT - Final latent result and recommendation

| Configuration | Backend | EP | Iris job | State | Median tok/s | Median step | GB200 MFU | XLA plan |
|---|---|---:|---|---|---:|---:|---:|---:|
| Reduced latent L3072/I3072/e128 | sonic_cute | 1 | `/mwittmann/b200lmoe-003-a-e128-r1` | succeeded | **269,630** | **15.6722 s** | **20.391%** | not emitted |
| Reduced latent L3072/I3072/e128 | ring_cute | 4 | `/mwittmann/b200lmoe-003-b-e128-r1`, `-r1b` | CUBIN load failure, 2/2 | — | — | — | 134.62 GiB |
| Reduced latent L3072/I3072/e128 | ring_cute | 8 | `/mwittmann/b200lmoe-003-c-e128-r1`, `-r1b` | CUBIN load failure, 2/2 | — | — | — | 147.49 GiB |
| Reduced latent L3072/I3072/e128 | ragged_all_to_all_cute | 8 | `/mwittmann/b200lmoe-003-d-e128-r1`, `-r1b` | CUBIN load failure, 2/2 | — | — | — | 140.07 GiB |
| Efficient latent L3072/I3072/e256 | ring_cute | 4 | `/mwittmann/b200lmoe-004-b-e256-r1` | succeeded | **256,059** | **16.3862 s** | **19.401%** | 172.86 GiB |

The e256 candidate is the recommended parameter-count-preserving latent
configuration. Relative to the best #7201 baseline, ring EP4, it improves
throughput by 16.0% and reduces step time by 13.8%. It preserves total routed
expert parameters while halving active routed-expert work and EP payload. Its
architecture-aware MFU is 0.28 percentage points lower after accounting for the
15.1% reduction in analytic work/token.

The e128 candidate is the throughput-first option if halving routed-expert
parameters is acceptable. Its EP1 result is 42.7% faster than the EP1 baseline
and improves architecture-aware MFU by 3.53 percentage points. The current data
cannot quantify its incremental EP benefit: all six EP>1 attempts failed before
the first step in the intermittent JAX 0.10.1 CUBIN loader tracked by #7421,
despite HBM plans below the successful baseline and successful EP1 execution.

The baseline all-to-all EP8 arm also failed at the same CUBIN load in both clean
allocations. The failure is therefore recorded as an infrastructure-blocked
measurement rather than a backend throughput result. All jobs launched by this
investigation are terminal; no Iris gang remains to babysit.
