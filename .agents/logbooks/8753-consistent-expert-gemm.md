---
topic: 8753-consistent-expert-gemm
description: Compare consistent QuACK and cuDNN expert-GEMM kernels on the ragged all-to-all EP hero path.
author: mcwitt
---

# Consistent Expert GEMM Kernels: Task Logbook

## Scope

- Goal: Starting from PR #8753, determine whether the ragged all-to-all expert MLP can use QuACK or cuDNN consistently across forward and backward while preserving full hero shape and checkpoint restore, loss fidelity, and at least control MFU without materially increasing complexity.
- Primary metrics: mean MFU over restored steps 5-19; train loss and routing-drop parity against a matched control; successful one-rack EP64 execution from the same complete production hero checkpoint.
- Constraints: BF16 expert math with FP32 accumulation as in control; no added quantization; one GB200 NVL72 rack per full-shape arm; no Iris cluster lifecycle changes.
- Coordinating PR: [#8753](https://github.com/marin-community/marin/pull/8753)
- Experiment prefix: `CEG`.

## Current TL;DR

- PR #8753 is not a fidelity-valid control as written: its cuDNN Wgrad adapter pads expert groups to 8 rows, but cuDNN Frontend 1.27.0 requires 256-row group alignment. The violation silently corrupts `dw13` and `dw2`. The matched control must apply the 256-row correction from [PR #8793](https://github.com/marin-community/marin/pull/8793) to both control and treatment.
- The existing backend is not simply “QuACK forward, cuDNN backward.” QuACK runs four of six grouped GEMMs: the fused gate/up and down projections in forward, plus `dh` and `dx` in backward. Only the two weight gradients run on cuDNN. QuACK consistency therefore requires one additional varlen-K wrapper and replacing two calls; cuDNN consistency requires JAX adapters for grouped GLU, grouped GEMM, grouped dGLU, and Wgrad plus 256-row layout conversion across the activation path.
- The exact-shape single-GB200 sweep favors tuned QuACK varlen-K: `dw13` is 7.23 ms versus corrected cuDNN's 10.15 ms, and `dw2` is 3.33 ms versus 4.68 ms. The chosen shared QuACK setting is a 256x256 tile, 2x2x1 cluster, and CLC off. Full-rack validation remains open.

## Baseline

- Date: 2026-08-30.
- Code ref: PR #8753 head `04026e94fba29f6c852bdb89ea65b0b6f12c57ab`; corrected-control prerequisite `5be7ff19e21a82ee696ab88a0ac251b97c01d2ab` / PR #8793.
- Published PR #8753 baseline: 23.50 MFU at the 6k restore measurement point, measured over steps 5-19 on one NVL72 rack. This number predates discovery of the cuDNN Wgrad contract violation and must be re-established on the corrected stack.

## Hypothesis Queue

### Active

- `CEG-H1`: QuACK varlen-K grouped GEMM can compute `dw13` and `dw2` directly from the transport layout, eliminate cuDNN's two multi-GB 256-alignment copies, match BF16/FP32-accumulation gradients, and meet or exceed corrected-control MFU. Evidence: exact-shape kernel gate passed at 1.40x/1.41x the corrected cuDNN kernel-only speed. Next test: matched restored EP64 draws.
- `CEG-H2`: A cuDNN-only activation and gradient path is technically expressible using the pinned Frontend's grouped GLU, unfused grouped GEMM, dGLU, and Wgrad kernels. Next test: estimate and then measure adapter/layout cost at the two hero per-call shapes; promote to an EP64 arm only if the microbenchmark can plausibly match control and the adapter stays bounded.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Background Research Brief

- Effort / stop rule / date: high; stop when internal history, pinned kernel source, and an adversarial alternative-kernel pass no longer change the ranked experiments; 2026-08-30.

### Question

Why does the ragged EP expert MLP mix QuACK and cuDNN, and can either family cover all six grouped GEMMs without fidelity, shape, speed, or complexity regressions?

### Current Marin Context

The local `sonic_cute.py` custom VJP performs gate/up+SwiGLU and down projection in forward on QuACK. Backward uses QuACK for the two activation-path transposed contractions (`dh`, `dx`), JAX for the pointwise SwiGLU derivative, and cuDNN Frontend only for `dw2` and `dw13`. The original QuACK shim exposed variable group lengths only along M; weight gradients group the contraction dimension K, so cuDNN's purpose-built grouped Wgrad adapter was the available fast path.

### Internal Prior Work

- [PR #8549](https://github.com/marin-community/marin/pull/8549) records that QuACK activation-path GEMMs plus cuDNN Wgrad were the compute core of the earlier 16.1% to 21.9% MFU stack. It also records that full-shape execution requires two expert chunks because one chunk uses 192-194 GiB against 184 GiB HBM.
- [Issue #8077](https://github.com/marin-community/marin/issues/8077#issuecomment-5244502055) measured the cuDNN Wgrad custom call at the production shapes and predicted a 0.698 s/step saving versus the then-Pallas path. That evidence explains why cuDNN was selected for the two weight gradients; it does not compare against QuACK varlen-K.
- [Issue #8339](https://github.com/marin-community/marin/issues/8339#issuecomment-5318708904) corrected the integration contract: every group extent must be divisible by 256 and the final offset must equal the total row count. The old 8-row wrapper was outside contract and silently wrong.
- [PR #8793](https://github.com/marin-community/marin/pull/8793) validates the 256-row fix through the four-GPU ragged EP gradient gate: relative gradient error falls from 0.0229-0.0301 to 0.000443-0.000807, matching the ring reference range 0.000532-0.000738.
- [Echo incident 101](https://echo.oa.dev/wiki/101) attributes the remaining ragged-EP performance gap after transport tuning to expert compute and records that QuACK covers forward, down, activation-gradient, and input-gradient contractions; the two residual weight gradients previously consumed 2.084 s/step on Pallas.

### External Prior Art

- [QuACK's official GEMM implementation](https://github.com/Dao-AILab/quack/blob/main/quack/gemm.py) supports either `cu_seqlens_m` or `cu_seqlens_k`. The latter directly expresses per-expert weight gradients over contiguous ragged rows, so the limitation is Marin's vendored JAX shim rather than the SM100 kernel.
- [NVIDIA cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) ships grouped GLU, unfused grouped GEMM, grouped dGLU, and grouped Wgrad kernels for BF16 MoE training on Blackwell. The pinned 1.27.0 source in the environment requires 256-aligned grouped-M buffers for these BF16 paths, making layout conversion a shared cost rather than a Wgrad-only detail.

### Negative / Failed Leads

- Treating PR #8753 unchanged as the control is invalid because the discovered cuDNN contract violation affects every ragged training step.
- Treating the current backend as a clean forward/backward family split is inaccurate: backward is already mixed internally, with QuACK doing two contractions and cuDNN doing two.

### Evidence Map

#### Claim: QuACK-only is a small implementation change with a credible performance upside.

- Support: pinned QuACK supports varlen-K; the new path can consume existing cumulative group sizes without padding or copying; only the two Wgrad calls change.
- Contradictions: no Marin end-to-end measurement yet; the varlen-K tile choice may be slower than cuDNN's kernel even after copy overhead.
- Directness to Marin: exact pinned package and hero shapes.
- Confidence: medium pending accelerator results.
- Action: run the single-GPU correctness/tile benchmark, then matched EP64 control/treatment.

#### Claim: cuDNN-only is technically possible but unlikely to simplify or win without new adapter work.

- Support: the pinned package contains all required grouped kernels.
- Contradictions: every BF16 activation-path kernel requires a 256-aligned grouped-M layout, and Marin has no JAX adapters for three of the four required cuDNN kernel classes.
- Directness to Marin: exact pinned package; performance inference remains unmeasured.
- Confidence: medium on feasibility, low on competitiveness.
- Action: build the smallest representative microbenchmark before considering a rack arm.

### Recommended Next Experiments

#### 1. QuACK varlen-K Wgrad gate

- Minimum experiment / baseline: exact hero per-call `dw13` and `dw2` shapes on one GB200; compare QuACK varlen-K against corrected cuDNN+padding and an FP32 per-expert oracle.
- Expected signal / falsifier: gradient error at the BF16 rounding floor and lower combined wall time; any out-of-group read, unsupported empty group, or slower combined time falsifies promotion.
- Cost or risk / sources: one GPU, bounded; QuACK official varlen-K implementation and PR #8793 correctness gate.

#### 2. Corrected full-shape control and QuACK-only treatment

- Minimum experiment / baseline: same complete hero checkpoint, one NVL72 rack, same seed/config, restored steps scored over the PR #8753 window.
- Expected signal / falsifier: treatment MFU at least control, loss/drop parity within normal GPU nondeterminism, and successful full-shape restore.
- Cost or risk / sources: two bounded rack draws; PR #8753 methodology.

#### 3. cuDNN-only representative gate

- Minimum experiment / baseline: fused grouped GLU forward plus unfused down, dGLU, and Wgrad over the exact per-call layout, including 256-row conversion costs.
- Expected signal / falsifier: end-to-end kernel-family time competitive with the current QuACK activation path and an adapter no larger or more complex than the existing QuACK shim. A material copy/HBM or code-size penalty stops the arm before rack scale.
- Cost or risk / sources: one GPU; pinned cuDNN Frontend 1.27.0 source.

### Hypothesis Queue Update

`CEG-H1` ranks first because it removes the only second-family calls with one native QuACK feature and no new layout. `CEG-H2` remains active but is gated on a microbenchmark because “consistent cuDNN” currently adds adapters and alignment transforms rather than removing them.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| PR #8753 | PR | https://github.com/marin-community/marin/pull/8753 | Baseline methodology and reported MFU | High | Baseline must be rerun after Wgrad correction. |
| PR #8549 | PR | https://github.com/marin-community/marin/pull/8549 | Mixed-kernel origin and full-shape chunking constraint | High | Direct implementation history. |
| Issue #8077 result | issue | https://github.com/marin-community/marin/issues/8077#issuecomment-5244502055 | Why cuDNN Wgrad was chosen | High | Exact GB200 production shapes. |
| Issue #8339 correction | issue | https://github.com/marin-community/marin/issues/8339#issuecomment-5318708904 | 256-row cuDNN contract | High | Direct pinned-kernel investigation. |
| PR #8793 | PR | https://github.com/marin-community/marin/pull/8793 | Corrected-control gradient evidence | High | Four-GPU ragged gate. |
| Echo wiki 101 | report | https://echo.oa.dev/wiki/101 | Transport/compute decomposition | High | Durable incident record. |
| QuACK GEMM | external code | https://github.com/Dao-AILab/quack/blob/main/quack/gemm.py | Native varlen-K capability | High | Official upstream; pinned local 0.6.1 source checked separately. |
| cuDNN Frontend | external code | https://github.com/NVIDIA/cudnn-frontend | Available grouped forward/backward families | High | Official upstream; pinned 1.27.0 source checked separately. |

### Handoff

- No coordinating issue was created; PR #8753 is the source reference and this logbook is the durable research record.
- Open questions: exact latest complete hero checkpoint, matched rack launch identities, and measured QuACK varlen-K/corrected-cuDNN times.

## Entry Log

### 2026-08-30 23:35 - CEG-001: Forage and control correction

- Hypothesis: one kernel family should reduce layout work and conceptual surface without losing fidelity or throughput.
- Commit Hash: `04026e94fba29f6c852bdb89ea65b0b6f12c57ab` (PR #8753 source).
- Command: `gh pr view 8753 --repo marin-community/marin --json ...`; four Echo searches covering ragged EP, QuACK, cuDNN Wgrad, and issue #8339; local inspection of `sonic_cute.py`, `quack_moe_cute.py`, `cudnn_wgrad_cute.py`, and pinned package source.
- Config: read-only investigation; no accelerator job submitted.
- Result: current backward already uses QuACK for `dh`/`dx`; cuDNN is limited to `dw13`/`dw2`. The PR-head cuDNN wrapper violates its 256-row contract, so corrected control is required. QuACK varlen-K is the smallest complete-family candidate; cuDNN-only is possible but needs materially more adapter/layout work unless measurements justify it.
- Interpretation: promote the QuACK-only gate first. Do not spend a full rack on cuDNN-only until its exact-shape microbenchmark clears both time and complexity gates.
- Next action: validate the QuACK varlen-K implementation against the corrected cuDNN baseline on GB200, then prepare matched one-rack restore draws.

### 2026-08-31 06:45 - CEG-002: QuACK varlen-K gate and restored-run contract

- Hypothesis: QuACK's native varlen-K mode removes both the cuDNN Wgrad kernel and its alignment copies while preserving the BF16/FP32-accumulation calculation.
- Commit Hash: `c4724656e7` for the measured tile; corrected control `9da4d79c02`.
- Command: `python lib/levanter/scripts/bench/bench_grouped_wgrad.py --sweep`; `uv run fsutil cat|ls|du s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-42000`; targeted pytest and repository pre-commit checks.
- Config: one GB200; hero per-call rows=301466, experts=3, latent/intermediate=3072; QuACK sweep over 128/256 tiles, 2x1/2x2 clusters, and CLC. Full-run source is the permanent step-42000 checkpoint, schedule length 4470000, absolute stop 42030, EP64 on 16 four-GPU workers, interactive priority, no checkpoints or eval.
- Result: QuACK 256x256/2x2x1/CLC-off runs `dw13` at 7.23 ms and `dw2` at 3.33 ms; corrected cuDNN kernel-only takes 10.15 and 4.68 ms before alignment-copy time. Checkpoint metadata reports step 42000, `is_temporary=false`, timestamp 2026-08-31T03:06:00.923412; `manifest.json` and `manifest.ocdbt` are present; size is 5,363,757,878,827 bytes across 45,329 objects. Focused tests pass and the diff-scoped lint/type suite passes.
- Interpretation: the QuACK treatment clears correctness, performance, and full-shape-input preconditions for rack testing. cuDNN-only remains behind its complexity gate: the pinned APIs require three new JAX kernel adapters, a distinct 32-column gate/up layout VJP, and 256-row activation-path padding.
- Next action: snapshot and push the clean tree, then submit serialized C-T-C-T one-rack draws with fresh identities and compilation caches.
