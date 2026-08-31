---
topic: 8753-consistent-expert-gemm
description: Compare consistent QuACK and cuDNN expert-GEMM kernels on the ragged all-to-all EP hero path.
author: mcwitt
---

# Consistent Expert GEMM Kernels: Task Logbook

## Scope

- Goal: Starting from PR #8753, determine whether the ragged all-to-all expert MLP can use QuACK or cuDNN consistently across forward and backward while preserving full hero shape and checkpoint restore, loss fidelity, and at least control MFU without materially increasing complexity.
- Primary metrics: median MFU over restored steps 5-19, with mean MFU and stall count as guards; train loss and routing-drop parity against a matched control; successful one-rack EP64 execution from the same complete production hero checkpoint.
- Constraints: BF16 expert math with FP32 accumulation as in control; no added quantization; one GB200 NVL72 rack per full-shape arm; no Iris cluster lifecycle changes.
- Coordinating PR: [#8753](https://github.com/marin-community/marin/pull/8753)
- Experiment prefix: `CEG`.

## Current TL;DR

- PR #8753 is not a fidelity-valid control as written: its cuDNN Wgrad adapter pads expert groups to 8 rows, but cuDNN Frontend 1.27.0 requires 256-row group alignment. The violation silently corrupts `dw13` and `dw2`. The matched control must apply the 256-row correction from [PR #8793](https://github.com/marin-community/marin/pull/8793) to both control and treatment.
- The existing backend is not simply “QuACK forward, cuDNN backward.” QuACK runs four of six grouped GEMMs: the fused gate/up and down projections in forward, plus `dh` and `dx` in backward. Only the two weight gradients run on cuDNN. QuACK consistency therefore requires one additional varlen-K wrapper and replacing two calls; cuDNN consistency requires JAX adapters for grouped GLU, grouped GEMM, grouped dGLU, and Wgrad plus 256-row layout conversion across the activation path.
- The exact-shape single-GB200 sweep favors tuned QuACK varlen-K: `dw13` is 7.23 ms versus corrected cuDNN's 10.15 ms, and `dw2` is 3.33 ms versus 4.68 ms. The chosen shared QuACK setting is a 256x256 tile, 2x2x1 cluster, and CLC off.
- Two matched full-shape EP64 control/treatment pairs restored the immutable production hero step-42000 checkpoint and scored the same steps 42005-42019. QuACK's two-draw median is 23.8154 MFU versus corrected cuDNN's 23.1203, a +0.6951 / +3.01% improvement. All four arms produced complete finite loss/drop series; pairwise maximum absolute loss deltas are 1.32e-4 and 1.04e-4, against 7.6e-5 control-to-control nondeterminism. No quantization or dtype policy changed.
- QuACK-only is viable and simpler in the production source slice: it removes the cuDNN wrapper, padding copies, runtime dependency probe, and family-specific private entry point. Relative to PR #8753, the runtime MoE source changes by +115/-174 lines (net -59). cuDNN-only is technically expressible but not viable under the complexity gate: it needs three additional direct-JAX kernel adapters, a different gate/up layout VJP, and 256-row activation-buffer conversion while computing the inactive receiver tail. It was therefore stopped before a rack arm.

## Baseline

- Date: 2026-08-30.
- Code ref: PR #8753 head `04026e94fba29f6c852bdb89ea65b0b6f12c57ab`; corrected-control prerequisite `5be7ff19e21a82ee696ab88a0ac251b97c01d2ab` / PR #8793.
- Published PR #8753 baseline: 23.50 MFU at the 6k restore measurement point, measured over steps 5-19 on one NVL72 rack. This number predates discovery of the cuDNN Wgrad contract violation and must be re-established on the corrected stack.

## Hypothesis Queue

### Active

- None.

### Blocked

- None.

### Falsified / Dead End

- `CEG-H2`: A cuDNN-only activation and gradient path is technically expressible, but fails the no-significant-complexity-increase gate before rack testing. The pinned Frontend requires three new JAX adapters beyond Wgrad, a distinct gate/up layout VJP, and 256-row layout conversion for the activation path. The conversion also forces work over the receiver buffer's roughly 15% inactive tail; a prior fidelity-correct aligned-transport arm lost 0.173 MFU. A rack arm cannot rescue the already-failed complexity requirement.

### Promoted

- `CEG-H1`: QuACK varlen-K computes both weight gradients directly from existing cumulative group sizes. Two restored EP64 pairs show +3.01% median MFU with loss/drop parity, and the final runtime source is 59 lines smaller than PR #8753.

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
- Open questions: none for the viability decision. A production PR should use the QuACK-only head and include the corrected-control caveat so PR #8753's uncorrected Wgrad result is not treated as a fidelity baseline.

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

### 2026-08-31 15:55 - CEG-003: Restored EP64 C-T-C-T verdict

- Hypothesis: replacing the remaining cuDNN `dw13`/`dw2` calls with tuned QuACK varlen-K will preserve the full hero computation while removing padding copies and improving MFU beyond run noise.
- Commit Hash: treatment `1bd73f6f26d1e64e0336c5abffea98395d619dd5`; corrected control `33abad46a2ca476b90db7fcc7a16b12ae461ad1d`. The final cleanup only removes the now-dead cuDNN source/probe and renames the private QuACK entry point; it does not change the measured JAX graph or kernel settings.
- Command: serialized `autoresearch/quack-wgrad/arm.sh` and `watchdog.sh` draws at production priority, each with `ARM_TIMEOUT=3600`, followed by `score.py <run-id> --relative --lo 5 --hi 19`. The first interactive submission `ceg-wg-c1-42000-20260831` was Kueue-gated, consumed no GPU time, and was canceled before the production-priority campaign; it is not a measurement.
- Config: 16 workers x 4 GB200 GPUs (canonical EP64 slice of one NVL72 rack); full hero d6144/48-layer/384-expert/top-8/latent-i3072 shape; global batch 1024; capacity factor 1.15; mixture data; device master parameters; ragged all-to-all; schedule length 4,470,000; immutable `s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-42000`; no checkpoint writes, eval, or profiling. Every coordinator completed in 17-22 minutes and was canceled immediately after its scoring window, within the authorized one-hour cap.
- Result:

| Arm | Kernel for `dw13`/`dw2` | Median MFU | Mean MFU | Stalls | Mean tok/s | Peak GiB | Max drop |
|---|---|---:|---:|---:|---:|---:|---:|
| [C1](https://wandb.ai/marin-community/marin_moe/runs/ceg-wg-c1p-42000-20260831) | corrected cuDNN | 23.0535 | 23.0327 | 0 | 246,114.6 | 116.57 | 0.00027344 |
| [T1](https://wandb.ai/marin-community/marin_moe/runs/ceg-wg-t1p-42000-20260831) | QuACK varlen-K | 23.8427 | 23.7480 | 0 | 253,758.3 | 115.86 | 0.00026883 |
| [C2](https://wandb.ai/marin-community/marin_moe/runs/ceg-wg-c2p-42000-20260831) | corrected cuDNN | 23.1870 | 23.0821 | 0 | 246,642.4 | 116.57 | 0.00027483 |
| [T2](https://wandb.ai/marin-community/marin_moe/runs/ceg-wg-t2p-42000-20260831) | QuACK varlen-K | 23.7880 | 23.6940 | 1 | 253,181.6 | 115.86 | 0.00026291 |

  The two-draw median centers are 23.1203 control and 23.8154 treatment: +0.6951 MFU / +3.01%. Mean MFU centers are 23.0574 and 23.7210: +0.6636 / +2.88%. The treatment uses 0.71 GiB less peak memory. T2 has one step below 97% of its run median, but its full-window mean and median remain above both controls. No task failure or unplanned preemption occurred during a score window. The deliberate post-window cancellation leaves each terminal 16-worker gang with `preemptions=16`; those counters are the release mechanism, not a measurement event.
- Fidelity: all four score windows contain 15/15 MFU, loss, and drop points and no non-finite values. C1-T1 and C2-T2 maximum absolute loss deltas are 1.32e-4 and 1.04e-4; their mean absolute deltas are 6.36e-5 and 4.86e-5. The two controls themselves differ by as much as 7.6e-5, so the treatment deltas are at the expected BF16 reduction-order/nondeterminism scale. Drop means are 8.2e-5/8.2e-5 and 8.3e-5/8.1e-5. Both treatment runs restored the checkpoint and reached the exact full hero score window without a dtype, quantization, optimizer, data, or schedule change.
- Interpretation: `CEG-H1` passes every viability gate. The performance gain is 3.9-4.3 times the 0.16-0.18 MFU keep threshold and repeats across both pairs. The loss/drop series support equivalence, the exact checkpoint restore proves full-shape compatibility, and deleting the cuDNN-only wrapper makes the runtime source smaller and conceptually single-family. `CEG-H2` is closed without a rack arm because it already fails the independent complexity gate and carries a layout/work penalty absent from QuACK.
- Next action: run final focused tests and repository lint after deleting the dead cuDNN path, snapshot the research head, and prepare the production change for review if requested.
