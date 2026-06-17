🤖

## TL;DR

Enable **FP8 quantized training of the Grug MoE LM on NVIDIA H100** (Hopper, per-tensor delayed
scaling), flag-gated so default behavior is unchanged. Sibling to the B200 FP8 work (#5816) but on
**Hopper** with a **per-tensor E4M3/E5M2** recipe, and the throughput lever for the BF16 H100 MFU push
in #6367. Target: lift training MFU from the BF16 GPU baseline (~15%) toward **20–25%**. Status:
kickoff — reproducing the BF16 baseline and running the Phase-1 lowering-path spike. No FP8 speedup
claim yet.

## Description

The team is bringing up a 5B-active Grug MoE on H100 (first GPU run, first MoE arch, first data mix;
ref: Russell's 90B-total/5.3B-active bring-up at ~3% MFU on `cw-us-east-02a`, and the d2560 "May
Recipe" reference run in #6044). FP8 is on the critical path for that run's throughput, not a side
experiment.

This issue tracks enabling FP8 quantized training in the Grug MoE path. The recipe is
Transformer-Engine-style **per-tensor delayed scaling**: **E4M3** for forward activations/weights,
**E5M2** for output gradients, **FP32 accumulation**. We quantize the attention Q/K/V/O projections,
the dense/shared MLP, and the routed-MoE expert GEMMs, and keep router logits, top-k, softmax, norm,
QB beta, attention-score softmax, and loss in full precision. Everything is gated behind
`GrugFp8Config(enabled=False)` so the default path is byte-for-byte unchanged.

Work spans three packages, respecting dependency direction `haliax → levanter → experiments/grug`:
- `lib/haliax/.../fp8_dot.py` — `Fp8Dot`, `fp8_einsum`; `nn/ragged_dot.py` — `precision` arg +
  `QuantizedRaggedDot`/`fp8_ragged_dot`. (Haliax already has a dense FP8 impl — `Fp8DotGeneralOp` +
  `OverwriteWithGradient`; the gaps are Grug adoption, MoE/ragged FP8, and *verified* H100 kernel use.)
- `lib/levanter/.../grug/grug_moe.py` — `MoEExpertMlp.quantized_w13/quantized_w2`.
- `experiments/grug/moe/model.py` — `GrugFp8Config`, `GrugModelConfig.fp8`, model fields; `train.py`
  train-step contract (`OverwriteWithGradient` plumbing for scale/amax-history updates).

## Hypothesis or Goal

A per-tensor delayed-scaling FP8 recipe can be lowered to genuine H100 FP8 tensor-core kernels for
both the dense projections and the routed-expert GEMMs — verified in HLO + profiler counters, by
whichever custom call the selected path actually emits (`__cublas$lt$matmul$f8` for dense,
`__cudnn$blockScaledDot`, the relevant ragged-dot FP8 kernel for experts, or another) — is NaN-free on
real data, and yields a measured per-step / MFU win over the pinned BF16 baseline — targeting
~15% → 20–25% MFU.

**Phase 1 decides the lowering path on HLO + profiler evidence (do not assume A/B/C). Dense and MoE
may pick different paths.**
- Dense candidates: **A** delayed-scaling Q/DQ around `dot_general`; **B** Qwix `dot_general_qt`;
  **C** `jax.nn.scaled_dot_general`.
- MoE candidates: Pallas-Triton ragged-dot, XLA `ragged_dot_general(precision=?)`, Qwix
  `ragged_dot_qt`. Prior: dense `Fp8Dot` presumed wrong for ragged until proven otherwise.

### Links

* Spec / design / research: `.agents/projects/grug_fp8_h100/{spec,design,research}.md` (branch
  `codex/grug-fp8-h100-spec`)
* Research logbook: `.agents/logbooks/grug-fp8-h100.md`
* BF16 H100 MFU baseline (the number FP8 builds on): #6367
* Sibling FP8 work on B200/Blackwell: #5816
* Reference recipe (May Recipe, d2560 256-expert): #6044 (comment 4607416665)
* Grug GPU MoE profile epic: #5608

## Results

Pending. Report **dense and MoE separately**, each against the same BF16 baseline shape/config.

## Scope

In: Hopper per-tensor E4M3/E5M2 FP8 for attn Q/K/V/O, dense/shared MLP, routed-expert GEMMs;
delayed scaling w/ amax history; `OverwriteWithGradient` train-step plumbing; FP8 state checkpointing;
H100 profiles (dense + expert GEMMs); BF16-vs-FP8 benchmark.

Out: Blackwell MXFP8 / blockwise grouped-MoE (Hopper blockwise = stretch), FP8 optimizer states,
FP8 router/softmax/loss/norm, portable checkpoint export, Grug→Haliax-module rewrite, full TE API.

## Decision log

* 2026-06-17: Open a dedicated Hopper/H100 per-tensor FP8 issue (distinct from #5816 B200/Blackwell
  and #6367 BF16-MFU). Initial recipe = per-tensor delayed scaling; Phase 1 may redirect to a
  different lowering path on evidence.

## Negative results index

* (none yet)

## Current baseline

* BF16 Grug MoE, `params=fp32 / compute=bf16 / output=bf16`. FP8 benchmark shapes: dense MLP at
  **hidden dim 3072 & 4096**, seq 4096 (d3072 matches Russell's 90B-scale H100 run). Reference MFU:
  ~15% BF16 → 20–25% FP8 target. Concrete baseline numbers pending R4.
