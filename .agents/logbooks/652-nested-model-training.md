---
topic: nested-model-training
issue: Weaver issue #652
description: Co-train an extractable small MoE inside a larger expert bank.
author: Marin research
---

# Nested model training: task logbook

## Scope

- Goal: determine whether one MoE pretraining run can produce competitive large
  and small checkpoints for less than 10% overhead.
- Primary metrics: Paloma macro loss at fixed FLOPs; tokens/s; GPU-hours.
- Constraints: 128–256 GB200 GPUs, batch priority, deadline 2026-07-27 09:00 UTC.
- Coordinating issue: Weaver issue #652.
- Experiment prefix: `NEST-MOE`.

## Current TL;DR

- Google Gemma 3n is a shipped dense-model precedent: E2B was optimized inside
  E4B using MatFormer.
- The preregistered Marin test nests a fixed 128-expert prefix inside a
  256-expert bank and assigns whole sequences to prefix or full routes.
- Four arms compare conventional large, conventional small, 25%-prefix, and
  50%-prefix training.

## Hypothesis queue

### Active

- `NEST-MOE-H1`: 25%-prefix nesting preserves full-model loss, produces a
  near-control small prefix, and costs less than 10% throughput.
- `NEST-MOE-H2`: 50%-prefix training improves the extracted prefix at a larger
  full-model cost.
- `NEST-MOE-H3`: QB compensation prevents outer-expert undertraining.

### Blocked

- None.

### Falsified / dead end

- Full-width masking as a free small model: it retains the large GEMM and does
  not reproduce the small hidden trajectory.
- Random expert dropout as a breakout model: no fixed subset is guaranteed to
  be independently usable.

### Promoted

- None.

## Decision log

- 2026-07-26: test expert-bank nesting before hidden-width or depth nesting.
  It directly targets the 300B/700B total-parameter choice and permits one
  forward with no extra active experts.
- 2026-07-26: keep four arms through the common token target instead of
  screening a broad hyperparameter sweep.

## Entry log

### 2026-07-26 18:55 - Prior work and preregistration draft

- Hypothesis: whole-sequence expert-prefix routing can make an extractable small
  model with near-zero training overhead.
- Commit hash: pending first research snapshot.
- Commands: `weaver summary`; repository `rg`; targeted `gh issue view`; primary
  source searches for MatFormer, Gemma 3n, LayerSkip, Flextron, MoNE, and sparse
  upcycling.
- Config: planned d1280/L13/E256 top-4 large with E128 prefix.
- Result: direct production precedent exists for dense FFN/depth nesting, but
  no frontier-lab result was found for the proposed fixed MoE expert prefix.
  Marin's existing QB router and single-rack GB200 path can express the test.
- Interpretation: preregister a narrow four-arm proxy; do not infer frontier
  viability from one seed.
- Next action: publish the Weaver artifact, obtain Claude Fable review, revise,
  then implement Gate 0.

### 2026-07-26 19:05 - Fable review requested

- Hypothesis: an independent code-grounded review will identify preregistration
  flaws before implementation or launch.
- Commit hash: pending first research snapshot.
- Command: `claude --model fable -p <review prompt>`.
- Config: first prompt included the artifact and four relevant Grug files; the
  retry was restricted to the artifact and 1,200 words.
- Result: both processes remained alive without output and were stopped after
  bounded waits of roughly three minutes. No review content was received.
- Interpretation: the requested review was sent but the direct Fable lane was
  unavailable. Launch one asynchronous Loom Fable session from the research
  snapshot while Gate 0 implementation proceeds; do not launch cluster jobs
  until that review returns or a bounded review deadline is reached.
- Next action: snapshot the preregistration and start the asynchronous review.

### 2026-07-26 19:20 - Gate 0 passed and preregistration frozen

- Hypothesis: interleaving the fixed E128 subset over the E256 expert bank
  preserves ordinary full routing, defines an exactly extractable model, and
  remains compatible with the existing sharded training step.
- Commit hash: Gate 0 implementation snapshot pending.
- Commands: focused nested tests; complete
  `tests/test_grug_variant_contracts.py`; launcher lowering for the nested
  smoke arm; changed-file pre-commit; focused Pyrefly.
- Config: d1280/L13, sequence length 8192, E256/E128, top-4, global batch 256,
  EP64 over 16 four-GPU GB200 nodes.
- Result: the restricted E256 and extracted E128 logits match within `1e-5`;
  all-prefix outer gradients are exactly zero; full rows are bitwise unchanged
  and update both banks; the balanced row schedule and nested train step lower.
  The complete variant contract suite passed with 17 tests and one
  accelerator-only skip. The launcher resolves 5,275 steps and 11.06B tokens
  for the full phase. Focused Pyrefly passes for the launcher, trainer, and
  tests; checking the model still reports two unchanged pre-existing errors
  (`jax.shard_map` import discovery and the existing HF converter bound).
- Interpretation: Gate 0 passes. Federate the whole coordinator to
  `cw-us-east-08a`; child jobs run locally on that peer and inherit the batch
  band from their parent coordinator.
- Review: an asynchronous `loom ... --agent claude --model fable` launch also
  returned HTTP 405. No Fable content was received before the bounded review
  deadline. Revision 2 records this failed lane and the implementation review's
  interleaved-subset correction.
- Next action: publish revision 2, snapshot the implementation, then launch the
  four 20-step batch smokes.

### 2026-07-26 19:26 - Gate 1 smokes submitted

- Hypothesis: the four preregistered arms compile and complete 20 steps on the
  target single-rack GB200 geometry.
- Commit hash: `4dd4d09e9a344f836dd904c2efb08e852041f7d2`.
- Config: calendar version `2026.07.26`; main Iris submission federated to
  `cw-us-east-08a`; batch priority; 16 replicas with four GB200s each.
- Artifact check: `/power/nest-moe-artifact-check` succeeded in-cluster and
  found records for SlimPajama-6B and sampled Paloma caches. No data build or
  cross-region transfer is expected.
- Canonical coordinator jobs:
  - `/power/nest-moe-001-smoke-coord`
  - `/power/nest-moe-002-smoke-coord`
  - `/power/nest-moe-003-smoke-coord`
  - `/power/nest-moe-004-smoke-coord`
- Initial state: all four are pending federation admission on
  `cw-us-east-08a`; no child training job has started.
- Next action: monitor admission, compilation, steady-state metrics, and final
  checkpoint writes; resubmit only architecture or infrastructure failures
  within their preregistered retry rules.

### 2026-07-26 19:27 - Smoke relaunch required before GPU allocation

- Result: the four coordinators reached the target peer, but each child
  submission failed before allocation. The child `ResourceConfig` redundantly
  pinned `cw-us-east-08a`; from a coordinator already running on that peer,
  Iris interpreted the self-pin as a second federation request and rejected it
  because a cluster is not its own configured peer.
- Interpretation: this is an orchestration configuration error, not an
  architecture or accelerator failure. Remove the child pin and retain
  whole-job federation from the main Iris controller.
- GPU cost: zero; no child training job was created.
- Next action: publish a corrected snapshot and submit `r1` coordinators with
  the same preregistered configs.

### 2026-07-26 19:38 - Gate 1 attention fallback

- Result: all four `r1` smoke jobs received their 16-node allocations. The
  E256 and E128 controls failed before step 0 after the FA4/CuTe backward path
  imported `quack`; `cutlass.cute.core.ThrMma` was absent from the installed
  dependency stack. The other arms were stopped before they could repeat the
  same architecture-independent fault.
- Evidence: no W&B history rows or checkpoints were written; Iris reported no
  preemptions or hardware diagnostics. The traceback is recorded in
  `.agents/ops/2026-07-26-nested-moe-fa4-cute.md`.
- Change: use the supported `gpu_fa4_thd` backend. Five focused tests passed
  locally, the GPU-only lowering test skipped on the CPU host, and the
  experiment plan resolves the corrected backend.
- Commit hash: `03bcd5c74b`.
- Canonical corrected coordinators:
  - `/power/nest-moe-001-smoke-r2-coord`
  - `/power/nest-moe-002-smoke-r2-coord`
  - `/power/nest-moe-003-smoke-r2-coord`
  - `/power/nest-moe-004-smoke-r2-coord`
- Interpretation: the fallback changes only the attention kernel
  implementation shared by every arm; it does not alter a preregistered
  treatment variable.
- Next action: require finite step telemetry from all four `r2` runs before
  freezing the production token target.

### 2026-07-26 19:43 - Preserve THD metadata in layer masks

- Result: all four `r2` attempts failed before compilation because the Grug
  transformer discarded fixed-shape THD segment metadata while deriving its
  per-layer masks. This was a shared model bug, not a treatment effect.
- Change: `_layer_attention_masks` now preserves the incoming structured mask,
  uses the intended 2,048-token short window and full-causal long window, and
  is called by the model forward pass. The contract test asserts both window
  values and metadata identity.
- Validation: six focused layer-mask/nested tests passed. Focused Pyrefly
  reported only the two unchanged model-file errors already recorded at Gate
  0.
- Next action: snapshot the fix and submit one final four-arm smoke attempt.

### 2026-07-26 19:48 - Select fixed-shape THD examples

- Result: the `r3` model preserved its input mask, but the default text mixture
  selected the streaming causal dataset, whose examples do not carry static
  THD metadata. All arms stopped before compilation and before step 0.
- Change: apply `with_pack(data, 1)` to the resolved train/validation mixture,
  matching the production THD canary. This preserves one document per example
  while adding the required fixed-size segment representation.
- Validation: seven focused launcher/mask/nested tests passed, including a new
  materialized-config assertion that every nested experiment component has
  `pack=1`.
- Next action: snapshot and relaunch the four smoke arms.

### 2026-07-26 19:53 - Gate 1 dependency incompatibility confirmed

- Result: all four `r4` arms reached the THD FlashAttention import after the
  data fix, then failed before step 0 with the same missing
  `cutlass.cute.core.ThrMma` symbol seen under the CuTe backend.
- Diagnosis: the repository currently resolves CUTLASS DSL 4.6.0 while
  FlashAttention 4.0.0b16 and Quack 0.5.0 still import the CUTLASS 4.5
  `ThrMma` API. This is shared infrastructure: E256, E128, and both nested
  treatments produced no metric rows or checkpoints.
- Change: restore the last known-compatible CUTLASS DSL 4.5.2 constraint in
  both GPU extras and restore the root solver override for its overlapping base
  wheel.
- Next action: require a single-GB200 import canary to pass before relaunching
  the four-arm smoke.
