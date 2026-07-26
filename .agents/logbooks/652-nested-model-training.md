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

### 2026-07-26 20:05 - Gate 1 dependency pair identified

- Result: CUTLASS 4.5.2 passed the import canary, but all four `r5` arms failed
  during the first train-step compilation with
  `TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'`.
  Iris retried each child once; the retries and coordinators were stopped
  explicitly. No arm wrote a metric row or checkpoint.
- Diagnosis: the inverse incompatibility is JAX 0.11 with CUTLASS 4.5. The
  repository's CUTLASS 4.6 upgrade retained Quack 0.5.0 in the lock, although
  Quack 0.6.1 is the release line that requires CUTLASS 4.6.0. FA4's package
  constraint accepts Quack 0.6.1.
- Change: restore the current CUTLASS 4.6.0 pin and require
  `quack-kernels>=0.6.1,<0.7`. Add a one-GPU test that JIT-compiles both the
  FA4 THD forward and backward kernels rather than testing import alone.
- Next action: require the forward/backward canary to pass before another
  four-rack smoke.

### 2026-07-26 20:39 - Gate 1 reference-attention amendment

- Result: the CUTLASS 4.6.0 and Quack 0.6.1 pair fixed the import and lowering
  failures. A one-GB200 forward-only FA4/CuTe canary compiled in 13 seconds,
  executed successfully, and returned finite output. The corresponding
  forward/backward canaries compiled and dispatched, but did not return from
  GPU execution. Matching the upstream dense-backward `subtile_factor=2` did
  not change the result. Iris reported zero preemptions and zero task failures
  while the processes remained resident; both exact canary jobs were stopped.
- Interpretation: FA4 forward is functional on SM100, but its backward path is
  not currently usable for this training window. This is an
  architecture-independent kernel failure rather than evidence about a
  treatment arm.
- Amendment: run every arm with Levanter's `reference` attention backend.
  Preserve the preregistered model, tokens, data order, optimizer, routing
  treatments, and 64-GB200 allocation. Relative loss comparisons remain valid.
  Measured wall-clock overhead is backend-specific and will not be presented as
  production-FA4 overhead.
- Validation: nine focused Levanter attention tests and two launcher tests
  passed; the required pre-commit entry point passed. Commit `e2f4036439`
  contains the amendment and final FA4 diagnostic.
- Canonical smoke coordinators:
  - `/power/nest-moe-001-smoke-ref-r8-coord`
  - `/power/nest-moe-002-smoke-ref-r8-coord`
  - `/power/nest-moe-003-smoke-ref-r8-coord`
  - `/power/nest-moe-004-smoke-ref-r8-coord`
- Next action: require finite optimizer-step telemetry from all four arms, then
  freeze the production step count from observed throughput and the 09:00 UTC
  hard stop.

### 2026-07-26 20:43 - Reference smoke admitted

- Result: all four arms received their 16-node, 64-GB200 allocations. Iris
  reported all 64 tasks running, with zero preemptions and zero failures. Arm
  001 reached the train-step compile on every host.
- Next action: hold Gate 1 until every arm writes at least three finite
  post-warmup steps and completes its 20-step smoke.

### 2026-07-26 20:58 - Reference backward data failure and proxy amendment

- Result: all four d1280/length-8192 arms returned finite first-forward losses.
  The E256 control and both nested arms then produced nonfinite gradients and
  stopped cleanly at step 2. Their first-step throughputs were 8,015, 7,883,
  and 8,103 tokens/s. The E128 control exposed the same nonfinite-gradient
  signature and was stopped explicitly before repeating the long second step.
  No task was preempted or retried.
- Diagnosis: `pack=1` was introduced for the THD kernel contract, but the
  reference attention backend does not require THD metadata. Fully masked
  padding query rows produce undefined all-masked softmax values under the
  reference path, which contaminate backward despite zero token loss weights.
- Change: retain `pack=1` only for `gpu_fa4_thd`; use ordinary causal examples
  for `reference`. Add bounded hidden-dimension and sequence-length overrides,
  and rebuild heuristic optimizer hyperparameters from the actual batch and
  sequence length.
- Proxy amendment: d768, 8 layers, length 2,048, global batch 1,024, four
  steps. This retains 2,097,152 tokens per step and approximately 2.0B/1.1B
  large/small parameter counts while making the reference backend feasible.
- Validation: two materialized-launcher tests and the required focused
  pre-commit entry point passed.
- Next action: snapshot and submit all four corrected arms. Gate 1 still
  requires three finite steps and at least 85% relative nested throughput.

### 2026-07-26 21:17 - Proxy optimizer and router arithmetic amendment

- Result: the d768 controls produced finite first gradients but nonfinite
  weights after one full-rate update. The nested arms produced nonfinite first
  gradients. All final evaluation losses were nonfinite.
- Diagnosis: a four-step run rounds the fractional 1% warmup to zero. Nested
  eligibility also used `-inf`, which entered QB subtraction and reduction.
- Change: use a finite `-1e9` eligibility sentinel and five explicit warmup
  steps, matching 1% of the bounded 500-step production schedule. A focused
  regression requires every nested-router gradient leaf to be finite.
- Validation: four focused launcher, router-gradient, extraction, and
  full-row tests passed; the required pre-commit entry point passed.
- Next action: publish the amended snapshot and repeat the common four-arm
  smoke. No arm is promoted without three finite updates.

### 2026-07-26 21:35 - Final fp32 feasibility gate

- Result: the corrected r10 E128 arm produced a finite first gradient. The
  E256 control and nested arms produced NaN or infinite gradient norms, and
  every final validation loss was NaN. Iris reported zero failures and zero
  preemptions.
- Interpretation: no architecture arm promotes. The bf16 reference backend is
  not a stable scientific vehicle for this E256 proxy.
- Final bounded test: E256 control and nested25 at batch 256 with full-fp32
  compute. Both require three finite updates, finite Paloma, overflow at or
  below 1%, and a saved checkpoint. Failure of either closes Gate 1 without
  production launches.
