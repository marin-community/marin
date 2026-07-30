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

- The NEST-BURN-001/002 control is invalid for the original architecture
  comparison. It trained one document per 8,192-token example with padding
  excluded from loss and also changed the model/router source. Its nominal
  token count overstates useful phase-0 targets by at least 3.57x.
- The immutable `aug-dk-d768-ev-sw2k-g4-nomtp-noconv-f1` bundle reproduced
  through update 1,000. Median absolute pointwise loss error was `0.00229`
  nat and Paloma macro differed by `+0.00368`.
- The corrected treatment uses the same augmented Grug source, dense packing,
  Datakit mixture, optimizer, batch, and one-node/8-H100 topology as the
  reproduced control. It restricts 12.5% of sequences to experts 0--127 and
  12.5% to experts 0--15, with independent QB state for E256/E128/E16.
- At update 1,000, corrected fixed25 costs `+1.13%` median optimizer-step time.
  Full-model Paloma is `+0.02267` nat worse than the matched control; E128 and
  E16 are `+0.05502` and `+0.17999` worse. Both arms continue to the matched
  4.42B-token endpoint.

## Hypothesis queue

### Active

- `NEST-MOE-H1`: 25%-prefix nesting preserves full-model loss, produces a
  near-control small prefix, and costs less than 10% throughput.
- `NEST-MOE-H2`: 50%-prefix training improves the extracted prefix at a larger
  full-model cost.
- `NEST-MOE-H3`: QB compensation prevents outer-expert undertraining.

### Blocked

- Production FA4 throughput measurement: the SM100 backward canary dispatches
  but does not return.

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
- 2026-07-28: reject NEST-BURN-001/002 as evidence for model quality after the
  control failed to reproduce the augmented d768 reference.
- 2026-07-29: continue the corrected E256/fixed25 pair after the exact-source
  control passed the preregistered update-1,000 gate.

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

### 2026-07-26 21:59 - Common-capacity discovery rerun

- Result: the fp32 nested25 canary completed four finite updates, saved a
  checkpoint, and reached Paloma macro loss 11.71754 in full mode and 11.72052
  in nested mode. Assignment overflow was 9.84%, 4.33%, 5.37%, and 5.93%
  across the four updates. The matched control exhausted four JAX
  gang-incarnation attempts before step 0 and was stopped.
- Interpretation: capacity factor 1.0 is not a usable comparison vehicle, but
  the run did not establish that nesting caused the overflow. The user
  directed the study to fix the common routing inefficiency and continue the
  architecture comparison.
- Change: expose `capacity_factor` in `GrugModelConfig` and the nested launcher.
  Seven focused contracts and the required pre-commit entry point passed.
  Commit `c7bad5119a` contains the routing-only change.
- Jobs: submit E256, E128, nested25, and nested50 together for 20 updates,
  batch 256, full fp32, capacity factor 1.25, on 64 GB200s each. The run suffix
  is `cf125-r13`.
- Next action: require finite checkpoints and less than 1% overflow from all
  four arms, then compare matched loss, throughput, routing balance, and cost.

### 2026-07-26 22:22 - Gate 2 nested50 rejection and retry recovery

- Gate 1 result: E256 control, nested25, and nested50 completed 20 finite
  updates. Terminal Paloma was 11.24659, 11.24700, and 11.25210. Terminal
  overflow was 0.072%, 0.044%, and 0%; median steady throughput was 2.496M,
  2.536M, and 2.530M tokens/s. Both nested arms passed the feasibility and
  throughput gates. The E128 control remained blocked by JAX gang retries.
- Gate 2 result: the E256 control and nested50 completed 500 matched updates,
  262.144M tokens each. Terminal full-mode Paloma was 6.14223 and 6.21521;
  nested50's nested-mode Paloma was 6.20705. The +0.07298 full-mode and
  +0.06482 nested-mode penalties exceed the preregistered +0.02 rejection
  boundary. The 50% schedule is rejected at this proxy scale.
- Routing interpretation: capacity factor 1.25 removed terminal overflow, but
  the E256 control itself had transient overflow during early optimization.
  Overflow is therefore a common router/capacity dynamic rather than evidence
  that nesting alone caused the original 5.93% observation.
- Infrastructure diagnosis: r16 workers restarted at attempt 1 and split
  between task 0's stale attempt-0 and current attempt-1 coordinator
  addresses. Commit `b9029615fa` scopes JAX coordinator endpoint names by
  nonzero gang attempt; 31 focused tests and pre-commit passed. The incident is
  recorded in
  `.agents/ops/2026-07-26-jax-retry-stale-coordinator.md`.
- Jobs: E128 Gate 1 and nested25 Gate 2 were relaunched at batch priority as
  `cf125-r17`.
- Review blocker: Claude Fable launch/poll still returns Loom ACP HTTP 403.
- Next action: finish nested25 Gate 2, then run the E128 matched-token baseline
  if Gate 1 succeeds. Promote only arms that satisfy the loss boundary; do not
  spend the remaining window on rejected nested50 follow-ups.

### 2026-07-26 22:36 - Gate 2 selects nested25

- Result: the corrected nested25 run completed 500 updates and 262.144M
  training tokens. Final Paloma macro loss was `6.15123` in full E256 mode and
  `6.18123` in fixed E128 mode. The preregistered E256 control was `6.14223`;
  the full-model delta is `+0.00900`, inside the `+0.010` gate.
- Result: the standalone E128 control completed after attempt-scoped JAX
  coordination fixed the retry fault. Its final Paloma macro loss was
  `6.21359`; the extracted-model delta is `-0.03237`, better than the
  preregistered `+0.030` threshold and better on all 16 Paloma domains.
- Throughput: nested25 median steady throughput was 2.484M tokens/s, `100.31%`
  of the E256 control. Mean assignment overflow was `0.565%`, with zero
  terminal overflow.
- Decision: nested25 passes Gate 2. Nested50 remains rejected.

### 2026-07-26 22:40 - Untreated fixed-subset diagnostic

- Result: an E256 control repeat with the fixed even E128 subset exposed only
  at evaluation finished at `6.13964` full-mode and `6.27433` subset-mode
  Paloma.
- Sensitivity: nested25 is `+0.01159` behind this repeated full control,
  missing the `+0.010` margin by `0.00159`. The primary registered comparison
  passes, but its binary threshold is sensitive to one-run control variation.
- Architecture evidence: the trained nested E128 is `-0.09311` better than the
  untreated fixed half and wins on 15 of 16 domains. Its benefit is not
  explained by selecting a naturally strong half of the expert bank.
- Routing: untreated core/outer assignment CV was `0.1037`/`0.1012`;
  nested25 was `0.1041`/`0.1005`. Nested25 sent `49.32%` of assignments to the
  core bank. After subtracting the 25% core-only rows, full rows sent an
  inferred `32.43%` core and `67.57%` outer, consistent with QB-driven
  residual allocation.

### 2026-07-26 22:46 - Gate 3 breakout cooldown passes

- Result: the fixed even E128 subset was compacted from nested25 step 500,
  loaded with a fresh optimizer, and trained directly for 50 E128 updates.
  Paloma at 10/20/30/40/50 updates was
  `6.18579`/`6.16572`/`6.15394`/`6.14163`/`6.13423`.
- Decision: Gate 3 passes. The final checkpoint is `-0.04700` better than the
  extracted start and `-0.07936` better than the independently trained E128.
  The cooldown adds `9.956%` analytic model FLOPs to the E256 pretraining run.
- Cost caveat: the isolated 50-step job adds 5.104 charged GPU-hours, making
  nested plus cooldown `+67.9%` versus the E256 proxy. Compilation, checkpoint
  loading, and five evaluations dominate this short job; an in-process
  appended cooldown is required to validate the analytic 10% wall-clock
  estimate.

### 2026-07-26 22:49 - Claude Fable review recovered

- Result: the Loom ACP connection recovered. Review artifact
  `review` was fetched from Fable session `fdjx43mm`; its child Weaver issue
  `#659` and the Fable session were closed.
- Incorporated findings: common capacity 1.25 is a valid proxy comparison but
  is not production overhead evidence; 50% restriction is routing-degenerate;
  exact extraction assumes no overflow; the proxy is undertrained; and a
  one-expert-per-rank production layout may require capacity at least
  `1 + restricted_fraction` before route rebalancing.
- Decision: retain the positive discovery result while requiring production
  sharding and capacity accounting in the next scale gate.

### 2026-07-26 23:02 - SFT cache collision invalidates the first transfer loss

- Result: four matched eight-step SFT jobs completed from the E256 control,
  E128 control, nested25 full checkpoint, and cooled E128 breakout. A provenance
  check found that Levanter logged tokenizer and chat-template metadata
  mismatches but loaded an existing cache anyway.
- Interpretation: checkpoint loading and trainability succeeded, but the r21
  loss values are not valid evidence for the intended Llama 3.1 instruct,
  assistant-only WildChat recipe. Pretraining, routing, extraction, and
  cooldown findings are unaffected.
- Change: step-count SFT cache paths now include cache version, tokenizer,
  template, and packing identity. Fifteen focused SFT tests, Pyrefly, and
  pre-commit passed. Commit `ecead81e68`.
- Next action: build the isolated cache once, rerun the four eight-step SFT
  arms as `r22`, and exclude every `r21` loss value from the report.

### 2026-07-26 23:15 - WildChat null tool-call compatibility

- Result: two recipe-isolated cache coordinators exited 137 before all workers
  registered and were classified as preemptions. A later attempt reached all
  five shards and deterministically failed while rendering the Llama 3
  trainable template.
- Diagnosis: canonical WildChat messages carry `"tool_calls": null`. The
  template tested only for key presence, then evaluated `len(None)`. It also
  would have omitted a null-bearing ordinary message from the rendered
  conversation.
- Change: the template now distinguishes nonempty tool-call lists from null or
  absent values. A regression applies the real tokenizer and template to user
  and assistant messages with null tool calls and verifies the supervised
  assistant span. Pytest, Pyrefly, and pre-commit passed. Commit `540d07e2c9`.
- Cache identity: the template change moves the corrected recipe from key
  `216d2f` to `0b14cd`, so no partial shard from the failed build can be reused.
- Next action: run the large-control SFT arm as `r23` to materialize the shared
  corrected cache, then launch the other three matched arms.

### 2026-07-26 23:30 - Corrected SFT transfer gate completes

- Result: the large control built recipe-isolated WildChat cache `0b14cd`,
  loaded it without a metadata warning, trained for eight updates, and saved a
  checkpoint. Three coordinators launched before its final consolidation was
  visible and entered redundant cache builds. They were stopped before
  concurrent materialized-cache copies and excluded from analysis.
- Result: fresh `r24` small-control, nested-full, and breakout jobs loaded the
  completed cache directly. All three completed eight updates and saved
  checkpoints. The large-control run remains `r23`.
- Loss: mean assistant-token loss over updates 2–7 was `7.08179` for E256,
  `7.15806` for E128, `7.08169` for nested25 full, and `7.03852` for the cooled
  breakout. Nested25 full differs from E256 by `-0.00010`; breakout differs
  from E128 by `-0.11954`.
- Routing: mean assignment overflow was `12.815%`, `11.060%`, `11.927%`, and
  `10.794%`, respectively. Gate 4 establishes loadability and short-horizon
  trainability only; it does not establish loss-safe post-training or agentic
  capability under this cold-router distribution shift.
- Verification: the changed-file pre-commit suite passed. Repository-wide
  pytest collection is blocked by the existing Flax/JAX environment mismatch:
  installed Flax references `jax.core.Effect`, which JAX 0.11 removed.

### 2026-07-26 23:55 - Final branch review

- Result: the required branch-wide lint review completed. Concrete findings
  were fixed: launcher phase values and checkpoint initialization are enums,
  attention selection occurs at the environment boundary, checkpoint restore
  logic is no longer duplicated, the SFT model source has one concrete
  checkpoint-path type, tracker naming reuses the existing MoE launcher
  helper, stale docstrings were corrected, and result histories now use a
  typed record with immutable run maps.
- Disposition: the remaining environment and function-size findings describe
  the intended experiment entrypoint boundary. `build()` resolves one
  reproducible arm from launch-time environment variables and assembles its
  Iris artifact; it is not a reusable configuration API. Splitting that
  orchestration after the completed runs would add indirection without changing
  the scientific result.
- Verification: seven directly affected nested/extraction contracts passed.
  The 26 SFT, chat-template, and Iris retry tests passed. A wider 52-test run
  had 49 passes, two skips, and one unrelated failure in the Grug base CPU
  metric smoke: JAX explicit sharding rejected concatenation of
  `P(('replica_dcn', 'data'), None)` with `P(None, None)`.

### 2026-07-27 15:40 - Cost and power-ladder extension preregistration

- Status: `preregistered`; no extension runs have been launched or inspected.
- Hypotheses:
  - `NEST-MOE-COST`: once compilation, evaluation, and checkpoint time are
    separated, restricting router eligibility without adding a second forward
    pass costs at most 10% steady-state time per optimizer step versus the
    matched E256 control. An overhead above 25% rejects the implementation as a
    scale candidate.
  - `NEST-MOE-LADDER`: a rotating eligibility ladder over E128, E32, E8, and
    E1 subsets can expose useful miniature models while retaining full-model
    Paloma within `+0.02` at 25% restricted rows. The 50% arm is exploratory
    and is rejected if its full-model penalty exceeds `+0.02`.
- Common configuration: d768, eight layers, sequence length 2,048, global
  batch 256, 8,192 optimizer updates (4.295B tokens), seed 0, full-fp32
  compute, reference attention, 256 total experts, top-4 routing, capacity
  factor 1.25, 64 GB200s per arm on `cw-us-east-08a`, batch priority. Paloma
  runs every 2,048 updates and at termination.
- Four concurrent arms:
  - E256 control;
  - standalone E128 control;
  - rotating E128/E32/E8/E1 ladder on 25% of batch rows;
  - the same ladder on 50% of batch rows.
- Schedule: restricted rows cycle through the four eligible-bank sizes. Within
  a size, the eligible coset rotates across the E256 bank: two E128 cosets,
  eight E32 cosets, 32 E8 cosets, and 256 E1 cosets. This makes each
  down-sampling event use a different subset and avoids concentrating the E1
  traffic on one expert rank. Evaluation uses offset-zero representatives for
  E128, E32, E8, and E1; these are exchangeable representatives of the
  balanced rotation, not a claim that one fixed E1 expert received every
  restricted E1 example.
- E1 routing rule: top-4 is reduced semantically to top-1. The three inactive
  dispatch slots carry zero combine weight and are assigned uniformly across
  experts so dispatch shape and expert FLOPs remain matched. They do not count
  in semantic routing statistics. This tests model nesting at equal training
  FLOPs; it is not a claim that an extracted E1 model should retain top-4
  inference compute.
- Cost estimand: median post-warmup step duration derived from
  `tokens_per_step / throughput/tokens_per_second`, plus 1,000-step block
  medians and a contiguous-block bootstrap confidence interval. End-to-end
  runtime is modeled as
  `startup + steps * steady_step + evaluations * eval_time + checkpoint_time`.
  Report both steady-state overhead and charged GPU-hours; do not use the
  latter alone because the 500-step jobs were dominated by fixed work.
- Quality estimand: full-model and representative-submodel Paloma curves
  against tokens. Fit late-run power-law and log-linear sensitivity models,
  report held-out residuals, and extrapolate only over a stated token range.
- Gate: all arms must reach 8,192 finite updates with terminal overflow below
  1%. If both ladder arms pass, use the remaining GPU window for a matched
  second seed of E256 plus the better ladder arm and/or a longer continuation,
  prioritizing replication of the cost ratio. Deadline for termination and
  wrap-up is 2026-07-28 03:31 UTC.

### 2026-07-27 15:42 - Four-arm 4.295B-token wave submitted

- Commit: `25c52f7dd83941a0e5bc851de96fb2081f8cfe6e`.
- Validation: 12 focused launcher, nesting, extraction, schedule, E1 routing,
  and lowered-train-step contracts passed. The required changed-file
  pre-commit checks passed. All four materialized plans contain 8,192 steps,
  eval interval 2,048, seed 0, full fp32, and the expected arm configuration.
- Command template:
  `uv run iris --config lib/iris/config/marin.yaml job run --no-wait
  --priority batch --job-name <name> -e NESTED_ARM <arm> <common env> --
  uv run python experiments/grug/moe/launch_nested_experts.py --version
  2026.07.27 --run`.
- Canonical coordinator jobs:
  - `/power/nest-moe-cost-large-r25-coord`;
  - `/power/nest-moe-cost-small-r25-coord`;
  - `/power/nest-moe-cost-ladder25-r25-coord`;
  - `/power/nest-moe-cost-ladder50-r25-coord`.
- Monitoring state:
  `scratch/20260727-1542_monitoring_state.json`.
- Next action: verify federation to 16 four-GPU GB200 workers per arm, inspect
  compilation and first finite steps, and recover job-level preemptions without
  changing the preregistered run IDs.

### 2026-07-27 15:49 - Correct regional prefix relaunch

- Result: no GPU workers or W&B runs had started. The first E256 coordinator
  inherited the main cluster's GCS prefix and began rebuilding the 14GB
  SlimPajama cache from Hugging Face instead of reusing the existing
  us-east S3 cache. The other coordinators were still installing dependencies.
- Action: stopped the four exact `r25` coordinator roots, including the
  tokenizer descendant, before training allocation. Relaunched the four arms
  as `r26` coordinators with
  `MARIN_PREFIX=s3://marin-us-east-02a/marin`. Model run IDs and every
  preregistered scientific parameter remain unchanged (`cost-r25`).
- Active coordinator roots:
  - `/power/nest-moe-cost-large-r26-coord`;
  - `/power/nest-moe-cost-small-r26-coord`;
  - `/power/nest-moe-cost-ladder25-r26-coord`;
  - `/power/nest-moe-cost-ladder50-r26-coord`.
- Interpretation: this is a launch-boundary correction, not an experimental
  rerun. No loss, timing, routing, or quality outcome existed to inspect.

### 2026-07-27 15:54 - Explicit CoreWeave federation relaunch

- Result: the `r26` coordinators ran on the main cluster and therefore lacked
  an AWS credential chain. Ladder25 completed environment setup and failed on
  `botocore.exceptions.NoCredentialsError` while resolving the S3 artifact
  records. No accelerator descendants, W&B runs, or optimizer steps existed.
- Action: stopped all four `r26` roots and relaunched as `r27` through the main
  Marin controller with `--target-cluster cw-us-east-08a`. The S3 prefix,
  model run IDs, commit, and scientific configuration are unchanged.
- Active coordinator roots:
  - `/power/nest-moe-cost-large-r27-coord`;
  - `/power/nest-moe-cost-small-r27-coord`;
  - `/power/nest-moe-cost-ladder25-r27-coord`;
  - `/power/nest-moe-cost-ladder50-r27-coord`.

### 2026-07-27 16:00 - Long-run training is active

- All four `r27` children reached 16 running tasks with four GB200s per task:
  64 GB200s per arm and 256 total. Iris reported zero failures and zero
  preemptions.
- The first finite W&B observations appeared between steps 119 and 190.
  Instantaneous throughput was 2.36M--2.43M tokens/s, or 0.216--0.222 seconds
  per 524,288-token update. At that rate, 8,192 updates require 28.9--30.3
  minutes of pure optimizer-step time. The end-to-end estimate remains
  40--55 minutes until the first periodic Paloma evaluation measures its
  incremental cost.
- Early capacity overflow ranged from 0.40% to 1.84% and was already declining;
  this is a stability monitor, not a terminal result. The preregistered gate
  uses terminal overflow below 1%.

### 2026-07-27 16:10 - First 1.074B-token quality checkpoint

- Paloma macro loss at the common step-2,048 checkpoint:
  - E256 control: `5.622856`;
  - E128 control: `5.747447`;
  - ladder25 full: `5.530346` (`-0.092510` versus E256);
  - ladder50 full: `5.677487` (`+0.054631` versus E256).
- Representative ladder25 submodels were E128 `5.709478`, E32 `6.366071`,
  E8 `7.031826`, and E1 `7.416303`. The E128 representative was `-0.037970`
  better than the standalone E128 control at common tokens.
- Representative ladder50 submodels were E128 `5.774388`, E32 `6.365642`,
  E8 `7.016877`, and E1 `7.173573`.
- Interpretation: this is the first of four validation checkpoints, not a
  promotion decision. Ladder25 has the ideal early sign at matched step FLOPs:
  better full-model validation and a better E128 submodel. Ladder50 again
  violates the `+0.02` full-model margin.

### 2026-07-27 16:22 - Second 2.147B-token quality checkpoint

- Paloma macro loss at step 4,096:
  - E256 control: `5.486187`;
  - E128 control: `5.542410`;
  - ladder25 full: `5.410714` (`-0.075473` versus E256);
  - ladder50 full: `5.494916` (`+0.008729` versus E256).
- Ladder25 representatives were E128 `5.613199`, E32 `6.513178`, E8
  `7.403059`, and E1 `7.631603`. E128 is now `+0.070789` worse than the
  standalone E128 control. E32, E8, and E1 all regressed from their step-2,048
  evaluations despite more total training.
- Ladder50 representatives were E128 `5.588541`, E32 `6.449406`, E8
  `7.471175`, and E1 `7.666183`; E128 is `+0.046131` worse than standalone.
- Interpretation: ladder25's full-model advantage has persisted for two
  checkpoints, but the smaller representatives do not improve monotonically.
  Rotating exposure balances the full expert bank while any one extractable
  coset receives only intermittent small-mode updates and can be overwritten
  by full-mode updates. Promotion must therefore distinguish structured
  regularization from the original goal of stable breakout checkpoints.

### 2026-07-27 16:33 - Third 3.221B-token quality checkpoint

- Paloma macro loss at step 6,144:
  - E256 control: `5.409805`;
  - E128 control: `5.475025`;
  - ladder25 full: `5.348059` (`-0.061746` versus E256);
  - ladder50 full: `5.458055` (`+0.048250` versus E256).
- Ladder25 representatives were E128 `5.574710` (`+0.099685` versus
  standalone), E32 `6.672468`, E8 `7.572572`, and E1 `7.977216`.
- Ladder50 representatives were E128 `5.539650` (`+0.064625` versus
  standalone), E32 `6.565496`, E8 `7.629114`, and E1 `7.858928`.
- Ladder25's full model has beaten E256 at all three checkpoints. Its fixed
  offset-zero E128 has lost to standalone at the latest two, while E32, E8,
  and E1 continue to regress. This promotes ladder25 as a possible structured
  regularizer but not yet as evidence for stable breakout checkpoints.

### 2026-07-27 16:50 - 4.295B-token gate and continuation preregistration

- All four arms completed 8,192 updates, final evaluation, and checkpoint
  commit with no retries or preemptions.
- Final Paloma macro:
  - E256 `5.480636`;
  - E128 `5.455853`;
  - ladder25 full `5.332881`, E128 `5.679830`, E32 `6.982411`, E8
    `7.537161`, E1 `7.782655`;
  - ladder50 full `5.451588`, E128 `5.534664`, E32 `6.745502`, E8
    `7.631391`, E1 `7.843251`.
- Ladder25 improved the full-model endpoint by `-0.147755` and lost to the
  standalone E128 by `+0.223978`. Ladder50 improved the full endpoint by
  `-0.029048` and lost to E128 by `+0.078812`.
- Median post-warmup step overhead versus E256 was `+0.832%` for ladder25 and
  `+0.275%` for ladder50. All terminal overflow rates were below `0.53%`.
- Interim report:
  `docs/reports/nested-model-training-interim.md`.

The user requested another 10--20B tokens. The continuation is frozen before
launch:

- Resume the full model, optimizer, scheduler, and global step from each
  step-8,192 checkpoint. Do not initialize weights into a fresh optimizer.
- Continue E256, E128, ladder25, and ladder50 through global step 38,912:
  30,720 new updates and 16.106B additional tokens per arm, 20.401B total.
- Keep seed, data stream, batch, precision, capacity, hardware, and every
  architecture setting unchanged. Use 64 GB200s per arm and 256 total.
- Evaluate every 8,192 global steps and at termination. Ladder evaluation uses
  all two E128 offsets plus four evenly spaced offsets at E32, E8, and E1.
- Continuation gates:
  - ladder25 median step overhead remains below 10%;
  - terminal overflow remains below 1%;
  - ladder25 full-mode Paloma remains no worse than E256 at the terminal
    checkpoint;
  - report the median and range across offsets at every small size. E128
    breakout quality passes only if its median is within `+0.05` of standalone
    E128.
- The SlimPajama-6B stream restarts after exhaustion. The extension therefore
  measures common-data long-horizon behavior and forgetting, not novel-token
  sample efficiency.
- Expected pure training time is 1.85 hours. Recompilation, multi-offset
  evaluation, and checkpointing raise the wall estimate to 2.5--3.0 hours.

### 2026-07-27 17:01 - 16.106B-token continuation launched

- Submitted four batch-priority coordinator jobs through the main `marin`
  controller, federated to `cw-us-east-08a`:
  - `/power/nest-moe-extend-large-r28-coord`;
  - `/power/nest-moe-extend-small-r28-coord`;
  - `/power/nest-moe-extend-ladder25-r28-coord`;
  - `/power/nest-moe-extend-ladder50-r28-coord`.
- Each coordinator created a 16-node, 64-GB200 training child in a new
  `extend16b-r28` output directory. The original 4.295B-token checkpoints
  remain immutable.
- The first submission attempt was rejected by the local Iris CLI because the
  mixed-precision environment value was quoted as part of its key. No job was
  created and no cluster time was consumed. The corrected submissions
  separated the environment key and value.
- Environment setup is in progress. Scientific validation remains pending
  until the training logs confirm both the expected source checkpoint and a
  restored global step of 8,192.

### 2026-07-27 17:05 - Invalid continuation stopped; LR restart amended

- All four `r28` arms loaded the intended immutable step-8,192 checkpoints.
- E256 became NaN at step 8,197 and E128 at step 8,194. The `r28` wave was
  stopped and will not be used as experimental evidence.
- Cause: rebuilding the linear decay schedule with a 38,912-step horizon
  changed the learning rate at the restored step:
  - MuonH jumped from `0.00019696` to `0.00315176`;
  - Adam jumped from `0.00004545` to `0.00072733`.
  The approximately 16x discontinuity is common to every arm and independent
  of nested routing.
- Continuation amendment, frozen before replacement launch:
  - preserve the original schedule exactly through step 8,192;
  - begin a second optimizer cycle at step 8,192;
  - linearly rewarm from the old 5% floor to the original peak over 512
    updates, then linearly decay to the 5% floor at step 38,912;
  - resume model, optimizer moments, and global step from the original `r25`
    checkpoints, never the NaN-tainted `r28` outputs;
  - keep every paired architecture, data, evaluation, and hardware setting
    unchanged.
- The 512-step rewarmup is 0.268B tokens. Replacement jobs must remain finite
  through both the first resumed updates and the peak at step 8,704 before the
  incident is considered resolved.

### 2026-07-27 17:14 - Full-state continuation abandoned

- The schedule-continuous `r29` retry logged the intended learning rates at
  step 8,192, exactly matching the old schedule floor.
- E256 nevertheless became NaN at step 8,195 and E128 at step 8,194. Both had
  finite first-step losses. This falsifies LR discontinuity as the sufficient
  cause and implicates restored optimizer state or another checkpoint-resume
  mechanism.
- The original checkpoint weights remain finite: initial full-mode Paloma was
  `5.501` for E256, `5.321` for ladder25, and `5.447` for ladder50 before any
  failed update affected evaluation. The `r29` outputs are invalid and were
  stopped.
- The user asked for architecture discovery rather than cleanroom resume
  semantics. The next wave is therefore a paired weights-only continuation:
  - initialize each arm from its original step-8,192 model weights;
  - create fresh optimizer state and reset the phase-local step to zero;
  - train 30,720 updates, adding the same 16.106B tokens;
  - use 10% of the original pretraining peak learning rates, warm up over 512
    updates, then linearly decay;
  - keep the four architectures, data, batch, precision, capacity, hardware,
    and multi-offset evaluation plan unchanged.
- Final analysis will add the 8,192-step offset when splicing phase-local
  histories. It will report this optimizer reset prominently and will not call
  the result a seamless full-state continuation.

### 2026-07-27 17:21 - Weights-only launcher bug caught

- `r30` coordinators resolved the correct source checkpoints, but their
  training children reported scratch initialization.
- Cause: `GrugMoeLaunchConfig.init_from` was passed into `TrainerConfig` but
  the custom Grug training loop only executes that path when
  `GrugTrainerConfig.initialization_mode` is explicitly `WEIGHTS_ONLY`.
- All four jobs were stopped. No `r30` metrics are evidence for the
  continuation.
- The generic launcher now derives `WEIGHTS_ONLY` from its documented
  `init_from` field. Replacement jobs must log
  `Initialized weights from .../step-8192` and begin with losses near the
  4.295B-token endpoints before they pass initialization validation.

### 2026-07-27 17:26 - Valid weights-only continuation running

- Launched `r31` through the main `marin` controller on
  `cw-us-east-08a`, 16 nodes and 64 GB200s per arm:
  - `/power/nest-moe-extend-large-r31-coord`;
  - `/power/nest-moe-extend-small-r31-coord`;
  - `/power/nest-moe-extend-ladder25-r31-coord`;
  - `/power/nest-moe-extend-ladder50-r31-coord`.
- Every training child searched its new output directory, found no prior run
  state, and then explicitly loaded the corresponding immutable `r25`
  `step-8192` checkpoint through the weights-only path.
- E256 reached phase step 248 and E128 step 263 with finite losses around
  `4.6`; ladder50 was also finite. Ladder25 was still amortizing its first
  compilation.
- All four arms subsequently passed phase step 512, the peak of the
  fresh-optimizer warmup, with finite losses. The valid continuation therefore
  passes both initialization and early stability gates.
- Median step durations after phase step 100 were `0.2134 s` for E256,
  `0.2179 s` for E128, `0.2122 s` for ladder25, and `0.2121 s` for ladder50.
  At this early point the nested arms are within measurement noise of E256
  (`-0.55%` and `-0.60%`), while E128 is `+2.11%`. Final cost estimates will
  use the full steady-state histories.

### 2026-07-27 17:46 - E256 W&B tail has durable fallbacks

- The E256 W&B uploader reported a fatal network upload error after phase step
  1,906. Training continued normally past step 3,000; the other three W&B
  uploaders remained current.
- Levanter's telltale tracker continued mirroring the same scalar metrics into
  durable finelog. A four-arm query recovered global step, loss, step duration,
  tokens/s, and routing overflow from task index zero. Finelog's early sampled
  medians after step 1,024 were `0.2130 s`, `0.2176 s`, `0.2123 s`, and
  `0.2124 s` for E256, E128, ladder25, and ladder50.
- The E256 task-zero pod also retains its append-only local W&B run file under
  `/app/wandb`; it was 204 MiB while training at roughly step 3,500. Copy it
  with `kubectl cp` near the end of training and sync it after completion if
  the uploader does not recover.
- This is a reporting-path incident, not an architecture or training failure.
  Finelog is sufficient for the continuous cost series even if W&B repair
  fails.

### 2026-07-27 18:12 - Fixed E16 ⊂ E128 ⊂ E256 amendment launched

- User correction: the intended breakout architecture is a true fixed nested
  chain, not rotation across every expert coset. The rotating run remains
  useful for testing structured regularization but is not expected to produce
  well-trained individual miniature checkpoints.
- Frozen treatment before observing any fixed-chain metrics:
  - logical E16 is always the same subset of logical E128, which is always the
    same subset of E256;
  - physical E16 uses expert indices `{0,16,...,240}` and physical E128 uses
    even indices, keeping both subsets balanced across the 64-way expert axis;
  - extraction compacts these physical sets into conventional logical
    `0..15` and `0..127` checkpoints;
  - restricted rows alternate E128 and E16; unrestricted rows use E256;
  - fixed25 restricts 25% of rows and fixed50 restricts 50%;
  - both treatments initialize from the E256 control's immutable `r25`
    step-8,192 weights, then use the same fresh optimizer, 512-step warmup,
    0.1x peak learning rate, data seed, batch, hardware, capacity, and
    30,720-update budget as the concurrent E256 continuation control;
  - Paloma evaluates full E256, fixed E128, and fixed E16 at phase steps
    8,192, 16,384, 24,576, and 30,720.
- Hypothesis: fixed-chain training will improve each extractable subset
  materially relative to the rotating offset-zero checkpoint, at the cost of
  a smaller full-model regularization benefit. Promote only if the fixed full
  model remains competitive with E256, routing overflow remains below 1% at
  the evaluation endpoints, and steady-state step overhead remains below 10%.
- Implementation commit: `1aa735b277`. Focused fixed-chain eligibility and
  launcher tests passed, along with changed-file lint and Pyrefly.
- Submitted through `marin` at batch priority to `cw-us-east-08a`:
  - `/power/nest-moe-fixed25-r32-coord`;
  - `/power/nest-moe-fixed50-r32-coord`.

### 2026-07-27 18:25 - Continuation Paloma diagnosis and fixed25 retry

- The first continuation gate increased absolute Paloma macro loss from
  `5.480636` to `5.597265` for E256 and from `5.455853` to `5.494336` for
  standalone E128. The increase is therefore not specific to nested routing.
- E128 worsened on 11 of 16 fixed Paloma domains. The paired domain delta had
  mean `+0.03848` and median `+0.03259`; wikitext, manosphere, and the Dolma
  subreddit slice had the largest increases. This is broader than a single
  macro-average outlier.
- The continuation is not a schedule-continuous training curve. It resets
  MuonH and Adam moments, restarts the SlimPajama stream, and uses a new
  30,720-step linear schedule. At phase step 8,192, MuonH LR was
  `0.000298781`, 1.52x the original step-8,192 endpoint LR `0.000196961`.
- The original full proxy also overrode the heuristic's intended 1% warmup
  with five steps. Batch accounting is correct at 524,288 tokens per update,
  and the MuonH parameter groups and fitted peak rates match the launcher
  configuration, but the absolute continuation should not be used for a
  scaling-law quality projection.
- Paired architecture deltas remain interpretable because all continuation
  controls and treatments share optimizer reset, data replay, batch,
  precision, and LR schedule. The fixed-chain arms remain matched to the E256
  continuation for that reason.
- Fixed50 initialized and passed its 512-step warmup. Fixed25 failed before
  model initialization when one worker inferred `num_slices=3` for 64
  devices. The terminal attempt had no W&B run, checkpoint, or optimizer
  update. It was resubmitted without scientific changes as
  `/power/nest-moe-fixed25-r33-coord`, retaining the canonical
  `fixed16b-r32` run and output identity.

### 2026-07-27 18:27 - Fixed-chain routing-capacity amendment

- Fixed50 reached 8.7% assignment overflow by step 1,840. The run was stopped
  before its first quality gate. The fixed25 topology retry was also stopped
  before model initialization completed.
- Cause: EP=64 stores four consecutive experts per rank. E128 is evenly
  represented across all ranks, but fixed E16 occupies only 16 of the 64
  expert ranks. With half the rows restricted and restricted rows alternating
  E128/E16, an E16-bearing rank receives about 1.75x average rank traffic.
  Capacity factor 1.25 necessarily clips those assignments. This is a
  topology/capacity property of true fixed nesting, not random router
  imbalance.
- Amendment frozen before replacement metrics:
  - use expert axis 16 on the same 64 GPUs, leaving four-way data parallelism;
  - each EP rank then owns one E16 expert, eight E128 experts, and sixteen E256
    experts, making every routing mode rank-balanced;
  - run a matched E256, fixed25, and fixed50 pilot for 600 updates from the
    immutable E256 step-8,192 weights;
  - keep capacity factor 1.25, batch 256, sequence length 2,048, fresh
    optimizer, 0.1x peak LR, 512-step warmup, full fp32, and seed 0;
  - promote only if all arms remain finite through the warmup peak, endpoint
    overflow is below 1%, and treatment median step overhead versus the EP=16
    E256 control is below 10%.
- The pilot is a topology and throughput gate, not a quality result. Promoted
  long arms restart from the common source weights.

The promoted fixed-chain experiment will instead restart from scratch to avoid
carrying the weights-only continuation confound into the primary architecture
result:

- E256, fixed25, and fixed50 use identical scratch initialization and data
  seed, EP=16, batch 256, sequence length 2,048, capacity factor 1.25, and
  full fp32;
- train for 8,192 updates, or 4.295B tokens, which stays within the nominal 6B
  corpus before replay;
- use the heuristic MuonH and Adam peak rates with 82 warmup updates, the
  rounded 1% warmup documented by the May recipe;
- evaluate at steps 2,048, 4,096, 6,144, and 8,192; fixed arms evaluate full
  E256, fixed E128, and fixed E16;
- compare treatment quality and step cost only against the concurrent EP=16
  E256 control. The rotating 20.4B continuation remains a separate
  long-horizon paired sensitivity analysis.

### 2026-07-27 18:36 - EP=16 topology gate passed

- All three 600-step pilots remained finite through the 512-step warmup peak.
- Median step duration after step 100:
  - E256 `156.715 ms`;
  - fixed25 `158.379 ms`, `+1.061%`;
  - fixed50 `157.059 ms`, `+0.219%`.
- Median capacity overflow after step 512 was `0.0050%`, `0.0252%`, and
  `0.0938%`; the largest observed values were `0.128%`, `0.162%`, and
  `0.229%`.
- EP=16 therefore passes the less-than-1% overflow and less-than-10% treatment
  overhead gates. The three promoted scratch arms may launch with the frozen
  8,192-step configuration.

### 2026-07-27 18:43 - Fixed50 scratch optimizer gate

- The first clean scratch wave used the heuristic peak LR and an 82-step
  warmup. E256 and fixed25 remained finite past steps 200 and 180. Fixed50
  logged finite losses through step 2, then the training loop detected a NaN
  on the next update and stopped. Iris marked the child successful because
  the loop handles NaN as a clean early return; the run is scientifically
  failed.
- Fixed50 overflow was zero at the finite updates. This is not recurrence of
  the EP=64 capacity problem.
- E256 and fixed25 were stopped before their first Paloma evaluation so the
  primary comparison does not mix schedules.
- Amendment frozen before retry metrics: run a 600-step fixed50 scratch gate
  at the same full peak LR with a 512-step warmup. If it remains finite through
  the peak, restart all three long arms with that schedule. If it fails,
  fixed50 fails the optimization gate and only a matched E256/fixed25 pair
  advances.

### 2026-07-27 18:52 - Fixed50 optimizer gate passed

- The 512-step warmup pilot reached the full MuonH peak learning rate
  `0.0039392302` at step 512 with finite loss `5.8143`.
- Routing overflow was `0.278%` at the peak. The earlier scratch NaN was
  therefore an optimization-ramp failure, not recurrence of the EP=64 routing
  hotspot.
- Promoted three matched scratch arms with 8,192 updates, 4.295B tokens,
  expert axis 16, capacity factor 1.25, batch 256, sequence length 2,048, full
  fp32, reference attention, seed 0, heuristic peak rates, a 512-step warmup,
  and Paloma every 2,048 updates:
  - `/power/nest-moe-fixedep16-large-w512-cost-r37-coord`;
  - `/power/nest-moe-fixedep16-fixed25-w512-cost-r37-coord`;
  - `/power/nest-moe-fixedep16-fixed50-w512-cost-r37-coord`.

### 2026-07-27 19:04 - Fixed-chain QB correction

- The first promoted wave was stopped before evaluation. At roughly matched
  early steps, fixed50's median router-bias norm reached `469`, versus `37`
  for E256 and `36` for fixed25. The earlier 600-step fixed50 scratch pilot
  had reached roughly `1,803`.
- Cause: QB used one uniform target count,
  `local_tokens * top_k / 256`, for every expert. True nested eligibility
  requires different legitimate assignment totals. An E16 core expert is
  eligible on full, E128, and E16 rows, while an outer E256 expert is eligible
  only on full rows. The impossible uniform target continually increased
  biases for ineligible experts and distorted subsequent full-row routing.
- Corrected QB computes the target for expert `e` as the sum, over tokens where
  `e` is eligible, of `active_top_k / eligible_expert_count`. The local
  quantile now uses that per-expert target. A deterministic fixed-chain test
  produces exact counts `[16, 4, 8, 4, 16, 4, 8, 4]`, which sum to the number
  of assignments.
- Launched a new 600-step fixed50 scratch gate:
  `/power/nest-moe-fixedep16-qbfix-w512-pilot-r38-coord`. Promotion additionally
  requires router-bias norms to remain comparable in order of magnitude to the
  untreated control.

### 2026-07-27 19:05 - Second rotating-ladder continuation gate

- At 12.885B effective tokens, full-mode Paloma macro loss was:
  - E256 `5.59614`;
  - E128 `5.53563`;
  - ladder25 `5.43270`, or `-0.16344` versus E256;
  - ladder50 `5.58540`, or `-0.01074` versus E256.
- The E256 control was nearly unchanged from its first continuation gate
  (`-0.00112`). E128 rose another `+0.04130`; 9 of 16 domains worsened, but
  the median domain delta was only `+0.00560`.
- Ladder25 retains a material full-model advantage at the second gate.
  Ladder50's advantage has narrowed substantially. All are still
  weights-only continuations with reset optimizer and data streams, so only
  the paired differences are used.

### 2026-07-27 19:21 - Matched no-QB fixed-chain gate launched

- The eligibility-weighted QB pilot failed before its first optimizer update
  on four consecutive attempts. Rank 0 exited with code 1 during first-step
  compilation/runtime; the persisted diagnostic contained only the secondary
  XLA coordination-service cancellation. No loss, LR, overflow, or parameter
  update was emitted. This is a systems failure of that QB implementation, not
  architecture-quality evidence.
- Decision: use an explicit `router_balance_mode=none` for the discovery
  experiment. QB remains the default for existing Grug arms. With balancing
  disabled, every layer emits zero pending QB beta and router biases remain
  zero. A compact fixed-eligibility forward test verifies this behavior.
- The matched scratch arms use the same no-QB mode, EP=16, capacity factor
  1.25, heuristic MuonH/Adam peak rates, 512-step warmup, batch 256, sequence
  length 2,048, full fp32, reference attention, seed 0, 8,192 updates, and
  Paloma every 2,048 updates:
  - `/power/nest-moe-fixedep16-large-noqb-w512-cost-r39-coord`;
  - `/power/nest-moe-fixedep16-fixed25-noqb-w512-cost-r39-coord`;
  - `/power/nest-moe-fixedep16-fixed50-noqb-w512-cost-r39-coord`.
- Gate: all three must remain finite through the warmup peak, endpoint
  overflow must stay below 1%, and treatment median step overhead versus the
  concurrent E256 control must remain below 10%. Quality comparisons are
  matched within r39; r39 is not compared absolutely to the earlier QB runs.

### 2026-07-27 19:30 - No-QB gate failed; conditioned auxiliary gate launched

- The r39 no-QB gate was stopped before update 600. Capacity overflow reached
  `50.14%` for E256 at update 571, `16.77%` for fixed25 at update 499, and
  `1.91%` for fixed50 at update 344. Router-bias norms remained exactly zero,
  confirming uncontrolled router collapse rather than recurrence of the QB
  target mismatch.
- Added an eligibility-conditioned auxiliary load-balance loss. It computes
  the standard assignment-frequency/probability loss separately for full E256,
  fixed E128, and fixed E16 rows, then averages those losses by token count.
  This asks each routing mode to balance only across experts that are eligible
  in that mode.
- A numerical regression has balanced E256 and E2 groups each contribute their
  independent optimum of `1.0`. A nested train step lowers with coefficient
  `0.01`; focused tests and changed-file lint/type checks pass.
- Launched matched 600-update r40 gates with coefficient `0.01`, otherwise
  preserving the r39 optimizer, topology, data, and capacity configuration:
  - `/power/nest-moe-fixedep16-large-eaux01-w512-pilot-r40-coord`;
  - `/power/nest-moe-fixedep16-fixed25-eaux01-w512-pilot-r40-coord`;
  - `/power/nest-moe-fixedep16-fixed50-eaux01-w512-pilot-r40-coord`.

### 2026-07-27 19:45 - Conditioned-router coefficient bracket

- The coefficient `0.01` r40 gate remained finite through update 599 and
  controlled the no-QB collapse, but missed the frozen endpoint-overflow gate:
  fixed25 ended at `1.235%` and fixed50 at `0.940%`. The E256 arm encountered
  a retry/W&B lifecycle fault before yielding a clean matched endpoint.
- A matched coefficient `0.02` r42 diagnostic improved endpoint overflow to
  `0.132%` for E256 and `0.037%` for fixed50. Fixed25 instead became
  non-finite at update 3, so r42 failed the optimization gate.
- Launched one final bottleneck-arm bracket at coefficient `0.015`, preserving
  the r42 topology, optimizer, data, seed, warmup, precision, and capacity:
  `/power/nest-moe-fixedep16-fixed25-eaux015-w512-pilot-r43-coord`.
- If r43 remains finite and ends below `1%` overflow, promote all three
  matched arms at coefficient `0.015`. Otherwise stop controller tuning and
  report fixed nesting as blocked on a narrow or absent routing-control window
  in this proxy configuration.

### 2026-07-27 19:53 - Fixed-chain quality experiment promoted

- The coefficient `0.015` fixed25 r43 pilot passed. It reached update 599 with
  finite loss `5.6194`, cross-entropy `5.5601`, auxiliary contribution
  `0.05929`, and endpoint capacity overflow `0.0717%`. It remained finite
  through the full Muon peak at update 512.
- Promoted three matched scratch arms for 8,192 updates, or 4.295B tokens,
  using coefficient `0.015`, expert axis 16, capacity factor 1.25, batch 256,
  sequence length 2,048, full fp32, reference attention, seed 0, 512 warmup
  updates, and Paloma every 2,048 updates:
  - `/power/nest-moe-fixedep16-large-eaux015-w512-cost-r44-coord`;
  - `/power/nest-moe-fixedep16-fixed25-eaux015-w512-cost-r44-coord`;
  - `/power/nest-moe-fixedep16-fixed50-eaux015-w512-cost-r44-coord`.
- Fixed arms evaluate full E256 plus extractable fixed E128 and E16 modes. All
  quality and throughput comparisons use the concurrent r44 E256 control.

### 2026-07-27 20:03 - Long auxiliary-controller gate failed

- The promoted r44 arms crossed the 512-step peak with finite loss, then
  assignment overflow rose to `14.50%` for E256, `14.39%` for fixed25, and
  `14.19%` for fixed50 by updates 1,189--1,219.
- The 600-step coefficient pilots had cooled from peak immediately after
  update 512. They did not exercise the sustained high-LR interval in the
  8,192-step schedule. The promotion gate was therefore too short for the
  schedule it was intended to validate.
- All three arms were stopped before the first Paloma evaluation. This is a
  common eligibility-aux controller failure, not evidence about fixed nesting
  quality.
- Do not continue coefficient sweeps. The only bounded follow-up considered is
  separate QB bias state for E256, E128, and E16 modes, which directly removes
  the shared-target conflict. If that cannot be implemented and gated cleanly
  in the remaining window, report fixed nesting as router-controller blocked.

### 2026-07-27 20:12 - Eligibility-specific QB launched

- Added an `eligibility_qb` controller with separate QB bias rows for E256,
  E128, and E16. Each row computes a uniform assignment quantile only from
  tokens in that routing mode. The full model applies the E256 row;
  extraction compacts the matching E128 or E16 row with its expert subset.
- Ordinary QB retains its existing one-dimensional bias state. The focused
  contracts verify finite group betas, zero beta for ineligible experts,
  correct extraction, and train-step lowering. Six focused tests pass. The
  complete variant-contract file has 35 passes, two skips, and one unrelated
  dense-base CPU explicit-sharding failure.
- Launched matched 8,192-step scratch arms:
  - `/power/nest-moe-fixedep16-large-eqb-w512-cost-r45-coord`;
  - `/power/nest-moe-fixedep16-fixed25-eqb-w512-cost-r45-coord`;
  - `/power/nest-moe-fixedep16-fixed50-eqb-w512-cost-r45-coord`.
- Updates 0--1,600 are the controller gate. The arms continue in place only
  if all are finite, remain below 1% overflow through the sustained high-LR
  interval, and stay below 10% treatment step overhead.

### 2026-07-27 20:18 - Eligibility-QB sharding repair and clean relaunch

- The first r45 attempt failed before update 0. Building each group beta by
  taking `router_logits[0]` produced a length-one slice from a tensor sharded
  over the 64-device data/expert mesh. Explicit sharding correctly rejected
  that output shape.
- This failure produced no loss, routing, or architecture-quality observation.
  The stale retries were stopped.
- Replaced the sharded slice with a directly allocated 256-expert vector before
  scattering the compact group beta. Changed-file lint and the focused
  eligibility-QB contract pass.
- Relaunched the same three matched 8,192-update schedules under fresh r46 run
  identities:
  - `/power/nest-moe-fixedep16-large-eqb-w512-cost-r46-coord`;
  - `/power/nest-moe-fixedep16-fixed25-eqb-w512-cost-r46-coord`;
  - `/power/nest-moe-fixedep16-fixed50-eqb-w512-cost-r46-coord`.
- The preregistered sustained-peak controller gate remains updates 0--1,600.
  No data, initialization, optimizer, LR, topology, capacity, or evaluation
  setting changed.

### 2026-07-27 20:22 - Eligibility-QB token gather repair

- r46 reached the compiled train step but failed before update 0. Selecting a
  routing-mode bias row for every token left the gather output sharding
  ambiguous to JAX. No optimizer or routing observation was produced.
- Declared the token-by-expert gather output as sharded over the batch mesh
  axes and replicated over experts. Lint and the focused eligibility-QB
  lowering test pass.
- Stopped all r46 retries and launched the unchanged arms as r47. This is the
  final controller-lowering retry in this investigation; another
  controller-specific lowering failure ends the fixed-chain arm as blocked.

### 2026-07-27 20:28 - First valid eligibility-QB optimization

- r47 compiled and entered training. E256, fixed25, and fixed50 reached updates
  492, 441, and 447 with finite loss. Instantaneous overflow was zero,
  `0.0016%`, and `0.0018%`.
- Through roughly update 700, post-warmup median step time was 160.16 ms for
  E256, 159.70 ms for fixed25, and 162.61 ms for fixed50. These preliminary
  deltas are `-0.29%` and `+1.53%`; the frozen cost decision waits for update
  1,600.
- E256 briefly reached `1.063%` overflow at update 293 before returning to
  zero. Fixed25 and fixed50 maxima were `0.529%` and `0.336%`. The strict
  below-1% threshold is therefore technically missed by the untreated control,
  but there is no sustained collapse and no treatment-specific failure. Keep
  running for the discovery objective and report the transient explicitly.

### 2026-07-27 20:28 - Paloma rise diagnosis

- In the rotating E256 continuation, SlimPajama validation improved
  `4.69321 -> 4.66644 -> 4.64010` across the 8.59B, 12.89B, and 17.18B gates.
  Paloma macro moved `5.59726 -> 5.59614 -> 5.64185`; micro moved
  `5.57194 -> 5.57506 -> 5.62366`.
- The model continues to improve on its training distribution while degrading
  out of distribution. This rules out generic optimizer divergence as the
  primary explanation and points to the weights-only optimizer/data restart,
  finite-corpus replay, and narrow fixed Paloma slices.
- The original proxy used a five-update warmup and is not an optimizer-quality
  reference. r47 uses the heuristic MuonH/Adam peak rates and a matched
  512-update warmup in every scratch arm.

### 2026-07-27 20:33 - Eligibility-QB sustained-peak gate passed

- All r47 arms passed update 1,600 with finite loss and gradients.
- Over updates 512--1,600, median step time and maximum overflow were:
  - E256: `159.94 ms`, `0.0480%`;
  - fixed25: `159.80 ms`, `0.0367%`;
  - fixed50: `162.50 ms`, `0.0489%`.
- The fixed25 and fixed50 step deltas are `-0.084%` and `+1.60%`. Continue all
  arms to the first Paloma gate at update 2,048 and the 4.295B-token endpoint.
- The earlier one-step E256-control excursion to `1.063%` remains a disclosed
  formal threshold miss. It was not sustained and did not occur in either
  fixed-chain treatment, so it does not block the discovery objective.

### 2026-07-27 20:36 - First fixed-chain quality gate

- At 1.074B tokens, full-mode Paloma was `5.45031` for E256, `5.47354` for
  fixed25, and `5.72693` for fixed50. Treatment deltas are `+0.02323` and
  `+0.27662`.
- Fixed25 extracted E128 and E16 were `5.54035` and `5.77017`; fixed50 were
  `5.59894` and `5.75990`.
- Fixed25 is the leading branch: no measurable step cost, viable fixed
  submodels, and only a small early full-model penalty. Fixed50 gains only
  `0.01026` on E16 while giving up `0.27662` on the full model.
- The corrected E256 control is `0.17255` better than the original r25 E256 at
  the same token count. This is not a pure warmup ablation because EP topology
  and QB implementation also changed, but it confirms that the original
  five-update-warmup absolute curve is not an optimizer-quality reference.

### 2026-07-27 20:38 - Rotating continuation endpoint

- All four r31 arms completed 30,720 continuation updates, or 20.401B
  effective tokens.
- Final full-mode Paloma was approximately `5.674` for E256, `5.70816` for
  E128, `5.45195` for ladder25, and `5.56803` for ladder50. The E256 final
  evaluator log also reports micro `5.661` and SlimPajama validation `4.602`;
  its crashed W&B uploader did not preserve exact final scalars.
- Domain breadth rejects a general-quality interpretation:
  - E128 improved on 9/16, median `-0.00461`;
  - ladder25 improved on 7/16, median `+0.01741`;
  - ladder50 improved on 6/16, median `+0.16814`.
- Ladder macro gains remain concentrated in programming languages, gab, and
  TwitterAAE. Rotating miniatures worsened: ladder25 E128 offset 0 reached
  `6.13170`, ladder50 `5.68603`; final E32/E8/E1 values were above `7.27`.

### 2026-07-27 20:45 - Second fixed-chain quality gate

- At 2.147B tokens, full-mode Paloma was `5.30234` for E256, `5.28739` for
  fixed25, and `6.02401` for fixed50.
- fixed25 now improves the full model by `0.01496`; its E128 and E16 improve to
  `5.37677` and `5.49938`.
- fixed50 falls `0.72167` behind the control; its E128 and E16 are `5.90928`
  and `5.68742`. Overflow remains zero, so this is not a routing-capacity
  failure.
- Promote fixed25 as the only scale-up candidate. Continue fixed50 only to
  complete the preregistered curve.

### 2026-07-27 20:54 - Third fixed-chain quality gate

- At 3.221B tokens, full-mode Paloma was `5.21788` for E256, `5.18625` for
  fixed25, and `6.13546` for fixed50.
- Fixed25 improves full mode by `0.03163`; its fixed E128 and E16 checkpoints
  improve to `5.23026` and `5.35568`.
- Fixed50 is `0.91759` behind E256. Its E128 and E16 checkpoints are `6.06548`
  and `5.72020`.
- SlimPajama validation tells the same story: E256 `4.73528`, fixed25
  `4.71905`, fixed50 `5.55067`. The fixed50 reversal is therefore not a
  Paloma-only distribution artifact.
- All three arms remain finite with zero instantaneous overflow. The common
  optimizer is healthy in E256 and fixed25; fixed50's failure is specific to
  the 50%-restriction objective under this recipe.

### 2026-07-27 21:05 - Fixed-chain endpoint

- All three r47 arms completed 8,192 updates, or 4.295B tokens, without
  preemption.
- Full-mode Paloma was `5.17725` for E256, `5.13033` for fixed25, and
  `6.08237` for fixed50. Fixed25's delta is `-0.04692`; fixed50's is
  `+0.90512`.
- Fixed25 extracted E128 and E16 reached `5.18978` and `5.28666`. Fixed50
  reached `6.01412` and `5.68469`.
- Fixed25 improved on 12/16 Paloma domains with median delta `-0.04158`.
  Fixed50 lost on all 16 with median `+0.72534`.
- SlimPajama validation was `4.64909`, `4.60884`, and `5.45717`, confirming
  that fixed25's gain and fixed50's failure are not Paloma-only artifacts.
- Post-update-1,024 median step time was `161.23 ms` for E256, `161.50 ms`
  for fixed25, and `163.26 ms` for fixed50. Surcharges are `+0.17%` and
  `+1.26%`.
- Promote fixed25 to a longer, multi-seed, no-replay proxy with the production
  expert layout and a direct E128 cooldown. Do not promote fixed50 without
  separate optimizer tuning.

### 2026-07-27 20:00 - Third rotating-ladder continuation gate

- At 17.180B effective tokens, full-mode Paloma macro loss was:
  - E256 `5.64185`;
  - E128 `5.60826`, or `-0.03359` versus E256;
  - ladder25 `5.44328`, or `-0.19857` versus E256;
  - ladder50 `5.58798`, or `-0.05386` versus E256.
- Ladder25's full-model advantage persisted and widened again after narrowing
  from `-0.19652` at 8.59B to `-0.16344` at 12.89B. Ladder50 likewise
  recovered from `-0.01074` to `-0.05386`.
- Absolute checkpoint-to-checkpoint macro movement remains noisy on the fixed
  one-batch-per-domain Paloma slice. Paired arm deltas at the same checkpoint
  are the primary signal; domain-level medians are checked before attribution.
- The domain check materially narrows the macro interpretation:
  - E128 improved on 10/16 domains with median delta `-0.01635`;
  - ladder25 improved on 8/16 with median delta `+0.01882`;
  - ladder50 improved on 5/16 with median delta `+0.20203`.
- Ladder25's macro advantage is concentrated in programming languages
  (`-1.921`), gab (`-0.946`), and TwitterAAE (`-0.781`) versus E256. This is a
  specialization shift, not a broad full-model quality improvement.

### 2026-07-27 22:31 - Compute-optimal fixed25 burn-in amendment

- Opened Weaver issue `#662` for the matched longer run. No job from the
  superseded d640/50B draft was submitted.
- The operator supplied the production comparison point: compute budget
  `4.14e18`, hidden dimension `768`, sequence length `8192`, and heuristic
  target steps `2^15`.
- The current `MoeHeuristic` derives:
  - L8, six query heads, one KV head, E256 top-4, expert intermediate 384;
  - `4,414,492,433` target tokens, global batch 32, and 16,840 updates;
  - MuonH/AdamH-group LR `0.0083798354`, plain-Adam LR `0.0019338082`;
  - beta1 `0.9062`, beta2 `0.998001`, epsilon `1.2556433e-15`;
  - linear schedule, 1% warmup, minimum LR ratio `0.05`, and no clipping.
- The quoted epsilon estimate was `1.03e-15`; the run uses the heuristic's
  computed `1.2556433e-15`, following the instruction to derive the recipe
  from base inputs rather than override individual outputs.
- Compare only E256 and fixed25. Both use the canonical two-phase datakit mix,
  seed 0, capacity factor 1.25, and identical evaluation cadence. Fixed25
  restricts 25% of rows, alternating the fixed E128 and E16 subsets.
- Use full FSDP over 32 GB200s per arm with expert axis 1. This removes expert
  parallelism from the comparison; both arms use the same topology.
- Expected control Paloma macro loss is approximately `3.22`. Treat this as a
  calibration check, not an outcome threshold for the treatment comparison.

### 2026-07-27 23:13 - Burn-in launch

- The r1 jobs reached the canonical Datakit cache but failed before model
  initialization and step 0. fsspec consumed the injected renewable ADC, while
  TensorStore's native GCS driver attempted anonymous Zarr reads and returned
  HTTP 401. Both arms failed identically, so r1 contains no experimental data.
- Stopped the exact r1 coordinator trees after confirming the deterministic
  failure. Recorded the incident in
  `.agents/ops/2026-07-27-coreweave-tensorstore-gcs-auth.md`.
- Added process-start credential materialization: structured fsspec GCS ADC is
  written to a mode-0600 task-local file and exposed through
  `GOOGLE_APPLICATION_CREDENTIALS`. Credential contents are not logged.
- A CPU preflight on `cw-us-east-08a` opened the exact private Zarr offsets
  array through TensorStore with OAuth2. Job:
  `/power/nest-burn-001-tensorstore-adc-probe-r2`.
- Submitted the matched r2 pair at batch priority:
  `/power/nest-burn-001-e256-c4p14e18-r2-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-r2-coord`.

### 2026-07-27 23:31 - Lazy single-document packing and r3 launch

- The r2 pair passed native TensorStore authentication, allocated all 32
  devices per arm, and initialized W&B. It did not reach model initialization:
  `pack=1` eagerly read every document offset and length in the 10.37T-token
  Datakit store and materialized one Python `range` per document. Worker RSS
  passed 61 GB while still growing.
- Stopped both r2 coordinator trees before host OOM. Neither arm produced a
  training step, so r2 contains no scientific observation.
- Added an exact lazy path for single-document packing. It reads only the
  selected documents at batch time and preserves the existing left/right
  truncation, padding, and segment-ID semantics. Also made `with_pack`
  propagate into concatenated tail components.
- Regression tests pass: 29 tests across
  `tests/test_data_configs.py` and `lib/levanter/tests/test_packing.py`.
  Pre-commit and Pyrefly pass on the modified data-path files.
- The complete 168-component training mix instantiated in a 16 GB CPU
  preflight:
  `/power/nest-burn-001-datakit-pack1-probe-r3`.
- Submitted the fresh matched pair:
  `/power/nest-burn-001-e256-c4p14e18-r3-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-r3-coord`.

### 2026-07-27 23:42 - Production sharded-read coverage and r4 launch

- r3 reached its first actual dataset sample, where both arms failed before
  model initialization because the production `_ShardedJaggedArrayStore` did
  not expose the `get_batch` interface available on a materialized
  `JaggedArrayStore`. Iris began matched retries; both coordinator trees were
  stopped after one failure. No update was computed.
- Added batch reads to the sharded jagged-array view through its owning
  `TreeCache` and added production-shaped regression coverage. The focused
  suite now passes 30 tests; pre-commit and Pyrefly pass.
- A second 16 GB CPU preflight authenticated native GCS, instantiated all 168
  components, fetched a real sharded row, and emitted the expected packed
  8,192-token leaves:
  `/power/nest-burn-001-datakit-sample-probe-r4b`.
- Submitted the matched r4 pair:
  `/power/nest-burn-001-e256-c4p14e18-r4-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-r4-coord`.

### 2026-07-28 00:08 - CoreWeave-local Datakit relaunch

- The operator stopped r4 before update 0 after confirming that its absolute
  source pin read the canonical GCS store directly. Neither arm produced
  scientific data.
- An in-cluster S3 inventory confirmed the complete Datakit store at
  `s3://marin-us-east-02a/marin/datakit/store_8ac06c74`, including artifact
  metadata and the expected cluster partitions. No corpus copy is required.
- Replaced the GCS source pin with the CoreWeave-local S3 path and submitted
  the matched r5 pair at batch priority:
  `/power/nest-burn-001-e256-c4p14e18-r5-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-r5-coord`.
- Both coordinators use `MARIN_PREFIX=s3://marin-us-east-02a/marin`; training
  data, evaluations, checkpoints, and W&B replicas remain on CoreWeave S3.

### 2026-07-28 01:22 - First-step distributed hang

- Both r5 arms compiled and entered their first PJRT training dispatch but
  produced no update 0. All 16 task main threads have the same
  `pxla -> pjit -> train.py:780` stack, while their data-loader queues are now
  full.
- All 64 GPUs hold approximately 157.7 GiB and report 100% utilization at only
  190--235 W. Three-second InfiniBand samples show only a few identical
  keep-alive packets and no bulk traffic. NVLink fabric and NVML error status
  are healthy.
- NCCL RAS initially saw every rank and communicator `RUNNING/OK`. Subsequent
  global status collection accepts the local socket connection but times out,
  so the per-rank JSON collective counters are unavailable in the wedged
  state.
- Both environments contain the CUDA 12 and CUDA 13 NCCL wheels. E256 loaded
  the CUDA 13 build, while fixed25 loaded the CUDA 12.9 build despite identical
  package metadata. This makes the matched runtime nondeterministic and must be
  removed before the scientific run, but it cannot alone explain E256's
  CUDA-13 hang.
- The input stall on E256 task 1 preceded the hang but is not the continuing
  mechanism: all ranks eventually entered the executable and every prefetch
  thread is blocked on a full queue.
- No job has been kicked while live evidence is captured. The incident record
  is `.agents/ops/2026-07-28-gb200-first-step-collective-hang.md`.

### 2026-07-28 01:37 - Instrumented first-step reproduction

- Stopped the exact r5 coordinator trees after collecting stacks, RAS state,
  GPU/NIC telemetry, runtime mappings, and durable logs. Neither arm completed
  an optimizer update.
- Added deterministic CUDA 13 library precedence to the Iris GPU setup:
  reinstall both the installed cuDNN and NCCL CUDA-13 wheels after mixed CUDA
  packages are synced. The focused behavior tests cover both shared-library
  paths and pass.
- The first eight-step smoke, diagnostic r6, failed before its first dispatch
  because using eight steps for the Datakit simulated-epoch plan made a finite
  component empty. It was stopped before retry.
- `BURNIN_DATA_STEPS` now separates the Datakit planning horizon from the
  requested smoke length. Diagnostic r7 trains for eight steps against the
  full 16,840-step mix plan.
- r7 has loaded NCCL 2.28.9 with CUDA runtime 13.0 on all ranks. During FA4
  forward/backward compilation, RAS JSON reports four eight-rank
  communicators, identical `AllReduce=1` counts on every rank, no missing
  ranks, and a 1 ms collection with no timeout. GPUs are at 0% utilization
  while the Python main thread lowers FA4, distinguishing compilation from the
  r5 steady state of 100% GPU utilization inside the returned PJRT dispatch.

### 2026-07-28 01:52 - Collective transport isolation

- r7 reproduced the first-dispatch hang under deterministic CUDA 13 and NCCL
  2.28.9. Every rank remained at exactly `AllGather=147` and `AllReduce=6` on
  the 32-rank communicator across four RAS samples, with no missing rank,
  error, or count skew. GPUs were 100% busy at 202--236 W and no update or
  loss was emitted.
- NCCL INFO shows the executable selected NVLS SIMPLE for small collectives and
  RING LL128/SIMPLE over the P2P/MNNVL topology for large all-gathers. The
  evidence now points to a common GPU collective or transport path, rather
  than compilation, Datakit starvation, or one host entering a different
  collective.
- The reproductions span physical racks 392, 393, and 394. The resolved rack
  137 tray alert is unrelated.
- Stopped r7 after capturing its stable state. Launched two matched two-step
  isolates: r8 `nvls0` disables NVLS on rack 393, while r8 `mnnvl0` disables
  MNNVL on rack 392. Both retain the full 16,840-step Datakit planning horizon.
- Both r8 arms reproduced the same first-dispatch stall. Every rank again
  stopped at `AllGather=147`, `AllReduce=6` with no skew, missing rank, or
  error. Disabling either NVLS or MNNVL independently changes the selected
  algorithms but is insufficient.
- Stopped both r8 coordinator trees after capture. Submitted r9
  `/power/nest-burn-001-e256-c4p14e18-diag-transport0-r9-coord` with both
  `NCCL_NVLS_ENABLE=0` and `NCCL_MNNVL_ENABLE=0` for two optimizer updates.
- r9 reproduced the exact stationary `AllGather=147`, `AllReduce=6` state with
  both transport features disabled. It was stopped before retry.
- Submitted r10 with both features disabled plus `NCCL_ALGO=Ring` and
  `NCCL_PROTO=Simple`. Submitted r11 with default algorithms plus
  `NCCL_LAUNCH_ORDER_IMPLICIT=1` and `NCCL_LAUNCH_RACE_FATAL=1`; the loaded
  NCCL 2.28.9 library exposes both launch-order controls.
- r10 reproduced the same zero-update `147/6` state under Ring+Simple. Stopped
  it after capture. Protocol and algorithm selection are not sufficient to
  explain the failure; r11 is the active diagnostic.
- r11 reproduced the same state with implicit NCCL launch ordering enabled;
  launch-race fatal diagnostics remained silent. Stopped it after capture.
- Added `BURNIN_REPLICA_AXIS_SIZE` to make the burn mesh explicit. r12 uses
  Grug's standard small-model layout: eight data replicas with four-GPU
  node-local FSDP. This is mathematically matched to the original global FSDP
  arms but removes the 147 cross-node parameter all-gathers from each update.
- A one-node, four-GPU FSDP control reproduced the stall at
  `AllGather=147`, `AllReduce=3`, proving that multi-node transport is not
  required. The common trigger is the FSDP parameter-all-gather sequence.
- Stopped r12 and the single-node control after capture. r14 uses 32-way fully
  replicated data parallelism, retaining one sequence per GPU and the same
  global batch while eliminating parameter all-gathers.

### 2026-07-28 02:54 - FSDP issue filed and EP=16 recovery launched

- The nominally replicated r15 single-node diagnostic still retained sharded
  Grug parameters and reproduced the first-dispatch hang at
  `AllGather=66`, `ReduceScatter=1`, and `AllReduce=5`. r14 and r15 were
  stopped after the result made further replicated-mesh compilation redundant.
- Filed [#7694](https://github.com/marin-community/marin/issues/7694) with the
  deterministic CUDA 13 reproduction, single-node control, collective
  counters, transport-isolation matrix, rack evidence, and known-good EP
  comparison. This is distinct from the nondeterministic mid-training wedge in
  #7344.
- The original ladder gate used EP=64 across all 64 GPUs. True fixed E16
  nesting cannot use EP=64 without concentrating E16 traffic on one quarter of
  the expert ranks; the successful fixed-chain topology was EP=16/data=4 on
  64 GPUs. It completed 8,192 updates and 4.295B tokens.
- Added an explicit burn expert-axis override and validated both arm plans.
  The matched two-update recovery smokes use EP=16/data=2 on 32 GPUs per arm,
  the largest device count permitted by the fixed global batch of 32:
  `/power/nest-burn-001-e256-c4p14e18-diag-ep16-r16-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-diag-ep16-r16-coord`.
- Both child gangs allocated all eight four-GPU nodes with zero failures or
  preemptions. Promotion waits for finite updates from both arms.

### 2026-07-28 03:09 - Size-two FSDP reproduction and pure-EP retry

- Both r16 EP=16/data=2 arms compiled, initialized all communicators, and then
  reproduced the shallow first-dispatch hang before update 0. GPUs stayed at
  100% utilization and roughly 188--234 W.
- RAS returned in milliseconds with no missing rank, async error, or count
  skew. Every size-two data communicator stopped at 83 all-gathers. E256's
  32-rank communicator stopped at 64 all-gathers and six all-reduces;
  fixed25 stopped at 55 and five. Both expert groups reached 28 all-gathers
  and ten reduce-scatters.
- This localizes the remaining trigger to the size-two FSDP parameter shard,
  not expert parallelism. The evidence is published on
  [#7694](https://github.com/marin-community/marin/issues/7694#issuecomment-5099473868).
- Stopped both r16 trees without an optimizer update. Launched matched
  two-update r17 smokes on four nodes per arm with pure EP=16:
  `(replica=1, data=1, expert=16, model=1)`. This removes every FSDP parameter
  shard while preserving the model, global batch, data, optimizer, and
  architecture comparison.
- Active coordinators:
  `/power/nest-burn-001-e256-c4p14e18-diag-eponly-r17-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-diag-eponly-r17-coord`.

### 2026-07-28 03:33 - Pure EP reproduces; embedding-gather fix applied

- Both r17 EP=16/data=1 arms compiled and then reproduced the first-dispatch
  hang before update 0. The E256 16-rank communicator was aligned at 92
  all-gathers, ten reduce-scatters, and eight all-reduces. Four auxiliary
  size-four communicators were aligned at one all-reduce. RAS reported no
  missing rank or async error. Fixed25 had the same shallow PJRT stack and
  100% low-power GPU state.
- This disproves the narrower FSDP-only diagnosis. EP and data-axis changes
  alter the stalled collective sequence but do not restore progress.
- The `grug/embedding-gather-shard-map` branch contains a production-validated
  fix for a first-step rendezvous: replicate the token embedding and perform
  the lookup inside a batch-sharded `shard_map`. It also keeps the LM-head
  contraction shard on the data axis. Applied those layouts specifically to
  the MoE model; the shared Grug constants remain unchanged because the
  current tree's base model also consumes them.
- The nested MoE train-step lowering test and changed-file lint pass. The
  unrelated base-model runtime test still fails in its existing explicit
  sharding label concatenation path; the shared sharding module has no diff.
- Stopped r17 and submitted the identical matched two-update pure-EP smokes
  with the embedding fix:
  `/power/nest-burn-001-e256-c4p14e18-diag-embedfix-r18-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-diag-embedfix-r18-coord`.

### 2026-07-28 03:52 - Embedding fix ruled out; CuTe attention isolate

- E256 r18 compiled and reproduced the exact r17 stationary communicator
  counts: the 16-rank communicator reached 92 all-gathers, ten
  reduce-scatters, and eight all-reduces, while four size-four communicators
  each reached one all-reduce. The production embedding-gather change is
  valuable independently but does not clear this burn's hang. Stopped both
  r18 trees.
- The remaining compiler warning rematerializes `s32[32,2]` on device zero.
  That shape matches FA4-THD's Datakit packed-sequence metadata
  `[global_batch, max_segments]`, which is globally replicated before the THD
  custom call. The CuTe FA4 path instead encloses batch-sharded activations and
  bounds in a batch-axis `shard_map`.
- Ported the branch's replicated CuTe bound constants with a no-mesh guard.
  Focused attention and nested-MoE lowering tests pass (6 passed, 5
  hardware-dependent skips), as do changed-file lint and type checks.
- Submitted matched two-update r19 smokes on the pure EP=16 topology with only
  the attention implementation changed to `gpu_fa4_cute`:
  `/power/nest-burn-001-e256-c4p14e18-diag-cute-r19-coord` and
  `/power/nest-burn-001-fixed25-c4p14e18-diag-cute-r19-coord`.
- r19 failed before executable dispatch while compiling the CuTe backward
  kernel: CUTLASS DSL 4.6 removed `cute.make_fragment`, but the checked-in
  segmented backward kernel still called it. Stopped the retrying trees; this
  result is a source/dependency incompatibility, not an attention or
  collective result.
- Ported commit `5833e329ea99` from the supplied branch, replacing the four
  accumulator allocations with CUTLASS 4.6's `cute.make_rmem_tensor`. Focused
  tests and checks still pass. Submitted a single E256 two-update on-hardware
  API smoke:
  `/power/nest-burn-001-e256-c4p14e18-diag-cuteapi-r20-coord`.
- r20 passed CuTe kernel compilation, then exposed eight per-layer SPMD
  warnings for device-zero `s32[32,8192]` conditional outputs being scattered
  across the expert axis. A live thread dump showed
  `backend_compile_and_load`, not PJRT dispatch, so it was compiling rather
  than wedged. Stopped it once the warning was captured.
- Ported the supplied branch's precomputed FA4 bounds interface into the
  unrolled eight-layer MoE: compute long and sliding-window packed bounds once,
  explicitly batch-shard them, and attach the selected bounds to each layer's
  attention mask. Focused tests and checks pass.
- Submitted the replacement E256 two-update smoke:
  `/power/nest-burn-001-e256-c4p14e18-diag-cutebounds-r21-coord`.
- r21 cleared the collective failure and executed both train-step variants.
  Step 0 reported finite cross-entropy 11.7966, but the CuTe backward returned
  NaN/Inf gradients throughout the attention stack before the optimizer
  update. The next step therefore saw a non-finite loss and stopped at state
  step 2. This is a CuTe segmented-backward correctness failure, not
  learning-rate divergence.
- Returned to the THD backend used by the prior finite experiments. Its two
  compiled metadata validations wrap `[B,M]` segment lengths and prefix sums in
  `eqx.error_if`; the first is the exact `s32[32,2]` conditional output that
  XLA pins to device zero before scattering. Removed those redundant compiled
  checks because `ThdSegmentMetadata` is derived from validated packed segment
  IDs. The attention and nested-lowering test set passes (18 passed, 5 GPU
  skips), along with lint and type checks.
- Submitted the E256 two-update THD recovery:
  `/power/nest-burn-001-e256-c4p14e18-diag-thdvalidate-r22-coord`.

### 2026-07-28 04:40 - THD localization and literal-prefix correction

- r22 completed CUTLASS and XLA compilation, then reproduced the first
  executable-dispatch freeze. All four process stacks were parked in the same
  PJRT call.
- Two live NCCL RAS snapshots 20 seconds apart were identical. The 16-rank
  communicator had 134 all-gathers, 16 reduce-scatters, and nine all-reduces
  on every rank; four size-four communicators each had one all-reduce. There
  were no missing ranks, count skews, async errors, or RAS timeouts.
- The successful r47 run used the same JAX 0.11, CUDA 13, NCCL 2.28.9, and
  CUTLASS 4.6 runtime at sequence length 2,048. The failures in this burn all
  use sequence length 8,192. This rules out a runtime-version regression as
  the leading explanation and localizes the remaining failure to the
  sequence-8,192 THD executable.
- The prior fixed-chain implementation reused deterministic nested sets, but
  they were evenly interleaved across expert ranks. The requested treatment is
  literal prefix nesting: E16 uses experts 0--15 and E128 uses 0--127. Added a
  prefix schedule and made training, eligibility-QB, evaluation, and model
  extraction use the same prefix.
- Literal prefixes are incompatible with balanced EP=16 dispatch: contiguous
  E16 weights occupy one expert rank and exceed the ring's per-rank capacity.
  The prefix burn therefore requires `expert=1` and uses full FSDP over 16
  devices. Prefix eligibility and extraction behavior tests pass, as do
  changed-file lint and type checks.
- Ported the supplied branch's cuDNN fused-attention fallback. Because the burn
  uses pack=1, disabling the redundant cross-document attention mask preserves
  valid-token semantics while allowing the O(sequence) cuDNN path; padding
  remains excluded from the loss.
- Stopped r22 and submitted a matched two-update, sequence-8,192,
  full-FSDP smoke:
  - `/power/nest-burn-001-e256-c4p14e18-diag-cudnnprefix-r23-coord`;
  - `/power/nest-burn-001-fixed25-c4p14e18-diag-cudnnprefix-r23-coord`.

### 2026-07-28 05:06 - r23 control compiler wedge

- The fixed25 r23 arm continued through first-step tracing. The E256 arm
  remained in initial-state `backend_compile_and_load` on all four ranks for
  more than 20 minutes and produced no optimizer step.
- A 20-second `/proc/1/stat` sample on E256 rank 0 accumulated seven CPU ticks,
  approximately 0.35 cores, while the same sample on fixed25 accumulated 2,446
  ticks, approximately 122 cores. The E256 processes were sleeping in
  `futex_do_wait`; this was a wedged compiler future, not merely a slower
  compile.
- An Iris gang kick was accepted but had not applied after two controller
  ticks. Stopped only the E256 r23 coordinator and resubmitted the identical
  two-update control as
  `/power/nest-burn-001-e256-c4p14e18-diag-cudnnprefix-r24-coord`.
  Fixed25 r23 continues unchanged. Neither arm has produced a quality
  observation.

### 2026-07-28 05:13 - cuDNN rejection and reference fallback

- Fixed25 r23 reached executable creation after 5:14 of first-step work and
  failed on every rank with
  `cudnn_frontend: No valid execution plans built`. The subsequent
  coordinator-connection failures were gang teardown. Retries are
  deterministic and produced no optimizer update.
- Stopped fixed25 r23 and E256 r24. Added the existing `reference` backend to
  the burn launcher and submitted a matched two-update sequence-8,192 smoke:
  - `/power/nest-burn-001-e256-c4p14e18-diag-reference-r25-coord`;
  - `/power/nest-burn-001-fixed25-c4p14e18-diag-reference-r25-coord`.
- This fallback changes only the attention implementation. If reference
  attention cannot fit or execute at sequence 8,192, fall back to the
  sequence-2,048 cell already demonstrated by r47 rather than add another
  unvalidated kernel.

### 2026-07-28 05:34 - Sequence-8,192 smoke passed and r26 promoted

- Both reference-attention r25 arms completed two optimizer updates with
  finite losses and gradients:
  - E256: step-1 CE `11.744956`, gradient norm `0.949056`;
  - fixed25: step-1 CE `11.745689`, gradient norm `0.950185`.
- Both arms reported zero mean capacity overflow. The step-1 loss delta is
  `+0.000733` nats for fixed25; two scratch updates are a numerical sanity
  check, not a quality result.
- Submitted the full 16,840-update, 4.414B-token matched pair:
  - `/power/nest-burn-001-e256-c4p14e18-reference-r26-coord`;
  - `/power/nest-burn-001-fixed25-c4p14e18-reference-r26-coord`.
- Posted the gate result and W&B links to PR 7667:
  <https://github.com/marin-community/marin/pull/7667#issuecomment-5100361541>.
- Reference attention preserves the exact scientific comparison but may not be
  operationally viable. Use repeated r26 updates after compilation to forecast
  completion time and retain backend startup overhead separately from matched
  optimizer-step overhead.

### 2026-07-28 05:51 - r26 steady-state forecast

- Both full burns passed 100 updates with finite loss and gradients and zero
  mean capacity overflow.
- Over matched updates 20--100:
  - E256 median step time `382.754 ms`, p90 `396.351 ms`, median throughput
    `684,888 tokens/s`;
  - fixed25 median step time `378.146 ms`, p90 `392.710 ms`, median throughput
    `693,235 tokens/s`.
- Fixed25 is `1.20%` faster by median step time in this early window; the
  measured co-training surcharge is therefore zero within run noise.
- At update 100, full-mode train CE was `9.561832` for E256 and `9.565185` for
  fixed25, a treatment delta of `+0.003352` nats.
- Excluding compile, checkpoint, and evaluation overhead, the measured medians
  project to `1.790` optimizer hours / `28.65` GPU-hours for E256 and `1.769`
  optimizer hours / `28.30` GPU-hours for fixed25 across all 16,840 updates.
  Keep the operational ETA separate until the first update-1,000 Paloma
  evaluation completes.

### 2026-07-28 06:01 - Gate 1 passed

- At matched update 1,000:
  - E256 full Paloma `6.752973`, uncheatable `6.463064`;
  - fixed25 full Paloma `6.795202`, uncheatable `6.502627`;
  - fixed25 E128 Paloma `6.818607`, uncheatable `6.555846`;
  - fixed25 E16 Paloma `6.929915`, uncheatable `6.711246`.
- Fixed25's full-mode Paloma delta is `+0.042229` nats, below the
  preregistered `+0.10` stop threshold. E128 is `+0.023406` behind the
  treatment full mode; E16 is `+0.134714` behind.
- Over matched updates 100--900, E256 median/p90 step time was
  `361.991/373.602 ms` at `724,173 tokens/s`; fixed25 was
  `357.296/374.399 ms` at `733,689 tokens/s`. Fixed25's median delta is
  `-1.30%`; both maximum mean capacity-overflow rates were zero.
- The update-1,000 evaluation hook took `81.90 s` for control and `244.59 s`
  for fixed25 because the treatment evaluates three modes. This is measurement
  overhead, not intrinsic co-training cost. Use later checkpoints to separate
  one-time eval compilation from repeated evaluation.
- Posted the Gate 1 result to PR 7667:
  <https://github.com/marin-community/marin/pull/7667#issuecomment-5100541325>.

### 2026-07-28 06:16 - Update 2,000 remains inside the quality gate

- At matched update 2,000:
  - E256 full Paloma `6.516537`, uncheatable `6.165142`;
  - fixed25 full Paloma `6.575028`, uncheatable `6.250004`;
  - fixed25 E128 Paloma `6.586526`, uncheatable `6.270319`;
  - fixed25 E16 Paloma `6.644937`, uncheatable `6.334846`.
- Fixed25's full-mode Paloma delta is `+0.058492` nats. This is the second
  checkpoint below the preregistered `+0.10` boundary. E128 is only
  `+0.011497` behind the treatment full mode; E16 is `+0.069909` behind.
- Repeated evaluation took `75.36 s` for E256 and `233.24 s` for fixed25.
  The approximately threefold treatment evaluation time tracks its three
  separately measured modes. It must not be counted as intrinsic co-training
  cost.
- Both arms remained finite with zero mean capacity overflow. The control
  reached update 3,000 before the treatment because the experimental
  evaluation suite intentionally measures two additional treatment modes.
  Compare quality only at aligned updates.

### 2026-07-28 06:26 - Fixed25 leads at update 3,000

- At matched update 3,000:
  - E256 full Paloma `6.693682`, uncheatable `6.332820`;
  - fixed25 full Paloma `6.500750`, uncheatable `6.143626`;
  - fixed25 E128 Paloma `6.576565`, uncheatable `6.217495`;
  - fixed25 E16 Paloma `6.584426`, uncheatable `6.250178`.
- Fixed25's full mode is now `0.192932` nats better than control. This resolves
  the unmatched update-3,000 control point: the control's held-out regression
  is not caused by the evaluator advancing to a new sample, because
  `TaggedEvaluator` reconstructs an iterator over the same deterministic,
  bounded dataset on each call.
- The result is consistent with a treatment-specific regularization or
  optimization-stability benefit, but one checkpoint is not enough to assign
  mechanism. Both arms remain finite and their router z-loss rises; fixed25's
  update-3,000 router z-loss (`2,833`) is not lower than control (`2,488`).
  Continue through later aligned evaluations before claiming stabilization.

### 2026-07-28 06:38 - Update 4,000 confirms a small fixed25 lead

- At matched update 4,000:
  - E256 full Paloma `6.438949`, uncheatable `6.084310`;
  - fixed25 full Paloma `6.426967`, uncheatable `6.040575`;
  - fixed25 E128 Paloma `6.464280`, uncheatable `6.084595`;
  - fixed25 E16 Paloma `6.511320`, uncheatable `6.127030`.
- Fixed25's full mode is `0.011982` nats better than control. The large
  update-3,000 advantage narrowed after control recovered, but the ordering
  persists. Fixed25 Paloma has decreased at every checkpoint; control's
  update-3,000 increase was a transient optimization oscillation at this
  high-LR point, not a persistent loss trend.
- Submitted matched two-update WildChat SFT smokes against temporary
  update-3,001 checkpoints. E256 referenced a checkpoint already pruned by the
  running checkpointer. Fixed25 discovered its checkpoint before building the
  missing Marin-template WildChat cache, but the pretraining checkpointer
  pruned it before the delayed weight load. Neither arm loaded weights. The
  fixed25 attempt did finish the shared in-region chat cache, so endpoint SFT
  can start from a permanent final checkpoint without the same data delay.
  This is an operational preflight, not a post-training result.
### 2026-07-28 06:55 - Aligned 5k and historical d768 comparison

- The matched r26 burns remain healthy on four four-GPU GB200 nodes per arm
  with deterministic CUDA 13, NCCL 2.28.9, reference attention, and full
  FSDP. Neither arm has failed, retried, overflowed, or produced a non-finite
  update.
- At 1.311B aligned tokens, fixed25 full-mode Paloma is `6.369439` versus
  `6.358023` for E256, a `+0.011416` treatment regression. The fixed25 E128
  and E16 modes are `6.403664` and `6.446313`. The transient full-mode
  advantage at step 3000 therefore did not persist through step 5000.
- The historical `moe_may_compute_opt_d768_ep1` run does report final Paloma
  macro loss `3.227273` at 4.424B tokens. It is not a matched control: it used
  `meta-llama/Meta-Llama-3.1-8B` tokenization, a different training mix,
  sequence length 4096, global batch 64, and the older Grug model/runtime
  contract. The current pair uses the Marin tokenizer, Datakit mix, sequence
  length 8192, and global batch 32. Its value remains a useful historical
  quality reference but is not an acceptance threshold for the treatment
  delta.
- The current worktree already contains the production branch's substantive
  embedding fix: replicated token embeddings and a replica-local
  `shard_map` gather. The pure-EP r18 diagnostic still froze after that
  change. THD then froze at sequence length 8192, CuTe produced non-finite
  backward gradients, and cuDNN found no execution plan; reference attention
  completed the matched smoke and sustained burn. This localizes the
  recovered failure to fused-attention execution at this shape more strongly
  than to CUDA family, NCCL transport, embedding gather, or FSDP topology.

### 2026-07-28 07:04 - SFT caches and independent diagnostics PR

- The fixed25 cache preflight materialized the canonical-thinking chat cache
  at
  `s3://marin-us-east-02a/marin/documents/nemotron_science_think-bae881d-a0f2bb/_chat_cache/2defa2/train`.
  It then reached the expected missing temporary-checkpoint error because
  step 4001 had been pruned. The cache-only coordinator was stopped before a
  redundant retry. WildChat and canonical-thinking inputs are now both
  prebuilt in-region for endpoint SFT.
- Published the NCCL RAS client and distributed-diagnostics design separately
  in [PR #7699](https://github.com/marin-community/marin/pull/7699). The
  research worktree no longer carries those files. The PR deliberately leaves
  the ProfileTask RPC bundle and Grafana stall alert as follow-up slices.
- At aligned update 6000, fixed25 full-mode Paloma is `6.316322` versus
  `6.270247` for E256, a `+0.046074` regression. Fixed25 E128 and E16 are
  `6.350322` and `6.398504`. This remains below the preregistered `+0.10`
  threshold and does not count toward the two-consecutive-evaluation stop
  condition.

### 2026-07-28 07:15 - First quality strike at update 7000

- At aligned update 7000, fixed25 full-mode Paloma is `6.398132` versus
  `6.224376` for E256, a `+0.173756` regression. This is the first
  preregistered quality strike. The stop rule requires two consecutive aligned
  regressions above `0.10`, so both healthy arms continue to update 8000.
- Fixed25 E128 and E16 are `6.348491` and `6.400418`. E128 is `0.049641`
  better than the treatment's full mode at this checkpoint, while E16 is
  effectively tied. The full-mode regression therefore is not a uniform
  degradation across the fixed hierarchy.
- Decoded cache row zero directly inside the live E256 coordinator with the
  configured `marin-community/marin-tokenizer`. The row contains 838 valid
  tokens, starts with `<|begin_of_text|>`, and decodes to coherent English
  accessibility documentation. The cache has 588,032,775 rows and token IDs
  stay inside the 128,256-token vocabulary. The cache metadata also names the
  Marin tokenizer. This rules out obvious tokenizer/cache corruption as the
  explanation for the poor absolute Paloma curve, but not a data-mixture or
  current model/evaluator-contract difference.

### 2026-07-28 07:31 - Update 8000 recovers and continuation remains open

- At aligned update 8000, fixed25 full-mode Paloma is `6.235766` versus
  `6.186363` for E256, a `+0.049403` regression. This breaks the consecutive
  strike sequence, so the preregistered stop rule does not fire.
- Fixed25 E128 and E16 are `6.254466` and `6.332193`. Uncheatable macro loss is
  `5.779078` for E256, `5.818924` for fixed25 full, `5.829023` for E128, and
  `5.903736` for E16.
- On the 16 aligned Paloma domains, fixed25 is better only on Wikitext. Mean
  delta is `+0.049403`, median `+0.037570`, and the largest regressions are PTB
  (`+0.178120`) and Twitter AAE (`+0.106899`).
- Through the common update-8307 timing horizon, both arms contribute 7,284
  post-warmup samples. Median compiled-step time is `366.695 ms` for E256 and
  `370.473 ms` for fixed25, a `+1.03%` surcharge. Block-bootstrap intervals
  overlap narrowly. Overflow remains zero.
- Updated Weaver report and chart artifacts to revision 3 and posted the
  aligned milestone to PR 7667:
  <https://github.com/marin-community/marin/pull/7667#issuecomment-5101235418>.

### 2026-07-28 07:39 - Update 9000 spike and SFT epoch sizing

- At aligned update 9000, fixed25 full-mode Paloma is `6.196991` versus
  `6.508448` for E256, a `-0.311457` treatment delta. Fixed25 E128 and E16
  are `6.226601` and `6.282750`. The apparent treatment win is dominated by
  a transient control spike: E256 is `6.186363` at update 8000,
  `6.097732` at update 10000, and `6.052866` at update 11000. It is evidence
  against a sustained treatment regression, not evidence for a stable
  0.31-nat gain.
- Through the common update-9134 timing horizon, median compiled-step time is
  `365.798 ms` for E256 and `370.991 ms` for fixed25, a `+1.42%`
  surcharge. Both arms remain finite with zero overflow, task failures,
  retries, or preemptions.
- Read the already-built CoreWeave S3 chat-cache ledgers from a live task pod.
  WildChat contains `537,585,868` tokens (`385,700` rows), resolving one
  packed epoch to 2,051 updates at batch 32 and sequence length 8192.
  Canonical thinking contains `1,318,244,179` tokens (`708,920` rows),
  resolving one epoch to 5,029 updates. At the measured pretraining step
  rate, the optimizer portions are approximately 13 and 31 minutes per arm,
  respectively, before compilation and checkpoint overhead.

### 2026-07-28 07:52 - Update 10000 returns to the paired trend

- Fixed25 full-mode Paloma is `6.162987` versus `6.097732` for E256, a
  `+0.065255` treatment regression. Fixed25 E128 and E16 are `6.192197` and
  `6.216753`. This remains below the stop boundary and follows a treatment
  win at update 9000, so no consecutive strike exists.
- Across the ten aligned gates, fixed25 wins three. The mean delta is
  `-0.006975`, dominated by the update-3000 and update-9000 control spikes;
  the paired median is a `+0.044152` treatment tax.
- Through the common update-10161 timing horizon, median compiled-step time is
  `364.547 ms` for E256 and `371.553 ms` for fixed25, a `+1.92%`
  surcharge. Published revision 4 of the Weaver interim report and all three
  chart artifacts.

### 2026-07-28 09:15 - Pretraining endpoint and post-training handoff

- Both matched pretraining arms completed 16,840 updates and 4.4145B tokens
  without a task failure, retry, preemption, non-finite update, or routing
  overflow. E256 took 2h57m of charged child time; fixed25 took 3h38m because
  every treatment gate evaluated full, E128, and E16 modes.
- Endpoint Paloma macro loss is `5.841133` for E256, `5.818610` for fixed25
  full mode, `5.860195` for E128, and `5.901186` for E16. Endpoint
  uncheatable macro loss is `5.329` for E256, `5.307` for fixed25 full,
  `5.357` for E128, and `5.381` for E16. The treatment full model therefore
  finishes `0.022523` better on Paloma, while E128 and E16 finish `0.019062`
  and `0.060053` behind the control.
- Across all 17 aligned Paloma gates, the treatment wins four. Mean paired
  delta is `+0.004950` and median delta is `+0.037985`; the endpoint win does
  not justify a regularization claim with one seed and oscillatory
  intermediate gates. At the endpoint, fixed25 is better on 10 of 16 Paloma
  domains, with mean delta `-0.022523` and median delta `-0.008501`.
- Across the full common post-warmup horizon, median compiled-step time is
  `368.296 ms` for E256 and `371.060 ms` for fixed25, a `+0.750%` surcharge.
  The bootstrap intervals overlap. This projects to `62.44` versus `62.91`
  GPU-hours per 10B tokens, or `0.47` extra GPU-hours for fixed nesting. The
  W&B charged runtimes are `47.01` and `58.02` GPU-hours; almost all of that
  11.01-GPU-hour difference is the deliberately tripled treatment evaluation
  suite (`3,835` versus `1,338` logged hook seconds).
- E256 WildChat SFT completed 2,051 updates from the permanent pretraining
  checkpoint in 25m38s with a permanent final checkpoint and console loss
  `6.09`. E256 thinking SFT loaded that exact WildChat step-2051 checkpoint,
  cleared its three-update liveness gate after the expected 10m54s compile,
  and is training. fixed25 WildChat SFT started from its permanent pretraining
  endpoint; its thinking stage is success-gated behind WildChat.

### 2026-07-28 10:28 - Matched SFT completes

- All four SFT jobs completed without failure, retry, preemption, non-finite
  loss, or routing overflow. Both thinking stages committed permanent
  step-5029 checkpoints and finished their W&B sync.
- On WildChat, fixed25 post-warmup completion-masked training
  cross-entropy averages `6.660172` versus `7.020264` for E256, a
  `-0.360092` paired delta. fixed25 is lower on 1,758 of 1,951 paired
  post-warmup batches (90.1%). Last-100 mean loss is `6.039649` versus
  `6.488408`.
- On the thinking stage, fixed25 averages `4.745278` versus `4.864716`, a
  `-0.119438` paired delta. fixed25 is lower on 4,928 of 4,929 paired
  post-warmup batches (99.98%). Last-100 mean loss is `4.425439` versus
  `4.531615`.
- WildChat median optimizer steps are `319.683 ms` for E256 and `319.458 ms`
  for fixed25. Thinking medians are `321.850 ms` and `317.169 ms`.
  Restricted routing is disabled during SFT, so these small negative
  differences are hardware variation, not a treatment speedup. Combined
  optimizer estimates are 10.108 and 10.001 GPU-hours.
- The SFT result tests transfer of the full E256 checkpoint after nested
  pretraining. It does not test post-training of extracted E128/E16
  checkpoints or held-out agentic behavior. The final report states this
  boundary explicitly.
- Published the final burn report and SFT chart as Weaver artifacts:
  <https://loom.rjp.io/s/wk4wnbee/artifacts/burnin-final> and
  <https://loom.rjp.io/s/wk4wnbee/artifacts/burnin-sft>.
- The branch-wide pre-commit pass and Pyrefly complete without error. The
  combined Grug/data/cache regression suite reports 97 passed and 2 skipped.
  Lint review identified and prompted fixes for a URL-prefix join, stale
  launcher descriptions, duplicate assertions, an unused synchronous cache
  method, checkpoint-loader typing, nested-size normalization, and a breakout
  source-model router-state mismatch.

### 2026-07-28 15:19 - NEST-BURN-002 100B-token extension starts

- Preregistered a fresh 99,999,547,392-token comparison in
  `.agents/projects/662-nested-moe-24h-burnin.md`. Each arm runs 95,367
  updates at sequence length 8,192 and global batch 128 on 64 GB200s. A
  four-way replica axis preserves the prior two sequences per device. The
  100B-derived MuonH/AdamH LR is `0.00488646`, plain-Adam LR is `0.00112764`,
  beta2 is `0.992028`, and epsilon is `2.98810e-15`.
- Started from scratch instead of resuming the 4.414B-token checkpoints.
  Resuming while changing batch 32 to 128 would cross an optimizer-schedule
  discontinuity. This d768 run is about 23 times beyond its compute-optimal
  token count and tests long-horizon treatment behavior, not compute-optimal
  scaling quality.
- r1 used an invalid descriptive version label and r2 lacked S3 credentials
  on the main-cluster coordinator. Neither attempt allocated a GPU or
  initialized W&B. r3 correctly pinned the complete root job to 08a, but the
  main controller reported `cw-us-east-08a` unreachable with a capacity
  heartbeat stale by about 50 minutes. Both r3 roots remained in
  `QUEUED_HANDOFF` and were terminated before delivery.
- Commit `3e09affbf75af0c19055190064903e2b2fecfb4b` removes the invalid
  child-level cluster pin. Iris federates complete root job trees, so GPU
  children must stay local to a federated coordinator.
- Submitted r4 directly to `cw-us-east-08a` at batch priority:
  - `/power/nest-burn-002-e256-100b-b128-r4-coord`;
  - `/power/nest-burn-002-fixed25-100b-b128-r4-coord`.
- Both child gangs have 16/16 workers running, with zero failures and
  preemptions. Every worker is loading the regional Datakit cache from
  `s3://marin-us-east-02a/marin/datakit/store_8ac06c74`. The next gate is
  successful compilation and finite matched updates.
- Consolidated the prior Weaver reports and charts with the r4 launch links in
  <https://github.com/marin-community/marin/pull/7667#issuecomment-5105929440>.

### 2026-07-28 15:51 - Replace wedged E256 startup

- fixed25 cleared compilation and reached step 1,121 with finite loss and zero
  routing overflow. Across steps 100-884, median throughput is 2.536M
  tokens/s, or 413.6ms per 1,048,576-token update. That projects to 10.96
  optimizer-hours for 100B tokens before evaluation and checkpoint overhead.
- fixed25 also recorded an 82.6s data-loader stall at step 641, then resumed.
  Steady-state compiled-step cost and end-to-end elapsed cost will therefore
  be reported separately.
- Every sampled E256 r4 rank remained in JAX
  `backend_compile_and_load` for `threefry_split` after more than 27 minutes.
  Three ranks had identical stacks, each node reported 0% GPU utilization,
  and rank 0 accumulated only 0.8 CPU-seconds over 128 wall-seconds. This was
  a wedged startup compilation, not a training collective or an architecture
  result.
- Submitted the identical E256 control as r5. Once its 16-node gang was
  running, terminated r4 and released its 64 GB200s. r5 cleared the RNG-key
  stage in under two minutes and entered train-step compilation on all ranks.
  The treatment arm was not interrupted.
- The independent stale-federation fault is tracked in
  <https://github.com/marin-community/marin/issues/7705> and
  `.agents/ops/2026-07-28-iris-08a-federation-heartbeat-stale.md`.

### 2026-07-28 16:08 - Both arms train; fixed25 recovers one failed gang

- E256 r5 reached step 510 with finite loss, zero overflow, and median
  post-step-100 throughput of 2.575M tokens/s across the first 137 steady
  samples. fixed25's corresponding median through step 2,501 is 2.558M
  tokens/s. The preliminary compiled-step surcharge is 0.66%.
- The fixed25 step-2,500 evaluation reported Paloma macro loss `6.065558` in
  full mode, `6.088614` for E128, and `6.212317` for E16. The 77.42s logged
  hook includes all three modes.
- The fixed25 gang then lost XLA coordination at 16:04 UTC:
  `WatchTasksAsync ... 10.186.213.145:8476: Connection refused`. Iris
  recorded one failed task-0 attempt and cascaded all 16 ranks into attempt 1.
  The replacement gang restored from its latest temporary checkpoint near
  step 1,550 and resumed training. This is a recoverable infrastructure
  interruption; replayed elapsed time will be separated from optimizer-step
  cost.

### 2026-07-28 16:31 - Fixed25 fails the first long-horizon stability gate

- E256 r5 completed its step-2,500 evaluation and remained finite through
  step 2,575. fixed25 r4 instead produced a non-finite loss at state step
  2,503 after returning from the repeated full/E128/E16 evaluation. The
  control's survival past the identical step makes this treatment-path
  specific under the 64-GB200, batch-128 topology.
- The fixed25 child accumulated three failed attempts: two XLA coordination
  service connection failures followed by the explicit non-finite loss. Iris
  restored the temporary checkpoint and reproduced the same boundary. The
  fixed25 retry loop was stopped after the numerical failure so it would not
  repeatedly replay 1B tokens.
- At matched update 2,500, full-mode Paloma macro loss is `6.557286` for E256
  and `6.065558` for fixed25, a `-0.491728` treatment delta. fixed25 is better
  on 15 of 16 Paloma domains, while its E128 and E16 losses are `6.088614` and
  `6.212317`. This unusually large one-gate improvement is paired with an
  unstable optimizer trajectory and is not promotion evidence.
- Across 1,478 matched compiled updates through step 2,501, median step time
  is `381.952 ms` for E256 and `409.075 ms` for fixed25, a `+7.10%`
  surcharge. The single full-mode control evaluation took `26.14 s`; the
  three-mode fixed25 evaluation took `77.42 s`. The evaluation premium is
  measurement cost, not co-training cost.
- Launched
  `/power/nest-burn-002-fixed25-100b-b128-r6-noeval2500-coord` from scratch
  with the same model, seed, optimizer, data, batch, and 100B horizon, changing
  only the first evaluation boundary from step 2,500 to step 10,000. If r6
  passes step 2,503, the triple nested evaluation or its interaction with
  restore is implicated. If it fails at the same update, the nested training
  trajectory itself is unstable in this cell.

### 2026-07-28 17:00 - Deferred-evaluation fixed25 passes step 2,503

- fixed25 r6 reached step 2,537 with finite loss, finite gradient norm, zero
  routing overflow, and no failed task attempt. Its first 290 matched losses
  reproduce r4 within `0.00076` nats.
- Deferring the full/E128/E16 callback is the only configuration change. The
  uninterrupted nested trajectory therefore does not deterministically fail
  at step 2,503. The remaining causes are the multi-mode evaluation callback,
  restored train state, or their interaction.
- r6 continues to its first three-mode evaluation at step 10,000. An immediate
  post-evaluation failure will implicate the callback; survival will implicate
  r4's restored or gang state.
- A matched ten-step XPlane pair was captured at steps 128--138 and the two
  64-GB200 profiling gangs were stopped after upload. The control and treatment
  landed on different leafgroups. Control all-gather time was about eight
  times treatment all-gather time, while treatment reduce-scatter was slower.
  This cross-rack communication variance overwhelms the architecture signal,
  so the profiles are retained for debugging and excluded from cost
  attribution.

### 2026-07-28 18:18 - Clean fixed25 passes the three-mode evaluation gate

- fixed25 r6 completed full/E128/E16 evaluation at update 10,000 and remained
  finite through update 10,268 with zero overflow, failures, or preemptions.
  The clean gate therefore rules out both the uninterrupted treatment
  trajectory and the evaluation callback alone. The r4 non-finite boundary
  requires its prior coordination failures, restored state, or their
  interaction.
- Matched update-10,000 Paloma macro loss is `6.281861` for E256 and
  `5.817962` for fixed25 full, a `-0.463899` treatment delta. fixed25 is better
  on all 16 domains. Its E128 and E16 modes are `5.796685` and `6.028710`.
  This is one clean paired gate, not promotion evidence.
- Across 9,291 matched post-warmup updates, median compiled step time is
  `384.600 ms` for E256 and `397.876 ms` for fixed25, a `+3.45%` surcharge.
  The corresponding 100B optimizer forecast is `10.188 h` versus `10.540 h`,
  or 22.51 additional GB200-hours.
- Control Paloma is non-monotonic: `6.557286`, `5.627037`, `5.967525`,
  `6.281861`, and `5.978082` at updates 2,500 through 12,500. Median training
  loss over the preceding 200 updates moves in the same direction, so this is
  a model-trajectory signal rather than an evaluator-only artifact. Router
  entropy and z-loss do not move monotonically with Paloma, so routing is not
  identified as the cause.
- Updated the committed interim report and added clean-run training-loss,
  Paloma, step-time, and machine-readable result assets.

### 2026-07-28 18:26 - Nested evaluation perturbs treatment; resume pre-eval

- Visual inspection of the clean-run loss curve revealed a treatment-specific
  jump immediately after the update-10,000 nested evaluation. fixed25 median
  loss is `4.69196` over updates 9,900--9,999, then `7.53152`, `7.30613`, and
  `6.94942` over the next three 100-update windows. E256 falls from `5.49351`
  to `5.25149` across its one-mode evaluation.
- The clean callback did not reproduce r4's immediate NaN, but it did perturb
  the treatment materially. The exact severity depends on prior gang or
  optimizer state. The update-10,000 Paloma result remains valid because
  evaluation computes it before the post-callback training updates.
- The temporary checkpoint root still contained `step-9936`, before nested
  evaluation. Stopped fixed25 r6 before the next ten-minute checkpoint could
  replace it, then resubmitted the same run identity and artifact version with
  `BURNIN_EVAL_INTERVAL=1000000`. It will restore update 9,936 and continue
  without periodic evaluation; the forced terminal evaluation occurs only
  after the final optimizer update.

### 2026-07-28 19:25 - No-eval counterfactual isolates callback perturbation

- The 64-GPU replacement could not reacquire its gang. At the original
  64-CPU/512-GiB worker request Kueue could fit 2/16 workers. A live control
  worker used 1.86 CPU cores and 57.84 GiB RSS, so the launcher now accepts
  explicit per-node CPU and RAM reservations. At 16 CPU/80 GiB, memory no
  longer excluded nodes, but 199/202 GB200 nodes were occupied.
- Used the two available nodes for a bounded 8-GPU counterfactual. It restored
  the clean update-9,936 checkpoint, crossed update 10,000 with periodic
  evaluation disabled, and held loss near `4.86`--`5.08` through update
  10,100. The contaminated run was `6.95`--`7.53` over the corresponding
  windows. The treatment trajectory does not jump without nested evaluation.
- The diagnostic saved clean temporary checkpoint `step-10114` and was
  stopped. Queued a 64-GPU continuation from that checkpoint with periodic
  evaluation disabled. The 178 updates computed on an 8-GPU reduction
  topology are excluded from cost estimates and remain a quality caveat.

### 2026-07-28 20:02 - Same-topology pair confirms nested-evaluation failure

- Copied the clean state-step-10,114 checkpoint into two new run identities
  on identical two-node, 8-GB200 meshes. The arms shared global batch 128,
  optimizer state, data offset, model, and CUDA/JAX stack. Losses over global
  steps 10,114--10,119 matched within `0.00866` nats maximum absolute error.
- The treatment ran full/E128/E16 evaluation after global step 10,119. It
  logged losses `5.12179` and `5.53051` on the next two updates, then raised
  `FloatingPointError: Non-finite loss at step 10123` before logging global
  step 10,122.
- The no-evaluation arm logged finite losses `5.10738`, `5.52237`, `5.05476`,
  and `4.83240` over global steps 10,120--10,123 and remained finite through
  global step 10,137. The callback, or state it leaves behind, is causal; the
  reduction topology and natural nested-training trajectory are ruled out.
- Router z-loss was `10,668.9` on the first post-callback row versus `3,595.1`
  in the counterfactual. The counterfactual later reached `10,937.4` without
  failing, so router z-loss alone does not explain the failure.
- Stopped both bounded diagnostic roots after collecting the result and
  updated issue #7712. The production continuation remains queued from the
  clean checkpoint with periodic evaluation disabled.

### 2026-07-28 23:52 - Exact augmented control reproduces through update 1,000

- Hypothesis: the poor NEST-BURN control curve comes from its data/model
  contract, not the requested d768 compute-optimal cell.
- Bundle:
  `adc2aad8a60b45f4a105d4d6e4134cb7fff350caa77d7e56ab23fbe66bd3479b`.
- Reference:
  <https://wandb.ai/marin-community/marin_moe/runs/aug-dk-d768-ev-sw2k-g4-nomtp-noconv-f1>.
- Reproduction:
  <https://wandb.ai/marin-community/marin_moe/runs/nest-burn-control-augdk-repro1000-r1>.
- Config: d768/L8, six query heads, one KV head, E256 top-4, one shared
  expert, sequence length 8,192, batch 32, 16,840-step MuonH schedule, dense
  Datakit packing, one 8-H100 node. The launch changes only run identity,
  output paths, credentials, priority, and stop time.
- Result: median absolute pointwise train-loss error over 999 matched updates
  is `0.002285` nat; p95 is `0.013817`; LR matches exactly. Update-1,000
  Paloma macro is `4.224867` versus `4.221188`; uncheatable macro is
  `3.788470` versus `3.790526`.
- Interpretation: the reference cell is reproducible. NEST-BURN-001 forced
  `pack=1` and disabled cross-document attention while changing model/router
  source. Phase-0 cache ledgers imply a mean document length of 2,297.7
  tokens, so one-document examples fill at most 28.05% of the nominal
  8,192-token context on average. The old control does not answer the nested
  architecture question.
- Next action: port fixed-prefix E128/E16 routing onto the immutable augmented
  source and require a matched control to pass the same gate.

### 2026-07-29 00:22 - Corrected fixed-prefix pair passes update-1,000 gate

- Hypothesis: restricting 25% of whole sequences to fixed nested expert
  prefixes produces usable E128/E16 models for less than 10% optimizer-step
  overhead without materially degrading full E256.
- Source: immutable augmented reference bundle plus changes limited to
  `experiments/grug/moe/model.py`, `train.py`, and `launch_cw_scale.py`.
  E128 always routes within experts 0--127; E16 always routes within experts
  0--15. The groups use independent QB balance state.
- Control:
  <https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e256-4b-r2>.
- Treatment:
  <https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-fixed25-4b-r2>.
- Data: 168 top-level train components expand to 200 physical caches because
  the tail concat contains 33 caches. The 23 evaluation components bring the
  expanded graph to 223. The earlier 168-versus-200 alarm was a representation
  mismatch, not mixture drift.
- Control gate: across updates 2--1,000, median absolute train-loss error
  versus the reference is `0.002182` nat, p95 is `0.011893`, and LR matches
  exactly. Median loss over updates 900--999 is `3.230410` versus `3.226803`.
  Paloma macro is `4.219130` versus `4.221188`; uncheatable macro is
  `3.788652` versus `3.790526`.
- Cost: over updates 100--999, median compiled step time is `453.553 ms` for
  control and `458.670 ms` for fixed25, a `+1.13%` surcharge. Means are
  `459.107 ms` and `463.865 ms`, a `+1.04%` surcharge. The three-mode
  treatment evaluation takes about 108 seconds per gate versus 44 seconds for
  the control and is excluded from co-training cost.
- Quality at 262M tokens: control/fixed25 full/E128/E16 Paloma macro losses
  are `4.219130`, `4.241795`, `4.274152`, and `4.399118`. Full fixed25 is
  `+0.022665` nat worse than control. Uncheatable macro losses are `3.788652`,
  `3.814934`, `3.856844`, and `4.002663`.
- Decision: the control gate passes and optimizer-step overhead is below the
  preregistered 10% ceiling. Continue both arms to 16,840 updates and
  4.4145B nominal tokens. The early quality penalty remains exploratory until
  the complete curve is available.

### 2026-07-29 00:38 - Second corrected gate and checkpoint replicas

- Both measurement arms crossed the update-2,000 evaluation and resumed
  finite training. Full-mode Paloma macro is `3.961045` for E256 and
  `3.992682` for fixed25, a `+0.031638` treatment delta. Fixed25 E128 and E16
  are `4.029015` and `4.173859`.
- Across the two aligned gates, fixed25 full mode is `+0.022665` and
  `+0.031638` nat worse. Neither gate approaches the preregistered `+0.10`
  regression threshold.
- Through common update 2,689, post-warmup median compiled step time is
  `455.993 ms` for control and `461.389 ms` for fixed25, a `+1.18%`
  surcharge. Optimizer-only cost projects to `38.66` versus `39.11` H100
  GPU-hours per 10B tokens.
- The exact reference disables periodic checkpoints. Launched a second
  matched E256/fixed25 pair with periodic evaluation disabled and checkpoints
  enabled, so final state can feed matched SFT without changing the
  measurement pair. Both replicas compiled and began finite training.
- Published an interim corrected report and marked the first d768 burn
  invalidated. Its pack-one control cannot support the earlier positive
  quality conclusion.

### 2026-07-29 00:58 - Four corrected gates and SFT transfer smoke

- The matched measurement pair remains finite through update 4,000. Full-mode
  fixed25 Paloma deltas at updates 1,000 through 4,000 are `+0.022665`,
  `+0.031638`, `+0.025615`, and `+0.024656` nat. The median delta is
  `+0.025136`; no gate approaches the preregistered `+0.10` rejection
  threshold.
- Through common update 4,194, fixed25 adds `1.21%` to median compiled
  optimizer-step time. Both measurement arms and both checkpoint replicas are
  healthy.
- Added an exact-source Grug SFT path that initializes model weights from a
  pretraining checkpoint while retaining SFT step zero and a fresh optimizer.
  It uses the existing CoreWeave WildChat cache. Unit tests cover full-E256 and
  fixed25 model construction plus weights-only state replacement.
- The E256 one-update smoke loaded checkpoint step 1,190, produced finite loss
  `3.01`, and saved SFT checkpoint step 1. The first fixed25 smoke raced the
  pretraining checkpointer's temporary-checkpoint rotation: step 1,173 was
  discovered by the coordinator but deleted when step 2,394 became current
  before the worker loaded it. This is a checkpoint-lifetime handoff failure,
  not a numerical or model-shape failure. A bounded retry targets the current
  checkpoint; production SFT will consume each replica's durable final
  checkpoint after pretraining terminates.

### 2026-07-29 01:35 - Eight corrected gates and exact-source snapshot

- The matched measurement pair remains finite through the update-8,000
  evaluation. Fixed25 full-mode Paloma deltas at updates 1,000 through 8,000
  are `+0.022665`, `+0.031638`, `+0.025615`, `+0.024656`, `+0.030234`,
  `+0.026350`, `+0.029090`, and `+0.027246` nat. The median is `+0.026798`;
  fixed25 is worse at all eight gates but remains well inside the
  preregistered `+0.10` rejection threshold.
- A log-linear fit projects a `+0.030585` full-mode delta at 10B tokens and
  `+0.031602` at 20B. The slope fit has `R²=0.119`, so it is weak evidence;
  the directly observed stable eight-gate band is more reliable.
- Through common update 8,823, median compiled step time is `456.507 ms` for
  E256 and `461.972 ms` for fixed25, a `+1.197%` surcharge. Optimizer-only
  forecasts are `38.699` versus `39.162` H100 GPU-hours per 10B tokens.
- Both no-evaluation checkpoint replicas are healthy and writing rotating
  temporary checkpoints. The validated weights-only SFT smoke succeeds for
  both model shapes.
- Pushed the exact augmented-reference overlay used for corrected pretraining,
  SFT, and native generation evaluation as commit `b283929fd9` on
  `weaver/652-augdk-exact-source`. The running jobs continue to use their
  already-uploaded immutable source bundle.

### 2026-07-29 02:31 - Corrected 4.414B-token pair completes

- The exact augmented-datakit E256 control and fixed25 treatment both completed
  16,840 optimizer updates without a failure or preemption.
- Terminal full-mode Paloma macro loss is `3.258327` for E256 and `3.289642`
  for fixed25, a `+0.031316` treatment delta. Uncheatable macro is `2.676667`
  and `2.711534`, a `+0.034867` delta. Fixed25 was worse at every one of the
  17 paired evaluation gates.
- Median compiled optimizer-step time is `457.102 ms` for E256 and
  `462.578 ms` for fixed25, a `+1.198%` mechanical surcharge.
- Pre-phase-1 log-linear Paloma fits imply that matching the control's
  4.4145B-token loss requires about `4.8795B` fixed25 tokens. Combining the
  `+10.53%` token multiplier with measured step time gives an `+11.86%`
  time-to-equivalent-loss estimate. The corresponding uncheatable estimate is
  `+11.79%`.
- Interpretation: fixed-prefix co-training is mechanically cheap but the
  naive 75% E256 / 12.5% E128 / 12.5% E16 allocation imposes a small,
  persistent full-model quality tax. It overexposes experts 0--15 at `3x`
  baseline assignment frequency and underexposes experts 128--255 at `0.75x`.

### 2026-07-29 02:48 - NEST-MOE-003 single-prefix 10B preregistration

- Question: how much of the fixed25 quality tax comes from E128 versus E16
  restriction, and can rotating the restriction across layers retain an
  extractable prefix while reducing expert-update imbalance?
- Fixed contract for all five runs: augmented d768/L8, six query heads, one KV
  head, E256 top-4 plus one shared expert, sequence length 8,192, global batch
  32, 38,147 updates (`9.9997B` nominal tokens), dense two-phase Datakit
  mixture, MuonH/AdamH learning rates `0.00838`, beta1 `0.9062`, beta2
  approximately `0.998`, epsilon approximately `1.03e-15`, linear schedule
  with 1% warmup and 0.05 minimum ratio, no clipping, seed 0, one 8-H100 node.
- Arms:
  - `E256-10B`: no training restriction; required matched control.
  - `E128-naive25`: 25% of sequences restrict every MoE layer to experts
    0--127.
  - `E16-naive25`: 25% restrict every MoE layer to experts 0--15.
  - `E128-layer25`: the same 25% sequence schedule, but each restricted
    sequence applies the prefix to a rotating two of eight layers.
  - `E16-layer25`: the analogous rotating two-of-eight-layer schedule.
- The rotating schedule gives every layer equal nested exposure and never
  changes attention, shared-expert, embedding, or LM-head computation. At 10B
  tokens each layer sees approximately 625M restricted token-events. Expected
  per-expert assignment frequencies relative to E256 are `1.0625x/0.9375x`
  for the E128 inner/outer groups and `1.9375x/0.9375x` for E16, versus
  `1.25x/0.75x` and `4.75x/0.75x` in the all-layer naive arms.
- Gate 0: existing all-layer masks remain numerically unchanged; layerwise
  masks restrict exactly two layers for each nested sequence; restriction
  rotates evenly over layers; full rows remain eligible for all experts; all
  five launch configs lower.
- Gate 1 at 1B tokens: all arms finite; observed nested layer-sequence
  fractions within 0.5 percentage points of their targets; median compiled
  step overhead below 5%; full-mode Paloma regression below `+0.10` versus
  E256. Stop an arm that fails numerical or routing correctness. A quality
  miss stops only if it exceeds `+0.10` at two consecutive gates.
- Gate 2 at 4.4145B tokens: compare against the completed fixed25 result and
  estimate full-mode time to equivalent loss. Continue arms whose full-mode
  penalty is below `+0.10` and whose extracted prefix beats the same fixed
  prefix evaluated from the E256 control.
- Gate 3 at 10B tokens: rank by full E256 Paloma and uncheatable loss,
  extracted-prefix loss, per-domain deltas, optimizer-step overhead, and
  time-to-equivalent loss. The primary comparison is each single-prefix naive
  arm versus E256. The primary enhancement comparison is layerwise versus its
  same-prefix naive arm.
- Post-training gate: carry E256 and at most two Pareto-optimal treatments
  through the fixed WildChat-then-Nemotron SFT recipe and heldout/generation
  evaluation. Do not spend SFT compute on a dominated or failed arm.
- Limitations frozen before launch: one seed; layerwise training does not show
  all restricted layers jointly during a training forward, so extracted
  all-layer prefix quality is an empirical question; H100 EP size 1 does not
  validate multi-rack expert-parallel routing.

### 2026-07-29 17:53 - NEST-MOE-003 optimizer amendment before launch

- The 10B cell uses the same `MoeHeuristic` contract as the reproduced
  augmented reference, evaluated at its new 10B token budget. This gives MuonH
  learning rate `0.0060668502`, Adam learning rate `0.0014000424`, beta1
  `0.9062`, beta2 `0.998001`, epsilon `1.8898444e-15`, 1% warmup, linear
  decay to a 0.05 minimum ratio, and no clipping.
- The `0.00838`/`0.00193` rates in the preceding preregistration describe the
  completed 4.4145B-token cell. Stretching those rates to 10B would no longer
  follow the user's requested heuristic. This amendment was recorded before a
  full NEST-MOE-003 job was launched or any arm result was observed.

### 2026-07-29 18:05 - NEST-MOE-003 Gate 0 passes and five arms launch

- Commit `613e570564` adds a rotating layerwise eligibility mask while
  preserving the existing all-layer path. Focused nested-routing and SFT tests
  pass, and the exact-source worktree passes the changed-file lint gate.
- A 20-update E16 layerwise smoke completed on eight H100s. It logged an E16
  nested sequence fraction of `0.25`, a nested layer-sequence fraction of
  `0.0625`, and finite loss. This passes the preregistered mask and compilation
  gate.
- Launched the matched E256 control, E128-naive25, E16-naive25,
  E128-layer25, and E16-layer25 arms on `cw-us-east-02a`. Each arm targets
  38,147 updates and 9.9997B nominal tokens. The control evaluates fixed E128
  and E16 counterfactual prefixes without using either restriction during
  training.
- Initial live telemetry confirms `0.25` sequence and layer-sequence fractions
  for both naive arms and `0.25` sequence / `0.0625` layer-sequence fractions
  for both layerwise arms. All arms are finite. Early compiled steps are
  approximately `0.46--0.49` seconds before the timing warmup gate.
- W&B runs:
  [E256](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e256-10b-r1),
  [E128 naive](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-naive25-10b-r1),
  [E16 naive](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e16-naive25-10b-r1),
  [E128 layerwise](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-layer25-10b-r1),
  and
  [E16 layerwise](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e16-layer25-10b-r1).

### 2026-07-29 18:44 - NEST-MOE-003 Gate 1 passes at 1.049B tokens

- All five arms remain finite and the four treatments retain their exact
  routing targets. Naive arms log `0.25` nested sequence and layer-sequence
  fractions; layerwise arms log `0.25` sequence and `0.0625` layer-sequence
  fractions.
- At update 4,000, control full Paloma is `3.772742`. Full-mode treatment
  deltas are `+0.009955` E128 naive, `+0.032944` E16 naive, `+0.003407` E128
  layerwise, and `+0.000511` E16 layerwise. The corresponding uncheatable
  deltas are `+0.013886`, `+0.038707`, `+0.002849`, and `+0.003660`.
- Against the same control checkpoint's untrained prefix, Paloma gains are
  `-0.208415` E128 naive, `-0.793201` E16 naive, `-0.112081` E128 layerwise,
  and `-0.357386` E16 layerwise. Every treatment improves its intended prefix
  at every one of the first four gates.
- Through common update 4,289, median compiled-step overhead is `+0.50%`,
  `+0.38%`, `+0.44%`, and `+0.53%` in the same arm order. All are below the
  preregistered 5% Gate 1 bound.
- An anchored log-linear local slope model estimates time-to-equivalent full
  Paloma penalties of `+2.87%`, `+8.46%`, `+1.26%`, and `+0.65%`. This is an
  interim four-point estimate, not a final extrapolation.
- Decision: Gate 1 passes for all arms. Continue to the 4.4145B comparison
  gate. The current Pareto arms are E128 naive for stronger E128 extraction
  and both layerwise arms for minimal full-mode loss.
- Commit `9e9a44a4fb` extends the validated weights-only SFT launcher to all
  four single-prefix model-state shapes. The post-training selection remains
  capped at two treatments plus control.

### 2026-07-29 20:58 - NEST-MOE-003 Gate 2 passes at 4.456B tokens

- All four treatments remain finite, retain their exact routing fractions,
  and improve the intended extracted prefix at every one of 17 aligned
  evaluations.
- At update 17,000, full Paloma deltas versus E256 are `+0.015783` E128
  naive, `+0.037078` E16 naive, `+0.006652` E128 layerwise, and `+0.009429`
  E16 layerwise. Full uncheatable deltas are `+0.017628`, `+0.040146`,
  `+0.004921`, and `+0.005452`.
- Prefix Paloma gains versus the same prefix cut from control are `-0.256294`,
  `-0.893008`, `-0.151111`, and `-0.385389`. Every treatment has a prefix win
  at all 17 gates.
- Tail-slope time-to-equivalent estimates at the exact update-17,000 horizon
  are `+8.00%`, `+18.59%`, `+3.49%`, and `+4.81%`. E16 naive is outside the
  10% economic viability line, while E128 naive and both layerwise arms
  remain inside it.
- Through common update 17,205, median compiled-step overhead is `+0.49%`,
  `+0.35%`, `+0.47%`, and `+0.43%`. Several multi-minute S3 checkpoint
  commits changed instrumented wall progress but not optimizer throughput;
  no arm stalled or restarted.
- Decision: all arms satisfy the preregistered Gate 2 continuation rule
  because full Paloma remains within `+0.10` and every intended prefix beats
  control. Continue all five jobs to the 10B endpoint.

### 2026-07-30 00:52 - NEST-MOE-003 10B draft snapshot

- All five arms finished 38,147 updates without a restart. Terminal full
  Paloma is `3.143487` for E256 control. Deltas are `+0.020221` E128 naive,
  `+0.036839` E16 naive, `+0.007248` E128 layerwise, and `+0.011082` E16
  layerwise. Full uncheatable deltas are `+0.018201`, `+0.037782`,
  `+0.004206`, and `+0.009424`.
- The trained-prefix Paloma gains against the same fixed prefix cut from
  control are `-0.340158`, `-1.195134`, `-0.223621`, and `-0.616815`.
  Every intended prefix wins all 39 aligned evaluations.
- Mean throughput over the final 100 updates is 556,653 tok/s for control and
  553,997, 555,481, 555,469, and 555,591 tok/s for the four treatments.
  Compiled-step overhead is only `0.41--0.49%`.
- The primary cost conversion now follows `experiments/grug/moe/agent.md`:
  recenter `loss(C) = 1.6 + A*C^-0.0941` at each endpoint, invert at the
  control loss, and apply the terminal throughput ratio. Equivalent extra
  update counts are 5,659 E128 naive, 10,867 E16 naive, 1,947 E128
  layerwise, and 3,013 E16 layerwise. Equivalent wall costs are `+15.38%`,
  `+28.76%`, `+5.33%`, and `+8.11%`.
- The earlier empirical tail-slope estimates (`+4.88%`, `+8.03%`, `+2.03%`,
  and `+2.81%`) remain a sensitivity analysis. The fixed-exponent Grug
  conversion is the primary scale decision because it does not let each
  short run choose a different terminal slope.
- E128 naive and E128 layerwise are non-dominated and completed matched
  WildChat/Nemotron SFT or are in its final stage. Corrected GPU-native
  full/prefix evaluations are running.
- A matched standalone E128 10B run launched as
  [nest-augdk-e128-standalone-10b-r1](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-standalone-10b-r1).
  The ordinary E256 checkpoint's fixed first-half chop is already measured at
  `3.555385`; calibration-ranked and greedy post-hoc E128 pruning will provide
  a stronger uncompromised-E256 extraction control.
- Draft report:
  `docs/reports/nested-model-training-single-prefix-10b.md`. The final version
  waits for standalone E128, post-hoc pruning, SFT, and generation results.

### 2026-07-30 01:42 - NEST-MOE-004 paired residual experts launched

- Hypothesis: storing experts 0--127 as extractable bases and experts
  128--255 as residuals over the paired base lets every E256 token train the
  E128 checkpoint without masking rows away from the outer bank.
- Commit hash: exact-source commit `f2104e7f60`.
- Command: `scratch/augdk-reference-nested-wt/scratch/resubmit_paired_residual_10b.sh`.
- Config: the matched d768/L8/E256 top-4, batch-32, 10B-token augmented
  Datakit cell. Full route `128+i` materializes each expert weight as
  `(base[i] + residual[i]) / sqrt(2)`; route `i` and E128 extraction use
  `base[i]`. The scaling preserves initialization variance, stored parameter
  count, top-4 active expert count, optimizer, data stream, and LR schedule.
- Result: focused residual and nested-routing tests pass. Job
  `/power/nest-augdk-e128-pairedresidual-10b-r1-coord` submitted to one
  eight-H100 node. W&B run:
  [nest-augdk-e128-pairedresidual-10b-r1](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-pairedresidual-10b-r1).
- Interpretation: this is a Gate 1 architecture probe, not yet a production
  implementation. It does not test expert-parallel sharding or folded E256
  checkpoint export.
- Next action: at update 4,000, compare full E256 and extracted E128 Paloma,
  compiled-step overhead, overflow, and uncheatable loss against the existing
  E256 control. Continue only if full Paloma remains within `+0.10`, E128 beats
  the control chop, and step overhead remains below 5%.

### 2026-07-30 01:51 - NEST-MOE-005 paired router residuals launched

- Hypothesis: pairing only expert weights sends outer-route examples into the
  E128 base but does not make the compact router send related examples to that
  base. Applying the same base-plus-residual parameterization to router columns
  should align the route hierarchy and improve extraction.
- Commit hash: exact-source commit `4a32e6ee22`.
- Command:
  `scratch/augdk-reference-nested-wt/scratch/resubmit_paired_residual_10b.sh nest-augdk-e128-pairedrouterresidual-10b-r1 1`.
- Config: identical to NEST-MOE-004, with outer router column `128+i`
  materialized as `(base_router[i] + router_residual[i]) / sqrt(2)`.
- Result: focused tests pass and job
  `/power/nest-augdk-e128-pairedrouterresidual-10b-r1-coord` is running.
  W&B run:
  [nest-augdk-e128-pairedrouterresidual-10b-r1](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e128-pairedrouterresidual-10b-r1).
- Interpretation: the two residual arms isolate weight sharing from
  hierarchical route sharing without adding parameters or active expert
  matmuls.
- Next action: apply the same update-4,000 gate to both arms.

### 2026-07-30 02:33 - NEST-MOE-004/005 stop at Gate 1

- Both residual arms reached update 4,000 and 1.049B nominal tokens without a
  restart. Weight-only residuals finish at `3.785311` full E256 Paloma and
  `4.178945` extracted E128 Paloma. Pairing the router finishes at `3.788318`
  full and `4.116328` extracted.
- Against the matched control, the full-model deltas are `+0.012569` for
  weight-only residuals and `+0.015575` for weight-plus-router residuals. Both
  pass the `+0.10` full-model bound.
- The control's fixed E128 chop is `4.023812` at the same update. Extracted
  E128 deltas are therefore `+0.155132` for weight-only residuals and
  `+0.092516` for weight-plus-router residuals. Both fail the preregistered
  requirement to beat the untrained control chop. Pairing router columns
  recovers `0.062616` nat relative to pairing expert weights alone, but it
  does not make the extracted model competitive.
- Median compiled-step durations over updates 500--4,000 are `463.392` ms
  control, `462.800` ms weight-only residuals, and `462.928` ms paired-router
  residuals. The residual materialization has no measurable optimizer-step
  surcharge in this one-node full-FSDP test.
- The jobs were intentionally stopped after the gate. Iris reports the
  user-requested termination as one preemption on each coordinator; neither
  run suffered an infrastructure preemption.
- Decision: do not spend 10B tokens on residual sharing without compact-mode
  supervision. A follow-up should combine paired expert/router residuals with
  a sparse explicit E128 objective, or test a balanced-complement schedule
  that restores each bank's expected control assignment rate.
