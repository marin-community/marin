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
- The Marin test nests a fixed, interleaved 128-expert subset inside a
  256-expert bank and assigns whole sequences to subset or full routes.
- A four-step fp32 nested25 canary was finite and showed a `+0.00298`
  full-versus-nested Paloma gap, but capacity factor 1.0 dropped 5.93% of
  assignments and the control did not reach step 0.
- A user-directed discovery rerun compares all four architectures for 20 fp32
  updates at common capacity factor 1.25.

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
