---
topic: Gated latent router ablation
issue: https://github.com/marin-community/marin/issues/9110
description: Compare full-width routing with a GatedNorm latent shared by router and experts.
author: Kaiyue Wen
---

# Gated latent router: task logbook

## Scope

Idea from Zihan Qiu. Follow Agent MoE Gate 1 at d512/d768 and proceed to
Gate 2 at d1024/d1280 only if both smaller scales improve effective speedup.
Reuse the existing Agent MoE baseline runs; do not launch new baseline jobs.
Experiment prefix: MOE-LGR. W&B group: moe-latent-gated-router-9110 in marin_moe.

## Current baseline

EP hero main `d891fba48a7d3729fdac2595df3c6ca292a53b74`, scaled to small widths.
The control is `RMSNorm(down(h))` for experts and `router(h)` for routing.
The treatment is `z=GatedNorm(RMSNorm(down(h)))`, used by both router and experts.
The control is retained for numerical checks and analytic budget calculation.
Only the treatment is trained. Historical comparisons have architecture,
optimizer, hardware, and token-count differences.
See the [plan](../projects/moe-latent-gated-router/plan.md) and
[variant README](../../experiments/grug/moe_latent_gated_router/README.md).

## Hypothesis queue

- Active MOE-LGR-001: The shared gated latent improves loss enough to offset
  runtime overhead at d512 and d768 versus the existing Agent MoE reference.
  Evidence is pending; fail if either effective speedup is <=1. This reference
  comparison cannot isolate the requested intervention.
- Conditional MOE-LGR-002: The gain persists at d1024/d1280 and improves
  projected loss at 1e21 and 1e23 FLOPs. Run only after MOE-LGR-001 passes.

## Entry log

### 2026-09-11 — Design and prior work

- Effort: low. The requested intervention is concrete; stop after checking
  current hero code, nearby latent experiments, and norm/optimizer history.
- Source: [hero model](https://github.com/marin-community/marin/blob/d891fba48a7d3729fdac2595df3c6ca292a53b74/experiments/grug/moe_hero_ep/model.py#L914).
  Router reads full width; latent RMSNorm feeds experts.
- [#8105](https://github.com/marin-community/marin/issues/8105): dynamic latent
  retained RMSNorm and found a dropless tie (3.0329 vs 3.0335). It does not
  answer whether the requested shared gated latent helps.
- Echo query: `latent MoE router gated norm ablation`, execution 5444.
  Related norm-placement discussion in #4952 and optimizer-routing work in
  #5750; neither snippet established an existing result for this intervention.
- Scope: combined change, no component-level attribution. GatedNorm has no
  built-in RMS normalization; use it literally as the replacement requested.
- Four initial full cells: d512 control/treatment and d768 control/treatment.
  Nemotron mix and v5p-8 follow Agent MoE; current hero shape and Aug optimizer
  are matched across arms. Harrier data and GB200 transport are not reproduced.
- Cluster read-only preflight: `uv run --no-sync iris --cluster=marin cluster status`.
  Running True, Healthy True, 363/363 workers healthy.
- Issue #9110 created and attached as a sub-issue of #4281. Zihan Qiu credited.
- Next: numerical tests, lint, bounded accelerator smoke, then full Gate 1.

### 2026-09-11 — Local verification

- Numerical suite: `uv run --no-project infra/ci/run_tests.py --base-ref marin/main --workers 2`:
  4 passed. Control output/routing exactly matches the upstream hero; the treatment
  is invariant to down-projection-nullspace perturbations; router gradients reach
  the down projection and latent gate; dense all-expert output matches grouped dispatch.
- Variant contracts: `uv run --no-sync pytest tests/test_grug_variant_contracts.py -q -n 2`:
  19 passed, 1 pre-existing skip.
- Explicit Pyrefly checks on the new model, trainer, launcher and test file: 0 errors.
- Required `infra/pre-commit.py` lint/format checks on all changed files: passed.
- The new gate key is folded with 4, disjoint from the existing four initialization
  splits. The source model's obsolete shard_map compatibility import was replaced
  with the current JAX import. HFCheckpointConverter support is omitted from this
  training-only variant; the research checkpoint format is the native Grug state.
- Dry-run lowering: only the research checkpoint is mutable; the nine training
  caches are pinned and all report SUCCESS in us-central1.
- Computed full schedules: d512 13,642 steps / 1,788,084,224 tokens;
  d768 19,378 / 5,079,826,432; d1024 16,425 / 8,611,430,400;
  d1280 14,473 / 15,176,040,448. Both arms use these same schedules.
- A scratch dense-reference test initially had ambiguous explicit sharding.
  The reference now uses NumPy dense expert evaluation; no tolerance was relaxed.

### 2026-09-11 — Placement preflight

- All 32 pinned dependencies report SUCCESS in us-central1: nine training,
  sixteen Paloma and seven Uncheatable caches. This check read status metadata only.
- The first smoke submission was rejected before creating a job: the CPU parent
  combined a us-central1 region constraint, inferred non-preemptible placement,
  and `--reserve v5p-8` availability. The controller reported no matching groups.
- Pin the TPU child explicitly to us-central1 and retry the CPU parent without
  the accelerator availability constraint. No production job was changed.

### 2026-09-11 — Smoke submission and scope correction

- Source checkpoint: `da78e56d0`. The smoke parent
  `/kaiyuew/moe-lgr-9110-d512-gated-smoke` is running and dispatched only
  `/kaiyuew/moe-lgr-9110-d512-gated-smoke/grug-train-moe-lgr-9110-d512-gated-smoke`.
  The child is capacity-pending, with no training progress verified yet.
- User clarification: “我们不需要跑baseline吧”. Do not submit any new baseline
  training. Reuse existing Agent MoE references; only two treatment cells per gate.
  The fresh-control plan above is superseded. No baseline job was submitted.
- The existing May Recipe references differ in architecture, optimizer, token count
  and v4-32 hardware. Report compute-equivalent gain and measured wall-clock
  comparison separately; this reference cannot isolate the requested intervention.
- Monitoring ownership transfers to the current task's heartbeat
  `follow-gated-latent-moe-ablation-9110` (10-minute cadence). The state file is
  `scratch/20260911-1028_moe-lgr-9110_monitoring_state.json`; the durable
  [handoff](../projects/moe-latent-gated-router/handoff.md) records commands, gates,
  checkpoint locations, and recovery limits. Stop the heartbeat after the gate
  decision or an unrecoverable failure. Normal capacity waits should stay quiet.
- Verified the existing W&B references are both `finished`: d512 Paloma macro
  3.54216671 at 1,439,170,560 tokens; d768 3.22727251 at 4,423,680,000 tokens.
  Full config/summary snapshots are in `scratch/9110-existing-baselines.json`.
  Retrieve their final 100 throughput samples when calculating the final comparison.
- 17:36 UTC update: smoke child allocated, JAX initialized, expected W&B run
  started, and checkpoint loader confirmed a fresh start. Training loss and final
  smoke completion are still pending.

### 2026-09-11 — Automatic preemption recovery

- At 17:48 UTC W&B showed `crashed` while Iris still showed the child running.
  `iris task describe` resolved the apparent conflict: attempt 0 was scheduler-
  preempted, and attempt 1 was already running on another v5p-8 worker.
- Attempt-specific logs show the same run ID reinitialized, read the existing
  caches and entered training at 0/5 at 17:49:13 UTC. A thread profile confirmed
  a live Python process during cache loading. No model exception was observed.
- No job was manually stopped/resubmitted and no code or cluster settings changed.
  Manual recovery count remains zero. Full Gate 1 is still conditional on smoke success.
- CLI correction: this checkout has `task describe` and `attempt logs`, but no
  `job summary`; use the current task/attempt inspection surface on future ticks.

### 2026-09-11 — Smoke passed; normalization clarification

- Gate-only smoke succeeded in Iris and W&B: five completed steps (global_step 4),
  655,360 tokens, train loss 11.80024719, Paloma macro 11.78380013. These startup
  values are not ablation-quality results. Permanent step-5 checkpoint metadata
  was read successfully (timestamp 17:51:34.166678 UTC, is_temporary=false).
- Full d512 parent `/kaiyuew/moe-lgr-9110-d512-gated` was submitted at 17:56:53 UTC
  from `e9cf08610`. Before d768 submission the user asked whether GatedNorm itself
  includes normalization. It does not: the hero composes the gate with RMSNorm.
- Clarification pending: retain latent RMSNorm and add the gate, or keep the current
  gate-only treatment. The full d512 parent was cancelled and the heartbeat paused;
  d768 and all baseline jobs were never submitted. Do not auto-resume until resolved.

### 2026-09-11 — Confirmed RMSNorm then gate

- User confirmed: “down(h) → RMSNorm → gate → router + experts 我指的是这个”.
  Retain the latent RMSNorm and apply the existing rank-128 gate after it.
  The earlier gate-only interpretation and its smoke result are superseded.
- Updated the model to normalize before gating for both router and experts.
  Added regression coverage using a zero gate matrix and nonuniform learned norm
  scales; this distinguishes missing normalization and normalization after gating.
  Router gradients reach the down projection, norm weights and gate matrix.
- Safe diff-driven test runner: all four numerical tests passed. Existing control
  parity and the dense selected-expert oracle remain valid.
- Use fresh `rmsgated` run IDs and output roots for corrected smoke and Gate 1.
  Never restore the gate-only checkpoint. No new baseline training is authorized.
- Corrected source checkpoint: `9dcc3438b`. Type checks and required lint passed.
- Corrected smoke parent submitted at 18:07:10 UTC; its exact TPU child appeared
  at 18:08:24 UTC and is capacity-pending with no training steps yet.
- The same heartbeat is ACTIVE again with the corrected architecture and run IDs.
  Current state: `scratch/20260911-1105_moe-lgr-9110-rmsgated_monitoring_state.json`.
  It will verify this smoke before submitting the two corrected Gate 1 cells.

### 2026-09-11 — Direct full Gate 1 launch

- User instruction: “别smoke浪费卡了，直接跑”. Cancelled the corrected smoke parent
  and descendants while its TPU child was still pending with no allocated attempt.
  No further smoke runs are authorized or required for this experiment.
- Submitted the two full variants from `edf9b2871`: d512 parent
  `/kaiyuew/moe-lgr-9110-d512-rmsgated` at 18:53:16 UTC (13,642 steps), and d768
  `/kaiyuew/moe-lgr-9110-d768-rmsgated` at 18:53:58 UTC (19,378 steps).
- Both use `down -> RMSNorm -> gate -> router + experts`, v5p-8 in us-central1,
  the existing schedule, and fresh corrected output roots. No baseline submitted.
- The active heartbeat now monitors full Gate 1 directly, without a smoke prerequisite.
  Each full-run identity, expected checkpoint and sanitized recovery command is in
  the existing monitoring state file's `gate1` list.
- At 18:58 UTC, both expected training children were confirmed pending for v5p-8
  capacity, with no allocated attempts; both CPU parents were running. The corrected
  smoke parent and child were both confirmed killed. No duplicate jobs were submitted.

### 2026-09-11 — Full Gate 1 training verified

- At 20:56 UTC, both corrected full cells were running on attempt 0, each on a
  us-central1 v5p-8 worker. No manual recovery was needed.
- W&B source `edf9b2871`, GATED_LATENT mode, full schedules, batches 32/64,
  seed 0, data seed 1, EP1 and bf16 compute were verified. Latent RMSNorm and
  gate parameters are both present. Checkpoint logs confirm fresh starts.
- d512: 800 completed updates, train loss 4.27357, current 343,555 tokens/sec.
  d768: 257 completed updates, train loss 5.55921, current 201,524 tokens/sec.
  Recorded numeric metrics are finite and routing overflow is zero. These
  startup values are not final evaluation or steady-state throughput results.
- Compact startup evidence: `scratch/9110-gate1-startup-wandb.json`. State and
  issue #9110 now record training progress; the same heartbeat remains owner.

### 2026-09-11 — First full-run worker recovery

- At 21:09 UTC, both attempt 0 workers had failed with `worker reconcile failure
  threshold exceeded`. Iris automatically started attempt 1 on replacement workers.
  No manual resubmission, code change or cluster modification was made.
- Last attempt 0 W&B observations: d512 global step 937, loss 4.13557; d768
  global step 300, loss 5.27849. All numeric metrics were finite, overflow zero.
- Both attempt 1 loaders confirmed no checkpoint and a step-0 restart: the failure
  happened before the first 15-minute checkpoint. Corrected run IDs/source persist.
- W&B suppresses replayed lower steps, so its summary may stay at the prior attempt
  until catch-up. Use attempt-specific Iris progress logs in this interval.

### 2026-09-11 — Checkpoints survive scheduler preemption

- At 21:34 UTC, both attempt 1 tasks were scheduler-preempted and pending
  automatic attempt 2. No manual resubmission or cluster change was made.
- Both temporary checkpoints were saved and metadata read successfully:
  d512 step 1921 at 21:24:39 UTC; d768 step 648 at 21:24:53 UTC. Expected
  automatic resumes use these checkpoints under the existing corrected roots.
- Last global steps: d512 2492, train loss 3.73923; d768 659, loss 3.94279.
  Numeric metrics remain finite, overflow zero. Intermediate d512 step-2000
  Paloma macro loss is 4.25001; no final result or gate decision yet.
- Checkpoint evidence: `scratch/9110-gate1-checkpoint-metadata.json`.

### 2026-09-12 — Checkpoint restoration verified

- At 09:17 UTC, Iris allocated d512 attempt 3 and d768 attempt 2 on new
  us-central1 v5p-8 workers after the extended capacity wait.
- Attempt-specific logs confirm successful restoration of corrected d512 step
  1921 and d768 step 648 checkpoints. Source remains `edf9b2871`.
- Both W&B runs are running again; their summaries still show preemption-era
  steps 2492/659. New training advancement and evaluations remain unverified.
- No manual recovery, smoke, baseline, or cluster change was performed.

### 2026-09-12 — Resumed training advances

- At 09:29 UTC, both current attempts had advanced beyond the preemption-era
  steps: d512 global step 3284, d768 step 1070. Attempt logs corroborate progress.
- W&B train losses are 3.95643/3.97246, with no nonfinite summary metrics and
  zero routing overflow. Intermediate Paloma losses are 4.29929/4.32959.
- Final evaluation and gate decisions remain pending; no additional jobs launched.

### 2026-09-12 — d512 final result

At 10:51 UTC on September 12, d512 finished successfully (Iris attempt 3 exit 0; W&B finished). Its permanent step-13642 checkpoint metadata is verified. d768 remains healthy on attempt 2, at approximately 4,600/19,378 updates with finite metrics and zero routing overflow.

| d512 result | Gated latent treatment | Existing May reference |
| --- | ---: | ---: |
| Final Paloma macro loss | 3.84245491 | 3.54216671 |
| Total tokens | 1,788,084,224 | 1,439,170,560 |
| Mean tokens/sec, last 100 training steps | 340,373.87 | 431,263.12 |

The recentered compute ratio is 0.217010 and the token/throughput-adjusted effective speedup is 0.137854. Thus d512 fails the required >1 threshold; no d1024/d1280 cells will launch. The existing d768 run will finish for the complete two-width report. These are comparisons against a different historical recipe on v4-32, versus this treatment on v5p-8; they do not isolate the effect of gating.

Training source remains `edf9b2871`. No new baseline or smoke was run. Idea credit: **Zihan Qiu**. Detailed evidence is recorded in the experiment logbook and monitoring state.

### 2026-09-12 — Final Gate 1 results

Both RMSNorm-before-gate runs in the first, small-model experiment stage (Gate 1) completed successfully on September 12: d512 Iris attempt 3 and d768 attempt 2 exited 0, and both W&B runs are finished. Permanent checkpoint metadata is verified at steps 13,642 and 19,378. Routing overflow remained zero and recorded metrics were finite. Source: `edf9b2871`; architecture: `down(h) -> RMSNorm -> gate -> router + routed experts`. Idea: **Zihan Qiu**.

| Hidden width | Treatment Paloma loss | Reference Paloma loss | Tokens, treatment / reference | Last-100 mean tokens/sec, treatment / reference | Treatment non-embedding training FLOPs | Recentered compute ratio | Effective speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 3.84245491 | 3.54216671 | 1,788,084,224 / 1,439,170,560 | 340,373.87 / 431,263.12 | 3.798955e+17 | 0.217010 | 0.137854 |
| 768 | 3.55822515 | 3.22727251 | 5,079,826,432 / 4,423,680,000 | 200,154.66 / 291,668.98 | 2.798080e+18 | 0.139819 | 0.083556 |

Both widths fail the >1 effective-speedup threshold. Gate 1 fails; no d1024/d1280 runs or Gate 2 scaling fit will be launched. No additional baseline or smoke was run.

Using the [Agent MoE guide](https://github.com/marin-community/marin/blob/codex/moe-latent-gated-router/experiments/grug/moe/agent.md) scaling-law assumption (loss asymptote 1.6, exponent 0.0941), the compute ratio is `((L_reference - 1.6)/(L_treatment - 1.6))**(1/0.0941)`. Effective speedup is a model-based estimate, not a measured acceleration. It divides the estimated reference time to reach the treatment loss (`compute_ratio * reference_tokens / reference_TPS`) by treatment time (`treatment_tokens / treatment_TPS`). TPS is the mean of exactly the last 100 training steps. These times exclude queueing, preemption, compilation, evaluation and checkpoint overhead. Analytic non-embedding training FLOPs are `3 * (forward_FLOPs_per_token - 2*hidden_dim*vocab_size) * treatment_tokens`; they are distinct from the loss-derived compute ratio.

The reused May references use a different architecture, optimizer/data recipe and v4-32 hardware; treatment uses the [scaled hero latent configuration](https://github.com/marin-community/marin/tree/codex/moe-latent-gated-router/experiments/grug/moe_latent_gated_router) on v5p-8. This comparison does not isolate the causal effect of adding the gate or changing router input.

- [d512 treatment](https://wandb.ai/marin-community/marin_moe/runs/moe-lgr-9110-d512-rmsgated) / [reference](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d512_ep1)
- [d768 treatment](https://wandb.ai/marin-community/marin_moe/runs/moe-lgr-9110-d768-rmsgated) / [reference](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d768_ep1)

Final checkpoints: `gs://marin-us-central1/users/kaiyuew/grug/moe-lgr-9110-d512-rmsgated/dev/checkpoints/step-13642` and `gs://marin-us-central1/users/kaiyuew/grug/moe-lgr-9110-d768-rmsgated/dev/checkpoints/step-19378`.

### Comparison correction — September 12

The May reference is not a matched control for this ablation. The previous Gate 1 failure wording must not be interpreted as evidence that RMSNorm-then-gate routing hurts quality. The measured losses remain valid for these runs, but the ablation conclusion is inconclusive.

Actual W&B configuration differences include 256/top-4 versus 384/top-8 experts, a full-width MoE versus the hero latent bottleneck, different shared-expert structures, optimizer beta1 0.9062 versus 0.9, minimum LR ratio 0 versus 0.05, different learning rates and token budgets, and training z-loss weight 0 versus 1e-4. Evaluation batch size is 512 in the references versus 32/64 in treatment, with max_eval_batches=8 in both, so the evaluation sample budgets also differ. Hardware is v4-32 versus v5p-8; the estimated speedups do not isolate architectural efficiency.

Tokenizer identity is the same (`meta-llama/Meta-Llama-3.1-8B`), and named training-source mixture weights agree after removing the treatment's `-llama3` suffix. Cache identity and exact evaluation examples have not been established as equivalent. Different cache paths alone do not prove different source data.

No new runs are being launched. A valid gate ablation needs a reference with the same hero architecture, training recipe, and evaluation protocol, differing only in the intended gate/router intervention. Until such a matched existing reference is found or a new comparison is explicitly agreed, no causal pass/fail conclusion is warranted. Idea credit remains Zihan Qiu.

### Existing baseline search — September 12

Located the hero-shape scaling-ladder references:
- d512: [h100-ladder-d512-ep8-bs1024-791tpp-20260824-rno2a](https://wandb.ai/marin-community/marin_moe/runs/h100-ladder-d512-ep8-bs1024-791tpp-20260824-rno2a), 16,483,614,720 tokens, batch 1024.
- d768: [h100-ladder-d768-2xep8-bs1024-791tpp-20260824](https://wandb.ai/marin-community/marin_moe/runs/h100-ladder-d768-2xep8-bs1024-791tpp-20260824), 47,898,951,680 tokens, batch 1024. A batch-512 sibling also exists.

These share the 384/top-8, hidden/2 latent and expert widths, two shared experts, 6/8 layers, SConv k/attn/mlp, HIST QB with 10k bins, and capacity 1.15 configuration. However, they use the Harrier mixture, top-level Marin tokenizer setting, H100 pooled EP transport, different LR/beta2, and different evaluation budgets. Treatment used the Agent MoE Nemotron/StarCoder/ProofPile recipe with the Llama-3.1 tokenizer setting and batch 32/64 on v5p-8. The ladder's evaluation cache names contain `-llama3`; actual cache-tokenizer equivalence requires further verification, so tokenizer config differences alone should not be interpreted as proven evaluation-token differences.

Completed filtered W&B searches across `marin_moe`, `marin_moe_ragged`, `grug_latent_ablation`, and `qbb_latentmoe_ringEP` for hidden width 512/768, 384 experts, top-8, sequence length 4096, and the treatment's exact top-level tokenizer setting returned only the two treatment runs and the superseded smoke. No matched existing control was found in that scope. Issue #8227 and #8105 baselines use 192/top-4 and are excluded.

The original experiment combined the hero architecture with a different training recipe instead of matching one of these existing hero baselines. To reuse an existing baseline, the treatment must be aligned to its full configuration and schedule. The completed treatment runs cannot currently support a controlled gate comparison. No new training was launched during this search.

## 2026-09-12 — reuse Larry's gated TPU controls

User approved matching #6822's already-finished RMSNorm + GatedNorm controls,
changing only router input to gated latent. Recovered W&B config and code
artifacts v698/v696; model/train files match across the two widths. New source
uses larry_model/larry_train/larry_launch, recorded configs, steps 10980/16875,
batches 32/64, eval batch512/max8, v5p-8 us-east5, project dial_moe. Superseded
unsubmitted H100 launcher drafts moved to scratch. No smoke/new control planned.
CPU probe verified baseline forward parity after API adaptation, unchanged
shared initialization, latent null-space invariance, router logits, and a
forward/backward optimizer update. Runtime dependencies are current; historical
throughput and analytic FLOP estimates retain that limitation.

### Submission confirmed

Source 86e5826e4 pushed. Both parents /kaiyuew/moe-lgr-9110-larry-d{512,768}
and expected grug-train children exist; TPU children pending capacity, zero
failures. State: scratch/20260912-1033_larry-router-monitoring-state.json.
Single monitoring owner: heartbeat follow-larry-gated-router-tpu-ablation,
every ten minutes. Issue #9110 updated. No accelerator smoke was run.

### 17:47 UTC startup verified

Both children training on attempt0. Live W&B model/data/optimizer/eval and
seed/batch/steps/precision match historical snapshots except router_input.
Finite loss and advancing steps verified; no Paloma result yet.

### 17:58 UTC first evaluation

d512 Paloma7.1434 at logged999 versus Larry control4.5498 at1000.
Large early gap; not a final verdict. Both treatments running, finite loss;
d768 has no eval yet. Posted interim evidence to #9110; no recovery needed.

## 18:01 UTC — optimizer class mismatch found and contained

User asked which optimizer GatedNorm uses. Runtime inspection found that the
launcher decoded scalars into library MuonHConfig, not GrugMoeMuonHConfig. The
generic mask routes raw Grug arrays to Adam. Thus prior scalar-config matching
was insufficient, and first-eval loss comparisons cannot isolate routing.
Cancelled both /kaiyuew/moe-lgr-9110-larry-d{512,768} parents and descendants.
Recovered Larry optimizer source: current experiments/grug/moe/optimizer.py
executable behavior matches. Fix launcher class, test masks, and use new
-muonh identities/checkpoint roots from scratch. No smoke or new baseline.
