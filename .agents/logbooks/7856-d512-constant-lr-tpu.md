---
topic: 7856-d512-constant-lr-tpu
issue: https://github.com/marin-community/marin/issues/7856
description: Re-run the issue #7856 d512 LR-by-token-budget matrix on TPU with constant post-warmup learning rates.
author: kaiyuew
---

# Issue #7856 d512 Constant-LR TPU: Research Logbook

## Scope

- Goal: measure how replacing the issue #7856 linear decay with a constant
  post-warmup LR changes d512 loss scaling over 30x, 60x, 150x, 300x, and
  600x active-parameter token budgets.
- Primary metrics: final `eval/paloma/macro_loss`; per-budget optimum peak LR
  from a log-quadratic fit; fitted loss-versus-token-budget exponent.
- Constraints: preserve the historical model, datakit mixture, batch 64, seed
  0, 1% warmup, five peak-LR multipliers, steps, and evaluation cadence. Run
  in us-central2 on TPU so the datakit store is read in-region.
- Coordinating issue: [#7856](https://github.com/marin-community/marin/issues/7856)
- Branch: `codex/research-kaiyuew-7856-d512-constant-lr`
- Experiment prefix: `AUG-LRC-TPU`
- Shared W&B tags: `AUG-LRC-TPU`, `issue-7856`, `d512`, `constant-lr`

## Current TL;DR

- Twenty terminal constant-LR cells cover 30x through 300x. Every budget chose
  the original `0.7x` lower boundary and all four loss curves are monotone over
  `0.7x` through `1.4x`, so the sweep did not bracket the optimum.
- A budget-fixed-effect log-quadratic fit predicts a shared optimum at `0.323x`;
  leave-one-budget-out fits span `0.304x` to `0.363x`. The follow-up grid is
  `0.10x / 0.20x / 0.32x / 0.45x / 0.70x`.
- The five nonterminal 600x cells were intentionally stopped because this
  reproduction writes no checkpoints. A unique 21-cell follow-up reuses the
  completed `0.7x` points at 30x through 300x and reruns all five 600x points.

## Baseline

- Date: 2026-08-25
- Code ref: `marin/aug_hero_run_ablations` at `53488bff8`; the completed W&B
  runs are named `aug-hero-d512-{budget}x-lr{multiplier}-v2` in
  `marin-community/marin_moe`.
- Historical best Paloma macro loss by budget: 30x 3.844, 60x 3.666, 150x
  3.502, 300x 3.409, 600x 3.336.
- Historical schedule: `linear`, warmup `0.01`, `min_lr_ratio=0.05`.

## Hypothesis Queue

### Active

- `AUG-LRC-TPU-H1`: without decay, the high-LR noise floor will make terminal
  Paloma loss improve more slowly at long token budgets than the historical
  linear-decay curve. Minimum test: all five 1.0x cells; full test: the 25-cell
  matrix. Falsifier: the constant-LR loss-versus-token slope is as steep or
  steeper after fitting each budget at its LR optimum.
- `AUG-LRC-TPU-H2`: the constant schedule will move the fitted peak-LR optimum
  below the historical sweep's optimum, especially at 300x and 600x. Minimum
  test: the five LR multipliers at 300x and 600x. Falsifier: fitted optima match
  or exceed the historical multipliers within fit uncertainty. Current
  evidence: supported but boundary-censored at 30x through 300x; next test is
  the bracketed low-LR sweep.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Background Research Brief

- Effort: medium
- Stop rule: stop when issue, W&B, source-branch, and schedule evidence agree
  on the exact control matrix and additional sources do not change the launch.
- Date: 2026-08-25

### Question

What is the narrowest matched TPU experiment that determines how constant LR
changes d512 token-budget scaling relative to issue #7856?

### Current Marin Context

- Issue #7856 specified six widths, five budgets, and five peak-LR
  multipliers. This extension selects all 25 d512 cells.
- W&B is the ground truth for the completed cells because the issue body and a
  later issue comment disagree on d512 batch and step counts. The completed
  `-v2` configs use batch 64 and steps 1,058 / 2,115 / 5,288 / 10,575 / 21,150.
- The completed configs use the datakit `mixture-3.csv` two-stage mixture,
  sequence length 8192, seed 0, and Paloma evaluation every 1,000 steps.
- The existing datakit launcher pins TPU training to us-central2 to avoid
  cross-region reads. This extension uses a v4-8 in `us-central2-b` for the
  same reason.

### Internal Prior Work

- The issue's completed d512 matrix establishes a strong linear-decay control
  across all five budgets and LR multipliers.
- W&B configs show that the intended algorithmic delta is precisely
  `lr_schedule: linear -> constant`; warmup remains 1%, and the historical
  peak MuonH/AdamH rates are preserved per cell.
- Historical d512 runs reported zero dropped assignments, so the TPU local/EP
  backend must also be checked for zero routing loss before treating the loss
  comparison as matched.

### External Prior Art

- *Understanding Warmup-Stable-Decay Learning Rates* predicts that a stable
  high-LR phase can keep observed loss elevated through parameter oscillation
  while still making underlying progress, and that decay reveals that progress
  by reducing the oscillation. This supports measuring the no-decay terminal
  loss directly rather than assuming the historical scaling exponent transfers.
  Source: https://arxiv.org/abs/2410.05192.
- *Scaling Law with Learning Rate Annealing* models loss with both cumulative
  LR and a separate annealing contribution. It predicts a schedule-dependent
  terminal-loss offset and motivates fitting the constant-LR curve independently.
  Source: https://arxiv.org/abs/2408.11029.
- *A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules*
  reports a shared power-law component plus an additional loss-reduction term
  from LR decay across constant, cosine, and step schedules. This is direct
  evidence that the desired comparison needs multiple token horizons and cannot
  be inferred from one constant-LR cell. Source: https://arxiv.org/abs/2503.12811.

### Evidence Map

#### Claim: constant LR may preserve optimization progress while worsening observed terminal loss

- Support:
  - WSD river-valley paper: stable high LR drives motion along the valley but
    sustains transverse oscillation; decay suppresses the visible loss penalty.
  - Annealing scaling-law papers: decay contributes additional terminal loss
    reduction beyond accumulated LR.
- Contradictions:
  - WSD shows the stable branch can continue making useful progress for long
    horizons, so a shallower observed-loss curve need not mean optimization has
    stopped; a later cooldown could recover latent gains.
- Directness to Marin: moderate. The cited work uses dense Adam-like language
  models, while this study uses d512 MoE with MuonH and evaluates an uncooled
  endpoint.
- Confidence: exploratory until the 25 matched Marin cells complete.
- Action: report both terminal constant-LR scaling and the limitation that this
  does not measure a cooldown branched from the same checkpoints.

### Negative / Failed Leads

- The issue summary's d512 batch 32 and 2,115 / 4,230 / ... step table was not
  the matrix that produced the reported results. Launching it would double the
  actual token budgets relative to the completed W&B controls.
- The issue branch's checked-in `launch.py` is a 25-step d6144 GB200 throughput
  run, not the 150-cell LR launcher. The completed W&B configs plus the shared
  datakit builder are required to reconstruct the cells.
- Running v5p in a different GCS region would violate the repository's
  cost-sensitive data-locality guidance; TPU generation is not the independent
  variable in this study.

### Recommended Next Experiments

#### 1. Representative TPU cell

- Minimum experiment: `AUG-LRC-TPU-003-d512-30x-lr1`.
- Baseline/control: `aug-hero-d512-30x-lr1-v2`.
- Expected signal: successful TPU compile, finite advancing loss, zero routing
  drops, constant post-warmup W&B LR, and a final Paloma evaluation.
- Falsifier: incompatible TPU kernel, nonzero routing drops, or a materially
  different config beyond the documented hardware/backend substitutions.
- Cost/risk: one v4-8 through 1,058 steps; compilation is the main early risk.

#### 2. Full d512 matrix

- Minimum experiment: all 25 cells with five-way parent concurrency after the
  representative cell is healthy.
- Baseline/control: the 25 `aug-hero-d512-*-v2` W&B runs.
- Expected signal: enough terminal losses to fit the per-budget LR optima and
  the loss-versus-token curve.
- Falsifier: missing terminal evaluations or an accelerator/backend confound
  visible in routing-drop or optimizer config telemetry.
- Cost/risk: up to five concurrent v4-8 tasks; StepRunner reuses the completed
  representative artifact and prevents duplicate materialization.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Issue #7856 | GitHub issue | https://github.com/marin-community/marin/issues/7856 | Matrix intent and historical result summary | High | Body step table is stale relative to W&B. |
| d512 `-v2` runs | W&B | `marin-community/marin_moe` | Exact model, data, steps, LR, schedule, metrics, and hardware | High | Direct completed-run configs. |
| Aug hero branch | Marin code | `53488bff8` | Historical model/optimizer implementation | High | Fixed source snapshot. |
| Datakit MoE launcher | Marin code | `experiments/grug/moe/launch_datakit_moe_mix.py` | Exact mixture and data-local TPU placement | High | Reused rather than copied. |
| WSD river-valley paper | paper | https://arxiv.org/abs/2410.05192 | Stable-LR loss elevation and decay interpretation | Medium | Dense models; no MuonH/MoE result. |
| LR annealing scaling law | paper | https://arxiv.org/abs/2408.11029 | Separate cumulative-LR and annealing effects | Medium | Motivates an independent schedule fit. |
| Multi-power LR-schedule law | paper | https://arxiv.org/abs/2503.12811 | Constant/cosine/step schedule-specific loss terms | Medium | Direct schedule comparison, different regime. |

## Decision Log

- 2026-08-25: use W&B materialized configs over stale issue-body step counts.
- 2026-08-25: define constant LR as the same 1% warmup followed by the fixed
  historical peak LR; do not remove warmup.
- 2026-08-25: run on v4-8 in us-central2-b to keep the datakit store in-region.
- 2026-08-25: submit one full 30x/1.0x representative cell before lowering the
  remaining 24 cells, then run the matrix with max concurrency 5.
- 2026-08-26: use the explicit `gs://marin-us-central2` datakit prefix because
  TensorStore cannot consume Marin's fsspec-only `mirror://` scheme. The v4-8
  remains pinned to `us-central2-b`, so reads stay in-region.
- 2026-08-27: stop the five nonterminal 600x cells after all four completed
  budgets selected the `0.7x` boundary. Replace the unbracketed range with
  `0.10x / 0.20x / 0.32x / 0.45x / 0.70x` and submit only the 21 missing cells.

## Negative Results Index

- None yet.

## Entry Log

### 2026-08-25 23:59 PDT - Reconstructed matrix and started TPU launcher

- Hypothesis: a config-only linear-to-constant schedule change can be isolated
  while all 25 d512 cells remain otherwise matched to the completed sweep.
- Commit Hash: `c14bd6b09`.
- Command: read issue #7856; queried W&B configs matching
  `^aug-hero-d512-.*-v2$`; inspected branch `53488bff8`.
- Config: d512, batch 64, sequence length 8192, 128 routed experts top-4 plus
  two shared experts, five budgets, five LR multipliers, 1% warmup, TPU v4-8.
- Result: exact materialized controls recovered; isolated research worktree and
  launcher implementation in progress.
- Interpretation: the issue body is insufficient by itself, but W&B and Marin
  code provide a reproducible 25-cell comparison.
- Next action: finish focused tests and materialization, snapshot the branch,
  check duplicates, then submit `AUG-LRC-TPU-003`.

### 2026-08-26 00:06 PDT - TPU matrix validates locally

- Hypothesis: every historical d512 cell can materialize with the same peak LR
  and a flat post-warmup schedule before any accelerator is allocated.
- Commit Hash: `c14bd6b09`.
- Commands:
  - `uv run pytest -q tests/test_d512_constant_lr_tpu.py`
  - `uv run pytest -q tests/test_grug_variant_contracts.py -k 'moe_hero_fsdp_constant_lr_tpu'`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run python` materialization probe for `AUG-LRC-TPU-003`.
- Config: stable artifact version `2026.08.26`; v4-8 in
  `us-central2-b`; max matrix concurrency 5.
- Result: eight focused tests and two copied-variant contracts pass; lint,
  formatting, Pyrefly, AST, and repository hygiene checks pass. The
  representative cell materializes with 1,058 steps, `lr_schedule=constant`,
  and a `v4-8` child resource in `us-central2-b`.
- Interpretation: the launcher is ready for duplicate/auth checks and the
  representative submission. The TPU kernel compile remains the first live
  risk.
- Next action: push the research snapshot, query Iris and W&B for
  `AUG-LRC-TPU-003`, and submit only that cell if no duplicate exists.

### 2026-08-26 01:03 PDT - Submitted representative TPU cell

- Hypothesis: the 30x/1.0x cell is a sufficient live gate for TPU compilation,
  constant-LR telemetry, finite loss, and zero routing drops before launching
  the remaining matrix.
- Commit Hash: `4610d50ea`.
- Command: `/Users/kaiyuew/Downloads/Project/marin-iris-client-current/.venv/bin/iris --controller-url http://127.0.0.1:19000 job run --no-wait --job-name issue-7856-d512-constant-lr-smoke --user kaiyuew --cpu 1 --memory 2G --priority interactive --extra cpu -e WANDB_API_KEY ${WANDB_API_KEY} -- python -m experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch --token-multiple 30 --lr-multiplier 1 --max-concurrent 1`.
- Config: CPU-only StepRunner parent
  `/kaiyuew/issue-7856-d512-constant-lr-smoke`; expected child
  `AUG-LRC-TPU-003-d512-30x-lr1` on a v4-8 in `us-central2-b`; W&B
  project `marin-community/marin_moe` and group
  `issue-7856-d512-constant-lr-tpu`.
- Result: parent submitted after Iris and W&B duplicate checks found no prior
  constant-LR jobs or runs.
- Interpretation: submission identity is clean; parent/child startup and TPU
  compile still require live verification.
- Next action: wait two minutes, verify the child and W&B config/progress, then
  submit the remaining 24 cells if the representative gate is healthy.

### 2026-08-26 01:10 PDT - Fixed remote datakit cache resolution

- Hypothesis: the representative failure is caused by a relative cache prefix,
  not by TPU execution or the constant-LR optimizer, and should be fixed by
  resolving the same us-central2 objects through Marin's `mirror://` filesystem.
- Commit Hash: `8515b04ad`.
- Commands:
  - inspected the child traceback and verified
    `gs://marin-us-central2/datakit/store_8ac06c74/cluster=1/quality=1/shard_ledger.json`;
  - cancelled `/kaiyuew/issue-7856-d512-constant-lr-smoke` after the same
    missing-cache error entered its automatic retry;
  - `uv run pytest -q tests/test_d512_constant_lr_tpu.py`;
  - `uv run pytest -q tests/test_grug_variant_contracts.py -k 'moe_hero_fsdp_constant_lr_tpu'`;
  - `./infra/pre-commit.py --changed-files --fix`.
- Config: the historical launcher retains its explicit relative store prefix;
  the TPU extension now passes
  `mirror://datakit/store_8ac06c74`, which resolves to the in-region Marin
  bucket on the us-central2 worker.
- Result: the first child reached a v4-8 and created the intended W&B run with
  `lr_schedule=constant`, then failed before model initialization with
  `ValueError: No source and no cache found for component c01q1 split train`.
  The path fix materializes `c01q1` as a `mirror://` cache; all focused tests
  and changed-file checks pass.
- Interpretation: this is a bounded launcher-path bug. It provides no evidence
  against either research hypothesis and should not be counted as an
  experimental result.
- Next action: snapshot and push the fix, resubmit the same representative run
  identity, and require an advancing finite loss before launching the other 24
  cells.

### 2026-08-26 01:20 PDT - Replaced mirror URI at the TensorStore boundary

- Hypothesis: the second preflight failure comes from passing an fsspec-only
  `mirror://` path into TensorStore; the native us-central2 GCS URI should serve
  the same objects without cross-region I/O.
- Commit Hash: `9e6a55dfd`.
- Commands:
  - inspected attempt
    `/kaiyuew/issue-7856-d512-constant-lr-smoke/grug-train-AUG-LRC-TPU-003-d512-30x-lr1/0:0`;
  - checked Iris capacity: 28 READY v4-8 slices in `us-central2-b`;
  - cancelled the exact parent before an automatic retry could repeat the
    known error.
- Config: replace `mirror://datakit/store_8ac06c74` with
  `gs://marin-us-central2/datakit/store_8ac06c74`; keep the v4-8 in
  `us-central2-b`.
- Result: the attempt loaded all 200 ledgers, then failed before model
  initialization with `Unsupported URI scheme for tensorstore: 'mirror'`.
  W&B contains no loss or step, and no checkpoint was written. Nine focused
  tests, two copied-variant contracts, and changed-file pre-commit checks pass;
  the native path builds a GCS TensorStore spec, and the failed shard's
  `zarr.json` exists in `marin-us-central2`.
- Interpretation: the two attempts are training-preflight failures, not
  constant-LR observations. Native GCS is required below the cache-ledger
  layer.
- Next action: validate, snapshot, and resubmit the representative identity.

### 2026-08-26 01:42 PDT - Passed the live gate and launched the full matrix

- Hypothesis: one healthy 30x/1.0x run is enough to release the other 24 cells
  without duplicating the representative artifact.
- Commit Hash: `4593a8f95`.
- Commands:
  - monitored W&B `AUG-LRC-TPU-003-d512-30x-lr1` through step 86;
  - submitted parent `/kaiyuew/issue-7856-d512-constant-lr-matrix` with default
    max concurrency 5;
  - created the 15-minute heartbeat `babysit-7856-d512-constant-lr`.
- Config: all 25 d512 cells on v4-8 in `us-central2-b`; batch 64; 1% warmup;
  constant post-warmup LR; W&B group `issue-7856-d512-constant-lr-tpu`.
- Result: representative loss fell from 11.782 at step 2 to 8.794 at step 20
  and 5.685 at step 86. LR reached 0.028575256 at step 10 and remained there;
  dropped assignments and routing overflow stayed zero. The matrix parent
  waited on the active representative lock, started `001`, `002`, `004`, and
  `005`, and created no duplicate `003` child. The first two new W&B runs also
  report `lr_schedule=constant`, warmup 0.01, batch 64, and 1,058 steps.
- Interpretation: the TPU/data/optimizer path is healthy enough to run the
  matrix. Startup has not introduced a schedule or routing confound.
- Next action: babysit all 25 cells, verify checkpoints and terminal Paloma
  metrics, then fit the constant-LR optimum and loss scaling against #7856.

### 2026-08-27 10:40 PDT - Stopped the boundary-censored sweep and bracketed the follow-up

- Hypothesis: the completed constant-LR curves contain enough curvature to
  choose a lower grid that brackets the optimum without repeating completed
  `0.7x` artifacts.
- Commit Hash: `53ec1fb6ec`.
- Commands:
  - queried the 20 finished `AUG-LRC-TPU` W&B runs and fit
    `budget fixed effects + log(multiplier) + log(multiplier)^2`;
  - `/Users/kaiyuew/Downloads/Project/marin-iris-client-current/.venv/bin/iris --controller-url http://127.0.0.1:19001 job cancel /kaiyuew/issue-7856-d512-constant-lr-matrix`;
  - `uv run pytest -q tests/test_d512_constant_lr_tpu.py tests/test_d512_constant_lr_lower_sweep.py`;
  - `./infra/pre-commit.py --changed-files --fix`.
- Config: shared fitted optimum `0.323x`; individual optima
  `0.201x / 0.358x / 0.377x / 0.352x`; leave-one-budget-out range
  `0.304x–0.363x`; new grid `0.10x / 0.20x / 0.32x / 0.45x / 0.70x`.
- Result: the original parent and five 600x descendants are `killed`; their
  latest W&B partial metrics remain available at steps 16,296 / 16,196 /
  10,377 / 10,514 / 3,319. No checkpoint can be preserved because this
  throughput reproduction explicitly sets `checkpointer = None`. The new
  launcher contains 21 unique cells: four new multipliers for each completed
  budget and all five multipliers for 600x. Twelve focused tests and all
  changed-file checks pass.
- Interpretation: `0.7x` is an upper bound, not a measured interior optimum.
  The shared fit is stable to dropping one budget, while the lower `0.10x`
  endpoint protects against the 30x-only extrapolation near `0.20x`.
- Next action: snapshot and push the launcher and fit artifacts, check Iris and
  W&B for duplicate `AUG-LRC-LOW` identities, then submit the CPU StepRunner
  parent with five-way v4-8 concurrency.

### 2026-08-27 10:54 PDT - Launched the lower-LR follow-up

- Hypothesis: the `0.10x / 0.20x / 0.32x / 0.45x / 0.70x` grid brackets the
  constant-LR optimum, with the existing four completed `0.70x` cells reusable
  at 30x through 300x.
- Commit Hash: `fdc6d312f5`.
- Commands:
  - checked Iris, W&B, and GCS for existing `AUG-LRC-LOW` jobs, runs, or
    artifacts;
  - submitted parent `/kaiyuew/issue-7856-d512-constant-lr-low` with five-way
    concurrency;
  - verified the first five child jobs and W&B identities.
- Config: 21 new d512 cells on v4-8 in `us-central2-b`; batch 64; 1% warmup;
  constant post-warmup LR; W&B group
  `issue-7856-d512-constant-lr-low-tpu`; artifact version `2026.08.27`.
- Result: the parent is running with zero failures and zero preemptions. The
  first allocation is exactly `AUG-LRC-LOW-001` through `005`. All five runs
  have finite, advancing train loss at steps 18 through 23 and have reached
  their configured constant LRs. Dropped assignments and routing capacity
  overflow are zero. No OOM, TPU fault, duplicate, or unrequested child is
  present.
- Interpretation: the replacement sweep passed its startup gate without
  introducing a schedule, routing, identity, or hardware confound. The four
  completed old `0.70x` cells remain excluded from the new work.
- Next action: babysit at 15-minute cadence, combine the 21 new terminal Paloma
  results with the four reusable old `0.70x` results, then refit the optimum and
  compare the selected LR curve with the issue #7856 linear-decay baseline.
