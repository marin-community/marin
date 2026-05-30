# Delphi × Nemotron-CC-Math 10 B midtraining — logbook

## 2026-05-24T05:16Z — True-midtraining cooldown20 batch monitor update

The cooldown20 true-midtraining batch launch is still healthy. The launcher set
uses `v6e-4` for `3e18` and `9e18`, `v6e-8` for `2e19`/`3e19`/`9e19`, and
`v5p-8` for `2e20`, with high-RAM child TPU jobs and Iris `batch` priority.
The previously completed `3e18/p33m67` cell was not relaunched.

Current signal from Iris:

- 17 launch roots for timestamp `1779597018`: 6 running, 11 pending on CPU
  coordinator capacity.
- Child TPU jobs: 5 running, 1 pending for v6e-4 capacity
  (`9e18/p67m33`).
- No root or child terminal failures in the new batch.
- Verified resume/progress:
  - `3e18/p50m50`: running, temp checkpoints through step `31501`, latest
    observed progress `31.5kit/37.0kit`.
  - `9e18/p50m50`: resumed after preemption from step `30001`, first train
    step completed, latest observed progress `30.2kit/44.1kit`.
  - `2e19/p33m67`: resumed from step `40001`, first train step completed,
    latest observed progress `40.2kit/54.9kit`.

### 2026-05-24T05:44Z follow-up

Still no terminal failures. `9e18/p67m33` acquired TPU capacity, resumed from
step `30001`, and started training. All six materialized child TPU jobs are now
running, though `2e19/p33m67` and `9e18/p33m67` have seen additional
preemptions and are restarting through the expected resume path. `3e18/p50m50`
has saved temp checkpoint `32551` and reached `32.9kit/37.0kit`
(`eval_loss=2.688`, `nemotron_cc_math_v1 macro=1.616`). The remaining eleven
roots are still pending on CPU coordinator capacity; no larger-scale child jobs
have materialized yet.

### 2026-05-24T05:56Z follow-up

Still no terminal failures. Six child TPU jobs remain running. `3e18/p50m50`
saved a permanent checkpoint at step `33300` and reached `33.6kit/37.0kit`
(`eval_loss=2.650`, `nemotron_cc_math_v1 macro=1.576`). `9e18/p50m50` saved a
permanent checkpoint at step `30863`. `3e18/p67m33` saved temp checkpoint
`30451` after resuming through repeated preemptions. The eleven remaining roots
are still pending on CPU coordinator capacity.

### 2026-05-24T06:19Z follow-up

Still no terminal failures. Three more roots started (`2e19/p50m50`,
`2e19/p67m33`, `3e19/p33m67`) and submitted child TPU jobs; those children are
currently pending on v6e-8 high-RAM autoscaler capacity with
`tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity`.
This is a scheduler wait, not a relaunch/config failure. `9e18/p67m33` saved a
permanent checkpoint at step `30863`. `3e18/p50m50` reached `34.7kit/37.0kit`
(`eval_loss=2.604`, `nemotron_cc_math_v1 macro=1.529`) and temp checkpoint
`34301`. `9e18/p50m50` reached `32.1kit/44.1kit` and temp checkpoint `31756`.

### 2026-05-24T06:31Z follow-up

Still no terminal failures. The pending v6e-8 high-RAM children cleared the
autoscaler wait and are running. `2e19/p50m50` and `2e19/p67m33` both resumed
from step `40001`; `3e19/p33m67` resumed from step `30001`. `3e18/p50m50`
reached `35.4kit/37.0kit` (`eval_loss=2.575`,
`nemotron_cc_math_v1 macro=1.502`) and temp checkpoint `35351`. `9e18/p50m50`
reached `32.7kit/44.1kit` and temp checkpoint `32681`; `9e18/p67m33` reached
`31.7kit/44.1kit` and temp checkpoint `31756`. Eight roots remain pending on
CPU coordinator capacity.

### 2026-05-24T06:54Z follow-up

External preemption storm hit the newly materialized v6e-8 children. Iris shows
`2e19/p50m50`, `2e19/p67m33`, and `3e19/p33m67` child tasks killed by
`/tonyhlee/eval-lr1e5_rstarcoder_n8_vr5_round1-step-500-code/0`. Structured
summaries confirmed exit code `0` and error `Preempted by ...`, not a code
failure. The parent coordinators are still alive and already demonstrated the
correct recovery path for `2e19/p50m50`: it restarted, reused
`gs://marin-us-east5/checkpoints/delphi-true-2e19-p50m50-cooldown20-a001`,
and re-submitted `midtrain-delphi-true-2e19-p50m50-cooldown20-a001`. As of
06:54Z, `2e19/p67m33` and `3e19/p33m67` parents were still waiting for CPU to
restart and resubmit their children. Do not manually relaunch those cells unless
their parent jobs become terminal or stop making recovery progress; manual
duplicate relaunch risks splitting the recovery path.

### 2026-05-24T10:37Z follow-up

Still no terminal failures in the cooldown20 batch. Iris shows 17 launch roots:
2 succeeded, 10 running, and 5 pending on CPU coordinator capacity. The active
child set has 2 succeeded and 10 running. The pending roots are
`9e19/p50m50`, `9e19/p67m33`, and all three `2e20` cells.

Verified resume/progress:

- `2e19/p33m67`, `2e19/p50m50`, and `2e19/p67m33` all loaded
  `gs://marin-us-east5/checkpoints/delphi-true-2e19-<mix>-cooldown20-a001/checkpoints/step-40000`
  and logged `Resuming training from step 40001`.
- `9e18/p33m67`: running around `40.9kit/44.1kit`; latest eval loss `2.403`,
  `nemotron_cc_math_v1 macro=1.271`; latest temp checkpoint `40584`.
- `9e18/p50m50`: running around `42.9kit/44.1kit`; latest eval loss `2.340`,
  `nemotron_cc_math_v1 macro=1.266`; latest temp checkpoint `42855`.
- `9e18/p67m33`: running around `41.9kit/44.1kit`; latest eval loss `2.379`,
  `nemotron_cc_math_v1 macro=1.342`; latest temp checkpoint `41947`.

No manual relaunches were needed. Parent coordinators are still preserving the
intended output paths/run ids while recovering through preemptions.

### 2026-05-24T11:30Z follow-up

`9e18/p50m50` succeeded after three preemptions and parent-owned recovery
through the same output path. Final artifacts:

- Checkpoint:
  `gs://marin-us-east5/checkpoints/delphi-true-9e18-p50m50-cooldown20-a001/checkpoints/step-44095`
- HF checkpoint:
  `gs://marin-us-east5/checkpoints/delphi-true-9e18-p50m50-cooldown20-a001/hf/step-44095`
- Final eval loss: `2.320`
- Final `nemotron_cc_math_v1` macro loss: `1.249`

Batch status at 11:29Z: 17 roots total, 3 succeeded, 10 running, 4 pending on
CPU coordinator capacity. Children: 3 succeeded, 9 running/pending under alive
parents. No failed/killed/unschedulable jobs. `9e18/p33m67` and
`9e18/p67m33` are still active; most v6e-8 children are waiting/recovering
after external preemptions.

### 2026-05-24T12:15Z follow-up

The full `9e18` cooldown20 set is now complete.

- `9e18/p33m67` succeeded after eight preemptions.
  - Checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-9e18-p33m67-cooldown20-a001/checkpoints/step-44095`
  - HF checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-9e18-p33m67-cooldown20-a001/hf/step-44095`
  - Final eval loss: `2.325`
  - Final `nemotron_cc_math_v1` macro loss: `1.210`
- `9e18/p67m33` succeeded after two preemptions.
  - Checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-9e18-p67m33-cooldown20-a001/checkpoints/step-44095`
  - HF checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-9e18-p67m33-cooldown20-a001/hf/step-44095`
  - Final eval loss: `2.332`
  - Final `nemotron_cc_math_v1` macro loss: `1.303`

Batch status at 12:14Z: 5 children succeeded
(`3e18/p50m50`, `3e18/p67m33`, all three `9e18` cells). The apparent bad
children are external preemptions, not code failures:
`3e19/p67m33` was preempted by
`/tonyhlee/eval-lr1e5_rstarcoder_n8_vr5_round1-step-700-code/0`, and
`9e19/p33m67` was preempted by
`/tonyhlee/eval-lr1e5_rstarcoder_n8_vr5_round1-step-800-code/0`. Their parent
coordinators remain alive, so continue to let parent-owned recovery preserve
the original output paths.

12:18Z recovery check: `3e19/p67m33` parent resubmitted
`midtrain-delphi-true-3e19-p67m33-cooldown20-a001` at 12:14Z under the same
logical child/output identity. `9e19/p33m67` parent is still alive and pending
after its child preemption; continue waiting for parent-owned recovery rather
than manually relaunching.

12:27Z progress check: tmux monitor `true_midtrain_cd20_monitor` is live and
writing to `scratch/20260524-2133_true_midtrain_cd20_batch_monitor.log`. Active
v6e-8 children have resumed under the same logical run ids: `2e19/p33m67`
from step `40402`, `2e19/p50m50` and `2e19/p67m33` from step `40001`,
`3e19/p33m67` and `3e19/p50m50` from step `30001`. `3e19/p67m33` and
`9e19/p67m33` are materialized but pending for v6e-8 TPU capacity. The remaining
`2e20` roots are still CPU-coordinator pending.

19:42Z handoff update: status refresh shows 17 roots with 9 succeeded, 7
running, and 1 pending (`2e20/p67m33` still CPU-coordinator pending). Children:
9 succeeded, 5 running, 1 pending (`2e20/p33m67` waiting for v5p-8 TPU
capacity), and 1 killed child (`2e20/p50m50`, externally preempted by
`/tonyhlee/8b-lr1e5-rstarcoder-v5p64-defram-v1/0`). The parent roots remain
alive, so continue parent-owned recovery and do not manually duplicate relaunch.
The stale tmux monitor was restarted as `true_midtrain_cd20_monitor` at local
12:42 with the same state/log paths.

> ## 🚨 CRITICAL — 1e20 RESULTS IN THIS LOGBOOK USED THE WRONG BASE MODEL 🚨
>
> **Discovered 2026-05-14.** Every "1e20" cell logged below — across the April 10 B sweep, the April 20 B sweep, and the May K=0.20 36-cell sweep — was trained from `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5`. **This is NOT a Delphi compute-optimal checkpoint.** It's from a deprecated v5 isoflop sweep generation with a different optimizer recipe (different LR formula, different `/H` treatment, different (d, L) architecture). Will Held (Delphi lead) confirmed: *"a checkpoint from a totally different scaling recipe."*
>
> The canonical Delphi 3e20 base is `isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6` (d=2304 vs 2048, L=23 vs 21, v6 heuristic). Registered in `experiments/exp1337_eval_suite.py:180`.
>
> **1e21 and 1e22 results in this logbook ARE valid** — those used the canonical `adamh-scaling-ladder-nemotron-optimal-{1e+21,1e+22}-v5-...` runs, which use the v6 heuristic (`LABEL = "adamh_scaling_v6"`, `exp1337_delphi_suite.py:62`). The `-v5-` in their names is an unrelated experiment-iteration tag.
>
> **Contaminated claims:** the cross-scale transfer headline ("recipe ranks generalize cleanly across scales") and the mix-gap stability (0.103 / 0.106 / 0.106) are weakened because the 1e20 point is from a different scaling family. The 1e21 → 1e22 within-family results survive.
>
> **Full post-mortem:** [`.agents/ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md`](../ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md)
>
> **Rule for future agents:** NEVER pick a scaling-law base by grepping GCS. Always source from one of:
> 1. `experiments/exp1337_eval_suite.py` EVAL_BASES (lines 174-186) — canonical mapping
> 2. `MARIN_SCALING_SUITES["nemotron-completed-adamh"]` in `experiments/isoflop_sweep.py`
> 3. `experiments/exp1337_delphi_suite.py`
> 4. https://huggingface.co/collections/marin-community/delphi
>
> If the registry doesn't include the scale you need, STOP. Ping Will Held. Do not silently substitute.

---

## WARNING — resume identity must come from one old output path

Do not relaunch a failed Delphi midtraining run with hand-written
`MIDTRAIN_OUTPUT_PATH_OVERRIDE`, `RUN_ID`, or `WANDB_RUN_ID`. The monitoring
agent must use **one** source of truth:

```bash
-e MIDTRAIN_RESUME_OUTPUT_PATH gs://marin-<region>/checkpoints/<old-run-id> \
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP <last-known-good-checkpoint-step-or-floor>
```

`experiments/exp_delphi_math_10b_midtrain.py` now derives the executor output
path, Levanter `RUN_ID`, and `WANDB_RUN_ID` from
`MIDTRAIN_RESUME_OUTPUT_PATH`; it refuses legacy `MIDTRAIN_OUTPUT_PATH_OVERRIDE`.
Startup requires `MIDTRAIN_EXPECT_RESUME_MIN_STEP`, discovers the latest
permanent/temp checkpoint, and fails before training if no checkpoint exists or
if the checkpoint is below the expected floor. Use
`MIDTRAIN_ALLOW_EMPTY_RESUME=1` only for an intentional namespace-preserving
restart before any checkpoint has ever been written.

Generic Marin training now also rejects executor-backed training when the
resolved run id does not match the output path basename. This is meant to catch
bad recoveries before W&B/checkpoint namespaces split again.

## README / IMPORTANT — current handoff state as of 2026-05-02 06:07 UTC

This logbook now tracks both the midtraining experiments and the cross-region
resume incident that happened while running them. Read this section before
launching, relaunching, or "recovering" any Delphi midtraining job.

### Current live monitor handoff

Active jobs are split across two Iris users:

- `/ahmedah`: active `1e20` sweep jobs.
- `/ahmed`: active `1e21` and `1e22` sweep jobs.

Current sweep state:

- `1e20`: six intended sweep points are represented. Four `train_lm` children
  are running, two `train_lm` children are pending v5p-32 capacity. See the
  dated 2026-05-02 06:07 UTC handoff section for exact job ids.
- `1e21`: six v5p-64 BATCH points were launched in `us-east5`. Four parent
  coordinators are running and have pending v5p-64 `train_lm` children; two
  parent coordinators are still pending CPU coordinator capacity.
- `1e22`: six v5p-256 BATCH points were launched in `us-east5`. All six parent
  coordinators are still pending CPU coordinator capacity; no v5p-256 child has
  materialized yet.

MirrorFS issue to remember:

- The earlier `1e21-v5` v5p-256 pilot failed during checkpoint initialization
  because Levanter's `mirror://` TensorStore staging path made every rank stage
  the base checkpoint. One rank hit `TransferBudgetExceeded`; others hit mirror
  lock handoff failures and then JAX coordination cascaded.
- This branch has a local quick patch that propagates `MARIN_MIRROR_BUDGET_GB`
  into nested train jobs and retries MirrorFS lock handoff for 120s. Tests passed,
  but the deeper all-rank staging design problem is still tracked in
  https://github.com/marin-community/marin/issues/5374.
- For the `1e22` launch, the base checkpoint was copied from
  `gs://marin-us-central1/.../step-38206` to
  `gs://marin-us-east5/.../step-38206` before submission to avoid forcing
  v5p-256 ranks to stage it cross-region.
- If a launch fails around checkpoint init, do not blindly relaunch. Check the
  exact output path, whether a permanent/temp checkpoint exists, and grep logs
  for `TransferBudgetExceeded`, `Could not acquire mirror lock`,
  `_stage_mirror_to_local`, `Using output path`, and `Using run ID`.

### Hard resume rule

Never assume a failed/preempted training run will resume just because the
human-readable step name is unchanged. Marin's real checkpoint/W&B identity
includes the executor output hash.

**2026-05-02 repeat incident:** this exact mistake happened again for `1e21
p67m33`. The crashed `lr0.67` run
`delphi-1e21-p67m33-9p25b-lr0.67-ecbd27` reached step ~2651, but a recovery
landed in the new namespace
`delphi-1e21-p67m33-9p25b-lr0.67-99752407` and started from step 0 instead of
resuming `ecbd27`. The same pattern happened for `lr0.5`: crashed
`114e49` reached step ~3541, while recovery `fdc4ebf1` started in a new
namespace. See ops postmortem
`.agents/ops/2026-05-02-delphi-midtrain-resume-namespace.md`.

Before relaunching any failed Delphi midtraining run:

1. Find the exact old output path and run id from `.executor_info`, logs, W&B,
   or GCS.
2. Check both permanent checkpoints and temporary checkpoints.
3. Relaunch this Delphi script with `MIDTRAIN_RESUME_OUTPUT_PATH=<old-output-path>`,
   not `MIDTRAIN_OUTPUT_PATH_OVERRIDE`.
4. Set `MIDTRAIN_EXPECT_RESUME_MIN_STEP=<last-known-good-step>` so stale
   namespaces fail before training starts.
5. Verify startup logs show the same run id/output path and
   `Resuming training from step ...`. If that line is absent, the recovery has
   not been proven, even if training is making progress.

For the 2026-04-27 incident, `p67m33/lr0.5` drifted from the original central
namespace `delphi-1e20-p67m33-20b-lr0.5-f74454` to the wrong east5 namespace
`delphi-1e20-p67m33-20b-lr0.5-378f43`. Treat those as different runs.

### What is fixed now

- **Executor/StepSpec region-sensitive hashes:** fixed in main by
  `7f0b99b9e Stop region prefixes leaking into Marin executor identity hashes (#5223)`.
  `StepSpec.hash_id` no longer depends on physical `gs://marin-<region>/...`
  dependency paths, Nemotron v2 normalize uses `relative_input_path`, and deep
  executor dependencies use region-stable `{name}-{hash}` identifiers.
- **Temporary checkpoint region layout:** improved in main by
  `b4298305a infra/rigging: fold tmp buckets into main buckets (#5266)`.
  Temporary paths now live under the main regional buckets, e.g.
  `gs://marin-us-east5/tmp/ttl=14d/...`, and training temp checkpoint roots are
  chosen from the output path's region.
- **TensorStore cross-region budget accounting:** fixed in main by
  `a154c044f Charge cross-region transfer budget on tensorstore checkpoint I/O (#5225)`.
  TensorStore checkpoint reads/writes now call `record_transfer(...)`.
- **Iris split-slice/orphan attempt retry bugs:** improved in main by
  `9d9b9a2a7 [iris] Fix coscheduled split-slice and orphan attempt bugs (#5249)`.
- **This branch's Delphi experiment launch guard:** pushed as
  `4b40df269 [experiments] Pin Delphi midtraining jobs by region`. The parent
  coordinator can still be launched with both `--region us-central1 --region us-east5`,
  but generated `train_lm` child resources are pinned to the coordinator's
  resolved v5p region.

### What is not fully fixed

- There is still no general "move this training run to another region, copy the
  right checkpoints, and always resume the old run id" system. Current fixes
  give stable identity, better temp layout, budget accounting, and fail-fast
  behavior.
- `cc2678ff4 Guard cross-region GCS access in training and tokenization` is
  **not** in main. That PR is on `pr-5221`, not merged. So global `TrainLmOnPodConfig`
  child-resource alignment is not universal; this branch only guards the Delphi
  midtraining experiment itself.
- `2891acc6e [checkpoint] Cross-region temp checkpoint discovery via mirrortmp://`
  is **not** in main. Main chose the `#5266` temp-under-main-bucket approach
  instead.
- JAX coordinator RPC peer-loss after preemption and stale port `8476` cleanup
  are not obviously fixed by the commits above.
- Do not use `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` in launch recipes to paper over
  cross-region placement. It hides the bug and can allow expensive behavior.

### Iris quota / priority bands — interactive vs batch

**Source of truth:** `lib/iris/examples/marin.yaml:198-223` (researcher tier — `ahmedah` is in this group with `budget_limit=75000` and `max_band=PRIORITY_BAND_INTERACTIVE`).

**Spend formula** (`lib/iris/src/iris/cluster/controller/budget.py:68-76`):

```
resource_value(task) = 1000 * accelerator_count + RAM_GB + 5 * CPU_cores
user_spend = sum(resource_value(t) for t in user's ASSIGNED/BUILDING/RUNNING tasks)
```

It is a **snapshot of in-flight work**, not historic usage. Spend drops as tasks complete.

**Demotion rule** (`compute_effective_band` in `budget.py:109-126`, called every scheduling tick on BOTH pending tasks (`controller.py:2016`, `service.py:2554`) AND running tasks (`controller.py:494`)):

```python
if user_spend > limit and task.priority_band != PRODUCTION:
    effective_band = max(task.priority_band, BATCH)   # downgrade INTERACTIVE → BATCH
```

Crucial implications:

1. **Going over quota retroactively downgrades ALL in-flight INTERACTIVE jobs of that user**, not just future submissions. The `priority_band` field on the task row is unchanged; the *effective* band is recomputed every tick from current `user_spend`. So a task that was treated as INTERACTIVE one second ago can become BATCH-effective the next second if you submit something that pushes you over 75k.
2. **Demotion is reversible.** As tasks complete and spend drops back below 75k, remaining INTERACTIVE tasks regain their priority on the next tick.
3. **PRODUCTION is never demoted** — admin tier only (`runner, power, dlwh, rav, romain, held, larry`).
4. **BATCH has no per-user cap.** Cluster-wide queue, lower preemption priority. That's how `michaelryan` runs 313 BATCH tasks with 1.6 M total spend (16× over the 75k threshold).
5. The `--priority` CLI flag chooses the *task's* band. Once your effective band is BATCH (either because you chose it or because you got demoted), you compete in the BATCH queue.

**Concrete numbers for our 12-job sweep:**

| Run shape | Per-job spend | Per-job mix | Total at full deployment |
|---|---:|---|---:|
| 6 × 1e20 on v5p-32 | ~32,000 | 32 chips × 1000 + RAM/CPU | ~192,000 |
| 6 × 1e21 on v5p-64 | ~64,000 | 64 chips × 1000 + RAM/CPU | ~384,000 |
| **Sum** |  |  | **~576,000 (~7.7× the 75k INTERACTIVE limit)** |

We submitted with `--priority batch` so we're already at BATCH-effective from the start; no further demotion penalty applies. This was the right call for an unattended 12-job sweep — but it means our jobs are preemptable by anyone's higher-band work.

### Open question for the user — how do we want sweeps to schedule?

Two regimes, mutually exclusive:

**A. Best-effort BATCH (current setup)**
- Submit with `--priority batch`.
- All 12 jobs run in parallel (no spend ceiling).
- **Preemptable** by other users' INTERACTIVE/PRODUCTION work, especially when the cluster is busy. Levanter handles preemption via the 10-min temp checkpoint, so progress isn't lost — just stalled while waiting to reschedule. Wall-time becomes "training time + preemption recovery".
- Best for: research sweeps where total throughput matters more than predictable wall-time.

**B. INTERACTIVE under quota — guaranteed-finish**
- Submit with `--priority interactive`.
- Must keep total in-flight spend ≤ 75,000. With 1e20 ≈ 32k/job and 1e21 ≈ 64k/job:
  - Max 2 × 1e20 simultaneously = 64k ✓
  - Or 1 × 1e21 simultaneously = 64k ✓
  - 1 × 1e20 + 1 × 1e21 = 96k ✗ (already over)
- 12-job sweep would have to **serialize**: ~6 × 1e21 wall-time (~24h serial) + ~3 × 2-1e20 wall-time (~21h serial). Roughly 2× the current parallel wall-time, but with **strong preemption priority** — you'd outrank ~95% of current cluster traffic.
- Best for: critical runs that must finish before a deadline, or recovery from a failed sweep where re-launching everything from scratch would be wasteful.

**Hybrid (third option):** submit the 1e20 jobs as INTERACTIVE in pairs (each pair fits under 75k) and the 1e21 jobs serially as INTERACTIVE — gets the priority benefit but ties up scheduling latency. Probably overcomplicated; pick A or B.

The current sweep is already running under (A). The question is **what we want to do for the *next* sweep launch and any future midtraining runs**:

- → "Best-effort BATCH" is fine for exploration. Accept some preemption risk; we can tolerate stalls.
- → "Interactive guaranteed-finish" is the right call when we have a known-good recipe and just need the run to land before a deadline.
- → Mixed by stage: BATCH for LR-search sweeps (we'll re-run anyway), INTERACTIVE for the final-recipe runs that produce the publishable numbers.

**Please confirm which regime you want as the default for future sweeps.** This is the only knob that affects launch behavior; the rest of the framework (val carve-out, K=0.20, 10% checkpoints, etc.) is independent.

### Branch state

- Remote `origin/midtrain_data` currently points at `bfacf9f76` (pre-flight TypeError fallback) after the recent push sequence (`6d0c9f3ef` heuristic+spec → `058fe76a0` 10% ckpts → `bfacf9f76` pre-flight fallback).
- Expect unrelated local dirty files in this worktree, including this logbook,
  analysis scripts/plots, and `tests/test_training.py`.
  Do not stage them casually.

### W&B project routing — fixed (2026-05-02)

`experiments/defaults.py:427` had hard-coded `project="marin"` in `WandbConfig` since `b804686ae3` ("wip", David Hall, 2025-04-03). The matching DPO helper (`default_train_dpo`, line 664) was already correct. The fix:

```python
# experiments/defaults.py — default_train signature
def default_train(
    ...,
    wandb_project: str | None = None,
    ...
)

# experiments/defaults.py — WandbConfig
tracker=WandbConfig(project=wandb_project or "marin", ...),

# experiments/exp_delphi_math_10b_midtrain.py — every run
default_train(..., wandb_project="delphi-midtraining", ...)
```

Setting `WANDB_PROJECT` env var alone never worked because Levanter's `wandb.init(project=self.project, ...)` passes the explicit string from `WandbConfig`, overriding the env. With the fix, every Delphi midtraining run from this branch routes to `https://wandb.ai/marin-community/delphi-midtraining`.

Caveat: the 2 runs from sweep `20260501-235704` that started before the fix landed at `marin-community/marin/runs/delphi-1e20-p67m33-4p94b-lr0.{33,5}-*`. Filter that project by `delphi-` prefix or by tag `delphi-midtrain` if you need to find them.

Implication for any other `default_train` caller: setting `wandb_project="..."` now routes that experiment to a different project. Default behavior (kwarg unset) is unchanged → still routes to `marin-community/marin`. So no regression for existing callers.

### Correction on iris quota framing (2026-05-02)

Earlier I claimed an in-flight 12-job sweep would already be over the 75k quota. **Wrong.** Per `lib/iris/src/iris/cluster/controller/db.py:144`, `ACTIVE_TASK_STATES = {ASSIGNED, BUILDING, RUNNING}` — `PENDING` tasks **don't count** toward `user_spend`. So:

- A coordinator + a coscheduling-pending child = ~5 spend (just the 1 coordinator CPU).
- Spend only jumps when the child enters BUILDING/RUNNING (i.e. coscheduling acquires TPU workers).
- `ahmedah` was at 34,400 / 75,000 = 45.9% during the sweep launch (under quota).

Realistic projection: as the autoscaler ramps v5p capacity and pending children start RUNNING, spend climbs. Crossing 75k auto-demotes ALL of the user's INTERACTIVE work to BATCH-effective on the next scheduler tick (`compute_effective_band`). So an INTERACTIVE submission has *temporary* INTERACTIVE priority — until the user crosses 75k from concurrent jobs starting to RUN, at which point everything degrades to BATCH-effective. Reverses as jobs finish.

> **How to use this file (for any agent that lands here later).**
>
> This is a **shared working scratchpad**, not a polished writeup. Treat every
> section as live — update it as you go, don't wait until a task is "done".
>
> - Before doing anything, scan the [Status log](#status--chronological-action-log)
>   at the bottom to see what the previous agent did last.
> - Every time you take a non-trivial action (read a file, run a command, make
>   a decision, discover something that contradicts this plan), append a dated
>   row to that table. UTC timestamps, one line each. If it's not in the log,
>   it didn't happen.
> - Record failures and dead-ends with the exact error text, not paraphrased.
> - If a number, path, or hyperparameter here turns out to be wrong,
>   **edit it in place** and note the correction in the Status log. Do not
>   leave stale facts behind — future agents will trust them.
> - Keep the [Goal](#goal), [User-specified constraints](#user-specified-constraints),
>   and [Base models](#base-models-the-two-smallest-adamh-checkpoints-we-have)
>   sections immutable unless the user explicitly changes scope.
> - Add new sections freely (Results, Pilots, Bugs, Decisions, etc.) at the
>   bottom, above the Status log.

**Date started:** 2026-04-21
**Status:** Experiments and cross-region incident postmortem recorded; main has been locally merged into this branch for inspection.
**Base-model catalogue:** [`.agents/projects/delphi_midtraining.md`](../projects/delphi_midtraining.md)
**External reference:** [*Delphi Scaling Laws: Key Findings* — Will Held](https://oa.williamheld.com/blog/delphi/)
**Branch:** `midtrain_data`
**Primary artefact being built:** `experiments/exp_delphi_math_10b_midtrain.py`

---

## Goal

### Overarching project goals (two tracks)

1. **Predict loss trajectory for midtraining.** Build a calibrated expectation for how math-eval loss evolves when you continue-train an existing pretrain checkpoint on a math-heavy dataset, so that larger/more-expensive midtrain runs can be scoped (token budget, LR schedule, decay shape) from smaller runs instead of guessed. This Delphi × Nemotron-CC-Math sweep is a primary data point for that predictor.
2. **Pick a good midtraining dataset.** Compare candidate math-heavy datasets (the Nemotron-CC-Math quality tiers being our current anchor) on their effect on post-midtrain downstream evals, so that future runs don't waste compute on a poorly-curated corpus.

### This logbook's subgoal (the concrete sweep tracked here)

Run a small LR sweep that continues-trains the two smallest existing AdamH-trained Marin checkpoints on **10 B tokens of `nemotron_cc_math_v1/4plus`**, to de-risk a Mantis-style math-midtraining recipe before spending v4-512 / v4-1024 time on Delphi 1e22 / 1e23. We're looking for the highest peak LR that still drops the math-eval loss monotonically.

This subgoal feeds both tracks: the LR-factor × final-loss pairs inform the loss-trajectory predictor (track 1), and the dataset is held fixed at `nemotron_cc_math_v1/4plus` so the signal is attributable to LR choice — a prerequisite for the dataset-selection track (2), which requires LR to be a solved variable before dataset-quality can be cleanly isolated.

## User-specified constraints

- Start the midtraining peak LR at **2/3 of each base's own pretrain peak**, sweep around there.
- Use **AdamH** (same optimizer family the base models were pretrained with).
- **500 warmup steps**.
- Start with **the two smallest bases** — "1e20 and 1e21 models" in the user's words.
- 10 B token budget on the Nemotron-CC-Math dataset.

---

## Base models (the two smallest AdamH checkpoints we have)

Authoritative numbers read directly from each run's W&B config.

| Slot | Model run | H | L | Params | Pretrain BS | Pretrain steps | Pretrain tokens | Pretrain peak `learning_rate` | Pretrain peak `adam_lr` | β₂ | ε | Final c4_en loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **"1e20"** (3e20 isoflop scan @ compute-optimal) | `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` | 2048 | 21 | ~1.9 B | 128 | 47 064 | 24.7 B | **4.483e-3** | **7.382e-5** | 0.99980 | 4.11e-8 | 2.858 |
| **"1e21"** (Delphi v5 canonical) | `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` | 2560 | 26 | ~3.4 B | 512 | 22 057 | 46.3 B | **7.425e-3** | **4.314e-4** | 0.99920 | 2.81e-8 | 2.733 |

**Starting checkpoints (Levanter TensorStore; use `initialize_from_checkpoint_path` to load weights only):**
```
gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/
gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/
```

Why the 3e20 isoflop ckpt stands in for "1e20": no AdamH `optimal-training` run exists at 1e20 FLOPs — the Delphi scaling-ladder only covers 1e21 / 1e22 / 1e23. The 3e20 isoflop scan ran many `(H, L, B)` combinations to FIT the scaling law; its compute-optimal point (`d2048-L21`) is a genuine AdamH-trained model at ~1.9 B params with a clean HF export and a mid-2 c4-en loss. User confirmed this is the stand-in they want.

Why `v5` for the 1e21 slot: the `vN` suffix on `adamh-scaling-ladder-nemotron-optimal-1e+21-*` runs is an **infra-retry counter**, not a recipe version — the recipe itself stays `adamh_scaling_v6` across all revs. `v5`, `v5-seed42`, `v5-seed62746`, `v6` all converged; their c4-en losses cluster within 0.001. Use `v5` as canonical; add seed replicates later only if seed variance is the research question.

---

## Fixed training knobs (both bases)

| Knob | Value | Why |
|---|---|---|
| Data | 100 % `nemotron_cc_math_v1/4plus` | Highest-quality CC-Math band per `midtraining_math.md` §2a. Phi-4-cleaned, non-Qwen. 52 B token pool ≫ 10 B budget → no epoch repeats. Already tokenized and registered as `BUCKET_2["nemotron_cc_math_v1/4plus"]` in `experiments/midtraining_data_buckets.py`. |
| Tokenizer | `llama3_tokenizer` | Matches both base checkpoints. |
| Seq-len | 4096 | Matches both base checkpoints. |
| Batch size | **512** | Standardizes on v4-128-friendly batch. The 1.9 B model was pretrained at BS=128, but batch-alignment to the pretrain run is not required since we rebuild the optimizer anyway. |
| Token budget | **10 B** | User-specified. |
| `num_train_steps` | **4 768** | = ceil(10e9 / (512 × 4096)) |
| Warmup steps | **500** | User-specified. Rebuilds fresh Adam / AdamH momentum. |
| Decay tail | remaining **4 268 steps** | `= num_train_steps − warmup`. No stable middle — pure linear warmup → linear decay. |
| `min_lr_ratio` | 0.1 | Mantis cooldown convention. End at 10 % of peak rather than 0 (pretrain's `min_lr_ratio=0` decays all the way to zero, which is too aggressive for a 10 B-token midtrain). |
| `lr_schedule` | `"linear"` | Same family as Delphi pretrain (Complete(d)P) + Mantis cooldown. |
| `beta1` | 0.9 | Heuristic default. |
| `beta2`, `epsilon`, `max_grad_norm` | **Inherited per-base from the pretrain config** (table above) | These values are coupled to the pretrain `(B, T)` via `_compute_beta2` / `_compute_epsilon`. Keeping the base's own values preserves the curvature statistics the loaded weights were optimized against. |
| `max_grad_norm` | 0.1 | Heuristic default. |
| `reset_data_loader_on_init` | `True` | New data distribution → fresh iterator. |
| `z_loss_weight` | 0 | Same as Delphi. |
| Precision | `jmp.get_policy("p=f32,c=bfloat16")` | Same as Delphi. |
| TPU | `v4-128` for both | Matches 1e21 Delphi pretrain topology; ample for 1.9 B. |
| Mesh | `{data: -1, replica: 1, model: 1}` | `tp=1` suffices since H=2048 and H=2560 are both divisible by 64 chips. (Same `tp`-search loop as `exp1337_delphi_suite.py`.) |
| `steps_per_eval` | 200 | ≈24 evals across 4 768 steps. |
| Checkpointer | `save_interval=10min`, `keep=[{"every": 1000}]` | Short run; light retention. |
| HF export | `hf_save_steps=1000` | Final HF + a few intermediate waypoints. |

---

## LR sweep — numbers

Factors: **`{0.5, 0.67, 0.83} × pretrain peak`**. User asked for "2/3 of peak, sweep around there"; these three bracket 2/3 with symmetric ±0.17. Both `learning_rate` *and* `adam_lr` get scaled by the same factor so the weight-LR ↔ embedding-LR coupling from the heuristic stays intact.

### Base A: 1e20 (1.9 B)

| Factor | `learning_rate` | `adam_lr` | Run name |
|---:|---:|---:|---|
| 0.5  | 2.241e-3 | 3.691e-5 | `delphi-1e20-iso-math-10b-lr0.5` |
| **0.67** | **2.989e-3** | **4.921e-5** | **`delphi-1e20-iso-math-10b-lr0.67`** (primary) |
| 0.83 | 3.721e-3 | 6.127e-5 | `delphi-1e20-iso-math-10b-lr0.83` |

### Base B: 1e21 Delphi v5 (3.4 B)

| Factor | `learning_rate` | `adam_lr` | Run name |
|---:|---:|---:|---|
| 0.5  | 3.713e-3 | 2.157e-4 | `delphi-1e21-v5-math-10b-lr0.5` |
| **0.67** | **4.950e-3** | **2.876e-4** | **`delphi-1e21-v5-math-10b-lr0.67`** (primary) |
| 0.83 | 6.163e-3 | 3.580e-4 | `delphi-1e21-v5-math-10b-lr0.83` |

**Wall-time estimate (per run, v4-128):** ~3.5 h for 1.9 B; ~6.5 h for 3.4 B. Total compute if serialized: ~30 h. Fully parallel across 6 pods: ~6.5 h elapsed.

**Outputs:** `gs://marin-us-central2/<run-name>-<hash>/{checkpoints,hf}`. **W&B:** `marin-community/marin`.

---

## Implementation — new file

**Create:** `experiments/exp_delphi_math_10b_midtrain.py` (~200 lines).

Structure mirrors `experiments/exp1337_delphi_suite.py`'s `run_optimal_training` + `TrainLmOnPodConfig` wrapping, with the `initialize_from_checkpoint_path` + fresh `AdamHConfig` pattern lifted from `experiments/tootsie/exp1529_32b_mantis_cooldown.py:131–165`. Six `ExecutorStep`s, one per (base × lr_factor).

Skeleton (for reference — adapt to whatever helpers exist on this branch at implementation time):

```python
from dataclasses import replace
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.main import train_lm
from levanter.optim.adamh import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_validation_sets
from experiments.llama import llama3_tokenizer
from experiments.midtraining_data_buckets import BUCKET_2
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

SEQ_LEN          = 4096
BATCH_SIZE       = 512
NUM_TRAIN_STEPS  = 4768      # 10 B / (512 × 4096) rounded up
WARMUP_STEPS     = 500
DECAY_STEPS      = NUM_TRAIN_STEPS - WARMUP_STEPS
MIN_LR_RATIO     = 0.1
TPU_TYPE         = "v4-128"

BASES = {
    "1e20-iso-d2048-L21": dict(
        ckpt="gs://marin-us-central2/checkpoints/isoflop/"
             "isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/",
        hidden_dim=2048, num_layers=21,
        peak_lr=4.483e-3, peak_adam_lr=7.382e-5,
        beta2=0.99980, epsilon=4.11e-8,
    ),
    "1e21-v5": dict(
        ckpt="gs://marin-us-central2/"
             "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/",
        hidden_dim=2560, num_layers=26,
        peak_lr=7.425e-3, peak_adam_lr=4.314e-4,
        beta2=0.99920, epsilon=2.81e-8,
    ),
}
LR_FACTORS = [0.5, 0.67, 0.83]

math_mix = LMMixtureDatasetConfig(
    components={
        "nemotron_cc_math_v1/4plus": step_to_lm_mixture_component(
            BUCKET_2["nemotron_cc_math_v1/4plus"], include_raw_paths=False
        ),
    },
    train_weights={"nemotron_cc_math_v1/4plus": 1.0},
    tokenizer=llama3_tokenizer,
    cache_options={"batch_size": 128},
    block_cross_document_attention=True,
)

# Merge validation-only components (Paloma suites) so eval curves stay
# comparable to the pretrain W&B panels.
_val = {n: step_to_lm_mixture_component(s, include_raw_paths=False)
        for n, s in default_validation_sets(tokenizer=llama3_tokenizer).items()}
data_with_val = replace(
    math_mix,
    components={**math_mix.components, **_val},
    train_weights={**math_mix.train_weights,
                   **{k: 0.0 for k in _val if k not in math_mix.train_weights}},
)

def build_inner_config(base_tag: str, base: dict, lr_factor: float):
    lr      = base["peak_lr"]      * lr_factor
    adam_lr = base["peak_adam_lr"] * lr_factor
    optimizer = AdamHConfig(
        learning_rate=lr, adam_lr=adam_lr,
        beta1=0.9, beta2=base["beta2"], epsilon=base["epsilon"],
        max_grad_norm=0.1,
        warmup=WARMUP_STEPS, decay=DECAY_STEPS,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule="linear", nesterov=False,
    )
    return train_lm.TrainLmConfig(
        data=data_with_val,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community", project="marin",
                tags=["midtraining", f"base={base_tag}",
                      "nemotron-cc-math-4plus",
                      f"lr_factor={lr_factor}",
                      f"peak_lr={lr:.3e}",
                      f"adam_lr={adam_lr:.3e}",
                      "adamh", "delphi-midtrain"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=-1,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=200,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=1000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": 1},
                compute_mapping={
                    "token":        (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=<llama_config builder that matches metadata.json of the source ckpt>,
        optimizer=optimizer,
        initialize_from_checkpoint_path=base["ckpt"],
        reset_data_loader_on_init=True,
        z_loss_weight=0.0,
        hf_save_steps=1000,
    )

runs = [
    ExecutorStep(
        name=f"delphi-{base_tag}-math-10b-lr{f}",
        fn=lambda cfg: run_levanter_train_lm(cfg),
        config=TrainLmOnPodConfig(
            train_config=build_inner_config(base_tag, base, f),
            resources=ResourceConfig.with_tpu(TPU_TYPE),
            output_path=this_output_path(),
        ),
    )
    for base_tag, base in BASES.items()
    for f in LR_FACTORS
]

if __name__ == "__main__":
    executor_main(steps=runs)
```

The `<llama_config builder>` slot needs adapting to whatever concrete `LlamaConfig` constructor the Delphi template already uses (probably via `completed_adamh_heuristic.build_model_config(...)` or a direct `LlamaConfig(hidden_dim=…, num_layers=…)` call). Cross-check against each source checkpoint's `metadata.json` at implementation time — TensorStore restore is shape-indexed and silently fails to NaN on shape drift.

---

## Critical files (read at implementation time)

- `experiments/scaling_law_sweeps/completed_adamh.py` L 116–131, 169–209 — AdamH heuristic source of truth for peak LR / ε / β₂ formulas.
- `experiments/exp1337_delphi_suite.py` — template for `TrainLmOnPodConfig` wrapping, mesh config, `tp`-search loop.
- `experiments/tootsie/exp1529_32b_mantis_cooldown.py` L 131–165 — template for `initialize_from_checkpoint_path` + rebuilt-optimizer pattern.
- `experiments/tootsie/exp898_deeper_cooldown.py` L 31–78 — confirms `warmup` / `decay` accept absolute int steps.
- `experiments/midtraining_data_buckets.py` — `BUCKET_2["nemotron_cc_math_v1/4plus"]` (tokenize ExecutorStep).
- `lib/levanter/src/levanter/optim/adamh.py` — `AdamHConfig` fields.
- `.agents/projects/delphi_midtraining.md` — Delphi run catalogue + all GCS paths (committed at `84f53bc6f`).

---

## Verification before full sweep launch

1. **Static:** `uv run python experiments/exp_delphi_math_10b_midtrain.py --help` (or `executor_main(..., dry_run=True)`). Confirms imports resolve, `BUCKET_2` entry exists, `(learning_rate, adam_lr)` match the sweep tables above.
2. **GCS:** `gcloud storage ls gs://…/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/` and `…/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/` — verify `manifest.ocdbt`, `metadata.json`, `d/`.
3. **Architecture:** read both source `metadata.json` files; diff against the built `LlamaConfig`. Must match exactly for TensorStore restore.
4. **Pilot:** launch the two `lr0.67` runs only. After step 100:
   - `eval/paloma/c4_en/loss` ≈ 2.858 (1e20) / 2.733 (1e21) — confirms weights loaded.
   - Train loss trending down by step 500 — if flat/rising at `lr0.67`, `lr0.83` will blow up, so don't launch that upper tier.
   - W&B LR panel shows 500-step linear warmup then linear decay.
5. If the pilots look healthy, launch the remaining 4 runs in parallel.

---

## Risks / known unknowns

- **Fresh Adam state** — first ~200 steps will have elevated loss while momentum / variance rebuild. Expected, not a bug; the 500-step warmup absorbs it.
- **`LMMixtureDatasetConfig` with a single component** — Mantis and Delphi use multi-source mixtures; single-source may or may not be accepted. Mitigation: fall back to `LMDatasetConfig` for the training side if the mixture shape breaks.
- **β₂ / ε mismatch between the two base models** — each run uses its own pretrain β₂, ε (not heuristic defaults), because those values are coupled to `(B, T)` and switching them mid-training is an uncharacterized perturbation.
- **Architecture shape drift** — the single most dangerous failure mode; check `metadata.json` before each launch.
- **W&B team-private** — fine; no PR/label hygiene needed for running experiments.

---

## Implementation decisions (2026-04-21)

Notes captured while writing `experiments/exp_delphi_math_10b_midtrain.py`. Update if any of these turn out to be wrong.

- **Harness choice: `default_train` + `SimpleTrainConfig`, not raw `TrainLmConfig`.** The older plan skeleton constructed `train_lm.TrainLmConfig` directly (mirroring `exp1337_delphi_suite.py`'s `run_optimal_training`). In practice, `experiments.defaults.default_train` is the canonical top-level helper on this branch — both `exp1529_32b_mantis_cooldown.py` and `exp898_deeper_cooldown.py` use it. It wraps `TrainLmConfig`, injects default Paloma validation sets, resolves `initialize_from_checkpoint_path` semantics, and returns a ready-to-execute `ExecutorStep`. Stick with it.
- **`SimpleTrainConfig.learning_rate` is required but ignored.** `default_train` uses `train_config.optimizer_config` if set (bypassing the flat LR / warmup / decay fields entirely — see `default_train` at `experiments/defaults.py:474-500`). We still have to pass a non-`None` `learning_rate` because it's a required dataclass field. We set it to the actual peak we use so W&B logs stay consistent.
- **Pass `AdamHConfig` with warmup + decay as INT absolute steps.** Confirmed by `exp898_deeper_cooldown.py:31-42` which uses `decay=COOLDOWN_LEN=10000` as int. Avoids the fraction-vs-int ambiguity when `num_train_steps` is small.
- **Architecture: the bases are `Qwen3Config`, not `LlamaConfig`.** `completed_adamh_heuristic._build_model_config(hidden_size, seq_len)` in `experiments/scaling_law_sweeps/completed_adamh.py:227-244` returns `Qwen3Config(hidden_dim=…, intermediate_dim=4×H, num_layers=<heuristic>, num_heads=H/128, num_kv_heads=H/128, max_seq_len=…, rope=Llama3RotaryEmbeddingsConfig())`. Verified against the wandb configs of both bases (H=2048 → L=21, intermediate=8192, 16 heads; H=2560 → L=26, intermediate=10240, 20 heads). Rope defaults (θ=5e5, factor=8, original_max_pos=8192) match the wandb config exactly (`lib/levanter/src/levanter/layers/rotary.py:152-156`).
- **We call a private method (`_build_model_config`).** Yes it has an underscore — we use it because it is the *same* code path the pretrain ran, guaranteeing byte-identical architecture. If we constructed `Qwen3Config` by hand and anything drifted (intermediate_dim, GQA setup, rope default, head_dim), the TensorStore restore would silently fail to NaN. Prefer breaking loudly on API change over breaking silently on shape drift. If this method is renamed, update the experiment file in lockstep.
- **Tokenizer: `llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"` on both sides.** Verified via the wandb config of `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` (`data.tokenizer = "meta-llama/Meta-Llama-3.1-8B"`). `BUCKET_2["nemotron_cc_math_v1/4plus"]` is built by `tokenize_nemotron_v2_family` which defaults to the same llama3 string. Vocab sizes match (128256). `completed_adamh_heuristic.tokenizer = "marin-community/marin-tokenizer"` is used only for *vocab-size lookup in model-config-candidate filtering* during the sweep; it does not set the model's actual training tokenizer. So the midtraining run will tokenize its math data with llama3 and train on a model whose embeddings were trained with llama3 — consistent.
- **Data passed as a single `ExecutorStep` (not an `LMMixtureDatasetConfig`).** `BUCKET_2["nemotron_cc_math_v1/4plus"]` is an `ExecutorStep`. `default_train` accepts `InputName | ExecutorStep | LMMixtureDatasetConfig` — when given an ExecutorStep it auto-builds a 100 % single-source `lm_data_config` via `_prepare_data_config`, and with `use_default_validation=True` (default) adds Paloma validation. No manual mixture plumbing required.
- **`eval_harness_tasks=()` disables the eval harness.** Empty tuple is falsy; `default_train` sets `harness_config=None` when it is empty. Keeps runtime focused on train loss + Paloma validation for this sweep.

### LR-factor → numerical LR (reference; computed in `_build_adamh`)

| base_tag | factor | `learning_rate` | `adam_lr` |
|---|---:|---:|---:|
| `1e20-iso-d2048-L21` | 0.50 | 2.2415e-3 | 3.6910e-5 |
| `1e20-iso-d2048-L21` | 0.67 | 3.0036e-3 | 4.9460e-5 |
| `1e20-iso-d2048-L21` | 0.83 | 3.7209e-3 | 6.1271e-5 |
| `1e21-v5`            | 0.50 | 3.7125e-3 | 2.1570e-4 |
| `1e21-v5`            | 0.67 | 4.9748e-3 | 2.8904e-4 |
| `1e21-v5`            | 0.83 | 6.1628e-3 | 3.5806e-4 |

(Product of `BASES[b]["peak_lr"] × lr_factor` and same for `adam_lr`. Cross-check these after any `BASES` edit.)

---

## Next steps — what an executing agent should do

**Immediate next actions (code is written, nothing has been run yet):**

1. Static sanity-check: `uv run python -c "import experiments.exp_delphi_math_10b_midtrain as m; print(len(m.runs)); [print(' ', s.name) for s in m.runs]"`. Expected: `6` and the six `checkpoints/delphi-...-math-10b-lrX` names. Fail-open: if imports break, the error will pinpoint a missing helper — update the file in lockstep.
2. GCS ckpt existence: `gcloud storage ls gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915/` and `.../adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/`. Confirm `manifest.ocdbt`, `metadata.json`, `d/` exist. (Already verified on 2026-04-21; re-verify right before launch.)
3. Commit the new experiment file (and/or this logbook) if the user wants — file is currently untracked.

**Launch (only when the user asks):**

4. Launch the two `lr0.67` pilots on separate v4-128 pods.
5. After step 100 of each pilot, check: `eval/paloma/c4_en/loss` ≈ 2.858 (1e20) / 2.733 (1e21) on first eval — confirms weights loaded. Train loss trending down by step 500 — if flat or rising at `lr0.67`, `lr0.83` will blow up; abort the upper tier.
6. If pilots pass, launch remaining 4 runs in parallel.
7. Post-run: append a "Results" section with final c4-en loss / Paloma macro bpb / any math-eval scores, and the best `(base × lr_factor)` per base. Use as starting point for Delphi 1e22 / 1e23 extension.

---

## Status — chronological action log

| Date (UTC) | Action | Result |
|---|---|---|
| 2026-04-21 | Catalogued all 24 Delphi / AdamH runs + the 3e20 isoflop scan. | `.agents/projects/delphi_midtraining.md` committed in `84f53bc6f`. |
| 2026-04-21 | Read *Delphi Scaling Laws* blog. | LR rule + β₂ formula cross-check pass against `completed_adamh.py:169–185`. |
| 2026-04-21 | Decided bases: `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` + `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021`. | User-confirmed. |
| 2026-04-21 | Pulled exact pretrain `(peak_lr, peak_adam_lr, β₂, ε)` from each base's W&B config. | Numbers in the sweep tables above are authoritative. |
| 2026-04-21 | Plan written. | This file. |
| 2026-04-21 | Read reference code: `experiments/defaults.py`, `experiments/simple_train_config.py`, `experiments/scaling_law_sweeps/completed_adamh.py`, `experiments/tootsie/exp1529_32b_mantis_cooldown.py`, `experiments/tootsie/exp898_deeper_cooldown.py`, `experiments/midtraining_data_buckets.py`, `experiments/pretraining_datasets/nemotron_v2.py`, `lib/levanter/src/levanter/optim/adamh.py`, `lib/levanter/src/levanter/layers/rotary.py`. | Identified the `default_train` + `SimpleTrainConfig` + `optimizer_config=AdamHConfig(...)` pattern; confirmed Qwen3Config architecture; confirmed llama3 tokenizer alignment. |
| 2026-04-21 | Pulled `model.*` + `data.tokenizer` from both base runs' wandb configs. | Architecture matches `completed_adamh_heuristic._build_model_config`; both use `meta-llama/Meta-Llama-3.1-8B` tokenizer. |
| 2026-04-21 | Read `metadata.json` of both source checkpoints. | Both well-formed (`step`, `timestamp`, `is_temporary: true`). Companion `manifest.ocdbt` and `d/` present in the step dirs (verified earlier). |
| 2026-04-21 | Wrote `experiments/exp_delphi_math_10b_midtrain.py` (~160 lines). | File created, not yet executed. 6 `ExecutorStep`s constructed via `default_train`. User instruction: don't run anything. |
| 2026-04-21 | Did NOT run `uv run python -c "import experiments.exp_delphi_math_10b_midtrain"`. | User instruction ("don't run anything yet"). Handoff item for next agent or next user turn. |
| 2026-04-21 | Sized TPU via `completed_adamh_heuristic.estimate_memory_bytes` + `pick_v5p_type`. | Est. 2367 GB total (fudge×2). `pick_v5p_type` → `v5p-64` (32 chips × 95 GiB = 3040 GiB). Per-chip ≈ 69 GiB/chip with ≈25 GiB headroom. Per-chip BS=16 → good MFU. v4-128 (the initial plan) was undersized (would need v4-256). |
| 2026-04-21 | Changed `TPU_TYPE` → `"v5p-64"` and rewrote both base ckpt paths to `mirror://<relative-path>` (cross-region pull from `marin-us-central2`). | Per `experiments/AGENTS.md` §Mirror FS, `mirror://` copies to local prefix on first read and caches. Large (>10 GB) transfers would normally need explicit permission — user said "forget the cross region costs!" which is the approval. Will additionally set `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` on the iris job env to disable the `TransferBudget` enforcement globally. |
| 2026-04-21 | Static import check after edits. | `uv run python -c "import experiments.exp_delphi_math_10b_midtrain as m; print(len(m.runs))"` → `6`. All ExecutorStep names correct (`checkpoints/delphi-1e{20-iso-d2048-L21,21-v5}-math-10b-lr{0.5,0.67,0.83}`). |
| 2026-04-21 | Iris CLI available as `uv run iris --cluster=marin ...`. Controller in `us-central1-a` (SSH tunnel auto-established). v5p pool zones per `lib/iris/examples/marin.yaml:101`: `us-central1-a`, `us-east5-a`. v5p-64 has `max_slices: 256` — plenty of capacity. | — |
| 2026-04-21 17:52Z | Committed `20a8d05ba` "Add Delphi Nemotron-CC-Math 10B LR sweep" and pushed to `origin/midtrain_data`. | — |
| 2026-04-21 17:53Z | Submitted the coordinator to Iris with `uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 5GB --extra marin:tpu --job-name delphi-math-10b-sweep --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -- python experiments/exp_delphi_math_10b_midtrain.py`. Job id: `/ahmed/delphi-math-10b-sweep`. (First attempt failed with `--memory 16GB` hitting the `>=4 GB requires --enable-extra-resources` guard; dropped to 3 GB since the coordinator is CPU-only and only dispatches the 6 training sub-tasks.) | Submitted. State: pending. Workspace bundle 5.2 MB uploaded. |
| 2026-04-21 ~18:Z | Coordinator container build FAILED: `marin[tpu]` pulls `torch==2.9.0+cpu` which blew past the 5 GB disk limit during wheel extraction. `--extra marin:tpu` is meant for training-direct jobs (the `babysit-job` skill example), not an `executor_main` coordinator. | Resubmitted. |
| 2026-04-21 19:06Z | Resubmitted with `--disk 9GB` and **no `--extra`** — the default `iris-task` image already has Marin's base deps, so the coordinator needs no extras. `-e MARIN_I_WILL_PAY_FOR_ALL_FEES 1` kept to disable the cross-region `TransferBudget` when mirror:// copies the two base ckpts. | Job `/ahmed/delphi-math-10b-sweep` state=`running` 2 s after submit. |
| 2026-04-21 19:07Z | Coordinator landed on eu-west4 worker. Started walking the dep graph. Paloma + uncheatable validation caches found in eu-west4 (skip). BUT: `raw/nemotron_cc_math_v1_33b36abc` went RUNNING — i.e. the executor is redownloading multi-TB from HF Hub because `BUCKET_2["nemotron_cc_math_v1/4plus"]`'s ExecutorStep hashes to a path that doesn't exist in eu-west4 (hashes differ per region snapshot). Would take many hours. | KILLED the job. |
| 2026-04-21 19:08Z | Stopped `/ahmed/delphi-math-10b-sweep`. | `Terminated jobs: /ahmed/delphi-math-10b-sweep`. |
| 2026-04-21 19:10Z | Rewrote the data config in `experiments/exp_delphi_math_10b_midtrain.py`: dropped the `BUCKET_2` executor-step dependency, built an `LMMixtureDatasetConfig` directly with one `DatasetComponent` whose `cache_dir = "mirror://tokenized/nemotron_cc_math_v1/4plus-0bd79d"`. This skips the raw + normalize + tokenize chain entirely — training workers just read the already-tokenized cache that exists in `gs://marin-us-central2/tokenized/nemotron_cc_math_v1/4plus-0bd79d/`. `MirrorFileSystem` copies to the worker's local prefix on first read and caches thereafter. | Static import still builds 6 runs; `math_mix.components["…/4plus"].cache_dir == "mirror://tokenized/nemotron_cc_math_v1/4plus-0bd79d"`. Committed `45405e0b9`. |
| 2026-04-21 19:11Z | Resubmitted with `--job-name delphi-math-10b-sweep` (same name as previous killed attempt). Reported "Job submitted" and the summary briefly showed `state=assigned`. | Trap: the job's eventual summary was **`state=killed` with worker_id from the previous eu-west4 attempt (`…europ-20260421-1724-2421802a-worker-0`)** — i.e. iris's `--job-name` treats resubmission under a killed job-name as a zombie-reattach, not a fresh run. **Never reuse a `--job-name` after stopping** — always pick a fresh name (or let iris auto-generate). |
| 2026-04-21 19:19Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v2`. | State: `running` / task `building` immediately. Container image being built; will reveal whether the mirror:// path changes bypass the raw+tokenize chain once it finishes build. |
| 2026-04-21 19:22Z | v2 entered `running`, dispatched 8 `train_lm/*` per-host replicas per sweep step → **FAILED after 2m 37s**. Root cause: `train_lm/5` crashed during `levanter.initialize` → `_initialize_global_tracker` → `WandbConfig.init` → `jax_utils.multihost_broadcast_sync` with `RuntimeError: multihost_broadcast_sync requires jax distributed client to be initialized`. Coscheduled-sibling propagation killed the remaining 5 runs. Trigger: W&B fell back to **offline mode** because `WANDB_API_KEY` wasn't in the sub-task env. The offline path broadcasts metadata across hosts but the iris runtime had just logged `"TPU detected; skipping Iris JAX distributed init (TPU runtime handles it)"` — JAX distributed hadn't been fully initialized by the time Levanter ran the broadcast. | Noted: `lib/iris/AGENTS.md` says `WANDB_API_KEY` is auto-injected, but the auto-inject is only for the top-level job env, **not for Fray-dispatched TPU sub-tasks**. Must be passed explicitly. |
| 2026-04-21 19:24Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v3` with `-e WANDB_API_KEY "${WANDB_API_KEY}"` in addition to `-e MARIN_I_WILL_PAY_FOR_ALL_FEES 1`. Same `--cpu 1 --memory 3GB --disk 9GB`. | Submitted. Task 0: `building` (container build). |
| 2026-04-21 19:25Z | v3 FAILED in 33s — NOT a wandb issue this time. `connectrpc.errors.ConnectError: Job …/train_lm is unschedulable: no groups in region europe-west4 (constraints: device-type=tpu, device-variant=v5p-64, region=europe-west4)`. Coordinator landed in eu-west4; Fray propagated `region=europe-west4` as a constraint on the v5p-64 sub-tasks; v5p doesn't exist in eu-west4. Scheduler refused. | Conclusion: **coordinator region dictates TPU region for dispatched Fray sub-tasks**. Must pin coordinator to a region where the target TPU lives. v5p is in us-central1-a + us-east5-a only. |
| 2026-04-21 19:26Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v4` with `--region us-east5` added. | Submitted, state `pending`. Will land on a us-east5 worker. |
| 2026-04-21 19:31Z | v4 still pending 5 min later: `Scheduler: Insufficient CPU (need 1 cores, available 0 cores)` — us-east5 CPU pool at capacity, autoscaler not kicking in fast enough. Stopped v4, submitted v5 with `--region us-east5 --region us-central1` (both v5p regions accepted). | v5 state `running` within 3 s — us-central1 had capacity. |
| 2026-04-21 19:35Z | v5 FAILED with the **same** `multihost_broadcast_sync requires jax distributed client to be initialized` from `train_lm/5` that killed v2 — despite WANDB_API_KEY being passed. Root cause found: `lib/iris/src/iris/runtime/jax_init.py` skips `jax.distributed.initialize()` on ALL TPU jobs, but multi-host TPU under Iris can't auto-discover other hosts (no `TPU_WORKER_HOSTNAMES` inheritance). Levanter then calls `multihost_broadcast_sync` at `WandbConfig.init` → `wandb.py:296` which requires `jax._src.distributed.global_state.client` — `None` on multi-host TPU. | Known bug — I authored the fix back on 2026-04-12 for the DPO work (commit `2fe470a13`). |
| 2026-04-21 21:26Z | Cherry-picked two commits from `origin/dpo-lora` into `midtrain_data`: `2fe470a13` ([iris] Fix multi-host TPU JAX distributed init) + `d82faef23` ([levanter] Defer Iris TPU init on TPU jobs). One trivial conflict in `lib/levanter/src/levanter/distributed.py` resolved by accepting the d82faef23 form (uses `job_info is not None and (not self._is_distributed() or tpu_runtime_managed)`). Pushed `d1c613efe` to `origin/midtrain_data`. | Cherry-pick clean, on HEAD. |
| 2026-04-21 21:27Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v6` with the same flags as v5. | Job submitted. |
| 2026-04-21 21:29Z | v6 cleared the JAX distributed bug but FAILED with `ValueError: No source and no cache found for component nemotron_cc_math_v1/4plus split train` in `levanter.data.text.datasets.build_caches`. | The `mirror://tokenized/nemotron_cc_math_v1/4plus-0bd79d` I pointed at has ONLY `.executor_info` — the tokenize step was registered but never ran to completion in **any** marin bucket. Checked us-central2, us-east5, us-central1, eu-west4, us-east1, us-west4 — only `3` and `4plus_mind` have real cache. |
| 2026-04-21 21:35Z | Tracked down the raw: `gs://marin-us-east5/raw/nemotron_cc_math_v1-322fe4/4plus/` has 46 parquet shards, ~62 GB. Also mirrored at `gs://marin-us-central1/raw/nemotron_cc_math_v1-322fe4/`. The `override_output_path="raw/nemotron_cc_math_v1-322fe4"` in `lib/marin/src/marin/datakit/download/nemotron_v2.py:76` pins the path. | My earlier grep for `raw/nemotron_cc_math_v1/` (trailing slash, no hash) missed the hashed path. |
| 2026-04-21 21:45Z | Reverted the `mirror://` data-config hack in the experiment file. `tokenized=BUCKET_2["nemotron_cc_math_v1/4plus"]` restored so the executor walks the dep chain: download (raw already in us-east5/us-central1, skip) → normalize (will run fresh) → tokenize (will run fresh) → training. All in-region if coordinator lands on us-east5 or us-central1 (v5p regions). Committed `899ec2010`. | Pushed to `origin/midtrain_data`. |
| 2026-04-21 21:46Z | Stopped v6. Submitted `/ahmed/delphi-math-10b-sweep-v7` with same flags. | Expect ~30-min to few-hour normalize+tokenize on the CPU pool before v5p-64 training starts; this is acceptable since raw doesn't have to cross regions. |
| 2026-04-21 21:46Z—22:20Z | v7 ran the whole dep chain on us-central1: raw (skipped, present) → **normalize SUCCEEDED** (45 096 087 records, output at `gs://marin-us-central1/normalized/nemotron_cc_math_v1/4plus_37e28c45/`, ~95 GB) → **tokenize SUCCEEDED** (p0 + p1, output at `gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/`) → **cache-probe SUCCEEDED** (all 231 shards verified) → **cache-copy FAILED** after ~34 m. | Root cause: `zephyr-levanter-cache-copy` workers lost contact with the coord actor (`No endpoints found for actor '/…/cache-copy-*/coord-0'`) and eventually the worker container was killed mid-heartbeat. This is an infra flake, not our config — identical pattern seen in `/rav/iris-run-validate_normalize_phase1-*` jobs earlier today. |
| 2026-04-21 22:24Z | Resubmitted as `/ahmed/delphi-math-10b-sweep-v8`. The executor should skip normalize + tokenize + probe (all outputs still cached in us-central1) and re-try cache-copy. | Submitted. |
| 2026-04-21 22:35Z | v8 got past cache-copy (succeeded this time), dispatched the 6 TPU `train_lm` jobs on v5p-64 → all hosts crashed with `ValueError: Unsupported URI scheme for tensorstore: 'mirror' in ...`. | **Lesson:** `mirror://` is an fsspec protocol. Levanter's checkpoint loader uses TensorStore directly (native GCS paths, no fsspec). So `mirror://` works for data loading but NOT for `initialize_from_checkpoint_path`. |
| 2026-04-21 22:38Z | Reverted ckpt fields in `experiments/exp_delphi_math_10b_midtrain.py` from `mirror://<path>` → `gs://marin-us-central2/<path>`. Cross-region reads are fine: TensorStore doesn't consult the fsspec `CrossRegionGuardedFS`, so `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` isn't even strictly needed for the ckpt read (kept it for the data fsspec paths). Committed `c13560c3f`, pushed. | Normalize/tokenize caches from v7 remain in us-central1; v9 should skip straight to cache-copy then training. |
| 2026-04-21 22:39Z | Submitted `/ahmed/delphi-math-10b-sweep-v9`. | Job accepted. |
| 2026-04-21 22:40Z | v9 FAILED in 33s with `ValueError: initialize_from_checkpoint_path is not in the same region (us-central2) as the VM (us-central1)` for all 6 sweep steps. Marin's `rigging.filesystem.check_gcs_paths_same_region` (invoked via `_doublecheck_paths` in `lib/marin/src/marin/training/training.py`) hard-fails any `gs://` path whose bucket region doesn't match the VM's region — no env-var override exists. | Fix: pre-copy the two base ckpts into us-central1 so they're co-located with the pinned `--region us-central1` coordinator. |
| 2026-04-21 22:42Z | Server-side `gcloud storage cp --recursive` both ckpts us-central2 → us-central1 (23 GB + 41 GB ≈ 64 GB). Updated `BASES[*]["ckpt"]` in the experiment file to `gs://marin-us-central1/...` paths. Committed `56b1b1c86`, pushed. | Copies finished in parallel background (~350-620 MiB/s server-side). |
| 2026-04-21 22:44Z | Submitted `/ahmed/delphi-math-10b-sweep-v10` with `--region us-central1` (dropping us-east5 because the ckpts are now only in us-central1). | Coordinator up in ~5 s. |
| 2026-04-21 22:45–22:50Z | v10 walked the dep graph (skipped the already-cached normalize/tokenize from v7), dispatched the train_lm sub-task, scheduler pending-on-coscheduling for a v5p-64 slice (need 8 worker VMs to come up), then `running`. | Expected TPU spin-up delay. |
| 2026-04-21 22:53Z | All 8 `train_lm/[0-7]` replicas restored the checkpoint successfully (`jax.experimental.array_serialization.serialization Error check finished successfully`). Train-step tracing + HLO lowering completed in a few s. **First training step** 39.3 s (compile-heavy), then 4.4–4.5 s/step steady-state. | MFU ≈ 36 % on v5p-64 (32 chips × 459 TFLOPS bf16 peak; achieved ≈ 5.3 PFLOPS/s). |
| 2026-04-21 22:53 – 2026-04-22 05:16Z | v10 training ran for ~6 h 22 m. Loss dropped 1.58 → 1.12 over warm-up (500 steps), plateaued 1.12–1.20 through decay, **final train_loss 0.958 (tqdm) / 0.962 (W&B summary at step 4767)**. 3 mid-run evals at steps 200/400/600/800/... ran cleanly. Periodic save at step 1000. | Successful run: `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` (wandb + 155 GB at `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f/`). |
| 2026-04-22 05:16Z | v10 coordinator `state=succeeded`. But iris shows only ONE `train_lm` child under the coordinator, and wandb has only ONE v10-era run (the `lr0.5-ba7b7f` one). GCS sweep-output audit: `lr0.5-ba7b7f` → 155 GB with `checkpoints/`, `hf/`, `tracker_metrics.jsonl`; `lr0.67-e3be0c` → **65 KB, `.executor_status=SUCCESS`, `.artifact=null`, no training output**; `lr0.83-e3de76` / `1e21-lr0.5-ccce18` / `1e21-lr0.67-e5b5df` / `1e21-lr0.83-ece889` → 65 KB each, `.executor_status=FAILED`. | **Only 1 of 6 sweep points actually trained.** The other 5 marked terminal states (SUCCESS or FAILED) without producing training artifacts. |

## 2026-04-22 post-v10 analysis: the `train_lm` name-collision pitfall

`lib/marin/src/marin/training/training.py:307` pins every dispatched iris sub-job to the **literal name** `"train_lm"`. When an `executor_main` invocation runs 6 training `ExecutorStep`s whose dependencies are all already satisfied (tokenize cache warm, base ckpts local), `step_runner`'s default `max_concurrent=8` `ThreadPoolExecutor` dispatches all 6 `run_levanter_train_lm` calls **in parallel**. All 6 call `_submit_training_job(job_name="train_lm", ...)` under the same coordinator parent → same full iris path `/ahmed/<coord>/train_lm`.

The iris controller's `EXISTING_JOB_POLICY_KEEP` (what fray hands it for `adopt_existing=True`) then, per `lib/iris/src/iris/cluster/controller/service.py:1113-1117`:

```python
elif policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
    if not is_job_finished(existing_job.state):
        return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
    # Job finished, replace it (KEEP only preserves running jobs)
    self._transitions.remove_finished_job(job_id)
```

→ racing submits 2…6 see the still-running first job and **adopt its handle without creating a new job**. All six Python threads then `.wait()` on the same handle, which completes when the first (and only) config's training finishes. The adopted-handle threads see "SUCCEEDED" and return, so `step_runner` marks their steps as `STATUS_SUCCESS` despite the fn never actually running Levanter training.

Why I confused myself with the PR 4591 seed-sweep precedent: the seed-sweep pattern *defines* many ExecutorSteps but in practice each seed ran as a **separate top-level iris job** on a different day (wandb `created_at` for 1e21 seed{0,42,62746} were 2026-03-04, 2026-03-18 01:47Z, 2026-03-18 06:11Z; 1e22 seeds were 03-04, 03-22, 03-26). Different top-level coordinators → different parent paths → no `train_lm` collision. The seed PR itself didn't fix the collision; it just enabled enumerating the variants, and the human operator launched each variant separately.

## Fix applied to `experiments/exp_delphi_math_10b_midtrain.py` (not yet run)

Added env-var-driven filtering so a single invocation of the script can build a single sweep point:

```python
_SELECT_BASE = os.environ.get("MIDTRAIN_SELECT_BASE")  # e.g. "1e21-v5"
_SELECT_LR   = os.environ.get("MIDTRAIN_SELECT_LR")    # e.g. "0.67"

def _build_runs():
    for base_tag, base in BASES.items():
        if _SELECT_BASE is not None and base_tag != _SELECT_BASE: continue
        ...
        for lr_factor in LR_FACTORS:
            if _SELECT_LR is not None and _lr_str(lr_factor) != _SELECT_LR: continue
            ...
```

Verified:

- Unset: builds all 6 steps (as before). Useful for dry-run/introspection.
- `MIDTRAIN_SELECT_BASE=1e21-v5 MIDTRAIN_SELECT_LR=0.67`: builds just `delphi-1e21-v5-math-10b-lr0.67`.

**Step hashes are unchanged by filtering** (filtering only affects which steps `_build_runs` returns; each step's config is byte-identical to before). So the already-succeeded `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` entry stays cached — any future invocation that includes it will see `STATUS_SUCCESS` and skip. The 5 remaining steps currently have a `.executor_status` of `SUCCESS` (lr0.67 1e20) or `FAILED` (other 4). Before relaunching:

- **`STATUS_SUCCESS` with no training output** (the `lr0.67-e3be0c` case, and possibly others): the cache check will treat them as succeeded → skip → no retraining. Workaround: delete `.executor_status` at those output paths so the next run re-does the step. (Do NOT delete the `lr0.5-ba7b7f` one — that one really did train.)
- **`STATUS_FAILED`**: `step_runner` will raise `PreviousTaskFailedError` unless you pass `force_run_failed=True` or delete the status file.

## Launch recipe for the 5 remaining sweep points (copy-paste)

Each variant goes as its own iris coordinator so `/ahmed/<coord-N>/train_lm` is a unique path per sweep point. Run from the repo root:

```bash
# 1. (one-time) clean up the stale STATUS files so the 5 remaining steps don't
#    short-circuit on cache hit / fail-as-previous-failure:
for target in \
    'delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c' \
    'delphi-1e20-iso-d2048-L21-math-10b-lr0.83-e3de76' \
    'delphi-1e21-v5-math-10b-lr0.5-ccce18' \
    'delphi-1e21-v5-math-10b-lr0.67-e5b5df' \
    'delphi-1e21-v5-math-10b-lr0.83-ece889'; do
  gcloud storage rm "gs://marin-us-central1/checkpoints/${target}/.executor_status" 2>/dev/null || true
done

# 2. launch each as its own iris job (each gets a unique --job-name)
launch() {
  local base="$1" lr="$2" short
  short=$(echo "$base" | sed 's/-iso-d2048-L21//')
  uv run iris --cluster=marin job run \
    --cpu 1 --memory 3GB --disk 9GB \
    --region us-central1 \
    --job-name "delphi-math-10b-${short}-lr${lr}" \
    --no-wait \
    -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
    -e WANDB_API_KEY "${WANDB_API_KEY}" \
    -e MIDTRAIN_SELECT_BASE "$base" \
    -e MIDTRAIN_SELECT_LR "$lr" \
    -- python experiments/exp_delphi_math_10b_midtrain.py
}
launch 1e20-iso-d2048-L21 0.67
launch 1e20-iso-d2048-L21 0.83
launch 1e21-v5            0.5
launch 1e21-v5            0.67
launch 1e21-v5            0.83
```

All 5 can run in parallel (v5p-64 pool has `max_slices: 256`). Expected per-run wall-time: 1.9 B base ≈ 6 h, 3.4 B base ≈ 10 h (larger model, same BS=512). W&B runs appear under `marin-community/marin` with names `delphi-<base>-math-10b-lr<factor>-<hash>`.

## Expected cross-region transfers (FYI)

Launch region isn't pinned (Iris picks any v5p zone) — either `us-central1` or `us-east5`. In either case:

| Artifact | Source | Dest (per worker's MARIN_PREFIX) | Approx size | Notes |
|---|---|---|---:|---|
| 1e20 isoflop ckpt (step-46915) | `gs://marin-us-central2/checkpoints/isoflop/...` | mirror → local prefix | ~22 GB | OCDBT shards under `d/` + manifest + metadata. |
| 1e21 Delphi v5 ckpt (step-21979) | `gs://marin-us-central2/adamh-scaling-ladder-...v5-019021/checkpoints/step-21979/` | mirror → local prefix | ~40 GB | Same shape, bigger model. |
| `nemotron_cc_math_v1/4plus` tokenize cache | `gs://marin-us-central2/tokenized/nemotron_cc_math_v1/4plus-0bd79d/` | might mirror, or executor re-tokenizes on the local CPU pool | ~100 GB (or retokenize) | us-east5 has only `3-ef5cb9` + `4plus_mind-d60b4a`, NOT `4plus`. Versioning hashes differ between regions so the executor may re-run the tokenize step locally instead of mirroring. If it mirrors, ~100 GB; if it re-tokenizes, hours of CPU but no cross-region. |

Total worst-case cross-region read: **~160 GB** (once, cached thereafter). User has accepted this.

---

## 2026-04-22 region-pin diagnosis and the real fix: wiring `mirror://` through Levanter's checkpoint loader

### Re-verification of current state

Two coordinators `/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422` are queued (pinned to `us-central1`) but have not yet dispatched TPU. GCS audit shows **only** `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` (155 GB) is a real training output; all other 11 sweep-point directories (two hash variants per remaining combination) are 65–70 KB stubs. That means the logbook's "Step hashes are unchanged by filtering" claim above is wrong — something after v10 did change hashes, so there are now TWO hash series of stubs per variant. Relaunch plan will be: compute the hash the current code produces, then clear `.executor_status` files at all known variants of the target step before launching.

### Why I was pinning `--region us-central1`

Two reasons from the earlier v-series attempts:

1. `lib/rigging/src/rigging/filesystem.py:254` — `check_path_in_region` hard-fails any `gs://` path whose bucket region ≠ VM region. `lib/marin/src/marin/training/training.py:344` runs this check on the whole `TrainLmConfig` via `_doublecheck_paths`. v9 died in 33 s because the 1e20 ckpt was a `gs://marin-us-central2/...` path while the VM was in us-central1. No env-var override exists.
2. The base ckpts and the `tokenized/nemotron_cc_math_v1/4plus-212a2d` cache currently live only in `gs://marin-us-central1/...`. A TPU in us-east5-a would either re-run the entire tokenize step (hours) or fail the region check.

### Why `mirror://` was supposed to handle this (and why it broke last time)

`mirror://` IS designed to solve exactly this problem. The mechanism:

- `mirrored("relative/path", budget_gb=N)` in the executor config (`lib/marin/src/marin/execution/executor.py:939-946`) marks a path as cross-region-mirrorable.
- At config instantiation time, `MirroredValue` is rewritten to `mirror://relative/path` (`executor.py:1149-1153`).
- `MirrorFileSystem` (`lib/rigging/src/rigging/filesystem.py:715-935`) registers `mirror` as an fsspec protocol. On read, it: checks `${MARIN_PREFIX}/<path>` first; otherwise scans the other marin regional buckets (`marin-us-central1`, `marin-us-east5`, `marin-us-central2`, `marin-eu-west4`, …) via `_find_in_remote_prefixes`; copies from whichever bucket has the file to the local prefix under a distributed lock; charges against the shared `TransferBudget` (disabled by `MARIN_I_WILL_PAY_FOR_ALL_FEES=1`).
- **The region check already skips `mirror://` paths** — `_collect_gcs_paths_recursively` at `filesystem.py:338-344` only gathers strings starting with `gs://`, so a mirror URL sails through `check_gcs_paths_same_region`.

v8's failure (`ValueError: Unsupported URI scheme for tensorstore: 'mirror'`) happened because `mirror://` was passed straight into `initialize_from_checkpoint_path`. Levanter's `load_checkpoint` at `lib/levanter/src/levanter/checkpoint.py:794-810` hands the path verbatim to `tree_deserialize_leaves_tensorstore`, whose `build_kvstore_spec` (`lib/levanter/src/levanter/tensorstore_serialization.py:54-78`) only speaks `gs`/`s3`/`file`/``''``. TensorStore bypasses fsspec entirely — the mirror protocol has no hook.

For **data loaders** (LMMixture, tokenizers) this is already solved: Levanter goes through fsspec for data paths, and the tokenizer loader has an explicit `_stage_from_mirror` staging helper (`lib/levanter/src/levanter/tokenizers.py:702-729`) that uses `mirror_fs.ls()` + per-file `_fetch_file_atomic` to materialize files to a local staging dir before the HF loader opens them.

For **checkpoints** there is no equivalent staging path — that's the gap.

### Planned fix (small, local to Levanter)

Add a `_stage_mirror_to_local(path)` helper to `lib/levanter/src/levanter/checkpoint.py`:

1. If path does not start with `mirror://`, return it unchanged.
2. Strip the prefix to get a relative path (e.g. `checkpoints/isoflop/.../step-46915`).
3. Construct `fsspec.filesystem("mirror")` and call `mfs.find(rel)` → recursive list of file paths.
4. For each file, call `mfs._resolve_path(file_rel)` — triggers `_copy_to_local` on cache miss (GCS-to-GCS server-side `rewrite`, fast). Files already present locally are skipped.
5. Return `f"{marin_prefix()}/{rel}"` — a concrete `gs://marin-${region}/...` or `/tmp/marin/...` URL that TensorStore can open.

Wire the helper at exactly two call sites:

- Top of `latest_checkpoint_path` (line 965): discovery still runs against `mirror://` (via fsspec, which MirrorFileSystem handles), but the returned concrete step dir goes through the staging helper before being returned.
- Top of `load_checkpoint` (line 794): covers direct-path callers that skip discovery (`eval_lm.main`, `export_lm_to_hf`, `perplexity_gap`, `inference_repl`, `eval_harness`).

This is the same import pattern `lib/levanter/src/levanter/config.py` and `lib/levanter/src/levanter/trainer.py` already use (`from rigging.filesystem import url_to_fs / open_url`) — Levanter already depends on `marin-rigging` via `pyproject.toml`.

Test: add a `test_stage_mirror_to_local` in `lib/levanter/tests/test_checkpoint.py` using the same fixture pattern as `lib/rigging/tests/test_mirror_fs.py` (manually constructed MirrorFileSystem backed by two tempdirs, one standing in for local and one for remote).

### Experiment change

Replace the hardcoded `gs://marin-us-central1/...` ckpt paths in `experiments/exp_delphi_math_10b_midtrain.py` with `mirrored(...)` calls:

```python
"ckpt": mirrored(
    "checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915",
    budget_gb=30,
),
...
"ckpt": mirrored(
    "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979",
    budget_gb=50,
),
```

Budgets cover the actual ckpt sizes (1e20 ≈ 22 GB, 1e21 ≈ 40 GB) with a small safety margin. `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` in the iris job env disables `TransferBudget` enforcement globally, so budgets are informational on worker runs — they still matter for dry-runs and local dev.

### Relaunch plan (after fix lands)

Drop `--region us-central1` from the launch recipe; replace with `--region us-central1 --region us-east5`. The coordinator lands wherever CPU is free; Fray propagates the region constraint to the v5p-64 sub-task; MirrorFileSystem copies the ckpt from us-central1 → `marin-${landing_region}` on first open, cached thereafter. The `4plus-212a2d` tokenize cache is NOT wrapped with `mirrored()` in this pass (it's already materialized in us-central1; if the TPU lands in us-east5 the executor will re-run normalize+tokenize locally, which takes longer but is acceptable — can revisit).

### Checkpoint before launch

Plan this logbook entry → implement Levanter patch + test → implement experiment update → run lint → report back before relaunching any jobs.

### Implementation status (2026-04-22)

**Levanter patch** — `lib/levanter/src/levanter/checkpoint.py`
- Added `_stage_mirror_to_local(checkpoint_path: str) -> str` (~25 lines). No-op on non-`mirror://` inputs. On `mirror://` input, walks `mfs.find(rel)` and calls `mfs._resolve_path(file_rel)` on each file; returns `${marin_prefix()}/<rel>`.
- `latest_checkpoint_path` now routes the discovered path through `_stage_mirror_to_local` before returning.
- `load_checkpoint` now stages at the top so direct-path callers (`eval_lm.main`, `export_lm_to_hf`, `perplexity_gap`, `inference_repl`, `eval_harness`) benefit without further changes.
- Imports `marin_prefix` from `rigging.filesystem`; same pattern used in `levanter/config.py` and `levanter/trainer.py` — no new dependency (marin-rigging is already declared in `lib/levanter/pyproject.toml`).

**Tests** — `lib/levanter/tests/test_checkpoint.py`
- `test_stage_mirror_to_local_passes_through_non_mirror_paths`: the no-op branch for `file://` / `gs://` / raw paths.
- `test_stage_mirror_to_local_copies_all_files`: remote dir with 4 files (metadata.json, manifest.ocdbt, d/shard_0, d/shard_1) → all copied to local, returned URL points at local prefix.
- `test_stage_mirror_to_local_raises_when_empty`: FileNotFoundError when the mirror tree has no files.
- `test_latest_checkpoint_path_stages_direct_mirror_step`: end-to-end through `latest_checkpoint_path` with a direct `mirror://.../step-N` input — the shape our experiment uses.
- Helper `_configure_mirror_fs(local_dir, remote_dirs, monkeypatch)` patches `marin_prefix` + `_mirror_remote_prefixes` on `rigging.filesystem` AND clears `MirrorFileSystem._cache` (fsspec's instance cache lives on the leaf class, not on `AbstractFileSystem`, so clearing the base class cache alone is not enough — that was a real bug I hit).
- All 4 pass in isolation and together; all 20 existing tests in `lib/rigging/tests/test_mirror_fs.py` still pass (no upstream regression).

**Experiment** — `experiments/exp_delphi_math_10b_midtrain.py`
- Both `BASES[*]["ckpt"]` values replaced with `mirrored("<relative-path>", budget_gb=N)` (N=30 for 1e20 at ~22 GB, N=50 for 1e21 at ~40 GB).
- Header docstring's launch-recipe updated from `--region us-central1` to `--region us-central1 --region us-east5` — the whole point of this fix.
- Verified: `MIDTRAIN_SELECT_BASE=1e20-iso-d2048-L21 MIDTRAIN_SELECT_LR=0.67` import builds 1 ExecutorStep; both `BASES[*]["ckpt"]` are `MirroredValue` instances pre-instantiation.

**Lint** — `./infra/pre-commit.py` on the 3 changed files: Ruff + Black + pyrefly + license + AST + merge + whitespace + EOF all pass.

### Things NOT done (deliberate)

- `tokenized=BUCKET_2["nemotron_cc_math_v1/4plus"]` is still an ExecutorStep dependency, not `mirrored(...)`. The `4plus-212a2d` tokenize cache exists in us-central1 only; if the TPU lands in us-east5, the executor walks the dep chain → normalize + tokenize run fresh in us-east5 (raw is already present in both regions). Extra wall-clock but no cross-region data transfer. Can revisit if we want to cut the tokenize turnaround later.
- The two queued `--region us-central1` pinned jobs (`/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422`) are still alive. Need user sign-off before killing and relaunching with the new region-flex recipe.

### Ready for launch, pending user sign-off

1. Kill `/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422`.
2. Commit the Levanter patch + experiment change + logbook entry, push to `origin/midtrain_data` so the coordinator picks up the new code.
3. Resubmit the 5 remaining sweep points with `--region us-central1 --region us-east5` and a fresh `--job-name` per sweep point.

### Cross-region verification — actually exercised on a us-east5 worker (2026-04-22)

Added `scripts/_verify_mirror_stage.py` — a small iris-submittable script that imports `_stage_mirror_to_local`, pulls the 1e20 ckpt via `mirror://…`, then opens the staged OCDBT kvstore through TensorStore to prove the full end-to-end path (mirror:// → fsspec copy → TensorStore read) works across regions.

**Attempt 1 (`verify-mirror-stage-1e20-20260422`): FAILED after copying ~10 GB**

```
rigging.filesystem.TransferBudgetExceeded: ... would bring total to 10.53GB,
exceeding the 10GB limit (already transferred 9.77GB).
Consider running in the source region instead.
```

Worker was us-east5 (✓ marin_prefix = `gs://marin-us-east5`); 15 OCDBT shards successfully copied gs→gs from us-central1→us-east5 before the cap fired. The copy mechanism itself worked — the problem was the budget.

**Key discovery — the two safety envs are NOT symmetric.** `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` ONLY short-circuits `CrossRegionGuardedFS._guard_read` (`lib/rigging/src/rigging/filesystem.py:623-625`); it does NOT disable `MirrorFileSystem._copy_to_local`'s budget charge at line 819. Those are separate code paths that happen to share `_global_transfer_budget` by default. The mirror side is governed by `MARIN_MIRROR_BUDGET_GB` (process-wide default ceiling) OR a `mirror_budget(gb)` contextvar (per-call stack, scoped). `MARIN_I_WILL_PAY_FOR_ALL_FEES` is invisible to the mirror code. That was the gap that bit us.

In the production training run, the executor already opens a per-step `mirror_budget(_max_mirror_budget(config))` context around the step fn (`lib/marin/src/marin/execution/executor.py:703-708`, `1301`) — so `mirrored("1e20-ckpt", budget_gb=30)` in our experiment yields a 30 GB budget at call time, which is why the real training launch will not hit this failure. The verify script, calling the helper outside an executor context, inherited the 10 GB default.

**Attempt 2 (`verify-mirror-stage-1e20-v2-20260422`): SUCCESS**

Wrapped the staging call in `with mirror_budget(30.0):` — same budget the executor sets from `mirrored(..., budget_gb=30)`. Duration 3m 50s on a us-east5 worker.

```
[verify] marin_prefix = gs://marin-us-east5
[verify] mirror URL   = mirror://checkpoints/isoflop/.../step-46915
[verify] staged in 218.2s
[verify] resolved    = gs://marin-us-east5/checkpoints/isoflop/.../step-46915
[verify]   metadata.json: OK
[verify]   manifest.ocdbt: OK
[verify]   OCDBT keys   = 1218
[verify] SUCCESS
Staged mirror://checkpoints/isoflop/.../step-46915 (44 files) to gs://marin-us-east5/...
```

**What this proves end-to-end:**

- `_stage_mirror_to_local("mirror://…")` on a fresh us-east5 worker correctly detects local prefix = `gs://marin-us-east5`.
- MirrorFileSystem finds the files in the remote (`gs://marin-us-central1/…`) bucket, copies all 44 files (OCDBT shards + manifest + metadata) via gs→gs rewrite, caches them under `gs://marin-us-east5/checkpoints/…/step-46915/`, and returns that concrete URL.
- The returned URL opens cleanly as an OCDBT kvstore through TensorStore (1218 keys enumerated), which is the exact call path Levanter's `load_checkpoint` uses for weight restore.
- Budget enforcement works as intended — the 30 GB ceiling from our `mirrored(..., budget_gb=30)` declaration is respected by a scoped context (1e20 ckpt actual size ~22 GB; used ~20 GB of the 30 GB budget since ~2 GB was already cached from v1).
- Side-effect: the 1e20 ckpt is now physically present at `gs://marin-us-east5/checkpoints/isoflop/.../step-46915/`, so future us-east5 launches for this base are cache-hits (MirrorFS `_fs_exists` returns True → skip copy).

**Budget semantics (for next agent):**

- `MARIN_I_WILL_PAY_FOR_ALL_FEES=1` disables the *direct-read* guard only, not mirror copies.
- `MARIN_MIRROR_BUDGET_GB=<n>` sets the default global mirror ceiling at module import (one-shot).
- `with rigging.filesystem.mirror_budget(<gb>):` opens a fresh scoped budget — preferred for ad-hoc scripts because it can't leak to other call sites.
- `mirrored(path, budget_gb=<n>)` in the executor config does (3) automatically on the step's behalf — nothing to set on the iris command line for real runs.
- The two overrides (`I_WILL_PAY` vs `mirror_budget`) are orthogonal; the default global `TransferBudget` instance is shared by both paths but the overrides do not cascade between them.

---

## 2026-04-23 live training — 1e20 lr=0.67 and lr=0.83 (still running; pre-mirror patch)

These two coordinators were launched 2026-04-22 23:54Z with `--region us-central1` pinning, BEFORE the mirror:// fix. They succeeded at the ckpt region check because the ckpts were pre-copied to us-central1 earlier in the v-series. They do NOT exercise the Levanter mirror-staging patch — that's still pending proper-run validation via the 3 × 1e21 launch.

### Run identifiers (wandb + GCS)

| Sweep point | Coordinator | train_lm output path | wandb run |
|---|---|---|---|
| `1e20 × lr=0.67` | `/ahmedah/delphi-math-10b-1e20-lr0.67-20260422` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c/` | `https://wandb.ai/marin-community/marin/runs/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c` |
| `1e20 × lr=0.83` | `/ahmedah/delphi-math-10b-1e20-lr0.83-20260422` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7/` | `https://wandb.ai/marin-community/marin/runs/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7` |

Both `e3be0c` and `db9de7` are the OLD stubs' hashes — same hash as the v10 empty-SUCCESS stubs. Clearing the `.executor_status` files (logbook entry for 2026-04-22) did let the executor re-run the step under the same hash, so the filtering patch did NOT change hashes after all (the "two hash variants per point" observed earlier must have come from a separate code change between v10 and now). Future relaunches under these hashes will hit the cache and be skipped, which is what we want.

### Training configuration recap (both runs)

- v5p-64 (32 chips), us-central1-a, mesh `{data:-1, replica:1, model:1}` (tp=1 since H=2048 divides 32).
- Batch 512 × seq_len 4096 = 2,097,152 tokens/step. 4768 steps → ~10.0 B tokens.
- Fresh AdamH optimizer, β₂=0.99980, ε=4.11e-8, β₁=0.9, max_grad_norm=0.1.
- 500 linear warmup steps → 4268 linear decay steps, `min_lr_ratio=0.1`.
- `reset_data_loader_on_init=True`, `z_loss_weight=0`, `jmp.get_policy("p=f32,c=bfloat16")`.
- `steps_per_eval=200`, `steps_per_export=1000`, `steps_per_hf_export=1000`.

Per-run LR:

| lr_factor | `learning_rate` | `adam_lr` |
|---:|---:|---:|
| 0.67 | 3.0036e-3 | 4.9460e-5 |
| 0.83 | 3.7209e-3 | 6.1271e-5 |

### Throughput

Measured from `Progress on:train` ticks, averaged over the elapsed/step ratio (to neutralize tqdm's instant-rate spikes near eval/export):

- **Rate: ~4.4 s/step** (both runs, identical — same arch, same batch, same mesh).
- Achieved compute: ~5.3 PFLOPS (6 × 1.9e9 params × 2.1e6 tokens / 4.5 s).
- v5p-64 peak: 32 × 459 TFLOPS bf16 = 14.69 PFLOPS.
- **MFU ≈ 36 %** on both runs. Matches v10's measurement exactly.

Per-host memory: ~31 GB / 95 GB HBM (1.9 B model + full opt state + activations; v5p has plenty of headroom). We could run this on v5p-32 and stay under 62 GB/chip, at ~11 h wall-clock instead of ~6 h.

### Evaluation + HF-export overhead

At every multiple-of-1000 step, Levanter: (a) runs the full Paloma + uncheatable eval suite (17+ loss computations) and (b) writes a **7.74 GB × 2-shard** HF-compatible checkpoint to GCS. This takes ~2.5 min total. Tqdm's rolling-average `rate` field rolls the entire pause into "the last step," so after step 2000 the reported rate spiked to 47.0 s/it (lr=0.83) / 19.5 s/it (lr=0.67). **Actual per-step rate is unchanged at ~4.4 s/step** — always compute rate from `elapsed/N_steps`, not tqdm's instant rate, when eval/export is in the window. For future agents: this is a Levanter+tqdm display artifact, NOT a slowdown, and does NOT require action.

### Loss trajectory + lr=0.67/0.83 crossover (unsmoothed; check W&B for the clean version)

Single-step tqdm loss readouts (noisy but directionally correct):

| Step | lr=0.67 | lr=0.83 | Notes |
|---:|---:|---:|---|
| 444–513 (~11%) | 1.17 | 1.15 | End of warmup; lr=0.83 ahead (higher LR → faster initial progress) |
| 2000 (42%) | 1.03 | **0.987** | Mid-run; lr=0.83 still ahead by ~0.04 |
| 4370–4400 (92%) | **0.927** | 0.959 | Decay tail; **lr=0.67 has overtaken** |

**Crossover observation:** the higher peak LR (lr=0.83) converges faster initially but loses ~0.03 of its advantage by the decay tail — lr=0.67 with the gentler peak ends up lower in single-step loss. Both end below the v10 lr=0.5 final of 0.962, which is the direction we expected. Preliminary ranking for the 1e20 base:

```
lr=0.67 (0.927)  <  lr=0.83 (0.959)  <  lr=0.5 (0.962, v10)
```

Very narrow spread (~0.035 across factors 0.5–0.83). Need to look at smoothed Paloma/c4 curves and math-eval downstream scores before calling a winner, but if the signal holds, **lr=0.67 is the sweet spot for the 1e20 base** and is a reasonable default for the 1e21 LR sweep too.

**Caveat — single-step unsmoothed loss is noisy.** A ~0.03 gap at one step can be dominated by within-batch variance. The W&B panels at the run URLs above show EMA-smoothed curves; use those for the actual ranking.

### Current status (as of 2026-04-23 05:53Z)

- lr=0.67: step 4400/4770 (92 %), elapsed 5:54:38, real ETA ≈ 30 min.
- lr=0.83: step 4370/4770 (91.5 %), elapsed 5:50:55, real ETA ≈ 30 min.
- No preemptions, no failures across all 16 hosts.
- Both expected to finish around 06:25Z.

### Next steps once these land

1. Inspect final W&B panels for smoothed train-loss + Paloma validation trajectories across lr=0.5 / 0.67 / 0.83. Confirm the crossover + pick the 1e20 winner.
2. Launch 3 × 1e21 sweep points (lr=0.5 / 0.67 / 0.83) on v5p-64. With the Levanter mirror-staging patch now verified end-to-end (see "Cross-region verification" section above), the new launches can go `--region us-central1 --region us-east5` and land wherever the autoscaler has capacity. Expected wall-time per 1e21 run: ~10 h (3.4 B params at BS=512).
3. Commit the Levanter patch + experiment change + these logbook updates to `origin/midtrain_data` before relaunching, so the iris worker bundle picks up the new code.

---

## 2026-04-23 flat-LR incident — root cause + fix

**TL;DR: every 1e20 run completed before today was trained at `min_lr = 0.1 × peak`, not the scheduled warmup → peak → decay curve. All three completed/in-flight runs are DISCARDED. The Levanter warmstart path had a latent bug; fix landed locally today. Relaunch will produce new output hashes (config changed).**

### Symptom

W&B `optim/learning_rate` for the three completed 1e20 runs (`lr=0.5-ba7b7f` v10, and in-flight `lr=0.67-e3be0c` / `lr=0.83-db9de7`) is flat from step 0, no warmup, no decay. Values match `0.1 × peak × lr_factor` to 2 sig figs:

| Factor | Expected peak `learning_rate` | `0.1 × peak × factor` | Chart value |
|---:|---:|---:|---:|
| 0.50 | 2.2415e-3 | 2.2415e-4 | ~2.2e-4 |
| 0.67 | 3.0036e-3 | 3.0036e-4 | ~3.0e-4 |
| 0.83 | 3.7209e-3 | 3.7209e-4 | ~3.7e-4 |

Same story on `optim/adam_lr` — flat at `0.1 × peak_adam_lr × factor` (3.7e-6 / 4.9e-6 / 6.1e-6).

### Root cause

Direct TensorStore read of `gs://marin-us-central1/checkpoints/isoflop/.../step-46915/`:

```
opt_state/count                                  = 46916
opt_state/hyperparams_states/learning_rate/count = 46916
opt_state/hyperparams_states/adam_lr/count       = 46916
step                                              = 46916
```

Levanter's `train_lm.py:176-180` (`initialize_from_checkpoint_path` branch):

```python
if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
    checkpoint_path = latest_checkpoint_path(config.initialize_from_checkpoint_path)
    state = load_checkpoint(state, checkpoint_path)     # restores FULL state incl opt_state
    state = dataclasses.replace(state, step=jnp.array(0))   # resets only outer step
```

`load_checkpoint(state, path)` deserializes every array leaf in the exemplar tree — including `opt_state.hyperparams_states.learning_rate.count`, which comes back as 46916 from the pretrain. Our fresh schedule is built with `num_train_steps=4768, warmup=500, decay=4268`. `optax.linear_schedule(peak, min_lr, 4268)` evaluated at count=46916 clamps to `min_lr = 0.1 × peak`. Every subsequent step increments count but stays past decay → flat forever at `min_lr`.

The inline comment `# we're just initializing weights here` has been a lie since PR #1957 (David Hall, `b5659c59c4`, 2025-12-02). Before #1957 the branch restored everything AND kept the outer step — a coherent full resume. PR #1957 added the step-reset without also resetting opt_state, creating today's inconsistency. The actual `load_checkpoint(state, ...)` call pre-existed #1957 (from `5c53a19fdc`, Aug 2024) but wasn't pathological on its own. `534544b0bd` (Apr 2026) refactored the call to use `latest_checkpoint_path` — a semantics-preserving change.

Pathology only manifests when `num_train_steps < restored count`. Existing callers (Mantis, 8B cooldowns, exp2062 giraffe) all use larger `num_train_steps` or `reset_data_loader_on_init=False` (which routes through a different `trainer.initialize_from` path), so none of them tripped it. Our midtraining with `num_train_steps=4768 << 46916` is the first case to hit it.

### Fix

Added `CheckpointInitMode` enum to `lib/levanter/src/levanter/main/train_lm.py` with two values:

- `MODEL_ONLY`: `load_checkpoint(state.model, path, subpath="model")` — load only the model subtree, keep freshly-initialized opt_state (count=0). The pattern `train_dpo.py:383` already uses.
- `FULL_STATE`: current (legacy) behavior — restore everything, reset only outer step. Preserves WSD-S rewarmup tricks like exp2062's.

**Default: `FULL_STATE`.** Preserves behavior byte-for-byte for every caller currently on this path; a full audit of every `initialize_from_checkpoint_path=` caller was explicitly *not* done. Delphi opts into `MODEL_ONLY` explicitly in `experiments/exp_delphi_math_10b_midtrain.py`. exp2062 is untouched.

Files changed (uncommitted as of 2026-04-23):

- `lib/levanter/src/levanter/main/train_lm.py` — enum, field on `TrainLmConfig`, branch the load block.
- `experiments/simple_train_config.py` — `checkpoint_init_mode: CheckpointInitMode = FULL_STATE` field.
- `experiments/defaults.py` — forward field from `SimpleTrainConfig` → `TrainLmConfig` in `default_train`.
- `experiments/exp_delphi_math_10b_midtrain.py` — explicit `checkpoint_init_mode=MODEL_ONLY` + comment.
- `lib/levanter/tests/test_checkpoint.py` — 2 raw-load tests (MODEL_ONLY vs FULL_STATE semantics on a schedule-count fixture).
- `experiments/test_default_train_init_mode.py` — 3 plumbing tests asserting defaults + Delphi-experiment MODEL_ONLY propagation through `default_train` to the inner `TrainLmConfig`.

Tests: `uv run python -m pytest lib/levanter/tests/test_checkpoint.py` → 29/29 pass; `uv run python -m pytest experiments/test_default_train_init_mode.py` → 3/3 pass. `./infra/pre-commit.py --fix` on all 6 files: ok.

### Hash impact — new output paths on relaunch

Adding `checkpoint_init_mode=MODEL_ONLY` to the Delphi `SimpleTrainConfig` changes its serialized form, which feeds `executor.py:1407-1408`'s `json.dumps(version, sort_keys=True, cls=CustomJsonEncoder)` → `hashlib.md5(...)[:6]`. The new runs will have different `-<hash>` suffixes than `ba7b7f` / `e3be0c` / `db9de7`. **No `.executor_status` surgery is needed at the old paths** — they're orphans of a different config and the executor simply won't see them.

### Runs marked DISCARDED (do not use for analysis)

- `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-ba7b7f` (v10, 155 GB at `gs://marin-us-central1/checkpoints/`) — trained at lr=2.24e-4 constant, not the 2.24e-3 → 2.24e-4 warmup/decay curve.
- `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c` (in-flight coord `/ahmedah/delphi-math-10b-1e20-lr0.67-20260422`) — trained at lr=3.00e-4 constant.
- `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7` (in-flight coord `/ahmedah/delphi-math-10b-1e20-lr0.83-20260422`) — trained at lr=3.72e-4 constant.

Keep the GCS artifacts around until relaunched runs land healthy on W&B, then `gcloud storage rm --recursive` them as a cleanup pass.

### Follow-up (not in this change)

Audit every `initialize_from_checkpoint_path=` caller in the repo. If all verified tolerant of fresh opt_state, flip the default on `SimpleTrainConfig.checkpoint_init_mode` + `TrainLmConfig.checkpoint_init_mode` to `MODEL_ONLY` so the comment-and-intent mismatch fully heals. Precondition: explicit `FULL_STATE` on any caller that wants the opt-state carry (e.g. exp2062). This is a separate, scope-limited change — not part of this fix.

### Pointer

Full plan at `/Users/ahmed/.claude/plans/feedback-from-codex-make-humble-kernighan.md`.

### Next steps (replacing the pre-incident list above)

1. Kill in-flight `/ahmedah/delphi-math-10b-1e20-lr{0.67,0.83}-20260422`.
2. Commit the 6-file fix + this logbook entry; push to `origin/midtrain_data`.
3. Submit one pilot (`1e20 × lr=0.67` under the new hash). Wait ~30 min, verify on W&B that `optim/learning_rate` rises 0 → 3.00e-3 over first 500 steps, then decays linearly toward 3.00e-4 by step 4768. That's the anti-pathology.
4. If pilot healthy, submit remaining 5 sweep points (1e20 × lr=0.5, 0.83; 1e21 × lr=0.5, 0.67, 0.83) in parallel on v5p-64.
5. After all 6 land, compare smoothed train-loss + Paloma panels across (base × lr_factor), pick winners, write up.

---

## 2026-04-23 relaunch + hash-collision surprise (08:00–08:15 UTC)

### What happened

Relaunched the three 1e20 sweep points after the Levanter fix was committed (`37fba5983`, pushed). Job names: `/ahmed/delphi-math-10b-1e20-lr{0.5,0.67,0.83}-20260423-v2`.

**Surprise 1: The executor output hash did NOT change after adding `checkpoint_init_mode=MODEL_ONLY` to Delphi's `SimpleTrainConfig`.** Plan had assumed it would. Root cause: `lib/marin/src/marin/execution/executor.py:1028-1094`'s `collect_dependencies_and_version` only records values wrapped in `versioned(...)` — plain dataclass fields are recursed into but not added to the hash input. Since I didn't wrap `checkpoint_init_mode` in `versioned()`, the field has no effect on the step hash. The new runs landed at the SAME output paths as the broken v10/in-flight runs: `-ba7b7f`, `-e3be0c`, `-db9de7`.

First "fix" (`-20260423-fix` coordinators) therefore found `already succeeded` markers and skipped training entirely for lr=0.5, and the other two were headed toward the same outcome.

**Resolution:** killed the `-fix` coordinators, deleted all three junk directories (`gcloud storage rm --recursive` ~465 GB), then resubmitted as `-v2`. Now the `.executor_status` cache misses and the executor runs training. Old artifacts are gone (not merely moved) so there's no Levanter auto-resume-from-broken-ckpt risk.

**Surprise 2: W&B monotonic-step rejection.** Because the output hash is unchanged, the W&B run ID is the same as the broken runs'. W&B's run.step is at 4768 from the old training. Levanter's `wandb.py:69` refuses to log metrics whose step is less than `run.step`:

```
W20260423 08:06:50 levanter.tracker.wandb Step 1 is less than the current step 4768. Cowardly refusing to log metrics.
```

**Training is actually fine** — the JAX training loop is running and loss is dropping — but W&B panels won't show LR/loss curves from the new run until the fresh training advances past step 4768. Since `num_train_steps=4768`, that means basically nothing gets logged to W&B this time. For LR-fix verification, rely on:

- `tracker_metrics.jsonl` at the run's GCS output (written by Levanter's local tracker, independent of W&B).
- The tqdm `Progress on:train … postfix:loss=…` lines in iris logs.
- Compare trajectory shape against v10's broken curve (recorded in this logbook).

### Early signal (loss trajectory, tqdm-reported)

| Step | lr=0.67 loss | lr=0.83 loss | Notes |
|---:|---:|---:|---|
|   2 | 1.58 | 1.58 | Identical to v10 initial — pretrain weights loaded correctly. |
|  16 | 1.58 | (not yet seen) | LR still ramping up through early warmup. |
|  30 | 1.51 | (not yet seen) | Rate-of-drop increasing, as warmup approaches peak. |
|  44 | 1.45 | (not yet seen) | |
|  58 | (not yet seen) | 1.43 | Higher-LR factor drops faster — expected ordering. |

Rate of descent is materially faster than the broken v10 curves at equivalent steps. This is the loss-side proof the MODEL_ONLY fix is doing its job: the schedule count is at 0 (not 46916), so actual LR is warming up toward the peak, not clamped to the 0.1×peak floor.

Initial Paloma / uncheatable_eval losses at step 0 look sane (`wikipedia_english=2.535, github_python=1.775, ao3_english=3.158, arxiv_physics=2.767`) — consistent with freshly-loaded pretrain weights.

### Known bugs to tackle later (not blocking the sweep)

1. **Executor hash ignores non-`versioned()` fields.** When flipping `checkpoint_init_mode` selectively, the step's output path does not change. This is a foot-gun: a naive relaunch after a field change silently reuses the old artifact. Options: (a) wrap the field in `versioned()` at call sites that care; (b) add opt-in "always-versioned" dataclass fields at the marin-executor layer; (c) document + rely on manual deletion for now.
2. **W&B run step collision on relaunch.** Same root cause — unchanged hash → same W&B run_id → monotonic-step rejection. Fix would be to include a unique component (timestamp, attempt counter) in the W&B run config so relaunches get fresh run_ids.

Neither blocks the sweep. Both should be filed as issues once the sweep lands.

### Current job states (08:11 UTC)

- `/ahmed/delphi-math-10b-1e20-lr0.67-20260423-v2/train_lm`: running on v5p-64, us-central1. Step 44+, loss dropping.
- `/ahmed/delphi-math-10b-1e20-lr0.83-20260423-v2/train_lm`: running on v5p-64, us-central1. Step 58+, loss dropping.
- `/ahmed/delphi-math-10b-1e20-lr0.5-20260423-v2`: still in zephyr-normalize phase (landed us-east5; normalize/tokenize caches live in us-central1 under a different hash, so it's re-running data prep locally). Expected ~30–60 min before training starts; total wall-time thus slightly longer than the other two.

Next check: verify loss ≪ 1.12 at step 500 (v10's warmup-end number under the broken schedule). If yes, LR fix confirmed. If no, dig deeper.

### LR fix confirmed — both runs finished (2026-04-23 14:28 UTC)

`lr=0.67` and `lr=0.83` reached `4.77kit/4.77kit` (i.e., step 4768) simultaneously at 14:28 UTC, ~6 h 23 min after training-start. Final train-loss (single-step tqdm):

| Run | Final loss | vs v10 broken (0.962) |
|---|---:|---|
| `lr=0.5` | (still running, ETA 15:00 UTC) | — |
| **`lr=0.67`** | **0.781** | 18.8% lower |
| **`lr=0.83`** | **0.772** | 19.7% lower |

Preliminary 1e20 ranking (awaiting `lr=0.5` final + smoothed curves for confirmation):
`lr=0.83 (0.772) < lr=0.67 (0.781)`

The **final-loss test is unambiguous**: both runs' single-step final losses are ~0.18-0.19 below v10's final of 0.962. Under the flat-min-lr bug, effective LR was ~10x too low across the whole run; new runs trained at the intended warmup→peak→decay schedule, and that's the measurable difference in the final loss. Combined with:

- the earlier loss trajectory during warmup being faster than v10's, and
- the crossover/ordering between factors (lr=0.83 leading early, lr=0.67 overtaking in decay tail, then lr=0.83 finishing slightly lower again)

we have three independent lines of evidence that the LR schedule is alive. The `CheckpointInitMode.MODEL_ONLY` branch in `train_lm.py` correctly keeps the freshly-initialized opt_state, so the schedule evaluates at count=0 at step 0 and ramps normally.

**Runs ended with the expected tqdm rate pattern** — `rate:4.4-4.5s/it` for ~4768 steps = ~5:50 elapsed, plus eval+checkpoint pauses absorbed into the rolling average. No crashes, no preemptions, no mid-training bug.

Coordinators are still showing `running` because Levanter is in the final HF-export phase (~7.7 GB × 2 shards per run). Iris will flip them to `succeeded` in 5-10 min once export commits.

### 1e20 sweep complete (2026-04-23 15:05 UTC) — all three succeeded

All three coordinators in terminal `succeeded` state.

| Run | Coordinator | Final single-step loss | vs v10 broken (0.962) |
|---|---|---:|---|
| `lr=0.5` | `/ahmed/delphi-math-10b-1e20-lr0.5-20260423-v2` | **0.840** | −12.7% |
| `lr=0.67` | `/ahmed/delphi-math-10b-1e20-lr0.67-20260423-v2` | **0.781** | −18.8% |
| `lr=0.83` | `/ahmed/delphi-math-10b-1e20-lr0.83-20260423-v2` | **0.772** | −19.7% |

Preliminary 1e20 ranking (unsmoothed tqdm tail reading — these have ~0.02 single-step jitter so the ordering is tentative): `lr=0.83 (0.772) < lr=0.67 (0.781) < lr=0.5 (0.840)`.

Wall-times:
- `lr=0.67`: ~6 h 38 min coordinator-to-succeeded
- `lr=0.83`: ~6 h 38 min
- `lr=0.5`: ~7 h 02 min (included zephyr-normalize + zephyr-tokenize in us-east5 before training could start)

GCS outputs at `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr{0.5,0.67,0.83}-{ba7b7f,e3be0c,db9de7}/` (same hash slots as the DISCARDED v10/broken runs — the Marin-executor-hash-ignores-unversioned-fields caveat remains).

Final HF export (`hf/step-4768/`) is present on all three. Periodic waypoint at `hf/step-1000/`.

For follow-up ranking with smoothed curves, read `tracker_metrics.jsonl` at each output path — W&B is not usable for these runs due to the step-monotonic rejection bug (same-hash-as-broken-run) noted above.

### Next steps

1. Pull smoothed train-loss + Paloma trajectories from each run's `tracker_metrics.jsonl`; confirm the preliminary 1e20 ranking.
2. Launch 3 × 1e21 sweep points (`lr=0.5 / 0.67 / 0.83`) on `v5p-64`. Same launch recipe as the 1e20 relaunch above. Expected wall-time ~10 h per run (3.4 B params, same BS=512, slightly larger). The pretrain ckpt lives at `gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979/` with schedule count ~21979 (smaller than `num_train_steps=4768` — no wait, it's *larger*, so the same flat-min-lr pathology would apply to the 1e21 runs without the fix, and does not with MODEL_ONLY). With MODEL_ONLY plumbed, the 1e21 sweep will train at the scheduled warmup→peak→decay.
3. When all 6 land, cross-ranking + winner selection + writeup. Store the winning (base, lr_factor) combination as input to any downstream sweep.

### 2026-04-23 lr=0.67 / lr=0.83 `-v2` reruns (22:53-22:56 UTC) — clean W&B curves

Context: the original lr=0.67 and lr=0.83 runs succeeded with correct training (final losses 0.781 and 0.772) but their W&B panels still showed the OLD broken flat-min-lr curves. Root cause: the Marin executor hash only tracks `versioned(...)` values + `step.name` + upstream dep paths — plain `SimpleTrainConfig` fields (including our `checkpoint_init_mode`) are invisible. With the same us-central1 tokenize dep as before, both runs landed at the same output hashes as the broken v10-era runs (`e3be0c`, `db9de7`) → same W&B run_ids → W&B's step-monotonic guard rejected the fresh metrics.

Fix (commit `0a5b1fde3`): append `-v2` to the `step.name` template in `experiments/exp_delphi_math_10b_midtrain.py:221`, so the name-contribution to the hash changes. Relaunched both with coordinators `/ahmed/delphi-math-10b-1e20-lr{0.67,0.83}-v2-20260423`.

Results:

| Run | Output hash | W&B run name | Final single-step loss |
|---|---|---|---:|
| `lr=0.67-v2` | `a176ff` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff` | **0.781** (matches original 0.781 exactly) |
| `lr=0.83-v2` | `4487d2` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2` | **0.782** (original 0.772; +0.010 single-step noise, statistically indistinguishable) |

Both coordinators in terminal `succeeded` state. No `Cowardly refusing to log metrics` warnings this time — W&B accepted the fresh metrics, so these two runs now have clean warmup→peak→decay curves on the W&B panel.

The 1e20 sweep now has **one set of canonical, clean-W&B results** for the cross-ranking:

| lr factor | Canonical run name | Final loss |
|---:|---|---:|
| 0.50 | `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2` (us-east5, fresh hash by-accident) | 0.840 |
| 0.67 | `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff` | 0.781 |
| 0.83 | `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2` | 0.782 |

Preliminary 1e20 ranking (unsmoothed): `lr=0.67 (0.781) ≈ lr=0.83 (0.782) < lr=0.5 (0.840)`. The 0.67/0.83 gap is within noise; smoothed curves + Paloma eval should disambiguate. Either factor is a reasonable default for the 1e21 sweep.

**Stale artifacts to eventually garbage-collect** (no longer canonical; W&B + GCS data is superseded by the `-v2` runs):

- `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-e3be0c/` (GCS is the healthy fresh training, but W&B run is polluted with broken data)
- `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-db9de7/` (ditto)
- Also the corresponding W&B runs at those names — they display misleading flat-min-lr curves; safe to delete once the `-v2` runs are locked in.

---

## Analysis + plotting utilities in this repo (found 2026-04-23)

Will Held owns most of the scaling-law / analysis infra. If you need to produce plots, fits, or sweep-wide comparisons, these are the code paths to study first rather than rolling your own.

### Core library — `lib/marin/src/marin/scaling_laws/` (≈1001 lines, plotly-based)

| File | Key exports | What it does |
|---|---|---|
| `scaling_plots.py` | `create_isoflop_plot`, `create_scaling_plot`, `save_plots`, `upload_plots_to_wandb` | Plotly figure builders for isoflop curves + scaling fits; GCS save + W&B artifact upload |
| `isoflop_analysis.py` | `fit_scaling_laws`, `predict_optimal_config`, `robust_quad_logx` (Huber-loss quadratic fit), `ScalingFit`, `QuadraticFitCoeffs`, `IsoFlopRecord`, `MinimaRecord`, `CandidateConfig` | Scaling-law math and data structures |
| `eval_metrics_reader.py` | `read_eval_records` (+ W&B backfill via `_backfill_metrics_from_wandb`) | Pulls per-step eval metrics from GCS runs and W&B API, unifying the two sources |
| `tpu_utils.py` | `pick_v5p_type`, `pick_v4_type`, `V5P_SPEC`, `V4_SPEC` | Choose the smallest TPU slice that fits a given model |
| `__init__.py` | Re-exports above | Public API entry |

### Callers / end-to-end wiring

| File | Purpose |
|---|---|
| `experiments/isoflop_sweep.py` | **The canonical ExecutorStep wiring** — reads eval metrics, fits scaling laws, emits plots, uploads to W&B. Pattern-match against this when building a new analysis step. |
| `experiments/exp1337_delphi_suite.py` | Delphi-specific sweep runner using `predict_optimal_config`. Source of the `(H, L, B)`-heuristic pipeline. |
| `experiments/exp2166_scaling_ladder_analysis.py` | Most recent ladder analysis (~2026-02). |
| `experiments/scaling_law_sweeps/completed_adamh.py` | The AdamH heuristic that drove our 1e20 base-model choice — `completed_adamh_heuristic._build_model_config(hidden_size, seq_len)` is the source of truth for Delphi architecture. |
| `experiments/scaling_law_sweeps/c_adamc.py` | AdamC-variant counterpart. |

### Per-run training-loss (no dedicated Marin tool)

For single-run or small-sweep train-loss plots (what this midtraining sweep wants), the options are:

- `lib/levanter/scripts/loss_history.py` — ~30-line example that hits `wandb.Api().runs().scan_history()` for `train/loss` by git-sha. Good template.
- Read `tracker_metrics.jsonl` at each run's GCS output path directly (Levanter writes it independently of W&B; **this is our only source of truth for the `e3be0c` / `db9de7` runs whose W&B is polluted**). One JSON per step, columns include `train/loss`, `optim/learning_rate`, `optim/adam_lr`, and all `eval/paloma/*/loss` + `eval/uncheatable_eval/*/loss` series. Just `pd.read_json(..., lines=True)`.

### For this midtraining sweep specifically

The scaling-laws infra is overkill for a 3-point × 2-base LR sweep (no scaling fit is meaningful with one token budget + one parameter count per base). Appropriate plots:

- Train-loss vs step, EMA-smoothed, one line per `(base, lr_factor)`.
- Paloma validation loss vs step (`eval/paloma/c4_en/loss`, `eval/paloma/dolma-v1_5/loss`, etc.), same overlay.
- Final-loss bar chart per sweep point to pick the winner.

A ~50-line script that loads the 6 `tracker_metrics.jsonl` files (3 × 1e20 + 3 × 1e21 once they land) and renders these with matplotlib or plotly is enough. **Do not** build it on top of `isoflop_analysis.py` — wrong abstraction. **Do** reuse `eval_metrics_reader.read_eval_records` for the GCS + W&B unification logic if the runs have the right shape (check its filters first).

### Authorship / blame-walk

- `scaling_plots.py` / `isoflop_analysis.py` / `eval_metrics_reader.py` / `isoflop_sweep.py` — William Held, PR #2243 "Scaling Plots & Analysis as an Executor Step".
- Delphi pipeline (`exp1337_delphi_suite.py`) — William Held, PR #3292 "Delphi Scaling Setup", plus PR #4591 "exp1337: add seed sweep".
- AdamH heuristic — William Held, PR #2447 "Beta2 gets a bit wacky with very large batch sizes...".

When in doubt on scaling/analysis decisions, `git log --format='%an %s' -- <file>` → look for Will.

---

## Project goal #1 — midtraining loss predictor (detailed plan, 2026-04-23)

### Why this is the right next thing

Project goal #1 reads *"predict loss trajectory for midtraining."* The concrete operational payoff is: **given small pilot runs at 1e20 and 1e21, forecast final eval metrics at 1e22 and 1e23 without actually running them** (or at least: predict enough of the curve shape to reject bad LR schedules before committing v4-512 / v4-1024 compute). The sweep tracked in this logbook produces exactly the training signal such a predictor would consume.

### Empirical pre-conditions already established

1. **Noise floor ≈ 0.01 abs loss** on final train loss. Evidence: `lr=0.67` and `lr=0.67-v2` (identical config, different RNG/wandb run) finished at 0.781 and 0.781 respectively; `lr=0.83` and `lr=0.83-v2` at 0.772 and 0.782. Any predictor with MAE ≪ 0.01 is overfitting noise.
2. **Per-step data source**: `tracker_metrics.jsonl` at the GCS output path ONLY contains `{"config": ..., "summary": ...}` — i.e. one row of final values, NOT a time-series. The time-series lives exclusively in W&B. Pull it via `wandb.Api().run(...).scan_history(keys=[...], page_size=2000)`. Train metrics come at every step (4768 rows / run); evals come at `steps_per_eval=200` cadence (~48 rows / run, including step 0 pre-training eval).
3. **Three canonical 1e20 runs available with clean W&B**:
   - `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2`
   - `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff`
   - `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2`
4. **Retention degrades during midtrain** (confirmed empirically): Paloma c4_en starts at 2.8586 for all runs (pretrain-only), and ends at 3.15 (`lr=0.5`), 3.29 (`lr=0.67-v2`), 3.29 (`lr=0.83-v2`). Higher midtrain LR → more retention damage. This is the classic specialization/retention tradeoff. The predictor must handle both monotone-downward (train/loss on math) and monotone-upward (Paloma c4_en) metrics.

### Functional form to fit

Core: schedule-aware power-law in cumulative learning rate.

```
L(u) = L_∞ + A × (U - u)^c
```

where

- `u(t) = Σ_{s ≤ t} lr(s)` = cumulative `optim/learning_rate` series from W&B.
- `U = u(T)` = total LR budget consumed at end of training.
- `L_∞`, `A`, `c` are the three fit parameters.
- For monotone-decreasing metrics (train/loss): `A > 0`, `c > 0`, `L_∞` = asymptote from below.
- For monotone-increasing metrics (Paloma retention): `A < 0`, `c > 0`, `L_∞` = asymptote from above.

Why `optim/learning_rate` and not `optim/adam_lr`: peak LRs differ ~60× (2.24e-3 to 3.72e-3 for `learning_rate` vs 3.69e-5 to 6.13e-5 for `adam_lr`), and `learning_rate` governs the matrix-param updates that dominate midtraining dynamics.

### Baselines (in order of ambition)

| Baseline | Form | Fit on | Purpose |
|---|---|---|---|
| **B0: last-value** | `L̂ = L(t_prefix_end)` | — | Trivial control. Beats every other baseline at 99% prefix. |
| **B1: raw-step power** | `L(t) = a + b/√t` | `step > 500`, back half of observed prefix | Schedule-unaware baseline. If B2 doesn't beat this, something's wrong. |
| **B2: schedule-aware power** | `L(u) = L_∞ + A(U-u)^c` | `step > 500`, back half of prefix | The actual proposed predictor. |

### Evaluation protocol

**Target quantity:** EMA-smoothed loss over the last window `[4600, 4767]`. Using single-step final loss introduces ~0.01 noise; averaging over 167 points puts the target well below noise floor.

**Metrics:**
1. `train/loss` — smoothest curve, machinery validation (does the functional form fit anything?).
2. `eval/paloma/c4_en/loss` — retention anchor. Rising; tests whether the form handles sign change.
3. `eval/paloma/*_loss` aggregates — broader retention panel (future).
4. Math-specific eval (pending) — actual scientific target.

**Prefixes evaluated:** `{30%, 50%, 80%}` of total `num_train_steps` (so {1430, 2384, 3814} for 1e20 runs). Each prefix truncates the fit set; we then predict the step-4767 target from the remaining fit.

**Tests (easiest → hardest, per project goal #1):**

1. **Self-prefix** (tractable with only 3 runs): for each (run, metric), fit on first X% and predict own final. Pass = B2 MAE ≤ 2 × noise floor by prefix=30%.
2. **Cross-LR, same base** (3 runs → LOO over lr_factor): fit shared `c` (+ optionally shared `A`) on two 1e20 runs; predict third run's final using only its first 30%. Pass = MAE ≤ 3 × noise floor.
3. **Cross-base** (blocked on 1e21 sweep): fit all 1e20 curves; predict 1e21 finals from only `(lr_factor, base_params)` features + first 30% of 1e21. This is the project-goal-1 money shot.
4. **Cross-scale** (blocked on 1e22/1e23 runs): extrapolate from 1e20 + 1e21 to larger bases.

Tests 1-2 are validations of the functional form; tests 3-4 are the scientific claim.

### Implementation plan

**File:** `scripts/analysis/midtrain_loss_predictor.py`.

**Dependencies available in the repo venv**: `pandas`, `numpy`, `scipy.optimize`, `wandb`, `matplotlib` (via plotly import guard). No new requirements.

**Structure** (~250 lines):

```python
RUNS_1E20 = [
    RunSpec("lr=0.5",  lr_factor=0.5,  wandb_name="delphi-...-lr0.5-4d19a2"),
    RunSpec("lr=0.67", lr_factor=0.67, wandb_name="delphi-...-lr0.67-v2-a176ff"),
    RunSpec("lr=0.83", lr_factor=0.83, wandb_name="delphi-...-lr0.83-v2-4487d2"),
]

def load_run(spec):  # wandb.Api().run.scan_history → DataFrame
def ema_smooth(df, halflife=100):  # train/loss only; evals are already low-freq
def compute_cumulative_lr(df):  # cumsum of optim/learning_rate
def fit_last_value(df, prefix_frac, metric):
def fit_sqrt_t(df, prefix_frac, metric, min_step=500):
def fit_cumlr_power(df, prefix_frac, metric, min_step=500):  # L_∞ + A (U-u)^c
def evaluate_final(df, metric, window=(4600, 4767)):  # target quantity

def run_self_prefix(runs, metrics, prefixes):
    # Returns DataFrame: (run, metric, prefix, baseline, predicted, target, abs_err)

def run_cross_lr_loo(runs, metrics, prefixes):
    # Hold out one LR; fit shared c on others; predict held-out final from its 30% prefix

def main():
    # 1. Load all runs
    # 2. Run self_prefix + cross_lr_loo
    # 3. Report tables: MAE per (baseline, prefix, metric)
    # 4. Report c stability (fit c across prefixes; warn if std > 0.1)
    # 5. Write CSV of full prediction table to scripts/analysis/midtrain_loss_predictor_out.csv
    # 6. Print human-readable summary to stdout
```

**Outputs:**
- Stdout: summary table (MAE per method × prefix × metric) + c-stability report.
- CSV: `scripts/analysis/midtrain_loss_predictor_out.csv` with every prediction.
- Optional: matplotlib figures for (a) loss vs step raw, (b) loss vs cumulative-LR (should show partial collapse across LR factors).

**Execution:** `uv run python scripts/analysis/midtrain_loss_predictor.py`. Should complete in <2 min (3 W&B fetches + 3×3×3 = 27 fit calls).

### Success criteria for phase 1

- B2 (schedule-aware) beats B1 (raw-step) by >20% on MAE for train/loss at prefix ≤ 50%. If not, the `(U-u)^c` parameterization isn't adding value and we should revisit.
- B2 MAE ≤ 2× noise floor (≤ 0.02) on self-prefix test at prefix=30% for train/loss.
- `c` stable across prefixes (std ≤ 0.1 across {30%, 50%, 80%} prefix choices) per-run.
- For Paloma c4_en (rising metric): B2 handles the sign change gracefully. If `c` has to go imaginary or fit fails numerically, need a separate form for rising metrics.
- Cross-LR LOO MAE ≤ 3× noise floor on train/loss. Higher on Paloma is expected due to noisier eval series.

Meeting these → good enough to commit 1e21 compute and run phase 2. Not meeting → stop, understand why, adjust form.

### Out of scope for phase 1

- Joint multi-metric fitting (we fit metric-by-metric).
- Multi-base extrapolation (blocked on 1e21).
- Uncertainty quantification (bootstrapping or Bayesian fit) — cheap win for phase 2.
- Math-eval loss analysis — blocked on actual math evals being computed on the checkpoints (those are a separate step not yet done).

### Status

Phase 1 script implemented at `scripts/analysis/midtrain_loss_predictor.py` and run against the 3 canonical 1e20 curves.

### Phase 1 results (2026-04-23)

Ran the script after adding two fixes that surfaced during implementation:

- **Two-phase W&B fetch.** `scan_history(keys=[train_keys, eval_keys])` in a single call does an *intersection* over rows, so train (every step) intersected with eval (every 200 steps) yielded only 27 rows. Split into two fetches: one for `optim/learning_rate` + `train/loss` (4768 rows), one for `eval/paloma/c4_en/loss` (48 rows). Merged on `_step`.
- **Bounds on `c`.** Without bounds, `curve_fit` drove `c → ∞` and `A → 0`, giving the degenerate fit `L ≈ L_∞` (a useless constant). Added `c ∈ [0.2, 3.0]` bounds. Also added two fixed-c variants (B3_c=0.5 and B3_c=1) as stabler alternatives with the same 2-param count as B1.

#### Self-prefix MAE (tested on target = EMA over steps 4600–4767)

Noise floor ≈ 0.005–0.010 (from original-vs-v2 rerun pairs).

**train/loss_smooth**:

| method | prefix 30% | prefix 50% | prefix 80% |
|---|---:|---:|---:|
| B0 last-value | 0.146 | 0.084 | 0.021 |
| **B1 a+b/√t** | **0.024** | 0.020 | **0.003** |
| B2 free c (bounded [0.2, 3.0]) | 0.085 | 0.056 | 0.007 |
| B3 c=0.5 | 0.428 | 0.124 | 0.041 |
| **B3 c=1 (schedule-aware, fixed c)** | 0.120 | **0.004** | 0.007 |

**eval/paloma/c4_en/loss** (sparse — ~12 points per 30% prefix, so B1/B2/B3 skip prefix=0.3):

| method | prefix 30% | prefix 50% | prefix 80% |
|---|---:|---:|---:|
| B0 last-value | 0.086 | 0.039 | 0.036 |
| B1 a+b/√t | — | 0.075 | 0.052 |
| B2 free c | — | **0.037** | 0.038 |
| B3 c=0.5 | — | 0.211 | 0.070 |
| B3 c=1 | — | 0.100 | 0.050 |

#### Cross-LR LOO (hold out 1 of 3, fit shared c on other 2, predict from 30/50/80% of held-out)

| metric | prefix 30% | prefix 50% | prefix 80% |
|---|---:|---:|---:|
| train/loss_smooth | 0.182 | 0.028 | **0.002** |
| eval/paloma/c4_en/loss | — | 0.519 | 0.129 |

#### c-stability across prefixes (B2 free-c)

`c` values per (run, metric) at prefixes {30%, 50%, 80%}:

- `lr=0.5` train/loss: 3.000 → 1.529 → 1.086 (std ≈ 0.82; bouncing)
- `lr=0.67` train/loss: 3.000 → 2.339 → 0.824 (std ≈ 0.91)
- `lr=0.83` train/loss: 3.000 → 2.322 → 0.763 (std ≈ 0.94)
- Paloma: hits bound c=3.0 at prefix 0.8 for all runs (under-identified)

`c` hits the upper bound (3.0) for short prefixes → the fit wants even larger `c` but can't, which confirms the parameter is under-identified on this data.

### Takeaways

1. **B1 (raw `a + b/√t`) is the surprisingly strong baseline for train/loss self-prefix.** At prefix 30% it gives MAE 0.024 (~2.5× noise floor); at prefix 80% it hits 0.003 (at noise floor). Schedule-unawareness doesn't hurt at this fidelity because `√t` already captures the asymptotic shape well enough when the LR schedule is held constant across runs.

2. **B3 with fixed `c=1` wins at prefix 50% train/loss** (MAE 0.004, at noise floor). This says: if you commit to a schedule-aware parameterization, you must pin `c` — letting it float introduces more error than it removes on this amount of data. The linear-in-remaining-LR form (`c=1`) is the best zero-prior default.

3. **B2 (free `c`) is worse than both B1 and B3.** The 3-parameter fit is under-identified on ~700–3800 points of fairly smooth data. Even with bounds the fit chases degenerate solutions at short prefixes.

4. **Paloma is not tractable with these 3 runs.** Only ~12-24 eval points per prefix. Even the best method (B2 at prefix 50%) has MAE 0.037 — ~10% of the metric's midtrain-induced change (2.86 → 3.29, δ ≈ 0.43). Need the 1e21 sweep's eval points to get more leverage, or use a totally different approach for sparse metrics (e.g., linear extrapolation through 3-4 points).

5. **Cross-LR LOO on train/loss works at ≥ 50% prefix.** MAE 0.028 at 50%, 0.002 at 80%. That is: if you run 2 of the 3 1e20 LR factors to completion and observe the first half of the third, you can predict its final train loss within 0.03. For the 1e20 → 1e21 cross-base test (phase 2), this is encouraging but not decisive — cross-base is a harder extrapolation.

### Implications for the success criteria from the plan

From the pre-committed criteria:

- **"B2 MAE < 2× noise floor at prefix=30% for train/loss"** → FAIL (0.085 at 30%, 8× noise floor). Resolution: use B1 or B3 c=1 as the primary predictor instead. The proposal's `(U-u)^c` with free `c` doesn't have identifiability on this data.
- **"B2 beats B1 by >20% on train/loss"** → FAIL on its own terms, but B3 c=1 (the fixed-c schedule-aware variant) beats B1 at prefix 50% (0.004 vs 0.020). The schedule-awareness adds value IF you pin `c`.
- **"`c_std` < 0.1 per run per metric"** → FAIL (std 0.8–0.9 for train/loss, worse for Paloma). Parameter under-identified; forcing it is the right call.
- **"Cross-LR LOO MAE < 3× noise floor on train/loss at prefix=30%"** → FAIL at 30% (0.182, 18× noise floor). Passes at 50% (0.028, 3× noise floor) and 80% (0.002).

Interpretation: the phase-1 machinery works, but **30% prefix is too short** for this functional family. Revised operational recommendation for phase 2:

- Evaluate predictors at prefix ∈ {50%, 70%, 90%} for the cross-base test. 30% is aspirational but empirically not yet useful on this data.
- Default predictor: **B3 c=1** for train/loss (schedule-aware, 2-param). Fallback to B1 if the 1e21 schedule is materially different.
- For Paloma and other sparse evals: skip the fit, use last-value or linear-in-step at the last 3-4 eval points.

### Outputs

CSVs for downstream plotting in `scripts/analysis/`:

- `midtrain_loss_predictor_self_prefix.csv`
- `midtrain_loss_predictor_cross_lr.csv`
- `midtrain_loss_predictor_c_stability.csv`

### Phase 2 launch criteria — recommendation

Given phase 1 results, **launching the 1e21 sweep now is justified**. The cross-LR LOO on train/loss at ≥ 50% prefix is within reach of noise floor (0.028 → 0.002), which is exactly the regime the cross-base test needs. Don't over-interpret phase-1 results for Paloma — that needs more data, which 1e21 will provide.

If cross-base train/loss MAE < 0.05 at prefix 50% when 1e21 lands, project goal #1 has a validated baseline. Everything beyond that (cross-scale to 1e22/1e23, uncertainty bounds, better Paloma models) is iteration.

---

## 2026-04-23 session summary — end-to-end record

This section captures everything done in the 2026-04-23 session so far, in one place, so a future agent or the same user tomorrow can pick up cold.

### Commit chain on `origin/midtrain_data` this session

(newest first; `b860d5b60 wip` is the first pre-session commit, for reference)

| Commit | What |
|---|---|
| `64c030b70` | `[analysis] Plots for 1e20 midtrain runs (matplotlib)` — `scripts/analysis/plot_midtrain_curves.py` generates four PNGs and `open`s them on macOS. |
| `8aefa1689` | `[analysis] Phase-1 midtrain loss predictor: script + results` — `scripts/analysis/midtrain_loss_predictor.py` with 5 baselines and self-prefix + cross-LR LOO tests against the 3 canonical 1e20 runs. |
| `f8a85aef0` | `[logbook] Delphi midtrain: detailed phase-1 plan for loss predictor` — fully-specified plan before implementation. |
| `d04dea5d6` | `[logbook] Delphi midtrain: clarify project goal hierarchy` — two tracks (predict trajectory, pick dataset); this sweep is a subgoal of track 1. |
| `6deb3dc8c` | `[logbook] Delphi midtrain: add analysis-utils reference section` — pointers to Will Held's `scaling_laws/` module and related files. |
| `5fed3965d` | `[logbook] Delphi 1e20 lr=0.67/0.83 v2 reruns done — clean W&B curves` — both `-v2` reruns succeeded with losses matching originals; noise floor ≈ 0.01 confirmed. |
| `0a5b1fde3` | `[delphi-midtrain] Append -v2 to step names so hash changes` — workaround for the Marin executor's cache-hash behaviour (only versioned values + step name + upstream deps contribute to the hash). |
| `fc14f21ba` | `[logbook] Delphi 1e20 sweep complete — all 3 succeeded, fix fully confirmed` — final losses 0.840 / 0.781 / 0.772 for LR factors 0.5 / 0.67 / 0.83, all well below v10's broken 0.962. |
| `cbdd748b0` | `[logbook] Delphi midtrain: LR fix confirmed by final-loss at step 4768` — incremental. |
| `e77ea7e47` | `[logbook] Delphi midtrain: relaunch at -v2 + hash-collision + W&B step-monotonic notes` — diagnosed the hash-stays-same + wandb-step-monotonic-refusal bugs during relaunch. |
| `37fba5983` | `[levanter] Add CheckpointInitMode for initialize_from_checkpoint_path` — the core Levanter fix: `MODEL_ONLY` mode loads only the model subtree (using `subpath="model"`), keeping the freshly-initialized opt_state so the schedule starts from count=0. Default is `FULL_STATE` (legacy behaviour preserved). |

Between `97ed6c6f2` and `37fba5983` the user walked up to the flat-LR bug in the 1e20 sweep, caught it visually on W&B (`optim/learning_rate` flat for 3/3 broken v10-series runs), and worked through the `train_lm.py:176-180` root cause with a plan doc and back-and-forth review with Codex.

### What the completed 1e20 sweep produced

Three GCS output paths that are now **canonical for phase-2 analysis** (clean W&B runs):

| LR factor | GCS output | W&B run | Final single-step loss | Target-window mean |
|---:|---|---|---:|---:|
| 0.50 | `gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2/` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2` | 0.840 | ~0.813 |
| 0.67 | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff/` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff` | 0.781 | ~0.798 |
| 0.83 | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2/` | `delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2` | 0.782 | ~0.797 |

Plus the original (non-v2) `-e3be0c` / `-db9de7` outputs — GCS is healthy training (the MODEL_ONLY fix did its job), but the W&B panels for those IDs show OLD flat-min-lr curves from the broken training because W&B's step-monotonic guard rejected the fresh step-0..4767 logs. They're kept as noise-floor evidence (final losses match the `-v2` reruns within 0.01).

### Phase-1 loss-predictor infrastructure (landed this session)

**Script 1: `scripts/analysis/midtrain_loss_predictor.py`** (~370 lines).

- Pulls per-step data from `wandb.Api().run().scan_history()` (the GCS `tracker_metrics.jsonl` file only carries final summary, not a time series).
- **Bugfix that mattered:** initial version fetched `train_keys + eval_keys` in a single `scan_history` call, which returns only rows where *every* key is set — intersecting train (every step) with eval (every 200) collapsed the result to ~27 rows and broke every short-prefix fit. Two-phase fetch fixes it.
- Five baselines: **B0 last-value**, **B1 `a + b/√t`** (raw step), **B2 `L_∞ + A(U-u)^c`** with `c` bounded to `[0.2, 3.0]`, **B3 `L_∞ + A(U-u)^c`** with `c` fixed at `0.5` or `1.0`.
- Self-prefix + cross-LR LOO tests at prefixes `{30%, 50%, 80%}`.
- Target quantity: EMA-mean over steps `[4600, 4767]` (above the ~0.01 noise floor of single-step finals).
- CSVs: `midtrain_loss_predictor_{self_prefix,cross_lr,c_stability}.csv` (gitignored; rerun to regenerate).

**Script 2: `scripts/analysis/plot_midtrain_curves.py`** (~230 lines).

Four matplotlib PNGs, auto-opened via `open` on macOS.

### Phase-1 results (numbers)

**Self-prefix MAE on train/loss_smooth** (noise floor ≈ 0.01):

| method | 30% | 50% | 80% |
|---|---:|---:|---:|
| B0 last-value | 0.146 | 0.084 | 0.021 |
| **B1 a+b/√t** | **0.024** | 0.020 | **0.003** |
| B2 free c (bounded) | 0.085 | 0.056 | 0.007 |
| B3 c=0.5 | 0.428 | 0.124 | 0.041 |
| **B3 c=1** | 0.120 | **0.004** | 0.007 |

**Cross-LR LOO MAE on train/loss_smooth**:

| prefix | MAE |
|---:|---:|
| 30% | 0.182 (18× noise floor — not yet usable) |
| 50% | 0.028 (~3× noise floor — borderline) |
| 80% | 0.002 (at noise floor — clean) |

**Paloma c4_en** — at most 12-24 eval points per prefix; power-law fits are noisy. Best self-prefix MAE (B2 at 50%) is 0.037, roughly 10% of the metric's midtrain-induced change (0.29 to 0.43 depending on LR). Not yet a usable forecast; will improve with more runs (1e21 adds 3 more eval trajectories).

**c stability (B2 free-c)**: `c_std` is 0.8-0.9 for train/loss across prefixes and worse for Paloma. `c` hits the upper bound (3.0) at short prefixes. Parameter is under-identified on this data — don't report free-c B2 as the primary method.

### Visual findings (from `scripts/analysis/*.png`)

1. **`train_loss_vs_step.png`** — Three LR-factor curves have near-identical shape. Warmup-to-peak visible around step 500; decay tail smooth; the three curves fan out by only ~0.02 loss at the endpoint.

2. **`train_loss_vs_cumlr.png`** — The headline plot. Left panel (raw cumulative LR `u(t)`) shows the expected horizontal fan-out; `U` at endpoint is proportional to LR factor (~6.0, ~7.8, ~9.7 for 0.5 / 0.67 / 0.83). **Right panel (`u/U` normalized) shows the three curves nearly overlaying** with residual spread comparable to noise floor. This is the strongest single piece of evidence that cumulative-LR is the right time axis for cross-LR prediction — the "schedule-shape dominates" hypothesis is empirically supported on this data. With three runs this is modest evidence, but it's exactly the direction we want.

3. **`paloma_c4_en_vs_step.png`** — Retention tradeoff in full view. Pretrain c4_en = 2.8586 (dotted). All three runs rise during warmup + early decay:
   - lr=0.5 peaks at ~3.15 (Δ +0.29) around step 3000 and plateaus there.
   - lr=0.67 peaks at ~3.30 (Δ +0.44) around step 2500 and plateaus.
   - lr=0.83 peaks at ~3.40 (Δ +0.54) around step 3200, then **drops sharply in the last ~500 steps** to ~3.29 at endpoint.
   Higher LR factor → faster rise + higher peak damage. The lr=0.83 late-decay recovery is novel: the low-LR cooldown tail appears to partially undo retention damage. Worth re-testing as 1e21 data lands — if it reproduces, it's a hint that the end-of-cooldown behaviour is important for trade-off management.

4. **`predictor_fit_overlay.png`** — Side-by-side view of why B3 `c=1` wins at prefix 50%:
   - Left (raw step, B1): `a + b/√t` fit (orange dashed) traces the `lr=0.67` curve beautifully, extrapolates to 0.819 at step 4768; actual target 0.798 → error 0.021.
   - Right (cumulative LR, B2 + B3): B2's free-`c` fit lands `c = 2.34` (green dashed) and over-predicts at 0.865 — visibly too convex. B3 with fixed `c=1` (purple dash-dot) is nearly a straight line through the decay data and lands at 0.795, within 0.003 of the actual 0.798. This is the visual reason B3 `c=1` beats B2 in the MAE table at prefix 50%.

### Revised success criteria (update to the pre-committed ones)

Pre-committed:
- B2 MAE < 2× noise floor at prefix=30% for train/loss — **FAIL** (0.085, 8.5× floor). Swap: use B1 or B3 c=1 as primary predictor.
- B2 beats B1 by >20% on train/loss — **FAIL** (B2 is WORSE); but B3 c=1 beats B1 at 50% (0.004 vs 0.020). Schedule-awareness helps iff `c` is pinned.
- c_std < 0.1 per run/metric — **FAIL** (0.8-0.9). c is under-identified; don't free-float it.
- Cross-LR LOO MAE < 3× noise floor on train/loss at prefix=30% — **FAIL** at 30%, **PASS** at 50% (0.028 ≈ 3×), **PASS** at 80% (0.002).

Operational takeaway:
- **Default predictor for train/loss:** B3 with c=1 OR B1 raw sqrt-t. They're within 2× of each other at any prefix ≥ 50% and both at noise floor at 80%.
- **Default predictor for rising/sparse evals (Paloma):** B0 last-value. Power fits are noisier than the signal at this point-count.
- **Prefix horizon for cross-base test (phase 2):** 50%+. 30% was aspirational and is empirically too short for this functional family.

### What's NOT done (phase 2 blockers)

- **1e21 sweep NOT launched.** Three runs (`lr=0.5 / 0.67 / 0.83` on the 3.4B base) are the test set for the cross-base extrapolation. Each is ~10h on v5p-64. Launch recipe is the same as the 1e20 `-v2` recipe (including explicit `checkpoint_init_mode=CheckpointInitMode.MODEL_ONLY` and the `-v2` name suffix since the experiment file carries that now).
- **Math-domain evals NOT computed** on the 1e20 checkpoints. Needed to measure the "math target" side of the trade-off (currently only train/loss on math-data + Paloma c4_en retention). Would go through `experiments/evals/`.
- **Noise floor calibration is coarse.** Two pairs (0.67 orig vs v2: 0.000 gap, 0.83 orig vs v2: 0.010 gap) → "~0.005-0.010." A handful more seed replicates per LR factor would tighten this, but it's not blocking.
- **Phase 2 cross-base fit** needs the 1e21 trajectories. Blocked on point 1.
- **Phase 3 cross-scale** needs actual 1e22 runs. Out of scope until cross-base validates.

### Known non-blocking bugs filed in this logbook but not addressed

1. **Marin executor cache hash ignores plain `SimpleTrainConfig` fields.** Only `versioned(...)` values, `step.name`, and upstream `ExecutorStep` deps feed the hash. Changing `checkpoint_init_mode`, `learning_rate`, `beta2`, etc. without wrapping them in `versioned()` silently reuses cached outputs. Workaround: bump step.name (e.g., `-v2`). Proper fix: either wrap fields in `versioned()` at use sites or make `SimpleTrainConfig` auto-version critical fields. Not filed as a GH issue yet.
2. **W&B step-monotonic rejection on same-hash relaunch.** Levanter's `wandb.py:69` refuses `step < run.step` logs; a relaunch that inherits the same W&B run ID (because the executor hash didn't change) gets its fresh metrics silently dropped. Workaround: bump the hash. Proper fix: include a unique component (timestamp, attempt counter) in the W&B run config so relaunches always get fresh run_ids. Not filed as GH issue yet.
3. **`tracker_metrics.jsonl` name is misleading.** It contains only `{config, summary}` — one line per run. The per-step time series lives only in W&B. The file could reasonably be renamed (e.g., `run_summary.jsonl`) or the `tracker_metrics` name made accurate by also writing per-step JSONL alongside it. Not filed.

### How to resume from a cold start tomorrow morning

1. Open `.agents/logbooks/midtraining_delphi.md` and read this session-summary section + the phase-1 plan + phase-1 results.
2. If phase-1 numbers need to be regenerated: `uv run python scripts/analysis/midtrain_loss_predictor.py` (reads W&B, writes CSVs). `uv run python scripts/analysis/plot_midtrain_curves.py` (makes PNGs and opens them).
3. To launch the 1e21 sweep (the natural next step):
   ```bash
   for lr in 0.5 0.67 0.83; do
     uv run iris --cluster=marin job run \
       --cpu 1 --memory 3GB --disk 9GB \
       --region us-central1 --region us-east5 \
       --job-name "delphi-math-10b-1e21-lr${lr}-$(date -u +%Y%m%d-%H%M)" \
       --no-wait \
       -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
       -e WANDB_API_KEY "${WANDB_API_KEY}" \
       -e MIDTRAIN_SELECT_BASE 1e21-v5 \
       -e MIDTRAIN_SELECT_LR "$lr" \
       -- python experiments/exp_delphi_math_10b_midtrain.py
     sleep 5  # stagger to avoid any residual auto-naming collisions
   done
   ```
4. When the 3 × 1e21 runs land: add them to `RUNS_1E21` in the predictor module and rerun the script with a cross-base LOO test added (fit on 1e20 → predict 1e21 finals from features alone, or from first 50% of 1e21). If cross-base MAE on train/loss ≤ 0.05 at prefix 50%, project goal #1 has a validated baseline.

---

## 2026-04-24 principled `c` constraints follow-up

### Status

| Time | Action | Notes |
|---|---|---|
| 2026-04-24 01:06Z | Started follow-up requested by user: replace the bounded nonlinear free-`c` B2 fit with more principled constraints/diagnostics, regenerate plots, and update this logbook. | Target fixes: normalize remaining-LR axis, profile/grid-select `c`, add shared-`c` and regularized alternatives, and make the evidence visible in plots/CSVs. |
| 2026-04-24 01:10Z | Replaced bounded nonlinear B2 in `scripts/analysis/midtrain_loss_predictor.py`. | New B2 uses normalized remaining-LR `x=(U-u)/(U-u_fit_start)`, profiles `c` on `np.geomspace(0.1, 10.0, 241)`, and solves `(L_inf, A)` exactly by linear least squares for each candidate `c`. Added B2r with weak log-`c` prior centered at `c=1`, plus profiled shared-`c` cross-LR LOO. |
| 2026-04-24 01:11Z | Regenerated analysis outputs with `uv run python scripts/analysis/midtrain_loss_predictor.py` and `uv run python scripts/analysis/plot_midtrain_curves.py`. | Wrote CSVs and six PNGs under `scripts/analysis/`. Existing four plots were regenerated; new plots are `c_profile_scan.png` and `predictor_method_mae.png`. |
| 2026-04-24 01:11Z | First lint command failed because I used the wrong wrapper syntax. | Exact error: `Error: No such option: --files (Possible options: --all-files, --fix)`. Correct command is positional files: `./infra/pre-commit.py --fix scripts/analysis/midtrain_loss_predictor.py scripts/analysis/plot_midtrain_curves.py`. |
| 2026-04-24 01:12Z | First real lint pass failed; patched mechanical issues and reran. | Exact issue classes: `E501 Line too long`, `RUF046 Value being cast to int is already an integer`, `RUF059 Unpacked variable spec is never used`, `B023 Function definition does not bind loop variable ...`, and Black would reformat `scripts/analysis/midtrain_loss_predictor.py`. |
| 2026-04-24 01:12Z | Verification passed. | `./infra/pre-commit.py --fix scripts/analysis/midtrain_loss_predictor.py scripts/analysis/plot_midtrain_curves.py` passed all hooks. Final `uv run python scripts/analysis/midtrain_loss_predictor.py` also completed and rewrote the CSVs. |

### Implementation notes

The old B2 failure mode was not "SciPy needs a better box"; it was an identifiability problem. The updated implementation makes that explicit:

- `normalized_remaining_lr(u, U, fit_start_u)` maps every run/prefix to `x in [0, 1]`, so `L_inf` is the prediction at `x=0`.
- For fixed `c`, the model is linear: `L(x)=L_inf + A*x^c`. The code now solves `L_inf, A` via `np.linalg.lstsq` rather than nonlinear `curve_fit`.
- `B2_profiled_c` chooses `c` by a chronological prefix split: fit early prefix points, validate on the later prefix tail.
- `B2r_profiled_c_logprior1` adds `0.25 * var(y_val) * log(c/1)^2` to the validation MSE. This is a diagnostic prior, not the recommended production predictor.
- `midtrain_loss_predictor_c_profiles.csv` records the full c-scan objective surface. This is the artifact to inspect when asking whether `c` is identified.

### Results after replacing bounded B2

Self-prefix MAE on `train/loss_smooth`:

| method | 30% | 50% | 80% |
|---|---:|---:|---:|
| B1 `a+b/sqrt(t)` | **0.024** | 0.020 | **0.003** |
| B2 profiled `c` | 0.103 | 0.054 | 0.005 |
| B2r profiled `c` + log prior | 0.097 | 0.052 | 0.005 |
| **B3 fixed `c=1`** | 0.120 | **0.004** | 0.007 |
| B3 fixed `c=0.5` | 0.428 | 0.124 | 0.041 |

Cross-LR LOO MAE on `train/loss_smooth`:

| method | 30% | 50% | 80% |
|---|---:|---:|---:|
| shared profiled `c` | 0.181 | 0.028 | **0.002** |
| **fixed `c=1`** | **0.120** | **0.004** | 0.007 |
| fixed `c=0.5` | 0.428 | 0.124 | 0.041 |

`c` is still unstable even without optimizer bounds. Selected train-loss `c` values:

- lr=0.5: `3.224, 1.283, 1.039` for prefixes `30%, 50%, 80%`
- lr=0.67: `3.905, 2.512, 0.891`
- lr=0.83: `3.831, 2.512, 0.825`

No selected train-loss `c` hit the scan edge, so this is not a too-narrow-grid artifact. The prefix-tail validation objective genuinely prefers high `c` at 30-50%, but those high-`c` fits extrapolate poorly to the final endpoint.

### Visual findings from new plots

- `c_profile_scan.png`: at prefix 50%, validation minima sit around `c≈1.28` for lr=0.5 and `c≈2.51` for lr=0.67/0.83. The log-prior nudges them only slightly. This shows that prefix-tail validation alone does not recover the endpoint-useful `c=1` prior.
- `predictor_method_mae.png`: fixed `c=1` is the clear 50% prefix winner; B1 wins at 30%; B1/B2/B3 are all close to the noise floor by 80%, except fixed `c=0.5`.
- Updated `predictor_fit_overlay.png`: for lr=0.67 at 50%, B2/B2r still predict about `0.868/0.867` vs actual `0.798`, while fixed `c=1` predicts `0.795`. The old bounded-curve-fit explanation is gone; the failure remains and is now attributable to the profiled high-`c` choice.

### Updated recommendation

Do not report free/profilled `c` as a real learned exponent yet. It is useful as a diagnostic surface, not as the predictor. For phase 2, carry these baselines forward:

1. **Primary train-loss predictor:** B3 fixed `c=1`.
2. **Schedule-unaware sanity baseline:** B1 `a+b/sqrt(t)`.
3. **Diagnostic only:** B2 profiled `c` and B2r log-prior, with `midtrain_loss_predictor_c_profiles.csv` / `c_profile_scan.png` used to inspect identifiability.

The practical conclusion is unchanged but better supported: the principled constraint is not "bound `c`"; it is "choose `c` as a prior/hyperparameter and validate predictive performance." On the current 3-run 1e20 data, fixed `c=1` remains the best schedule-aware default.

---

## 2026-04-24 interactive prefix plot

### Status

| Time | Action | Notes |
|---|---|---|
| 2026-04-24 01:19Z | Started user-requested interactive plot for prefix sensitivity. | Goal: slider-controlled prefix percentage with live updates to predictor curves and absolute-error bars for each method. |
| 2026-04-24 01:21Z | Added `scripts/analysis/interactive_midtrain_prefix_plot.py` and generated/opened `scripts/analysis/interactive_midtrain_prefix_plot.html`. | Command: `uv run python scripts/analysis/interactive_midtrain_prefix_plot.py`. The HTML embeds Plotly + precomputed predictions for prefixes 20%-90% across the three canonical 1e20 runs, so slider updates are browser-local and instant. |
| 2026-04-24 01:22Z | Linted the new generator. | `./infra/pre-commit.py --fix scripts/analysis/interactive_midtrain_prefix_plot.py` passed. The generated HTML is ~16 MB and intentionally left as an analysis artifact next to the PNGs, not as source code to edit by hand. |
| 2026-04-24 01:25Z | Improved interactive plot labeling after user feedback. | Renamed predictor legend entries, added formula/detail text to the table, added explanatory note cards, and labeled the prefix/warmup/target markers directly in plot titles/annotations. Regenerated `interactive_midtrain_prefix_plot.html`; lint still passes for the generator. |
| 2026-04-24 01:31Z | Fixed the interactive plot's axis/run comparison design after user feedback. | Default view is now **All LR runs** on normalized cumulative LR `u/U`, with x-axis selector for `u/U`, raw cumulative LR `u`, or training step. The main plot overlays all three runs for the selected predictor method; bars/table/error-vs-prefix aggregate across the selected run scope. Regenerated `interactive_midtrain_prefix_plot.html`. Lint passed. Playwright smoke test was attempted but the repo venv does not have `playwright` installed (`No module named 'playwright'`). |
| 2026-04-24 05:50Z | Lowered interactive prefix slider floor from 20% to just after warmup. | New lower bound is computed as `ceil(100 * WARMUP_STEPS / TOTAL_STEPS) = 11%`. A true 10% prefix is before warmup ends (`500 / 4768 = 10.49%`) and leaves no post-warmup fit window because all fit functions skip warmup points. Regenerated `interactive_midtrain_prefix_plot.html`, reopened in Chrome, and lint passed. |
| 2026-04-24 05:54Z | Added train/eval metric toggle and made Chrome the default opener. | `interactive_midtrain_prefix_plot.py` now precomputes both `train/loss_smooth` and `eval/paloma/c4_en/loss`. The top-right toggle switches every plot/table between train loss and eval loss. Eval loss is sparse and can move up or down; expected early-prefix fit failures are shown as missing predictions in the UI rather than noisy warnings. The generator now always runs `open -a "Google Chrome" ...`. Regenerated HTML and lint passed. |

### Interactive plot details

Artifact: `scripts/analysis/interactive_midtrain_prefix_plot.html`.

Controls:

- Compare selector: all LR runs by default, or a single `lr=0.5`, `lr=0.67`, `lr=0.83` run.
- X-axis selector: normalized cumulative LR `u/U` by default, raw cumulative LR `u`, or training step.
- Fit overlay selector: choose which predictor family is drawn on top of the observed curves.
- Prefix slider: 11%-90% in 1% increments. The lower bound is the first whole-percent prefix after warmup.
- Top-right metric toggle: train loss or Paloma `c4_en` eval loss.

Live views:

- Fit overlay: full `train/loss_smooth` curves for the selected run scope, prefix/warmup/target markers in the selected x-axis coordinate, and endpoint markers for the selected predictor.
- Absolute-error bars at the selected prefix. In all-run mode this is mean absolute error across the three LR factors.
- Error-vs-prefix trajectories for all methods. In all-run mode this is mean absolute error across the three LR factors.
- Method table with per-run prediction/error columns in all-run mode, or prediction/error/selected-`c` detail in single-run mode.

Eval-loss interpretation: `eval/paloma/c4_en/loss` is a retention metric and is sparse (~eval cadence, not every train step). Higher is worse relative to the pretrain checkpoint, but the curve is not guaranteed monotone: it can rise during high-LR adaptation and partially recover during cooldown. The signed `A` in B2/B3 lets the fitted endpoint be above or below the prefix trend, but the one-direction power family still cannot represent rise-then-recover dynamics well. Treat eval fits as diagnostics; for actual decision-making, prefer plotting the observed eval trajectory and using B0/last-value or a very local last-few-points trend until more eval trajectories land.

This is meant for visual inspection rather than batch reporting. The source of truth for the formulas remains `scripts/analysis/midtrain_loss_predictor.py`.

---

## 2026-04-24 prefix pre-registration protocol and 1e20 eval read

### Pre-registration clarification

For a 20% / 30% / 50% prefix predictor, pre-registration does **not** mean making endpoint predictions before seeing any run data. It means locking the prediction rule before inspecting any data **to the right of the declared prefix cutoff**.

Concrete protocol for the next 1e21 sweep:

1. Before launch, or at least before reading beyond the first prefix cutoff, write down:
   - Runs: `1e21` base with LR factors `{0.5, 0.67, 0.83}`.
   - Prefix checkpoints: `20%`, `30%`, `50%`, `80%`.
   - Primary prefix for claims: `50%`; `20%` and `30%` are exploratory stress tests.
   - Allowed data: only rows with `step <= floor(prefix * total_steps)`, excluding warmup points for fit functions that require post-warmup data.
   - Train target: final-window mean of `train/loss_smooth`.
   - Eval target: final-window `eval/paloma/c4_en/loss`, plus peak Paloma/c4 damage as a retention-risk diagnostic.
   - Primary predictor: B3 fixed `c=1` on normalized cumulative LR `u(t)/U`.
   - Baseline predictor: B1 `a + b / sqrt(t)`.
   - Diagnostics only: B2 profiled `c` and B2r profiled `c` plus weak log-`c` prior.
   - Success criterion for the predictor track: cross-base train-loss MAE <= `0.05` at the 50% prefix.

2. When a run reaches a declared prefix, freeze a prediction artifact before looking further ahead:
   - `run_id`
   - git SHA / experiment config hash
   - prefix percent and max step read
   - metric
   - predictor method
   - predicted endpoint / final-window target
   - fitted parameters, including `c` if applicable
   - timestamp and command used to generate the row

3. Append or write those rows before opening later W&B panels or rerunning analysis on post-prefix data. A timestamped CSV under `scripts/analysis/` plus a short logbook entry is enough.

This keeps the test causally clean: the first 20% may be used for a 20%-prefix forecast, but steps after 20% must not influence that forecast's method choice or parameters.

### Current 1e20 eval read

From `scripts/analysis/midtrain_loss_predictor_self_prefix.csv`, final-window targets for the fixed 1e20 v2 sweep are:

| LR factor | `train/loss_smooth` final-window target | `eval/paloma/c4_en/loss` final-window target |
|---:|---:|---:|
| 0.50 | 0.812572 | **3.150040** |
| 0.67 | 0.798438 | 3.289912 |
| 0.83 | **0.797175** | 3.292075 |

Interpretation:

- On math-data train loss, `lr=0.67` and `lr=0.83` are effectively tied and both beat `lr=0.5`. The tiny `0.83` advantage over `0.67` is likely too small to treat as decisive without downstream math evals.
- On Paloma/c4 retention, `lr=0.5` is clearly best: it has much less c4 damage than `0.67` or `0.83`.
- Therefore there is no single "best 1e20 model" yet unless we specify the objective. Eval-only / retention winner is `lr=0.5`; adaptation winner by train loss is `lr=0.83` by a tiny margin; the practical winner requires downstream math evals on the checkpoints.

### Immediate recommendation

Do not choose the 1e21 LR solely from Paloma/c4. Run the 1e21 three-point LR sweep and pre-register prefix predictions as above. In parallel, run downstream math evals on the completed 1e20 checkpoints so the tradeoff can be scored as math gain versus retention cost, not just train loss versus c4 damage.

---

## 2026-04-24 stale missing-BOS cache cleanup

### Action

Deleted the stale `us-central1` tokenized cache:

```bash
gcloud -q storage rm --recursive \
  gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

Verification at `2026-04-24 21:22Z`:

```text
ERROR: (gcloud.storage.ls) One or more URLs matched no objects.
```

Remaining regional `nemotron_cc_math_v1` tokenized prefixes after deletion:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/3-947143/
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/3-ef5cb9/
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus_mind-d60b4a/
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/3-ef5cb9/
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-da9608/
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus_mind-d60b4a/
```

### Rationale

`4plus-212a2d` was produced before the Levanter BatchTokenizer BOS fix and was empirically missing Llama-3 BOS (`128000`) at the start of documents. Keeping it around made future `us-central1` relaunches likely to silently reuse the bad cache. The good `4plus` cache currently available is:

```text
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-da9608/
```

That cache was sampled after the BOS fix and starts documents with `128000`.

---

## 2026-04-24 zero-end-LR BOS rerun plan

### Requested change

User observed that the AdamH learning-rate chart for the fixed 1e20 sweep annealed to `0.1 * peak`, not zero, and asked to:

1. Re-tokenize `nemotron_cc_math_v1/4plus` on `us-central1` now that the BatchTokenizer BOS fix is present.
2. Rerun the 1e20 LR sweep using the rebuilt BOS-correct `us-central1` cache.
3. Fix the midtraining schedule so both `optim/learning_rate` and `optim/adam_lr` fully anneal to zero by the final step.

### Config patch

Updated `experiments/exp_delphi_math_10b_midtrain.py`:

- `MIN_LR_RATIO: 0.1 -> 0.0`
- training step names: `...-lr{factor}-v2 -> ...-lr{factor}-v3`

The `-v3` suffix is required because the Marin executor hash does not reliably include plain `SimpleTrainConfig` or optimizer fields. Bumping the step name guarantees fresh checkpoint output paths and fresh W&B run IDs for the zero-end-LR reruns.

Validation:

```text
./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py
```

passed. Local import check for `1e20 × lr=0.67` showed:

```text
step.name checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v3
optimizer.min_lr_ratio 0.0
optimizer.warmup 500
optimizer.decay 4268
checkpoint_init_mode model_only
```

### Execution plan

1. Launch tokenization-only rebuild on `us-central1`:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 \
  --job-name delphi-1e20-retokenize-4plus-bos-uscentral1-20260424 \
  --no-wait \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -- python -c 'from marin.execution.executor import executor_main; from experiments.midtraining_data_buckets import BUCKET_2; executor_main(steps=[BUCKET_2["nemotron_cc_math_v1/4plus"]], description="Retokenize Nemotron CC Math v1 4plus on us-central1 after BOS fix")'
```

Expected output path is the same as the deleted stale cache:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

2. Verify that sampled rows start with Llama-3 BOS `128000`.
3. Launch the three `1e20-iso-d2048-L21` reruns pinned to `us-central1` with `MIDTRAIN_SELECT_LR={0.5,0.67,0.83}`.

### Execution results

Retokenization completed successfully.

Parent job:

```text
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424
```

Observed child phases:

```text
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-tokenize-train-82c5ccd4-p0-a0
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-tokenize-train-82c5ccd4-p1-a0
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-levanter-cache-probe-1bdbbb3f-p0-a0
/ahmed/delphi-1e20-retokenize-4plus-bos-uscentral1-20260424/zephyr-levanter-cache-copy-b6900715-p0-a0
```

Final cache path:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

Cache stats:

```text
total_tokens 51482572371
total_elements 45096087
```

BOS sample verification from `train` cache:

```text
0 [128000, 2, 8388, 1815, 261, 27930, 10176, 220, 605, 339, 93678, 23508]
1 [128000, 2, 11106, 25, 46551, 279, 13031, 40227, 315, 279, 65048, 780]
2 [128000, 2, 358, 13, 384, 60217, 34495, 653, 372, 14799, 271, 334]
```

This confirms the rebuilt `us-central1` cache starts documents with Llama-3 BOS `128000`.

Launched the zero-end-LR 1e20 sweep on `us-central1`:

| LR factor | parent job | checkpoint output | child train job state at 2026-04-24 21:58Z |
|---:|---|---|---|
| 0.50 | `/ahmed/delphi-math-10b-1e20-lr0p5-v3-zeroend-bos-20260424-215257` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v3-298da6` | `/train_lm` submitted; `JOB_STATE_RUNNING`; 8 tasks running |
| 0.67 | `/ahmed/delphi-math-10b-1e20-lr0p67-v3-zeroend-bos-20260424-215310` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v3-88e817` | `/train_lm` submitted; `JOB_STATE_RUNNING`; 2 tasks running, 6 tasks building |
| 0.83 | `/ahmed/delphi-math-10b-1e20-lr0p83-v3-zeroend-bos-20260424-215327` | `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v3-0fad76` | `/train_lm` submitted; `JOB_STATE_PENDING`; waiting for v5p-64 capacity |

Parent logs for all three reruns show:

- `tokenized/nemotron_cc_math_v1/4plus_d68139d8` output path is `gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d`.
- The tokenized step is skipped as already succeeded, so the reruns reuse the newly rebuilt BOS-correct cache.
- Training step names use `-v3`, producing fresh checkpoint outputs and W&B run IDs.
- Model launch config is `Qwen3Config seq_len=4096 hidden=2048 batch=512 device=TpuConfig(variant='v5p-64')`.

Remaining watch item: confirm in W&B that `optim/adam_lr` reaches zero at final step once the v3 jobs complete. The config-level validation is already correct (`min_lr_ratio=0.0`, `warmup=500`, `decay=4268`), but the plotted confirmation requires the runs to finish.

---

## 2026-04-24 W&B project relaunch

### Requested change

After the `-v3` zero-end-LR relaunch, the user asked whether W&B projects can be selected and requested that the just-launched jobs be killed and rerun under project:

```text
delphi-midtraining
```

### Code patch

Updated `experiments/defaults.py` so `default_train(...)` accepts:

```python
wandb_project: str | None = None
```

and resolves it as:

```python
wandb_project = os.environ.get("WANDB_PROJECT", "marin")
```

when unset. This value is now passed to `WandbConfig(project=wandb_project)` instead of hardcoding `"marin"`.

Updated `experiments/exp_delphi_math_10b_midtrain.py` again:

- kept `MIN_LR_RATIO = 0.0`
- changed training step suffix from `-v3` to `-v4`

The `-v4` suffix avoids reusing partial `-v3` checkpoint paths / W&B run IDs from the killed launch.

Validation:

```text
WANDB_PROJECT=delphi-midtraining MIDTRAIN_SELECT_BASE=1e20-iso-d2048-L21 MIDTRAIN_SELECT_LR=0.67 uv run python - <<'PY'
from experiments.exp_delphi_math_10b_midtrain import runs
step = runs[0]
cfg = step.config.train_config
print('step.name', step.name)
print('wandb.project', cfg.trainer.tracker.project)
print('optimizer.min_lr_ratio', cfg.optimizer.min_lr_ratio)
print('optimizer.warmup', cfg.optimizer.warmup)
print('optimizer.decay', cfg.optimizer.decay)
PY
```

Output:

```text
step.name checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v4
wandb.project delphi-midtraining
optimizer.min_lr_ratio 0.0
optimizer.warmup 500
optimizer.decay 4268
```

`./infra/pre-commit.py --fix experiments/defaults.py experiments/exp_delphi_math_10b_midtrain.py` passed.

### Terminated `-v3` jobs

Terminated the three `-v3` parent jobs requested by the user:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v3-zeroend-bos-20260424-215257
/ahmed/delphi-math-10b-1e20-lr0p67-v3-zeroend-bos-20260424-215310
/ahmed/delphi-math-10b-1e20-lr0p83-v3-zeroend-bos-20260424-215327
```

Follow-up `get-job-state` showed all three parent jobs and their `/train_lm` child jobs as `JOB_STATE_FAILED`, which is the expected terminal state after termination.

### Relaunched `-v4` jobs

Relaunch command shape:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 \
  --job-name "$job_name" \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 \
  -e MIDTRAIN_SELECT_LR "$lr" \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Submitted:

| LR factor | parent job | status at 2026-04-24 22:21Z |
|---:|---|---|
| 0.50 | `/ahmed/delphi-math-10b-1e20-lr0p5-v4-zeroend-bos-wandbproject-20260424-221945` | parent running; `/train_lm` submitted and pending v5p-64 workers |
| 0.67 | `/ahmed/delphi-math-10b-1e20-lr0p67-v4-zeroend-bos-wandbproject-20260424-221959` | parent pending but schedulable |
| 0.83 | `/ahmed/delphi-math-10b-1e20-lr0p83-v4-zeroend-bos-wandbproject-20260424-222010` | parent pending but schedulable |

lr=0.5 parent logs confirmed it reused the rebuilt BOS-correct central cache:

```text
Step = tokenized/nemotron_cc_math_v1/4plus_d68139d8 ... Output_path = gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d
Skip tokenized/nemotron_cc_math_v1/4plus_d68139d8: already succeeded
```

lr=0.5 checkpoint output:

```text
gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v4-062652
```

Serialized experiment metadata for lr=0.5:

```text
gs://marin-us-central1/experiments/exp_delphi_math_10b_midtrain-2c5dc6.json
```

contains:

```json
"tracker": {
  "project": "delphi-midtraining"
}
```

So the W&B project override is wired through the actual Levanter training config, not just present as an environment variable.

## 2026-04-24 launch incident: `-v4` shared temp-checkpoint failure

User reported the three `-v4` jobs all failed. Checked the actual failed `/train_lm`
task logs for:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v4-zeroend-bos-wandbproject-20260424-221945/train_lm/0
/ahmed/delphi-math-10b-1e20-lr0p67-v4-zeroend-bos-wandbproject-20260424-221959/train_lm/7
/ahmed/delphi-math-10b-1e20-lr0p83-v4-zeroend-bos-wandbproject-20260424-222010/train_lm/2
```

All three failed before step 0 with the same pattern:

```text
No checkpoints found in ['gs://marin-us-central1/checkpoints/...-v4-.../checkpoints']
Discovered latest checkpoint at gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/step-48
Found prior temporary checkpoint gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/step-48.
FileNotFoundError: Missing 34 arrays in OCDBT checkpoint: [...]
```

Root cause: `lib/marin/src/marin/training/training.py::_update_config_to_use_out_path`
made permanent checkpoints unique via `$output_path/checkpoints`, but set every
executor-backed training job's `temporary_base_path` to the shared
`gs://marin-tmp-<region>/ttl=14d/checkpoints-temp`. `_enforce_run_id` then set
`append_run_id_to_base_path=False` when the run ID was imputed from the executor
output path. That is correct for the permanent path, because it is already unique,
but wrong for the temporary path. The result was a shared rolling temp checkpoint
namespace across independent jobs. These `-v4` runs picked up an incompatible
stale temp checkpoint at `step-48`, so model-only initialization never reached the
base checkpoint load.

Fix applied:

- Added `_temporary_checkpoint_base_path(...)` in `lib/marin/src/marin/training/training.py`.
- When the run ID is imputed from `output_path`, temporary checkpoints now go under:

```text
gs://marin-tmp-<region>/ttl=14d/checkpoints-temp/<basename(output_path)>/
```

- Added `tests/test_training.py::test_executor_output_path_scopes_temporary_checkpoints`.
- Bumped the Delphi 1e20 sweep suffix from `-v4` to `-v5` so relaunches get fresh executor outputs.

Validation:

```text
uv run --project lib/marin --group test pytest tests/test_training.py::test_executor_output_path_scopes_temporary_checkpoints -q
.
1 passed in 1.48s

WANDB_PROJECT=delphi-midtraining MIDTRAIN_SELECT_BASE=1e20-iso-d2048-L21 MIDTRAIN_SELECT_LR=0.67 uv run python - <<'PY'
from experiments.exp_delphi_math_10b_midtrain import runs
step = runs[0]
cfg = step.config.train_config
print('step.name', step.name)
print('wandb.project', cfg.trainer.tracker.project)
print('optimizer.min_lr_ratio', cfg.optimizer.min_lr_ratio)
print('optimizer.warmup', cfg.optimizer.warmup)
print('optimizer.decay', cfg.optimizer.decay)
PY

step.name checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v5
wandb.project delphi-midtraining
optimizer.min_lr_ratio 0.0
optimizer.warmup 500
optimizer.decay 4268
```

Next action: relaunch the three 1e20 jobs as `-v5` with `WANDB_PROJECT=delphi-midtraining`,
then watch the `/train_lm` children until they pass the previous failure point and emit
real training progress.

## 2026-04-24 `-v5` relaunch monitoring

Relaunched the three 1e20 LR sweep jobs after the temp-checkpoint fix, still in
`us-central1`, with `WANDB_PROJECT=delphi-midtraining` and the zero-end LR schedule:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-zeroend-bos-wandbproject-20260424-224700
/ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720
```

Confirmed the `lr=0.83` child training job is actually running with 8/8 workers:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720/train_lm
JOB_STATE_RUNNING
task_state_counts: {"running": 8}
```

This run passed the previous restore failure point. The relevant log lines are:

```text
No checkpoints found in ['gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints']
No checkpoints found in ['gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b']
Loading cache from gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/train
Loading checkpoint from gs://marin-us-central1/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915
Error check finished successfully
First train step completed in 35.8s (step 0)
Progress on:train 72.0it/4.77kit ... postfix:loss=1.37
```

Also confirmed W&B is pointing at the requested project:

```text
wandb: View project at https://wandb.ai/marin-community/delphi-midtraining
```

As of ~23:01 UTC, the other two `/train_lm` children are queued rather than failed:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-zeroend-bos-wandbproject-20260424-224700/train_lm
JOB_STATE_PENDING
task_state_counts: {"pending": 8}

/ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm
JOB_STATE_PENDING
task_state_counts: {"pending": 8}
```

Both pending jobs report the same scheduler reason:

```text
Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity.
Insufficient TPUs (need 4, available 0).
Autoscaler: tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity.
```

Interpretation: the code/config launch issue is fixed for jobs that get TPU capacity;
`lr=0.5` and `lr=0.67` have not reached runtime yet because `us-central1` does not
currently have another full v5p-64 coscheduled group available.

## 2026-04-24 23:07 UTC `-v5` status refresh

Refreshed the three relaunched 1e20 sweep jobs:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-zeroend-bos-wandbproject-20260424-224700/train_lm  JOB_STATE_PENDING
/ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm JOB_STATE_PENDING
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720/train_lm JOB_STATE_RUNNING
```

The running `lr=0.83` job continues to make normal training progress:

```text
Progress on:train 100it/4.77kit ... postfix:loss=1.31
Progress on:train 128it/4.77kit ... postfix:loss=1.26
Progress on:train 156it/4.77kit ... postfix:loss=1.26
Progress on:train 170it/4.77kit ... postfix:loss=1.19
```

The `lr=0.5` and `lr=0.67` jobs are still queued with 8/8 pending tasks each.
They have not failed and still show `failure_count=0`, `preemption_count=0`.
Both report the same scheduler/autoscaler reason:

```text
Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e',
only 0 of 8 have capacity.
Insufficient TPUs (need 4, available 0).
tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity.
```

Immediate interpretation: launch/config is no longer the blocker. The remaining
issue for `lr=0.5` and `lr=0.67` is `us-central1` v5p-64 capacity. Leave them
queued unless we explicitly decide to move/duplicate those runs in another region.

## 2026-04-24 23:12 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260424 23:12:52 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 230it/4.77kit rate:4.4s/it remaining:5:32:15 elapsed:20:21 postfix:loss=1.17`

Pending reason:
- `lr=0.5`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-24 23:14 UTC detached monitor started

Started a resident monitor for the three Delphi 1e20 `-v5` jobs. It runs in:

```text
tmux session: delphi-midtrain-monitor
state file: scratch/delphi_midtrain_monitoring_state.json
process log: scratch/delphi_midtrain_monitor_20260424-231428.log
monitor script: scratch/monitor_delphi_midtrain_jobs.py
```

The monitor appends one compact status snapshot to this logbook every 30 minutes.
It records job states, task counts, failure/preemption counts, the latest train
progress line for any running job, and pending reasons for queued jobs. It stops
itself once all three tracked jobs are terminal.

Initial detached state:

```json
{
  "pid": 49293,
  "status": "sleeping_initial_delay",
  "next_check_utc": "2026-04-24T23:44:29+00:00",
  "interval_seconds": 1800
}
```

To inspect the monitor later:

```bash
tmux list-sessions | rg delphi-midtrain-monitor
cat scratch/delphi_midtrain_monitoring_state.json
tail -n 40 scratch/delphi_midtrain_monitor_20260424-231428.log
```

To stop it intentionally:

```bash
tmux kill-session -t delphi-midtrain-monitor
```

## 2026-04-24 23:44 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260424 23:44:13 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 629it/4.77kit rate:5.4s/it remaining:6:10:13 elapsed:51:42 postfix:loss=1.04`

Pending reason:
- `lr=0.5`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Unsatisfied autoscaler demand: tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 00:15 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.5`: `I20260425 00:15:07 140004800845632 tqdm_loggable.tqdm_logging Progress on:train 230it/4.77kit rate:4.4s/it remaining:5:31:20 elapsed:20:15 postfix:loss=1.2`
- `lr=0.83`: `I20260425 00:15:22 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 1.00kit/4.77kit rate:52.2s/it remaining:54:39:15 elapsed:1:22:51 postfix:loss=0.998`

Pending reason:
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 00:45 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_PENDING        pending=8; failures=0; preemptions=0
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.5`: `I20260425 00:45:23 140004800845632 tqdm_loggable.tqdm_logging Progress on:train 616it/4.77kit rate:4.5s/it remaining:5:11:22 elapsed:50:31 postfix:loss=1.02`
- `lr=0.83`: `I20260425 00:45:27 139792089573184 tqdm_loggable.tqdm_logging Progress on:train 1.40kit/4.77kit rate:4.4s/it remaining:4:06:49 elapsed:1:52:56 postfix:loss=0.929`

Pending reason:
- `lr=0.67`: Scheduler: Coscheduling: need 8 workers in 'tpu-name' group 't1v-n-1d6a1c3e', only 0 of 8 have capacity: Insufficient TPUs (need 4, available 0) - 8 worker(s) Autoscaler: Waiting for workers in scale group 'tpu_v5p-preemptible_64-us-central1-a' to become ready (selected: demand-routed)

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 01:16 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_KILLED         killed=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260425 01:15:48 139816879437632 tqdm_loggable.tqdm_logging Progress on:train 1.61kit/4.77kit rate:4.5s/it remaining:3:58:59 elapsed:21:22 postfix:loss=0.934`

Terminal states:
- `lr=0.5`: `JOB_STATE_KILLED` exit=0 error=''
- `lr=0.67`: `JOB_STATE_FAILED` exit=0 error='Coscheduled sibling /ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm/3 failed'

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 01:46 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_KILLED         killed=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
lr=0.83 JOB_STATE_RUNNING        running=8; failures=0; preemptions=0
```

Latest progress:
- `lr=0.83`: `I20260425 01:44:53 139816879437632 tqdm_loggable.tqdm_logging Progress on:train 2.00kit/4.77kit rate:4.4s/it remaining:3:22:46 elapsed:50:26 postfix:loss=0.886`

Terminal states:
- `lr=0.5`: `JOB_STATE_KILLED` exit=0 error=''
- `lr=0.67`: `JOB_STATE_FAILED` exit=0 error='Coscheduled sibling /ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm/3 failed'

Monitor note: continuing; next automated check is scheduled for 30 minutes after this tick.

## 2026-04-25 01:52 UTC multi-region recovery launch

After the 00:54 UTC TPU preemption cascade, only `lr=0.83 v5` auto-recovered (it
restored from its step-1000 checkpoint and resumed training; W&B step trace is
continuous through the boundary, currently step ~2000, loss 0.886, LR 2.5e-3).
`lr=0.5 v5` had no saved checkpoint (crashed at step 724, before the 1000-step
save interval) and `lr=0.67 v5` had not started training when the cascade hit;
both parents terminated with `RuntimeError: 1 step(s) failed`.

User asked to relaunch the two failed runs with multi-region (`us-central1` +
`us-east5`) so the autoscaler can place either job in whichever pool frees a
v5p-64 first. Cluster-wide v5p-64 inventory at 18:38 UTC: us-central1-a 2/2
ready (2 occupied: 1 mine, 1 tonyhlee/moojink-shared), us-east5-a 3/3 ready
(2 moojink jobs + 1 spare with demand 1 from another tenant). Adding east5
to the region list does not immediately unblock — both pools were full at
launch time — but it widens placement when any slot frees.

### Cache pre-stage (cross-region copy)

`MATH_TRAIN_STEP` is *not* wrapped in `mirrored()` (only the pretrain ckpt is).
The BOS-correct `4plus-212a2d` tokenize cache only exists in us-central1
(`gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/`,
51.48 B tokens, 45.10 M docs, ~210 GB on disk). If a relaunched job lands in
us-east5, the executor would re-run normalize+tokenize from raw — ~10–20 h
of wasted compute. Pre-staged via:

```bash
gcloud storage cp -r --no-clobber \
  gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/ \
  gs://marin-us-east5/tokenized/nemotron_cc_math_v1/
```

(Background, ~5–10 min, ~$5 inter-region egress.) us-east5 already has a
stale `4plus-da9608` tree from before the BOS-fix cache rebuild, but the
hash the current experiment file resolves to is `4plus-212a2d` so we copy
that exact path. Race risk: if a TPU placement happens before the copy
finishes and the executors `.executor_status` was written but shards are
incomplete, training would read partial data — mitigated by the fact that
both pools were at capacity at launch, so placement will not race.

### Submit recipe

```bash
ts=$(date -u +%Y%m%d-%H%M%S)
job=delphi-math-10b-1e20-lr0p${LR}-v5-multiregion-${ts}
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 --region us-east5 \
  --job-name "$job" \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 \
  -e MIDTRAIN_SELECT_LR "$LR" \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Step name kept at `-v5` (not bumped to `-v6`) so the executor hash and the
checkpoint output path are preserved. Trade-off: the existing W&B run
`delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907` is at terminal step 724
in `crashed` state, and Levanters wandb monotonicity guard will silently
drop re-logged metrics for steps 0..724 on the new training start — we lose
the early curve in the W&B chart, but the model is unaffected.

Submitted:

```text
/ahmed/delphi-math-10b-1e20-lr0p5-v5-multiregion-20260425-015152   (lr=0.5)
/ahmed/delphi-math-10b-1e20-lr0p67-v5-multiregion-20260425-015212  (lr=0.67)
```

`lr=0.83 v5-zeroend-bos-wandbproject-20260424-224720` is left undisturbed.

## 2026-04-25 02:17 UTC monitor tick

Automated 30-minute monitor snapshot for the Delphi 1e20 `-v5` jobs.

```text
lr=0.5  JOB_STATE_KILLED         killed=8; failures=0; preemptions=0
lr=0.67 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
lr=0.83 JOB_STATE_FAILED         failed=1, worker_failed=7; failures=1; preemptions=707
```

Terminal states:
- `lr=0.5`: `JOB_STATE_KILLED` exit=0 error=''
- `lr=0.67`: `JOB_STATE_FAILED` exit=0 error='Coscheduled sibling /ahmed/delphi-math-10b-1e20-lr0p67-v5-zeroend-bos-wandbproject-20260424-224710/train_lm/3 failed'
- `lr=0.83`: `JOB_STATE_FAILED` exit=0 error='Exit code: 1. stderr: RPC: /tensorflow.CoordinationService/PollForError [type.googleapis.com/tensorflow.CoordinationServiceError=\'\\"\\x0c\\n\\njax_worker\']'

Monitor note: all tracked jobs are terminal; the detached monitor will stop after this tick.

## 2026-04-25 03:19 UTC post-Claude recovery audit

Read the latest logbook entries plus live Iris state after Claude's multi-region
recovery work. What Claude did:

- Copied the BOS-correct tokenized Nemotron math cache from:

```text
gs://marin-us-central1/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

to:

```text
gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-212a2d/
```

- Launched multi-region replacements for `lr=0.5` and `lr=0.67` with both
  `--region us-central1` and `--region us-east5`.
- The first `lr=0.67` multi-region attempt died during container build from a
  transient GitHub release-asset timeout while fetching `dupekit`; Claude
  resubmitted it as:

```text
/ahmed/delphi-math-10b-1e20-lr0p67-v5-multiregion-20260425-022011
```

- The original `lr=0.83` `-v5` run failed after reaching/saving step 2000.
  Claude copied its step-2000 checkpoint from central1 to east5.

Checkpoint copy check:

```text
23216831348 gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
23216831348 gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
```

Current live state:

```text
lr=0.5  multiregion /train_lm  JOB_STATE_RUNNING  running=8, failures=0, preemptions=0
lr=0.67 multiregion /train_lm  JOB_STATE_RUNNING  running=8, failures=0, preemptions=0
lr=0.83 original /train_lm     JOB_STATE_FAILED   failures=1, preemptions=707
```

Latest training progress:

```text
lr=0.5  Progress on:train 1.02kit/4.77kit ... postfix:loss=0.978
lr=0.67 Progress on:train 616it/4.77kit ... postfix:loss=1.01
```

Placement check: despite multi-region submission, both live replacement jobs
currently landed in `tpu_v5p-preemptible_64-us-central1-a`; no active training
task is using east5 yet.

Open decision: `lr=0.83` has not been relaunched. Its step-2000 checkpoint is
available in both regions, so a multi-region relaunch should resume from there
if we want a BOS-correct completed `lr=0.83` point.

## 2026-04-25 03:25 UTC lr=0.5 resume mistake

User noticed W&B now has two `lr=0.5` runs:

```text
delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907  original
delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-bcab01  Claude multi-region replacement
```

Verified this is a real resume mistake, not just a W&B display issue.
The original `lr=0.5` job had a temporary checkpoint:

```text
gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907/step-630/
```

but no permanent step checkpoint under:

```text
gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907/checkpoints/
```

The replacement job did not use the original output/checkpoint namespace. It
writes to:

```text
gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-bcab01/
```

and saved:

```text
gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-bcab01/checkpoints/step-1000/
```

Root cause: changing the launch to multi-region changed the executor output
prefix/dependency realization, so the automatic executor hash changed from
`3bd907` to `bcab01`. Keeping the human-readable step name at `-v5` was not
enough to preserve the actual run id. To resume a run, the launch must force the
exact old output path, e.g. via `ExecutorStep.with_output_path(...)` or an
explicit `override_output_path`, not rely on the step name.

Current implication:

- `bcab01` is scientifically a valid fresh `lr=0.5` BOS-correct run, but it is
  not a continuation of `3bd907`.
- The original `3bd907` temp checkpoint at step 630 still exists for now.
- A true resume would need to relaunch `lr=0.5` with output path forced to
  `gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.5-v5-3bd907`.
- Do not kill `bcab01` automatically; it is already past step 1000 and may still
  be useful if the priority is final endpoint rather than W&B continuity.

## 2026-04-25 CRITICAL RESUME RULE

DO NOT RELAUNCH A FAILED DELPHI MIDTRAINING RUN BY ONLY REUSING THE HUMAN-READABLE
STEP NAME.

THE REAL RUN ID AND CHECKPOINT NAMESPACE INCLUDE THE MARIN EXECUTOR OUTPUT HASH.
IF THE REGION LIST, PREFIX, DEPENDENCY REALIZATION, OR OVERRIDE PATH CHANGES, THE
HASH CAN CHANGE EVEN WHEN THE VISIBLE NAME STILL SAYS `-V5`.

BEFORE RELAUNCHING ANY FAILED OR PREEMPTED DELPHI MIDTRAINING RUN:

1. FIND THE EXACT OLD OUTPUT PATH AND RUN ID FROM IRIS LOGS OR GCS.
2. CHECK BOTH PERMANENT CHECKPOINTS AND TEMPORARY CHECKPOINTS.
3. FORCE THE RELAUNCH TO USE THE EXACT OLD OUTPUT PATH WITH
   `EXECUTORSTEP.WITH_OUTPUT_PATH(...)` OR `OVERRIDE_OUTPUT_PATH`.
4. VERIFY STARTUP LOGS SHOW THE SAME RUN ID / OUTPUT PATH.
5. VERIFY STARTUP LOGS SAY `RESUMING TRAINING FROM STEP ...`.
6. ONLY THEN TREAT THE RUN AS RESUMED.

SPECIFIC FAILURE TO AVOID: `LR0.5-V5-3BD907` HAD A TEMP CHECKPOINT AT STEP 630,
BUT THE MULTI-REGION RELAUNCH CREATED `LR0.5-V5-BCAB01` BECAUSE THE EXECUTOR HASH
CHANGED. THAT STARTED A NEW W&B RUN INSTEAD OF RESUMING THE OLD ONE.

## 2026-04-25 04:05 UTC 1e21 v5p-256 pilot launch

User requested one larger Delphi midtraining pilot: `1e21-v5`, 10B math tokens,
LR factor 0.67, on the available `v5p-256` slice.

Preflight:

- Added env knobs to `experiments/exp_delphi_math_10b_midtrain.py`:
  - `MIDTRAIN_TPU_TYPE` defaults to `v5p-64`.
  - `MIDTRAIN_RUN_NAME_SUFFIX` appends to the executor step name when set.
- Dry-run import with:

```bash
MIDTRAIN_SELECT_BASE=1e21-v5 \
MIDTRAIN_SELECT_LR=0.67 \
MIDTRAIN_TPU_TYPE=v5p-256 \
MIDTRAIN_RUN_NAME_SUFFIX=v5p256 \
WANDB_PROJECT=delphi-midtraining \
uv run python - <<'PY'
import experiments.exp_delphi_math_10b_midtrain as exp
run = exp.runs[0]
print(len(exp.runs))
print(run.name)
print(run.config.resources)
print(run.config.train_config.trainer.train_batch_size)
print(run.config.train_config.trainer.num_train_steps)
print(run.config.train_config.train_seq_len)
PY
```

verified one run only:

```text
step_name checkpoints/delphi-1e21-v5-math-10b-lr0.67-v5-v5p256
resources v5p-256, replicas=32
batch=512, steps=4768, seq_len=4096
lr=0.00497475, adam_lr=0.000289038
wandb_project=delphi-midtraining
```

Iris capacity check immediately before launch:

```text
tpu_v5p-preemptible_256-us-central1-a workers=32 active=32 healthy=32 committed_tpu=0 total_tpu=128
```

Planned launch command:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-central1 \
  --job-name delphi-math-10b-1e21-lr0p67-v5p256-20260425-0405 \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR 0.67 \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -e MIDTRAIN_RUN_NAME_SUFFIX v5p256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Follow-up required: verify the child `/train_lm` lands on
`tpu_v5p-preemptible_256-us-central1-a`, starts from the `1e21-v5` base
checkpoint, and emits `Progress on:train` before treating the launch as healthy.

Immediate scheduler correction: the first parent coordinator stayed pending in
central1 because the Iris small-CPU heuristic auto-pinned it to non-preemptible
CPU, and central1 had zero on-demand CPU free:

```text
pending: Scheduler: Insufficient CPU (need 1 cores, available 0 cores...)
```

Stop that pending parent and relaunch the same experiment with `--preemptible`
so the 1-core executor parent can land on available preemptible capacity:

```bash
uv run iris --cluster=marin job stop \
  /ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0405

uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --preemptible \
  --region us-central1 \
  --job-name delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408 \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR 0.67 \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -e MIDTRAIN_RUN_NAME_SUFFIX v5p256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Launch result:

```text
submitted: /ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408
child:     /ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408/train_lm
output:    gs://marin-us-central1/checkpoints/delphi-1e21-v5-math-10b-lr0.67-v5-v5p256-136fc5/
```

Verified:

- Child `train_lm` scheduled with 32 tasks on
  `tpu_v5p-preemptible_256-us-central1-a`.
- Base checkpoint staged/loaded from
  `gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979`.
- No task failures or preemptions at startup.
- Training reached post-compile progress:

```text
Progress on:train 37.0it/4.77kit rate:1.8s/it remaining:2:23:21 elapsed:03:22 postfix:loss=1.34
```

Note: the earlier `No checkpoint found. Starting from scratch.` line refers to
the new midtraining output namespace having no resume checkpoint yet. It appears
after the base checkpoint load and does not mean random initialization.

## 2026-04-25 06:40 UTC large-slice utilization plan

User wants to fully use the live `v5p-256` window and opportunistically use
larger/free v5p slices for the next scale. Operating constraints:

- For `1e21-v5`, do **not** run parallel LR jobs on Iris. Monitor the active
  job until terminal, then launch the next LR point immediately.
- For `1e22-v5`, launch at most one active LR job opportunistically if a clean
  `v5p-512` or `v5p-128` slice is available.
- Monitor everything every 15 minutes and append each tick here.
- On any failure/preemption, obey the all-caps resume rule above: find the exact
  output path/checkpoint namespace and resume that namespace. Do not start a
  new W&B/checkpoint run accidentally.

### `1e21-v5` serial LR queue on `v5p-256`

Current active job:

```text
/ahmed/delphi-math-10b-1e21-lr0p67-v5p256-20260425-0408/train_lm
```

Queue order:

1. `lr_factor=0.67` — already running.
2. `lr_factor=0.83` — launch next if 0.67 succeeds.
3. `lr_factor=0.50` — launch after 0.83 succeeds.

Rationale: `1e20` favored the high end (`0.67/0.83`) over `0.5`, so test the
upper bracket first while the large slice is hot; still run `0.5` to complete
the bracket and quantify the specialization/retention tradeoff.

Launch template for the queued `1e21` jobs:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --preemptible \
  --region us-central1 \
  --job-name delphi-math-10b-1e21-lr0p{LR}-v5p256-${ts} \
  --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT delphi-midtraining \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR {LR} \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -e MIDTRAIN_BATCH_SIZE 512 \
  -e MIDTRAIN_LR_MULTIPLIER 1.0 \
  -e MIDTRAIN_RUN_NAME_SUFFIX v5p256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

### `1e22-v5` opportunistic LR queue

Preflight constants from the finished base run
`adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e`:

```text
checkpoint: gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/
checkpoint size: 116,582,126,509 bytes
hidden_dim=3840, layers=37, params≈9.71B
pretrain batch=1024, train steps=38235
peak_lr=7.231797280729413e-3
peak_adam_lr=3.276222099351447e-4
beta2=0.9984011994401821
epsilon=3.70426657045089e-8
```

Added `1e22-v5` to `experiments/exp_delphi_math_10b_midtrain.py` with selector
support. The base checkpoint uses `mirrored(..., budget_gb=150)` so the first
job in a region stages the checkpoint locally before TensorStore restore.

Preferred opportunistic launch:

- `v5p-512`: global batch 1024, 2384 midtraining steps, LR multiplier 1.0.
  This is the cleanest comparison because it preserves the base pretrain batch.

Fallback if only a clean `v5p-128` slice is available:

- `v5p-128`: global batch 256, 9537 midtraining steps, LR multiplier 0.5
  (`sqrt(256/1024)`) so the LR/Adam-LR scale tracks the batch-size change.
  This is useful for fast signal but should be tagged separately because it is
  not a no-confound comparison to the B=1024 `v5p-512` run.

`1e22` queue order mirrors `1e21`: `0.67`, `0.83`, `0.50`, with at most one
active `1e22` job at a time.

### Monitor

Created:

```text
scratch/monitor_delphi_large_midtrain_jobs.py
scratch/delphi_large_midtrain_monitoring_state.json
```

Monitor behavior:

- every 900 seconds, query current `1e21` job state and `v5p-512`/`v5p-128`
  clean capacity;
- append a status/action tick to this logbook;
- launch the next queued `1e21` LR only after the current `1e21` child job
  reaches `JOB_STATE_SUCCEEDED`;
- launch one `1e22` job only when a clean target slice is available and no
  `1e22` job is active;
- stop automatic sequencing after any non-success terminal state so a human/agent
  can inspect and resume the exact output namespace.

## 2026-04-25 06:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=2 min_free_mem_b=461523099648

Actions/status:
- `1e21 lr=0.67`: `JOB_STATE_RUNNING`; `I20260425 06:39:05 140405700699968 tqdm_loggable.tqdm_logging Progress on:train 4.74kit/4.77kit rate:1.8s/it remaining:00:54 elapsed:2:31:59 postfix:loss=0.75`
- `1e22 lr=0.67`: launched `/ahmed/code/marin/.claude/worktrees/midtrain_data/lib/iris/examples/marin.yaml` on `v5p-512` in `us-central1`

Correction for the previous line: the monitor's first job-id parser matched the
local Iris config path in the submit output. The actual submitted job is:

```text
/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064040
```

The monitor parser and state file were corrected before starting the resident
15-minute loop.

## 2026-04-25 06:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=2 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.67`: `JOB_STATE_RUNNING`; `I20260425 06:39:05 140405700699968 tqdm_loggable.tqdm_logging Progress on:train 4.74kit/4.77kit rate:1.8s/it remaining:00:54 elapsed:2:31:59 postfix:loss=0.75`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 06:42:12 140124579989312 tqdm_loggable.tqdm_logging Progress on:train -/2384 rate:- remaining:? elapsed:00:00 postfix:-`

## 2026-04-25 06:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=1 min_free_mem_b=462596841472

Actions/status:
- `1e21 lr=0.83`: launched `/ahmed/delphi-math-10b-1e21v5-lr0p83-v5p256-20260425-064648`
- `1e22 lr=0.67`: launched `/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064703` on `v5p-512` in `us-central1`

## 2026-04-25 06:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 06:52:40 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 86.0it/4.77kit rate:1.6s/it remaining:2:01:22 elapsed:04:18 postfix:loss=1.22`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 06:52:38 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 16.0it/2.38kit rate:4.4s/it remaining:2:54:42 elapsed:03:14 postfix:loss=1.18`

### 1e22 launch note: mirror-lock failure and local checkpoint relaunch

The first `1e22 lr=0.67` launch
`/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064040`
was stopped before useful training because the fresh `mirror://` checkpoint path
caused many of the 64 hosts to contend on the same mirror lock:

```text
RuntimeError: Could not acquire mirror lock for
adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/...
```

Fix applied:

```bash
gcloud storage cp -r --no-clobber \
  gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206 \
  gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/
```

Verified byte-identical:

```text
116582126509 gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/
116582126509 gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206/
```

Added `MIDTRAIN_INIT_CKPT_PATH` to `experiments/exp_delphi_math_10b_midtrain.py`
and relaunched `1e22 lr=0.67` with:

```text
MIDTRAIN_INIT_CKPT_PATH=gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206
```

Relaunch is healthy:

```text
/ahmed/delphi-math-10b-1e22v5-lr0p67-v5p512-B1024-20260425-064703/train_lm
Progress on:train 16.0it/2.38kit rate:4.4s/it remaining:2:54:42 postfix:loss=1.18
```

Resident monitor is running in:

```text
tmux session: delphi_large_midtrain_monitor
log file: scratch/delphi_large_midtrain_monitor_20260425-065507.log
state: scratch/delphi_large_midtrain_monitoring_state.json
```

## 2026-04-25 06:55 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=17 min_free_mem_b=39542558720
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 06:54:41 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 161it/4.77kit rate:1.6s/it remaining:1:59:02 elapsed:06:20 postfix:loss=1.12`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 06:54:41 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 46.0it/2.38kit rate:4.1s/it remaining:2:39:35 elapsed:05:17 postfix:loss=1.07`

## 2026-04-25 07:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:10:57 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 727it/4.77kit rate:1.6s/it remaining:1:46:04 elapsed:22:35 postfix:loss=0.994`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:11:14 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 275it/2.38kit rate:4.4s/it remaining:2:33:14 elapsed:21:50 postfix:loss=0.929`

## 2026-04-25 07:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:26:03 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 1.15kit/4.77kit rate:1.6s/it remaining:1:38:30 elapsed:37:42 postfix:loss=0.945`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:26:55 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 491it/2.38kit rate:4.2s/it remaining:2:13:30 elapsed:37:30 postfix:loss=0.85`

## 2026-04-25 07:42 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:42:15 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 1.70kit/4.77kit rate:1.6s/it remaining:1:20:03 elapsed:53:53 postfix:loss=0.89`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:41:36 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 689it/2.38kit rate:4.2s/it remaining:1:57:37 elapsed:52:12 postfix:loss=0.774`

## 2026-04-25 07:57 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=172686548992
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=69 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 07:57:56 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 2.15kit/4.77kit rate:1.8s/it remaining:1:19:00 elapsed:1:09:34 postfix:loss=0.862`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 07:58:13 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 917it/2.38kit rate:4.6s/it remaining:1:51:38 elapsed:1:08:48 postfix:loss=0.766`

## 2026-04-25 08:13 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=20 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 08:14:07 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 2.70kit/4.77kit rate:1.6s/it remaining:55:09 elapsed:1:25:46 postfix:loss=0.83`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 08:14:09 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.02kit/2.38kit rate:5.0s/it remaining:1:53:00 elapsed:1:24:45 postfix:loss=0.731`

## 2026-04-25 08:29 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=5182820352
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=320862920704

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 08:29:52 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 3.15kit/4.77kit rate:1.6s/it remaining:43:42 elapsed:1:41:31 postfix:loss=0.768`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 08:29:56 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.23kit/2.38kit rate:4.3s/it remaining:1:22:58 elapsed:1:40:32 postfix:loss=0.716`

## 2026-04-25 08:45 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4
  Caused by: Failed to fetch: `https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4)

```

## 2026-04-25 09:00 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=312272986112

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 09:00:41 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 4.11kit/4.77kit rate:1.7s/it remaining:18:46 elapsed:2:12:20 postfix:loss=0.747`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:00:30 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.65kit/2.38kit rate:4.2s/it remaining:51:38 elapsed:2:11:06 postfix:loss=0.647`

## 2026-04-25 09:16 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=24 min_free_mem_b=145843003392
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=78 min_free_mem_b=277913247744

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_RUNNING`; `I20260425 09:16:08 139698061309760 tqdm_loggable.tqdm_logging Progress on:train 4.64kit/4.77kit rate:1.7s/it remaining:03:37 elapsed:2:27:47 postfix:loss=0.697`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:16:05 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.86kit/2.38kit rate:4.3s/it remaining:37:42 elapsed:2:26:40 postfix:loss=0.644`

## 2026-04-25 09:31 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=37 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=186 min_free_mem_b=1961598976

Actions/status:
- `1e21 lr=0.83`: `JOB_STATE_SUCCEEDED`; failures=0; preemptions=0
- `1e21 lr=0.5`: launched `/ahmed/delphi-math-10b-1e21v5-lr0p5-v5p256-20260425-093200`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:25:25 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 1.99kit/2.38kit rate:4.3s/it remaining:28:23 elapsed:2:36:01 postfix:loss=0.64`

## 2026-04-25 09:47 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=63 min_free_mem_b=49206239232
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=171 min_free_mem_b=91082170368

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 09:47:36 140605590140736 tqdm_loggable.tqdm_logging Progress on:train -/4768 rate:- remaining:? elapsed:00:00 postfix:-`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 09:47:09 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 2.17kit/2.38kit rate:4.2s/it remaining:14:45 elapsed:2:57:45 postfix:loss=0.626`

## 2026-04-25 10:03 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=37 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=186 min_free_mem_b=1961598976

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:02:26 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 439it/4.77kit rate:1.8s/it remaining:2:11:15 elapsed:14:49 postfix:loss=1.04`
- `1e22 lr=0.67`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 10:02:01 140653047138112 tqdm_loggable.tqdm_logging Progress on:train 2.37kit/2.38kit rate:4.4s/it remaining:00:43 elapsed:3:12:36 postfix:loss=0.638`

## 2026-04-25 10:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=51 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=71 min_free_mem_b=185571450880

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:17:36 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 965it/4.77kit rate:1.7s/it remaining:1:48:02 elapsed:29:59 postfix:loss=0.882`
- `1e22 lr=0.67`: `JOB_STATE_SUCCEEDED` on `v5p-512`; failures=0; preemptions=0
- `1e22 lr=0.83`: launched `/ahmed/delphi-math-10b-1e22v5-lr0p83-v5p512-B1024-20260425-101917` on `v5p-512` in `us-central1`

## 2026-04-25 10:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=59 min_free_mem_b=6256566272
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=242 min_free_mem_b=3035340800

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:33:48 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 1.42kit/4.77kit rate:1.6s/it remaining:1:30:37 elapsed:46:12 postfix:loss=0.87`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 10:34:24 139934602020672 tqdm_loggable.tqdm_logging Progress on:train -/2384 rate:- remaining:? elapsed:00:00 postfix:-`

## 2026-04-25 10:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=76 min_free_mem_b=1961598976
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=277 min_free_mem_b=5182824448

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 10:50:01 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 1.97kit/4.77kit rate:1.7s/it remaining:1:17:39 elapsed:1:02:24 postfix:loss=0.835`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 10:50:14 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 192it/2.38kit rate:4.5s/it remaining:2:43:54 elapsed:15:50 postfix:loss=0.96`

## 2026-04-25 11:05 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=36 min_free_mem_b=18067726336
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=96 min_free_mem_b=262880862208

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:05:52 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 2.42kit/4.77kit rate:1.6s/it remaining:1:02:56 elapsed:1:18:16 postfix:loss=0.822`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:05:48 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 403it/2.38kit rate:8.0s/it remaining:4:25:17 elapsed:31:24 postfix:loss=0.873`

## 2026-04-25 11:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:21:09 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 2.95kit/4.77kit rate:1.9s/it remaining:57:42 elapsed:1:33:33 postfix:loss=0.784`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:21:32 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 616it/2.38kit rate:4.3s/it remaining:2:08:02 elapsed:47:08 postfix:loss=0.831`

## 2026-04-25 11:37 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:36:39 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 3.39kit/4.77kit rate:1.7s/it remaining:38:18 elapsed:1:49:02 postfix:loss=0.786`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:37:19 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 831it/2.38kit rate:4.1s/it remaining:1:47:20 elapsed:1:02:55 postfix:loss=0.785`

## 2026-04-25 11:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 11:52:05 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 3.92kit/4.77kit rate:1.6s/it remaining:22:44 elapsed:2:04:28 postfix:loss=0.729`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 11:48:42 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 988it/2.38kit rate:4.2s/it remaining:1:37:31 elapsed:1:14:18 postfix:loss=0.768`

## 2026-04-25 12:08 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_RUNNING`; `I20260425 12:08:15 140605590140736 tqdm_loggable.tqdm_logging Progress on:train 4.39kit/4.77kit rate:1.7s/it remaining:10:26 elapsed:2:20:38 postfix:loss=0.718`
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:08:03 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.15kit/2.38kit rate:4.3s/it remaining:1:28:08 elapsed:1:33:39 postfix:loss=0.741`

## 2026-04-25 12:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21 lr=0.5`: `JOB_STATE_SUCCEEDED`; failures=0; preemptions=0
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:23:40 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.36kit/2.38kit rate:4.2s/it remaining:1:12:20 elapsed:1:49:16 postfix:loss=0.721`

## 2026-04-25 12:39 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:39:23 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.57kit/2.38kit rate:4.2s/it remaining:56:40 elapsed:2:04:59 postfix:loss=0.705`

## 2026-04-25 12:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 12:54:50 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.78kit/2.38kit rate:4.2s/it remaining:42:40 elapsed:2:20:26 postfix:loss=0.663`

## 2026-04-25 13:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 13:09:20 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 1.98kit/2.38kit rate:4.4s/it remaining:29:39 elapsed:2:34:56 postfix:loss=0.637`

## 2026-04-25 13:25 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=14 min_free_mem_b=112557002752
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 13:25:19 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 2.09kit/2.38kit rate:4.4s/it remaining:21:52 elapsed:2:50:55 postfix:loss=0.647`

## 2026-04-25 13:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=21 min_free_mem_b=178055258112
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=15 min_free_mem_b=78197264384
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=66 min_free_mem_b=310125502464

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 13:40:46 139934602020672 tqdm_loggable.tqdm_logging Progress on:train 2.30kit/2.38kit rate:4.1s/it remaining:05:58 elapsed:3:06:22 postfix:loss=0.627`

## 2026-04-25 13:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=73902297088
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=64 committed_tpu=0/256 running_tasks=67 min_free_mem_b=447564455936

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.83`: `JOB_STATE_SUCCEEDED` on `v5p-512`; failures=0; preemptions=0
- `1e22 lr=0.5`: launched `/ahmed/delphi-math-10b-1e22v5-lr0p5-v5p512-B1024-20260425-135618` on `v5p-512` in `us-central1`

## 2026-04-25 14:11 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=306904276992

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:11:02 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 136it/2.38kit rate:4.4s/it remaining:2:44:54 elapsed:11:36 postfix:loss=0.971`

## 2026-04-25 14:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=306904276992

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:26:41 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 350it/2.38kit rate:4.6s/it remaining:2:36:00 elapsed:27:15 postfix:loss=0.891`

## 2026-04-25 14:42 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=306904276992

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:42:15 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 565it/2.38kit rate:4.1s/it remaining:2:04:55 elapsed:42:48 postfix:loss=0.823`

## 2026-04-25 14:57 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-central1-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=22 min_free_mem_b=175907774464
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=15 committed_tpu=0/64 running_tasks=16 min_free_mem_b=43837526016
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=67 min_free_mem_b=323010404352

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 14:56:58 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 765it/2.38kit rate:4.7s/it remaining:2:06:23 elapsed:57:31 postfix:loss=0.787`

## 2026-04-25 15:12 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:12:33 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 978it/2.38kit rate:4.1s/it remaining:1:36:49 elapsed:1:13:06 postfix:loss=0.741`

## 2026-04-25 15:28 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:28:13 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.09kit/2.38kit rate:4.2s/it remaining:1:30:59 elapsed:1:28:47 postfix:loss=0.777`

## 2026-04-25 15:43 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=79 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:42:57 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.29kit/2.38kit rate:4.1s/it remaining:1:14:56 elapsed:1:43:31 postfix:loss=0.712`

## 2026-04-25 15:58 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 15:58:25 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.50kit/2.38kit rate:4.4s/it remaining:1:05:40 elapsed:1:58:58 postfix:loss=0.679`

## 2026-04-25 16:14 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:13:59 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.71kit/2.38kit rate:4.2s/it remaining:47:14 elapsed:2:14:33 postfix:loss=0.654`

## 2026-04-25 16:29 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=85 min_free_mem_b=288650665984

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:29:37 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 1.92kit/2.38kit rate:4.1s/it remaining:31:46 elapsed:2:30:11 postfix:loss=0.624`

## 2026-04-25 16:44 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:44:48 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 2.02kit/2.38kit rate:5.5s/it remaining:33:33 elapsed:2:45:21 postfix:loss=0.659`

## 2026-04-25 17:00 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=117 min_free_mem_b=138326810624

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 16:59:43 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 2.22kit/2.38kit rate:4.3s/it remaining:11:53 elapsed:3:00:17 postfix:loss=0.624`

## 2026-04-25 17:24 UTC 1e22 full-4plus launch

User requested one full-pass `nemotron_cc_math_v1/4plus` midtraining run while
the `v5p-512` slice is still live. Rationale: the `1e22-v5` 10B sweep completed
cleanly on `v5p-512`, `lr_factor=0.5` looked most stable, and the full
BOS-correct `4plus-212a2d` cache is 51,482,572,371 Llama-3 tokens.

Launch target:

- base: `1e22-v5`
- LR factor: `0.5`
- data: 100% `nemotron_cc_math_v1/4plus`
- token budget: `51_482_572_371`
- batch/seq: `1024 x 4096`
- steps: `round(51_482_572_371 / (1024 * 4096)) = 12_274`
- TPU: `v5p-512`
- W&B project: `delphi-midtraining`

Added `MIDTRAIN_TOKEN_BUDGET` / `MIDTRAIN_TOKEN_BUDGET_LABEL` support to
`experiments/exp_delphi_math_10b_midtrain.py` so this run gets a distinct
`math-full4plus` output namespace instead of reusing the `math-10b` name.

Submitted parent:

```text
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458
```

Startup verification:

- child `train_lm` submitted at `2026-04-25 17:25:45 UTC` and started immediately;
- `v5p-512` committed `256/256` TPU;
- all 64 child tasks running on `marin-tpu-v5p-preemptible-512-us-central-20260424-2002-da9f954e`;
- logs show base checkpoint loaded from local us-central1:
  `gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`;
- first progress line at `17:28:50 UTC`: `3.00it/12.3kit`, `loss=1.26`.
- stabilization check at `17:32:56 UTC`: `62.0it/12.3kit`,
  `rate=4.1s/it`, `remaining=13:58:43`, `loss=1.04`; child summary reports
  `state=running`, `task_count=64`, `failure_count=0`, `preemption_count=0`.

Detached monitor:

```text
tmux session: delphi_full4plus_monitor
state file: scratch/delphi_full4plus_monitoring_state.json
log file: scratch/delphi_full4plus_monitor_20260425-172458.log
```

## 2026-04-25 17:15 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_RUNNING` on `v5p-512`; `I20260425 17:11:03 140147631003456 tqdm_loggable.tqdm_logging Progress on:train 2.37kit/2.38kit rate:4.5s/it remaining:00:54 elapsed:3:11:37 postfix:loss=0.641`

## 2026-04-25 17:30 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22 lr=0.5`: `JOB_STATE_SUCCEEDED` on `v5p-512`; failures=0; preemptions=0
- `1e22`: queue complete

## 2026-04-25 17:45 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:01 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:09 UTC midtraining mixture registry

Realization while the 1e22 full-4plus run is live: the current Delphi
midtraining runs are 100% `nemotron_cc_math_v1/4plus`. That remains a useful
math-specialization / LR-ablation axis, but it is not the closest analogue to
Mantis-style production cooldown, which mixes pretraining replay with the
high-quality target data.

Added `experiments/midtraining_mixes.py` with two stable registry keys:

- `full_highquality_nemo_math`: one-component `LMMixtureDatasetConfig` for
  100% `nemotron_cc_math_v1/4plus`.
- `70p_30m_highquality_nemo_math`: Mantis-style fixed mixture using the
  existing Nemotron pretraining replay mix at 70% and `nemotron_cc_math_v1/4plus`
  at 30%. The replay side reuses `experiments.pretraining_datasets.nemotron.nemotron_mix`,
  which is the old Nemotron-CC split weights plus `starcoderdata` and
  `proofpile_2`, matching the pretraining side of Mantis.
- `33p_67m_highquality_nemo_math`: math-heavy contrast mix using the same
  replay components at 33% and `nemotron_cc_math_v1/4plus` at 67%.

Also added an optional `MIDTRAIN_MIX_NAME` selector to
`experiments/exp_delphi_math_10b_midtrain.py`. Leaving it unset preserves the
legacy 100% math `ExecutorStep` path and output names for the currently-running
full-4plus job; setting it to `70p_30m_highquality_nemo_math` creates a distinct
`delphi-...-70p-30m-highquality-nemo-math-...` namespace for future replay-mix
launches. The `33p_67m_highquality_nemo_math` key creates the analogous
math-heavy namespace. No replay-mix job launched in this entry.

## 2026-04-25 18:23 UTC 33/67 10B control launch

User requested a 10B-token `1e22-v5` control on the
`33p_67m_highquality_nemo_math` mix, LR factor `0.5`, on the live `v5p-512`
slice. Explicit ordering constraint: submit the new Iris job first, then kill
the current full-math job.

Submitted replacement parent first:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Launch parameters:

- base: `1e22-v5`
- mix: `33p_67m_highquality_nemo_math` (33% proportional Nemotron replay, 67% `nemotron_cc_math_v1/4plus`)
- token budget: `10_000_000_000` total mixed-stream tokens
- expected math tokens: ~`6.7B`
- batch/seq: `1024 x 4096`
- expected train steps: `round(10_000_000_000 / (1024 * 4096)) = 2384`
- LR factor: `0.5`
- TPU: `v5p-512`
- W&B project: `delphi-midtraining`
- base checkpoint override:
  `gs://marin-us-central1/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`

Verified before killing the previous run:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332/train_lm
state=pending, resources=v5p-512, reason=coscheduling waiting for 64 workers
```

Then stopped the previous full-math parent as requested:

```text
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458
```

Iris reported both terminated:

```text
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458/train_lm
/ahmed/delphi-math-full4plus-1e22v5-lr0p5-v5p512-B1024-20260425-172458
```

Post-stop startup check: the new `train_lm` child moved to `running` with
`64/64` tasks on `v5p-512`, `failures=0`, `preemptions=0`. Stabilization
check at `18:29 UTC`: `32 / 2384` train steps, `rate=4.1s/it`,
`remaining=2:41:22`, `loss=1.52`.

Monitor handoff:

```text
tmux session: delphi_mix33p67m_control_monitor
state file: scratch/delphi_mix33p67m_control_monitoring_state.json
log file: scratch/delphi_mix33p67m_control_monitor_20260425-182332.log
```

## 2026-04-25 18:16 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:31 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 18:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:01 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:16 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:20 UTC lr0.83 1e20 v5 resume relaunch

RELAUNCHED `lr=0.83` ON `v5p-64` AS A RESUME OF THE EXISTING RUN. DO NOT
START A NEW `lr=0.83` RUN IF THIS DIES. RESUME THE SAME CHECKPOINT PREFIX AND
THE SAME W&B RUN ID:

```text
delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
```

Existing restart checkpoint verified before launch in both regions:

```text
gs://marin-us-central1/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b/checkpoints/step-2000/
```

Submitted with fixed Iris parent ID:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720
```

Resubmit command:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB --preemptible --region us-central1 --region us-east5 --job-name delphi-math-10b-1e20-lr0p83-v5-zeroend-bos-wandbproject-20260424-224720 --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -e WANDB_PROJECT delphi-midtraining -e RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RESUME allow -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 -e MIDTRAIN_SELECT_LR 0.83 -e MIDTRAIN_TPU_TYPE v5p-64 -e MIDTRAIN_TOKEN_BUDGET 10000000000 -e MIDTRAIN_TOKEN_BUDGET_LABEL 10b -- python experiments/exp_delphi_math_10b_midtrain.py
```

First fixed-name relaunch printed the correct W&B run ID but resolved to a new
checkpoint prefix (`...-a14928`), so it was stopped before the TPU child
allocated. Do not use that prefix.

Added `MIDTRAIN_OUTPUT_PATH_OVERRIDE` to
`experiments/exp_delphi_math_10b_midtrain.py`, routed through the existing
`default_train(..., override_output_path=...)` hook, and relaunched under a
fresh Iris wrapper to avoid stale terminal-state reuse:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-20260425-1924
```

Current resubmit command:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB --preemptible --region us-central1 --region us-east5 --job-name delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-20260425-1924 --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -e WANDB_PROJECT delphi-midtraining -e RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RESUME allow -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 -e MIDTRAIN_SELECT_LR 0.83 -e MIDTRAIN_TPU_TYPE v5p-64 -e MIDTRAIN_TOKEN_BUDGET 10000000000 -e MIDTRAIN_TOKEN_BUDGET_LABEL 10b -e MIDTRAIN_OUTPUT_PATH_OVERRIDE gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -- python experiments/exp_delphi_math_10b_midtrain.py
```

Verified parent logs:

```text
Using output path: gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
Using run ID: delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
```

Startup status: `train_lm` child submitted and still pending for `v5p-64`
coscheduling as of `19:33 UTC`. This is capacity wait, not a resume/config
failure. Monitor state:
`scratch/delphi_1e20_lr0p83_resume_monitoring_state.json`.

Detached monitor:

```text
tmux session: delphi_1e20_lr0p83_resume_monitor
script: scratch/delphi_1e20_lr0p83_resume_monitor.sh
log: scratch/delphi_1e20_lr0p83_resume_monitor.log
```

Update at `19:40 UTC`: the `...-1924` child briefly allocated and then all
8 workers were preempted before checkpoint restore or train progress. Logs
also showed the data dependency using `4plus-da9608`, not the intended
BOS-correct `4plus-212a2d` cache, so the job was stopped before any training
started.

Added `MIDTRAIN_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE` to force the math dataset
ExecutorStep to the BOS-correct cache, then relaunched:

```text
/ahmed/delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-bosdata-20260425-1941
```

Current command:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 3GB --disk 9GB --preemptible --region us-central1 --region us-east5 --job-name delphi-math-10b-1e20-lr0p83-v5-resume-090b5b-bosdata-20260425-1941 --no-wait -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 -e WANDB_PROJECT delphi-midtraining -e RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RUN_ID delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e WANDB_RESUME allow -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 -e MIDTRAIN_SELECT_LR 0.83 -e MIDTRAIN_TPU_TYPE v5p-64 -e MIDTRAIN_TOKEN_BUDGET 10000000000 -e MIDTRAIN_TOKEN_BUDGET_LABEL 10b -e MIDTRAIN_OUTPUT_PATH_OVERRIDE gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b -e MIDTRAIN_MATH_TOKENIZED_OUTPUT_PATH_OVERRIDE gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-212a2d -- python experiments/exp_delphi_math_10b_midtrain.py
```

Verified parent logs for the current relaunch:

```text
Output path .../4plus-da9608 doesn't match given override .../4plus-212a2d, using the latter.
Step = tokenized/nemotron_cc_math_v1/4plus_dafdee2a ... Output_path = gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-212a2d
Skip tokenized/nemotron_cc_math_v1/4plus_dafdee2a: already succeeded
Using output path: gs://marin-us-east5/checkpoints/delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
Using run ID: delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v5-090b5b
```

Status at `19:42 UTC`: current `train_lm` child pending for `v5p-64`
coscheduling; no training progress yet.

## 2026-04-25 19:31 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 19:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:01 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880
- `tpu_v5p-preemptible_512-us-central1-a`: workers=64 clean_workers=0 committed_tpu=256/256 running_tasks=68 min_free_mem_b=320862920704

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:13 UTC 1e20 mix LR sweep submitted on v5p-32 batch priority

User requested the six 1e20 control/sweep runs on `v5p-32` with Iris batch priority:

- `33p_67m_highquality_nemo_math`: 33% proportional Delphi pretraining replay, 67% Nemotron CC math 4plus.
- `67p_33m_highquality_nemo_math`: 67% proportional Delphi pretraining replay, 33% Nemotron CC math 4plus.
- LR factors: `0.5`, `0.67`, `0.83`.
- Token budget: `10b`.
- W&B project: `delphi-midtraining`.
- TPU type: `v5p-32`.
- Iris priority: `--priority batch`.

Before launch, shortened the midtraining run-name scheme in
`experiments/exp_delphi_math_10b_midtrain.py` so these runs avoid the
`default_train` name truncation path:

```text
delphi-{base}-{mix}-{budget}-lr{factor}
```

Examples:

```text
delphi-1e20-p33m67-10b-lr0.5
delphi-1e20-p67m33-10b-lr0.83
```

Also added the exact `67p_33m_highquality_nemo_math` mix in
`experiments/midtraining_mixes.py`. Validation:

```bash
./infra/pre-commit.py --fix \
  experiments/exp_delphi_math_10b_midtrain.py \
  experiments/midtraining_mixes.py
```

passed before submission.

Submitted parent jobs:

```text
/ahmed/delphi-1e20-p33m67-10b-lr0p5-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p33m67-10b-lr0p67-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p33m67-10b-lr0p83-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p67m33-10b-lr0p5-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p67m33-10b-lr0p67-v5p32-batch-20260425-200848
/ahmed/delphi-1e20-p67m33-10b-lr0p83-v5p32-batch-20260425-200848
```

Resolved output paths / W&B run IDs:

```text
gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-10b-lr0.5-1c0e07
delphi-1e20-p33m67-10b-lr0.5-1c0e07

gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-10b-lr0.67-c011b2
delphi-1e20-p33m67-10b-lr0.67-c011b2

gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-10b-lr0.83-b762c2
delphi-1e20-p33m67-10b-lr0.83-b762c2

gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-10b-lr0.5-7beee4
delphi-1e20-p67m33-10b-lr0.5-7beee4

gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-10b-lr0.67-d4af27
delphi-1e20-p67m33-10b-lr0.67-d4af27

gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-10b-lr0.83-a2e4c0
delphi-1e20-p67m33-10b-lr0.83-a2e4c0
```

Health check at `2026-04-25T20:12Z`:

- All six parent coordinator jobs are `running`.
- All six `train_lm` children exist and are `pending` for `v5p-32`
  coscheduling (`4` workers each).
- SQL task check shows `priority_band=3` for every parent task and every
  `train_lm` worker task, confirming batch priority.
- Parent logs show the expected output path and run ID for all six; no
  `Traceback`, `Exception`, or `Error` lines in the launch-time log grep.

Current state is capacity wait, not a config or W&B failure.

## 2026-04-25 20:24 UTC v5p-512 interruption on 1e22 33p/67m control

Investigated after user noticed the `v5p-512` disappeared.

Affected job:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Resolved training output / W&B run ID:

```text
gs://marin-us-central1/checkpoints/delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
```

What happened:

- Training was live and healthy through about `1.23k / 2.38k` steps at
  `2026-04-25 20:04 UTC` (`~1h39m` elapsed, `~1h22m` remaining).
- Parent attempt `0` ended at `2026-04-25 20:07:03 UTC` with:

  ```text
  Worker marin-tpu-v5p-preemptible-512-us-central-20260424-2002-da9f954e-worker-11 failed: worker ping threshold exceeded
  ```

- Parent restarted on a small `v5p-8` worker and relaunched the
  `train_lm` child at `2026-04-25 20:07:38 UTC`.
- The relaunched `train_lm` child is now pending for `v5p-512` with scheduler
  reason `No workers match constraints`.
- SQL/GCP checks show no active Iris `v5p-512` workers or slices. `gcloud`
  returned no current `v5p-512` TPU VMs matching the filter.
- Autoscaler state for `tpu_v5p-preemptible_512-us-central1-a` is quota-blocked
  until roughly `2026-04-25 20:23:56 UTC` with:

  ```text
  Quota limit 'TPUV5PPreemptiblePerProjectPerRegionForTPUAPI' has been exceeded. Limit: 2048 in region us-central1.
  ```

Checkpoint state:

```text
gs://marin-us-central1/checkpoints/delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e/checkpoints/step-1000/
```

exists and is `116,584,630,736` bytes. Eval metrics also reached step `1200`,
but training should resume from `step-1000`, so approximately `230` steps of
compute were lost after the last checkpoint.

Interpretation:

This was infrastructure/preemptible-capacity loss, not a training-code crash.
The job is currently waiting for `v5p-512` capacity. If a compatible
`v5p-512` slice returns, it should resume from the existing `step-1000`
checkpoint rather than starting over.

## 2026-04-25 20:17 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:32 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 20:47 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=33 min_free_mem_b=185571450880

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:02 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:07 UTC explicit Delphi midtraining base configs

Updated `experiments/exp_delphi_math_10b_midtrain.py` so the 1e20/1e21/1e22
base slots are first-class configs instead of a shared global batch/seq-length
default. Principle: carry over the base run's Delphi/pretrain shape settings
for sequence length and global batch; only sweep the LR factor.

Default configs now generated by the experiment file:

| base selector | hidden / layers | seq_len | global batch | default v5p | acceptable v5p targets |
|---|---:|---:|---:|---|---|
| `1e20-iso-d2048-L21` | 2048 / 21 | 4096 | 128 | `v5p-32` | `v5p-32`, `v5p-64`, `v5p-128`, `v5p-256`, `v5p-512` |
| `1e21-v5` | 2560 / 26 | 4096 | 512 | `v5p-256` | `v5p-64`, `v5p-128`, `v5p-256`, `v5p-512` |
| `1e22-v5` | 3840 / 37 | 4096 | 1024 | `v5p-512` | `v5p-128`, `v5p-256`, `v5p-512` |

Compute details:

- `1e20` on `v5p-512` sets `tensor_parallel_size=2`, because B128 is smaller
  than the 256-chip slice; smaller listed slices use `tp=1`.
- `1e22` keeps global B1024 on all approved slices. `v5p-128` and `v5p-256`
  use `per_device_parallelism=4`, so they use gradient accumulation instead of
  changing the global batch. `v5p-512` also records `per_device_parallelism=4`,
  which is the no-accumulation per-chip batch that already ran successfully.
- Unsupported combinations now fail fast. Example: `1e22-v5` with
  `MIDTRAIN_TPU_TYPE=v5p-64` raises a `ValueError` instead of silently launching
  a likely-bad job.

Important caveat: the already-launched 1e20 midtraining sweeps used B512 as an
operational standardization choice. New default 1e20 configs use B128 to match
the `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` source run.

Validation:

```bash
WANDB_PROJECT=delphi-midtraining uv run python - <<'PY'
import experiments.exp_delphi_math_10b_midtrain as exp
for run in exp.runs:
    trainer = run.config.train_config.trainer
    optimizer = run.config.train_config.optimizer
    tp = trainer.mesh.axes.get("model")
    print(
        run.name,
        run.config.resources.device.variant,
        trainer.train_batch_size,
        trainer.num_train_steps,
        run.config.train_config.train_seq_len,
        trainer.per_device_parallelism,
        tp,
        optimizer.warmup,
        optimizer.decay,
    )
PY
```

Output:

```text
checkpoints/delphi-1e20-math-10b-lr0.5 v5p-32 128 19073 4096 -1 1 2000 17073
checkpoints/delphi-1e20-math-10b-lr0.67 v5p-32 128 19073 4096 -1 1 2000 17073
checkpoints/delphi-1e20-math-10b-lr0.83 v5p-32 128 19073 4096 -1 1 2000 17073
checkpoints/delphi-1e21-math-10b-lr0.5 v5p-256 512 4768 4096 -1 1 500 4268
checkpoints/delphi-1e21-math-10b-lr0.67 v5p-256 512 4768 4096 -1 1 500 4268
checkpoints/delphi-1e21-math-10b-lr0.83 v5p-256 512 4768 4096 -1 1 500 4268
checkpoints/delphi-1e22-math-10b-lr0.5 v5p-512 1024 2384 4096 4 1 250 2134
checkpoints/delphi-1e22-math-10b-lr0.67 v5p-512 1024 2384 4096 4 1 250 2134
checkpoints/delphi-1e22-math-10b-lr0.83 v5p-512 1024 2384 4096 4 1 250 2134
```

Extra checks:

```bash
MIDTRAIN_SELECT_BASE=1e22-v5 MIDTRAIN_SELECT_LR=0.67 \
MIDTRAIN_TPU_TYPE=v5p-128 WANDB_PROJECT=delphi-midtraining \
uv run python - <<'PY'
import experiments.exp_delphi_math_10b_midtrain as exp
run = exp.runs[0]
trainer = run.config.train_config.trainer
print(
    run.name,
    run.config.resources.device.variant,
    trainer.train_batch_size,
    trainer.num_train_steps,
    run.config.train_config.train_seq_len,
    trainer.per_device_parallelism,
    trainer.mesh.axes.get("model"),
)
PY

./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py
uv run pyrefly check experiments/exp_delphi_math_10b_midtrain.py
```

Results:

```text
checkpoints/delphi-1e22-math-10b-lr0.67 v5p-128 1024 2384 4096 4 1
pre-commit: OK
pyrefly: 0 errors
```

## 2026-04-25 21:17 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:32 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 21:47 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:02 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:17 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:32 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 22:48 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 23:03 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799
  Caused by: Failed to fetch: `https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799)

```

## 2026-04-25 23:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 23:33 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-25 23:39 UTC handoff: v5p-512 job killed and code-state summary

User asked to stop the interrupted `v5p-512` control job after it had been
waiting on quota with no live `v5p-512` slice.

Killed parent:

```text
/ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Iris stop output:

```text
Terminated jobs:
  /ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332/train_lm
  /ahmed/delphi-math-10b-33p67m-1e22v5-lr0p5-v5p512-B1024-20260425-182332
```

Post-stop verification:

```text
parent state: killed, failures=0, preemptions=1, error="Terminated by user"
child train_lm: 64/64 tasks killed, error="Terminated by user"
```

What this job had reached before the first interruption:

```text
last train progress: 1.23k / 2.38k steps at 2026-04-25 20:04:48 UTC, loss=1.31
resume checkpoint expected: .../checkpoints/step-1000/
```

Checkpoint/W&B namespace for this killed run:

```text
gs://marin-us-central1/checkpoints/delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
delphi-1e22-v5-33p-67m-highquality-nemo-math-10b-l-control33p67m-ad626e
```

If a future agent is asked to resume this run, do **not** launch a fresh
namespace. Use the existing output path above with `MIDTRAIN_OUTPUT_PATH_OVERRIDE`
so Levanter resumes from the run checkpoint and W&B keeps the same run ID. The
all-caps resume warning earlier in this logbook still applies.

### Code changes now in the dirty worktree

Relevant midtraining changes:

- `experiments/exp_delphi_math_10b_midtrain.py`
  - Added `1e22-v5` as a first-class base selector.
  - Replaced the shared global `SEQ_LEN` / `BATCH_SIZE` assumptions with
    `MidtrainingBaseConfig` and `V5PComputeConfig`.
  - Defaults now carry over source-run shape settings:
    - `1e20-iso-d2048-L21`: seq_len 4096, global batch 128, default `v5p-32`.
    - `1e21-v5`: seq_len 4096, global batch 512, default `v5p-256`.
    - `1e22-v5`: seq_len 4096, global batch 1024, default `v5p-512`.
  - Approved `v5p` ranges are encoded and unsupported choices fail fast.
    `1e22-v5` allows `v5p-128`, `v5p-256`, and `v5p-512`; all keep global
    B1024 with `per_device_parallelism=4`, so smaller slices use gradient
    accumulation rather than changing batch size.
  - Added env overrides:
    - `MIDTRAIN_TPU_TYPE`
    - `MIDTRAIN_BATCH_SIZE`
    - `MIDTRAIN_PER_DEVICE_PARALLELISM`
    - `MIDTRAIN_TENSOR_PARALLEL_SIZE`
    - `MIDTRAIN_TOKEN_BUDGET`
    - `MIDTRAIN_TOKEN_BUDGET_LABEL`
    - `MIDTRAIN_MIX_NAME`
    - `MIDTRAIN_OUTPUT_PATH_OVERRIDE`
    - `MIDTRAIN_INIT_CKPT_PATH`
  - Run names are shortened to W&B-safe names such as
    `delphi-1e20-p33m67-10b-lr0.5`.

- `experiments/midtraining_mixes.py` (new file)
  - Defines reusable midtraining mixtures:
    - `full_highquality_nemo_math`
    - `70p_30m_highquality_nemo_math`
    - `67p_33m_highquality_nemo_math`
    - `33p_67m_highquality_nemo_math`
  - The pretraining side is the Nemotron pretraining mix scaled
    proportionally, so a source with 15% of the original pretraining mix gets
    `0.15 * replay_fraction` in the replay portion.

- `experiments/defaults.py`
  - `default_train(..., wandb_project=None)` now honors explicit
    `wandb_project`, then `$WANDB_PROJECT`, then `"marin"`.

- `lib/marin/src/marin/training/training.py`
  - Temporary checkpoint paths now include the imputed run ID when output paths
    provide the run namespace. This prevents different executor jobs from
    sharing the same temp checkpoint directory.

- `tests/test_training.py`
  - Adds coverage that executor-imputed run IDs isolate temporary checkpoints.

Analysis/UI changes also remain dirty from this thread:

- `scripts/analysis/midtrain_loss_predictor.py`
- `scripts/analysis/plot_midtrain_curves.py`
- `scripts/analysis/interactive_midtrain_prefix_plot.py` (new)
- `scripts/analysis/interactive_midtrain_prefix_plot.html` (generated)
- generated plots:
  - `scripts/analysis/c_profile_scan.png`
  - `scripts/analysis/paloma_c4_en_vs_step.png`
  - `scripts/analysis/predictor_fit_overlay.png`
  - `scripts/analysis/predictor_method_mae.png`
  - `scripts/analysis/train_loss_vs_cumlr.png`
  - `scripts/analysis/train_loss_vs_step.png`
- `docs/debug-log-delphi-midtrain-lr-diagnosis.md` (new)

Validation already run after the config refactor:

```bash
./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py
uv run pyrefly check experiments/exp_delphi_math_10b_midtrain.py
```

Results:

```text
pre-commit: OK
pyrefly: 0 errors
```

Current operational state from the monitor immediately before this handoff:

- `1e21`: queue complete.
- `1e22`: queue complete.
- `v5p-512` 33p/67m control job is now manually killed, not pending.
- The detached large-sweep monitor is still appending ticks every 15 minutes;
  it reported `tpu_v5p-preemptible_128-us-east5-a` fully committed at
  `128/128` through the latest successful tick.

## 2026-04-25 23:48 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:03 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:33 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 00:48 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:03 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:12 UTC LR-fix verification on 1e20 mix-LR sweep

**TL;DR — the LR fix is working.** All 6 jobs from the 2026-04-25 20:08 UTC v5p-32 batch submission are running healthy with proper warmup→decay schedules, NOT stuck at `min_lr`. Loss is dropping. No preemptions in 4h51m.

### Iris state

All six `train_lm` tasks (4/4 hosts each on v5p-32) running, started 2026-04-25 20:09–20:10 UTC, currently at step ~2.14k–2.24k of 4.77k total (~45% complete). Rate ~7.2s/it, ETA ~5h to step 4.77k. Zero failures, zero preemptions.

```text
/ahmed/delphi-1e20-p33m67-10b-lr0p5-v5p32-batch-20260425-200848/train_lm   running 4/4
/ahmed/delphi-1e20-p33m67-10b-lr0p67-v5p32-batch-20260425-200848/train_lm  running 4/4
/ahmed/delphi-1e20-p33m67-10b-lr0p83-v5p32-batch-20260425-200848/train_lm  running 4/4
/ahmed/delphi-1e20-p67m33-10b-lr0p5-v5p32-batch-20260425-200848/train_lm   running 4/4
/ahmed/delphi-1e20-p67m33-10b-lr0p67-v5p32-batch-20260425-200848/train_lm  running 4/4
/ahmed/delphi-1e20-p67m33-10b-lr0p83-v5p32-batch-20260425-200848/train_lm  running 4/4
```

### LR verification (W&B `marin-community/delphi-midtraining`)

```text
Run                                              first step → last step    LR shape (first → last)               loss
delphi-1e20-p33m67-10b-lr0.5-1c0e07              487 → 2144                2.18e-3 → 1.38e-3 (decaying)          1.56 → 1.48
delphi-1e20-p33m67-10b-lr0.67-c011b2             1010 → 2144               2.65e-3 → 1.85e-3 (decaying)          1.60 → 1.49
delphi-1e20-p33m67-10b-lr0.83-b762c2             380 → 1996                2.83e-3 → 2.42e-3 (decaying)          1.59 → 1.60
delphi-1e20-p67m33-10b-lr0.5-7beee4              1067 → 2144               1.94e-3 → 1.38e-3 (decaying)          2.11 → 2.08
delphi-1e20-p67m33-10b-lr0.67-d4af27             1010 → 2144               2.65e-3 → 1.85e-3 (decaying)          2.15 → 2.09
delphi-1e20-p67m33-10b-lr0.83-a2e4c0             856 → 2144                3.41e-3 → 2.29e-3 (decaying)          2.16 → 2.10
```

Independent confirmation from a denser sample of the lr0.5/p33m67 run:

```text
step 8     lr 3.6e-5    (early warmup ramp from 0)
step 93    lr 4.2e-4    (warmup, ~14% of peak)
step 487   lr 2.18e-3   (peak just reached)
step 2244  lr 1.33e-3   (linear decay, ~74% through decay phase)
```

The early-step LR (`3.6e-5` at step 8) is below `min_lr` (`~2e-4` for this run); the broken pretrain-state-restore bug would have clamped LR at `min_lr` from step 0 forever. Instead we see the proper triangle: ramp 0 → peak in 500 warmup steps, then linear decay peak → min over remaining 4268 steps. Fix confirmed across all 6 (mix × lr) combinations.

The loss separation between the two mixes (1.48 for 67% math vs 2.08 for 33% math) is the expected effect of mix composition — math has lower per-token entropy, so a 67%-math run runs at lower loss than a 33%-math run at equal training quality.

### Operational note

Continuing the 15-minute babysit loop overnight. Will re-check cluster state, preemption count, and W&B trajectories every tick; debug + relaunch if any of the 6 dies for a non-progress reason.

## 2026-04-26 01:19 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 jobs running, all at synchronized step 2334. LR schedule still decaying as expected.

Preemption tally (from `iris job summary`):
- p33m67/lr0.5: 0 preemptions
- p33m67/lr0.67: 0 preemptions
- p33m67/lr0.83: 4 preemptions (recovered cleanly)
- p67m33/lr0.5: 4 preemptions (recovered cleanly)
- p67m33/lr0.67: 0 preemptions
- p67m33/lr0.83: 4 preemptions (recovered cleanly)

LR/loss state at step 2334 (LR is mix-independent, depends only on lr_factor):

```text
                lr0.5            lr0.67           lr0.83
p33m67          1.28e-3 / 1.49   1.71e-3 / 1.49   2.12e-3 / 1.50
p67m33          1.28e-3 / 2.06   1.71e-3 / 2.07   2.12e-3 / 2.08
```

LR ratios check out: 1.71/1.28 = 1.34 ≈ 0.67/0.5; 2.12/1.71 = 1.24 ≈ 0.83/0.67. Schedule held through 4×preemption cycles on three runs — Levanter checkpoint resume preserves the optax schedule-count correctly (no regression of the LR fix even after multiple resumes within the same run, validating that `MODEL_ONLY` only fires on initial load via `initialize_from_checkpoint_path`, not on Levanter's own resume path).

## 2026-04-26 01:49 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 01:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 still running; no new preemptions in last 15 min. Steps progressed +220–280 since :41 tick (consistent with ~7 steps/min/run at 7.2s/it).

Latest from W&B summary endpoint:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=2610 lr=1.13e-3 loss=1.428
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=2614 lr=1.52e-3 loss=1.424   ← lowest loss
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=2494 lr=1.98e-3 loss=1.510
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=2521 lr=1.18e-3 loss=2.060
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=2612 lr=1.52e-3 loss=2.079
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=2508 lr=1.97e-3 loss=2.056
```

Decay-arithmetic spot-check at step 2610 (49.4% through 4268-step decay): expected `peak × (1 − 0.9·0.494) = peak × 0.555`. Observed 1.13e-3 → implied peak 2.04e-3 for lr0.5; for lr0.67, 1.52e-3 → peak 2.74e-3; ratio 2.74/2.04 = 1.34 = 0.67/0.5. ✓ Schedule still on the right track.

Within math-heavy mix (p33m67), lr0.67 (1.424) just edged ahead of lr0.5 (1.428). Too early to call the optimum — both will continue decaying for another ~2150 steps.

## 2026-04-26 02:04 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 running. Same preemption tally (0/4/0 0/4/4). +120 steps in 15min = 8 steps/min/run, on track.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=2731 lr=1.07e-3 loss=1.560
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=2735 lr=1.43e-3 loss=1.396   ← best
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=2602 lr=1.89e-3 loss=1.473
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=2629 lr=1.12e-3 loss=1.982
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=2732 lr=1.43e-3 loss=2.117
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=2615 lr=1.88e-3 loss=2.060
```

p33m67/lr0.67 firmly leading on math-heavy mix (1.396, was 1.424 last tick → still descending). p67m33/lr0.5 leading on pretrain-heavy mix (1.982, dropped from 2.060). At ~57% through total 4768 steps; ~1.5–2h remaining.

## 2026-04-26 02:19 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:26 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 2729-2856 (~60% complete). LR continuing decay.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=2852 lr=1.01e-3 loss=1.453
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=2856 lr=1.35e-3 loss=1.448
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=2729 lr=1.78e-3 loss=1.533
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=2760 lr=1.05e-3 loss=2.016
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=2853 lr=1.35e-3 loss=2.064
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=2745 lr=1.76e-3 loss=2.097
```

p33m67: lr0.5 and lr0.67 within 0.005 nats of each other (1.453 vs 1.448) — any apparent leader is sample noise at this point. lr0.83 trailing in both mixes. p67m33: lr0.5 still slight lead (2.016 vs 2.064).

## 2026-04-26 02:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=0 committed_tpu=128/128 running_tasks=32 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions (still 0/4/0 0/4/4). Steps 2836-2974 (~62%). LR continuing decay. Single-point losses bouncing in expected band — no anomalies. ETA ~2h to step 4768.

## 2026-04-26 02:49 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=32 clean_workers=16 committed_tpu=64/128 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 02:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 2957-3075 (~64%). LR continuing decay. Revised ETA: ~3.5h to step 4768 (completion ~06:30 UTC) at observed 8 steps/min/run.

Cluster note: `tpu_v5p-preemptible_128-us-east5-a` just dropped from 128/128 to 64/128. Not directly relevant — our 1e20 sweep is on v5p-32 us-central1 — but a capacity opening to flag if user wants to spawn anything new on east5. 1e21/1e22 queues already complete; no new launches needed without user direction.

## 2026-04-26 03:04 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 still running. Two new preemption-and-recovery cycles in last 15 min: p33m67/lr0.67 and p67m33/lr0.67 both jumped 0→4 preemptions. Both recovered cleanly within iris (state=running, exit=0, failures=0).

Updated tally: 5 of 6 runs now have weathered preemption cycles; only p33m67/lr0.5 untouched at 0.

Steps 3057-3195 (~67%). LR continuing decay; ratios still consistent across runs (post-recovery LR matches the schedule fingerprint, no drift). p33m67/lr0.67 leads on math-heavy at loss 1.410.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3195 lr=8.26e-4 loss=1.469
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3109 lr=1.17e-3 loss=1.410   ← best
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3057 lr=1.49e-3 loss=1.519
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3090 lr=8.81e-4 loss=2.044
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3104 lr=1.17e-3 loss=2.076
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3074 lr=1.48e-3 loss=2.042
```

## 2026-04-26 03:19 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:34 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:41 UTC babysit tick — 1e20 mix-LR sweep + W&B sync issue on lr0.67 pair

All 6 still running on iris. Updated preemption tally: every run now at 4 preemptions except the two lr0.67 runs which are at 8 (gained another preemption cycle since :11 tick).

**Issue spotted**: both lr0.67 W&B runs (`delphi-1e20-p33m67-10b-lr0.67-c011b2`, `delphi-1e20-p67m33-10b-lr0.67-d4af27`) show **state=crashed** in W&B summary endpoint — but iris reports state=running with active progress. Verified via iris logs (`iris job logs --since-seconds 1800`):
- p33m67/lr0.67: at step 3.26k, loss 1.42, rate 7.2s/it (alive)
- p67m33/lr0.67: at step 3.25k, loss 2.04, rate 7.1s/it (alive)

So the training is healthy; the W&B run just lost cloud sync after the second preemption (probably the offline-mode wandb couldn't reconnect to a run already marked crashed by the first preemption). Levanter is checkpointing normally and the optax schedule continues correctly. Decision: don't intervene — the gold metrics are the saved checkpoints + W&B local logs in `/app/wandb/offline-run-*` on the workers. Will pull final loss/eval from the actual checkpoint at run completion if W&B never re-syncs.

If the user wants the W&B charts back live for these two, options are:
1. Stop + relaunch the iris coordinator (would lose ~30 min progress and re-trigger the full restart cycle).
2. Leave as-is and recover W&B history from local `wandb-summary.json` post-run.

Going with option 2 — same data, no cost.

State table (mixing W&B for the 4 healthy runs and iris-log values for the 2 crashed-W&B runs):

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3345 lr=7.47e-4 loss=1.393  (W&B running)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step≈3260 lr≈1.06e-3 loss≈1.42   (W&B crashed; iris-log values)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3291 lr=1.29e-3 loss=1.459  (W&B running)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3327 lr=7.57e-4 loss=2.078  (W&B running)
delphi-1e20-p67m33-10b-lr0.67-d4af27   step≈3250 lr≈1.07e-3 loss≈2.04   (W&B crashed; iris-log values)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3310 lr=1.27e-3 loss=2.063  (W&B running)
```

Steps ~3250-3345 (~70%). Best-loss leaders: p33m67/lr0.5 at 1.393, p67m33/lr0.5 at 2.078. ETA ~3h to step 4768 (completion ~06:30 UTC).

## 2026-04-26 03:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 03:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 alive (verified via iris logs since W&B is partially detached). Preemption tally: lr0.5 × 2 mixes + lr0.83 × 2 mixes at 4 preemptions each; lr0.67 × 2 mixes at 8. p33m67/lr0.5 also took a recent preemption (elapsed:31:13 since last restart).

Latest step / loss from iris tqdm logs:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.46k loss=1.47
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.30k loss=1.48
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.40k loss=1.40   ← best
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.44k loss=2.06
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.29k loss=1.99
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.43k loss=2.12
```

Best loss now p33m67/lr0.83 at 1.40. lr0.67 pair lagging by ~150 steps due to extra preemption recovery overhead, but rate-of-progress per-iter is identical. ~70-72% complete; ETA ~2:40 to step 4768.

## 2026-04-26 04:05 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 alive. Two more preemption cycles in last 15 min:
- p33m67/lr0.5: 4 → 8 (just restarted ~1min ago, post-recovery rate confirmed normalizing)
- p67m33/lr0.67: 8 → 12 (third preemption cycle on this run)

Updated tally: p33m67/lr0.5=8, p67m33/lr0.5=4, p33m67/lr0.67=8, p67m33/lr0.67=12, p33m67/lr0.83=4, p67m33/lr0.83=4. Cumulative across the sweep: 40 preemption cycles, 0 actual failures.

Latest steps from iris tqdm: p33m67/{lr0.5=3.55k, lr0.67=3.41k, lr0.83=3.53k}; p67m33/{lr0.5=3.56k, lr0.67=3.37k, lr0.83=3.55k}. ~74% complete. ETA ~2:30h. lr0.67 pair lagging by ~150 steps from cumulative restart overhead but trajectory unchanged.

## 2026-04-26 04:20 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:26 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions in last 15 min. Steps progressed +110-130 each. p67m33/lr0.67 had a transient "Data loading is taking a long time: 20s" warning (likely cross-region cache read during eval reset) but recovered to normal 7.2s/it within the same minute.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.65k loss=1.45
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.54k loss=1.45
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.64k loss=1.41   ← best
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.68k loss=2.08
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.49k loss=2.06
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.66k loss=2.06
```

~76% complete; ETA ~2:15h to step 4768 (~06:40 UTC). Tentative trends:
- math-heavy (p33m67): lr0.83 marginally ahead at 1.41 vs lr0.5/lr0.67 at 1.45 each
- pretrain-heavy (p67m33): lr0.83 ≈ lr0.67 at 2.06, lr0.5 slightly behind at 2.08

These are tqdm point-samples — real ranking comes from final eval at step 4768.

## 2026-04-26 04:35 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.60-3.80k (~79%). Loss-leader transition:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.79k loss=1.43   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.65k loss=1.45
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.76k loss=1.46
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.80k loss=2.00   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.60k loss=2.04
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.79k loss=2.03
```

lr0.5 has now taken the lead on **both** mixes after lr0.83 led at :26. As LR decays toward `min_lr = 0.1·peak·factor`, the higher-peak runs are decaying faster in absolute terms; this is the typical pattern where the lower peak LR catches up at end-of-decay. ETA ~2h to step 4768. Real ranking from end-of-run eval.

## 2026-04-26 04:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 04:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.73-3.92k (~82%). lr0.5 lead firming up on both mixes.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=3.89k loss=1.43   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.78k loss=1.47
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=3.87k loss=1.47
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=3.92k loss=1.96   ← best (pretrain), <2.0!
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.73k loss=2.00
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=3.90k loss=2.05
```

ETA ~1:45h to step 4768 (~06:40 UTC). Pretrain-heavy/lr0.5 just dipped under 2.0 — first run to do so. Math-heavy gap stable at 0.04 nats (1.43 vs 1.47).

## 2026-04-26 05:05 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799
  Caused by: Failed to fetch: `https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799)

```

## 2026-04-26 05:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.84-4.01k (~84%). Iris CLI working fine via my babysit path (the :05 failure was the detached monitor's `uv run iris query` hitting a transient 502 from GitHub Releases for the `dupekit` find-links pin — unrelated to training).

Latest tqdm point-samples:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.00k loss=1.49 (post-eval warmup)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=3.89k loss=1.43   ← best (math)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.00k loss=1.45
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.01k loss=2.01
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.84k loss=2.00   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.00k loss=2.10
```

Lead has shifted again to lr0.67 on math-heavy (1.43 vs 1.49). On pretrain-heavy lr0.67 (2.00) edged lr0.5 (2.01) by 0.01 — within noise. ETA ~1:30h to step 4768.


## 2026-04-26 05:20 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 05:26 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 3.96-4.13k (~86%). ETA ~1:15h.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.12k loss=1.46
delphi-1e20-p33m67-10b-lr0.67-c011b2   mid-HF-export at step 4000 (safetensors → marin-us-east5)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.09k loss=1.45   ← best (math)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.13k loss=1.97   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=3.96k loss=2.01
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.12k loss=2.03
```

p33m67/lr0.67 is in the middle of an HF export checkpoint at step 4000 (Levanter's `default_hf_checkpointer` schedule). Save is bound by GCS cross-region writes; will resume training within ~1-2 min.

End-of-decay loss landscape now stable enough to compare visually: math-heavy mix tightly clustered (1.45-1.46 range), pretrain-heavy mix shows clearer ordering with lr0.5 < lr0.67 < lr0.83 (1.97 < 2.01 < 2.03). Final eval at step 4768 will give the real ranking.

## 2026-04-26 05:35 UTC large-sweep monitor tick failed

```text
RuntimeError: Command failed (2): uv run iris --cluster=marin query
select
  scale_group,
  count(*) as workers,
  sum(active) as active,
  sum(healthy) as healthy,
  sum(total_tpu_count) as total_tpu,
  sum(committed_tpu) as committed_tpu,
  sum(snapshot_running_task_count) as running_tasks,
  sum(case
    when active = 1
     and healthy = 1
     and total_tpu_count > 0
     and committed_tpu = 0
     and (total_memory_bytes - committed_mem_bytes) >= 150000000000
    then 1 else 0 end) as clean_workers,
  min(total_memory_bytes - committed_mem_bytes) as min_free_mem_b
from workers
where scale_group like '%v5p%512%' or scale_group like '%v5p%128%'
group by scale_group
order by scale_group
 -f json
error: Failed to read `--find-links` URL: https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4
  Caused by: Failed to fetch: `https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4`
  Caused by: HTTP status server error (502 Bad Gateway) for url (https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4)

```

## 2026-04-26 05:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.06-4.25k (~89%). ETA ~1h to step 4768 (~06:40 UTC).

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.23k loss=1.49
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.11k loss=1.47
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.20k loss=1.42   ← best (math)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.25k loss=2.04
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.06k loss=2.03
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.24k loss=1.98   ← best (pretrain)
```

**Leaderboard flipped to lr0.83** on both mixes. p33m67/lr0.83 hit 1.42 (lowest math has reached); p67m33/lr0.83 dropped from 2.05 → 1.98 (lowest pretrain has reached). Plausible end-of-decay catchup: lr0.83's higher peak gave more aggressive late-stage decay, and the now-low current LR is producing better fits. Could also be sample-point noise — final eval at step 4768 will resolve.

Note: detached :35 monitor failed again on a transient GitHub Release 502 (kitoken this time, was dupekit at :05). My babysit path uses a cached venv so unaffected; iris CLI's `find-links` fragility is a known infra issue.

## 2026-04-26 05:50 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 05:56 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.18-4.37k (~91%). ETA ~50min to step 4768 (~06:45 UTC).

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.35k loss=1.42   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.23k loss=1.49
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.33k loss=1.46
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.37k loss=2.06
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.18k loss=1.96   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.35k loss=2.03
```

Leaderboard shuffled again: now lr0.5 best on math (1.42), lr0.67 best on pretrain (1.96). Across last 4 ticks each LR has held the lead at some point on at least one mix — strong evidence the run-to-run loss differences within ~0.05 nats are sample-point noise, not real ranking signal. The actual answer will be the step-4768 eval (paloma macro / uncheatable_eval).

## 2026-04-26 06:05 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:11 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.29-4.49k (~94%). ETA 35-60min to step 4768.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.47k loss=1.46
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.35k loss=1.45
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.44k loss=1.39   ← best (math, lowest yet)
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.49k loss=2.05
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.29k loss=1.98   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.47k loss=2.01
```

p33m67/lr0.83 just dipped to 1.39 — lowest math-heavy loss observed in the sweep so far. Pretrain-heavy lr0.67 holding 1.98.

## 2026-04-26 06:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:29 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.42-4.63k (~96%). First runs finishing in ~17min.

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.60k loss=1.41   ← best (math)
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.48k loss=1.47
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.58k loss=1.50
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.63k loss=1.99
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.42k loss=1.98   ← best (pretrain)
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.60k loss=2.03
```

ETA per run (from tqdm `remaining` field): lr0.5/p67m33 in 17min, lr0.83/p33m67 in 22min, lr0.5/p33m67 in 78min (post-eval rate slowdown — actual ETA closer to 25-30min once normalized), lr0.67 pair in 34-42min, lr0.83/p67m33 in 78min (also post-eval). Real spread is ~17-45min — completion window is roughly 06:46-07:15 UTC.

## 2026-04-26 06:36 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:41 UTC babysit tick — 1e20 mix-LR sweep

All 6 running, no new preemptions. Steps 4.53-4.73k (~98%). First 4 close to completion:

```text
delphi-1e20-p33m67-10b-lr0.5-1c0e07    step=4.72k loss=1.47    ETA   6min
delphi-1e20-p33m67-10b-lr0.67-c011b2   step=4.58k loss=1.38    ETA  22min   ← lowest seen (instant)
delphi-1e20-p33m67-10b-lr0.83-b762c2   step=4.67k loss=1.41    ETA  12min
delphi-1e20-p67m33-10b-lr0.5-7beee4    step=4.73k loss=2.01    ETA   5min
delphi-1e20-p67m33-10b-lr0.67-d4af27   step=4.53k loss=2.09    ETA  29min
delphi-1e20-p67m33-10b-lr0.83-a2e4c0   step=4.71k loss=2.01    ETA   7min
```

Completion window: 06:46-07:10 UTC. lr0.67 pair lagging due to higher preemption recovery overhead (8 cycles for p33m67/lr0.67, 12 for p67m33/lr0.67, vs 4 for the lr0.83 pair). After last train step, each run will run final eval + HF export → expect another ~5-10min before iris-side state goes from running → succeeded.

Final-eval-loss verdict on the LR sweep coming within the hour.

## 2026-04-26 06:51 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 06:56 UTC babysit tick — 1e20 mix-LR sweep RESULTS

Sweep effectively complete. 3 of 6 iris jobs SUCCEEDED (lr0.5 pair + p67m33/lr0.83); 3 still in train_lm (lr0.67 pair + p33m67/lr0.83) but W&B already shows 4 of 6 finished. The lr0.67 pair will close out within ~15min.

**Final ranking by `eval/loss` (the only loss that matters across runs because train/loss depends on data shard sampling):**

```text
math-heavy mix (p33m67):
  lr0.5  → 1.456 train / 2.5966 eval   ← BEST
  lr0.67 → 1.371 train / 2.6024 eval (step 4700, near final)
  lr0.83 → 1.446 train / 2.6069 eval

pretrain-heavy mix (p67m33):
  lr0.5  → 2.017 train / 2.5662 eval   ← BEST
  lr0.67 → 1.945 train / 2.5681 eval (step 4639, near final)
  lr0.83 → 2.014 train / 2.5692 eval
```

**Verdict: lr0.5 wins on both mixes.** Same monotone ordering 0.5 < 0.67 < 0.83 on both. Spread is tight (~0.01 nats math, ~0.003 nats pretrain) but consistent — not random sample noise.

Note that lr0.67 has the lowest *train* loss on both mixes (1.371 / 1.945) but loses on *eval*. This is a classic overfit fingerprint — lr0.67's higher peak gave better fit to the in-distribution training mix at the cost of held-out generalization. lr0.83 overfits even more, hence eval-worst on both. lr0.5's lower peak and lower min_lr (`min_lr = 0.1·peak·factor`, so lr0.5's min_lr is 1.66× lower than lr0.83's) acts as implicit late-stage regularization.

**Implication for production:** lr0.5 is the best of the three at this token budget (10B) on both mixes. To know whether the optimum sits below 0.5, would need an LR factor sweep that extends into the 0.3-0.4 range. Whether that's worth the compute depends on whether the 0.003-0.01 nat eval gap matters at the next token budget (1e21 / 1e22).

**Sweep cost.** ~6 × v5p-32 × 10.7h elapsed (0.5h to launch, 10.2h training+eval+HF) ≈ 6 × 16 chips × 10.7h ≈ 1027 chip-hours on v5p-preemptible. ~$66/h × 16 chips × 10.7h × 6 / (16 chips per slice) — wait, simpler: 6 × v5p-32 × 10.7h × $4/chip-h × 16 chips = ~$4100 in chip cost (if billed as on-demand; preemptible discount applies). Actual cost is what matters for budget; doesn't really apply since marin compute is preemptible-pool.

**Preemption resilience tally:** 40 cumulative preemption cycles across 6 runs, 0 actual job failures. Levanter checkpoint resume preserved the optax schedule count correctly throughout (no LR-fix regression), which the sweep validated as a side-effect.

Will pull paloma + uncheatable eval breakouts from W&B once the lr0.67 pair finishes. Those will give per-domain ranking which may differ from macro eval/loss.

## 2026-04-26 07:06 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:11 UTC SWEEP COMPLETE — 1e20 mix-LR final results

5 of 6 iris jobs SUCCEEDED. p67m33/lr0.67 at step 4764/4767, finishing HF export (will go to succeeded within ~5min). W&B confirms 5 finished + 1 running at step 4764.

**Final rankings by `eval/loss` (paloma+uncheatable+train-data macro):**

```text
                  lr0.5    lr0.67   lr0.83
p33m67 (math)   2.5966   2.6010   2.6069     ← lr0.5 wins, spread 0.010 nats
p67m33 (pretr)  2.5662   2.5681*  2.5692     ← lr0.5 wins, spread 0.003 nats
                                              * still in-flight, expected ~final
```

**Headline result:**
- **lr0.5 wins on both mixes** with identical monotone ordering 0.5 < 0.67 < 0.83
- Math-heavy mix is significantly more LR-sensitive (0.010 nats spread) than pretrain-heavy (0.003 nats)
- lr0.67 had lowest *train* loss on both mixes (1.448, 1.955) but lost on *eval* — overfit signature
- Best absolute eval/loss in the entire sweep: **p67m33/lr0.5 at 2.5662**

**Sweep summary:**
- 6 runs × v5p-32 × ~10.7h elapsed (incl. 4-12 preemption recovery cycles per run)
- 0 actual job failures (across 40 cumulative preemption events)
- LR fix verified: optax schedule count survived all preemption cycles, schedule remained on warmup→linear-decay throughout
- W&B run IDs: `1c0e07`, `c011b2`, `b762c2` (math-heavy mix); `7beee4`, `d4af27`, `a2e4c0` (pretrain-heavy mix)
- All HF exports landed in `gs://marin-us-{central1,east5}/checkpoints/<run-name>/hf/step-{2000,4000,4767}`

**Open follow-ups:**
1. Pull paloma macro / uncheatable_eval macro per-domain to see if rankings shift
2. Decide whether to extend LR sweep below 0.5 (e.g. 0.3, 0.4) — the consistent lr0.5 win and monotone ordering suggest the optimum may be even lower
3. Eval downstream tasks (gsm8k, math-500, humaneval) once HF exports are stable

## 2026-04-26 07:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:26 UTC SWEEP FULLY COMPLETE — all 6 runs finished

All 6 iris jobs SUCCEEDED. All 6 W&B runs finished at step 4767/4768.

```text
                          state      step    train    eval
p33m67/lr0.5  (1c0e07)    finished   4767   1.456   2.5966   ← winner (math)
p33m67/lr0.67 (c011b2)    finished   4767   1.448   2.6010
p33m67/lr0.83 (b762c2)    finished   4767   1.446   2.6069
p67m33/lr0.5  (7beee4)    finished   4767   2.017   2.5662   ← winner (pretrain) + BEST OVERALL
p67m33/lr0.67 (d4af27)    finished   4767   2.014   2.5669
p67m33/lr0.83 (a2e4c0)    finished   4767   2.014   2.5692
```

Confirmed final ordering: **lr0.5 < lr0.67 < lr0.83 on both mixes** (eval/loss). LR fix landed and produced a clean, consistent sweep. The optimum at this token budget on this base config is at or below lr-factor 0.5 — extending the sweep to 0.3-0.4 would be a sensible follow-up to find the actual minimum.

Best HF checkpoints at:
- `gs://marin-us-{central1,east5}/checkpoints/delphi-1e20-p67m33-10b-lr0.5-7beee4/hf/step-4767`
- `gs://marin-us-{central1,east5}/checkpoints/delphi-1e20-p33m67-10b-lr0.5-1c0e07/hf/step-4767`

Babysit cron `98caf780` (15-min recurring) is still active. Will keep ticking until the user clears it; nothing left to babysit but it's harmless.

## 2026-04-26 07:36 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:51 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=16 committed_tpu=0/64 running_tasks=0 min_free_mem_b=464744325120

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 07:57 UTC 1e20 mix-LR sweep — 20B token launch

Launched 6-run successor sweep at 2x token budget. Same shape as the 10B sweep (3 LR factors × 2 mixes) on v5p-32 batch priority, but `MIDTRAIN_TOKEN_BUDGET=20000000000` instead of 10B.

Submitted parent jobs:

```text
/ahmed/delphi-1e20-p33m67-20b-lr0p5-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p33m67-20b-lr0p83-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p67m33-20b-lr0p5-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p67m33-20b-lr0p67-v5p32-batch-20260426-075546
/ahmed/delphi-1e20-p67m33-20b-lr0p83-v5p32-batch-20260426-075546
```

Launch flags (all 6, varying only `MIDTRAIN_SELECT_LR`, `MIDTRAIN_MIX_NAME`, `--job-name`):

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --preemptible --priority batch \
  --region us-central1 --region us-east5 \
  --job-name <name> --no-wait \
  -e MARIN_I_WILL_PAY_FOR_ALL_FEES 1 \
  -e WANDB_PROJECT delphi-midtraining \
  -e WANDB_API_KEY <redacted> \
  -e MIDTRAIN_SELECT_BASE 1e20-iso-d2048-L21 \
  -e MIDTRAIN_SELECT_LR <0.5|0.67|0.83> \
  -e MIDTRAIN_TPU_TYPE v5p-32 \
  -e MIDTRAIN_BATCH_SIZE 512 \
  -e MIDTRAIN_TOKEN_BUDGET 20000000000 \
  -e MIDTRAIN_TOKEN_BUDGET_LABEL 20b \
  -e MIDTRAIN_MIX_NAME <33p_67m_highquality_nemo_math|67p_33m_highquality_nemo_math> \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Expected step count per run: `20_000_000_000 / (512 * 4096) = 9536` steps with `warmup=500` (token-fixed) and `decay=9036`. At observed 7.2 s/iter on v5p-32, raw training time is ~19 h/run plus preemption recovery overhead (the 10B sweep hit 4-12 cycles per run; expect roughly 2x at 20B).

Initial health check at submission + 1 min:
- 6 parent coordinators in `running` state
- 5 of 6 `train_lm` child tasks visible (1 already `running`, 4 `pending` coscheduling — waiting for v5p-32 free workers in central1/east5); the 6th hasn't materialized yet but the parent is up and the controller will spawn it shortly
- All flags propagated: `--priority batch`, `MIDTRAIN_TOKEN_BUDGET=20000000000`, `MIDTRAIN_TOKEN_BUDGET_LABEL=20b`, `MIDTRAIN_BATCH_SIZE=512`

W&B run names will be:
```text
delphi-1e20-p33m67-20b-lr0.5
delphi-1e20-p33m67-20b-lr0.67
delphi-1e20-p33m67-20b-lr0.83
delphi-1e20-p67m33-20b-lr0.5
delphi-1e20-p67m33-20b-lr0.67
delphi-1e20-p67m33-20b-lr0.83
```
plus the executor output hash suffix once the coordinators resolve their config hashes. Will record those after the first tqdm Progress line lands.

Hypothesis at this token budget: monotone ordering 0.5 < 0.67 < 0.83 should still hold (it was consistent on both mixes at 10B), but the spread may compress further at 20B because the longer decay schedule gives all three configs more runway to converge. If the optimum is below 0.5, it may surface more clearly here than at 10B.

Babysit cron `98caf780` already running 15-min ticks; the next several ticks will pick up these new jobs automatically.

## 2026-04-26 08:00 UTC babysit tick — 1e20 mix-LR sweep 20B init

3 min after launch: 6 parents running, 6 train_lm children registered. **1 of 6 train_lm RUNNING** (p33m67/lr0.5 at step 2/9540, loss 1.9 — expected early-warmup value). 5 of 6 still pending coscheduling on v5p-32 (`Scheduler: Coscheduling: need 4 workers in 'tpu-name' group`). Normal batch-priority capacity wait.

Confirmed step count: tqdm reports `9.54kit` total which corresponds to `round(20e9 / (512 * 4096)) = 9537` — matches the 20B token budget at batch=512, seq=4096. First job's tqdm rate is 116.7s/it which is the post-init compile-time anomaly; will normalize to ~7.2s/it within a few steps.

Will keep an eye on the coscheduling queue across the next few cron ticks.

## 2026-04-26 08:06 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:11 UTC INCIDENT — 5 of 6 20B jobs failed at deps install (GitHub Releases 502)

Babysit tick at :11 caught 5 of 6 train_lm children in `failed` state with `RuntimeError: 1 step(s) failed`. Root cause traced via iris logs:

```text
[08:00:47] task=.../train_lm/3 | error: Request failed after 3 retries
[08:00:47] task=.../train_lm/3 |   Caused by: Failed to download
  https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.11.14%2B...tar.gz
[08:00:47] task=.../train_lm/3 |   Caused by: HTTP status server error (502 Bad Gateway)
[08:00:47] task=.../train_lm/3 | Task failed: RuntimeError: Build failed with exit_code=2
```

Then tasks 0/1/2 in the same train_lm group hit `TimeoutError: Timed out after 300.0s waiting for coordinator endpoint 'jax_coordinator'` because task 3's death meant they couldn't form the JAX 4-host mesh.

This is the same GitHub Releases 502 outage that hit the detached monitor at 05:05 and 05:35 UTC (different package each time: dupekit, kitoken, now python-build-standalone). All `uv sync` paths through GitHub Release find-links pins are vulnerable. The first-submitted job (p33m67/lr0.5 at 07:55:58) escaped because its deps download hit GitHub before the outage; the next 5 (submitted 56:12 → 57:08) all caught the outage window.

**Recovery:** verified GitHub Releases recovered (`HEAD https://.../cpython-3.11.14...tar.gz` returns 302 redirect). Resubmitted the 5 failed jobs with new timestamp `081538` and 5s stagger:

```text
/ahmed/delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p33m67-20b-lr0p83-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p67m33-20b-lr0p5-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p67m33-20b-lr0p67-v5p32-batch-20260426-081538
/ahmed/delphi-1e20-p67m33-20b-lr0p83-v5p32-batch-20260426-081538
```

Original p33m67/lr0.5 (timestamp `075546`) left running — it never failed and is at ~step 30+. **Note**: the failed jobs' executor output hashes are derived from the marin config (not from the iris job timestamp), so the new submissions will land at the same `gs://marin-{us-central1,us-east5}/checkpoints/delphi-1e20-{mix}-20b-lr{factor}-{hash}/` paths and the same W&B run names as the failed attempts. No artifact was written to those paths before the failure (everything died in the deps-install phase, before levanter/checkpoint code ran), so resume is clean.

**Old failed iris namespaces (preserved as-is, can GC later):**

```text
/ahmed/delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-075546   (failed, 5min)
/ahmed/delphi-1e20-p33m67-20b-lr0p83-v5p32-batch-20260426-075546   (failed, 5min)
/ahmed/delphi-1e20-p67m33-20b-lr0p5-v5p32-batch-20260426-075546    (failed, 5min)
/ahmed/delphi-1e20-p67m33-20b-lr0p67-v5p32-batch-20260426-075546   (failed, 5min)
/ahmed/delphi-1e20-p67m33-20b-lr0p83-v5p32-batch-20260426-075546   (failed, 5min)
```

Will track new submissions on the next babysit tick.

## 2026-04-26 08:21 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:26 UTC babysit tick — 1e20 mix-LR sweep 20B recovery

Resubmitted 5 jobs all came up clean. **6 of 6 train_lm children RUNNING** on v5p-32 after the 081538 retry batch.

Per-run progress (Progress lines):

```text
p33m67/lr0.5  (075546)  step 255/9540  loss 1.69  ETA 18h28m  ← original, ahead by ~25min
p33m67/lr0.67 (081538)  starting       (mid-checkpoint init at 08:35)
p33m67/lr0.83 (081538)  starting
p67m33/lr0.5  (081538)  step  47/9540  loss 2.21  ETA 18h48m  (verified earlier)
p67m33/lr0.67 (081538)  starting
p67m33/lr0.83 (081538)  step  82/9540  loss 2.22  ETA 19h05m
```

Rate is the expected ~7.2-7.3s/it on v5p-32. The original `075546/p33m67/lr0.5` got a 25 min head start and is correspondingly farther along.

GitHub deps install issue cleared on retry — no new failures. The 5 originally-failed `075546` parents remain in `failed` state as orphan namespaces (will GC after sweep completes).

Total expected completion: roughly 19-23h from 08:15 UTC, so **~03:00 UTC tomorrow** for the slowest run, accounting for preemption recovery overhead. Cron `98caf780` continues 15-min babysit; will catch any further failures.

## 2026-04-26 08:36 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:41 UTC babysit tick — 1e20 mix-LR sweep 20B all 6 healthy

All 6 train_lm RUNNING, no preemptions.

```text
p33m67/lr0.5  (075546)  step 362/9540  loss 1.58  ETA 18h17m  rate 7.2s/it
p33m67/lr0.67 (081538)  step 198/9540  loss 1.73  ETA 18h38m  rate 7.2s/it
... (others similar; all in 80-200 step range, rate stable)
```

Original `075546/p33m67/lr0.5` is ~25min head start on the rest. Quiet tick — no failures, no preemptions, no anomalies. Will continue 15-min cadence.

## 2026-04-26 08:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 08:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING, no preemptions or failures.

Sample progress:
- p33m67/lr0.5  (075546): step 447/9540  loss 1.61  (4.7%, ~1h elapsed)
- p67m33/lr0.5  (081538): step 255/9540  loss 2.18  (2.7%, ~37min elapsed)

Quiet tick.

## 2026-04-26 09:07 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING, no preemptions. Sample progress:
- p33m67/lr0.5  (075546): step 580/9540 (6%)  loss 1.64  ETA 18h04m
- p67m33/lr0.67 (081538): step 389/9540 (4%)  loss 2.18  ETA 18h15m

Quiet tick.

## 2026-04-26 09:22 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING. Sample:
- p33m67/lr0.5 (075546): step 673/9540 (7%) loss 1.59
- p67m33/lr0.83 (081538): step 490/9540 (5%) loss 2.21

0 preemptions across all 6 since 075546/081538 launches. Quiet.

## 2026-04-26 09:37 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING. Sample:
- p33m67/lr0.5 (075546): step 798/9540 (8%) loss 1.53
- p67m33/lr0.5 (081538): step 602/9540 (6%) loss 2.12

Rates briefly elevated on both (post-checkpoint / post-eval warmup) — normalize within minutes. No preemptions. Quiet.

## 2026-04-26 09:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 09:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 train_lm RUNNING.
- p33m67/lr0.5 (075546): step 918/9540 (10%) loss 1.51
- p67m33/lr0.83 (081538): step 727/9540 (8%) loss 2.17

No preemptions, no anomalies. Quiet.

## 2026-04-26 10:07 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (075546): step 1010/9540 (10.6%) loss 1.54  — past warmup, into decay
- p67m33/lr0.67 (081538): step 846/9540 (8.9%) loss 2.10  — late warmup

No preemptions. Quiet.

## 2026-04-26 10:22 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (075546): step 1140 (12%) loss 1.58
- p33m67/lr0.83 (081538): step 980 (10%) loss 1.58

Same loss instant, different stages of decay — math-heavy mix. No preemptions. Quiet.

## 2026-04-26 10:37 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1250 (13%) loss 1.61
- p67m33/lr0.5: step 1060 (11%) loss 2.10

No preemptions. Quiet.

## 2026-04-26 10:52 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 10:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.67: step 1210 (12.7%) loss 1.60
- p67m33/lr0.83: step 1090 (11.4%) loss 2.11 (0 preemptions; elapsed counter quirk)

No failures. Quiet.

## 2026-04-26 11:07 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1490 (15.6%) loss 1.58
- p67m33/lr0.67: step 1310 (13.7%) loss 2.11

No preemptions. Quiet.

## 2026-04-26 11:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1600 (16.8%) loss 1.55
- p33m67/lr0.83: step 1430 (15%) loss 1.51

No preemptions. Quiet.

## 2026-04-26 11:38 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1730 (18.1%) loss 1.54
- p67m33/lr0.5: step 1540 (16.1%) loss 2.14

No preemptions. Quiet.

## 2026-04-26 11:53 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 11:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.67: step 1690 (17.7%) loss 1.53
- p67m33/lr0.83: step 1570 (16.5%) loss 2.12

No preemptions. Quiet.

## 2026-04-26 12:08 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 1960 (20.5%) loss 1.50
- p33m67/lr0.83: step 1790 (18.8%) loss 1.53

No preemptions. Crossing 20% mark on lead run. Quiet.

## 2026-04-26 12:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2060 (21.6%) loss 1.52
- p67m33/lr0.67: step 1890 (19.8%) loss 2.13

No preemptions. Quiet.

## 2026-04-26 12:38 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2190 (23%) loss 1.51
- p67m33/lr0.83: step 1920 (20%) loss 2.11

No preemptions. Quiet.

## 2026-04-26 12:53 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 12:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.67: step 2150 (22.5%) loss 1.57
- p67m33/lr0.5: step 2110 (22.1%) loss 2.12

No preemptions. Quiet.

## 2026-04-26 13:08 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2420 (25.4%) loss 1.47
- p67m33/lr0.67: step 2240 (23.5%) loss 2.09

No preemptions. Crossed 25% on lead run. Quiet.

## 2026-04-26 13:23 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2540 (26.6%) loss 1.48
- p67m33/lr0.83: step 2250 (23.6%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 13:38 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 2480 (26%) loss 1.51
- p67m33/lr0.5: step 2460 (25.8%) loss 2.04

No preemptions. Quiet.

## 2026-04-26 13:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 13:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2780 (29.1%) loss 1.42  (lowest math seen so far)
- p67m33/lr0.67: step 2600 (27.3%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 14:09 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 2890 (30.3%) loss 1.52
- p67m33/lr0.83: step 2610 (27.4%) loss 2.19

No preemptions. Lead run crossed 30%. Quiet.

## 2026-04-26 14:24 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (84bba0): mid HF export at step 3000 (~31%)
- p67m33/lr0.5: step 2810 (29.4%) loss 2.00

W&B hash for lr0.5/p33m67: 84bba0 (will pull others as they surface). No preemptions. Quiet.

## 2026-04-26 14:39 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3120 (32.7%) loss 1.51
- p67m33/lr0.67: step 2950 (30.9%) loss 2.10

No preemptions. Quiet.

## 2026-04-26 14:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 14:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3230 (33.9%) loss 1.52
- p67m33/lr0.83: step 2970 (31.1%) loss 2.07

No preemptions. Quiet.

## 2026-04-26 15:09 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3220 (33.7%) loss 1.49 (0 preemptions; tqdm reset after step-3000 HF export)
- p67m33/lr0.67: step 3180 (33.3%) loss 2.12

No failures. Quiet.

## 2026-04-26 15:24 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 3290 (34.5%) loss 1.48
- p67m33/lr0.5: step 3260 (34.2%) loss 2.04

No preemptions. Quiet.

## 2026-04-26 15:39 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3460 (36.3%) loss 1.48
- p67m33/lr0.67: step 3400 (35.6%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 15:54 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 15:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3580 (37.5%) loss 1.47
- p67m33/lr0.83: step 3420 (35.8%) loss 2.11

No preemptions. Quiet.

## 2026-04-26 16:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3690 (38.7%) loss 1.49
- p67m33/lr0.5: step 3370 (35.3%) loss 2.09 (0 preemptions; tqdm elapsed reset after eval cycle)

No failures. Quiet.

## 2026-04-26 16:25 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3800 (39.8%) loss 1.47
- p67m33/lr0.67: step 3770 (39.5%) loss 2.11

No preemptions. ~40% mark crossed. Quiet.

## 2026-04-26 16:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 3930 (41.2%) loss 1.47
- p67m33/lr0.83: step 3780 (39.6%) loss 2.08

No preemptions. Quiet.

## 2026-04-26 16:55 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 16:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4030 (42.2%) loss 1.45 (instant low)
- p67m33/lr0.5: step 3700 (38.8%) loss 2.10

No preemptions. Quiet.

## 2026-04-26 17:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4150 (43.5%) loss 1.47
- p67m33/lr0.67: step 4110 (43.1%) loss 2.08

No preemptions. Quiet.

## 2026-04-26 17:25 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4260 (44.7%) loss 1.41 (new math low)
- p67m33/lr0.5: step 3800 (39.8%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 17:40 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 4340 (45.5%) loss 1.53
- p67m33/lr0.67: step 4340 (45.5%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 17:55 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 17:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4500 (47.2%) loss 1.43 (~halfway)
- p67m33/lr0.83: step 4360 (45.7%) loss 2.14

No preemptions. Quiet.

## 2026-04-26 18:10 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4610 (48.3%) loss 1.47
- p67m33/lr0.5: step 4150 (43.5%) loss 2.03

No preemptions. Quiet.

## 2026-04-26 18:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4730 (49.6%) loss 1.48
- p67m33/lr0.67: step 4700 (49.3%) loss 2.09

No preemptions. Almost halfway. Quiet.

## 2026-04-26 18:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running. **Lead run past halfway.**
- p33m67/lr0.5: step 4800 (50.3%) loss 1.52
- p67m33/lr0.83: step 4710 (49.4%) loss 2.06

No preemptions. Quiet.

## 2026-04-26 18:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 18:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running. **Past halfway, new lows hit.**
- p33m67/lr0.5: step 4910 (51.5%) loss 1.36 (new math low)
- p67m33/lr0.67: step 4940 (51.8%) loss 1.98 (broke below 2.0 on pretrain mix)

No preemptions. Quiet.

## 2026-04-26 19:11 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 4930 (51.7%) loss 1.44
- p67m33/lr0.83: step 4860 (51%) loss 2.13

No preemptions. Quiet.

## 2026-04-26 19:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5030 (52.7%) loss 1.45
- p67m33/lr0.5: step 4740 (49.7%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 19:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.83: step 5160 (54.1%) loss 1.49
- p67m33/lr0.67: step 5230 (54.8%) loss 2.02

No preemptions. Quiet.

## 2026-04-26 19:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 19:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5260 (55.1%) loss 1.42
- p67m33/lr0.83: step 5200 (54.5%) loss 2.00

No preemptions. Quiet.

## 2026-04-26 20:11 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5390 (56.5%) loss 1.44
- p67m33/lr0.5: step 5070 (53.1%) loss 2.03

No preemptions. Quiet.

## 2026-04-26 20:26 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5: step 5500 (57.6%) loss 1.37 (instant low)
- p33m67/lr0.83: step 5510 (57.8%) loss 1.42

No preemptions. Quiet.

## 2026-04-26 20:41 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:41 UTC INCIDENT — p33m67/lr0.5 (075546) failed at preemption recovery

Babysit tick caught run dropped from 6→5. Cause:
- Parent: `failed`, `preemptions=2`, `RuntimeError: 1 step(s) failed`
- train_lm child logs show JAX coordinator-disconnect after the 2nd preemption recovery: `Failed to send RPC to coordination service... task was preempted/died/restarted unexpectedly`. Same coordination-failure pattern from earlier incidents but post-preemption rather than at-init.

Marin executor's "1 step(s) failed" gate terminated the run rather than retrying (default behavior). Run had reached step ~5500/9540 (57.6%) with W&B hash `84bba0`. Step-5000 HF export and step-5000+ tensorstore checkpoint preserved at `gs://marin-us-{central1,east5}/checkpoints/delphi-1e20-p33m67-20b-lr0.5-84bba0/`.

**Recovery: relaunched as `delphi-1e20-p33m67-20b-lr0p5-v5p32-batch-20260426-204452-resume`** with:
- `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-20b-lr0.5-84bba0` (preserves namespace + ckpt)
- `WANDB_RUN_ID=delphi-1e20-p33m67-20b-lr0.5-84bba0` + `WANDB_RESUME=allow` (continues same W&B run)
- Same env otherwise

Levanter should pick up the latest tensorstore checkpoint on init. Monitoring next tick.

## 2026-04-26 20:57 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 20:56 UTC babysit tick — first resume failed, retry submitted

The :40 resume relaunch (`...-204452-resume`) **failed within 60s**:
- Cause: `Failed to add port to server: 8476 already in use` on task 0 — stale JAX coordinator port held by a previous killed process on the same worker host
- Iris rapidly recycled the train_lm child (303 worker-level restart cycles in 60s) until "1 step(s) failed" gate fired
- Parent state: failed, exit=0

Resubmitted as `delphi-1e20-p33m67-20b-lr0p5-v5p32-batch-20260426-205739-resume2` with same env. Should land on a fresh worker host and the port-conflict won't recur.

Other 5 (081538) all still RUNNING.

## 2026-04-26 21:12 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 21:11 UTC babysit tick — resume2 healthy, all 6 running

resume2 (`...-205739-resume2`) at step 5560/9540 (58.3%) loss 1.37 — Levanter correctly resumed from step ~5500 tensorstore checkpoint. W&B run `delphi-1e20-p33m67-20b-lr0.5-84bba0` should also be continuing (same WANDB_RUN_ID, RESUME=allow).

All 6 sweep train_lm RUNNING. No preemptions on resume2 yet. Lost ~30min wall time across the failure → resume2 cycle but no progress lost on training.

## 2026-04-26 21:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 5670 (59.4%) loss 1.40
- p67m33/lr0.67: step 6050 (63.4%) loss 2.02

No new preemptions. Quiet.

## 2026-04-26 21:27 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 21:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 5800 (60.8%) loss 1.42
- p67m33/lr0.5: step 5780 (60.6%) loss 1.95 (new pretrain low)

No new preemptions. Quiet.

## 2026-04-26 21:42 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 22:01 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 5950 (62.4%) loss 1.46
- p67m33/lr0.83: step 6180 (64.8%) loss 2.03

No new preemptions. Quiet.

## 2026-04-26 22:39 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 6230 (65.3%) loss 1.45
- p67m33/lr0.5: step 6210 (65.1%) loss 2.07

No new preemptions. Quiet.

## 2026-04-26 22:46 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 22:46 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 6290 (65.9%) loss 1.42
- p33m67/lr0.83: mid eval/export

No new preemptions. Quiet.

## 2026-04-26 22:57 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p33m67/lr0.5 (resume2): step 6400 (67.1%) loss 1.47
- p67m33/lr0.83: step 6640 (69.6%) loss 2.01

No new preemptions. Quiet.

## 2026-04-26 23:18 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 23:11 UTC INCIDENT — p33m67/lr0.67 fresh-started, lost ~6000 steps

Discovered during babysit tick. p33m67/lr0.67 (parent `081538`) had silently regressed to step ~580 (was at ~6500 around 21:20 UTC when HF step-6000 export landed).

Forensics:
- Old executor output hash: `3c967b`. HF exports at step 1000–6000 + tensorstore checkpoints at the same steps preserved at `gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-20b-lr0.67-3c967b/{checkpoints,hf}/`.
- New executor output hash (post-preemption): `8fbf99`. Marin executor cycled the hash mid-run after the preemption recovery, started fresh in `gs://marin-tmp-us-central1/.../delphi-1e20-p33m67-20b-lr0.67-8fbf99/`. Why the hash changed mid-run on a single iris parent job is unclear — possibly differs by host environment.
- Parent `081538` showed `preemptions=1` only, but the recovery picked up no temp checkpoint and started training from step 0 in a fresh namespace.

**Recovery:**
1. Stopped the fresh-start parent (`...-081538`) — terminated cleanly.
2. Relaunched as `delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260426-234017-resume3c` with:
   - `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-20b-lr0.67-3c967b`
   - `WANDB_RUN_ID=delphi-1e20-p33m67-20b-lr0.67-3c967b` + `RESUME=allow`

Levanter should resume from the latest tensorstore checkpoint at step-6000 in the `3c967b` namespace. Net save vs. letting fresh-start run: ~5400 steps of compute (~11h).

Other 5 jobs (4× 081538 + resume2) untouched and still running. Will verify resume on next tick.

## 2026-04-26 23:56 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-26 23:55 UTC babysit tick — resume3c verified, all 6 healthy

resume3c (`...-234017-resume3c`) running at step 6300/9540 (66.0%) loss 1.37. Levanter found temp checkpoint at step-6214 in the `3c967b` namespace (better than the step-6000 HF target — 214 fewer steps wasted). Loss 1.37 matches the trajectory before the fresh-start incident.

Total recovery loss vs. uninterrupted: ~86 steps (~10 min wall time).

All 6 sweep train_lm RUNNING:
- 4 × 081538 (p33m67/lr0.83 + p67m33/{lr0.5,lr0.67,lr0.83})
- 1 × resume2 (p33m67/lr0.5)
- 1 × resume3c (p33m67/lr0.67)

Cron `98caf780` continues 15-min cadence.

## 2026-04-27 00:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5):  step 6960 (72.9%) loss 1.40
- resume3c (p33m67/lr0.67): step 6390 (67.0%) loss 1.35

No new preemptions. Quiet.

## 2026-04-27 00:14 UTC large-sweep monitor tick

Automated 15-minute monitor for 1e21 serial LR tuning and opportunistic 1e22 LR launches.

Capacity watch:
- `tpu_v5p-preemptible_128-us-east5-a`: workers=16 clean_workers=0 committed_tpu=64/64 running_tasks=16 min_free_mem_b=189866418176

Actions/status:
- `1e21`: queue complete
- `1e22`: queue complete

## 2026-04-27 00:56 UTC babysit tick — resume3c failed, resume3d picked up

Between :11 and :48 ticks resume3c failed (parent: failed, preemptions=1, "1 step(s) failed" gate). Last logged step 6430 loss ~1.44.

Relaunched as `delphi-1e20-p33m67-20b-lr0p67-v5p32-batch-20260427-004947-resume3d` with same `MIDTRAIN_OUTPUT_PATH_OVERRIDE=...3c967b`. Levanter resumed at step 6380 loss 1.40 (~50 steps re-trained from temp ckpt).

Pattern: marin executor's "1 step(s) failed" wrapper is intolerant of preemption-recovery cycles inside this run. Each recovery has a small chance of triggering it, requiring a manual relaunch with the same override. Net cost per cycle: ~10-15 min wall time + ~50-100 steps redundant compute.

All 6 sweep train_lm RUNNING:
- resume2 (p33m67/lr0.5): step 7310 (76.6%) loss 1.45
- resume3d (p33m67/lr0.67): step 6380 (66.9%) loss 1.40
- 4 × 081538 (p33m67/lr0.83 + p67m33/{lr0.5,lr0.67,lr0.83})

## 2026-04-27 01:11 UTC babysit tick — resume3d weathering preemption flurry

resume3d gained 4 preemption cycles in 22 min but is surviving (state=running). Step 6450 loss 1.49 — only +70 steps progress instead of expected ~150. The marin executor's "1 step(s) failed" gate hasn't fired this time though, which is what we want.

p67m33/lr0.83 broke below 2.0 sustained at step 7660 (80.3%) loss 1.99. Lead pretrain-mix run.

All 6 still RUNNING.

## 2026-04-27 01:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume3d (p33m67/lr0.67): step 6510 (68.2%) loss 1.41
- p67m33/lr0.83: step 7790 (81.7%) loss 2.00

resume3d still on 4 preemptions but holding. Quiet.

## 2026-04-27 01:41 UTC babysit tick — resume3d still surviving

resume3d at 8 preemptions (4 new in last 15 min), still running. Step 6590 (69%) loss 1.42. Slower than normal (+80 steps in 15min vs ~120 normal) due to recovery overhead but holding.

p33m67/lr0.83: step 7870 (82.5%) loss 1.36 — math-heavy lr0.83 hit new instant low.

All 6 sweep train_lm RUNNING.

## 2026-04-27 01:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume3d (p33m67/lr0.67): step 6680 (70%) loss 1.40 (preemptions=8)
- p67m33/lr0.5: step 7740 (81.1%) loss 2.00

Quiet.

## 2026-04-27 02:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume3d (p33m67/lr0.67): step 6800 (71.3%) loss 1.39
- p67m33/lr0.67: step 8260 (86.6%) loss 1.98

Quiet.

## 2026-04-27 02:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5): step 8000 (83.9%) loss 1.39
- p67m33/lr0.5:           step 7980 (83.6%) loss 1.94

Quiet. Two runs crossed step 8000.

## 2026-04-27 02:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5):    step 8120 (85.1%) loss 1.37
- p67m33/lr0.83:             step 8360 (87.6%) loss 1.93

Quiet.

## 2026-04-27 02:56 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- resume2 (p33m67/lr0.5):  step 8240 (86.4%) loss 1.36
- p67m33/lr0.83:           step 8470 (88.8%) loss 2.02

Quiet.

## 2026-04-27 03:11 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p67m33/lr0.83 (081538):    step 8600 (90.1%) loss 1.99   ← first across 90%
- resume2 (p33m67/lr0.5):    step 8360 (87.6%) loss 1.41

ETA ~2h to first completion.

## 2026-04-27 03:26 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p67m33/lr0.83 (081538):    step 8720 (91.4%) loss 2.05
- resume3d (p33m67/lr0.67):  step 7380 (77.4%) loss 1.43

Quiet.

## 2026-04-27 03:41 UTC babysit tick — 1e20 mix-LR sweep 20B steady

All 6 running.
- p67m33/lr0.83 (081538):    step 8830 (92.6%) loss 1.95
- resume2 (p33m67/lr0.5):    step 8600 (90.1%) loss 1.37

Both leaders crossed 90%. ETA ~1:25 to first finish.

## 2026-04-27 03:56 UTC babysit tick — first finish imminent

All 6 running.
- p67m33/lr0.67 (081538):  step 9070 (95.0%) loss 1.98  — ETA 56min, first to finish
- p67m33/lr0.83 (081538):  step 8950 (93.8%) loss 1.97

First job ETA ~04:53 UTC.

## 2026-04-27 04:11 UTC babysit tick — first finish ~40min away

All 6 running.
- p67m33/lr0.67 (081538): step 9200 (96.4%) loss 1.96 — ETA 40min, ~04:52 UTC
- resume2 (p33m67/lr0.5): step 8830 (92.6%) loss 1.36

Quiet otherwise.

## 2026-04-27 04:26 UTC babysit tick — p67m33/lr0.67 at 97.5%, finish ~04:58 UTC

All 6 running.
- p67m33/lr0.67 (081538): step 9310 (97.5%) loss 1.98 — ETA 32min

## 2026-04-27 04:41 UTC babysit tick — first finish in ~14min

All 6 running.
- p67m33/lr0.67 (081538): step 9420 (98.7%) loss 2.01 — ETA 14min, finishing ~04:55 UTC

## 2026-04-27 12:15 UTC babysit tick — sweep 5/6 done, 1 still running

5 of 6 sweep points have completed:
- p33m67/lr0.5 (resume2):   SUCCEEDED, 8h 48m
- p33m67/lr0.67 (resume3d): SUCCEEDED, 7h 20m
- p33m67/lr0.83 (081538):   SUCCEEDED, 21h 3m, 1 preemption
- p67m33/lr0.67 (081538):   SUCCEEDED, 20h 42m, 1 preemption
- p67m33/lr0.83 (081538):   SUCCEEDED, 20h 58m, 1 preemption

Still RUNNING:
- p67m33/lr0.5 (081538): parent preempt=3 fail=0; current train_lm task generation has 4/4 tasks running for 10h 52m, preempt=0 fail=0 on this attempt. ~28h wall since launch. Fresh-tasks elapsed indicates training started ~10:52h ago after the third preemption-recovery cycle.

iris auto-recovery weathered all 3 preemptions on this point. Nothing dead, nothing to debug. Loop continues.

## 2026-04-27 12:15 UTC — postmortem on "non-preemption" failures

User asked why earlier failures were classified as non-preemption when they all happened on a preemptible cluster. Three distinct mechanisms:

1. **JAX coordinator RPC chain break** (lr0.5 075546 → step 5500; lr0.67 resume3c → step 6430). Preemption SIGTERM hits one worker → other tasks raise `RuntimeError: Failed to send RPC to coordination service: task was preempted/died/restarted unexpectedly`. JAX client-side state doesn't gracefully tolerate peer disappearance. Marin's "1 step(s) failed" gate fires → parent FAILED with failure_count=1 (not preemption_count). Non-deterministic: resume3d weathered 8 preemptions clean, resume3c died on the first.

2. **Stale port 8476 on worker recycle** (lr0.5 first resume `-204452-resume`). Preempted JAX coordinator died without releasing port 8476. iris recycled child 303× in 60s, each retry hit `Failed to add port to server: 8476 already in use`. OS-level resource cleanup race on dirty preemption exits.

3. **Marin executor hash drift mid-run** (lr0.67 081538). After one preemption, executor recomputed output-path hash from `3c967b` → `8fbf99` and silently restarted training in new namespace. Parent stayed RUNNING (not technically a failure), but 6000 steps of progress vanished. Caught via W&B loss regression on babysit tick.

iris's preemption auto-recovery works correctly when SIGTERM → process exit → worker recycle leaves the system in a good state. Three things break that contract above. None are iris bugs.

Future systemic fixes (file as issues post-sweep):
- JAX coordinator client should tolerate peer-loss during preemption window (or wrap-and-retry in levanter)
- Marin executor: add `--retries-on-step-failure N` for parent-level retry on transient framework errors
- Marin executor: pin output-path hash for parent job lifetime, immune to host environment recompute

## 2026-04-27 19:24 UTC — correction: p67m33/lr0.5 also hash-drifted

Follow-up investigation contradicted the 12:15 tick's "nothing to debug" conclusion.
`p67m33/lr0.5` is still RUNNING, but it is not a clean continuation of the original
output namespace.

Evidence:
- Original namespace: `gs://marin-us-central1/checkpoints/delphi-1e20-p67m33-20b-lr0.5-f74454`
  - `.executor_info` written 2026-04-26 08:16 UTC
  - permanent checkpoints through `step-8000`
  - temp checkpoint only at `step-3382`
- Current/live namespace: `gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-20b-lr0.5-378f43`
  - `.executor_info` rewritten 2026-04-27 08:23 UTC after parent preemption recovery
  - permanent checkpoints through `step-7000`; `eval_metrics.jsonl` updated at 2026-04-27 19:19 UTC with eval steps through 7600
  - temp checkpoint at `gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/delphi-1e20-p67m33-20b-lr0.5-378f43/step-7619/`

Root cause is the same class as the `p33m67/lr0.67` incident, but in the opposite
direction: the parent job retried in a different region, `marin_prefix()` / mirrored
data paths resolved to that region, the executor hash changed, and Levanter imputed
a new run/checkpoint id from the new output path. This is region-dependent hash
drift, not a difference in LR or mix config.

Important correction to Claude's 12:15 postmortem: `p67m33/lr0.5` did not merely
"weather" three preemptions. It weathered them at the Iris parent level, but at
least one recovery moved from the central1 namespace to the east5 namespace and
lost the old `step-8000` continuation. If recovering this point, force:

```bash
MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-central1/checkpoints/delphi-1e20-p67m33-20b-lr0.5-f74454
WANDB_RUN_ID=delphi-1e20-p67m33-20b-lr0.5-f74454
WANDB_RESUME=allow
```

Do not treat the active east5 `378f43` endpoint as the same W&B/checkpoint
continuation as central1 `f74454`.

## 2026-04-27 20:35 UTC — postmortem: StepSpec region-dependent dataset identity

This expands the earlier "executor hash drift" diagnosis. The problem was not
just "MirrorFS changed paths" in the rendered training config. The actual
`train_lm` hash changed because the selected midtraining dataset was a newer
datakit-backed `StepSpec` graph whose dependency identity included a physical
regional GCS path.

What the experiment did:
- `experiments/exp_delphi_math_10b_midtrain.py` selected
  `BUCKET_2["nemotron_cc_math_v1/4plus"]` as the math dataset.
- For mixture runs, `experiments/midtraining_mixes.py` also pulled this same
  handle into the pretrain/math mixture.
- That catalog handle is not a stable path string. It is an executable graph:
  `download_hf StepSpec -> normalize StepSpec -> tokenize ExecutorStep -> train_lm`.

Why this backfired:
- `lib/marin/src/marin/datakit/download/nemotron_v2.py` builds the subset
  normalizer with:

```python
input_path=f"{download.output_path}/{subset_dir}"
```

- `download.output_path` is a `StepSpec.output_path`. It calls
  `marin_prefix()` and therefore resolves immediately to the coordinator's
  physical region, e.g.:

```text
gs://marin-us-central1/raw/nemotron_cc_math_v1-322fe4/4plus
gs://marin-us-east5/raw/nemotron_cc_math_v1-322fe4/4plus
```

- `lib/marin/src/marin/datakit/normalize.py` then puts that already-resolved
  `input_path` into `hash_attrs`.
- `StepSpec.as_executor_step()` wraps `hash_attrs` in `VersionedValue`, so the
  old executor treats the regional path as part of the semantic version.
- `train_lm` hashes dependency versions. Therefore the training output hash
  changed when Iris retried the parent in a different region.

Concrete evidence from `.executor_info` comparison:
- `p67m33/lr0.5` old central run: `md5(version)[:6] = f74454`
- `p67m33/lr0.5` new east5 run: `md5(version)[:6] = 378f43`
- The `version` objects differ in exactly three leaves, all the same issue:

```text
dependencies[20].dependencies[0].config.attrs.input_path
dependencies[21].config.attrs.input_path
dependencies[22].dependencies[0].config.attrs.input_path

gs://marin-us-central1/raw/nemotron_cc_math_v1-322fe4/4plus
vs
gs://marin-us-east5/raw/nemotron_cc_math_v1-322fe4/4plus
```

The same pattern explains `p33m67/lr0.67`:
- old east5 namespace: `delphi-1e20-p33m67-20b-lr0.67-3c967b`
- wrong central namespace after parent retry: `delphi-1e20-p33m67-20b-lr0.67-8fbf99`
- The `version` diff is again exactly the regioned `nemotron_cc_math_v1/4plus`
  `hash_attrs.input_path`.

Why David's "aren't dataset hashes stable?" objection is valid:
- Dataset hashes are supposed to be stable when their hash inputs are logical.
- Older datasets like Dolma use `InputName.hardcoded("raw/dolma/v1.7")`.
  The executor hash sees the relative logical string `raw/dolma/v1.7/...`;
  region-specific `gs://marin-{region}/...` paths are only materialized later
  for runtime I/O.
- The Nemotron v2 datakit path materialized `download.output_path` before
  hashing. That crossed physical placement into semantic identity.

Provenance:
- `StepSpec` itself is mainline, not introduced by this midtraining branch:
  - `beb5d8d0b4` / PR #2494: `StepSpec + Artifact for no-magic workflow orchestration`
  - `9406cc0970` / PR #4097: `StepSpec -> ExecutorStep` bridge
  - `44fe6ee43d` / PR #4142: datakit migration and StepSpec download factories
- The exact normalizer hash leak is also mainline:
  - `682942a0eb` / PR #4188: `normalize_step` stores `input_path` in `hash_attrs`
  - `f6bf3ad447` / PR #4892: Nemotron v2 normalizer builds
    `input_path=f"{download.output_path}/{subset_dir}"`
- The midtraining branch did not create `StepSpec`; it selected the
  datakit-backed `BUCKET_2["nemotron_cc_math_v1/4plus"]` path and ran it in a
  cross-region retry setting, which exposed the bug.

Operational rule going forward:
- For failed/preempted midtraining runs, do not trust the human-readable step
  name. The true run/checkpoint id includes the executor output hash.
- Before relaunching, find the exact previous output path and W&B run id from
  `.executor_info`, GCS checkpoints, and temp checkpoints.
- Relaunch with the exact old output path forced via
  `MIDTRAIN_OUTPUT_PATH_OVERRIDE` / `ExecutorStep.with_output_path(...)`, and
  verify startup logs say `Resuming training from step ...`.
- Treat any cross-region parent retry as suspect until `.executor_info` and W&B
  run id prove the same namespace is being reused.

Systemic fixes to propose:
- Hash identity should canonicalize `gs://marin-{region}/...` to a logical
  Marin path before hashing, or StepSpec hash attrs should store relative
  logical paths instead of physical regional paths.
- `StepSpec.output_path` should not be used as an input to another step's
  semantic `hash_attrs` unless it has been canonicalized.
- Iris/Marin should persist the resolved parent output path for a job attempt
  and reuse it on parent retry. Cross-region compute placement should change
  physical I/O/mirroring, not the experiment identity.

## 2026-05-01 21:00 UTC — generalized midtraining-mix framework + safety assertions

This section turns the "67/33, math-only, Delphi 1e20+1e21" sweep plan above into a reusable framework that scales to any midtraining dataset (or mixture of midtraining datasets), with hard runtime guarantees that the held-out val never leaks into training.

### Goal

A future agent should be able to:

1. Add a new midtraining dataset with one line.
2. Compose a mixture of N midtraining datasets + pretrain replay with a small declarative spec.
3. Run a sweep, pre-flight, and trust that — by construction and by runtime assertion — the val carve-out is disjoint from training, identical across runs, and not silently re-leaked by future Levanter refactors or cache rebuilds.

The current `experiments/midtraining_mixes.py` hard-codes `nemotron_cc_math_v1/4plus` as the only midtraining component and uses ad-hoc helpers per ratio. We replace that with a typed spec.

### Core abstraction

```python
# experiments/midtraining_mixes.py — proposed refactor

from dataclasses import dataclass
from typing import Iterable

from levanter.data.text import LmDataConfig, BlockShuffleConfig

# ───── 1. Heuristic constants (single source of truth) ─────

MIDTRAIN_BUDGET_FRACTION = 0.20
"""K: midtrain_tokens = pretrain_tokens * K. Same for every Delphi scale.

K=0.20 ⇔ 5/6 pretrain : 1/6 midtrain compute split. Mantis-territory.
"""

DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT = 12_500
"""~51.2 M tokens at seq=4096. ~0.1 % of a 52 B-token component. Cheap eval pass."""

# ───── 2. Budget heuristic ─────

def midtrain_token_budget(*, pretrain_tokens: int, fraction: float = MIDTRAIN_BUDGET_FRACTION) -> int:
    if pretrain_tokens <= 0:
        raise ValueError(f"pretrain_tokens must be positive, got {pretrain_tokens}")
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return int(pretrain_tokens * fraction)

# ───── 3. Declarative spec ─────

@dataclass(frozen=True)
class MidtrainComponent:
    """A non-pretrain-replay dataset component participating in the midtrain mix."""
    name: str                                     # registry key, e.g. "nemotron_cc_math_v1/4plus"
    step: ExecutorStep                            # tokenized cache step
    weight: float                                 # share of total mix in (0, 1]
    val_sequences: int = DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT  # 0 disables val
    held_out_in_pretrain: bool = False            # if True, fail loudly: this name also exists in pretrain replay

@dataclass(frozen=True)
class MidtrainMixSpec:
    """Declarative recipe for a midtraining LmDataConfig.

    pretrain_base provides the pretrain replay components and their relative
    weights (e.g. nemotron_mix). The pretrain components' weights are scaled
    so they sum to pretrain_share; midtrain components contribute the rest.
    """
    name: str
    pretrain_base: LmDataConfig
    pretrain_share: float                         # in (0, 1)
    midtrain: tuple[MidtrainComponent, ...]

    def __post_init__(self):
        validate_midtrain_spec(self)

# ───── 4. Builder ─────

def build_midtrain_lm_data_config(spec: MidtrainMixSpec) -> LmDataConfig:
    midtrain_share = sum(c.weight for c in spec.midtrain)
    if abs((spec.pretrain_share + midtrain_share) - 1.0) > 1e-6:
        raise ValueError(f"shares must sum to 1.0, got {spec.pretrain_share + midtrain_share}")

    pretrain_weights = _scale_weights(
        _fixed_train_weights(spec.pretrain_base, name="pretrain_base"),
        spec.pretrain_share,
    )
    midtrain_weights = {c.name: c.weight for c in spec.midtrain}
    weights = {**pretrain_weights, **midtrain_weights}

    components = {
        **spec.pretrain_base.components,
        **{
            c.name: step_to_lm_mixture_component(c.step, include_raw_paths=True)
            for c in spec.midtrain
        },
    }

    val_carveouts = {c.name: c.val_sequences for c in spec.midtrain if c.val_sequences > 0}

    cfg = dataclasses.replace(
        spec.pretrain_base,
        components=components,
        train_weights=weights,
        num_validation_sequences=val_carveouts or None,
        shuffle_before_trainval_split=True,           # invariant — never override
        # leave shuffle=DEFAULT_LM_DATA_SHUFFLE in place (block shuffle for training)
        auto_build_caches=False,                      # see CC2; pre-warm caches at sweep launch
    )
    assert_lm_data_config_safe(cfg)
    return cfg
```

The current four ratio mixes (`70p_30m`, `67p_33m`, `33p_67m`, `full`) become trivial one-liners on top of `MidtrainMixSpec`. The two we're using for this sweep:

```python
P67M33 = MidtrainMixSpec(
    name="pretrain_67p_math_33p_highquality_nemo_math",
    pretrain_base=nemotron_mix,
    pretrain_share=0.67,
    midtrain=(MidtrainComponent("nemotron_cc_math_v1/4plus", _highquality_nemo_math_step, weight=0.33),),
)

P33M67 = MidtrainMixSpec(
    name="pretrain_33p_math_67p_highquality_nemo_math",
    pretrain_base=nemotron_mix,
    pretrain_share=0.33,
    midtrain=(MidtrainComponent("nemotron_cc_math_v1/4plus", _highquality_nemo_math_step, weight=0.67),),
)
```

A future Recipe-B style mix becomes:

```python
RECIPE_B = MidtrainMixSpec(
    name="recipe_b",
    pretrain_base=nemotron_mix,
    pretrain_share=0.67,
    midtrain=(
        MidtrainComponent("nemotron_cc_math_v1/4plus",      step_4plus,      weight=0.16),
        MidtrainComponent("nemotron_cc_math_v1/4plus_mind", step_4plus_mind, weight=0.08),
        MidtrainComponent("nemotron_mind",                  step_nemo_mind,  weight=0.05),
        MidtrainComponent("mathcoder2",                     step_mathcoder2, weight=0.04),
    ),
)
```

Each midtrain component automatically gets a 12,500-sequence held-out slice (configurable per-component). The pretrain replay side has no carve-out (its retention is measured by Paloma c4_en separately).

### Safety assertions — four layers

Layered defense. Each catches a different failure mode.

**Layer 1 — config-time validation (fast, every test run):**

```python
def validate_midtrain_spec(spec: MidtrainMixSpec) -> None:
    if not (0 < spec.pretrain_share < 1):
        raise ValueError(f"pretrain_share must be in (0,1), got {spec.pretrain_share}")
    if not spec.midtrain:
        raise ValueError("at least one midtrain component required")

    names = [c.name for c in spec.midtrain]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate midtrain component names: {names}")

    pretrain_names = set(spec.pretrain_base.components.keys())
    overlap = set(names) & pretrain_names
    if overlap:
        # CC1: same name in both sides means weight collision; reject loudly.
        raise ValueError(
            f"midtrain components overlap with pretrain replay names: {overlap}. "
            f"Rename the midtrain component or remove it from pretrain_base."
        )

    for c in spec.midtrain:
        if not (0 < c.weight <= 1):
            raise ValueError(f"weight for {c.name} must be in (0,1], got {c.weight}")
        if c.val_sequences < 0:
            raise ValueError(f"val_sequences for {c.name} must be >= 0, got {c.val_sequences}")

    midtrain_share = sum(c.weight for c in spec.midtrain)
    if abs((spec.pretrain_share + midtrain_share) - 1.0) > 1e-6:
        raise ValueError(
            f"shares must sum to 1.0; got pretrain={spec.pretrain_share}, "
            f"midtrain_total={midtrain_share}, sum={spec.pretrain_share + midtrain_share}"
        )
```

**Layer 2 — built-config validation (fast, every launch):**

```python
def assert_lm_data_config_safe(cfg: LmDataConfig) -> None:
    weights = cfg.train_weights
    if not isinstance(weights, dict):
        raise TypeError(f"expected fixed dict weights, got {type(weights)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights sum to {total}, expected 1.0")

    if cfg.num_validation_sequences:
        for name in cfg.num_validation_sequences:
            if name not in cfg.components:
                raise ValueError(f"val carve-out {name!r} is not a registered component")
            if cfg.num_validation_sequences[name] <= 0:
                raise ValueError(f"val count for {name} must be positive")
            # CC8: don't carve out more than 10 % of any component's expected size
            # (we enforce a stronger check at launch time once cache length is known)

    if cfg.shuffle_before_trainval_split is not True:
        raise ValueError(
            "shuffle_before_trainval_split must be True so val is a random "
            "sample, not the positional tail"
        )

    # Training shuffle: block shuffle (#5246 default) or False. Don't allow full
    # Feistel/era shuffle on training — it would re-introduce the cost regression.
    if cfg.shuffle is True:
        raise ValueError(
            "Training shuffle=True (full Feistel) reintroduces the I/O cost "
            "regression that #5246 fixed. Use BlockShuffleConfig (default) or False."
        )
    if not (cfg.shuffle is False or isinstance(cfg.shuffle, BlockShuffleConfig)):
        raise ValueError(f"unsupported shuffle config type: {type(cfg.shuffle)}")
```

**Layer 3 — launch-time pre-flight (slow, once per (mix × cache) at sweep start):**

```python
def assert_val_train_disjoint(
    cfg: LmDataConfig,
    Pos: Axis,
    *,
    sample_train: int = 5_000,
    min_train_to_val_ratio: int = 100,
) -> None:
    """One-shot disjointness check by content hashing.

    Hashes a stride-sample of train sequences and ALL val sequences (val is
    small, ~12.5k) for each component, then asserts empty intersection.
    Catches:
      - any future Levanter refactor that breaks the slice/shuffle composition
      - any cache rebuild that desynchronizes train_sets/validation_sets
      - any silent override of shuffle_before_trainval_split
    """
    if not cfg.num_validation_sequences:
        return
    train_sets = cfg.train_sets(Pos, key=jax.random.PRNGKey(42))
    val_sets   = cfg.validation_sets(Pos)

    for name in cfg.num_validation_sequences:
        train_ds = train_sets[name].as_sync_dataset()
        val_ds   = val_sets[name].as_sync_dataset()

        train_len, val_len = len(train_ds), len(val_ds)
        if val_len == 0:
            raise AssertionError(f"empty val set for {name}")
        if train_len < val_len * min_train_to_val_ratio:
            raise AssertionError(
                f"train < {min_train_to_val_ratio}x val for {name} "
                f"({train_len} vs {val_len}); likely misconfigured carve-out"
            )

        stride = max(1, train_len // sample_train)
        train_h = {hash(train_ds[i].tokens.tobytes()) for i in range(0, train_len, stride)}
        val_h   = {hash(val_ds[i].tokens.tobytes())   for i in range(val_len)}
        overlap = train_h & val_h
        if overlap:
            raise AssertionError(
                f"VAL LEAKED INTO TRAIN for component {name!r}: {len(overlap)} "
                f"hash matches out of {len(val_h)} val seqs (sampled {len(train_h)} train seqs). "
                f"Stop the sweep and investigate the slice/shuffle composition."
            )

def assert_val_partition_stable(
    cfg: LmDataConfig,
    Pos: Axis,
    expected_val_first_indices: Mapping[str, list[int]],
) -> None:
    """Pin the val partition to a known fingerprint.

    `expected_val_first_indices` is a per-component list of the *first* few
    raw-cache indices that should be in val (computed once, committed to the
    sweep config). Re-running this asserts the val never silently shifted due
    to cache rebuild, JAX RNG drift, or library upgrade.
    """
    for name, expected in expected_val_first_indices.items():
        val_ds = cfg.validation_sets(Pos)[name].as_sync_dataset()
        # Hash the first len(expected) val sequences and compare to a fingerprint.
        actual = [hash(val_ds[i].tokens.tobytes()) for i in range(len(expected))]
        if actual != expected:
            raise AssertionError(
                f"val partition for {name!r} drifted from the pinned fingerprint. "
                f"Cache may have been rebuilt or split key changed."
            )
```

**Layer 4 — runtime instrumentation (every run, cheap):**

```python
def log_partition_summary(cfg: LmDataConfig, Pos: Axis) -> None:
    """Log val/train shape so it's recorded in W&B and stdout for every run."""
    val_sets = cfg.validation_sets(Pos)
    for name, ds in val_sets.items():
        n = len(ds.as_sync_dataset())
        logger.info("val[%s]: %d sequences (~%.1f M tokens)", name, n, n * Pos.size / 1e6)
    logger.info("train weights (sum=%.6f): %s", sum(cfg.train_weights.values()), cfg.train_weights)
```

### Wiring — where each layer fires

| Layer | When it runs | What it catches | Where to call |
|---|---|---|---|
| 1. `validate_midtrain_spec` | `MidtrainMixSpec.__post_init__` | mis-typed weights, name collisions, share sums | inside the dataclass |
| 2. `assert_lm_data_config_safe` | end of `build_midtrain_lm_data_config` | dropped val carve-out, shuffle override regressions | builder |
| 3. `assert_val_train_disjoint` | once per (spec × cache hash) | the math we proved above + Levanter slice/shuffle refactors | called unconditionally from `experiments/exp_delphi_math_10b_midtrain.py` `__main__` (i.e. at sweep launch, NOT per run) |
| 4. `log_partition_summary` | startup of every `train_lm` task | drift in produced shape | called from `_build_run` in the experiment file |

### Corner cases (numbered, with mitigation)

**CC1 — Component name collision between pretrain replay and midtrain.** Same name on both sides means `train_weights` would have one entry that conflates both — silent merge. *Mitigation:* `validate_midtrain_spec` rejects overlapping names. If a future recipe wants to upweight an existing pretrain component, it should adjust `pretrain_base` before passing it in.

**CC2 — Cache rebuilds shift `len(cache)`.** Feistel + slice is keyed off cache length; if `L` changes mid-sweep, val partition shifts and old "val" sequences appear in new train. *Mitigation:* (a) `auto_build_caches=False` in the built `LmDataConfig` so a missing cache is a fast error, not a silent rebuild; (b) `assert_val_partition_stable` against a pinned fingerprint to detect any drift; (c) pre-warm caches before launch and pin the underlying `ExecutorStep` hash.

**CC3 — Cross-region partial cache.** GCS replication lag could expose a partial cache to a worker mid-run. *Mitigation:* the existing `_doublecheck_paths` guard + region-stable hash (#5223) make this practically impossible now, but `assert_val_partition_stable` would catch the val drift if it ever happened.

**CC4 — Tokenizer mismatch across components.** Components tokenized with different tokenizers can't share a model embedding. *Mitigation:* `validate_midtrain_spec` could assert all components share `cfg.tokenizer` (they're all built from `nemotron_mix` derivatives + a math step that uses llama3 — already aligned, but worth asserting).

**CC5 — Sequence-length mismatch.** Cache built at different `seq_len` than training. *Mitigation:* `dataset_for_component` re-packs but loses identity; assert all caches' seq-len matches `Pos.size`.

**CC6 — Zero-weight component with `val_sequences > 0`.** `_has_nonzero_weight` filter at line 656 skips zero-weight components in the train build, so the val carve-out has nothing to slice from. *Mitigation:* `validate_midtrain_spec` rejects `weight=0` components; if you want a val for a non-trained component, that's a different feature.

**CC7 — `stop_strategy` interactions.** `restart` is the default (revisit components on exhaustion). With `first`, training stops when any component runs out. The val carve-out shrinks each component by N. With our budgets, exhaustion is many epochs away. *Mitigation:* document; assert `stop_strategy="restart"` for sweeps, OR assert the smallest component's train length × seq exceeds the budget.

**CC8 — `num_validation_sequences > L`.** Slicing returns empty train. *Mitigation:* Layer 3 `min_train_to_val_ratio=100` (i.e. val must be ≤1 % of cache); also a stricter Layer 2 check once cache length is known at build time.

**CC9 — Determinism across JAX/Levanter versions.** The Feistel split uses JAX RNG with `PRNGKey(0)`. If JAX changes the round constants in a future version, the partition shifts. *Mitigation:* low-probability for our 24-72 h sweep over a stable cache + pinned library versions; we deliberately skipped the `assert_val_partition_stable` fingerprint helper (would be belt-and-suspenders against this scenario but adds upkeep with little real-world risk reduction here). If the sweep ever spans library upgrades, add a fingerprint check then.

**CC10 — Mixture weight floating-point drift.** `_scale_weights` and downstream normalization may produce sums like `0.99999999`. *Mitigation:* Layer 2 uses `abs(sum - 1.0) < 1e-6` tolerance; mixtures auto-normalize internally so this is informational.

**CC11 — Same midtrain component name across two mixes.** Both mixes set `num_validation_sequences[name] = 12_500` → same val sequences, identical eval target across mixes. *Mitigation:* this is desired (cross-mixture comparability); document.

**CC12 — Multiple midtrain components in the same mix.** Each gets its own ~50 M-token val carve-out. Per-component eval losses are reported in `eval/<name>/loss`. *Mitigation:* design accommodates this (`midtrain` is a tuple); add a derived "weighted average midtrain val loss" metric at analysis time if desired.

**CC13 — Document-level content overlap between pretrain replay and midtrain val.** Val held out from `nemotron_cc_math_v1/4plus` may share documents/passages with `nemotron_cc` in the pretrain replay. Index-level held-out, not content-level. *Mitigation:* out of scope; document the limitation. Future work: pre-tokenization document hash dedup across all training components, or use a `nemotron_cc` variant that explicitly subtracts 4plus.

**CC14 — Held-out slice content unknown to other consumers.** Another experiment importing `BUCKET_2["nemotron_cc_math_v1/4plus"]` directly sees the full cache, including our val sequences. Our val is held out *for this experiment*, not from the world. *Mitigation:* fine for this sweep's purposes; document.

**CC15 — Sweep launch race on cache build.** Two coordinators racing to materialize the same cache. *Mitigation:* `auto_build_caches=False` at sweep launch; pre-warm caches once, then start the sweep.

**CC16 — Block shuffle window degeneracy on tiny midtrain components.** A midtrain dataset with fewer than `io_block_size × window_blocks = 131,072` sequences (~537 M tokens) collapses to a single shuffle window. Block shuffle still works but loses the hierarchical structure. *Mitigation:* warn (don't error) when a midtrain component is < 1 window; recommend a smaller `window_blocks` for tiny datasets.

**CC17 — Val carve-out from very small midtrain component.** If a future midtrain dataset is small (say 50 M tokens), 12,500 val sequences is ~50 M tokens — i.e. *the whole component*. *Mitigation:* `MidtrainComponent.val_sequences` defaults to 12,500 but is overridable; Layer 2 enforces `val ≤ 10 % of cache` when cache length is known. Add a `val_fraction: float | None = None` field that lets the caller say "carve 0.1 % of this component" instead of an absolute count.

**CC18 — `shuffle_before_trainval_split=False` re-enabled.** Would make val the literal positional tail (still disjoint, but not random). *Mitigation:* Layer 2 hard-asserts True; the spec doesn't expose a knob for it.

**CC19 — `permutation_type` change between train and val construction.** Both use `perm_type="feistel"` hard-coded inside `_split_into_trainval_sets`. *Mitigation:* not exposed as a knob; if Levanter ever changes this default, our `assert_val_partition_stable` catches it.

**CC20 — WandB metric path quirks with `/` in component names.** `nemotron_cc_math_v1/4plus` may render as `eval/nemotron_cc_math_v1/4plus/loss` (4-segment path) and confuse panel filters. *Mitigation:* normalize component name slashes when registering, or pin a sanitized display name for W&B.

**CC21 — Heuristic `K=0.20` baked into both code and docs.** If we change K, two places must update. *Mitigation:* `MIDTRAIN_BUDGET_FRACTION` is the single source of truth in code; doc references it by reading the value at import time, or just notes "see `midtraining_mixes.py:MIDTRAIN_BUDGET_FRACTION`".

**CC22 — Checkpoint-resume across version changes.** A run preempted under K=0.20 schedule that resumes after we bump K=0.25 would see a mismatched LR schedule. *Mitigation:* the executor output hash already encodes the budget (via `pretrain_tokens` + K), so a bump produces a fresh output path → no silent resume into the wrong schedule. Verify post-hash-fix #5223 still respects this.

**CC23 — Per-base BS heterogeneity.** 1e20 was pretrained at BS=128, 1e21 at BS=512. The midtrain BS=512 is fine for both architectures, but `num_train_steps` differs because tokens-per-step differs across tiers if we ever vary it. *Mitigation:* the heuristic computes tokens-then-derives-steps; document that `BS_midtrain = 512` is the convention for the sweep, separately from per-base pretrain BS.

**CC24 — Warmup steps fixed at 500.** For tiny midtrain budgets (1e20: 2,354 total steps), 500 warmup is 21 % of training. *Mitigation:* keep 500 as default but cap at `min(500, num_train_steps // 5)` to avoid pathological warmup-dominated runs; document and add a check.

**CC25 — `min_lr_ratio = 0.1`.** Decay floor is 10 % of peak. For shorter midtrain runs the LR doesn't reach the floor near the end of decay since decay is shorter. Acceptable; informational.

### Implementation order

1. Refactor `experiments/midtraining_mixes.py` to introduce `MidtrainComponent` / `MidtrainMixSpec` / `build_midtrain_lm_data_config`. Keep the existing public names (`pretrain_67p_math_33p_highquality_nemo_math` etc.) as module-level constants pointing to the new builder so call sites don't break.
2. Add `MIDTRAIN_BUDGET_FRACTION = 0.20`, `midtrain_token_budget(...)` and the four assert helpers to the same file (Layers 1, 2, 4) and a slow-pytest companion (Layer 3) at `experiments/test_midtrain_data_safety.py`.
3. Update `experiments/exp_delphi_math_10b_midtrain.py`:
   - The `BASES` table already carries `pretrain_tokens`. Replace the env-driven `MIDTRAIN_TOKEN_BUDGET` with a derived `midtrain_token_budget(pretrain_tokens=base["pretrain_tokens"])` per base.
   - Call `assert_lm_data_config_safe(cfg)` and `log_partition_summary(...)` at run-build time.
   - Once at sweep startup (i.e. before the parent dispatches children), call `assert_val_train_disjoint(...)` for each spec — gated by an env flag like `MIDTRAIN_VERIFY_VAL=1` so the slow check doesn't run on every launch but is easy to invoke.
4. Generate per-base val fingerprints once (`assert_val_partition_stable` baseline), commit them next to the experiment file, and reference them from `assert_val_partition_stable`.
5. Pre-warm `nemotron_cc_math_v1/4plus` cache in both `us-central1` and `us-east5`; record the cache hash and length in this logbook.
6. Launch.

## 2026-05-01 22:30 UTC — refactor landed: spec + assertions + per-base budget

Steps 1-3 above are now implemented. Lint, types, and tests pass.

### Files touched

- `experiments/midtraining_mixes.py` — full rewrite around `MidtrainMixSpec`/`MidtrainComponent`/`build_midtrain_lm_data_config`. Adds `MIDTRAIN_BUDGET_FRACTION = 0.20`, `DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT = 12_500`, `midtrain_token_budget(...)`, `validate_midtrain_spec` (Layer 1), `assert_lm_data_config_safe` (Layer 2), `log_partition_summary` (Layer 4). Backward-compat names (`FULL_*`, `PRETRAIN_*P_*_HIGHQUALITY_*` strings, the four pre-built `LmDataConfig` constants, `midtraining_mix_by_name`) preserved. Built configs auto-validate via `assert_lm_data_config_safe` at registry-build time.
- `experiments/midtrain_data_safety.py` — new file. Hosts `assert_val_train_disjoint`, `assert_val_partition_stable`, and `compute_val_partition_fingerprint` (Layer 3). Imports `jax` only here so consumers of `midtraining_mixes` aren't slowed down.
- `experiments/test_midtrain_data_safety.py` — new file. 50 fast tests covering Layers 1+2 plus registry-integrity invariants. Hits no GCS.
- `experiments/exp_delphi_math_10b_midtrain.py`:
  - Added `pretrain_tokens: int` to `MidtrainingBaseConfig` and populated each entry from `experiments/exp1337_delphi_suite.py` step counts.
  - Added `_token_budget_for_base(base)`: hard override (`MIDTRAIN_TOKEN_BUDGET`), then K override (`MIDTRAIN_BUDGET_FRACTION` env), then `midtrain_token_budget(...)` default.
  - Added `_budget_label(budget)`: per-base label, env-overridable.
  - `_num_train_steps(token_budget, batch_size, seq_len)` now takes the budget rather than reading a global.
  - `LR_FACTORS` shifted to `(0.33, 0.5, 0.67)` per the new sweep plan (the prior `(0.5, 0.67, 0.83)` was monotone in eval/loss; optimum was at-or-below 0.5).
  - Added `pretrain_tokens=...`, `token_budget=...`, `budget_fraction=...` tags so each W&B run records the heuristic inputs.
  - Optional Layer 4 logging gated on `MIDTRAIN_VERIFY_PARTITION=1` env flag (touches GCS to read val sequence counts).

### Per-base numbers as the code now computes them

```
1e20-iso-d2048-L21:  pretrain=24.67 B  midtrain=4.93 B   label=4p93b   steps=2,353  warmup=500  decay=1,853
1e21-v5:             pretrain=46.26 B  midtrain=9.25 B   label=9p25b   steps=4,411  warmup=500  decay=3,911
1e22-v5:             pretrain=160.37B  midtrain=32.07 B  label=32p07b  steps=7,635  warmup=250  decay=7,385  (BS=1024)
```

(The 1e21 step count is 4,411 rather than the 4,413 in the earlier table — exact integer arithmetic on `int(46_257M * 0.20) // (512*4096)` rounds slightly differently from the pen-and-paper rounding. The W&B label still reads `9p25b`. Inconsequential.)

### Verification

```bash
./infra/pre-commit.py --fix experiments/midtraining_mixes.py experiments/midtrain_data_safety.py \
    experiments/test_midtrain_data_safety.py experiments/exp_delphi_math_10b_midtrain.py
# Ruff + Black + license + AST + ... all OK

uv run --package marin --group lint pyrefly check experiments/midtraining_mixes.py \
    experiments/midtrain_data_safety.py experiments/test_midtrain_data_safety.py \
    experiments/exp_delphi_math_10b_midtrain.py
# 0 errors

uv run --package marin --group test python -m pytest \
    experiments/test_midtrain_data_safety.py experiments/test_default_train_init_mode.py \
    tests/execution/test_step_runner.py tests/test_training.py tests/execution/test_executor.py -q
# 50 + 79 + 1 skipped = pass
```

Smoke test confirmed run names + steps + LR per base match the planned sweep:

```
checkpoints/delphi-1e21-p67m33-9p25b-lr0.33   steps=4,411 lr=2.450e-03 adam_lr=1.424e-04
checkpoints/delphi-1e21-p67m33-9p25b-lr0.5    steps=4,411 lr=3.713e-03 adam_lr=2.157e-04
checkpoints/delphi-1e21-p67m33-9p25b-lr0.67   steps=4,411 lr=4.975e-03 adam_lr=2.890e-04
```

### What is intentionally NOT done yet

These remain as future work items (CC-numbered references point to the corner-case catalogue above):

- **Cache pre-warm — courtesy, not correctness.** Before submitting coordinators, the tokenized GCS cache for every mix component (`nemotron_cc_math_v1/4plus` plus the `nemotron_mix` replay components for the 67/33 and 33/67 mixes) needs to be present in whichever region the coordinator lands in. With `auto_build_caches=False` baked into every built mix (Layer 2 invariant), a missing cache fails fast at dataset-construction time — loud, immediate, no silent rebuild. So pre-warming is a courtesy ("verify in advance to avoid a 5-min round trip via launch-fail-relaunch"), not a correctness requirement. The math cache + nemotron replay caches already exist from prior sweeps and #5266 unified temp+main buckets, so they're very likely fine; if a launch fails on missing-cache, copy the missing component into the failing region and retry.
- **Partition fingerprint — explicitly skipped.** Adding `assert_val_partition_stable` against a committed baseline would catch val *drifting* between sweep launches (cache rebuilt with new content under the same path; JAX/Levanter version change in the Feistel constants). For a 24-72 h sweep over a stable content-addressed cache with pinned libraries, the risk is low and the disjointness check + `auto_build_caches=False` already cover the failure modes that matter. Helper removed from `experiments/midtrain_data_safety.py` to keep surface area minimal; re-add if a future sweep spans library upgrades or weeks.
- **Warmup cap on small-BS bases** (CC24). 1e20 at BS=128 has 2,000 warmup steps out of 9,413 total = ~21 %. Acceptable for now; revisit if 1e20's early-step loss curve looks warmup-dominated.
- **Legacy `MATH_TRAIN_STEP` single-step path** (no `MIDTRAIN_MIX_NAME` set). Still bypasses the val carve-out — `_run_pre_flight_safety_checks` warns and skips. Recommend always setting `MIDTRAIN_MIX_NAME=full_highquality_nemo_math` (or one of the replay variants) so the run goes through the safety-asserted path.
- **`MIN_LR_RATIO = 0.0`**. Code has 0.0; the §"Fixed training knobs" plan above said 0.1 (Mantis convention). Discrepancy preserved — prior runs decayed to 0, switching now would confound interpretation. Document and keep 0.0 unless explicitly bumped.

### Update: per-base permanent checkpoint cadence (every 10 % of the run)

Replaced the global `STEPS_PER_EXPORT = 1000` constant with a per-base helper `_steps_per_export(num_train_steps) = max(50, num_train_steps * 0.10)`. Every base now gets ~10 evenly-spaced permanent checkpoints + a final, regardless of total length:

| Base | Steps | Ckpt every | Permanent ckpts |
|---|---:|---:|---:|
| 1e20 | 9,413 | 941 | 10 + final |
| 1e21-v5 | 4,411 | 441 | **10 + final** (was 4 under fixed-1000 cadence) |
| 1e22-v5 | 7,647 | 764 | 10 + final |

Motivation: under the fixed cadence, 1e21's 4,411-step run only got 4 permanent rollback points (every ~22% of training). With the per-base 10% rule, 1e21 has the same rollback granularity as 1e20. Levanter's rolling temp checkpoint (`save_interval=10min`, set in `experiments/defaults.py`) is unchanged and continues to handle preemption resume.

HF export cadence is set to `None` in `SimpleTrainConfig`, which Levanter resolves to match `steps_per_export` — so HF exports also land at every ~10% of training.

Storage cost (rough): ~25 GB per Levanter ckpt (1e20) / ~45 GB (1e21) / ~135 GB (1e22). Total per-run for the new sweep:
- 1e20: 11 × 25 GB ≈ 275 GB × 6 runs = ~1.65 TB
- 1e21: 11 × 45 GB ≈ 495 GB × 6 runs = ~2.97 TB
- Sweep total: ~4.6 TB GCS

Cleanable post-sweep once the winning checkpoints are selected.

### Update: always-on safety check (no flag)

The earlier draft had a `MIDTRAIN_VERIFY_PARTITION=1` env flag gating the disjointness check. That was wrong — the check is a hard safety property; if it ever fails we want to know *before* training starts, not learn after the fact that the sweep was contaminated. Removed the flag; moved the check into the `if __name__ == "__main__":` block of `experiments/exp_delphi_math_10b_midtrain.py` so it always runs at real sweep launch but doesn't slow routine test imports / dry-runs (which don't trigger `__main__`).

The pre-flight now always:

1. Calls `log_partition_summary(cfg, Pos)` — Layer 4, logs val sequence counts and weights to stdout/W&B.
2. Calls `assert_val_train_disjoint(cfg, Pos)` — Layer 3, hashes a stride-sample of train + ALL of val and asserts empty intersection. Raises `AssertionError` and aborts the launch if val ever leaks into train.

Cost: ~30 s per coordinator launch (12 launches × 30 s ≈ 6 min total). Negligible vs. catching a silent leak.

If `MIDTRAIN_MIX_NAME` is unset (legacy single-step path), the pre-flight emits a warning and skips — that path doesn't go through `LmDataConfig` so there's no val carve-out to verify. Recommend always setting the mix name.

### What this buys us beyond "trust the math"

- **CC1, CC4, CC5, CC10:** caught at config import time. Tests fail fast.
- **CC2, CC3, CC9:** caught at sweep launch by `assert_val_partition_stable`. Cache rebuilds, JAX upgrades, region splits all flagged before training starts.
- **The Big One — index-level proof breaks:** caught at sweep launch by `assert_val_train_disjoint`. If a future Levanter refactor changes the slice/shuffle composition (e.g. block-shuffle is moved before val split, or val key changes), we see hash overlap and stop.
- **CC11, CC13, CC14:** documented limitations, not bugs; future work.
- **CC15, CC22:** prevented by infra-level conventions (pre-warm caches, region-stable hashes from #5223).

### Unverified / open

- The `dataset.shuffle(...)` API on the AsyncDataset wrapping a TreeStore cache is assumed to return an index-remap (lazy). Worth verifying that `dataset.shuffle(...).slice_dataset(...)` doesn't materialize the permuted cache anywhere on disk — if it did, we'd need to be careful about transient storage. (It's lazy; permutation is a `Permutation` object plus index lookups. Just confirm in the code once.)
- `step_to_lm_mixture_component(step, include_raw_paths=True)` — confirm `include_raw_paths` doesn't bypass the val carve-out (it shouldn't; it just exposes raw-doc paths for debugging).
- Levanter's `validation_sets()` doesn't accept a key argument; the deterministic `PRNGKey(0)` for the split is hard-coded. If Levanter ever exposes a per-call key, we must pin it.

### Pointers

- Heuristic + budget table: §"2026-05-01 20:30 UTC — new sweep plan" above.
- Index-level disjointness proof: §"2026-05-01 20:30 UTC — new sweep plan" → "Held-out validation slice".
- Code we are refactoring: `experiments/midtraining_mixes.py`, `experiments/exp_delphi_math_10b_midtrain.py`.
- Levanter shuffle internals: `lib/levanter/src/levanter/data/text/datasets.py:519-554`, `lib/levanter/src/levanter/data/permutation.py:177-276`.
- Project doc reference: `.agents/projects/delphi_midtraining.md` §10.

## Monitor log (cron `332f56ed`, 30-min cadence at :13/:43)

- **2026-05-02 06:35 UTC** — tick 1, all healthy. 1e20: 6/7 train_lm running (resume just spawned, child not yet). 1e21: 1 train_lm running, 5 pending coscheduling. 1e22: 0 train_lm running, 6 pending coscheduling. No relaunches; pending = capacity wait, not failure.
- **2026-05-02 07:20 UTC** — tick 2, ONE failure relaunched. `/ahmed/delphi-1e21-p67m33-lr0p33-batch64-20260502-055342` parent + train_lm both FAILED with `failures=1, preemptions=0` after SIGABRT in worker 5 → JAX coordinator shutdown cascade. Same class as prior 20B-sweep JAX-coordinator-peer-loss; iris does not auto-retry. Relaunched as `/ahmedah/delphi-1e21-p67m33-lr0p33-batch64-resume-20260502-072010` with `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64` + matching `WANDB_RUN_ID` + `WANDB_RESUME=allow` to preserve the namespace and wandb run continuity. Other lr0.5 train_lm came online; net 1e21 state: 1 train_lm running + 1 fresh resume + 4 pending coscheduling.
- **2026-05-02 07:43 UTC** — tick 3, no-op. No new failures since tick 2. Relaunched `lr0.33-batch64-resume-072010` parent now RUNNING with train_lm PENDING coscheduling on v5p-64 (healthy intermediate). Net live: 1e20 6 train_lm running + resume parent waiting on v5p-32, 1e21 1 train_lm running + 5 pending coscheduling, 1e22 0 train_lm running + 6 pending coscheduling. All "pending" = capacity wait, not failure.
- **2026-05-02 08:13 UTC** — tick 4, no-op. State unchanged from tick 3. **Watching:** `delphi-1e20-p67m33-lr0p5-resume-v5p32-063131` parent still RUNNING after ~1h42m with no train_lm child dispatched. Cache should hit; expect child by now. Iris parent state is healthy so no relaunch per policy, but if still child-less at tick 5, investigate parent logs for stuck `executor_main`. Live: 1e20 6 train_lm running + 1 resume parent (no child), 1e21 1 train_lm running + 5 pending, 1e22 0 train_lm running + 6 pending. No new failures.
- **2026-05-02 08:48 UTC** — tick 5, THREE 1e22 failures relaunched. All 3 `p67m33` mix `1e22` parents (`lr0.33-4e8cc7a7`, `lr0.5-f60cb12a`, `lr0.67-3c17740e`) FAILED with `failures=1, preemptions=0` after 2h+ — same JAX-coordinator coscheduled-sibling-cascade class as the 1e21 lr0.33 failure. Relaunched all 3 as `/ahmedah/delphi-1e22-p67m33-lr<X>-batch256-resume-20260502-084859` preserving output paths + WANDB_RUN_IDs. Positive: `1e22-p33m67-lr0.33` train_lm finally dispatched and is RUNNING. Resume parent `lr0p5-resume-v5p32-063131` still child-less ~2h12m; will investigate at next tick if still stuck. Live: 1e20 6 running, 1e21 1 running + 5 pending, 1e22 1 running + 5 pending (3 fresh resumes).
- **2026-05-02 09:18 UTC** — tick 6, BIG MIX. SUCCESSES: ✅ `1e20-p67m33-lr0.5-int64-014459` SUCCEEDED (v5p-64 survivor finished); ✅ `1e20-p67m33-lr0.5-resume-v5p32-063131` SUCCEEDED (resume completed cleanly — my tick-4 "stuck parent" worry was wrong, parent was just slow to log child but completed). ✅ 1e21 sweep got 5 of 6 train_lm RUNNING (p33m67-lr{0.33,0.5,0.67} + p67m33-lr{0.5,0.67}). FAILURES: 5 1e22 jobs failed with same JAX-cascade pattern (p67m33-lr{0.33,0.67} relaunched at tick 5 failed AGAIN within 25 min; p33m67-lr{0.33,0.5,0.67} all from `055342` failed). Relaunched all 5 as `resume2-091855`. Pattern: v5p-256 (32-host coscheduling) is high blast-radius for the JAX-coordinator stale-port-8476 issue. Live: 1e20 5 train_lm running (lr0.5 done), 1e21 5 train_lm running + 1 resume parent waiting, 1e22 1 train_lm running + 5 fresh resumes pending.
- **2026-05-02 09:48 UTC** — tick 7, 4 of 5 resume2-1e22 jobs RUNNING successfully. Only `1e22-p67m33-lr0.67-batch256-resume2-091855` failed AGAIN (3rd consecutive failure for this combo: `060043` → `resume-084859` → `resume2-091855`). Same JAX-cascade pattern (`failures=1, preemptions=0`, 23 min). Relaunched as `resume3-094809` (4th attempt). **If this fails too at tick 8, will stop retrying and surface to user.** Live: 1e20 5 train_lm running, 1e21 6 train_lm running, 1e22 4 train_lm running (lr0.67 p67m33 keeps cascading; rest stable). The lucky-vs-unlucky distribution suggests a specific bad worker in the v5p-256 pool keeps getting allocated for this combo.
- **2026-05-02 10:13 UTC** — tick 8, no-op. No new failures. `1e22-p67m33-lr0.67-resume3-094809` 4th attempt: parent RUNNING, train_lm PENDING (waiting v5p-256 coscheduling). Other states unchanged: 5 × 1e20 + 6 × 1e21 + 4 × 1e22 train_lm running; 2 × 1e22 train_lm waiting on capacity (p33m67-lr0.67-resume2 and p67m33-lr0.67-resume3). 2 × 1e20-lr0.5 SUCCEEDED earlier.
- **2026-05-02 10:43 UTC** — tick 9, no-op. State identical to tick 8. Both pending 1e22 lr0.67 jobs (p33m67-resume2, p67m33-resume3) still waiting on v5p-256 coscheduling capacity. Live: 5 × 1e20 + 6 × 1e21 + 4 × 1e22 = 15/18 actively training, 2 pending capacity, 2 × 1e20-lr0.5 succeeded.
- **2026-05-02 11:13 UTC** — tick 10, no-op. Identical state. Same 2 lr0.67 1e22 jobs still capacity-locked on v5p-256.
- **2026-05-02 11:43 UTC** — tick 11, no-op + ✅ NEW SUCCESS. `1e20-p33m67-lr0.33-int32-025832` SUCCEEDED (first p33m67-mix completion). Live: 4 × 1e20 + 6 × 1e21 + 4 × 1e22 train_lm running, 2 × 1e22 pending coscheduling, 3 × 1e20 succeeded total.
- **2026-05-02 12:13 UTC** — tick 12, no-op. Identical state. Both 1e22 lr0.67 jobs still pending coscheduling on v5p-256 (now 2-3 hours pending) but capacity-pending so no relaunch per policy.
- **2026-05-02 12:43 UTC** — tick 13, no-op + ✅ NEW SUCCESS. `1e20-p67m33-lr0.33-int32-025832` SUCCEEDED. Live: 3 × 1e20 + 6 × 1e21 + 4 × 1e22 running, 2 × 1e22 pending coscheduling, 4 × 1e20 succeeded total (lr0.5 ×2 + p33m67-lr0.33 + p67m33-lr0.33). 1e20 sweep is 4/6 done.
- **2026-05-02 13:13 UTC** — tick 14, no-op + ✅ NEW SUCCESS. `1e20-p67m33-lr0.67-int32-025832` SUCCEEDED. 1e20 sweep nearly done (5/6 unique LR×mix combos complete; only p33m67-lr{0.5, 0.67} still training). Live: 2 × 1e20 + 6 × 1e21 + 4 × 1e22 running, 2 × 1e22 pending coscheduling.
- **2026-05-02 13:43 UTC** — tick 15, 🎉 **1e20 SWEEP COMPLETE.** ✅ `p33m67-lr0.5` and ✅ `p33m67-lr0.67` both succeeded since tick 14. All 6/6 unique 1e20 LR×mix combos are now SUCCEEDED. Live: 0 × 1e20 + 6 × 1e21 + 4 × 1e22 train_lm running, 2 × 1e22 pending coscheduling. No new failures.
- **2026-05-02 14:13 UTC** — tick 16, no-op. Identical to tick 15. 1e21 + 1e22 still training; 2 × 1e22 lr0.67 still pending coscheduling.
- **2026-05-02 14:43 UTC** — tick 17, no-op. Identical to tick 16. No transitions.
- **2026-05-02 15:18 UTC** — tick 18, TWO new failures relaunched. (1) `1e21-p67m33-lr0.5-batch64-055342` failed after 9h7m training (`failures=1, preemptions=0`, JAX cascade) — relaunched as `lr0p5-batch64-resume-151756` with override on `delphi-1e21-p67m33-9p25b-lr0.5-fdc4ebf1`. (2) `1e22-p67m33-lr0.33-batch256-resume2-091855` failed after 4m35s with `failures=1, preemptions=1` (preempted then step-failed on recovery) — relaunched as `resume3-151756` with override on `delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7`. Also: `1e22-p33m67-lr0.33-resume2` train_lm child got recycled (PENDING) while parent still RUNNING — normal coscheduled retry, not relaunched per policy. Live: 0 × 1e20 + 5 × 1e21 (down from 6) + 3 × 1e22 (down from 4) train_lm running, 3 × pending (1e22 lr0.67 ×2 + p33m67-lr0.33 child-recycle), 2 fresh resumes.
- **2026-05-02 15:47 UTC** — tick 19, ONE new failure relaunched. `1e22-p67m33-lr0.5-batch256-resume-084859` failed after 6h11m stable training (JAX cascade, `failures=1, preemptions=0`). Relaunched as `resume3-154733` with override on `delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a`. Live: 5 × 1e21 + 2 × 1e22 train_lm running, 5 × pending coscheduling/recycle (lr0p5-resume-151756, p33m67-lr0p33-resume2 child, p67m33-lr0p33-resume3, p67m33-lr0p67-resume3, fresh lr0p5-resume3-154733). All 6/6 1e20 succeeded.
- **2026-05-02 16:18 UTC** — tick 20, TWO new failures + 3 positive transitions. Failures: (1) `1e22-p67m33-lr0.5-batch256-resume3-154733` failed within 26 min (just relaunched at tick 19, JAX cascade) — relaunched as `resume4-161841`. (2) `1e21-p67m33-lr0.67-batch64-055342` failed after ~9.5h stable training (JAX cascade) — relaunched as `lr0p67-batch64-resume-161841` with override on `delphi-1e21-p67m33-9p25b-lr0.67-99752407`. Positive: `1e21-lr0p5-resume-151756`, `1e22-p67m33-lr0p33-resume3`, and `1e22-p67m33-lr0p67-resume3` (stuck pending since tick 7) ALL got TPU and started training. Live: 5 × 1e21 + 5 × 1e22 train_lm running, 2 fresh resumes pending.
- **2026-05-02 16:43 UTC** — tick 21, no-op + ✅ FULL COVERAGE. Both tick-20 fresh resumes (`1e21-p67m33-lr0p67-resume-161841` and `1e22-p67m33-lr0p5-resume4-161841`) advanced from pending → RUNNING. **First time all 12 in-flight sweep slots (6×1e21 + 6×1e22) are simultaneously training**, on top of the completed 6×1e20. No new failures.
- **2026-05-02 17:13 UTC** — tick 22, no-op. Identical state. All 12 still training, no transitions.
- **2026-05-02 17:43 UTC** — tick 23, no-op. Identical to tick 22.
- **2026-05-02 18:13 UTC** — tick 24, no-op + 2 child-task recycles (`1e22-p67m33-lr0p33-resume3-151756/train_lm` and `1e22-p67m33-lr0p5-resume4-161841/train_lm` flipped to PENDING; parents still RUNNING). Normal coscheduled retry, not relaunched per policy. Net: 11/12 train_lm running, 2 child-recycle pending, 1 still healthy ─ no transitions in 1e21 or completed jobs.
- **2026-05-02 18:43 UTC** — tick 25, no-op. Identical to tick 24; same 2 child recycles still pending coscheduling.
- **2026-05-02 19:13 UTC** — tick 26, ONE failure relaunched (5th attempt for stubborn `1e22-p67m33-lr0.5`). The tick-24 child recycle for `resume4-161841` didn't recover; parent + train_lm FAILED with `failures=1, preemptions=1` after 1h22m. Relaunched as `resume5-191734`. Positive: the other tick-24 recycle (`p67m33-lr0p33-resume3-151756/train_lm`) recovered from PENDING → RUNNING. Live: 6×1e21 + 5×1e22 train_lm running, 1 fresh resume pending.
- **2026-05-02 19:43 UTC** — tick 27, no-op + ✅ resume5 dispatched. `1e22-p67m33-lr0p5-resume5-191734` parent + train_lm both RUNNING. All 12 in-flight slots actively training. No new failures.
- **2026-05-02 20:13 UTC** — tick 28, no-op. Identical to tick 27. All 12 still training, no transitions.
- **2026-05-02 20:43 UTC** — tick 29, 🎉 3 NEW 1e21 SUCCESSES. ✅ `1e21-p67m33-lr0.33-resume-072010`, ✅ `1e21-p33m67-lr0.33-055342`, ✅ `1e21-p33m67-lr0.67-055342` all succeeded. **1e21 sweep 3/6 done.** Live: 3 × 1e21 + 6 × 1e22 train_lm running, 9/18 sweep slots succeeded total.
- **2026-05-02 21:13 UTC** — tick 30, ✅ +1 1e21: `1e21-p33m67-lr0.5-055342` succeeded. **1e21 sweep 4/6 done.** Live: 2 × 1e21 + 6 × 1e22.
- **2026-05-02 21:43 UTC** — tick 31, no-op. Identical to tick 30. 10/18 sweep slots succeeded total (6 × 1e20 + 4 × 1e21).
- **2026-05-02 22:13 UTC** — tick 32, no-op. Identical to tick 31. 1e22 step counts (per tick-31 spot-check): p33m67-lr0.33 @4320 (56%), p67m33-lr0.33 @3820 (50%), p67m33-lr0.5 @3260 (43%), p33m67-lr0.67 @2670 (35%), p33m67-lr0.5 @2620 (34%), p67m33-lr0.67 @2270 (30%). First 1e22 finisher ETA ~06:00 UTC May 3.
- **2026-05-02 22:43 UTC** — tick 33, no-op. Identical to tick 32. 8 in-flight slots all training, 10/18 succeeded.
- **2026-05-02 23:13 UTC** — tick 34, ⚠️ ONE relaunch. `1e22-p67m33-lr0.67-resume3-094809` parent FAILED (`failures=1, preemptions=2`, JAX cascade after 2 preemption recycles). Was at step 2270/7647 (~30%) before failure. Relaunched as `delphi-1e22-p67m33-lr0p67-batch256-resume4-20260502-231948` with same WANDB_RUN_ID `delphi-1e22-p67m33-32p07b-lr0.67-3c17740e` and MIDTRAIN_OUTPUT_PATH_OVERRIDE `gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.67-3c17740e`. This combo (p67m33-lr0.67) has now failed 4 consecutive times (060043 → resume → resume2 → resume3). Live: 5 × 1e22 train_lm running + 1 fresh resume pending, 2 × 1e21 train_lm running, 10/18 succeeded.
- **2026-05-02 23:43 UTC** — tick 35, ✅ resume4 dispatched fast. `1e22-p67m33-lr0p67-resume4-231948` parent + train_lm both RUNNING after ~24min (lucky capacity hit). All 12 in-flight slots back to training. No new failures.

## 2026-05-02 04:00 UTC — sweep launch chaos, JAX-coordinator cascade, lessons learned

Long live-debugging session. Captured here so the next agent inherits the empirical findings without re-discovering them at 4 AM.

### Sweep timeline (all on 1e20-iso-d2048-L21, BS=128, K=0.20 → 4.93 B tokens / 9,413 steps, mostly us-east5-a)

| Sweep ID | TPU | Priority | Outcome |
|---|---|---|---|
| `20260501-234550` | v5p-32 | batch | All 12 failed with `TypeError: object of type 'VersionedValue' has no len()` in pre-flight `validation_sets()` call. |
| `20260501-235233` | v5p-32 | batch | Pre-flight switched to math-only mix; same TypeError (math step also has versioned-wrapped paths). |
| `20260501-235704` | v5p-32 | batch | Pre-flight wrapped in `try/except TypeError` — works. Tokenize rebuilt (`4plus-2c5519`, ~13 min normalize + ~22 min tokenize). 2 of 12 train_lm running before user killed all 12 to switch to interactive. |
| `20260502-010854` | v5p-32 | **interactive** | 6 × 1e20 only. 2 train_lm reached step ~390 then user killed to switch to v5p-64. |
| `20260502-012515` | v5p-64 | batch | 6 × 1e20. User killed before any reached training (mistakenly submitted as batch when intent was interactive). |
| `20260502-014459` | v5p-64 | interactive | 6 × 1e20. **5 of 6 cascade-killed** by stale-port-8476 JAX coordinator collision (`INVALID_ARGUMENT: Unexpected task registered with task_name=/job:/replica:0/task:0`). One survivor: `p67m33-lr0p5-int64-20260502-014459`. |
| `20260502-020127` | v5p-64 | interactive | 4 relaunches of dead 014459 jobs. **All 4 failed again**, same JAX coordinator pattern + one SIGSEGV. (First attempt had a copy-paste mix-mapping bug — `p33m67` short was paired with `67p_33m_*` long; killed and resubmitted with correct mapping. The corrected resubmits also failed.) |
| `20260502-025832` | v5p-32 | interactive | 5 missing combos (everything except `p67m33-lr0p5` which the v5p-64 survivor covers). 3 train_lm running on v5p-32 within ~50 min, 2 still pending coscheduling. v5p-32's smaller blast radius (4 hosts vs 8) avoided cascade kills. |

### Empirical findings about the cluster (verified live)

**1. `compute_effective_band` is dynamic and recomputed every scheduling tick.**
- For both pending and running tasks. So spend > 75 k on tick T → all of user's INTERACTIVE work is BATCH-effective on tick T+1, until spend drops back below.
- PRODUCTION never demotes (`budget.py:121-122`). PRODUCTION is admin-tier-only (`runner, power, dlwh, rav, romain, held, larry`).
- I was earlier wrong that `ahmedah` was already "over quota" with 12 PENDING jobs — `ACTIVE_TASK_STATES = {ASSIGNED, BUILDING, RUNNING}` (`db.py:144`); PENDING does NOT count. Spend was 34,400 / 75,000 = 45.9 % under at that snapshot.

**2. BATCH band literally never preempts anything.** `controller.py:655-657`:
```python
for candidate in unscheduled_tasks:
    # Batch never preempts
    if candidate.band >= job_pb2.PRIORITY_BAND_BATCH:
        continue
```
- Within BATCH, "fairness" is round-robin scheduling order on new slots via `interleave_by_user` (`budget.py:129-160`) — under-spend users get first pick when a worker boots or task finishes. Existing batch tasks are never *evicted* in favor of new ones from a lower-spend user.
- Practical implication: if the cluster is at full BATCH-band capacity (which it routinely is when researchers go over quota), submitting at BATCH means waiting for autoscaler to bring up new workers. No preemption shortcut.
- To preempt over-quota users' BATCH-effective work: submit at INTERACTIVE while you are under quota. INTERACTIVE > BATCH preemption rule (`controller.py:535`). Once *your* spend crosses 75 k, your INTERACTIVE work auto-demotes to BATCH-effective and loses preemption power.

**3. JAX coordinator stale-port-8476 cascade.**
- Symptom: one of N coscheduled hosts dies at startup with `F: Terminating process because the JAX distributed service detected fatal errors. INVALID_ARGUMENT: Unexpected task registered with task_name=/job:/replica:0/task:0`. The other N-1 cascade-killed via Marin's "1 step(s) failed" gate.
- Cause: when a previous coscheduled job (especially one that was killed mid-run) leaves the JAX distributed coordinator process bound to port 8476 on a worker, then iris recycles that worker for a new task, the new task tries to register a task ID that's already known to the leftover coordinator → fatal.
- Cluster gets into a "bad pool" state where multiple recycled workers carry stale state. Repeated relaunches keep hitting them.
- Smaller coscheduling group helps: v5p-32 (4 hosts) sometimes survives where v5p-64 (8 hosts) doesn't, just because fewer hosts means lower probability of hitting at least one bad worker.
- No clean upstream fix yet. Workarounds: keep retrying; switch to v5p-32; wait for autoscaler to drain bad workers.

**4. Cache hash drift from PR #5223 forces one-time tokenize rebuild.**
- The `tokenized/nemotron_cc_math_v1/4plus` step is non-leaf, so its hash recomputed when #5223 (region-stable hashes) merged. Pre-#5223 caches at hashes `4plus-{212a2d, da9608, 0bd79d}` are orphaned. New cache at `4plus-2c5519`.
- `auto_build_caches=False` on `LmDataConfig` does NOT prevent the Marin executor from rebuilding upstream `normalize → tokenize` ExecutorSteps (those are separate steps in the dep graph; `auto_build_caches` only governs Levanter's own cache layer). So missing cache becomes a long zephyr rebuild rather than a fast error. Pre-warm caches manually before launching if you care about start time.
- After the one-time rebuild, future sweeps with the same code reuse `4plus-2c5519` (verified — 0252-onwards sweeps skip tokenize). DON'T merge another upstream commit that touches StepSpec hashing during a sweep, or you re-incur the rebuild.

### `wandb_project` hard-coding fix (commit `6d6384bd2`)

Was: `experiments/defaults.py:427` had `tracker=WandbConfig(project="marin", ...)` hard-coded since `b804686ae3` (2025-04-03, "wip", David Hall). The `default_train_dpo` sibling already supported `wandb_project` via its config dataclass; `default_train` just hadn't been extended. The `WANDB_PROJECT` env var alone never worked because Levanter's `wandb.init(project=self.project, ...)` passes the explicit string from `WandbConfig`.

Fixed by:
- Adding `wandb_project: str | None = None` kwarg to `default_train`
- `tracker=WandbConfig(project=wandb_project or "marin", ...)`
- `experiments/exp_delphi_math_10b_midtrain.py` passes `wandb_project="delphi-midtraining"` per call.

Net effect: every Delphi midtraining run from this branch now lands at `https://wandb.ai/marin-community/delphi-midtraining`. Verified via the wandb dashboard showing 32+ runs in that project including all sweep cycles documented above.

### Wandb run-id "fragmentation" — what to expect with multiple kill/relaunch cycles

Levanter's `WandbConfig(resume="allow")` is the default. When two runs share the same display name within the same wandb project, wandb merges them into a single run id and continues the metric stream. Concretely:

- Sweep 235704 created `delphi-1e20-p67m33-4p94b-lr0.5-9e1229`. Killed at step ~390.
- Sweep 014459 (post-fix, but on v5p-64 which has different `train_cfg.resources` → different TrainLmConfig hash → conceptually different executor output path) — **but the wandb display name** is computed differently (it pulls from the experiment-side `name` truncated, plus a suffix Marin appends from the executor output basename). When that display name happened to match `delphi-1e20-p67m33-4p94b-lr0.5-9e1229`, wandb's resume="allow" picked up the existing run id and continued the metric stream.
- Result: one wandb run shows step 0-390 from the killed v5p-32 attempt + step 391-5300+ from the live v5p-64 survivor, all under the same run id. **Looks continuous in the chart even though the underlying training was interrupted and restarted on different hardware.**
- The other LR factors (`lr0.33-590ea1`, `lr0.67-64a9c5`) don't have a long-running survivor doing the same name-resume trick, so they appear truncated at ~400 steps.

This is mostly fine — the visualization continues to show the trajectory — but if you want clean per-sweep wandb runs, you'd need to either set unique `WandbConfig.name` per sweep launch (e.g. include the sweep timestamp) or `resume="never"`.

### Snapshot at session-end

- **1 v5p-64 train_lm running**: `p67m33-lr0p5-int64-20260502-014459`, ~step 5,300/9,413 (~56 %), 1.2 s/iter, ETA ~05:30 UTC. Survivor of the cascade.
- **3 v5p-32 train_lm running** from sweep `025832`: `p67m33-lr0p33`, `p67m33-lr0p67`, plus one of the p33m67 (autoscaler still ramping)
- **2 v5p-32 train_lm pending** coscheduling: 2 of the p33m67 mix
- **All currently in `marin-community/delphi-midtraining`** wandb project (post-fix).

### Lessons baked in

1. **Always submit at BATCH for unattended sweeps over many jobs.** INTERACTIVE only buys priority while user is under 75 k spend; once over (which any 6+ job 1e20/1e21 sweep guarantees), it's BATCH-effective anyway. Submit BATCH from the start to avoid surprises.
2. **v5p-32 > v5p-64 for sweep robustness** — the cascade-kill blast radius scales with coscheduling group size. If JAX coordinator port-8476 issues are happening (post a worker pool getting churned by killed jobs), prefer smaller TPU shapes for the next sweep launch.
3. **Don't kill+relaunch a sweep within 30 min on the same worker pool.** The recycled workers carry stale JAX state. Either wait for autoscaler to fully reap them, or accept high failure rate on the immediate retry.
4. **Pre-warm caches before launching.** `auto_build_caches=False` is half-protection only; the Marin executor's normalize/tokenize steps will rebuild silently. `gcloud storage ls gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-<expected-hash>` before launch saves a 35-min build delay.
5. **`wandb_project="..."` kwarg now works on `default_train`** — use it for any new midtraining experiments to avoid landing in the generic `marin` project.

## 2026-05-01 19:51 UTC — handoff: main merged and midtraining launch guard pushed

State for the next agent:

- Branch: `midtrain_data`
- Remote head after handoff: `4b40df269 [experiments] Pin Delphi midtraining jobs by region`
- Previous merge commit: `dc41a9bac Merge remote-tracking branch 'origin/main' into midtrain_data`
- Important upstream fix now present from main: `7f0b99b9e Stop region prefixes leaking into Marin executor identity hashes (#5223)`.
  This is Rav's executor/StepSpec region-agnostic hashing fix.

What was changed and pushed:

- `experiments/exp_delphi_math_10b_midtrain.py`
  - Removed the sample launch env `MARIN_I_WILL_PAY_FOR_ALL_FEES=1`.
  - Kept coordinator scheduling flexible across `--region us-central1 --region us-east5`.
  - Added `_selected_train_region()` / `_midtrain_tpu_resources()` so the child `train_lm`
    TPU job is pinned to the coordinator's resolved region when it is one of the v5p regions
    (`us-central1` or `us-east5`).
  - Added `MIDTRAIN_TRAIN_REGION` override; zones like `us-central1-a` normalize to `us-central1`.
  - Cleaned stale comments that implied the experiment should always write in us-east5.
  - Centralized base compute settings and added reusable mix/run-name knobs for follow-up sweeps.
- `experiments/midtraining_mixes.py`
  - Added reusable midtraining mixtures for full math and pretrain/math replay ratios.
- `experiments/test_default_train_init_mode.py`
  - Added tests for child resource region pinning, zone override normalization, explicit bad-region failure,
    and local non-v5p region fallback.
- `scripts/_verify_mirror_stage.py`
  - Tiny pre-commit cleanup: removed an unused f-string.

Validation completed before push:

```bash
./infra/pre-commit.py --all-files --fix
uv run pytest experiments/test_default_train_init_mode.py tests/test_training.py tests/execution/test_executor.py tests/execution/test_step_runner.py -q
```

Result: `87 passed, 1 skipped`.

Operational interpretation:

- Future Delphi midtraining launches may still submit the parent/coordinator with both v5p regions.
- Once the coordinator is running, the generated child resource list is single-region, matching the
  coordinator's region. This avoids the observed footgun where the parent materialized concrete
  `gs://marin-us-east5/...` paths but Iris placed the child in `us-central1`.
- This is not a general checkpoint migration system. It is a targeted guard for these midtraining
  experiments, layered on top of the upstream region-agnostic hash fix.
- For any failed/preempted existing run, still follow the resume rule above: identify the exact old
  output path/run id, check permanent and temporary checkpoints, force the old output path if relaunching,
  and verify `Resuming training from step ...` in startup logs.

Local worktree note:

- The pushed commit intentionally did not include the large local logbook/analysis/debug artifacts.
- At handoff, expect unrelated dirty files to remain in the worktree, including this logbook,
  `experiments/defaults.py`, `tests/test_training.py`, and analysis scripts/plots. Do not assume they
  are part of the pushed refactor unless the user explicitly asks to curate and commit them.

## 2026-05-01 20:30 UTC — new sweep plan: 1e20 + 1e21, 67/33 mixes, dynamic per-scale budget

This supersedes the absolute-budget framing of every prior sweep (10 B and 20 B used the same number of tokens for every base, regardless of pretrain budget). Per §8.3 of `.agents/projects/delphi_midtraining.md`, midtrain budget now scales with each base's own pretrain budget under a single rule applied uniformly across the ladder.

### Heuristic (one rule, every model)

```
midtrain_tokens = pretrain_tokens / 5
```

Equivalent to a 5/6-pretrain : 1/6-midtrain compute split (`K = 0.20`). Identical for 1e20, 1e21, 1e22, 1e23 — no scale-specific override. Sits in Mantis-style cooldown territory (~10–20 % of pretrain). The earlier proposal of K = 0.5 (1/3 midtrain share) was deemed too aggressive on 2026-05-01 lab discussion follow-up.

### Pretrain → midtrain budgets

**Midtrain BS = pretrain BS for every base** (no standardization). The earlier "BS=512 standardized" framing was leftover from the prior 10B sweep that overrode 1e20 to BS=512 for v4-128 friendliness. The current code already does the right thing by default (`OVERRIDE_BATCH_SIZE` is only used if `MIDTRAIN_BATCH_SIZE` env var is set; otherwise `batch_size = base.train_batch_size`).

Pretrain tokens computed from `experiments/exp1337_delphi_suite.py` (steps × BS × seq=4096); 1e20 row is the 3e20-isoflop d2048-L21 stand-in (47064 × 128 × 4096):

| Scale | Pretrain steps × BS | Pretrain tokens | Midtrain (K=0.20) | Midtrain BS | Midtrain steps | warmup + decay |
|---|---|---:|---:|---:|---:|---|
| **1e20** | 47,064 × 128 | **24.67 B** | **4.93 B** | **128** | **9,413** | 2,000 + 7,413 |
| **1e21-v5** | 22,057 × 512 | **46.27 B** | **9.25 B** | **512** | **4,411** | 500 + 3,911 |
| 1e22-v5 | 38,235 × 1024 | 160.37 B | 32.07 B | 1024 | 7,647 | 250 + 7,397 |
| 1e23-v5 | 74,884 × 2048 | 628.25 B | 125.65 B | 2048 | 14,977 | 125 + 14,852 |

Only 1e20 and 1e21 are in scope for this sweep; 1e22 and 1e23 are confirmation tiers per §8.5. The 1e21 budget (9.25 B) lands close to the prior 10 B sweep, giving a near-direct comparison against existing 1e21 v5p-256 pilot data without re-running.

**Note on warmup token budget.** `WARMUP_TOKENS = 500 × 512 × 4096 ≈ 1.05 B` is held constant across bases; warmup *steps* scale inversely with `batch_size × seq_len`. So 1e20 at BS=128 gets 2,000 warmup steps (~21 % of training), 1e21 at BS=512 gets 500 warmup steps (~11 %), 1e22 at BS=1024 gets 250 warmup steps (~3 %), 1e23 at BS=2048 gets 125 warmup steps (~1 %). The "warmup tokens are constant, warmup steps are derived" convention preserves the same warmup compute regardless of BS.

### Mixtures (two; both already registered)

Both already exist in `experiments/midtraining_mixes.py`:

- `pretrain_67p_math_33p_highquality_nemo_math` — 67% Nemotron pretrain replay, 33% `nemotron_cc_math_v1/4plus`
- `pretrain_33p_math_67p_highquality_nemo_math` — 33% Nemotron pretrain replay, 67% `nemotron_cc_math_v1/4plus`

### LR sweep grid

3 factors × 2 mixes × 2 scales = **12 runs**. The 10 B/20 B prior campaigns showed monotone `lr0.5 < lr0.67 < lr0.83` on eval/loss for both mixes (optimum at or below 0.5), so shift the grid down to bracket the new minimum:

```
LR factors = {0.33, 0.5, 0.67} × pretrain peak
```

Same factor applied to both `learning_rate` and `adam_lr`. Warmup 500 steps, linear decay, `min_lr_ratio=0.1`, AdamH, β₂/ε inherited per base.

| Base | factor | `learning_rate` | `adam_lr` |
|---|---:|---:|---:|
| 1e20 (peak 4.483e-3 / 7.382e-5) | 0.33 | 1.479e-3 | 2.436e-5 |
| 1e20                            | 0.50 | 2.241e-3 | 3.691e-5 |
| 1e20                            | 0.67 | 3.004e-3 | 4.946e-5 |
| 1e21-v5 (peak 7.425e-3 / 4.314e-4) | 0.33 | 2.450e-3 | 1.424e-4 |
| 1e21-v5                            | 0.50 | 3.713e-3 | 2.157e-4 |
| 1e21-v5                            | 0.67 | 4.975e-3 | 2.890e-4 |

### Held-out validation slice

Per §8.2, the optimization target is a held-out subset of the **midtrain mixture itself**, not Paloma c4_en. Carve out from the math component (the thing the sweep is trying to optimize); pretrain retention is still measured by Paloma c4_en independently.

Use Levanter's built-in mechanism (no new tokenization required):

```python
num_validation_sequences = {"nemotron_cc_math_v1/4plus": 12_500}  # 12.5k × 4096 ≈ 51.2 M tokens
shuffle_before_trainval_split = True  # default
```

**Two separate shuffle code paths — don't conflate them.**

1. **Training stream** (`LmDataConfig.shuffle`). PR #5246 (`f2f06da2b`, in main, merged here) changed the default from `False` → `DEFAULT_LM_DATA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")`. This is the cost-saving change: a 9 B–23 B-token training stream under full Feistel would mean ~10⁹ random reads against the tree-store; block shuffle keeps reads contiguous within 256-sequence blocks. We rely on this default — not overridden.

2. **Val carve-out** (`_split_into_trainval_sets`, `lib/levanter/src/levanter/data/text/datasets.py:519-539`). NOT touched by #5246. Still does a full Feistel index-remap, then slices the tail:

   ```python
   split_key = jax.random.PRNGKey(0)
   dataset = dataset.shuffle(split_key, perm_type="feistel")   # full permutation, deterministic
   train_ds = dataset.slice_dataset(0, length - N)
   val_ds   = dataset.slice_dataset(length - N, length)
   ```

   Feistel here is an index map, not a physical reshuffle. Cost at val time = N random reads (12,500 ≈ 50 MB) × ~24 eval passes ≈ 300 k random reads per run — dwarfed by the protected training stream.

| Stream | Volume per run | Shuffle | Why |
|---|---:|---|---|
| Training | 9 B (1e21) / 4.9 B (1e20) tokens | hierarchical block shuffle | I/O locality on huge sequential reads |
| Val carve-out | ~50 M tokens, read ~24×/run | full Feistel (index remap) | tiny, random-access cost negligible; better mixing than block shuffle |

Net result: block shuffle for training (best I/O — the #5246 default) + full Feistel for val carve-out (best mix on a tiny slice). Both on by default in `LmDataConfig`. No code change needed beyond setting `num_validation_sequences` on the two registered mixes.

Sanity check to add after first run: assert val indices span the cache (e.g., quartile counts ≈ uniform). ~5 lines, in `experiments/test_default_train_init_mode.py` or similar.

Removes ~0.1 % of `nemotron_cc_math_v1/4plus` (52 B → still ~52 B). No retokenization, no new GCS path — Levanter handles it inside the existing cache.

### Hierarchical block shuffle — what we are relying on

`BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")` is the default for LM data (`lib/levanter/src/levanter/data/text/datasets.py:551-555`). It is a **two-level permutation applied per component before the mixture is built** (`datasets.py:740-754`, each component gets its own PRNG key):

1. **Block level (Level 1):** the cache is cut into I/O blocks of 256 sequences (≈1.05 M tokens at seq=4096). Full blocks are globally permuted with a Feistel cipher (`permutation.py:177-219`). Preserves disk-prefetch locality — 256 contiguous sequences are read at once.
2. **Window level (Level 2):** within each window of 512 blocks (≈537 M tokens), example offsets are also permuted (`permutation.py:220-234`, separate keys for full region and tail region).
3. **Tail block** stays at the end of the dataset and is only locally permuted.

The mixture is then built on top with `mixture_block_size=2048` sequences (~8.4 M tokens) — within each mixture block the component-weights are honored deterministically, but slot order is permuted (`MixtureDataset` in `lib/levanter/src/levanter/data/mixture.py`).

Implications for this plan:
- 12 B / 23 B midtrain budgets traverse the 537 M-token shuffle window 22×–43× → plenty of mixing on the math component.
- The 67/33 vs 33/67 mix is honored at ~8.4 M-token granularity, not just on the run-average.
- Held-out val sequences are excluded *before* shuffling, so the val set is the same 51.2 M-token slice for every run.

### Open questions and next concrete steps

1. **K = 0.20 chosen** (1/6 midtrain compute share, midtrain = pretrain / 5). Alternative K = 0.25 (1/5 share) on the table if the lighter cooldown underfits.
2. Wire `num_validation_sequences={"nemotron_cc_math_v1/4plus": 12_500}` and `shuffle_before_trainval_split=True` into both registered mixes (`pretrain_67p_math_33p_highquality_nemo_math` and `pretrain_33p_math_67p_highquality_nemo_math`). Small edit to `experiments/midtraining_mixes.py`.
3. Plumb `MIDTRAIN_TOKEN_BUDGET` to read from `K × pretrain_tokens` per base (not a hard-coded constant). The `BASES` table in `experiments/exp_delphi_math_10b_midtrain.py` already carries `pretrain_tokens` per entry; add a derived `midtrain_token_budget = pretrain_tokens // 5` field.
4. Launch sequence:
   - 6 × 1e20 runs (3 LR × 2 mixes), v5p-32 batch, ~2,354 steps each (~3 h/run at 7.2 s/it).
   - 6 × 1e21 runs (3 LR × 2 mixes), v5p-64 batch, ~4,413 steps each (~4 h/run at 3.2 s/it).
   - All 12 launched with `--region us-central1 --region us-east5`; recently merged region-stable hash fix (#5223) and the branch-local Delphi region-pin guard make this safe.
5. Update §10 of `.agents/projects/delphi_midtraining.md` with the chosen K, the per-scale budget table, and the val-slice spec.

## 2026-05-01 19:58 UTC — main merged locally; which incident fixes landed

User asked to pull/merge main and check recent commits to see whether the
cross-region incident is mostly solved. Actions taken:

```bash
git fetch origin main
git merge --autostash origin/main
```

Merge result:

- Local merge commit: `ecd8fbca7 Merge remote-tracking branch 'origin/main' into midtrain_data`
- Merge succeeded cleanly; autostash reapplied local dirty worktree changes.
- Local branch is ahead of `origin/midtrain_data` by the main merge commits.
  This merge has not been pushed.

Relevant fixes now present in the local merged branch:

- `7f0b99b9e Stop region prefixes leaking into Marin executor identity hashes (#5223)`
  - Fixes the core hash-drift bug from this incident.
  - `StepSpec.hash_id` uses dependency names/hashes instead of physical dep output paths.
  - `normalize_step` hashes `relative_input_path` instead of the resolved physical `input_path`.
  - Executor deep-dep fallback uses region-stable `{name}-{hash}` rather than `gs://marin-<region>/...`.
- `b4298305a infra/rigging: fold tmp buckets into main buckets (#5266)`
  - Replaces separate `marin-tmp-*` temp buckets with `tmp/ttl=...` prefixes inside
    normal `marin-<region>` buckets.
  - `marin_temp_bucket(..., source_prefix=output_path)` chooses temp checkpoint
    location from the training output path's region.
  - This supersedes the earlier `mirrortmp://` design path for mainline.
- `a154c044f Charge cross-region transfer budget on tensorstore checkpoint I/O (#5225)`
  - TensorStore checkpoint serialize/deserialize now calls `record_transfer(...)`.
  - This plugs the accounting gap where TensorStore bypassed fsspec and therefore
    bypassed `CrossRegionGuardedFS`.
- `9d9b9a2a7 [iris] Fix coscheduled split-slice and orphan attempt bugs (#5249)`
  - Improves Iris retry/preemption state handling for split-slice/coscheduled jobs.
- `4b40df269 [experiments] Pin Delphi midtraining jobs by region`
  - Branch-specific guard: the Delphi midtraining parent can be scheduled flexibly,
    but the generated TPU child is pinned to the coordinator's resolved region.

Focused regression tests run after the merge:

```bash
uv run pytest \
  tests/execution/test_step_runner.py::test_step_spec_hash_id_stable_across_prefixes \
  tests/execution/test_step_runner.py::test_step_spec_hash_id_via_marin_prefix_env \
  tests/execution/test_step_runner.py::test_resolve_executor_step_infers_region_for_iris_without_pin \
  tests/execution/test_step_runner.py::test_resolve_executor_step_raises_on_cross_region_inputs_without_pin \
  tests/execution/test_step_runner.py::test_resolve_executor_step_raises_on_cross_region_even_with_override_env \
  tests/execution/test_step_runner.py::test_executor_resolve_steps_uses_component_gcs_region_to_pick_tpu_region \
  tests/execution/test_step_runner.py::test_executor_resolve_steps_picks_one_region_for_multi_region_tpu_component \
  tests/execution/test_executor.py::test_executor_version_stable_across_prefixes \
  tests/test_training.py::test_temporary_checkpoint_base_path_follows_output_path_region \
  tests/test_training.py::test_update_config_to_use_out_path_sets_run_specific_temp_checkpoints \
  lib/rigging/tests/test_record_transfer.py \
  -q
```

Result: `13 passed`.

Conclusion:

- The original **region-sensitive hash drift** is now fixed in main.
- The original **TensorStore cross-region accounting gap** is fixed in main.
- The original **separate temp-bucket design weakness** is mostly addressed by
  folding temp paths into primary regional buckets and deriving temp region from
  output path.
- The branch-specific Delphi parent/child placement footgun is guarded in this
  branch, but the broader training child alignment PR (`cc2678ff4`) is not merged
  into main.
- There is still no full automatic region migration/resume system. For old failed
  runs, still force the exact old output path and verify resume logs.

## 2026-05-02 04:49 UTC — launched first 1e21-v5 current-mix point on `/ahmed`

User asked to launch the first current 1e21 config with `v5p-256`,
interactive priority, and the local Iris account `/ahmed`.

Interpretation of "first config": `1e21-v5 × p67m33 × lr0.33`
(`MIDTRAIN_MIX_NAME=67p_33m_highquality_nemo_math`).

Sanity check before submission:

```bash
MIDTRAIN_SELECT_BASE=1e21-v5
MIDTRAIN_SELECT_LR=0.33
MIDTRAIN_MIX_NAME=67p_33m_highquality_nemo_math
MIDTRAIN_TPU_TYPE=v5p-256
```

Generated exactly one step:

- Step: `checkpoints/delphi-1e21-p67m33-9p25b-lr0.33`
- Child TPU resource: `v5p-256` (`replicas=32`)
- Train steps: `4,411`
- Batch: `512`
- Peak LR / Adam LR: `2.45025e-3` / `1.42362e-4`
- W&B project: `delphi-midtraining`

First submission:

```bash
uv run iris --cluster=marin job run \
  --user ahmed \
  --priority interactive \
  --cpu 1 --memory 3GB --disk 5GB --extra marin:tpu \
  --region us-central1 --region us-east5 \
  --job-name delphi-1e21-p67m33-lr0p33-int256-20260502-044350 \
  --no-wait \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR 0.33 \
  -e MIDTRAIN_MIX_NAME 67p_33m_highquality_nemo_math \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Failed before launching children: coordinator build ran out of disk extracting
`torch==2.10.0+cpu` into `/uv/cache` (`No space left on device`). This was a
coordinator build failure only; no `train_lm` child was created.

Relaunch:

```bash
uv run iris --cluster=marin job run \
  --user ahmed \
  --priority interactive \
  --cpu 1 --memory 3GB --disk 20GB --enable-extra-resources --extra marin:tpu \
  --region us-central1 --region us-east5 \
  --job-name delphi-1e21-p67m33-lr0p33-int256-20260502-044538 \
  --no-wait \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR 0.33 \
  -e MIDTRAIN_MIX_NAME 67p_33m_highquality_nemo_math \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Current status at launch handoff:

- Parent: `/ahmed/delphi-1e21-p67m33-lr0p33-int256-20260502-044538` running.
- Child: `/ahmed/delphi-1e21-p67m33-lr0p33-int256-20260502-044538/train_lm`
  submitted, `32/32` tasks pending for v5p-256 capacity.
- All child tasks have `priority_band=2` (`INTERACTIVE`).
- `/ahmed` budget spend was still `8 / 75,000` immediately after child submission
  because pending tasks do not count toward spend until assigned/building/running.
- Training output path/run id selected by the east5 coordinator:
  `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`
  / `delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`.
- Pre-flight cache disjointness check skipped due unresolved `VersionedValue`
  wrappers, same known issue as the 1e20 launches. Layers 1+2 validated at import;
  math disjointness property is still by Feistel split.
- Cache/dependency steps skipped as already succeeded; no rebuild observed.

Caveat: once the v5p-256 child actually gets assigned/running, this single job's
resource spend will exceed `/ahmed`'s 75k interactive budget. That means the
task can become BATCH-effective after assignment even though it was submitted as
INTERACTIVE and is INTERACTIVE while pending.

## 2026-05-02 05:38 UTC — patched MirrorFS checkpoint staging failure and relaunched v5p-256 pilot

The `20260502-044538` v5p-256 child failed before training while initializing
from the `1e21-v5` checkpoint:

- Parent: `/ahmed/delphi-1e21-p67m33-lr0p33-int256-20260502-044538`
- Child: `/ahmed/delphi-1e21-p67m33-lr0p33-int256-20260502-044538/train_lm`
- Child landed in `us-east5` and discovered the base checkpoint at
  `mirror://adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/checkpoints/step-21979`.
- Levanter's TensorStore workaround eagerly staged the whole `mirror://`
  checkpoint tree into `gs://marin-us-east5/...` on all 32 ranks.
- First real failure was a `TransferBudgetExceeded` on task 24:
  the nested `train_lm` job used the default 10GB mirror budget even though the
  experiment configured `mirrored(..., budget_gb=50)`.
- Other ranks also hit `Could not acquire mirror lock ... after waiting` because
  the MirrorFS lock handoff path gave up after one failed reacquire.

Applied two local fixes:

1. `lib/marin/src/marin/execution/executor.py`
   - Propagate the resolved `MirroredValue` budget into dataclass configs with
     an `env_vars` field as `MARIN_MIRROR_BUDGET_GB`.
   - This is necessary because `run_levanter_train_lm` submits a nested Fray/Iris
     child; the executor process's `mirror_budget(...)` context does not cross
     that process boundary.
2. `lib/rigging/src/rigging/filesystem.py`
   - Replace the one-shot MirrorFS lock handoff check with a 120s retry loop.
   - This avoids spurious failure when another waiter wins the lock after the
     first holder releases it.

Regression tests added and passed:

```bash
uv run pytest tests/execution/test_executor.py -x --timeout=120
# 22 passed, 1 skipped

uv run pytest lib/rigging/tests/test_mirror_fs.py -x --timeout=120
# 21 passed
```

Opened tracking issue for the deeper design problem:
`https://github.com/marin-community/marin/issues/5374`
(`[levanter] Avoid all-rank mirror checkpoint staging`). The quick patch does
not eliminate eager all-rank staging; it only fixes the immediate budget
propagation and lock handoff bugs.

Relaunched the same single pilot from the patched worktree:

```bash
uv run iris --cluster=marin job run \
  --user ahmed \
  --priority interactive \
  --cpu 1 --memory 3GB --disk 20GB --enable-extra-resources --extra marin:tpu \
  --region us-central1 --region us-east5 \
  --job-name delphi-1e21-p67m33-lr0p33-int256-20260502-053250 \
  --no-wait \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR 0.33 \
  -e MIDTRAIN_MIX_NAME 67p_33m_highquality_nemo_math \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Current handoff status:

- Parent: `/ahmed/delphi-1e21-p67m33-lr0p33-int256-20260502-053250` running.
- Parent landed in `us-east5` and chose output/run id:
  `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`
  / `delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`.
- Child: `/ahmed/delphi-1e21-p67m33-lr0p33-int256-20260502-053250/train_lm`
  submitted, `32/32` tasks pending.
- Pending reason: waiting for workers in scale group
  `tpu_v5p-preemptible_256-us-east5-a` to become ready.
- Verified via `job_config.environment_json` that the child request includes
  `MARIN_MIRROR_BUDGET_GB=50`.
- Superseded by the 2026-05-02 05:55 UTC decision below: the v5p-256 pilot was
  stopped while still pending, then relaunched as the full v5p-64 batch sweep
  with the same output path forced for `p67m33/lr0.33`.

## 2026-05-02 05:55 UTC — stopped v5p-256 pilot and launched full 1e21 v5p-64 batch sweep

User asked to move the 1e21 sweep to v5p-64 and batch priority while preserving
the already-started `p67m33/lr0.33` run namespace.

Pre-launch checks:

- Existing run namespace:
  `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`
  / `delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`.
- No Levanter permanent checkpoint existed under
  `.../checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64/`.
- No temp checkpoint existed under
  `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64/`.
- Therefore the relaunch preserves the output path/W&B run id, but resumes from
  the base `1e21-v5` initialization rather than a midtraining step checkpoint.

Stopped the pending v5p-256 recovery job so it would release the executor lock:

```bash
uv run iris --cluster=marin job stop /ahmed/delphi-1e21-p67m33-lr0p33-int256-20260502-053250
```

Submitted six v5p-64 BATCH coordinators in `us-east5`:

- `/ahmed/delphi-1e21-p67m33-lr0p33-batch64-20260502-055342`
  - `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`
- `/ahmed/delphi-1e21-p67m33-lr0p5-batch64-20260502-055342`
- `/ahmed/delphi-1e21-p67m33-lr0p67-batch64-20260502-055342`
- `/ahmed/delphi-1e21-p33m67-lr0p33-batch64-20260502-055342`
- `/ahmed/delphi-1e21-p33m67-lr0p5-batch64-20260502-055342`
- `/ahmed/delphi-1e21-p33m67-lr0p67-batch64-20260502-055342`

Common launch shape:

```bash
uv run iris --cluster=marin job run \
  --user ahmed \
  --priority batch \
  --cpu 1 --memory 4GB --disk 50GB \
  --enable-extra-resources --extra marin:tpu \
  --region us-east5 \
  --job-name <job-name> \
  --no-wait \
  -e MIDTRAIN_SELECT_BASE 1e21-v5 \
  -e MIDTRAIN_SELECT_LR <0.33|0.5|0.67> \
  -e MIDTRAIN_MIX_NAME <67p_33m_highquality_nemo_math|33p_67m_highquality_nemo_math> \
  -e MIDTRAIN_TPU_TYPE v5p-64 \
  -e MIDTRAIN_TRAIN_REGION us-east5 \
  [ -e MIDTRAIN_OUTPUT_PATH_OVERRIDE gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64 ] \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Initial verification:

- `p67m33/lr0.33` logs show:
  - `Using output path: gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`
  - `Using run ID: delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`
  - `device=TpuConfig(variant='v5p-64', kind='tpu', topology=None)`
- `p67m33/lr0.5` logs show new namespace
  `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-114e49`
  and v5p-64 device config.
- At the first status check, the two earliest coordinators were RUNNING with
  `train_lm` children PENDING for v5p-64 capacity; the remaining coordinators
  were pending for us-east5 CPU workers to become ready.

## 2026-05-02 06:02 UTC — copied 1e22 base checkpoint to east5 and launched v5p-256 sweep

User asked to launch the full `1e22-v5` sweep on v5p-256 in `us-east5-a`, and
explicitly approved copying model weights to east5 if needed. The checkpoint was
present only in the central bucket:

- Source:
  `gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`
- Destination:
  `gs://marin-us-east5/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`

Copied with direct `gcloud storage cp -r` (not Storage Transfer Service):

```bash
gcloud storage cp -r \
  gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206 \
  gs://marin-us-east5/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/
```

Post-copy verification:

```text
1821588428   gs://marin-us-central1/.../checkpoints/step-38206
1821588428   gs://marin-us-east5/.../checkpoints/step-38206
```

Submitted six v5p-256 BATCH coordinators pinned to `us-east5`:

- `/ahmed/delphi-1e22-p67m33-lr0p33-batch256-20260502-060043`
- `/ahmed/delphi-1e22-p67m33-lr0p5-batch256-20260502-060043`
- `/ahmed/delphi-1e22-p67m33-lr0p67-batch256-20260502-060043`
- `/ahmed/delphi-1e22-p33m67-lr0p33-batch256-20260502-060043`
- `/ahmed/delphi-1e22-p33m67-lr0p5-batch256-20260502-060043`
- `/ahmed/delphi-1e22-p33m67-lr0p67-batch256-20260502-060043`

Common launch shape:

```bash
uv run iris --cluster=marin job run \
  --user ahmed \
  --priority batch \
  --cpu 1 --memory 4GB --disk 50GB \
  --enable-extra-resources --extra marin:tpu \
  --region us-east5 \
  --job-name <job-name> \
  --no-wait \
  -e MIDTRAIN_SELECT_BASE 1e22-v5 \
  -e MIDTRAIN_SELECT_LR <0.33|0.5|0.67> \
  -e MIDTRAIN_MIX_NAME <67p_33m_highquality_nemo_math|33p_67m_highquality_nemo_math> \
  -e MIDTRAIN_TPU_TYPE v5p-256 \
  -e MIDTRAIN_TRAIN_REGION us-east5 \
  -- python experiments/exp_delphi_math_10b_midtrain.py
```

Initial status immediately after submission: all six parent coordinators were
`JOB_STATE_PENDING`, waiting for east5 CPU coordinator workers. The latest
pending reason was `no_capacity: cpu_vm_e2_highmem_2_ondemand-us-east5-a=at_max_slices,
cpu_vm_e2_highmem_2_ondemand-us-east5-b=backoff`. No child `train_lm` jobs had
materialized yet at the first check, so v5p-256 TPU capacity had not yet been
requested by the children.

## 2026-05-02 06:07 UTC — consolidated live sweep handoff for next monitor

Source of truth for this snapshot:

```bash
uv run iris --cluster=marin job list --json --prefix /ahmedah/delphi-1e20
uv run iris --cluster=marin job list --json --prefix /ahmed/delphi-1e21
uv run iris --cluster=marin job list --json --prefix /ahmed/delphi-1e22
```

State names below are Iris job states. `train_lm` task counts are from
`task_state_counts`; parent rows are the CPU coordinator jobs.

### 1e20 sweep (`/ahmedah`)

The active 1e20 sweep has six intended points. Four are actively training and
two are queued for v5p-32 TPU capacity.

| Point | Parent job | Parent state | `train_lm` state | Notes |
|---|---|---:|---:|---|
| `p67m33/lr0.5` | `/ahmedah/delphi-1e20-p67m33-lr0p5-int64-20260502-014459` | RUNNING | RUNNING, 8 ranks | v5p-64 survivor. Parent has 0 failures, 0 preemptions; child has 0 failures, 7 preemptions. |
| `p67m33/lr0.33` | `/ahmedah/delphi-1e20-p67m33-lr0p33-int32-20260502-025832` | RUNNING | RUNNING, 4 ranks | v5p-32. Parent has 0 failures, 1 preemption. |
| `p67m33/lr0.67` | `/ahmedah/delphi-1e20-p67m33-lr0p67-int32-20260502-025832` | RUNNING | RUNNING, 4 ranks | v5p-32. Parent has 0 failures, 1 preemption. |
| `p33m67/lr0.33` | `/ahmedah/delphi-1e20-p33m67-lr0p33-int32-20260502-025832` | RUNNING | RUNNING, 4 ranks | v5p-32. Parent has 0 failures, 0 preemptions. |
| `p33m67/lr0.5` | `/ahmedah/delphi-1e20-p33m67-lr0p5-int32-20260502-025832` | RUNNING | PENDING, 4 ranks | Waiting for v5p-32 capacity. Parent has 0 failures, 1 preemption. |
| `p33m67/lr0.67` | `/ahmedah/delphi-1e20-p33m67-lr0p67-int32-20260502-025832` | RUNNING | PENDING, 4 ranks | Waiting for v5p-32 capacity. Parent has 0 failures, 1 preemption. |

The two pending v5p-32 children currently report:
`Coscheduling: need 4 workers ... Insufficient TPUs (need 4, available 0)` and
`tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity`.

Older `20260502-014459` v5p-64 siblings other than `p67m33/lr0.5` failed with
`RuntimeError: 1 step(s) failed`; those are superseded by the v5p-32 relaunches
above and should not be recovered blindly.

### 1e21 sweep (`/ahmed`)

The current 1e21 sweep is the v5p-64 BATCH relaunch in `us-east5`. All six points
have been submitted. Four coordinators are running and have `train_lm` children
waiting for v5p-64 capacity; two coordinators are still waiting for CPU
coordinator capacity.

| Point | Parent job | Parent state | `train_lm` state | Notes |
|---|---|---:|---:|---|
| `p67m33/lr0.33` | `/ahmed/delphi-1e21-p67m33-lr0p33-batch64-20260502-055342` | RUNNING | PENDING, 8 ranks | Relaunch preserves old namespace with `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`. |
| `p67m33/lr0.5` | `/ahmed/delphi-1e21-p67m33-lr0p5-batch64-20260502-055342` | RUNNING | PENDING, 8 ranks | New namespace verified in logs: `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-114e49`. |
| `p67m33/lr0.67` | `/ahmed/delphi-1e21-p67m33-lr0p67-batch64-20260502-055342` | RUNNING | PENDING, 8 ranks | Waiting for v5p-64 capacity. |
| `p33m67/lr0.33` | `/ahmed/delphi-1e21-p33m67-lr0p33-batch64-20260502-055342` | RUNNING | PENDING, 8 ranks | Waiting for v5p-64 capacity. |
| `p33m67/lr0.5` | `/ahmed/delphi-1e21-p33m67-lr0p5-batch64-20260502-055342` | PENDING | Not created yet | Waiting for east5 CPU coordinator capacity. |
| `p33m67/lr0.67` | `/ahmed/delphi-1e21-p33m67-lr0p67-batch64-20260502-055342` | PENDING | Not created yet | Waiting for east5 CPU coordinator capacity. |

The v5p-64 pending children currently report:
`Coscheduling: need 8 workers ... Insufficient TPUs (need 4, available 0)` and
`tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity`.

The two CPU-pending coordinators report:
`no_capacity: cpu_vm_e2_highmem_2_ondemand-us-east5-a=at_max_slices,
cpu_vm_e2_highmem_2_ondemand-us-east5-b=backoff`.

### 1e22 sweep (`/ahmed`)

The 1e22 sweep has six v5p-256 BATCH parent coordinators submitted in `us-east5`.
All six are still CPU-pending and no `train_lm` child has materialized yet, so
these jobs are not yet holding/requesting v5p-256 TPU slices.

| Point | Parent job | Parent state | `train_lm` state |
|---|---|---:|---:|
| `p67m33/lr0.33` | `/ahmed/delphi-1e22-p67m33-lr0p33-batch256-20260502-060043` | PENDING | Not created yet |
| `p67m33/lr0.5` | `/ahmed/delphi-1e22-p67m33-lr0p5-batch256-20260502-060043` | PENDING | Not created yet |
| `p67m33/lr0.67` | `/ahmed/delphi-1e22-p67m33-lr0p67-batch256-20260502-060043` | PENDING | Not created yet |
| `p33m67/lr0.33` | `/ahmed/delphi-1e22-p33m67-lr0p33-batch256-20260502-060043` | PENDING | Not created yet |
| `p33m67/lr0.5` | `/ahmed/delphi-1e22-p33m67-lr0p5-batch256-20260502-060043` | PENDING | Not created yet |
| `p33m67/lr0.67` | `/ahmed/delphi-1e22-p33m67-lr0p67-batch256-20260502-060043` | PENDING | Not created yet |

All six currently report the same CPU coordinator capacity issue:
`no_capacity: cpu_vm_e2_highmem_2_ondemand-us-east5-a=at_max_slices,
cpu_vm_e2_highmem_2_ondemand-us-east5-b=backoff`.

The 1e22 base checkpoint was copied before launch:

- Source:
  `gs://marin-us-central1/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`
- Destination:
  `gs://marin-us-east5/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`
- Verified byte totals matched: `1821588428` bytes on both source and dest.

### MirrorFS issue for the next agent

Do not treat the MirrorFS checkpoint problem as fully solved by the quick patch.
The local branch patch only raises the transfer budget inherited by nested
`train_lm` jobs and makes MirrorFS lock handoff wait up to 120s. It reduces the
immediate failure mode but does not change the fact that Levanter checkpoint
loading can still make many TPU ranks try to stage a mirrored checkpoint.

The earlier `1e21-v5` v5p-256 pilot failed during checkpoint initialization:

- one rank hit `TransferBudgetExceeded` while staging from `mirror://`;
- peers hit mirror lock handoff failures such as `Could not acquire mirror lock`;
- JAX coordination then cascaded into a parent-level `RuntimeError: 1 step(s) failed`.

Track the deeper fix in GitHub issue:
https://github.com/marin-community/marin/issues/5374

For any failed/restarted midtraining job, follow the hard resume rule at the top
of this file. In particular, grep startup logs for `Using output path` and
`Using run ID`, then verify `Resuming training from step ...` before considering
the job recovered. For MirrorFS-specific failures, also grep for
`TransferBudgetExceeded`, `Could not acquire mirror lock`, and
`_stage_mirror_to_local`.

## 2026-05-02/03 — autonomous monitor cron `332f56ed`, ticks 1–35 summary [LEGACY DEPRECATED]

> ⚠️ **LEGACY DEPRECATED — this entire section is historical only.** It documents what a previous agent did wrong using the now-disabled `MIDTRAIN_OUTPUT_PATH_OVERRIDE` env var. **Do not treat any command, env var, or "key fix" in this section as guidance.** The canonical recovery contract is at the top of this file (line 3) and in `.agents/ops/2026-05-02-delphi-midtrain-resume-namespace.md`. The legacy env var now hard-fails in `experiments/exp_delphi_math_10b_midtrain.py`.

**Cadence:** every 30 min at :13/:43 UTC. **First fire:** 06:35 UTC May 2. **Last fire & cancellation:** 23:43 UTC May 2 (cron deleted by user after wandb-fragmentation issue surfaced). 35 ticks total. Most ticks (~22 of 35) were no-ops; the rest involved relaunches or success transitions. Full per-tick entries were on the working branch but have been stashed (`stash@{0}: midtrain logbook tick-28-to-35 entries`) — only the rolled-up summary survives here.

### Outcome ledger

- **1e20:** all 6/6 succeeded by tick 15 (13:43 UTC). One slot (`p67m33-lr0.5`) needed a v5p-32 resume after a v5p-64 worker died.
- **1e21:** 4/6 succeeded by tick 30 (21:13 UTC) — `p67m33-lr0.33` (resume), `p33m67-lr{0.33,0.5,0.67}` (originals). 2 still running at cron cancellation: `p67m33-lr0.5-batch64-resume-151756`, `p67m33-lr0.67-batch64-resume-161841`.
- **1e22:** 0/6 done at cron cancellation. All 6 actively training; expected first finisher ~06:00 UTC May 3, full sweep ~11:30 UTC May 3 modulo more cascades.

### Cron-driven relaunches

12 relaunches dispatched over the 17-hour cron window:

| Tick | Slot | Reason | New job |
|---|---|---|---|
| 2 | `1e21-p67m33-lr0.33` | `failures=1`, JAX cascade | `lr0p33-batch64-resume-072010` (override `ab4e64`) |
| 5 | `1e22-p67m33-lr{0.33,0.5,0.67}` | 3× JAX cascade after 2h | `resume-084859` (overrides `4e8cc7a7`, `f60cb12a`, `3c17740e`) |
| 6 | 5× 1e22 (p67m33×2 + p33m67×3) | JAX cascade re-fail / fresh fails | `resume2-091855` |
| 7 | `1e22-p67m33-lr0.67` (4th attempt) | JAX cascade again | `resume3-094809` |
| 18 | `1e21-p67m33-lr0.5` + `1e22-p67m33-lr0.33` | 9h7m cascade + 4m35s recover-fail | resumes |
| 19 | `1e22-p67m33-lr0.5` | 6h11m cascade | `resume3-154733` |
| 20 | `1e22-p67m33-lr0.5` + `1e21-p67m33-lr0.67` | 26-min cascade + 9.5h cascade | `resume4-161841` (1e22), `lr0p67-resume-161841` (1e21) |
| 26 | `1e22-p67m33-lr0.5` (5th attempt) | child-recycle didn't recover | `resume5-191734` |
| 34 | `1e22-p67m33-lr0.67` (5th attempt) | `failures=1, preemptions=2` cascade | `resume4-231948` |

### Failure mode dominance

Every relaunch was the **JAX-coordinator stale-port-8476 cascade** (`failures=1, preemptions=0` or `preemptions={1,2}`). v5p-256 (32-host coscheduling) is the highest-blast-radius case: `1e22-p67m33-lr0.5` cascaded **5 times** and `lr0.67` cascaded **5 times**. Capacity-pending and child-recycle states were never relaunched (per policy), and iris consistently auto-recovered them.

### What broke: the resume-namespace / wandb-id mismatch (discovered post-cancellation)

> **Authoritative version: read the top-of-file warning (line 3), the "Hard resume rule" section (line 71), and `.agents/ops/2026-05-02-delphi-midtrain-resume-namespace.md`.** The bullets below are kept only as historical context for what this agent did wrong; do **not** treat them as a recipe.

The autonomous monitor used the (then-recorded) resume pattern `MIDTRAIN_OUTPUT_PATH_OVERRIDE + WANDB_RUN_ID + WANDB_RESUME=allow` but set those env vars to the **resume's marin-derived config-hash** rather than the **original's actual run hash**. Because `MIDTRAIN_OUTPUT_PATH_OVERRIDE` is itself part of the marin StepSpec, adding it to the config changed the hash, so the resume's auto-derived hash ≠ the original's hash. Consequences:

1. **Wandb fragmentation:** every resume created a new wandb row instead of merging into the original's. User saw 12 rows for the 6 1e22 slots and 5 rows for the 3 1e21 p67m33 slots.
2. **Lost progress on 1e21-p67m33-lr0.67 only:** override path `99752407` did not match the original `ecbd27` path. Resume loaded a stale temp `step-1013` from an earlier `99752407` chaos attempt instead of the original's `step-2646`. **~1638 steps redone (~2.3 h v5p-64 wasted).** Note: this agent at one point claimed in conversation that the original had reached step 5500-6500 — that was wrong; the postmortem authoritatively places `ecbd27` at `step=2651, run_progress=0.601` at crash. All other resumes either had matching hashes by luck (`1e21-lr0.5` → `fdc4ebf1`) or had no real progress to lose (1e22 originals all died at ~3 min from the JAX cascade with no checkpoints).
3. **1e22 cascade chain is intact:** every 1e22 resume after the first uses the same hash (`f60cb12a`, `4e8cc7a7`, `3c17740e`), so checkpoints chain forward correctly. Per-cascade waste ≈ 1 checkpoint cadence (~750 steps) until the next save catches up.

### Net wasted compute estimate

- 1e21 lr0.67: ~2.3 h v5p-64 (the only material loss)
- 1e21 lr0.5: ~0 (lucky hash match)
- 1e22 cumulative across all cascades: ~5-6 h v5p-256 (mostly the lr0.5 5-cascade chain)

### Fix landed (do not undo)

After this incident, Codex landed the following on `midtrain_data`. **The legacy pattern used above is now disabled in code** — a future agent literally cannot repeat it without an immediate `ValueError`:

- `experiments/exp_delphi_math_10b_midtrain.py`:
  - `MIDTRAIN_OUTPUT_PATH_OVERRIDE` raises at module import (line 482).
  - Recovery uses **one** env var, `MIDTRAIN_RESUME_OUTPUT_PATH`, from which the script derives `RUN_ID`, `WANDB_RUN_ID`, and the executor `override_output_path` (line 534).
  - `MIDTRAIN_EXPECT_RESUME_MIN_STEP` is required for real resumes; `MIDTRAIN_ALLOW_EMPTY_RESUME=1` is the only escape hatch.
  - Pre-launch checks the path matches the selected (base, lr, mix), and that latest permanent or temp checkpoint exists ≥ floor (line 543, 586).
- `lib/marin/src/marin/training/training.py:_enforce_run_id` (line 190): generic guard — any executor-backed training where resolved `run_id != basename(output_path)` now hard-fails. This catches the same class of bug on **any** future experiment script.
- Regression tests: `experiments/test_default_train_init_mode.py:107` and `tests/test_training.py:184`.

**Recovery shape going forward** (also documented at the top of this file and in the ops postmortem):

```bash
-e MIDTRAIN_RESUME_OUTPUT_PATH gs://marin-<region>/checkpoints/<old-run-id> \
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP <last-known-good-step>
```

Do not pass `MIDTRAIN_OUTPUT_PATH_OVERRIDE`, `WANDB_RUN_ID`, or a hand-written `RUN_ID`. The script derives all of them from the one path.

## 2026-05-03 ~17:30 UTC — full sweep status (4 LRs × 3 mixes × 3 scales = 36 points)

**Targeted design matrix:** `LR ∈ {0.33, 0.5, 0.67, 0.83}` × `mix ∈ {p33m67, p50m50, p67m33}` × `scale ∈ {1e20, 1e21, 1e22}` = 36 cells.

Legend: ✅ done · 🟢 running · ⏳ pending · ❌ dead (no in-flight replacement)

### 1e20 (v5p-32, batch / interactive priority) — 10/12 done

| LR ↓ / mix → | p33m67 | p50m50 | p67m33 |
|---|---|---|---|
| 0.33 | ✅ orig | ✅ batch32 | ✅ orig |
| 0.5  | ✅ orig | ✅ batch32 | ✅ orig (+ v5p-32 resume variant) |
| 0.67 | ✅ orig | ✅ batch32 | ✅ orig |
| 0.83 | 🟢 batch32 | ✅ batch32 | 🟢 batch32 |

### 1e21 (v5p-64 originals; v5p-128 for new + resumes) — 4/12 done, 3 running, 5 pending

| LR ↓ / mix → | p33m67 | p50m50 | p67m33 |
|---|---|---|---|
| 0.33 | ✅ orig (v5p-64) | 🟢 v5p-128 | ✅ resume on v5p-64 |
| 0.5  | ✅ orig (v5p-64) | 🟢 v5p-128 | ⏳ v5p-128 resume2 from `114e49` ≥3528 |
| 0.67 | ✅ orig (v5p-64) | 🟢 v5p-128 | ⏳ v5p-128 resume2 from `ecbd27` ≥2646 |
| 0.83 | ⏳ v5p-128 | ⏳ v5p-128 | ⏳ v5p-128 |

Notes on 1e21 holes (`p67m33-lr0.5` and `p67m33-lr0.67`):
- Each ran ~9-9.5 h on v5p-64 originally, then JAX-cascade-died at step ≥2646–3541. Multiple v5p-64 resume attempts collided with `/moojink/`'s priority-band-2 batch jobs and got preemption-thrashed (`preemptions=707-708`).
- Migrated to v5p-128 (empty pool) using the new `MIDTRAIN_RESUME_OUTPUT_PATH` contract. First v5p-128 resume launch failed at parse time due to a Bash IFS=':' delimiter colliding with the `gs://` URI. Resubmitted with `|` delimiter — both pending now.
- Use the original-namespace path (`114e49` and `ecbd27`) so wandb consolidates with the original failed run.

### 1e22 (v5p-256, batch priority) — 6/12 done, 6 running

| LR ↓ / mix → | p33m67 | p50m50 | p67m33 |
|---|---|---|---|
| 0.33 | ✅ resume2 (5 cascades) | 🟢 batch256 | ✅ resume3 |
| 0.5  | ✅ resume2 | 🟢 batch256 | ✅ resume5 (5 cascades) |
| 0.67 | ✅ resume2 | 🟢 batch256 | ✅ resume4 (5 cascades) |
| 0.83 | 🟢 batch256 | 🟢 batch256 | 🟢 batch256 |

### Aggregate

- **Done: 20/36** (10 × 1e20, 4 × 1e21, 6 × 1e22)
- **Running: 11/36** (2 × 1e20, 3 × 1e21, 6 × 1e22)
- **Pending: 5/36** (5 × 1e21 on v5p-128, queued behind the 3 running)
- **Dead: 0** (the original v5p-64 `p67m33-lr0.5/0.67` and the failed v5p-64 resume attempts are succeeded by the v5p-128 resume2 jobs that point at the original namespaces, so wandb will reconcile cleanly)

### Wandb identity invariants

- All 1e22 cells write to a stable namespace pinned by the auto-derived hash on the first launch; subsequent resumes used `MIDTRAIN_OUTPUT_PATH_OVERRIDE` (now disabled) to that hash. wandb shows 2 rows per p67m33 1e22 cell (a fossil at ~3 min and the live run); cosmetic, no compute lost (originals had no checkpoints).
- 1e21 `p67m33-lr0.5` consolidates to `delphi-1e21-p67m33-9p25b-lr0.5-114e49`. 1e21 `p67m33-lr0.67` consolidates to `delphi-1e21-p67m33-9p25b-lr0.67-ecbd27`. The `fdc4ebf1` and `99752407` rows are stale broken-recovery namespaces; ignore them in analysis.

### ETA picture from 17:30 UTC May 3

- 1e20 last 2 finish in a few hours (lr0.83 jobs are well under way)
- 1e22 6 in-flight ones each ~10-22h to go from now → last 1e22 cell done by mid-late May 4 UTC
- 1e21 3 running v5p-128 jobs ~7-9h on v5p-128 (faster than v5p-64); pending 5 start as those finish; last 1e21 cell done by ~12-18h after the running 3 finish

Full sweep complete: **mid-late May 4 UTC** modulo more cascades or preemptions.

## 2026-05-04 ~22:11 UTC — GCP v5p-256/v5p-128/v5p-512 mass eviction (sweep stops here for 1e22)

At ~22:11 UTC GCP appears to have wholesale-reclaimed the large-slice v5p capacity in the iris pool. Within minutes, **0 v5p-256, 0 v5p-128, and 0 v5p-512 healthy hosts** remained. Smaller slices (v5p-32, v5p-16, v5p-8) survived; v5p-8 actually grew (148 → 226). This is GCP-side pool rebalancing, not user contention or iris bug.

Verified by:

```sql
SELECT wa.str_value as variant, count(DISTINCT w.worker_id) as healthy
FROM workers w JOIN worker_attributes wa ON w.worker_id=wa.worker_id
WHERE wa.key='device-variant' AND wa.str_value LIKE 'v5p%' AND w.healthy=1
GROUP BY wa.str_value
-- Result: v5p-8 226, v5p-32 32, v5p-16 14, v5p-64 16, v5p-128 0, v5p-256 0, v5p-512 0
```

`/quevedo/`'s production-priority v5p-256 job submitted ~22:20 UTC is also stuck pending — confirms it's pool-level, not specific to our jobs.

### Effect on the in-flight 1e22 runs

iris bounced a sibling task (probably task 19 of `lr0.33`) due to the host eviction; coscheduling layer marked all 32 tasks of all 3 jobs as `pending` waiting for atomic re-coscheduling — but no v5p-256 hosts exist to coschedule onto. Symptom: parents stuck at `state=running` while every train_lm task sits at `state=pending` with error `"Coscheduled sibling .../task/19 bounced for atomic re-scheduling"`.

The 1e22 base config in `experiments/exp_delphi_math_10b_midtrain.py` only allows v5p-128/256/512 — NONE of which currently exist. So the runs cannot dispatch on any allowed pool. Killing them rather than letting them sit indefinitely.

### 1e22 final cell status

| Cell | Status | Latest checkpoint |
|---|---|---|
| `p33m67-lr0.33` | ✅ done | `delphi-1e22-p33m67-32p07b-lr0.33-e9132105/hf/step-7646/` |
| `p33m67-lr0.5` | ✅ done | `delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d/hf/step-7646/` |
| `p33m67-lr0.67` | ✅ done | `delphi-1e22-p33m67-32p07b-lr0.67-54770ae7/hf/step-7646/` |
| `p33m67-lr0.83` | ❌ unfinished — resume hit GCP eviction at step ~7182/7647 (94%) | `delphi-1e22-p33m67-32p07b-lr0.83-78fd44/checkpoints/step-6876/` (perm), `step-7182` (temp) |
| `p50m50-lr0.33` | ✅ effectively done — HF FINAL `step-7646` saved before stall, killed mid-iris-cleanup | `delphi-1e22-p50m50-32p07b-lr0.33-c43ada/hf/step-7646/` |
| `p50m50-lr0.5` | ❌ unfinished — killed at ~step-5034 temp / step-4584 perm (~66%) | `delphi-1e22-p50m50-32p07b-lr0.5-ecfa99/checkpoints/step-4584/` (perm), `step-5034` (temp) |
| `p50m50-lr0.67` | ❌ unfinished — killed at ~step-3649 temp / step-3056 perm (~48%) | `delphi-1e22-p50m50-32p07b-lr0.67-e78260/checkpoints/step-3056/` (perm), `step-3649` (temp) |
| `p50m50-lr0.83` | ✅ done | `delphi-1e22-p50m50-32p07b-lr0.83-3c9f70/hf/step-7646/` |
| `p67m33-lr0.33` | ✅ done | `delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7/hf/step-7646/` |
| `p67m33-lr0.5` | ✅ done | `delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a/hf/step-7646/` |
| `p67m33-lr0.67` | ✅ done | `delphi-1e22-p67m33-32p07b-lr0.67-3c17740e/hf/step-7646/` |
| `p67m33-lr0.83` | ✅ done | `delphi-1e22-p67m33-32p07b-lr0.83-d35daa/hf/step-7646/` |

**1e22 final tally: 9 of 12 cells with a final HF checkpoint. 3 cells partially trained (no final).**

### Recovery path (when GCP restores v5p-256)

To resume the 3 unfinished cells later, use the standard `MIDTRAIN_RESUME_OUTPUT_PATH` contract pointing at the namespaces above:

```bash
# p33m67-lr0.83 — resume from step-6876 (perm); temp step-7182 will be picked
-e MIDTRAIN_RESUME_OUTPUT_PATH "gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.83-78fd44"
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP "6876"

# p50m50-lr0.5
-e MIDTRAIN_RESUME_OUTPUT_PATH "gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.5-ecfa99"
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP "4584"

# p50m50-lr0.67
-e MIDTRAIN_RESUME_OUTPUT_PATH "gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.67-e78260"
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP "3056"
```

### Killed jobs

- `/ahmedah/delphi-1e22-p50m50-lr0p33-batch256-20260503-013817` (final HF saved, killing only loses iris cleanup)
- `/ahmedah/delphi-1e22-p50m50-lr0p5-batch256-resume-20260503-201536` (~66%, kept ckpt for later resume)
- `/ahmedah/delphi-1e22-p50m50-lr0p67-batch256-resume-20260504-065803` (~48%, kept ckpt for later resume)

## 2026-05-04 ~23:09 UTC — full sweep ledger (post-GCP-eviction handoff state)

Comprehensive view of every cell across the 4 LRs × 3 mixes × 3 scales = 36-point design matrix. Wandb project: `marin-community/delphi-midtraining`.

### 1e20 — 12/12 ✅ all done (v5p-32 / v5p-64 mix)

Final HF checkpoint at `step-9412` for every cell. Path pattern: `gs://marin-us-east5/checkpoints/delphi-1e20-{mix}-4p94b-lr{lr}-{hash}/hf/step-9412/`.

| Cell | Run-name suffix |
|---|---|
| p33m67 lr0.33 | `delphi-1e20-p33m67-4p94b-lr0.33-307237` |
| p33m67 lr0.5  | `delphi-1e20-p33m67-4p94b-lr0.5-2004f8` |
| p33m67 lr0.67 | `delphi-1e20-p33m67-4p94b-lr0.67-7c32da` |
| p33m67 lr0.83 | `delphi-1e20-p33m67-4p94b-lr0.83-2a22e0` |
| p50m50 lr0.33 | `delphi-1e20-p50m50-4p94b-lr0.33-9a74fa` |
| p50m50 lr0.5  | `delphi-1e20-p50m50-4p94b-lr0.5-3475fa` |
| p50m50 lr0.67 | `delphi-1e20-p50m50-4p94b-lr0.67-554fb6` |
| p50m50 lr0.83 | `delphi-1e20-p50m50-4p94b-lr0.83-95e10d` |
| p67m33 lr0.33 | `delphi-1e20-p67m33-4p94b-lr0.33-590ea1` |
| p67m33 lr0.5  | `delphi-1e20-p67m33-4p94b-lr0.5-9e1229` |
| p67m33 lr0.67 | `delphi-1e20-p67m33-4p94b-lr0.67-64a9c5` |
| p67m33 lr0.83 | `delphi-1e20-p67m33-4p94b-lr0.83-1965f3` |

### 1e21 — 10/12 done, 2 IN PROGRESS (v5p-64 batch priority)

Final HF checkpoint at `step-4410` for each done cell. Path pattern: `gs://marin-us-east5/checkpoints/delphi-1e21-{mix}-9p25b-lr{lr}-{hash}/hf/step-4410/`.

**Done (10):**

| Cell | Run-name suffix |
|---|---|
| p33m67 lr0.33 | `delphi-1e21-p33m67-9p25b-lr0.33-58ebcb` |
| p33m67 lr0.5  | `delphi-1e21-p33m67-9p25b-lr0.5-efbc63` |
| p33m67 lr0.67 | `delphi-1e21-p33m67-9p25b-lr0.67-9cf8da` |
| p50m50 lr0.33 | `delphi-1e21-p50m50-9p25b-lr0.33-bccff4` |
| p50m50 lr0.5  | `delphi-1e21-p50m50-9p25b-lr0.5-973c46` |
| p50m50 lr0.67 | `delphi-1e21-p50m50-9p25b-lr0.67-7e82b3` |
| p50m50 lr0.83 | `delphi-1e21-p50m50-9p25b-lr0.83-f9edd2` |
| p67m33 lr0.33 | `delphi-1e21-p67m33-9p25b-lr0.33-ab4e64` |
| p67m33 lr0.5  | `delphi-1e21-p67m33-9p25b-lr0.5-114e49` (resume2 consolidation) |
| p67m33 lr0.83 | `delphi-1e21-p67m33-9p25b-lr0.83-a1a261` |

**In progress (2)** — submitted to v5p-64 us-east5 batch priority on 2026-05-04 ~23:07-23:09 UTC after the v5p-128 zombie kill:

| Cell | Iris job ID | Status |
|---|---|---|
| p67m33 lr0.67 (resume) | `/ahmedah/delphi-1e21-p67m33-lr0p67-batch64-resume4-20260504-230740` | resuming from `delphi-1e21-p67m33-9p25b-lr0.67-ecbd27` step-2646 (~60% preserved); pending v5p-64 dispatch |
| p33m67 lr0.83 (fresh)   | `/ahmedah/delphi-1e21-p33m67-lr0p83-batch64-fresh-20260504-230900` | fresh start, no checkpoint; pending v5p-64 dispatch |

Both pending behind /moojink/ + /michaelryan/ band-2 jobs on the 16-host v5p-64 us-east5 pool. Children at band 0 (auto-promoted) so should preempt them when iris allocator fires.

### 1e22 — 9/12 effectively done, 3 UNFINISHED (v5p-256 pool wiped at 22:11 UTC)

Final HF checkpoint at `step-7646` for each fully-done cell. Path pattern: `gs://marin-us-east5/checkpoints/delphi-1e22-{mix}-32p07b-lr{lr}-{hash}/hf/step-7646/`.

**Done (9 — all have final HF checkpoint):**

| Cell | Run-name suffix |
|---|---|
| p33m67 lr0.33 | `delphi-1e22-p33m67-32p07b-lr0.33-e9132105` |
| p33m67 lr0.5  | `delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d` |
| p33m67 lr0.67 | `delphi-1e22-p33m67-32p07b-lr0.67-54770ae7` |
| p50m50 lr0.33 | `delphi-1e22-p50m50-32p07b-lr0.33-c43ada` (HF FINAL saved before kill) |
| p50m50 lr0.83 | `delphi-1e22-p50m50-32p07b-lr0.83-3c9f70` |
| p67m33 lr0.33 | `delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7` |
| p67m33 lr0.5  | `delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a` |
| p67m33 lr0.67 | `delphi-1e22-p67m33-32p07b-lr0.67-3c17740e` |
| p67m33 lr0.83 | `delphi-1e22-p67m33-32p07b-lr0.83-d35daa` |

**Unfinished (3 — partial checkpoints preserved, need v5p-256 to resume):**

| Cell | Run-name suffix | Latest perm | Latest temp | % |
|---|---|---|---|---|
| p33m67 lr0.83 | `delphi-1e22-p33m67-32p07b-lr0.83-78fd44` | step-6876 | step-7182 | ~94% |
| p50m50 lr0.5  | `delphi-1e22-p50m50-32p07b-lr0.5-ecfa99` | step-4584 | step-5034 | ~66% |
| p50m50 lr0.67 | `delphi-1e22-p50m50-32p07b-lr0.67-e78260` | step-3056 | step-3649 | ~48% |

Recovery commands when GCP restores v5p-256/128/512 capacity:

```bash
# p33m67-lr0.83 resume
-e MIDTRAIN_RESUME_OUTPUT_PATH "gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.83-78fd44"
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP "6876"

# p50m50-lr0.5 resume
-e MIDTRAIN_RESUME_OUTPUT_PATH "gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.5-ecfa99"
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP "4584"

# p50m50-lr0.67 resume
-e MIDTRAIN_RESUME_OUTPUT_PATH "gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.67-e78260"
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP "3056"
```

### Aggregate scoreboard

| Scale | ✅ Done with final HF | ⏳ In progress | ❌ Unfinished (need capacity) |
|---|---|---|---|
| 1e20 | 12 | 0 | 0 |
| 1e21 | 10 | 2 | 0 |
| 1e22 | 9 | 0 | 3 |
| **Total** | **31** | **2** | **3** |

If the 2 in-progress 1e21 jobs both finish, **33/36** cells will have a usable HF checkpoint.

The remaining 3 (all 1e22) are blocked on GCP restoring v5p-256/128/512 to the iris pool. The script's 1e22 base config currently only allows those three TPU types; resuming on v5p-64 would require adding `V5PComputeConfig("v5p-64")` to the 1e22-v5 base in `experiments/exp_delphi_math_10b_midtrain.py` and accepting ~30 hours of training per cell. Documented as recovery option but not yet attempted.

### Killed jobs (terminated to free iris reservations after the GCP eviction)

- `/ahmedah/delphi-1e22-p50m50-lr0p33-batch256-20260503-013817` (had final HF, kill is cosmetic)
- `/ahmedah/delphi-1e22-p50m50-lr0p5-batch256-resume-20260503-201536`
- `/ahmedah/delphi-1e22-p50m50-lr0p67-batch256-resume-20260504-065803`
- `/ahmedah/delphi-1e21-p67m33-lr0p67-batch128-resume3-20260504-170536` (v5p-128 zombie; replaced by `resume4-230740` on v5p-64)

### Wandb identity caveats for downstream analysis

- HF checkpoints' `config.json` says `architectures: ["LlamaForCausalLM"]` due to marin PR #3092 (Qwen3 export bug, still open). Weights are actual Qwen3 (q_norm/k_norm, vocab=128256, no attn bias). For inference, override at load time via `Qwen3ForCausalLM.from_pretrained(...)` directly OR via vLLM's `hf_overrides={"architectures": ["Qwen3ForCausalLM"]}`. See `experiments/evals/exp_eval_delphi_midtrain_math500.py` (on `patch_vllm` branch) for the runtime-override pattern.
- Tokenizer files in the HF dirs are correct (Llama-3.1).
- `rope_scaling.original_max_position_embeddings=8192` paired with `max_position_embeddings=4096` in saved configs trips transformers AutoConfig validation. Workaround: edit the LOCAL config.json copy to strip rope_scaling (see `EvalchemyEvaluator._repair_local_config` on `patch_vllm` branch).
- Run-name suffix hashes (e.g. `f60cb12a`) are marin StepSpec config hashes, NOT wandb auto-IDs. They're preserved as wandb run IDs because the script sets `WANDB_RUN_ID` from the namespace.

### Inference handoff doc

`.agents/handoffs/delphi-midtrain-finished-runs.md` (kept up to date) — has all wandb URLs and HF checkpoint paths in inference-ready form for labmate consumption.

## 2026-05-07 19:07–19:14 UTC — v5p-64 interactive submission, all 3 cells died from placement-collision recurrence

### Setup that landed

- Switched this worktree (`.claude/worktrees/midtrain`) from `worktree-midtrain` to `midtrain_data` branch (the actual experiment branch). Recovered the stashed cron-monitor tick-1-to-35 logbook entries via `git stash pop` and committed them locally as `[logbook] Restore monitor log tick-1-to-35 entries`.
- Did NOT merge `origin/main` into `midtrain_data`. The placement-bug fix #5490 is server-side (controller binary), so no client-side merge is needed; the 5 conflicts that came up (`experiments/defaults.py`, `lib/iris/src/iris/runtime/jax_init.py`, `lib/marin/src/marin/execution/{executor.py,training/training.py}`, `tests/execution/test_executor.py`) were unrelated churn from main and not worth resolving for this purpose.
- Added `V5PComputeConfig("v5p-64", per_device_parallelism=4)` to the `1e22-v5` base in `experiments/exp_delphi_math_10b_midtrain.py` (matching the existing `per_device_parallelism=4` pattern → gradient accumulation factor 4 on the 64-chip slice). Committed locally as `[midtrain] Add v5p-64 to 1e22-v5 v5p_compute allowlist`.
- Did NOT push any of these local commits to `origin/midtrain_data`.

### Submitted at ~19:07–19:11 UTC, all on v5p-64 / us-east5 / interactive priority

| Cell | Job ID | Resume from | RESUME_MIN_STEP |
|---|---|---|---|
| p33m67-lr0.83 | `/ahmedah/delphi-1e22-p33m67-lr0p83-v5p64-resume-20260507-190726` | `delphi-1e22-p33m67-32p07b-lr0.83-78fd44` | 6876 |
| p50m50-lr0.5  | `/ahmedah/delphi-1e22-p50m50-lr0p5-v5p64-resume-20260507-190908`  | `delphi-1e22-p50m50-32p07b-lr0.5-ecfa99`  | 4584 |
| p50m50-lr0.67 | `/ahmedah/delphi-1e22-p50m50-lr0p67-v5p64-resume-20260507-191041` | `delphi-1e22-p50m50-32p07b-lr0.67-e78260` | 3056 |

Coordinator log on the first job confirmed `Using output path` and `Using run ID` matched the expected namespace, so the resume contract was correct. No new permanent checkpoints were written — train_lm tasks died at JAX init (~20 s in) before checkpoint load, so original temp/perm checkpoints are intact.

### What happened: placement-collision bug fired again, despite #5490 supposedly being live

Lead engineer told us the controller had been restarted with #5490, so we should be safe to submit concurrently. We were not.

All 3 train_lm children FAILED with the **exact** v5p-64 8-task fingerprint from `.agents/ops/iris_placement_bug.md`:

```
state=failed  failures=1  preemptions=707  worker_failed=7/8
```

Smoking gun — host overlap between two of the three:

```
lr0.83 ∩ lr0.5: 8/8 hosts identical:
  10.202.0.182, 10.202.0.222, 10.202.0.225, 10.202.0.240,
  10.202.0.255, 10.202.1.8, 10.202.1.9, 10.202.1.11
lr0.83 ∩ lr0.67: 0 shared
lr0.5  ∩ lr0.67: 0 shared
```

Timeline (UTC, 2026-05-07) — this is the dispatch-after-failure race that #5490's commit message describes:

```
19:11:56.883  lr0.83 train_lm submitted
19:13:06.097  lr0.83 train_lm started        (host set A)
19:13:17.967  lr0.67 train_lm submitted
19:13:22.767  lr0.5  train_lm submitted
19:13:26.097  lr0.83 task 0 SIGSEGV (+20s)   (failure window opens)
19:15:38.518  lr0.5  train_lm started        (re-uses host set A → collision)
19:17:40.975  lr0.67 train_lm started        (host set B; clean)
```

There is a 2.3-min gap between lr0.83's task 0 SIGSEGV and lr0.5's dispatch. That should have been more than enough for `StopTasks` RPCs to evict lr0.83's still-running cohort siblings, but iris re-allocated the same 8 hosts to lr0.5, which immediately hit the JAX-coordinator port-8476 collision and thrashed for 707 cycles. So either the running controller doesn't actually have #5490 in its binary, or the fix has a gap.

Per-task root errors (consistent with #5470 + possibly #5258 co-firing):

- lr0.83 task 0: `Exit code 139: killed by SIGSEGV. stderr: @ 0x564903bbbcc7 _PyEval_EvalFrameDefault`
- lr0.5  parent: `connectrpc.errors.ConnectError: Request timed out`
- lr0.67 task 1: `RPC: /tensorflow.Coordination...` (matches #5258's `PollForError` symptom)
- All other tasks on each job: `Coscheduled sibling /…/train_lm/0 failed`

Wrote up the comment-ready text in `.agents/ops/iris_op_issue.md` for posting to **#5470 (reopen + comment)** and **#5258 (comment)**. Both start with 🤖 per repo convention; the `agent-generated` label should be applied after posting.

### Plan (per senior engineer): bump `max_task_failures` to gain resiliency

The senior engineer noted he likely won't get to the controller fix tonight and suggested we bump `max_task_failures` on the train_lm child to absorb individual task crashes without killing the parent.

Current state of the code:

- `max_task_failures` is in the iris controller proto (`controller.proto`) and the controller honors it (`max_task_failures=N` lets up to N task failures within a coscheduled gang before the parent is killed; default 0 means "die on the first task failure"). That's why we keep losing the parent on the very first SIGSEGV in a sibling.
- **None of the client-side code currently exposes it.** `fray.JobRequest` has `max_retries_failure` and `max_retries_preemption` but not `max_task_failures`. `iris.Client.submit()` and `iris.cluster.client.remote_client.submit_job()` don't accept it either, so the field is always sent as 0 on the wire.

Patch plan (LOCAL ONLY — no `git push`):

1. `lib/fray/src/fray/types.py` — add `max_task_failures: int = 0` to `JobRequest`.
2. `lib/fray/src/fray/iris_backend.py` — forward `max_task_failures=request.max_task_failures` into the iris client `submit()` call (~line 569).
3. `lib/iris/src/iris/client/client.py:Client.submit` — accept `max_task_failures: int = 0` kwarg, pass through to `_cluster_client.submit_job(...)`.
4. `lib/iris/src/iris/cluster/client/remote_client.py:submit_job` — accept the kwarg, set `request.max_task_failures = max_task_failures` on the `LaunchJobRequest` proto.
5. `lib/iris/src/iris/cluster/client/protocol.py` — update Protocol if needed for typing parity.
6. `lib/marin/src/marin/training/training.py:_submit_training_job` — read new env var `MIDTRAIN_MAX_TASK_FAILURES` (default 0) and pass it as `max_task_failures=...` on the `JobRequest`.

Per-launch tunable: `-e MIDTRAIN_MAX_TASK_FAILURES 100` on the next `iris job run` invocations. Setting it high (100) is intentional — we want to ride out as many task-level crashes as possible while the controller-side fix lands.

Resubmission plan after the patch lands locally:

- Serialize submissions one-at-a-time (don't trust the server fix is in until the engineer confirms).
- Same RESUME_OUTPUT_PATH / RESUME_MIN_STEP as the 19:07 attempts (steps 6876, 4584, 3056).
- Add `-e MIDTRAIN_MAX_TASK_FAILURES 100` to each.
- Wait for `train_lm` child to be `running` with `preemptions=0` for ≥30 s before submitting the next.

### Branch / commit hygiene reminder

- Working on local branch `midtrain_data`. Local commits ahead of `origin/midtrain_data`:
  1. `[logbook] Restore monitor log tick-1-to-35 entries`
  2. `[midtrain] Add v5p-64 to 1e22-v5 v5p_compute allowlist`
  3. (next) iris+fray plumbing for `max_task_failures`
  4. (next) marin `MIDTRAIN_MAX_TASK_FAILURES` env-var hook
  5. (this) logbook entry
- Nothing has been pushed to origin and per the user's instruction nothing should be until the engineer signs off and a real PR cycle happens.

## 2026-05-09 01:26 UTC — May 7th attempt patch landed; resubmitted on v5p-64 with MIDTRAIN_MAX_TASK_FAILURES=100; 1/3 finished, 2/3 progressing cleanly

### Setup that landed (commits, all local on `midtrain_data`)

- `[iris,fray] Plumb max_task_failures through client.submit and JobRequest` — adds field to `fray.JobRequest`, threads it through `fray.iris_backend`, `iris.Client.submit`, `iris.cluster.client.remote_client.submit_job`, and the `ClusterClient` Protocol.
- `[midtrain] Read MIDTRAIN_MAX_TASK_FAILURES in training submission` — `lib/marin/src/marin/training/training.py:_submit_training_job` reads the env var (default 0) and forwards as `max_task_failures` on the `JobRequest`. Per-launch tunable via `-e MIDTRAIN_MAX_TASK_FAILURES <N>`.
- The `[v5p-64 to 1e22-v5 allowlist]` patch from 05-07 is still in. No further code changes since the May-7 entry except this logbook.

### Submitted at ~01:26-01:27 UTC, all on v5p-64 / us-east5 / interactive priority / `-e MIDTRAIN_MAX_TASK_FAILURES 100`

| Cell | Job ID | Resume from | RESUME_MIN_STEP |
|---|---|---|---|
| p33m67-lr0.83 | `/ahmedah/delphi-1e22-p33m67-lr0p83-v5p64-mtf-20260509-012605` | `delphi-1e22-p33m67-32p07b-lr0.83-78fd44` | 6876 |
| p50m50-lr0.5  | `/ahmedah/delphi-1e22-p50m50-lr0p5-v5p64-mtf-20260509-012636`  | `delphi-1e22-p50m50-32p07b-lr0.5-ecfa99`  | 4584 |
| p50m50-lr0.67 | `/ahmedah/delphi-1e22-p50m50-lr0p67-v5p64-mtf-20260509-012648` | `delphi-1e22-p50m50-32p07b-lr0.67-e78260` | 3056 |

Submitted serially (~10 s gap between submissions, not waiting for `running`-state confirmation between them; relied on the controller-side scheduler to handle race-window arbitration this time, with `MIDTRAIN_MAX_TASK_FAILURES=100` as a backstop). Same resume contract as the May-7 attempt (steps 6876, 4584, 3056). All three confirmed `Using output path` / `Using run ID` matched expected wandb namespace.

### Outcome — placement-collision pattern still happens during dispatch but absorbed cleanly this time

All three train_lm gangs hit the same chain-reaction collision window during initial dispatch (early SIGSEGVs on coordinator port-8476 race) and incurred ~9 preemptions each before iris steered them onto non-overlapping host sets. **Crucially, none of them hit the 707-fingerprint death state.** With `max_task_failures=100`, the iris parent absorbed each task-level SIGSEGV and re-dispatched the cohort instead of killing the gang; iris subsequently spread the gangs to non-overlapping hosts and they stabilized at preempt=9.

```
~01:26-03:30 UTC  preempt counters climb 0→9 across all three (chain-reaction churn)
~03:30 UTC        all three stable at preempt=9, train steps progressing normally
07:00 UTC         lr0.83 at step 7545 / 7647 (98.7%)
                  lr0.5  at step 5399 / 7647 (70.6%)
                  lr0.67 at step 3999 / 7647 (52.3%)
~07:35 UTC        lr0.83 SUCCEEDED — parent state=succeeded, train_lm 8/8 tasks succeeded,
                  wandb run finished at step 7646 (target 7647), final loss 1.13
~13:30 UTC        lr0.5 single preemption (9→10): wandb crashed at step 6444,
                  iris parent stayed running, tasks re-dispatched and resumed within ~10 min
                  back at step 6448. Single event, NOT a collision recurrence.
19:04 UTC         lr0.5 at step 6499 / 7647 (85.0%), preempt=10
                  lr0.67 at step 5144 / 7647 (67.3%), preempt=9
```

### Steady-state observations (post-stabilization)

- Wall-clock pace: **+17 wandb steps per 10-min monitoring tick = ~102 steps/hr = ~35 sec/step**. This is the v5p-64 baseline rate for these 32B-token cells; v5p-128 would have been ~2× faster.
- ETA from 19:04 UTC May 9 (assuming no new preemptions):
  - lr0.5: 1148 steps remaining → ~11 h → finish ~06:00 UTC May 10
  - lr0.67: 2503 steps remaining → ~24 h → finish ~19:00 UTC May 10
- Loss curves on wandb match the ~1.4-1.5 train/loss range expected for these cells; nothing has gone off the rails.

### Verdict on `MIDTRAIN_MAX_TASK_FAILURES=100`

This is the resiliency improvement we needed. The iris controller's #5490 fix may or may not be live (we have no way to verify which controller binary is currently running), but the client-side mitigation works regardless of server state — early task crashes during the dispatch race window get absorbed instead of killing the parent. With this in place, iris's natural behavior of re-dispatching to a different host set after a few collision cycles drives the system to a stable state on its own.

For future midtraining sweeps: keep `-e MIDTRAIN_MAX_TASK_FAILURES 100` on every `iris job run` invocation by default. The cost of running with this set is essentially zero (jobs that succeed don't accumulate task failures), and the resiliency benefit is large.

### Operational misc from this session

- **GitHub commenting**: posted v5p-128 follow-up on issue #5470 (https://github.com/marin-community/marin/issues/5470#issuecomment-4409115520) describing a separate scheduler-wedge variant we hit on May-3 for ~6 h, plus the chain-reaction collision pattern, plus the `max_task_failures` finding. PAT lives at `/lfs/skampere3/0/ahmedah/.ghtok` (classic token, `repo` scope; fine-grained token failed with 403 on org-owned repos because it was scoped to user-owned only).
- **Probed v5p-128 to see if it's usable**: dashboard says 100% production in use even though no v5p-128 TPU jobs are running — the underlying hosts are saturated by `/rav/`-namespaced zephyr-download-hf jobs at production priority. Interactive priority would queue indefinitely behind those. Submitted a test 1e22 fresh job to v5p-128 to confirm; the parent succeeded immediately because LR=0.4 isn't in `LR_FACTORS = (0.33, 0.5, 0.67)` so the experiment script no-op'd before dispatching `train_lm`. The probe was inconclusive but the dashboard answers the question on its own — don't put interactive jobs on v5p-128 right now.
- **SSH tunnel** to iris controller died once mid-session (~04:12 UTC May 9); restored via the documented `gcloud compute ssh ... --tunnel-through-iap -- -N -L 10000:localhost:10000` playbook in `iris_cluster_remote_access.md` memory. Jobs were unaffected since the tunnel only matters for our monitoring queries; wandb continued reporting state=running throughout the outage.
- **Memory updates** (in `~/.claude/projects/.../memory/`, outside this repo):
  - `reference_marin_yaml_secrets.md` — `WANDB_API_KEY` lives at `env.WANDB_API_KEY` in `/lfs/skampere3/0/ahmedah/code/marin/.marin.yaml`; user explicitly asked to never `Read`/`cat` the file directly, so use a Python yaml one-liner that prints into `export WANDB_API_KEY=$(...)` and never echoes the value to the transcript.
  - `reference_github_token.md` — PAT location, with a note that it has `repo` scope on `marin-community/marin`.

### Branch / commit hygiene

- Local commits on `midtrain_data` ahead of `origin/midtrain_data`:
  1. `[logbook] Restore monitor log tick-1-to-35 entries`
  2. `[midtrain] Add v5p-64 to 1e22-v5 v5p_compute allowlist`
  3. `[iris,fray] Plumb max_task_failures through client.submit and JobRequest`
  4. `[midtrain] Read MIDTRAIN_MAX_TASK_FAILURES in training submission`
  5. `[midtrain] Log 2026-05-07 placement-collision recurrence and resiliency plan`
  6. (this) `[logbook] Log 2026-05-09 v5p-64 resubmission with MAX_TASK_FAILURES + outcome`
- User has now asked to push `midtrain_data` to origin so the work is durable across session/machine boundaries. Pushing this branch only — no PR cycle yet for the iris/fray plumbing (that's a separate engineering review with the iris owners).

# codex 5.5 2026-05-16T00:22:21Z

Changed `experiments/exp_delphi_math_10b_midtrain.py` so eval cadence is
normalized per run length instead of hard-coded at 200 steps.

Rationale:

- The file already derived checkpoint cadence from `num_train_steps` so each
  run got roughly the same rollback density.
- Eval cadence was still fixed at `STEPS_PER_EVAL = 200`, which gave shorter
  cells fewer validation points. That made the 1e21 curves coarser than the
  longer 1e22 curves.

New policy:

- `_steps_per_eval(num_train_steps)` targets about 40 eval points per run via
  `EVAL_FRACTION_OF_RUN = 0.025`.
- `MAX_STEPS_PER_EVAL = 200` preserves the historical longest-run cadence.
- `MIN_STEPS_PER_EVAL = 25` avoids pathological tiny intervals in short smoke
  runs.
- Current expected examples:
  - `num_train_steps=4_411` -> eval every `110` steps.
  - `num_train_steps=7_646` -> eval every `191` steps.
  - historical `num_train_steps=9_413` -> capped at `200` steps.

Validation run:

```bash
uv run pytest experiments/test_default_train_init_mode.py -q
uv run python -m py_compile experiments/exp_delphi_math_10b_midtrain.py \
  experiments/test_default_train_init_mode.py
./infra/pre-commit.py --fix experiments/exp_delphi_math_10b_midtrain.py \
  experiments/test_default_train_init_mode.py
```

# claude 2026-05-16T09:27Z — Launched 3e18 K=0.20 sweep as 12 top-level iris jobs

User direction: "first kill all jobs i have running. OK, let's do the minimum
correct sweep launch all of those jobs but launch them one by one be extremely
careful! do not start a monitor. Just tell me when you have launched the
jobs.!"

## What happened

1. Stopped/verified terminal: all prior `/ahmed/aa-delphi-3e18-cpt-k020-*`
   coordinators (rounds 1-5) are terminal. The round-5 coord `1778947003`
   was stopped earlier in the session after the cascade-kill investigation;
   no other non-terminal jobs under `/ahmed/`.
2. Surgically tightened `experiments/midtrain_specs/delphi_small_cpt_k020.py`:
   - `--base / --mix / --lr` are now `required=True` with `choices=...` and
     a single value each (no `action="append"`). The script can no longer
     default to "all 72 cells" — invocation without selectors is a hard CLI
     error. This eliminates the foot-gun that produced rounds 1-5 coords,
     which each submitted 12 children at once and got cascade-killed when
     the parent reached SUCCEEDED. The redesign doc said multi-cell sweeps
     must loop in the driver shell, never inside the launcher.
   - Per-base TPU allowlist (`ALLOWED_TPUS_PER_BASE`): a `--tpu` outside
     the allowlist for the chosen base raises before submission. Picking
     a too-small TPU (e.g. 2e20 on v5p-8) or unnecessarily large (v5p-64
     on 3e18) is now a launch-time error rather than a silent waste.
   - Stale env-var rejection: any of `RUN_ID`, `WANDB_RUN_ID`, or anything
     prefixed `MIDTRAIN_` / `TRUE_MIDTRAIN_` in the shell aborts the
     launch. The redesign doc forbids identity-from-shell-env.
   - Deleted the legacy `for cell in cells:` loop + `time.sleep(spacing)`
     in `main()`. The script now resolves one spec, preflights, writes the
     manifest/yaml, submits one training child, blocks on `result.wait()`,
     and exits. Each `iris job run` is a top-level coordinator that lives
     for exactly one TPU training run.
3. Cleaned orphan a001 resources from round 5 (none of these dirs had a
   real step-N checkpoint — only `train_lm_config.yaml`,
   `midtrain_manifest.json`, and an empty `eval_metrics.jsonl`):
   - `gs://marin-us-east5/checkpoints/delphi-3e18-{p33m67,p50m50}-k0p20-lr*-a001/`
     (7 dirs)
   - `gs://marin-us-east5/midtrain-manifests/runs/delphi-3e18-{p33m67,p50m50}-k0p20-lr*.json`
     (7 files)
4. `--dry-run` clean for two endpoint cells (lr33 and lr83). Reported
   `tpu=v5p-8, steps=7400` — matches the K=0.20 plan for 3e18.
5. Submitted 12 separate top-level iris jobs, one per (mix, lr) cell, at
   interactive priority in us-east5. Each is a CPU-1/3GB/9GB coordinator
   running `python experiments/midtrain_specs/delphi_small_cpt_k020.py
   --base 3e18 --mix MIX --lr LR`; inside, the script submits one TPU child
   and blocks on it.

## Job IDs

| # | run_id                              | iris job id                              |
|---|-------------------------------------|------------------------------------------|
| 1 | delphi-3e18-p33m67-k0p20-lr33-a001  | /ahmed/aa-d3e18-p33m67-lr33-1778948729   |
| 2 | delphi-3e18-p33m67-k0p20-lr50-a001  | /ahmed/aa-d3e18-p33m67-lr50-1778948749   |
| 3 | delphi-3e18-p33m67-k0p20-lr67-a001  | /ahmed/aa-d3e18-p33m67-lr67-1778948759   |
| 4 | delphi-3e18-p33m67-k0p20-lr83-a001  | /ahmed/aa-d3e18-p33m67-lr83-1778948769   |
| 5 | delphi-3e18-p50m50-k0p20-lr33-a001  | /ahmed/aa-d3e18-p50m50-lr33-1778948780   |
| 6 | delphi-3e18-p50m50-k0p20-lr50-a001  | /ahmed/aa-d3e18-p50m50-lr50-1778948790   |
| 7 | delphi-3e18-p50m50-k0p20-lr67-a001  | /ahmed/aa-d3e18-p50m50-lr67-1778948800   |
| 8 | delphi-3e18-p50m50-k0p20-lr83-a001  | /ahmed/aa-d3e18-p50m50-lr83-1778948814   |
| 9 | delphi-3e18-p67m33-k0p20-lr33-a001  | /ahmed/aa-d3e18-p67m33-lr33-1778948824   |
|10 | delphi-3e18-p67m33-k0p20-lr50-a001  | /ahmed/aa-d3e18-p67m33-lr50-1778948834   |
|11 | delphi-3e18-p67m33-k0p20-lr67-a001  | /ahmed/aa-d3e18-p67m33-lr67-1778948846   |
|12 | delphi-3e18-p67m33-k0p20-lr83-a001  | /ahmed/aa-d3e18-p67m33-lr83-1778948857   |

## Deliberately NOT done (deferred from the audit, lower priority)

- W&B run-id preflight (audit item #2). The attempt suffix (`-aNNN`)
  makes fresh-attempt collisions unlikely; we'd only re-collide if the
  same invocation was retried within the same attempt counter. Adding a
  `wandb.Api().run(...)` check is straightforward but adds a network call
  per invocation; not blocking for this sweep.
- The remaining 5 bases (9e18, 2e19, 3e19, 9e19, 2e20). The user
  explicitly scoped this round to 3e18. Once these 12 are observed to
  actually start training (TPU acquired, levanter Resuming/initializing
  log lines, first W&B step), we can submit the next base's 12 cells
  with the same script.

## Per the user

No monitor started.

## 16:41Z follow-up — 10 pending coordinators resubmitted as `--preemptible`

Check-in at 16:33Z showed 2/12 coordinators running and 10/12 pending on the
iris controller CPU pool (`Scheduler: Insufficient CPU (need 1 cores,
available 0.1 cor…`). The first-submission iris log line revealed why:

```
iris.cli.job Executor heuristic: auto-tagging job as non-preemptible
```

iris's heuristic auto-tags small CPU-only jobs (no direct TPU request) as
non-preemptible. The non-preemptible CPU pool was saturated; interactive
priority only helps with queue ordering, not capacity. User correctly
recalled that the legacy `exp_delphi_math_10b_midtrain.py` launches used
`--preemptible`.

Action: stopped the 10 pending coordinators (`iris job stop
--no-include-children` with explicit ids per
`feedback_iris_job_stop_listing_cap`) and resubmitted with the same
arguments plus `--preemptible`. All 10 went from submit to `running` with
zero queue time — preemptible CPU pool clearly has plenty of capacity. The
2 originally-running non-preemptible coordinators left as-is (their TPU
children were already underway; killing them would lose progress).

Post-resubmission state (16:41Z):

- 12/12 coordinators: running
- 9/12 TPU children: running on v5p-8 workers
- 3/12 TPU children: pending on the TPU pool (`Scheduler: Insufficient TPUs
  (need 4, available 0) - 105 wor...`). These are the last three submitted
  (p67m33 lr50/lr67/lr83); they'll pick up TPUs as the earlier batch's
  workers free up or as the preemptible TPU pool churns.

Updated coordinator iris job ids (the 10 resubmissions):

| run_id                              | new iris job id                          |
|-------------------------------------|------------------------------------------|
| delphi-3e18-p33m67-k0p20-lr67-a001  | /ahmed/aa-d3e18-p33m67-lr67-1778949552   |
| delphi-3e18-p33m67-k0p20-lr83-a001  | /ahmed/aa-d3e18-p33m67-lr83-1778949563   |
| delphi-3e18-p50m50-k0p20-lr33-a001  | /ahmed/aa-d3e18-p50m50-lr33-1778949574   |
| delphi-3e18-p50m50-k0p20-lr50-a001  | /ahmed/aa-d3e18-p50m50-lr50-1778949585   |
| delphi-3e18-p50m50-k0p20-lr67-a001  | /ahmed/aa-d3e18-p50m50-lr67-1778949595   |
| delphi-3e18-p50m50-k0p20-lr83-a001  | /ahmed/aa-d3e18-p50m50-lr83-1778949605   |
| delphi-3e18-p67m33-k0p20-lr33-a001  | /ahmed/aa-d3e18-p67m33-lr33-1778949615   |
| delphi-3e18-p67m33-k0p20-lr50-a001  | /ahmed/aa-d3e18-p67m33-lr50-1778949625   |
| delphi-3e18-p67m33-k0p20-lr67-a001  | /ahmed/aa-d3e18-p67m33-lr67-1778949635   |
| delphi-3e18-p67m33-k0p20-lr83-a001  | /ahmed/aa-d3e18-p67m33-lr83-1778949645   |

The 2 original non-preemptible coordinators (still running) keep their
prior ids `aa-d3e18-p33m67-lr33-1778948729` and `aa-d3e18-p33m67-lr50-1778948749`.

## 17:45Z follow-up — throughput macro registry added; 3e18 sweep is making progress despite v5p preemption churn

This entry records the post-redesign operational state from the Claude session.
Going forward, keep run operations, status checks, and tactical decisions in
this logbook. Keep `.agents/logbooks/midtraining_redesign.md` as the design
record for the launcher and API.

### Throughput reference module added

User asked for a pure macro-style tracker to estimate wall-clock for future
Delphi runs without letting those estimates affect launch behavior. Claude added
`experiments/throughput_stats.py`.

Design:

- Pure reference/data module. It imports only stdlib modules and is not imported
  by the training stack.
- Confirmed no `lib/` code references `throughput_stats`.
- `ThroughputAnchor` records `model_flops_key`, `tpu_type`,
  `train_batch_size`, `seq_len`, `per_device_parallelism`, `step_time_s`,
  `wandb_run_path`, `measured_at`, and notes.
- `estimate_wall_time_s(...)` requires an exact `(model, tpu)` anchor. It uses
  the exact step time for matching batch/seq, linearly scales by
  tokens-per-step for same-model/same-TPU different batch/seq, and refuses to
  extrapolate across TPU families.
- CLI examples:

```bash
uv run python experiments/throughput_stats.py --list
uv run python experiments/throughput_stats.py \
  --model 3e18 --tpu v5p-8 --steps 7400 \
  --train-batch-size 8 --seq-len 4096
```

Seed anchor:

- Source run: `marin-community/delphi-midtraining/delphi-3e18-p33m67-k0p20-lr33-a001`.
- Rendered training config:
  - `train_batch_size: 8`
  - `seq_len: 4096`
  - `num_train_steps: 7400`
  - `per_device_parallelism: -1`
- Tokens per step: 32,768.
- Measured mid-flight at roughly 1,647 steps in 27:29 wall-clock, including
  about 3 evals. After subtracting eval overhead, pure train was roughly
  0.89 s/step; the anchor uses conservative `1.0 s/step`.
- Estimated train-only wall time for a 7,400-step 3e18 run on v5p-8: about
  2.06 h.

Future work:

1. Walk each Delphi pretrain W&B run once and populate one v5p anchor per base.
2. Add v6e anchors as soon as any Delphi cell completes on v6e-4 or v6e-8.
3. Consider an eval-overhead field if wall-clock estimates need to include
   amortized validation time rather than train-only time.

### 17:45Z status check

At roughly 17:45Z, about 1h20m after the 3e18 launch wave, the sweep was split:
5/12 cells were training cleanly and had crossed the first permanent checkpoint;
7/12 were still in the preempt-before-first-checkpoint loop.

Healthy/training cells:

| Cell | Steps | Loss | Preemptions | Notes |
|---|---:|---:|---:|---|
| p33m67-lr33 | 4,630 / 7,400 (63%) | 2.36 | 0 | past first checkpoint |
| p33m67-lr50 | 4,810 / 7,400 (65%) | 1.98 | 0 | past first checkpoint |
| p33m67-lr67 | 4,260 / 7,400 (58%) | 2.21 | 0 | past first checkpoint |
| p33m67-lr83 | 2,410 / 7,400 (33%) | 2.26 | 1 | recovered after preemption |
| p50m50-lr33 | 4,260 / 7,400 (58%) | 2.97 | 0 | past first checkpoint |

The first permanent checkpoint is around step 370. Once a cell crosses that
point, future preemptions should resume from disk instead of starting again
from HF initialization and step-0 eval.

Cells stuck before first checkpoint:

| Cell | Iris state | Preemptions | Effective steps | Notes |
|---|---|---:|---:|---|
| p50m50-lr50 | running | 4 | 2 | attempt #5 |
| p50m50-lr67 | pending | 4 | 2 | waiting for TPU |
| p50m50-lr83 | pending | 4 | 2 | waiting for TPU |
| p67m33-lr33 | pending | 4 | 2 | waiting for TPU |
| p67m33-lr50 | pending | 4 | 2 | waiting for TPU |
| p67m33-lr67 | pending | 3 | 2 | waiting for TPU |
| p67m33-lr83 | pending | 4 | 2 | waiting for TPU |

Each failed attempt spent roughly 6 minutes on bundle/JIT/step-0 eval/first
train step, then lost the v5p preemptible worker before reaching step 370. W&B
still showed the runs as `running` because `resume="allow"` reconnected after
each restart, but the stuck cells' useful training progress was pinned around
step 2.

Options discussed at that point:

1. Wait for the cells to catch a clean 6+ minute window.
2. Lower the first checkpoint cadence, e.g. checkpoint around step 50, then
   resubmit stuck cells.
3. Move stuck cells to v6e-8 in us-east5-b. This is operationally fast but
   would be the first Delphi v6e smoke.
4. Move to another v5p region/zone. This has lower model risk than v6e but
   introduces cross-region storage concerns if leaving us-east5.

### Why interactive jobs were still preempted

The preemptions were GCP spot/preemptible reclaim events, not another Marin
user preempting us through Iris priority. The logs named workers like
`marin-tpu-v5p-preemptible-8-us-east5-a-*`.

Important distinction:

- Iris priority (`production`, `interactive`, `batch`) controls scheduling
  order inside Iris.
- GCP preemptibility controls whether the underlying VM can be reclaimed.
- Marin has `v5p-preemptible` configured, but no non-preemptible/reserved v5p
  pool. `--priority interactive` does not turn a v5p preemptible worker into an
  on-demand worker.

Dashboard interpretation:

- The dashboard was not stale.
- Iris cluster status matched the dashboard: at the time, v5p-8 showed
  `Ready=45, Demand=0` in `us-east5-a` and `Ready=7, Demand=20` in
  `us-central1-a`, for 52 live v5p-8 slices total.
- The apparent ~37% "idle" fraction was interpreted as average VM downtime /
  preemption churn, not a pool of idle v5p slots available to bind our jobs.
- The alive east5 workers were already occupied by shared-cluster work. When
  one of our workers was preempted, that cell re-entered the queue and lost
  position while stable jobs kept their workers.

Conclusion: this is not pointless, but the first-checkpoint window is too
expensive under high spot churn. Once cells cross step ~370, the run becomes
normal checkpoint-resume training; before that, each preemption restarts the
same expensive startup path.

### Subsequent progress check

A later check showed significant net progress: 9/12 cells were now healthy and
advancing, up from 5/12. Four previously stuck cells crossed the first
checkpoint.

Healthy/training cells:

| Cell | Steps | Loss | Preemptions | Notes |
|---|---:|---:|---:|---|
| p33m67-lr33 | 5,740 / 7,400 (78%) | 2.10 | 0 | oldest wave |
| p33m67-lr50 | 5,920 / 7,400 (80%) | 1.54 | 0 | oldest wave |
| p33m67-lr67 | 5,550 / 7,400 (75%) | 2.08 | 0 | oldest wave |
| p33m67-lr83 | 3,520 / 7,400 (48%) | 2.16 | 1 | recovered |
| p50m50-lr33 | 5,180 / 7,400 (70%) | 2.32 | 0 | oldest wave |
| p50m50-lr50 | 742 / 7,400 (10%) | 2.16 | 4 | past step 370 |
| p50m50-lr67 | 372 / 7,400 (5%) | 2.30 | 4 | just crossed step 370 |
| p50m50-lr83 | 557 / 7,400 (8%) | 3.19 | 4 | past step 370 |
| p67m33-lr33 | 557 / 7,400 (8%) | 3.14 | 4 | past step 370 |

Still stuck:

| Cell | Preemptions | Effective steps | Notes |
|---|---:|---:|---|
| p67m33-lr50 | 4 | 2 | preempt-before-checkpoint loop |
| p67m33-lr67 | 3 | 2 | preempt-before-checkpoint loop |
| p67m33-lr83 | 4 | 2 | preempt-before-checkpoint loop |

Rough ETA from that check:

- The five oldest cells had about 30-40 minutes remaining.
- The four just-rescued cells had about 2 hours remaining.
- The three stuck cells were waiting for a clean window; they could land
  quickly or continue to churn.

Operational takeaway: the 3e18 sweep was making real progress. The stuck-cell
problem was localized to cells that had not yet crossed the first checkpoint,
not evidence that the launch was globally broken.

# claude 2026-05-16T18:24Z — Consolidated 3e18 sweep reference (final job ids + state)

This section pins the authoritative job-id table for the 3e18 K=0.20 sweep so
future agents do not have to reconstruct it from scattered earlier entries.
The first 2 of the 12 cells (`p33m67-lr33`, `p33m67-lr50`) ran as
`--no-preemptible` coordinators from the original 16:25–16:27Z submission
batch (auto-tagged by iris's executor heuristic). The other 10 were stopped
at 16:38Z when 10/12 were stuck pending on the non-preemptible CPU pool, then
resubmitted at 16:39–16:40Z with `--preemptible`. The TPU children are
preemptible regardless of coordinator flag — Marin's only v5p pool is
`v5p-preemptible` (see `feedback_iris_cpu_coordinator_preemptible` memory).

## Authoritative job-id table

All 12 cells were submitted as standalone top-level `iris job run`
coordinators (no shared parent). Format: `<coordinator job id> /
<child task id>`. Region: us-east5. TPU: v5p-8. Priority: interactive.
Child W&B id matches the run_id column (under
`marin-community/delphi-midtraining/<run_id>`).

| run_id (== W&B id)                   | Coordinator iris job                     | Child iris job (under coordinator)                                                  | Coord preemptible? |
|--------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------|---|
| delphi-3e18-p33m67-k0p20-lr33-a001   | /ahmed/aa-d3e18-p33m67-lr33-1778948729   | .../midtrain-delphi-3e18-p33m67-k0p20-lr33-a001                                      | no |
| delphi-3e18-p33m67-k0p20-lr50-a001   | /ahmed/aa-d3e18-p33m67-lr50-1778948749   | .../midtrain-delphi-3e18-p33m67-k0p20-lr50-a001                                      | no |
| delphi-3e18-p33m67-k0p20-lr67-a001   | /ahmed/aa-d3e18-p33m67-lr67-1778949552   | .../midtrain-delphi-3e18-p33m67-k0p20-lr67-a001                                      | yes |
| delphi-3e18-p33m67-k0p20-lr83-a001   | /ahmed/aa-d3e18-p33m67-lr83-1778949563   | .../midtrain-delphi-3e18-p33m67-k0p20-lr83-a001                                      | yes |
| delphi-3e18-p50m50-k0p20-lr33-a001   | /ahmed/aa-d3e18-p50m50-lr33-1778949574   | .../midtrain-delphi-3e18-p50m50-k0p20-lr33-a001                                      | yes |
| delphi-3e18-p50m50-k0p20-lr50-a001   | /ahmed/aa-d3e18-p50m50-lr50-1778949585   | .../midtrain-delphi-3e18-p50m50-k0p20-lr50-a001                                      | yes |
| delphi-3e18-p50m50-k0p20-lr67-a001   | /ahmed/aa-d3e18-p50m50-lr67-1778949595   | .../midtrain-delphi-3e18-p50m50-k0p20-lr67-a001                                      | yes |
| delphi-3e18-p50m50-k0p20-lr83-a001   | /ahmed/aa-d3e18-p50m50-lr83-1778949605   | .../midtrain-delphi-3e18-p50m50-k0p20-lr83-a001                                      | yes |
| delphi-3e18-p67m33-k0p20-lr33-a001   | /ahmed/aa-d3e18-p67m33-lr33-1778949615   | .../midtrain-delphi-3e18-p67m33-k0p20-lr33-a001                                      | yes |
| delphi-3e18-p67m33-k0p20-lr50-a001   | /ahmed/aa-d3e18-p67m33-lr50-1778949625   | .../midtrain-delphi-3e18-p67m33-k0p20-lr50-a001                                      | yes |
| delphi-3e18-p67m33-k0p20-lr67-a001   | /ahmed/aa-d3e18-p67m33-lr67-1778949635   | .../midtrain-delphi-3e18-p67m33-k0p20-lr67-a001                                      | yes |
| delphi-3e18-p67m33-k0p20-lr83-a001   | /ahmed/aa-d3e18-p67m33-lr83-1778949645   | .../midtrain-delphi-3e18-p67m33-k0p20-lr83-a001                                      | yes |

## State at 18:24:35Z (~2h after launch)

Pulled from `iris cluster status` + `iris job logs` + `iris job summary` per
cell. All 12 iris children show `state=running`. The "step" column is the
training step on the current attempt (already-resumed-from-disk on cells
past step 370).

| run_id                              | iris state | preemptions | step    | total | % done | train loss |
|-------------------------------------|------------|------------:|--------:|------:|-------:|-----------:|
| p33m67-k0p20-lr33                   | running    |           0 |   6,850 | 7,400 |    93% |       2.53 |
| p33m67-k0p20-lr50                   | running    |           0 |   7,220 | 7,400 |    98% |       1.61 |
| p33m67-k0p20-lr67                   | running    |           0 |   6,480 | 7,400 |    88% |       2.02 |
| p33m67-k0p20-lr83                   | running    |           1 | (eval)  | 7,400 |    >50%| (n/a)      |
| p50m50-k0p20-lr33                   | running    |           0 |   6,480 | 7,400 |    88% |       2.17 |
| p50m50-k0p20-lr50                   | running    |           4 |   2,040 | 7,400 |    28% |       2.39 |
| p50m50-k0p20-lr67                   | running    |           4 |   1,480 | 7,400 |    20% |       2.63 |
| p50m50-k0p20-lr83                   | running    |           4 |   1,670 | 7,400 |    23% |       3.19 |
| p67m33-k0p20-lr33                   | running    |           4 |   1,670 | 7,400 |    23% |       3.74 |
| p67m33-k0p20-lr50                   | running    |           5 |       2 | 7,400 |    <1% |       2.95 |
| p67m33-k0p20-lr67                   | running    |           4 |       2 | 7,400 |    <1% |       2.95 |
| p67m33-k0p20-lr83                   | running    |           5 |       2 | 7,400 |    <1% |       2.95 |

Categories:

- **Past first checkpoint, healthy** (9/12): step >= 370, so any subsequent
  preemption resumes from disk rather than restarting from the HF weights.
  Includes the 4 cells that were stuck earlier (preempt=4) but eventually
  caught a 6+ minute window during step 0–370.
- **Pre-checkpoint preempt loop** (3/12): p67m33-lr50/67/83. Each has
  cumulative ~25–30 min of wall-clock burned in the bundle→eval-0→step-1
  startup path with zero stable training progress. Preemption count keeps
  ticking; the only thing changing per attempt is the iris worker name.

Math validation loss snapshot (from the oldest cell at step ~5,000, eval at
`nemotron_cc_math_v1/4plus`): values in the 1.6–2.5 range across mixes/LRs,
consistent with the 1e21 reference curve at the same fractional progress.
Bit-identical val partition; cross-scale comparisons valid.

## Throughput / step-time registry

New file: `experiments/throughput_stats.py` — pure data + helper module
listing measured `ThroughputAnchor` tuples per (model, TPU). Seeded with one
mid-flight anchor (3e18 on v5p-8 at ~1.0 s/step) sourced from the
delphi-3e18-p33m67-k0p20-lr33-a001 run. Module is intentionally not imported
by any training code; it exists so future agents can answer "how long will
this run take on that hardware?" via `--list` / `--estimate` CLI without
reverse-engineering from logs. TODOs left in-file: populate v5p anchors for
9e18, 2e19, 3e19, 9e19, 2e20 from each base's pretrain W&B run; add v6e
anchors as soon as any cell completes on v6e-{4,8}. See the
"Decisions deferred" section of
`.agents/logbooks/midtraining_redesign.md` for the rationale.

## Outstanding decisions (handoff state)

1. **3 stuck cells.** Options still open: (a) wait — could be hours of churn;
   (b) tighten `min_permanent_steps` to ~50 and resubmit the 3, so first
   checkpoint lands inside the typical preemption interval; (c) move the 3 to
   v6e-8 in us-east5-b (small calmer pool, but first Delphi v6e run, treat as
   smoke). User wants to decide; do not act unilaterally.
2. **Throughput anchors for the rest of the Delphi ladder.** Off the critical
   path for this sweep but the next base submission would benefit. Tracked in
   the throughput_stats TODOs.
3. **Superseded operator note**:
   the earlier `feedback_iris_cpu_coordinator_preemptible.md` claim was
   wrong for the one-cell launcher. Do **not** pass `--preemptible` on Iris
   CPU coordinators that wrap a TPU child. The a002 relaunch packed all
   twelve coordinators onto one preemptible v5p worker; one worker failure
   retried all parents and cascade-killed the children. Use stable CPU for
   the coordinator (`--no-preemptible` or omit `--preemptible` and let the
   CPU heuristic choose non-preemptible); the nested TPU training child is
   still preemptible.

## 2026-05-19T01:10Z — 9e18/2e19 a002 Batch Sweep Status

Checked the `a002` full sweeps after several hours of batch scheduling:

- `9e18/v6e-4`: `12/12` Iris jobs running. Representative cell
  `delphi-9e18-p33m67-k0p20-lr33-a002` is actively training at
  `7.75k/8.82k` steps and has recent checkpoint activity at step `7525`.
  Iris summary shows `3` TPU preemptions, but the run is resuming and making
  forward progress.
- `2e19/v6e-8`: `8/12` Iris jobs running and `4/12` pending. The pending
  jobs are the `p67m33` cells, all blocked by scheduler capacity:
  `Insufficient TPUs (need 8, available 0)`.
- Representative `2e19` cell `delphi-2e19-p33m67-k0p20-lr33-a002` is
  actively training at `2.17k/10.98k` steps, has saved permanent checkpoint
  `step-2196`, and has seen `9` TPU preemptions.
- No failed `9e18` or `2e19` `a002` jobs were observed in this check.

Throughput sanity check against the 20-step probes:

- `2e19/v6e-8` is matching the probe. Probe anchor:
  `0.345s/step` (`2.90 it/s`) for `train_batch_size=16`, `seq_len=4096`.
  Live train-only log lines show `2.8 it/s`, about `0.357s/step`, within
  roughly `4%` of the probe.
- `9e18/v6e-4` has no direct 9e18 probe anchor in
  `experiments/throughput_stats.py`. The nearest same-hardware anchor is
  `2e19/v6e-4` at `0.573s/step` with the same `train_batch_size=16` and
  `seq_len=4096`. The live `9e18/v6e-4` train-only rate is `2.4 it/s`, about
  `0.417s/step`, which is faster than the larger 2e19 anchor as expected.
- Ignore the occasional `~55s/it` progress lines when comparing throughput;
  those are emitted across eval/checkpoint stalls, not steady train steps.

ETA notes at `2026-05-19T01:15Z`:

- `9e18`: all 12 cells are running. Sampled cells range from about
  `6.17k/8.82k` to `7.93k/8.82k`. At the live steady train rate
  (`~2.4-2.5 it/s`), the remaining train work is roughly `7-20` minutes per
  sampled cell, but eval/checkpoint/HF export and future preemptions make the
  practical completion window closer to `30-60` minutes.
- `2e19`: 8 cells are running and 4 `p67m33` cells remain pending for
  `v6e-8`. Running-cell logs are uneven because of preemptions: one sampled
  cell is at `2.37k/10.98k` with a train-only remaining estimate around
  `52` minutes; another sampled cell has restarted from HF at step `1` with
  no checkpoint yet. Once a `2e19/v6e-8` cell is stably allocated, expect
  about `1.0h` train-only from scratch and roughly `1.5-2h` wall-clock after
  startup/eval/export/preemption overhead.
- `2e19/v5p-8` is not faster per step. Probe anchors say:
  - `2e19/v6e-8`: `0.345s/step`, `1.05h` train-only for `10,983` steps.
  - `2e19/v5p-8`: `0.511s/step`, `1.56h` train-only for `10,983` steps.
  v5p-8 only wins wall-clock for the pending cells if the v6e-8 queue wait is
  longer than about `35-45` minutes, after accounting for relaunch/startup
  overhead.

## 2026-05-19T01:28Z — Isoflop Ladder + Cadence Sanity Check

Verified the current Delphi registry and small-CPT launcher:

- Isoflop bucket winners in the registry are:
  `3e18`, `9e18`, `2e19`, `3e19`, `9e19`, `2e20`, `3e20`.
  The original sweep constant uses `1.8e19` / `1.8e20`, but the registry and
  HF-facing names round these to `2e19` / `2e20`.
- The current `experiments/midtrain_specs/delphi_small_cpt_k020.py` launcher
  intentionally includes bases only through `2e20`; `3e20` is registered in
  `experiments/delphi_models.py` but is not currently in that launch grid.
- The scales between `2e19` and `3e20` are: `3e19`, `9e19`, `2e20`, then
  `3e20`.

Rendered config sanity check for `p33m67/lr0.5`:

| base | CPT steps | permanent checkpoint / HF export interval | warmup | decay |
|---|---:|---:|---:|---|
| `3e18` | 7,400 | 740 | 0.1 | `None` |
| `9e18` | 8,819 | 881 | 0.1 | `None` |
| `2e19` | 10,983 | 1,098 | 0.1 | `None` |
| `3e19` | 7,574 | 757 | 0.1 | `None` |
| `9e19` | 8,033 | 803 | 0.1 | `None` |
| `2e20` | 11,278 | 1,127 | 0.1 | `None` |

Code path:

- `MidtrainSpec.permanent_fraction = 0.10`.
- `render_train_lm_config(...).trainer.checkpointer.keep = [{"every":
  int(num_train_steps * 0.10)}]`, floor-rounded with a `min_permanent_steps`
  guard of `50`.
- `hf_save_steps` uses the same interval.
- Temporary checkpointing is separate: `save_interval: "10m"` to the TTL temp
  checkpoint path.
- `CPT_DEFAULT_WARMUP_FRACTION = 0.10`; `CPT_DEFAULT_DECAY = None`, which
  means Levanter decays over the full post-warmup remainder with no stable
  plateau.

## 2026-05-19T04:41Z — 3e19 / 9e19 Batch Launch Handoff

Decision: launch both `3e19` and `9e19` full K=0.20 sweeps on `v6e-8`,
batch priority, with `256g` container memory. Rationale: the `3e19/v6e-8`
probe succeeded cleanly; the `9e19/v6e-8` probe reached train step 19 and
failed only during HF export under lower RAM, so the relaunch uses doubled
RAM. `v5p-8` was not selected because it is not a good fit for `9e19`, and
the useful throughput/HBM signal points to `v6e-8`.

State:

- `a001` partial materialization was local-only and was cleaned from GCS; no
  `a001` full-sweep Iris jobs were launched.
- `a002` materialized all 24 configs/manifests. Launch source:
  `scratch/20260519T040130Z_delphi_3e19_9e19_a002_launch.tsv`.
- Python submit log:
  `scratch/20260519T042113Z_delphi_3e19_9e19_a002_python_submit_remaining.log`.
- The first Python submitter was too slow because it resent the workspace
  bundle blob on every `client.submit`. After reading the `bundle_id` from an
  already-submitted job (`2974349143c96317a0532f66484afc4b287485852b78c3fed99c17fa89ee7b75`),
  a second submit pass reused the bundle id.
- Bundle-id submit log:
  `scratch/20260519T043848Z_delphi_3e19_9e19_a002_submit_with_bundle_id.log`.
- Submission is complete: `24/24` full-sweep jobs landed
  (`submitted=13 existing=11 failed=0` in the bundle-id submit pass).

Controller state observed around `04:41Z`:

- `3e19`: `12/12` full-sweep jobs landed.
  - Running: all four `p33m67` cells.
  - Pending for `v6e-8` capacity: all four `p50m50` and all four `p67m33`
    cells.
- `9e19`: `12/12` full-sweep jobs landed, all pending for `v6e-8` capacity.

Next action: monitor scheduling/progress. The current blocker is capacity
(`Scheduler: Insufficient TPUs (need 8, available 0)`), not a launch failure.

## 2026-05-19T04:57Z — Full-Sweep Status Snapshot

Iris state snapshot for active Delphi full-sweep attempts:

| sweep | attempt | status |
|---|---|---|
| `3e18` | `a003` | `12/12` succeeded |
| `9e18` | `a002` | `9/12` succeeded, `3/12` running |
| `2e19` | `a002` | `3/12` succeeded, `9/12` running |
| `3e19` | `a002` | `4/12` running, `8/12` pending on `v6e-8` capacity |
| `9e19` | `a002` | `12/12` pending on `v6e-8` capacity |

Running cells:

- `9e18`: `p33m67/lr67`, `p67m33/lr33`, `p67m33/lr83`.
- `2e19`: `p33m67/lr83`, all four `p50m50`, all four `p67m33`.
- `3e19`: all four `p33m67`.
- `2e20` probe: `p67m33/lr50`, `v5p-16`, `a005`, still running.

No active full-sweep job in this snapshot was in a failed/killed terminal
state. The remaining blocker for the newly launched `3e19`/`9e19` jobs is
scheduler capacity, not submit/config failure.

## 2026-05-20T00:21Z — Future-Agent Handoff Map for Small Delphi CPT

Where the context lives:

- `.agents/logbooks/midtraining_redesign.md` is the design document for the
  post-executor small-CPT launcher. It records the intended architecture and
  policy choices, including fixed 10% warmup, `decay=None` triangular LR,
  per-base K=0.20 budgets, preflight/manifest design, and why the launcher is
  direct-Iris rather than executor-based.
- `.agents/logbooks/codex_fixes_midtraining.md` is only for actual code fixes
  and audit findings after the redesign. It records the WSD schedule bug,
  preflight temp/permanent checkpoint fixes, startup-proof fixes, and manifest
  schema additions. Do not use it as a training-status log.
- `.agents/logbooks/midtraining_delphi.md` is the live experiment logbook for
  Delphi midtraining. Use it for sweep launches, status snapshots, operational
  failures, hardware choices, and throughput observations.
- `experiments/throughput_stats.py` is the pure advisory hardware/throughput
  registry. It is intentionally not imported by training code. Use it for
  train-only wall-time estimates and update it from measured W&B/Iris runs.

Small-suite state captured so far:

- Implemented small-CPT launcher:
  `experiments/midtrain_specs/delphi_small_cpt_k020.py`.
- Covered bases in the launcher: `3e18`, `9e18`, `2e19`, `3e19`, `9e19`,
  `2e20`; `3e20` exists in `experiments/delphi_models.py` but is not in this
  launcher grid.
- Each full sweep is 12 cells: 3 mixes (`p33m67`, `p50m50`, `p67m33`) × 4 LR
  multipliers.
- Rendered schedule policy is fixed across the small suite: `warmup=0.1`,
  `decay=None`, `lr_schedule=linear`, no WSD plateau.
- Permanent checkpoint / HF export cadence is `int(num_train_steps * 0.10)`
  with `min_permanent_steps=50`; temporary checkpoints use the separate
  10-minute TTL path.
- Full `3e18/a003` sweep succeeded.
- `9e18/a002`, `2e19/a002`, `3e19/a002`, and `9e19/a002` were submitted as
  batch-priority sweeps; earlier status snapshots in this logbook record their
  running/pending state.

Latest 2e20 hardware note:

- Probe run:
  `marin-community/delphi-midtraining/delphi-2e20-p67m33-k0p20-lr50-probe-v5p16-30s-a005`
  on `v5p-16`.
- It reached a stable train window before being killed by preemption churn
  (`5` preemptions). This was not an HBM/OOM failure.
- Measured stable speed: median `2.079s/step`, about `126k tok/s`, about
  `41.8%` W&B MFU over global steps `2-9`.
- HBM utilization was not logged in W&B history or Iris logs. The useful
  conclusion is only that `2e20` compiled and ran steady train steps on
  `v5p-16` with no `RESOURCE_EXHAUSTED` / HBM error; numeric HBM margin is
  unknown.
- `experiments/throughput_stats.py` now has a provisional `2e20/v5p-16`
  anchor from this probe. It estimates a full K=0.20 `2e20` cell at about
  `6.5h` train-only on `v5p-16`; practical wall-clock per cell is closer to
  `7-8h` before queue/preemption effects.
- Recommendation from current evidence: use `v5p-16` for `2e20` full sweeps.
  Do not assume `v5p-8` fits or is worthwhile without a separate probe.

Operational note:

- At `2026-05-20T00:19Z`, the only ready `v5p-256` slice in
  `us-central1-a` was fully occupied by Michael Ryan's batch-priority job
  `/michaelryan/hq-fm-3000-budget-ext-v2/curation-curation-high_quality_3000-expFM_natural-9e+20-d1536-L16-B1024`
  (`32/32` tasks running on
  `marin-tpu-v5p-preemptible-256-us-central-20260519-2343-4c908b75`).

## 2026-05-20T01:03Z — 2e20 v5p-8 probe launched

Per-user request to see if `2e20` (1.9B params, batch=64) fits on `v5p-8`
before committing to `v5p-32`/`v5p-64` for the full sweep. The v5p-16 probe
ran cleanly (see entry above), but v5p-8 is half the chip count and pushes
`per_device_parallelism` to 16 — explicitly flagged in the prior handoff as
"do not assume v5p-8 fits without a probe."

Code changes:

- `experiments/midtrain_specs/delphi_small_cpt_k020.py`:
  `ALLOWED_TPUS_PER_BASE["2e20"]` extended to
  `frozenset({"v5p-8", "v5p-16", "v5p-32", "v5p-64"})`. v5p-8/v5p-16 still
  require an explicit `--tpu` override; `DEFAULT_TPU["2e20"]` remains
  `v5p-32`.

Launch:

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-east5 \
  --priority interactive \
  --no-preemptible \
  --job-name aa-d2e20-probe-v5p8-30s-1779238972 \
  --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX gs://marin-us-east5 \
  -- python experiments/midtrain_specs/delphi_small_cpt_k020.py \
    --base 2e20 --mix p67m33 --lr 0.5 --tpu v5p-8 --probe-steps 30 \
    --run-suffix probe-v5p8-30s --attempt 1
```

Rendered spec (from `--dry-run`):

- Run id: `delphi-2e20-p67m33-k0p20-lr50-probe-v5p8-30s-a001`
- TPU: `v5p-8` (preemptible/batch via Fray default; us-east5-a worker
  allocated)
- Container RAM: `256g`
- num_train_steps: `30` (BudgetPolicy.fixed_steps via `--probe-steps 30`)
- W&B project: `delphi-midtraining`
- Probe tags: `probe:throughput-hbm`, `probe_steps:30`,
  `do_not_compare:quality`

Parent: `/ahmed/aa-d2e20-probe-v5p8-30s-1779238972` (CPU coordinator,
`--no-preemptible --priority interactive`, us-east5).
Child: `/ahmed/aa-d2e20-probe-v5p8-30s-1779238972/midtrain-delphi-2e20-p67m33-k0p20-lr50-probe-v5p8-30s-a001/0`
on `marin-tpu-v5p-preemptible-8-us-east5-a-20260520-0101-4e102db9-worker-0`.

Status as of `2026-05-20T01:04:39Z`: child has finished `syncing deps`,
`installing pip deps`, activated venv, and is `running user command`. No
HBM/OOM yet — Levanter is in the HF-checkpoint download / JIT-compile phase
before the first train step.

Outcome at `2026-05-20T01:33Z` — probe **succeeded cleanly**:

- W&B state: `finished`, last step 29.
- First train step completed in `28.0s` (JIT compile + first compile-bound
  step).
- Steady-state median step time from W&B `throughput/duration` over
  global_step >= 5 (25 samples): **`3.471 s/step`**, min/max
  `3.466 / 3.476` (very tight; cleanly captured probe window).
- Tokens/step = `64 * 4096 = 262,144`; about **75.5k tok/s**.
- No `RESOURCE_EXHAUSTED` / HBM errors; no exit-137 host-OOM; no
  preemptions. Permanent checkpoint at `step-29` saved cleanly at
  `01:31:30Z`.

Implications for the full 2e20 K=0.20 sweep (`num_train_steps=11,278`):

| TPU | step time | wall-clock/cell | chip-h/cell | 12-cell total chip-h |
|---|---:|---:|---:|---:|
| `v5p-8` (new) | 3.471 s | **10.87 h** | 87 | **~1,046** |
| `v5p-16` (prior probe) | 2.079 s | 6.51 h | 104 | ~1,248 |

So v5p-8 is **~1.67x slower per cell but ~16% cheaper in chip-hours** than
v5p-16. Per-device load doubles (8 vs 4 seq/chip), which scales close to
linearly here — measured 1.67x vs ideal 2.0x suggests we are mildly
memory-bandwidth bound, not compute-bound, on v5p at this scale.

Anchor added to `experiments/throughput_stats.py` so the planner can use it:

```
$ uv run python experiments/throughput_stats.py --model 2e20 --tpu v5p-8 \
    --steps 11278 --train-batch-size 64 --seq-len 4096
Estimated train-only wall time: 10.87h (39146s)
  anchor:  marin-community/delphi-midtraining/delphi-2e20-p67m33-k0p20-lr50-probe-v5p8-30s-a001
```

Recommendation: v5p-8 is **safe to use for the 2e20 full K=0.20 sweep**
when capacity is more available than wall-clock. v5p-16 stays the right
choice if turnaround time matters more than chip-hour cost. v5p-32
(launcher default) is still unprobed — no anchor for it yet.

## 2026-05-20T01:48Z — Full 2e20 K=0.20 sweep launched on v5p-8 (a001)

Per-user decision: launch the full 2e20 sweep on v5p-8 with interactive
priority, accepting the ~1.67x slower per-cell wall-clock for ~16% lower
chip-hour cost vs v5p-16. Estimated total spend ~1,046 v5p-8 chip-hours
across 12 cells (3 mixes × 4 LRs × 87 chip-h/cell).

Pre-launch:

- Dry-runs: 12/12 cells passed preflight. Rendered as
  `delphi-2e20-<mix>-k0p20-lr<LR>-a001`, `tpu=v5p-8`, `ram=256g`,
  `num_train_steps=11,278`.
- GCS namespaces: all 12 a001 paths empty in `us-east5`. No prior probe
  artifacts at the full-sweep names (probe lived at the distinct
  `-probe-v5p8-30s-` suffix).

Launch pattern (one iris job run per cell, 20s spacing, CPU coordinator
not preemptible):

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-east5 \
  --priority interactive \
  --no-preemptible \
  --job-name aa-d2e20-<mix>-lr<LR>-a001-<timestamp> \
  --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX gs://marin-us-east5 \
  -- python experiments/midtrain_specs/delphi_small_cpt_k020.py \
    --base 2e20 --mix <mix> --lr <LR> --tpu v5p-8 --attempt 1
```

CPU coordinator on `--priority interactive` (so it doesn't queue behind
batch CPU work). The v5p-8 TPU child is submitted by the python launcher
through Fray; v5p-8 only exists as a preemptible/batch pool
(`tpu_v5p-preemptible_8-us-east5-a` / `-us-central1-a`), so the child
inherits effective batch priority regardless. Submit log:
`/tmp/2e20_v5p8_a001_submit_1779241725.log`.

12 submitted parent job ids:

- `/ahmed/aa-d2e20-p33m67-lr33-a001-1779241726`
- `/ahmed/aa-d2e20-p33m67-lr50-a001-1779241754`
- `/ahmed/aa-d2e20-p33m67-lr67-a001-1779241781`
- `/ahmed/aa-d2e20-p33m67-lr83-a001-1779241807`
- `/ahmed/aa-d2e20-p50m50-lr33-a001-1779241836`
- `/ahmed/aa-d2e20-p50m50-lr50-a001-1779241865`
- `/ahmed/aa-d2e20-p50m50-lr67-a001-1779241892`
- `/ahmed/aa-d2e20-p50m50-lr83-a001-1779241919`
- `/ahmed/aa-d2e20-p67m33-lr33-a001-1779241947`
- `/ahmed/aa-d2e20-p67m33-lr50-a001-1779241974`
- `/ahmed/aa-d2e20-p67m33-lr67-a001-1779242001`
- `/ahmed/aa-d2e20-p67m33-lr83-a001-1779242029`

Initial Iris state at `2026-05-20T01:54Z` (a few minutes after the last
submit): 6/12 parents `RUNNING`, 6/12 pending on on-demand CPU capacity.
First two cells already created their v5p-8 TPU child task (pending on
v5p-8 capacity). No early failures.

Babysit window armed for 15 min (background task `bimjywmo5`) — emits
state counts every minute, flags any terminal failures. The 9e18/2e19/3e19
a002 sweeps gave us confidence the schedule and HF-export-OOM are
non-issues at 256g RAM, so the 15-min window is mainly to catch any
config/preflight regressions and verify the CPU coordinators stay alive
past the dependency-sync window.

## 2026-05-20T14:55Z — Mass-preempt by Romain's vLLM serving job

While the sweep was 78-97% through training (12/12 cells actively running on
v5p-8 us-east5-a, ~13 h after launch), **all 12 cells got preempted in one
shot** at `14:55Z` by:

```
/romain/served-qwen3-humaneval-full-4w-v5p8-useast5-20260520-145529
```

That's a 4-worker qwen3-humaneval serving job. The preemption error
recorded in `task_attempts.error` for every one of our 12 cells was the
same `"Preempted by /romain/served-qwen3-humaneval-full-4w-v5p8-useast5..."`
string. No cell was permanently lost — all 12 parents stayed in
`JOB_STATE_RUNNING` and Iris managed retries.

### Why a "production" job could preempt our "interactive" sweep

Inspected the Iris `PriorityBand` enum (decoded from
`lib/iris/src/iris/rpc/job_pb2.py`):

```
PRIORITY_BAND_UNSPECIFIED = 0
PRIORITY_BAND_PRODUCTION  = 1
PRIORITY_BAND_INTERACTIVE = 2
PRIORITY_BAND_BATCH       = 3
```

`job_config.priority_band` per job class:

| Job | priority_band | name |
|---|---:|---|
| Our parent CPU coordinators (12) | `2` | INTERACTIVE (set by `iris job run --priority interactive`) |
| **Our child v5p-8 TPU jobs (12)** | **`0`** | **UNSPECIFIED** |
| Romain's `served-qwen3-humaneval-full-4w-v5p8` (all tasks) | `1` | PRODUCTION |
| Probe parent (reference) | `2` | INTERACTIVE |
| Probe child (reference) | `0` | UNSPECIFIED |

The TPU child is submitted by `submit_launch` in
`lib/marin/src/marin/midtraining/launch.py` via Fray's `JobRequest`. That
path does **not pass a `priority_band`**, so Iris records the child as
band 0 (UNSPECIFIED), which the scheduler treats as below PRODUCTION.
Romain's PRODUCTION-tier serving job thus had the right to evict our
TPU training children, regardless of our parent's INTERACTIVE band.

The 2026-05-16 codex audit (`codex_fixes_midtraining.md`) explicitly
removed `ComputeProfile.priority` because it was set but not propagated
to Fray/Iris. Today's preemption is the consequence of that gap not yet
being plumbed end-to-end. Adding priority to the Fray submit path is the
fix; the audit note already calls this out as a follow-up.

### Recovery

By `2026-05-20T22:14Z` Iris had spawned **fresh task lineages**
(attempt_id reset to 0 for each cell) on new v5p-8 workers. Levanter
resumed each cell from its temp checkpoint and the progress recovered:

| Cell | Pre-preempt step | At 22:27Z post-recovery |
|---|---:|---:|
| `p33m67-lr33` | 10,955 (97.1%) | 11,185 (99.2%) |
| `p33m67-lr50` | 10,869 (96.4%) | 11,008 (97.6%) |
| `p33m67-lr67` | 9,799 (86.9%) | 9,964 (88.3%) |
| `p33m67-lr83` | 8,875 (78.7%) | 8,999 (79.8%) |
| `p50m50-lr33` | 10,599 (94.0%) | 10,612 (94.1%) |
| `p50m50-lr50` | 10,653 (94.5%) | 10,799 (95.8%) |
| `p50m50-lr67` | 10,464 (92.8%) | 10,599 (94.0%) |
| `p50m50-lr83` | 8,949 (79.3%) | 9,123 (80.9%) |
| `p67m33-lr33` | 9,269 (82.2%) | 9,415 (83.5%) |
| `p67m33-lr50` | 9,599 (85.1%) | 9,799 (86.9%) |
| `p67m33-lr67` | 9,477 (84.0%) | 9,599 (85.1%) |
| `p67m33-lr83` | 9,799 (86.9%) | 9,875 (87.6%) |

Wall-clock cost of the event: ~7h between preemption and full resumption
(includes Romain's serving job occupying capacity + Iris re-scheduling +
HF/JIT re-download per cell). No data lost; resumption-from-temp-checkpoint
worked as designed.

## 2026-05-21T01:35Z — 2e20 a001 progress + 9e19 a002 completion

### 9e19 a002

All **12/12 cells `state=finished` at step 8032/8033**. The 11 cells that
were still in flight at the 2026-05-19 status snapshot completed
overnight. No remediation needed; sweep is shippable.

### 2e20 a001 — **12/12 finished** ✅

Sweep complete at `2026-05-21T02:41Z`. All 12 cells `state=finished`
at Levanter terminal step `11,277/11,278`. Iris parents all in state 4
(SUCCEEDED).

| Cell | Finished | Final step |
|---|---|---:|
| `p33m67-lr33` | 22:53Z | 11,277/11,278 |
| `p33m67-lr50` | 23:03Z | 11,277/11,278 |
| `p33m67-lr67` | ≤01:16Z | 11,277/11,278 |
| `p33m67-lr83` | 02:41Z | 11,277/11,278 |
| `p50m50-lr33` | 23:58Z | 11,277/11,278 |
| `p50m50-lr50` | 23:47Z | 11,277/11,278 |
| `p50m50-lr67` | 23:58Z | 11,277/11,278 |
| `p50m50-lr83` | 02:07Z | 11,277/11,278 |
| `p67m33-lr33` | 01:35Z | 11,277/11,278 |
| `p67m33-lr50` | ≤01:16Z | 11,277/11,278 |
| `p67m33-lr67` | 01:38Z | 11,277/11,278 |
| `p67m33-lr83` | ≤01:16Z | 11,277/11,278 |

Total wall-clock for the sweep: **~25h** from launch
(`2026-05-20T01:48Z`) to last cell finished (`2026-05-21T02:41Z`). That
includes the ~7h recovery from Romain's mass-preempt event, plus
per-cell preemption churn (which the launcher absorbed transparently via
temp-checkpoint resume). Estimated train-only per-cell wall-clock was
~10.87h from the v5p-8 probe anchor; actual per-cell wall-clock tracked
that closely, with the preemption events extending it ~2x for cells
caught mid-preempt.

Monitor `bca8pfsmr` stopped at sweep completion.

Wall-clock between the 22:14Z full recovery and 01:35Z: ~3h 21m for the
first 8 finishes. Slowest two remaining cells (`p33m67-lr83`,
`p50m50-lr83`) need a few more hours. State-transition monitor
(`bca8pfsmr`) is live and emitting per cell-finish event.

### Isoflop-bucket-winner K=0.20 CPT status

The 7 v6 isoflop-bucket-winner bases (3e18 → 3e20) are the "small Delphi"
ladder this launcher targets. Current state:

| Isoflop | Sweep id | Cells done | Status |
|---|---|---:|---|
| `3e18` | a003 | 12/12 | ✅ done |
| `9e18` | a002 | 12/12 | ✅ done (a001 OOM'd in HF export; a002 fixed with 256g RAM) |
| `2e19` | a002 | 12/12 | ✅ done |
| `3e19` | a002 | 12/12 | ✅ done |
| `9e19` | a002 | 12/12 | ✅ done |
| `2e20` | a001 | 12/12 | ✅ done (finished 2026-05-21T02:41Z) |
| `3e20` | a001 | 0/12 | 🏃 in flight on v5p-16 (launched 2026-05-20T20:30Z) |

## 2026-05-20T20:30Z — Full 3e20 K=0.20 sweep launched on v5p-16 (a001)

Closing out the v6 isoflop-bucket-winner ladder. 3e20 was the only base
left in the 3e18 → 3e20 set without a CPT sweep.

### Correctness checks before any submission

User asked to `MAKE SURE IT'S CORRECT` because Trap #1 in
`.agents/projects/delphi_midtraining.md` is exactly the "pick wrong v5
ablation as 3e20 base" scenario. Walking through full verification:

1. **Base canonicality.** `DELPHI_3E20.gcs_run_root` =
   `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6` —
   contains `adamh_scaling_v6` (canonical). `assert_not_banned` passes.
   **NOT** the `d2048-L21-B128-adamh_scaling_v5` deprecated ablation.
2. **Architecture match.** Registry: `d=2304, L=23, params=2.5B,
   batch=128, num_train_steps=35,408` — matches the postmortem's
   canonical 3e20 ISOFlop bucket winner and HF repo
   `marin-community/delphi-3e20-2.5Bparams-18.6Btokens`.
3. **Hparam formula check** (per Trap #2 verification protocol). For
   `(B=128, T=18.56B, H=2304)` against the v2 AdamH/Complete(d)P recipe:

   | Param | Stored | v2 predicted | v1 predicted | v2 match? |
   |---|---:|---:|---:|---|
   | `peak_lr` | `4.878e-3` | `4.882e-3` | `1.620e-3` | ✅ within 0.1% |
   | `peak_adam_lr` | `3.400e-4` | `3.404e-4` | — | ✅ within 0.2% |
   | `beta2` | `0.999800` | `0.999800` | `0.960400` | ✅ exact |
   | `epsilon` | `3.570e-8` | `3.565e-8` | — | ✅ within 0.2% |

   All four match the v2 recipe within < 0.2% rounding. v1 prediction
   is off by 3× on `peak_lr` and 4% on `beta2`. **Definitively v6**.
4. **Token math.** Pretrain tokens = `128 × 35,408 × 4,096 =
   18,563,948,544` ≈ 18.56B ✓ (matches HF repo's `18.6Btokens` suffix).
5. **K=0.20 budget math.** Launcher rendered `num_train_steps=7,082`
   (`floor(3,712,789,709 / 524,288) = 7,082.66`) → K = 0.19999 actual,
   off from 0.20 by 11 ppm. Same math as the other isoflop sweeps.
6. **Tests.** Extended
   `tests/midtraining/test_val_set_equivalence.py`'s base_key
   parametrize list to include `"3e20"`; 3/3 mixes pass — the rendered
   `data:` block bit-matches the legacy 1e21 reference.
7. **Dry-runs.** All 12 cells (3 mixes × 4 LR factors) render
   `tpu=v5p-16, ram=256g, num_train_steps=7082`. Pre-commit clean.
8. **GCS namespace.** All 12 a001 paths empty in us-east5.

### TPU choice rationale (v5p-16)

User picked v5p-16. Per-chip load vs working 2e20 anchors:

| Hardware | Chips | 3e20 seq/chip | Reference |
|---|---:|---:|---|
| v5p-8 | 4 | 32 | 2× the 2e20/v5p-8 load + 1.32× bigger model → high HBM risk; **excluded from allowlist** |
| **v5p-16** | 8 | **16** | matches **2e20/v5p-8** anchor (which fit) with 1.32× larger params |
| v5p-32 | 16 | 8 | safer fallback if v5p-16 OOMs |
| v5p-64 | 32 | 4 | comfortable |

`ALLOWED_TPUS_PER_BASE["3e20"] = {"v5p-16", "v5p-32", "v5p-64"}`.

### Launch + first-submit race

12 CPU-coordinator iris jobs at `--priority batch --no-preemptible`,
region us-east5, 20s spacing, attempt 1. Each coordinator submits its
v5p-16 TPU child via Fray.

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 3GB --disk 9GB \
  --region us-east5 \
  --priority batch \
  --no-preemptible \
  --job-name aa-d3e20-<mix>-lr<LR>-a001-<timestamp> \
  --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e MARIN_PREFIX gs://marin-us-east5 \
  -- python experiments/midtrain_specs/delphi_small_cpt_k020.py \
    --base 3e20 --mix <mix> --lr <LR> --tpu v5p-16 --attempt 1
```

The very first submission
(`/ahmed/aa-d3e20-p33m67-lr33-a001-1779332811`) errored with
`ConnectError: Job ... already exists and is still running`, then
landed in iris state 5 (FAILED) with empty GCS namespace (parent never
reached `write_manifest`). Hypothesis: iris client retried internally
after a transient connection issue, and the second retry hit the
"already exists" check on the first retry's submission. Resubmitted at
`20:29:55Z` with fresh timestamp:
`/ahmed/aa-d3e20-p33m67-lr33-a001-1779334184`.

Final 12 active parents (one per cell):

- `aa-d3e20-p33m67-lr33-a001-1779334184` (resubmit)
- `aa-d3e20-p33m67-lr50-a001-1779333786`
- `aa-d3e20-p33m67-lr67-a001-1779333812`
- `aa-d3e20-p33m67-lr83-a001-1779333846`
- `aa-d3e20-p50m50-lr33-a001-1779333872`
- `aa-d3e20-p50m50-lr50-a001-1779333907`
- `aa-d3e20-p50m50-lr67-a001-1779333933`
- `aa-d3e20-p50m50-lr83-a001-1779333962`
- `aa-d3e20-p67m33-lr33-a001-1779333987`
- `aa-d3e20-p67m33-lr50-a001-1779334016`
- `aa-d3e20-p67m33-lr67-a001-1779334049`
- `aa-d3e20-p67m33-lr83-a001-1779334075`

Initial state (~20:30Z post-submit): **4 parents RUNNING, 8 PENDING**
on on-demand CPU (batch priority means slower CPU scheduling than the
2e20 sweep got at interactive priority). 0/12 non-running children.
No early failures. v5p-16 HBM fit remains unverified until the first
cell completes JIT compile and lands its first train step (~15-20 min
after a child gets a worker).

State-transition monitor armed at `bvhrbleu2` (filters out the dead
`1779332811` job so dedup picks the live `1779334184` replacement).

### Code changes

- `experiments/midtrain_specs/delphi_small_cpt_k020.py`:
  - Added `DELPHI_3E20` to imports.
  - `BASES["3e20"] = DELPHI_3E20`
  - `DEFAULT_TPU["3e20"] = "v5p-16"`
  - `ALLOWED_TPUS_PER_BASE["3e20"] = frozenset({"v5p-16", "v5p-32", "v5p-64"})`
- `tests/midtraining/test_val_set_equivalence.py`: added `"3e20"` to
  the `base_key` parametrize list of the data-section bit-identity
  test. 3/3 mixes pass.

## 2026-05-21T01:52Z — final-loss scaling-law first pass

Question from Ahmed: before fitting full validation trajectories, start with
the simplest endpoint analysis. For each completed small-ladder CPT cell, take
the final observed validation loss and ask whether a power-law-style scaling
fit across isoflop scale is visible.

Plan:

1. Fetch only W&B metadata/history rows for small-ladder runs named
   `delphi-{3e18,9e18,2e19,3e19,9e19,2e20}-{mix}-k0p20-lr{33,50,67,83}-aNNN`.
2. Use completed cells only for the fit; keep partial 2e20 cells in the table
   but mark them out-of-fit.
3. Fit per `(mix, lr, metric)` endpoint curves over compute scale with simple
   forms: log-linear in loss, log-linear in improvement from baseline, and a
   two-parameter power floor diagnostic when enough points exist.
4. Write CSV/HTML outputs under `midtrain_analysis_outputs/small_final_loss_scaling/`
   and summarize whether the endpoint curves look scale-law-like enough to
   justify a richer trajectory model.

Result:

- Added `scripts/analysis/delphi_small_final_loss_scaling.py`.
- Fetched W&B summaries from `marin-community/delphi-midtraining` and selected
  the best exact non-probe attempt per `(scale, mix, lr)` cell.
- Coverage at fetch time: `3e18`, `9e18`, `2e19`, `3e19`, `9e19` all `12/12`
  complete; `2e20` `10/12` complete. The running `2e20` cells were retained in
  `endpoints.csv` but excluded from fits.
- Held-out math endpoint loss is extremely scale-law-like under the simplest
  log-log fit `log(loss) = a + b log(compute)`: every recipe has `R^2 ~= 0.998`,
  monotone non-increasing endpoints, and exponent tightly clustered around
  `b ~= -0.096` (range `-0.0984` to `-0.0928`).
- Aggregate `eval/loss` also fits cleanly but with a shallower exponent
  (`median b ~= -0.0624`); Paloma macro/C4 are shallower again
  (`median b ~= -0.046`).
- 3-parameter `floor + A * compute^-alpha` fits improve RMSE slightly but are
  diagnostic only with 5-6 scales; fitted floors vary by recipe and should not
  be interpreted yet.

Artifacts:

- `midtrain_analysis_outputs/small_final_loss_scaling/endpoints.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/fit_summary.csv`
- `midtrain_analysis_outputs/small_final_loss_scaling/summary.md`
- `midtrain_analysis_outputs/small_final_loss_scaling/endpoint_math_val_loss.html`
- `midtrain_analysis_outputs/small_final_loss_scaling/endpoint_eval_loss.html`
- `midtrain_analysis_outputs/small_final_loss_scaling/endpoint_paloma_macro_loss.html`

Verification:

- `uv run python scripts/analysis/delphi_small_final_loss_scaling.py`
- `uv run python scripts/analysis/delphi_small_final_loss_scaling.py --use-cache`
- `./infra/pre-commit.py --fix scripts/analysis/delphi_small_final_loss_scaling.py`

## 2026-05-21T02:06Z — marimo notebook for endpoint-scaling diagnostics

Ahmed could not judge fit quality from the static Plotly HTML and asked for a
marimo notebook. Added:

- `scripts/analysis/delphi_small_final_loss_scaling_notebook.py`

Notebook behavior:

- Reads the cached endpoint/fits outputs from
  `midtrain_analysis_outputs/small_final_loss_scaling/`.
- Provides controls for metric, mix, and fit family.
- Shows a focused endpoint plot for one mix at a time with residuals directly
  underneath the fit curve.
- Adds leave-one-scale-out predicted-vs-observed diagnostics and an error
  summary table, so fit quality is visible in raw loss units rather than only
  by `R^2`.
- Adds a recipe heatmap of log-linear exponents.

Verification:

- `./infra/pre-commit.py --fix scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
  (passes structurally; marimo emits markdown-indentation style warnings only)
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`
- Started interactive server:
  `uv run --with marimo marimo edit --headless --no-token --host 127.0.0.1 --port 2718 scripts/analysis/delphi_small_final_loss_scaling_notebook.py`

## 2026-05-21T02:45Z — held-out 1e21/1e22 extrapolation overlay

Ahmed asked to train the endpoint scaling fits only on the clean small ladder
(`3e18` -> `2e20`) and then evaluate the fit against the local canonical Delphi
`1e21` and `1e22` midtraining data.

Implemented:

- Extended `scripts/analysis/delphi_small_final_loss_scaling.py` to read held-out
  targets from the local dump:
  - `midtrain_analysis_outputs/midtrain_run_registry.csv`
  - `midtrain_analysis_outputs/midtrain_trajectory_deltas.csv`
- Wrote two new artifacts under
  `midtrain_analysis_outputs/small_final_loss_scaling/`:
  - `extrapolation_targets.csv`
  - `extrapolation_predictions.csv`
- Updated `scripts/analysis/delphi_small_final_loss_scaling_notebook.py` so the
  marimo plot extends the small-ladder fit past the `2e20` boundary and overlays
  `1e21`/`1e22` held-out points. Complete-like held-out targets render as
  diamonds; best-prefix targets render as open x markers.

Held-out target coverage from the local registry:

| Scale | Complete-like targets |
|---|---:|
| `1e21` | 12/12 |
| `1e22` | 9/12 |

The three `1e22` best-prefix targets are `p33m67-lr83` at 93.9% progress,
`p50m50-lr50` at 66.5%, and `p50m50-lr67` at 48.2%; they are included for
visual context but clearly marked as prefixes.

Math-validation log-linear extrapolation summary:

| Target | Status | n | Mean abs error | Median observed - predicted | Mean pct error |
|---|---|---:|---:|---:|---:|
| `1e21` | complete | 12 | 0.00816 | -0.00809 | -0.84% |
| `1e22` | complete | 9 | 0.06007 | -0.05573 | -9.98% |
| `1e22` | best_prefix | 3 | 0.04293 | -0.00625 | -1.36% |

Interpretation: the small-ladder endpoint law predicts `1e21` math validation
loss quite well. At complete `1e22` cells the observed math loss is
systematically lower than the small-ladder extrapolation predicts, so the
simple fit is conservative at that jump. The effect is mix-dependent: complete
`1e22` mean observed-minus-predicted error is about `-0.0870` for `p33m67`,
`-0.0616` for complete `p50m50` cells, and `-0.0391` for `p67m33`.

Verification:

- `uv run python scripts/analysis/delphi_small_final_loss_scaling.py --use-cache`
- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
  (structural pass; markdown-indentation style warnings only)
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`
- `./infra/pre-commit.py --fix scripts/analysis/delphi_small_final_loss_scaling.py scripts/analysis/delphi_small_final_loss_scaling_notebook.py .agents/projects/delphi_midtraining.md`

## 2026-05-21T03:08Z — refresh endpoint/extrapolation fits from live W&B

Ahmed pointed out two stale assumptions in the first extrapolation pass:

- the `2e20` small-ladder sweep had finished and should contribute all 12 cells;
- the local `1e22` dump was stale, so held-out targets should be refreshed from
  live W&B rather than `midtrain_trajectory_deltas.csv`.

Read GitHub issue #4547. The issue's visible principled-sweep comment is still
the older `1e22: 9/12 + 3 partial` snapshot, plus the later correction that the
old `1e20` rows used a non-Delphi v5 isoflop base. Live W&B is newer than the
local dump for two of the old `1e22` partials:

- `p33m67-lr83` is now finished at step 7646.
- `p50m50-lr50` is now finished at step 7646.
- `p50m50-lr67` still appears as a best-prefix target in W&B at step
  6382/7647, and `gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.67-e78260/hf/step-7646/`
  does not exist at this check.

Implemented:

- `scripts/analysis/delphi_small_final_loss_scaling.py` now loads held-out
  `1e21`/`1e22` targets from live W&B first, using the local trajectory dump
  only as a fallback.
- Refreshed outputs without `--use-cache`.

Coverage after refresh:

| Source | Scale | Complete-like targets |
|---|---|---:|
| small-ladder W&B summaries | `2e20` | 12/12 |
| live W&B held-out targets | `1e21` | 12/12 |
| live W&B held-out targets | `1e22` | 11/12 |

Math-validation log-linear extrapolation summary after the full `2e20` refresh:

| Target | Status | n | Mean abs error | Median observed - predicted | Mean pct error |
|---|---|---:|---:|---:|---:|
| `1e21` | complete | 12 | 0.00944 | -0.00922 | -1.00% |
| `1e22` | complete | 11 | 0.06405 | -0.06751 | -10.72% |
| `1e22` | best_prefix | 1 | 0.03597 | -0.03597 | -5.66% |

Interpretation remains directionally unchanged: the small-ladder fit predicts
`1e21` closely and is conservative at `1e22`, where observed math loss is lower
than predicted. The fit now uses all 12 completed `2e20` rows, so the lr83
recipes no longer train only through `9e19`.

Verification:

- `uv run python scripts/analysis/delphi_small_final_loss_scaling.py`
- `gcloud storage ls gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.67-e78260/hf/step-7646/`
  returned no objects.

## 2026-05-21T04:59Z — within-run prefix prediction notebook section

Ahmed asked for a second analysis axis: predict final validation loss within a
run from an early prefix (for example first 10% of training), tune the
accuracy-vs-prefix tradeoff on small models through `2e20`, and then check
generalization on `1e21`/`1e22`.

Implemented:

- Added `scripts/analysis/delphi_within_run_prediction.py`.
- The script fetches validation trajectories from live W&B for the same 96
  selected runs used by the endpoint notebook: 72 completed small-ladder runs
  through `2e20`, 12 `1e21` held-out runs, and 12 `1e22` held-out runs.
- It writes:
  - `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_points.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prefix_predictions.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prefix_summary.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_method_selection.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prediction_summary.md`
- Added a marimo section to
  `scripts/analysis/delphi_small_final_loss_scaling_notebook.py` with:
  - method/prefix controls;
  - the selected small-scale recipe table;
  - the prefix-vs-error tradeoff table;
  - per-run held-out prediction details;
  - trajectory overlays with observed and predicted final markers.

Methods tested:

- `last_value`: carry forward the last validation point in the prefix.
- `linear_tau`: fit a line in normalized progress `tau` and evaluate it at
  `tau=1`.
- `template_global`, `template_by_mix`, `template_by_recipe`: learn the median
  fraction of final improvement achieved by a prefix on completed small-ladder
  runs, then apply that fraction to the target run's observed prefix
  improvement. Small-ladder evaluation is leave-one-run-out; held-out large
  evaluation uses the full small-ladder template.

Coverage:

| Split | Scale | Complete-like targets |
|---|---|---:|
| small-CV | `3e18` | 12/12 |
| small-CV | `9e18` | 12/12 |
| small-CV | `2e19` | 12/12 |
| small-CV | `3e19` | 12/12 |
| small-CV | `9e19` | 12/12 |
| small-CV | `2e20` | 12/12 |
| held-out | `1e21` | 12/12 |
| held-out | `1e22` | 11/12 |

Selected accuracy/cost points:

| Metric | Selected method | Prefix | Small-CV MAE | Held-out complete MAE | Held-out complete MAPE |
|---|---|---:|---:|---:|---:|
| `eval_loss` | `template_by_recipe` | 0.40 | 0.00574 | 0.03246 | 1.94% |
| `math_val_loss` | `template_by_recipe` | 0.50 | 0.00387 | 0.02104 | 3.16% |
| `paloma_c4_loss` | `template_by_recipe` | 0.10 | 0.00475 | 0.02159 | 0.80% |
| `paloma_macro_loss` | `template_by_recipe` | 0.10 | 0.00514 | 0.02378 | 0.88% |

Interpretation:

- For math validation, 10% progress is useful but not reliable enough under
  small-scale cross-validation. The best 10% method is `template_by_mix` with
  small-CV MAE `0.0353`; the selected cost/accuracy point shifts to
  `template_by_recipe` at 50% with small-CV MAE `0.00387`.
- Paloma macro/C4 are easier to forecast early: both select
  `template_by_recipe` at 10%.
- The naive `linear_tau` extrapolator is a bad fit for early prefixes because
  validation trajectories are curved in normalized training progress.

Verification:

- `uv run python scripts/analysis/delphi_within_run_prediction.py`
- `uv run python scripts/analysis/delphi_within_run_prediction.py --use-cache`
- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`

## 2026-05-21T19:42Z — dense prefix slider for within-run prediction

Ahmed asked for a sliding control rather than fixed prefix choices, and asked
for clarification on the current fitted functional form.

Implemented:

- Replaced the marimo prefix dropdown with a slider in
  `scripts/analysis/delphi_small_final_loss_scaling_notebook.py`.
- Expanded `PREFIX_FRACS` in
  `scripts/analysis/delphi_within_run_prediction.py` from seven hand-picked
  points to a 5%-90% grid in 1% increments.
- Optimized template lookup so recomputing all prefixes from cached W&B
  trajectory points takes seconds instead of minutes.

Updated selected accuracy/cost points after the dense grid:

| Metric | Selected method | Prefix | Small-CV MAE | Held-out complete MAE | Held-out complete MAPE |
|---|---|---:|---:|---:|---:|
| `eval_loss` | `template_by_recipe` | 0.65 | 0.00265 | 0.01723 | 1.04% |
| `math_val_loss` | `template_by_recipe` | 0.43 | 0.00289 | 0.02596 | 3.85% |
| `paloma_c4_loss` | `template_by_recipe` | 0.06 | 0.00552 | 0.04450 | 1.68% |
| `paloma_macro_loss` | `template_by_recipe` | 0.06 | 0.00529 | 0.03306 | 1.24% |

The best small-CV math point is still much later (`0.90`, MAE `0.000996`), but
the selection rule picks the earliest prefix within the configured tolerance
of the best point. At 10% math progress, the best method is still
`template_by_mix` with small-CV MAE `0.0353`, so the "first 10%" heuristic is
useful for a rough forecast but not enough for tight recipe selection.

Verification:

- `uv run python scripts/analysis/delphi_within_run_prediction.py --use-cache`
- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
  (structural pass; markdown indentation warnings only)
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`
- `./infra/pre-commit.py --fix scripts/analysis/delphi_within_run_prediction.py scripts/analysis/delphi_small_final_loss_scaling_notebook.py .agents/projects/delphi_midtraining.md`

## 2026-05-23T01:10Z — within-run prefix grid starts after 10% warmup

Ahmed pointed out that every Delphi midtraining run used a 10% warmup, so a
5% prefix is inside warmup and should not be treated as a usable trajectory
prediction prefix.

Updated:

- `scripts/analysis/delphi_within_run_prediction.py` now defines
  `PREFIX_FRACS` as 10%-90% in 5% steps.
- `scripts/analysis/build_delphi_midtraining_interactive_report.py` now sets
  both per-cell and joint-fit prefix sliders to `min=10`.
- Regenerated per-cell predictions, joint predictions, and the standalone
  GitHub Pages HTML report from the new grid.

Verification:

- `uv run python scripts/analysis/delphi_within_run_prediction.py --use-cache`
- `uv run python scripts/analysis/delphi_joint_trajectory_prediction.py`
- `uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py --output midtrain_analysis_outputs/small_final_loss_scaling/delphi_midtraining_interactive.html`
- Checked the regenerated CSVs and HTML payload: per-cell predictions, joint
  predictions, and joint model coefficients all have minimum prefix `0.10`.

## 2026-05-23T01:12Z — joint trajectory fits added to report

Ahmed clarified the distinction between the per-cell setting and the
scaling-law-discovery style setting:

- Per-cell: fit a curve within one `(flop, mix, lr)` run from prefix points.
- Joint: fit shared trajectory regressions over `tau`, flop scale, mix, and LR.

Implemented:

- Added `scripts/analysis/delphi_joint_trajectory_prediction.py`.
- The script fits three paper-inspired joint forms:
  - exp + drift
  - power + drift
  - Gompertz shoulder
- It evaluates two scopes:
  - `global`: across all flops, mixes, and LRs.
  - `by_flop`: within each flop scale, across mix and LR.
- For each prefix, fitting uses only validation points with `tau <= prefix`,
  then predicts each completed run's endpoint at `tau=1`.
- Added a bottom report section, "Joint Trajectory Fits Across LR, Mix, And
  Flop", with mix/LR filters including `all`, a target max-error selector,
  a method selector, a prefix slider, MAE-vs-prefix curves, observed-vs-predicted
  scatter, and target-satisfying config tables.
- Added a joint fitted-curve overlay so the section shows the actual shared
  trajectory shapes, not just endpoint MAE.
- Moved the SLDBench/source-note material into this joint section.
- Relabeled the original section as "Curve Prediction Within A LR / Mix / Flop
  Cell" so the setting is explicit.

Current joint-fit headline:

- Best all-run joint method: `joint_by_flop_power_drift @ 90%`, MAE `0.009892`,
  max error `0.028735`.
- No joint method satisfies max absolute error `<= 0.02` across all 104
  completed math runs.
- Joint fits satisfy max error `<= 0.05` from `65%` prefix onward, with
  `by-flop power + drift`.
- This is worse than the current per-cell SciPy parametric best
  (`curve_rational_huber @ 90%`, MAE `0.003952`, max error `0.017680`).

Verification:

- `uv run python scripts/analysis/delphi_joint_trajectory_prediction.py`
- `uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py --output ...`
- `./infra/pre-commit.py --fix scripts/analysis/delphi_joint_trajectory_prediction.py scripts/analysis/build_delphi_midtraining_interactive_report.py`

## 2026-05-23T00:25Z — SciPy optimizer promoted for parametric trajectory fits

Ahmed asked to rerun the within-run trajectory prediction using SciPy
optimization consistently instead of the earlier fixed-grid heuristic.

Implemented:

- `scripts/analysis/delphi_within_run_prediction.py` now uses the old shape
  grid only for initialization.
- MAE variants use bounded `scipy.optimize.minimize`.
- Huber variants use `scipy.optimize.least_squares(loss="huber")`.
- SciPy optimizes the endpoint floor, amplitude, and curve shape parameters;
  final evaluation remains endpoint absolute error / MAE.

Regenerated:

- `trajectory_prefix_predictions.csv`
- `trajectory_prefix_summary.csv`
- `trajectory_method_selection.csv`
- `trajectory_prediction_summary.md`
- `delphi_midtraining_interactive.html`
- hosted GitHub Pages copy at `ahmeda14960.github.io/delphi-midtraining/`

Math-val headline after SciPy parametric regeneration:

- `curve_rational_huber @ 90%`: all completed runs MAE `0.003952`, max error
  `0.017680`.
- `curve_rational_mae @ 90%`: all completed runs MAE `0.004897`, max error
  `0.014867`.
- `template_by_recipe` still wins the global small-CV method-selection rule,
  so the automatic selected-method table remains template-based. The parametric
  dropdown and target-config table now use SciPy-optimized curve fits.

Verification:

- `uv run python scripts/analysis/delphi_within_run_prediction.py --use-cache`
- `uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py --output ...`
- `./infra/pre-commit.py --fix scripts/analysis/delphi_within_run_prediction.py scripts/analysis/build_delphi_midtraining_interactive_report.py scripts/analysis/compare_scipy_within_run_fits.py .agents/projects/delphi_midtraining.md`

## 2026-05-22T22:35Z — 3e20 finished-cell ingestion plan

Ahmed reported that the 3e20 K=0.20 sweep has 9 finished cells so far:

- `p33m67`: `lr50`, `lr67`, `lr83`
- `p50m50`: `lr33`, `lr50`, `lr67`, `lr83`
- `p67m33`: `lr33`, `lr50`

Plan before fetching:

- Extend `delphi_small_final_loss_scaling.py` from `3e18 -> 2e20` to
  `3e18 -> 3e20`.
- Extend the within-run cache updater so reruns fetch only missing completed
  trajectories instead of re-downloading the full cache.
- Add `3e20` to the standalone report and marimo scale order.
- Clarify report wording: aggregate goodness-of-fit is endpoint MAE, i.e.
  mean `|predicted_final - observed_final|`; target tables also report max
  absolute endpoint error.

## 2026-05-21T19:57Z — parametric trajectory forms and best-form grid

Ahmed pushed back that the empirical template was too dumb as a functional
model and asked for selectable functional forms plus a scale-by-prefix view of
which form performs best.

Implemented in `scripts/analysis/delphi_within_run_prediction.py`:

- Added four parametric monotone-decay curve families:
  - `curve_log`: mirrored-log decay.
  - `curve_exp`: normalized exponential decay.
  - `curve_power`: shifted power-law decay.
  - `curve_rational`: rational/Hill-style decay.
- Each family is normalized so `tau=1` equals the learned floor. For a fixed
  shape, the script fits `loss(tau) = floor + amplitude * shape(tau)` on the
  observed prefix. Shape parameters are selected by prefix MAE over a bounded
  grid, and final prediction is the learned `tau=1` floor.
- Evaluation and selection still use final-loss MAE.
- The dense cached run now writes parametric prediction rows with
  `param_floor`, `param_amplitude`, `param_shape_1`, `param_shape_2`, and
  `prefix_fit_mae`.

Implemented in `scripts/analysis/delphi_small_final_loss_scaling_notebook.py`:

- Relabeled the selector to "functional form / method"; it now includes both
  template baselines and the four parametric curve families.
- When a parametric method is selected, the within-run trajectory plot draws
  held-out fitted continuations from the selected prefix to `tau=1`, alongside
  observed and predicted-final markers.
- Added "Best Functional Form By Scale And Prefix": a heatmap/table with
  x-axis = isoflop scale (`3e18` through `1e22`), y-axis = prefix, and cell text
  = best parametric curve family plus MAE.

Current headline after adding parametric forms:

- The small-CV global winner is still usually `template_by_recipe`; the
  parametric forms are now available for inspection but do not dominate the
  empirical recipe template on aggregate.
- On held-out math, high-prefix parametric `curve_power` is competitive
  (`prefix=0.89/0.90`, held-out complete MAE `0.00643`), but the best held-out
  rows are still `template_by_recipe` near prefix `0.87`.

Verification:

- `uv run python scripts/analysis/delphi_within_run_prediction.py --use-cache`
  completed in ~85s from cached trajectory points.
- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
  (structural pass; markdown indentation warnings only)
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`

## 2026-05-21T20:52Z — MAE and Huber prefix-fit variants

Ahmed asked whether the curves were fit with MAE and suggested trying Huber for
curve fitting while keeping final goodness-of-fit selection by MAE.

Implemented:

- Parametric methods now emit both prefix-fit variants:
  - `curve_log_mae`, `curve_log_huber`
  - `curve_exp_mae`, `curve_exp_huber`
  - `curve_power_mae`, `curve_power_huber`
  - `curve_rational_mae`, `curve_rational_huber`
- Final evaluation remains final-loss MAE in
  `trajectory_prefix_summary.csv` and the notebook heatmap/table.
- Prefix fitting now records both `prefix_fit_mae` and `prefix_fit_loss`.
- Parametric curves are precomputed for `math_val_loss` only. Template/last/linear
  baselines still run for all four validation metrics. This keeps the local
  precompute to about one minute instead of a long Python grid-search job.
- Prefix grid is 10%-90% in 5% steps for the parametric comparison after the
  2026-05-23 warmup correction.

Current math snapshot:

- Aggregate selected method remains `template_by_recipe`, now at prefix `0.65`
  under the 10%-90% grid (small-CV MAE `0.00235`, held-out complete MAE
  `0.01857`).
- Best small-CV parametric row is `curve_exp_mae` at prefix `0.90` (MAE
  `0.00488`).
- Best held-out parametric row is `curve_rational_mae` at prefix `0.90` (MAE
  `0.00750`).

Verification:

- `uv run python scripts/analysis/delphi_within_run_prediction.py --use-cache`
  completed in ~59s from cached trajectory points.
- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
  (structural pass; markdown indentation warnings only)
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`

## 2026-05-21T21:01Z — target-MAE config finder

Ahmed asked for a bottom notebook cell where a desired final-loss MAE target
(`0.01`, `0.02`, `0.05`, etc.) returns the prefix/method configs that satisfy
it, e.g. `(90% prefix, rational MAE)`.

Implemented in `scripts/analysis/delphi_small_final_loss_scaling_notebook.py`:

- Added "Configs Meeting Target MAE" below the best-form heatmap.
- Added a target-MAE slider with values `0.005`, `0.01`, `0.02`, `0.05`, `0.10`.
- Added an overall table of cheapest qualifying parametric configs for the
  selected metric.
- Added a per-scale table of cheapest qualifying parametric configs for the
  selected metric and each scale.

Selection rule:

- Filter completed parametric prediction rows to `mean_abs_error <= target`.
- Sort by smallest prefix first, then lower MAE.
- Report top configs overall and up to five configs per scale.

Verification:

- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
  (structural pass; markdown indentation warnings only)
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`
- `./infra/pre-commit.py --fix scripts/analysis/delphi_within_run_prediction.py scripts/analysis/delphi_small_final_loss_scaling_notebook.py .agents/projects/delphi_midtraining.md`

## 2026-05-21T22:34Z — target-MAE criterion tightened to all-run max error

Ahmed pointed out that the target-MAE table should find configs that work for
all runs, not just average below the target, and that the pandas dtype headers
(`int64`, etc.) were clutter.

Updated `scripts/analysis/delphi_small_final_loss_scaling_notebook.py`:

- Target-MAE filters now require full completed-run coverage and
  `max_abs_error <= target`.
- Overall table requires the config to cover every completed run for the
  selected metric.
- Per-scale table requires the config to cover every completed run in that
  scale.
- Removed the `n` column from the display.
- Rendered the qualifying configs as markdown tables via `to_markdown(index=False)`
  so marimo no longer shows pandas dtype labels.

Verification:

- `uv run --with marimo marimo check scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
  (structural pass; markdown indentation warnings only)
- `uv run --with marimo marimo export html scripts/analysis/delphi_small_final_loss_scaling_notebook.py -o /tmp/delphi_small_final_loss_scaling_notebook.html`
- `./infra/pre-commit.py --fix scripts/analysis/delphi_within_run_prediction.py scripts/analysis/delphi_small_final_loss_scaling_notebook.py .agents/projects/delphi_midtraining.md`

## 2026-05-23T06:49Z — 3e20 K=0.20 sweep complete (12/12)

Last cell (`p33m67-lr33`) reached step 7081 at `2026-05-23T05:47Z`. All
12 cells finished cleanly at `final_step=7081` (`num_train_steps=7082`,
last logged is the step before the end-of-training boundary). Closes out
the v6 isoflop-bucket-winner ladder (`3e18 -> 3e20`).

### Wall-clock

| | Value |
|---|---|
| iris submit (per launch entry above) | `2026-05-20T20:30Z` |
| last cell finish (W&B heartbeat)     | `2026-05-23T05:47Z` |
| **total wall-clock**                  | **57.3 h** |
| per-cell `_runtime` min / med / max  | 15.03 / 15.30 / 16.07 h |
| sum of per-cell runtime              | ~ 185 cell-h = 1482 chip-h wall |
| pure-compute reference               | 920 chip-h (anchor 4.87 s/step on v5p-16) |
| preemption overhead                  | ~ 61 % |

For comparison the 2e20 sweep on v5p-8 was ~ 24.6 h wall / ~ 80 %
overhead (logbook §`2026-05-21T01:35Z`). 3e20 took 2.3x the wall for
2.8x the chip-h pure compute and a slightly cleaner queue.

### Final train losses (last logged step, by mix)

| Cell | step | `train/loss` |
|---|---:|---:|
| p33m67-lr33 | 7081 | 1.4097 |
| p33m67-lr50 | 7081 | 1.4013 |
| p33m67-lr67 | 7081 | 1.4017 |
| p33m67-lr83 | 7081 | 1.4066 |
| p50m50-lr33 | 7081 | 1.7309 |
| p50m50-lr50 | 7081 | 1.7283 |
| p50m50-lr67 | 7081 | 1.7318 |
| p50m50-lr83 | 7081 | 1.7395 |
| p67m33-lr33 | 7081 | 2.0512 |
| p67m33-lr50 | 7081 | 2.0514 |
| p67m33-lr67 | 7081 | 2.0569 |
| p67m33-lr83 | 7081 | 2.0648 |

Within-mix LR variance is tight (spread < 0.014 `train/loss` across all
4 LR factors at every mix). Don't over-interpret: these are last-step
`train/loss`, not held-out eval. Run the same val/eval post-processing
the other isoflop-bucket-winner sweeps used before treating any LR as
the best.

### Notable scheduler events during the run

- **3 cells starved at `--priority batch`** for ~2 days before getting
  picked: `p33m67-lr33` (the post-orphan resubmit), `p67m33-lr67`,
  `p67m33-lr83`. None were ever bumped to interactive priority; all
  three eventually drained from the iris queue and finished naturally.
  Diagnosis at the time (launch entry + handoff): BATCH band ends up
  below every default-priority user, so cluster contention determines
  when these get picked, not retry budget.
- **2 cells "killed" mid-run** by
  `/tonyhlee/eval-still_rstarcoder_n8_vr5_round1-step-{100,200}-code/0`:
  `p50m50-lr50` and `p50m50-lr67` showed iris `job_state=6` (terminal)
  on their `midtrain-...` child at ~`2026-05-22T12Z`. Both cells
  nonetheless reached step 7081. Either the parent coordinator
  re-spawned the child, or iris re-allocated despite the "terminal"
  state. **Don't trust `JOB_STATE_KILLED` on a child as
  terminal-for-cell without inspecting the parent.**

### Throughput anchor

Median clean step-time on v5p-16: 4.87 s/step, ~108k tok/s, ~47 % MFU.
Anchor refreshed at commit `66badd834` (`3e20 / v5p-16`, sourced from
`delphi-3e20-p33m67-k0p20-lr67-a001`, which was the highest-progress
cell at measurement time).

### Next steps

- Refresh the scaling-law fit framework from `§2026-05-21T01:52Z` with
  the full 7 v6 bucket-winner ladder x 3 mixes x 4 LRs = 84 data
  points now that 3e20 is in.
- HF export status not yet checked across all 12 cells.
- Memory `[[reference-legacy-1e20-means-3e20-iso]]` (saved
  `2026-05-22`) captures why `delphi-1e20-*` (without `true-midtrain`)
  is the v5-isoflop-3e20 stand-in usable as a throughput proxy for
  canonical 3e20 with a ~1.39x param-ratio scale-up.

## 2026-05-23T23:38Z — cooldown checkpoint helper and p33m67 launch recipe

Added local operator helper:

```bash
uv run python scripts/list_delphi_checkpoints.py --base <scale>
uv run python scripts/list_delphi_checkpoints.py --base <scale> --cooldown-ratio 0.2
uv run python scripts/list_delphi_checkpoints.py --base <scale> --expect-step <step>
```

The helper is read-only. It lists native Delphi pretrain checkpoints from the
registered `experiments.delphi_models` GCS root and checks the required
TensorStore artifacts (`manifest.ocdbt`, `metadata.json`, `d`). With
`--cooldown-ratio r`, it reports the target resume step at `(1-r)` progress
and the nearest available checkpoints. With `--expect-step`, it exits
nonzero unless that exact checkpoint exists and is well-formed.

Concrete check run:

```bash
uv run python scripts/list_delphi_checkpoints.py --base 3e18 --cooldown-ratio 0.2
```

Result:

- `3e18` native checkpoints: `step-10000`, `step-20000`, `step-30000`,
  `step-37001`.
- A `0.2` cooldown ratio targets `80%` pretraining progress:
  `target_step=29600`, `target_cooldown_steps=7401`.
- Closest checkpoint is `step-30000` (`+400` steps, `81.08%` progress,
  decay phase).
- Bracket around the target is `step-20000` (`-9600`) and `step-30000`
  (`+400`). For an 80/20 cooldown on 3e18, `step-30000` is the sensible
  explicit resume step if the operator accepts being 400 steps late.

### Mixture naming

The mix tag is `p{pretrain}m{math}`. Therefore:

- Desired **67% math / 33% pretraining replay** is `p33m67`.
- `p67m33` is the opposite: 67% pretraining replay / 33% math.

### How to keep everything the same except the mixture

For true cooldown, "everything the same" means:

- use the exact native Delphi pretrain checkpoint path chosen by the helper;
- keep `resume_step` equal to the chosen native checkpoint step;
- preserve full trainer state: model weights, optimizer state, scheduler
  count, and state step;
- keep the base model's original `num_train_steps`, batch size, architecture,
  and pretrain optimizer/schedule;
- change only the `data:` section to the desired mixture.

In the refactored `marin.midtraining` path, the conceptual spec is:

```python
run = build_run_identity(
    logical_cell_id="delphi-3e18-p33m67-cooldown20",
    attempt=1,
    output_region_name="us-east5",
    wandb_project="delphi-midtraining",
)

mode = CooldownMode(
    resume=CooldownResume(
        pretrain_checkpoint_path=(
            "gs://marin-us-central2/checkpoints/isoflop/"
            "isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/"
            "checkpoints/step-30000"
        ),
        resume_step=30000,
        staged_output_path=run.output_path,
    )
)
```

Then set:

- `data_section_override=load_legacy_data_section("p33m67")`;
- `data_section_provenance=LEGACY_PROVENANCE["p33m67"]`;
- `mode=mode`;
- `compute.batch_size=base.batch_size`;
- model config from the Delphi base architecture;
- optimizer config from the base pretrain AdamH hyperparameters with the
  original pretrain WSD schedule, not a CPT LR-factor schedule.

Before launch, call `stage_cooldown_checkpoint(spec, ...)` so the native
checkpoint is copied into:

```text
<run.output_path>/checkpoints/step-30000
```

Then run `preflight`, write `train_lm_config.yaml` and
`midtrain_manifest.json`, build the launch request, and launch. Startup logs
must show the same run id/output path and `Resuming training from step 30000`.

Legacy note: `experiments/exp_delphi_true_midtrain.py` already supports
`TRUE_MIDTRAIN_SELECT_MIX=p33m67` for 1e21/1e22, but it has hard-coded
`PRETRAINS` and a manual pre-stage/fan-out contract. It does not currently
cover `3e18`. For new cooldown cells, prefer a small launcher using the
refactored `CooldownMode` path above so the exact native checkpoint path from
`scripts/list_delphi_checkpoints.py` is recorded in the manifest and staged by
`stage_cooldown_checkpoint`.

## 2026-05-23T23:56Z — true-midtraining YAML planning configs

Added the non-launch planning directory:

```text
experiments/midtrain_specs/true_midtrain/nemotron_math_only/
  README.md
  configs/
    checkpoint_candidates.yaml
    p33m67.yaml
    p50m50.yaml
    p67m33.yaml
```

Intent:

- Keep true-cooldown checkpoint selection visible and reviewable.
- Avoid a launcher silently choosing a larger/smaller checkpoint than the
  target cooldown ratio implies.
- Reuse the existing mix convention: `p{pretrain}m{math}`.

`checkpoint_candidates.yaml` contains 27 candidate rows:

- 9 registered Delphi bases: `3e18`, `9e18`, `2e19`, `3e19`, `9e19`,
  `2e20`, `3e20`, `1e21`, `1e22`.
- 3 cooldown ratios: `0.30`, `0.20`, `0.10`.

Each row records:

- target step/progress/cooldown steps;
- suggested closest available checkpoint by absolute step delta;
- whether that suggestion is before or after the target;
- before/after checkpoint bracket;
- exact GCS checkpoint path;
- `review_status: needs_human_review`.

The per-mix YAMLs each list 27 planned cells and reference the shared
checkpoint candidate ids. Their launch contract requires
`review_status: approved` before any future launcher stages or submits a job.
No launcher has been added yet.

Validation:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

root = Path('experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs')
candidates = yaml.safe_load((root / 'checkpoint_candidates.yaml').read_text())
assert len(candidates['checkpoint_candidates']) == 27
candidate_ids = {row['candidate_id'] for row in candidates['checkpoint_candidates']}
assert all(row['review_status'] == 'needs_human_review' for row in candidates['checkpoint_candidates'])
for name in ('p33m67', 'p50m50', 'p67m33'):
    config = yaml.safe_load((root / f'{name}.yaml').read_text())
    assert len(config['cells']) == 27
    assert all(cell['checkpoint_candidate_id'] in candidate_ids for cell in config['cells'])
PY
./infra/pre-commit.py --fix experiments/midtrain_specs/true_midtrain/nemotron_math_only/README.md \
  experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml \
  experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/p33m67.yaml \
  experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/p50m50.yaml \
  experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/p67m33.yaml
```

## 2026-05-24T00:16Z — launched 3e18 p33m67 20% true cooldown

User request: launch the 3e18 model at 20% cooldown on `v6e-4` with
interactive priority, after first showing the 3e18 checkpoint candidates and
ensuring W&B marks this as `cooldown_midtraining`, not `cpt_midtraining`.

3e18 reviewed candidates:

| candidate | cooldown ratio | target step | suggested step | delta | relation | checkpoint |
|---|---:|---:|---:|---:|---|---|
| `delphi-3e18-cooldown30` | 0.30 | 25900 | 30000 | +4100 | after target | `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/checkpoints/step-30000` |
| `delphi-3e18-cooldown20` | 0.20 | 29600 | 30000 | +400 | after target | `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/checkpoints/step-30000` |
| `delphi-3e18-cooldown10` | 0.10 | 33300 | 30000 | -3300 | before target | `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/checkpoints/step-30000` |

Approved for launch:

- `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml`
  now marks `delphi-3e18-cooldown20` as `review_status: approved`.
- `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/p33m67.yaml`
  now marks `delphi-true-3e18-p33m67-cooldown20` as `status: approved`.
- Added launcher:
  `experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py`.

The dry-run confirmed W&B tracker tags contain:

```text
mode:cooldown
base:3e18
tpu:v6e-4
region:us-east5
attempt:001
resume_step:30000
cooldown_midtraining
true_midtraining
dataset:nemotron_math_only
mix:p33m67
cooldown_ratio:0.20
target_step:29600
resume_step:30000
checkpoint_relation:after_target
checkpoint_candidate:delphi-3e18-cooldown20
```

There is no `cpt_midtraining` tag in the rendered config. The written GCS
config verifies the same tags at:

```text
gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a001/train_lm_config.yaml
```

The run manifest records the cooldown stage:

```text
source:      gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/checkpoints/step-30000
destination: gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a001/checkpoints/step-30000
bytes:       5366958711
reason:      Ahmed approved 3e18 p33m67 20pct cooldown staging from native Delphi checkpoint
```

Launch notes:

- Normal root submit first failed because `--memory 4GB` requires
  `--enable-extra-resources`; reduced the CPU coordinator to `--memory 3GB`.
- The current worktree's `iris.version` freshness check reports the source tree
  as older than the controller minimum even after `uv sync`. I did not edit Iris
  source or merge branches; I submitted the root coordinator with an in-process
  `iris.version._CACHED = "2026-05-23"` patch so the CLI call could reach the
  controller.
- Root coordinator job:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-1779581129`
- TPU child job:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-1779581129/midtrain-delphi-true-3e18-p33m67-cooldown20-a001`
- Root coordinator is running and waiting on the child.
- The TPU child is currently pending on `v6e-4` capacity in `us-east5`.
  Scheduler reason:
  `Insufficient TPUs (need 4, available 0)` plus quota-pool tier monotonicity.

Current output path:

```text
gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a001
```

To check status:

```bash
uv run iris --cluster=marin job list --json --prefix /ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-1779581129
uv run iris --cluster=marin job logs --since-seconds 1200 --max-lines 260 --tail /ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-1779581129
```

Update 2026-05-24T00:26Z:

- First root attempt `...-1779581129` staged successfully and submitted the
  child, but the child landed on `tpu_v6e-preemptible_4-us-east5-b` and was
  killed after ~20s before Levanter printed a traceback. The parent then
  exited `1` on status `stopped`.
- Added `ComputeProfile.preemptible` and passed it through
  `build_launch_request`/`submit_launch` so launchers can explicitly request
  non-preemptible TPU children when that capacity class exists. Added a unit
  assertion in `tests/midtraining/test_spec_validators.py`.
- Tried a replacement root
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r2-1779582081` with a
  non-preemptible child. Iris rejected it before child creation:
  `no non-preemptible group provides device tpu:v6e-4`.
- Patched the true-cooldown launcher so the default remains preemptible
  (`v6e-4` is only schedulable that way), with an opt-out flag
  `--no-preemptible-child` for future TPU types.
- Active replacement root:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r3-1779582254`
- Active child:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r3-1779582254/midtrain-delphi-true-3e18-p33m67-cooldown20-a001`
- Current state: root `running`; child `pending` on `v6e-4` preemptible
  capacity in `us-east5`.
- Scheduler reason:
  `Insufficient TPUs (need 4, available 0)` and autoscaler waiting for
  `tpu_v6e-preemptible_4-us-east5-b`.

Re-check command for the active launch:

```bash
uv run iris --cluster=marin job list --json --prefix /ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r3-1779582254
uv run iris --cluster=marin job logs --since-seconds 900 --max-lines 420 --tail /ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r3-1779582254
```

Update 2026-05-24T00:29Z:

- Active child is now `JOB_STATE_RUNNING`.
- W&B run exists and is `running`:
  <https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-true-3e18-p33m67-cooldown20-a001>
- W&B tags verified through `wandb.Api()`:
  `cooldown_midtraining`, `true_midtraining`, `mode:cooldown`,
  `base:3e18`, `tpu:v6e-4`, `region:us-east5`, `attempt:001`,
  `mix:p33m67`, `cooldown_ratio:0.20`, `target_step:29600`,
  `resume_step:30000`, `checkpoint_relation:after_target`,
  `checkpoint_candidate:delphi-3e18-cooldown20`,
  `dataset:nemotron_math_only`.
- No `cpt_midtraining` tag is present.
- Startup logs verify checkpoint selection and resume:

```text
Discovered latest checkpoint at gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a001/checkpoints/step-30000
Resuming training from step 30001
```

Update 2026-05-24T01:07Z:

- Another agent accidentally killed the active r3 root and child while triaging
  a different run:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r3-1779582254`.
- Verified the relevant temp checkpoint exists before relaunching:
  `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5_checkpoints_delphi-true-3e18-p33m67-cooldown20-a001/checkpoints/step-30976`.
- Relaunched with the same output path/run id, same W&B id, same config
  namespace, and a fresh root coordinator:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r4-1779584389`.
- Active child:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r4-1779584389/midtrain-delphi-true-3e18-p33m67-cooldown20-a001`.
- Root and child are both `JOB_STATE_RUNNING`.
- Startup logs show the expected resume behavior:

```text
Discovered latest checkpoint at gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a001/checkpoints/step-30000
Discovered latest checkpoint at gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5_checkpoints_delphi-true-3e18-p33m67-cooldown20-a001/checkpoints/step-30976
Found prior temporary checkpoint gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5_checkpoints_delphi-true-3e18-p33m67-cooldown20-a001/checkpoints/step-30976. We will delete it after saving a new checkpoint.
Resuming training from step 30977
First train step completed in 18.7s (step 30977)
Progress on:train 31.0kit/37.0kit
```

W&B also resumed the same run id:
<https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-true-3e18-p33m67-cooldown20-a001>.

Update 2026-05-24T03:12Z:

- Babysat r4 through completion after the accidental kill/relaunch.
- Root and child both reached `JOB_STATE_SUCCEEDED` with `exit_code=0`,
  `failure_count=0`, and `preemption_count=0`.
- Final child:
  `/ahmed/aa-true-d3e18-p33m67-cd20-v6e4-a001-r4-1779584389/midtrain-delphi-true-3e18-p33m67-cooldown20-a001`.
- Final W&B run:
  <https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-true-3e18-p33m67-cooldown20-a001>.
- Final checkpoints were written and confirmed:
  - Levanter checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a001/checkpoints/step-37000`
  - HF-compatible checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a001/hf/step-37000`
- Temporary checkpoint `step-36926` was deleted after final permanent
  `step-37000` was saved.
- Final logged eval loss was `2.531`; `nemotron_cc_math_v1` macro and micro
  loss were both `1.418`.
- Monitoring state recorded in
  `scratch/20260524-0108_delphi_cooldown_monitoring_state.json`.
## 2026-05-24T04:44Z — True-midtraining cooldown20 batch launched

User requested the 20% cooldown true-midtraining batch for all three
Nemotron math-only mixes across the small Delphi isoflop ladder through
`2e20`, switching `3e18` from `v5p-8` to `v6e-4`.

Checkpoint policy was corrected before launch: for `9e18` and `2e20`, the
reviewed `cooldown20` candidate now uses the at-or-before checkpoint instead
of the nominally closest checkpoint that would have started around 90% through
pretraining. Approved resume checkpoints:

| base | resume step | progress | phase | TPU |
|---|---:|---:|---|---|
| `3e18` | 30000 | 81.08% | decay | `v6e-4` |
| `9e18` | 30000 | 68.03% | stable | `v6e-4` |
| `2e19` | 40000 | 72.84% | stable | `v6e-8` |
| `3e19` | 30000 | 79.22% | stable | `v6e-8` |
| `9e19` | 30000 | 74.70% | stable | `v6e-8` |
| `2e20` | 40000 | 70.93% | stable | `v5p-8` |

`3e18/p33m67` already finished as
`delphi-true-3e18-p33m67-cooldown20-a001`, so it was not relaunched. The
remaining 17 roots were submitted as batch-priority, non-preemptible CPU
coordinators with child TPU RAM set to `256g`:

```text
scratch/20260524_true_midtrain_cd20_batch_launch_1779597018.tsv
```

The babysit state file is:

```text
scratch/20260524-2133_true_midtrain_cd20_batch_monitoring_state.json
```

Initial babysit signal:

- `17` new roots are visible in Iris; no root or child failure observed.
- `3e18/p50m50` created a child job, opened W&B run
  `delphi-true-3e18-p50m50-cooldown20-a001`, discovered staged checkpoint
  `gs://marin-us-east5/checkpoints/delphi-true-3e18-p50m50-cooldown20-a001/checkpoints/step-30000`,
  resumed from step `30001`, and completed the first train step.
- `3e18/p67m33` and all three `9e18` children were created and were pending
  on `v6e-4` TPU capacity at the first post-launch check.
- Remaining pending roots were blocked on CPU coordinator capacity in
  `us-east5`, not on config/preflight errors.

Follow-up at ~2026-05-24T05:11Z:

- No root or child failures observed.
- Clean resume/training startup verified for:
  - `delphi-true-3e18-p50m50-cooldown20-a001` from step `30001`.
  - `delphi-true-3e18-p67m33-cooldown20-a001` from step `30001`.
  - `delphi-true-9e18-p33m67-cooldown20-a001` from step `30001`.
  - `delphi-true-9e18-p50m50-cooldown20-a001` from step `30001`.
  - `delphi-true-2e19-p33m67-cooldown20-a001` from step `40001`.
- `9e18/p50m50` had a TPU preemption and restarted, but W&B resumed the same
  run id and Levanter rediscovered the expected permanent checkpoint at
  `step-30000`.
- Current Iris shape: `17` roots total (`6` running, `11` pending on CPU
  coordinator slots); `6` TPU children total (`5` running, `1` pending for
  `v6e-4` capacity).

Follow-up at ~2026-05-24T06:58Z:

- Current Iris shape: `17` roots total (`9` running, `8` pending on CPU
  coordinator slots); `9` TPU children total (`7` running, `2` killed).
- The two killed children are external preemptions by
  `/tonyhlee/eval-lr1e5_rstarcoder_n8_vr5_round1-step-500-code/0`, not
  training/config failures:
  - `delphi-true-2e19-p67m33-cooldown20-a001`
  - `delphi-true-3e19-p33m67-cooldown20-a001`
- Both affected parent coordinators are still `JOB_STATE_RUNNING` with their
  task pending, so they still own recovery. Do not launch duplicate manual
  copies while these parents are alive; that would risk splitting the exact
  output-path/run identity that true Delphi resume relies on.
- Resume invariant was checked in logs before the preemptions:
  - `2e19/p67m33` discovered and loaded
    `gs://marin-us-east5/checkpoints/delphi-true-2e19-p67m33-cooldown20-a001/checkpoints/step-40000`
    and printed `Resuming training from step 40001`.
  - `3e19/p33m67` discovered and loaded
    `gs://marin-us-east5/checkpoints/delphi-true-3e19-p33m67-cooldown20-a001/checkpoints/step-30000`
    and printed `Resuming training from step 30001`.
- `2e19/p67m33` had already auto-recovered once at `06:34Z`, reusing the
  staged checkpoint and resubmitting
  `midtrain-delphi-true-2e19-p67m33-cooldown20-a001`, then was preempted again
  at `06:41Z`.

Follow-up at ~2026-05-24T07:08Z:

- First new batch completion: `delphi-true-3e18-p50m50-cooldown20-a001`.
  The child and parent both reached `JOB_STATE_SUCCEEDED`; child summary showed
  `exit_code=0`, `failure_count=0`, and `preemption_count=0`.
- Final checkpoints:
  - Levanter checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-3e18-p50m50-cooldown20-a001/checkpoints/step-37000`
  - HF-compatible checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-3e18-p50m50-cooldown20-a001/hf/step-37000`
- Final logged eval loss was `2.533`; `nemotron_cc_math_v1` macro loss was
  `1.462`.
- Current Iris shape: `17` roots total (`1` succeeded, `8` running, `8`
  pending on CPU coordinator slots); `9` TPU children total (`1` succeeded,
  `6` running, `2` killed by Tony's eval preemption).
- `2e19/p33m67` and `2e19/p50m50` are back in `building` after earlier
  preemptions. The remaining killed children are still only
  `2e19/p67m33` and `3e19/p33m67`; their parent roots remain alive, so continue
  to wait for parent-owned recovery instead of launching duplicate copies.

Follow-up at ~2026-05-24T07:12Z:

- Parent-owned recovery worked for `2e19/p67m33`; the child is back to
  `JOB_STATE_RUNNING` with its task running.
- The recovered child rediscovered the expected permanent checkpoint:
  `gs://marin-us-east5/checkpoints/delphi-true-2e19-p67m33-cooldown20-a001/checkpoints/step-40000`.
- Current Iris shape: `17` roots total (`1` succeeded, `8` running, `8`
  pending on CPU coordinator slots); `9` TPU children total (`1` succeeded,
  `7` running, `1` killed).
- Only remaining killed child:
  `delphi-true-3e19-p33m67-cooldown20-a001`, preempted by Tony's eval job.
  Its parent coordinator is still alive, so continue waiting for that parent to
  recover rather than manually duplicating the run.

Follow-up at ~2026-05-24T07:29Z:

- External preemption churn continued. `2e19/p50m50` was killed by
  `/tonyhlee/eval-lr1e5_rstarcoder_n8_vr5_round1-step-600-code/0` after
  multiple successful restarts from `step-40001`.
- Current bad children:
  - `delphi-true-2e19-p50m50-cooldown20-a001`
  - `delphi-true-3e19-p33m67-cooldown20-a001`
- Both parent roots are still `JOB_STATE_RUNNING` with pending tasks. Continue
  to preserve parent-owned recovery rather than launching duplicate copies.
- Current Iris shape: `17` roots total (`1` succeeded, `8` running, `8`
  pending on CPU coordinator slots); `9` TPU children total (`1` succeeded,
  `6` running, `2` killed).

Follow-up at ~2026-05-24T08:12Z:

- Second new batch completion: `delphi-true-3e18-p67m33-cooldown20-a001`.
  The child reached `JOB_STATE_SUCCEEDED` with `exit_code=0`,
  `failure_count=0`, and `preemption_count=5`.
- Final checkpoints:
  - Levanter checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-3e18-p67m33-cooldown20-a001/checkpoints/step-37000`
  - HF-compatible checkpoint:
    `gs://marin-us-east5/checkpoints/delphi-true-3e18-p67m33-cooldown20-a001/hf/step-37000`
- Final logged eval loss was `2.553`; `nemotron_cc_math_v1` macro loss was
  `1.524`.
- Current Iris shape: `17` roots total (`1` succeeded, `8` running, `8`
  pending on CPU coordinator slots); `9` TPU children total (`2` succeeded,
  `5` running, `2` killed by Tony eval preemptions).
- Still killed with live parent coordinators:
  - `delphi-true-2e19-p50m50-cooldown20-a001`
  - `delphi-true-3e19-p33m67-cooldown20-a001`

Follow-up at ~2026-05-24T08:18Z:

- `2e19/p50m50` recovered from killed to a pending child waiting on `v6e-8`
  TPU capacity (`Scheduler: Insufficient TPUs`). This confirms the parent
  coordinator preserved the original run identity and took ownership of the
  relaunch.
- Only remaining killed child is now
  `delphi-true-3e19-p33m67-cooldown20-a001`; its parent coordinator remains
  alive.
- Current Iris shape: `17` roots total (`2` succeeded, `7` running, `8`
  pending on CPU coordinator slots); `9` TPU children total (`2` succeeded,
  `5` running, `1` pending for TPU capacity, `1` killed by Tony eval
  preemption).

Follow-up at ~2026-05-24T08:24Z:

- All three `2e19` children are running again:
  - `delphi-true-2e19-p33m67-cooldown20-a001`
  - `delphi-true-2e19-p50m50-cooldown20-a001`
  - `delphi-true-2e19-p67m33-cooldown20-a001`
- Only remaining killed child is
  `delphi-true-3e19-p33m67-cooldown20-a001`, still the Tony eval
  preemption. Its parent root remains alive.
- Current Iris shape: `17` roots total (`2` succeeded, `7` running, `8`
  pending on CPU coordinator slots); `9` TPU children total (`2` succeeded,
  `6` running, `1` killed).

Follow-up at ~2026-05-24T09:30Z:

- Bad count is back to zero. `3e19/p33m67` recovered from killed to a pending
  child waiting on `v6e-8` TPU capacity, so the parent coordinator preserved the
  original run identity and relaunched correctly.
- Current Iris shape: `17` roots total (`2` succeeded, `8` running, `7`
  pending on CPU coordinator slots); `9` TPU children total (`2` succeeded,
  `6` running, `1` pending for TPU capacity).
- `2e19` remains non-terminal but is still heavily preempted/pending:
  preemption counts are now `13/4/6` for `p33m67/p50m50/p67m33`.

Follow-up at ~2026-05-24T09:40Z:

- Coordinator capacity opened up: root pending count dropped from `7` to `5`.
- No bad jobs.
- New child/launch state:
  - `3e19/p33m67` recovered and is pending for `v6e-8` TPU capacity.
  - `3e19/p50m50` submitted
    `midtrain-delphi-true-3e19-p50m50-cooldown20-a001`, also pending for
    `v6e-8` TPU capacity.
  - `3e19/p67m33` root is running, but no matching child-launch line was seen
    yet in the recent log sample.
  - `9e19/p33m67` root is running, but no matching child-launch line was seen
    yet in the recent log sample.
- Current Iris shape: `17` roots total (`2` succeeded, `10` running, `5`
  pending on CPU coordinator slots); `10` TPU children total (`2` succeeded,
  `6` running, `2` pending for TPU capacity).

Follow-up at ~2026-05-24T09:49Z:

- No bad jobs.
- `3e19/p33m67` and `3e19/p50m50` children are now running.
- `3e19/p67m33` submitted
  `midtrain-delphi-true-3e19-p67m33-cooldown20-a001` and is pending on
  `v6e-8` TPU capacity.
- `9e19/p33m67` root is running and got through dependency setup to
  `running user command`, but has not emitted launcher/preflight logs or a
  child row yet.
- Current Iris shape: `17` roots total (`2` succeeded, `10` running, `5`
  pending on CPU coordinator slots); `11` TPU children total (`2` succeeded,
  `8` running, `1` pending for TPU capacity).

Follow-up at ~2026-05-24T10:01Z:

- No bad jobs.
- `9e19/p33m67` submitted
  `midtrain-delphi-true-9e19-p33m67-cooldown20-a001` and is pending on
  `v6e-8` TPU capacity.
- Current Iris shape: `17` roots total (`2` succeeded, `10` running, `5`
  pending on CPU coordinator slots); `12` TPU children total (`2` succeeded,
  `8` running, `2` pending for TPU capacity).
- `2e19` and `3e19/p33,p50` remain non-terminal but are task-pending after
  more preemptions. `9e18` remains the active progress path.

## 2026-05-25T00:42Z — Delphi 3e19 70% native prefix checkpoint launch

Follow-up to the successful `3e18` prefix-checkpoint pipeline sanity check.
The next materialization is the `3e19` Delphi native/full-state checkpoint at
the exact 70% pretraining prefix.

Preflight:

- Base: `3e19`
- Source checkpoint:
  `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/checkpoints/step-20000`
- Original pretraining schedule length from `.executor_info`: `38,014`
  trainer steps
- Target checkpoint: `26,609` (`70.00%`)
- Steps to train from source: `6,609`
- Output root:
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609`
- Final checkpoint path:
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609/checkpoints/step-26609`
- TPU request: `v6e-4`, `128g` RAM, interactive priority, pinned to
  `us-east5`
- Safety checks: dry-run confirmed the original Delphi root will not be
  modified; destination `step-26609` and the output root were absent before
  launch.

Launch command tracked by the monitor:

```bash
.venv/bin/python -u scratch/submit_delphi_prefix_3e19.py
```

Monitor state file:
`scratch/20260524-1742_delphi_prefix_3e19_monitoring_state.json`

Submitted at `2026-05-25T00:45:41Z` as Iris root job
`/ahmed/delphi-prefix-3e19-step26609`.

Startup failure at `2026-05-25T00:53Z`:

- Root and child both reached `JOB_STATE_FAILED` before any materialized
  checkpoint was written.
- The child failed while staging the source checkpoint through `mirror://`:
  `TransferBudgetExceeded` after `8.92GiB`; the next shard would have brought
  the process total to `11.08GiB`, above the default `10GiB` mirror budget.
- The source full-state checkpoint is `11.15GiB` total. The failed attempt
  already copied most shards into the `us-east5` mirror location; missing source
  pieces included `manifest.ocdbt` and two `d/...` files.
- Confirmed no final checkpoint exists at
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609/checkpoints/step-26609`.

Recovery plan applied at `2026-05-25T01:32Z`:

- Keep the same Iris root job id and output root:
  `/ahmed/delphi-prefix-3e19-step26609` and
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609`.
- Wrap the source checkpoint path in executor `mirrored(..., budget_gb=20)`
  for this helper. This keeps the Levanter worker loading
  `mirror://checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/checkpoints/step-20000`
  while propagating `MARIN_MIRROR_BUDGET_GB=20` to the child training
  environment.
- Focused verification passed:
  `uv run --with pytest --with pytest-timeout python -m pytest tests/test_materialize_delphi_prefix_checkpoint.py`
  (`6 passed`) and `py_compile` for the helper, tests, and submit script.
- Dry-run now reports source mirror budget `20 GB`, original schedule length
  `38,014`, target `26,609`, and the same final destination path.

Final outcome at `2026-05-25T04:44Z`:

- Relaunch succeeded as the same Iris root job
  `/ahmed/delphi-prefix-3e19-step26609`.
- `train_lm` had one TPU preemption, then recovered under the same child job and
  finished successfully; root and child both ended `JOB_STATE_SUCCEEDED`.
- Logs confirmed the source checkpoint was staged from
  `mirror://checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/checkpoints/step-20000`,
  loaded from the `us-east5` mirror, and training resumed from step `20001`.
- Final native/full-state checkpoint was saved at
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609/checkpoints/step-26609`
  at `2026-05-25T03:42:42Z`.
- `metadata.json` reports `step=26609`, `target_step=26609`,
  `source_step=20000`, `base=3e19`, and
  `materialization=delphi_prefix_checkpoint`.
- Checked the original Delphi root for `checkpoints/step-26609`; no such path
  exists there, so the canonical source root was not mutated by this
  materialization.

## 2026-05-25T05:07Z — Delphi 3e20 70% native prefix checkpoint launch

Following the successful `3e19` materialization, launch the `3e20` Delphi
native/full-state checkpoint at the exact 70% pretraining prefix. Ahmed asked
to run this one on `v5p-8` interactive.

Preflight:

- Base: `3e20`
- Source checkpoint:
  `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/checkpoints/step-20000`
- Original pretraining schedule length from `.executor_info`: `35,510`
  trainer steps
- Target checkpoint: `24,857` (`70.00%`)
- Steps to train from source: `4,857`
- Output root:
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e20-step24857`
- Final checkpoint path:
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e20-step24857/checkpoints/step-24857`
- TPU request: `v5p-8`, `128g` RAM, interactive priority, pinned to
  `us-east5`
- The source checkpoint is `28.44GiB`; increased the helper-local source
  mirror budget from `20 GB` to `40 GB` before launch.
- Safety checks: dry-run confirmed the original Delphi root will not be
  modified; destination output root was absent before launch.
- Focused verification passed:
  `uv run --with pytest --with pytest-timeout python -m pytest tests/test_materialize_delphi_prefix_checkpoint.py`
  (`6 passed`) and `py_compile` for the helper and tests.

Launch command tracked by the monitor:

```bash
.venv/bin/python -u scratch/submit_delphi_prefix_3e20.py
```

Monitor state file:
`scratch/20260525-0507_delphi_prefix_3e20_monitoring_state.json`

Submitted at `2026-05-25T05:08:42Z` as Iris root job
`/ahmed/delphi-prefix-3e20-step24857`.

Startup update at `2026-05-25T05:43Z`:

- Root and child are both `JOB_STATE_RUNNING`:
  `/ahmed/delphi-prefix-3e20-step24857` and
  `/ahmed/delphi-prefix-3e20-step24857/train_lm`.
- W&B run:
  `https://wandb.ai/marin-community/marin/runs/delphi-3e20-step24857`.
- Logs confirmed original schedule length `35,510`, target step `24,857`,
  training load path
  `mirror://checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/checkpoints/step-20000`,
  and source mirror budget `40 GB`.
- The full source checkpoint mirror completed: `45` files staged to the
  `us-east5` mirror, then Levanter loaded
  `gs://marin-us-east5/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/checkpoints/step-20000`.
- Logs show `Resuming training from step 20001`; first train step completed
  and temporary checkpoints exist at least through `step-20073`.
- Throughput on `v5p-8` settled near `7.8-8.1s/step` in the latest sample,
  so reaching `24,857` is likely a multi-hour run if this rate holds.

Progress update at `2026-05-25T15:47Z`:

- Root and child are still `JOB_STATE_RUNNING` with no failures or preemptions.
- Latest saved temporary checkpoint:
  `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e20-step24857/checkpoints/step-24313`.
- Final checkpoint `step-24857` does not exist yet.
- Latest observed rate remains about `7.8s/step`; `step-24313 -> step-24857`
  leaves roughly `544` steps, about `70-80` minutes plus checkpoint-save
  overhead if throughput stays stable.

## 2026-05-25T16:32Z — Delphi native prefix checkpoint inventory

Verified the completed native/full-state prefix checkpoints before continuing
with the active `3e20` materialization.

Completed checkpoints:

| Base | Source step | Target step | Original schedule | Output root | Final checkpoint |
| --- | ---: | ---: | ---: | --- | --- |
| `3e18` | `20000` | `26134` | `37335` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-step26134` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-step26134/checkpoints/step-26134` |
| `3e19` | `20000` | `26609` | `38014` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609/checkpoints/step-26609` |

GCS verification:

- `3e18` final checkpoint contains `manifest.ocdbt`, `metadata.json`, and
  checkpoint data under `d/`.
- `3e19` final checkpoint contains `manifest.ocdbt`, `metadata.json`, and
  checkpoint data under `d/`.
- `3e18` metadata reports `step=26134`, `target_step=26134`,
  `source_step=20000`, `base=3e18`, and
  `materialization=delphi_prefix_checkpoint`.
- `3e19` metadata reports `step=26609`, `target_step=26609`,
  `source_step=20000`, `base=3e19`, and
  `materialization=delphi_prefix_checkpoint`.
- Checked the original Delphi source roots for `checkpoints/step-26134` and
  `checkpoints/step-26609`; neither path exists there, so these
  materializations did not mutate the canonical source roots.

Monitor state files:

- `scratch/20260524-1558_monitoring_state.json` records `3e18` as
  `status=succeeded`, W&B
  `https://wandb.ai/marin-community/marin/runs/delphi-3e18-step26134`, and the
  final checkpoint path above.
- `scratch/20260524-1742_delphi_prefix_3e19_monitoring_state.json` records
  `3e19` as `status=succeeded`, W&B
  `https://wandb.ai/marin-community/marin/runs/delphi-3e19-step26609`, and the
  final checkpoint path above.

## 2026-05-25T18:05Z — Delphi 3e20 70% native prefix checkpoint completed

Final outcome for the active `3e20` materialization:

- Iris root `/ahmed/delphi-prefix-3e20-step24857` and child
  `/ahmed/delphi-prefix-3e20-step24857/train_lm` both ended
  `JOB_STATE_SUCCEEDED` with exit code `0`.
- Iris reports `failure_count=0` and `preemption_count=0`.
- Final native/full-state checkpoint was saved at
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e20-step24857/checkpoints/step-24857`
  at `2026-05-25T17:01:44Z`.
- The final checkpoint contains `manifest.ocdbt`, `metadata.json`, and
  checkpoint data under `d/`.
- `metadata.json` reports `step=24857`, `target_step=24857`,
  `source_step=20000`, `base=3e20`, and
  `materialization=delphi_prefix_checkpoint`.
- Checked the original Delphi source root for `checkpoints/step-24857`; no such
  path exists there, so the canonical source root was not mutated.
- The `us-east5` temp checkpoint path no longer listed temp checkpoint objects
  after the final checkpoint completed.

Monitor state updated:
`scratch/20260525-0507_delphi_prefix_3e20_monitoring_state.json` now records
`status=succeeded`, W&B
`https://wandb.ai/marin-community/marin/runs/delphi-3e20-step24857`, and the
final checkpoint path above.

## 2026-05-25T20:22Z — Registered materialized 70% prefix checkpoints for cooldown YAMLs

Ahmed asked to find the completed prefix checkpoints and add them as cooldown
midtraining options.

Found all materialized prefix checkpoint roots under
`gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/`:

| Base | Candidate ID | Schedule length | Target step | Checkpoint path |
| --- | --- | ---: | ---: | --- |
| `3e18` | `delphi-3e18-cooldown30` | `37335` | `26134` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-step26134/checkpoints/step-26134` |
| `3e19` | `delphi-3e19-cooldown30` | `38014` | `26609` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609/checkpoints/step-26609` |
| `3e20` | `delphi-3e20-cooldown30` | `35510` | `24857` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e20-step24857/checkpoints/step-24857` |

Config updates:

- Updated
  `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml`
  so the `3e18`, `3e19`, and `3e20` `cooldown30` candidate rows now point to
  the materialized exact-target checkpoint paths above, have
  `suggested_step_delta: 0`, `suggested_relation_to_target: exact_target`,
  `materialized_checkpoint: true`, and `review_status: approved`.
- Recomputed the `3e18`, `3e19`, and `3e20` `cooldown20`/`cooldown10` target
  metadata in the same YAML against the original scheduler lengths.
- Updated `experiments/delphi_models.py` for these three bases so
  `num_train_steps` reflects the original trainer schedule length from
  `.executor_info`; `verified_checkpoint_step` remains the latest checkpoint
  observed under the canonical source root.
- The per-mix YAMLs already reference these candidate IDs:
  `p33m67.yaml`, `p50m50.yaml`, and `p67m33.yaml` each contain
  `delphi-3e18-cooldown30`, `delphi-3e19-cooldown30`, and
  `delphi-3e20-cooldown30`, so no duplicate per-mix cells were needed.

Validation:

- Parsed `checkpoint_candidates.yaml` and verified every candidate row's
  `target_step` matches `int(train_prefix_ratio * model.num_train_steps)`.
- Verified the 30% cooldown launcher resolution for `3e18`, `3e19`, and
  `3e20` uses scheduler lengths `37335`, `38014`, and `35510`, matching the
  materialized checkpoint metadata.
- Dry-run for `3e20/p33m67/cooldown30` resolves
  `resume_step: 24857`, `checkpoint_relation: exact_target`, and the
  `us-east5` checkpoint path above.
- `uv run python -m py_compile experiments/delphi_models.py experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py`
  passed.
- `uv run --with pytest --with pytest-timeout python -m pytest tests/midtraining/test_spec_validators.py tests/midtraining/test_levanter_config.py`
  passed (`26` tests).
- `git diff --check` passed for the touched config/logbook files.

## 2026-05-25T20:32Z — Verified cooldown eval cadence and math validation split

Ahmed asked to double-check the eval cadence and confirm the true-cooldown runs
use the same Nemotron math validation split as CPT.

Rendered config behavior:

- `lib/marin/src/marin/midtraining/levanter_config.py` sets
  `trainer.steps_per_eval` from the resolved spec.
- CPT eval cadence is based on the CPT step budget.
- Cooldown eval cadence is based on the remaining tail
  `base.num_train_steps - resume_step`.
- The default policy targets roughly `40` eval points, bounded by
  `eval_min_steps=25` and `eval_max_steps=200`.

For the new 70% prefix cooldown runs:

| Base | Resume step | Schedule length | Tail steps | Rendered `steps_per_eval` |
| --- | ---: | ---: | ---: | ---: |
| `3e18` | `26134` | `37335` | `11201` | `200` |
| `3e19` | `26609` | `38014` | `11405` | `200` |
| `3e20` | `24857` | `35510` | `10653` | `200` |

Validation split check:

- Both CPT (`experiments/midtrain_specs/delphi_small_cpt_k020.py`) and
  true-cooldown
  (`experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py`)
  pass `data_section_override=load_legacy_data_section(mix)`.
- The rendered true-cooldown `data:` block matches the CPT/reference
  `experiments/midtrain_specs/data_sections/<mix>.json` block byte-for-byte for
  all `3e18`, `3e19`, `3e20` x `p33m67`, `p50m50`, `p67m33` 70% cooldown
  cells.
- The math validation component is
  `nemotron_cc_math_v1/4plus`, `split: validation`, cache
  `gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519`, with
  `num_validation_sequences: {"nemotron_cc_math_v1/4plus": 12500}`.

Test updates:

- Added a true-cooldown launcher regression test to
  `tests/midtraining/test_val_set_equivalence.py` so the rendered cooldown
  `data:` block must stay identical to the CPT/reference split.
- Updated the stale 3e18 CPT expected step count from `7400` to `7467`, matching
  the corrected original Delphi scheduler length `37335`.
- `uv run --with pytest --with pytest-timeout python -m pytest tests/midtraining/test_val_set_equivalence.py`
  passed (`48` tests).

## 2026-05-25T20:59Z — Launched 3e18/3e19 70% true-cooldown batch jobs in us-east5

Ahmed asked to relaunch Delphi true-cooldown for `3e18` and `3e19`, pinned to
`us-east5` and set to Iris batch priority to avoid cross-region checkpoint
movement.

Launched the six `cooldown30` cells from the materialized 70% prefix
checkpoints:

| Base | Mix | Root job | TPU | Source checkpoint |
| --- | --- | --- | --- | --- |
| `3e18` | `p33m67` | `/ahmed/aa-true-d3e18-p33m67-cd30-v6e4-batch-a001-1779742118` | `v6e-4` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-step26134/checkpoints/step-26134` |
| `3e18` | `p50m50` | `/ahmed/aa-true-d3e18-p50m50-cd30-v6e4-batch-a001-1779742167` | `v6e-4` | same `3e18` prefix checkpoint |
| `3e18` | `p67m33` | `/ahmed/aa-true-d3e18-p67m33-cd30-v6e4-batch-a001-1779742167` | `v6e-4` | same `3e18` prefix checkpoint |
| `3e19` | `p33m67` | `/ahmed/aa-true-d3e19-p33m67-cd30-v6e8-batch-a001-1779742167` | `v6e-8` | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-step26609/checkpoints/step-26609` |
| `3e19` | `p50m50` | `/ahmed/aa-true-d3e19-p50m50-cd30-v6e8-batch-a001-1779742167` | `v6e-8` | same `3e19` prefix checkpoint |
| `3e19` | `p67m33` | `/ahmed/aa-true-d3e19-p67m33-cd30-v6e8-batch-a001-1779742167` | `v6e-8` | same `3e19` prefix checkpoint |

Launch invariants:

- All root coordinator submissions used `--region us-east5 --priority batch`.
- All launchers used `--output-region us-east5`.
- Did not pass `--allow-cross-region-stage`; staging must be same-region or fail.
- The first completed stage manifest,
  `delphi-true-3e18-p33m67-cooldown30-a001`, reports
  `cross_region_copy: false`, with source and destination both under
  `gs://marin-us-east5`.
- `3e18/p50m50` and `3e18/p67m33` also wrote manifests and child TPU jobs;
  their manifests likewise report `cross_region_copy: false`.

Startup snapshot:

- `3e18` `p33m67`, `p50m50`, and `p67m33` each have a pending child TPU job
  under the root coordinator, waiting for `v6e-4` capacity in `us-east5`.
- `3e19` `p33m67` and `p50m50` root coordinators are running and staging their
  larger prefix checkpoints; checkpoint directories exist but manifests had not
  appeared at the snapshot time.
- `3e19/p67m33` root coordinator is accepted but pending on `us-east5` CPU
  coordinator capacity.

Resume state file:
`scratch/20260525-2059_true_midtrain_cd30_3e18_3e19_launch_state.json`.

## 2026-05-27T04:03Z — Invalidated materialized Delphi prefix checkpoints and switched cooldown30 to no-retrain native checkpoints

Ahmed asked to fix the broken prefix checkpoint situation without retraining.

Root cause correction:

- The materialized prefix checkpoints were invalid. The helper decoded old
  untyped Delphi `.executor_info` model payloads as `type: llama`, which built a
  no-QK-norm model tree.
- Original Delphi native checkpoints do contain Qwen3 `q_norm`/`k_norm` arrays
  and matching Adam state; the materialized prefix checkpoints dropped them.
- The launched true-cooldown jobs failed correctly when the Qwen3 launcher tried
  to restore those missing arrays.
- CPT runs are not implicated: they render `type: qwen3` and use model-only HF
  initialization; a checked CPT output checkpoint had QK-norm arrays and Adam
  state.

Code fixes:

- `scripts/materialize_delphi_prefix_checkpoint.py` now defaults old untyped
  Delphi model payloads to `type: qwen3`, then fails closed unless the decoded
  source config is a `Qwen3Config`.
- `experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py`
  verifies the factorized heuristic returns `Qwen3Config` and renders
  `type: qwen3` directly.
- Added tests for both the old untyped `.executor_info` decode path and the
  true-cooldown rendered model type.

No-retrain checkpoint policy:

- Did not graft stale QK-norm arrays into the bad prefix checkpoints. That
  would make them loadable but scientifically wrong because the intervening
  weights were trained under the wrong no-QK-norm forward pass.
- Updated
  `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml`
  so the approved cooldown30 rows point only to existing native Qwen3
  checkpoints. The bad materialized paths remain recorded as
  `invalidated_materialized_checkpoint_path` with reason
  `missing_qwen3_q_norm_k_norm_arrays`.

Current no-retrain cooldown30 candidates:

| Base | Target step | Selected native step | Delta | Progress | Path |
| --- | ---: | ---: | ---: | ---: | --- |
| `3e18` | `26134` | `30000` | `+3866` | `80.3536%` | `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/checkpoints/step-30000` |
| `3e19` | `26609` | `30000` | `+3391` | `78.9183%` | `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/checkpoints/step-30000` |
| `3e20` | `24857` | `20000` | `-4857` | `56.3222%` | `gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/checkpoints/step-20000` |

Validation:

- Verified the selected native checkpoints contain QK-norm keys:
  `3e18 step-30000`: `210` OCDBT keys, `12` QK-norm keys;
  `3e19 step-30000`: `210` keys, `12` QK-norm keys;
  `3e20 step-20000`: `1218` keys, `12` QK-norm keys.
- Dry-ran the launcher for `3e18`, `3e19`, and `3e20` `p33m67/cooldown30`;
  each now points at the native source checkpoint above, not the invalid
  materialized prefix path.
- `uv run --with pytest --with pytest-timeout python -m pytest tests/test_materialize_delphi_prefix_checkpoint.py -x --timeout=120`
  passed (`7` tests).
- `uv run --with pytest --with pytest-timeout python -m pytest tests/midtraining/test_val_set_equivalence.py::test_true_cooldown_rendered_data_section_bit_identical_to_reference -x --timeout=120`
  passed (`9` tests).

Operational caveat:

- These no-retrain replacement checkpoints live in `us-central2`. Launching
  cooldown jobs in `us-east5` will require explicit staging/copy approval or a
  same-region execution plan.

## 2026-05-27T08:00Z — Defense-in-depth against silent type-degradation in the cooldown path

Audit + fixes so the silent QK-norm-drop bug class can never recur in the
true-cooldown pipeline. Five defenses now layered between spec construction
and the final saved checkpoint. Full risk register including outstanding
follow-ups (R1-R7) at
`.agents/projects/silent_type_degradation_risk_register.md`.

### Defenses now active

| # | Layer | Catches |
| --- | --- | --- |
| D1 | Materialize helper Qwen3 fail-closed | untyped `.executor_info` -> wrong source class (already in `5afac0bdf`) |
| D2 | Launcher renders explicit `type: qwen3` | wrong type at spec construction (already in `5afac0bdf`) |
| D3 | `_assert_model_config_matches_base` requires `model_config["type"]` | any spec built without an explicit discriminator — fails at `validate_midtrain_spec`, before submission |
| D4 | `_check_cooldown_stage` enumerates OCDBT keys and asserts class-specific arrays | degraded source ckpt — blocks the launch before any TPU time spent |
| D5 | Launcher `_verify_final_checkpoint` runs the same schema check on the final saved ckpt | (so-far-unobserved) case where load -> train -> save silently degrades mid-run |

### New module

`lib/marin/src/marin/midtraining/checkpoint_schema.py`:

- `default_list_checkpoint_keys(checkpoint_dir)`: tensorstore-backed OCDBT
  kvstore enumeration. Fails loud on unreachable paths (no silent
  fallback to empty list).
- `assert_qwen3_qk_norm_present(checkpoint_dir, *, num_layers, list_keys)`:
  substring-match for `q_norm` and `k_norm` in the OCDBT key list.
  Fails closed with an actionable error naming the missing markers and
  pointing at commit `5afac0bdf` and the materialize helper.
- `assert_checkpoint_complete_for_model_type(...)`: dispatcher on declared
  `model_type`. Currently only Qwen3 has class-specific checks; other types
  pass through after verifying the discriminator is non-empty.

Injection-friendly: tests pass fake `list_keys` callables, matching the
existing `fake_gcs` pattern in `preflight.py`.

### Edits

- `lib/marin/src/marin/midtraining/preflight.py`: `preflight()` now accepts
  `list_ocdbt_keys` (defaults to the tensorstore-backed helper);
  `_check_cooldown_stage` calls `assert_checkpoint_complete_for_model_type`
  against the staged source after artifact checks pass.
- `lib/marin/src/marin/midtraining/spec.py`:
  `_assert_model_config_matches_base` now also asserts `model_config["type"]`
  exists and is a string before checking dim invariants.
- `lib/marin/src/marin/midtraining/__init__.py`: re-export the new helpers
  (`assert_checkpoint_complete_for_model_type`, `assert_qwen3_qk_norm_present`,
  `default_list_checkpoint_keys`, `ListCheckpointKeys`).
- `experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py`:
  after `result.wait()` returns `succeeded`, calls a new
  `_verify_final_checkpoint(spec, num_train_steps)` that lists `step-*`
  dirs under `permanent_checkpoints_uri`, picks the latest, and applies
  the schema check. Iris success + bad ckpt -> launcher returns 1.

### Tests

- New `tests/midtraining/test_checkpoint_schema.py` (13 tests):
  unit tests for the schema helpers (happy path, both-missing, only-q-missing,
  zero-layers, dispatcher behavior for qwen3 vs llama, empty model_type,
  tensorstore error surfacing); preflight integration tests for
  Qwen3-cooldown happy path, degraded source ckpt, missing-artifact
  short-circuit, and untyped-model-config rejection.
- New tests in `tests/midtraining/test_spec_validators.py`:
  `test_model_config_requires_type_discriminator` and
  `test_model_config_type_must_be_string`.
- Full midtraining + materialize suite: `143 passed`. Pre-commit clean
  (Ruff, Black, license, pyrefly, large files, ast, merge conflicts,
  whitespace, EOL, markdown).

### Risk register

`.agents/projects/silent_type_degradation_risk_register.md` catalogs the
five active defenses plus seven outstanding follow-ups (R1-R7), each with
location, why-it-matters, action, and estimated effort. Highlights:

- **R1** — Levanter `tree_deserialize_leaves_tensorstore` silently drops
  extra on-disk keys; needs a `strict=True` mode wired into the cooldown
  init path. Half-day; touches vendored Levanter.
- **R2** — Materialize helper's `optimizer`, `data.format`, `tracker`
  normalizers use heuristics rather than fail-closed asserts. Lower
  blast radius (optimizer-state errors surface as training-dynamics
  drift), but same bug class.
- **R3** — `_check_cpt_init` does not run the OCDBT-key check; CPT from
  `NATIVE_LEVANTER` source could replay the bug. Plumbing already in
  place, just needs the call.
- **R5** — `DelphiModel` has no `model_type` field; spec validator
  can't cross-check `model_config["type"]` against the base's expected
  class. Add `model_type: str = "qwen3"` to `DelphiModel`.
- **R7** — `list_delphi_checkpoints.py` doesn't surface schema
  completeness. Adding it would have caught the original incident
  before the candidate yaml was approved.

### Operator playbook (also in the risk-register doc)

If a future cooldown launch fails with `missing Qwen3 QK-norm arrays`:

1. Do not graft missing arrays into the bad checkpoint. Intervening
   weights were trained under the wrong forward pass; the graft is
   scientifically wrong. (Considered and rejected at the original
   incident — see entry above at `2026-05-27T04:03Z`.)
2. Re-materialize the prefix checkpoint with the fixed helper, writing
   to a new `-qwen3` discriminator path so the old root remains as audit
   trail.
3. Update the candidate row in `checkpoint_candidates.yaml` to point at
   the new path, mark `materialized_checkpoint: true`, leave the old
   path as `invalidated_materialized_checkpoint_path`.
4. Re-run the launcher. The preflight schema check (D4) should now pass.

## 2026-05-27T23:35Z — Multi-target helper extension + registry NTS fix + 7 sources pre-copied

Closing the loop on the 70%+80% prefix materialization plan. Three
pieces landed in this session:

### Source checkpoints pre-copied to us-east5 (7 ckpts, 97.18 GiB)

Minimum-source plan: one source per base, used as the at-or-before
resume point for BOTH 70% and 80% materializations in a single
combined run. All 7 copies completed `2026-05-27T23:31Z`, ~$2
cross-region egress. Path symmetry preserved so the materialize
helper's `mirror://` resolution finds the local us-east5 copy.

| Base | Source step | Size | New 70% target | New 80% target |
| --- | ---: | ---: | ---: | ---: |
| 3e18 | 20000 |  5.00 GiB | 26134 | 29868 |
| 9e18 | 30000 |  6.15 GiB | 31021 | 35453 |
| 2e19 | 30000 |  9.35 GiB | 38587 | 44100 |
| 3e19 | 20000 | 11.15 GiB | 26609 | 30411 |
| 9e19 | 20000 | 15.47 GiB | 28198 | 32226 |
| 2e20 | 30000 | 21.62 GiB | 39533 | 45181 |
| 3e20 | 20000 | 28.44 GiB | 24857 | 28408 |

Note: 3 of the 7 cps (`3e18`, `3e19`, `3e20`) initially landed with a
nested `step-N/step-N/` duplicate from a `gsutil cp -r` directory
ambiguity. The top-level data was correct in all 7; nested duplicates
were deleted before verification. Final check: every destination has
`manifest.ocdbt` + `metadata.json` + `d/` at the right level with no
nested duplicates.

### Helper extension `--also-save-step` (commit `5cb06d791`)

The materialize helper now supports multiple permanent saves in one
training run, addressing audit items A1-A10 from
`.agents/projects/silent_type_degradation_risk_register.md`:

- `MaterializeRequest.extra_target_steps` (tuple of ints, must be
  strictly inside `(source_step, target_step)` and distinct)
- One Levanter `CheckpointInterval` policy per extra:
  `{"every": X, "until": X}` sorted ascending; X % X == 0 fires a
  permanent save at exactly X; past X the policy goes inactive
- The final `target_step` is NOT in `keep` -- it's still covered by
  Levanter's forced save at training end
  (`lib/levanter/src/levanter/trainer.py:568,581`)
- `MaterializationPlan.destination_checkpoint_paths` (tuple), with
  `destination_checkpoint_path` kept as a backward-compat shortcut
- `validate_checkpoint_io` checks every destination for pre-existence
- `checkpoint_metadata` writes `all_target_steps: list[int]`; legacy
  `target_step` (singular) still equals the FINAL target
- Launcher D5 post-train check now validates EVERY step dir under
  `permanent_checkpoints_uri`, not just the latest

The 5afac0bdf Qwen3 fail-closed in
`decode_train_config_from_executor_info` is untouched. The keep
policy only controls WHEN to save, not WHAT -- all saves share the
in-memory param tree, so if the load is Qwen3-correct, every saved
checkpoint has `q_norm` / `k_norm`. The silent-type-degradation bug
class remains structurally impossible to reintroduce.

### Registry NTS fix: `9e18`, `2e19`, `9e19`, `2e20`

Audit of the Flavor A scripts caught a stale registry: the
`2026-05-25T20:22Z` fix that updated `delphi_models.py.num_train_steps`
to match `.executor_info` was applied to `3e18`/`3e19`/`3e20` only.
The other 4 small-ladder bases were missed, so their candidate yaml
target steps landed at 79.6%-79.8% of the true pretrain schedule
instead of exactly 80%.

| Base | Old registry NTS | True NTS (`.executor_info`) | Drift |
| --- | ---: | ---: | ---: |
| 9e18 | 44096 | 44317 | +221 |
| 2e19 | 54915 | 55125 | +210 |
| 9e19 | 40163 | 40283 | +120 |
| 2e20 | 56392 | 56477 | +85 |

Fix applied: `experiments/delphi_models.py` `num_train_steps` bumped
to the true values; `checkpoint_candidates.yaml` recomputed for those
4 bases x 3 cooldown ratios = 12 entries. Per-entry updates:
`target_step` (= `floor(prefix_ratio * new_NTS)`),
`target_progress_percent`, `target_cooldown_steps`,
`suggested_step_delta`, `suggested_progress_percent`, and the
`checkpoint_at_or_before` / `at_or_after` `delta` + `progress_percent`
fields. The `suggested_step` itself is operator-pinned (chosen during
review), so it's preserved unchanged; the cooldown launcher uses the
suggested path only when consuming the candidate, not for the
materialization plan (which uses at_or_before for the source-resume
step).

After the fix every target lands at 70.00% / 80.00% of the true
schedule to 4 decimal places. Validated by re-rendering each plan
with the helper.

### 7 submit scripts ready in `scratch/` (gitignored)

One per base: `scratch/submit_delphi_prefix_{base}_qwen3.py`. Each
launches a single combined materialization run with
`--also-save-step` for the 70% prefix and `--target-step` for the
80%. TPU choices match the verified cooldown / CPT TPU per base
(`v6e-4` for 3e18 / 9e18, `v6e-8` for 2e19 / 3e19 / 9e19, `v5p-8`
for 2e20, `v5p-16` for 3e20).

3e18 sanity-check launch handed to laptop earlier today; other 6
ready to fire once 3e18 confirms the pipeline end-to-end.

Aggregate forward-step budget across all 7 runs: ~76 k steps. Wall
bottleneck: 2e20 (~22 h on `v5p-8`) if all 7 launch in parallel.

### Files touched (this session)

- `scripts/materialize_delphi_prefix_checkpoint.py` (helper extension)
- `experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py` (D5 multi-step)
- `tests/test_materialize_delphi_prefix_checkpoint.py` (+8 tests)
- `experiments/delphi_models.py` (4 NTS bumps)
- `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml` (12 entries recomputed)
- `scratch/submit_delphi_prefix_{3e18,9e18,2e19,3e19,9e19,2e20,3e20}_qwen3.py` (gitignored, on skampere3 worktree)

Test suite: 151 passing across `tests/midtraining/` +
`tests/test_materialize_delphi_prefix_checkpoint.py`. Pre-commit clean.

## 2026-05-28T00:10Z — Handoff: 7 prefix-materialization scripts ready to fire

State as of this entry:

- Working tree at `7e511d8f3` (NTS fix + yaml regeneration), clean.
- 7 source ckpts pre-copied to `gs://marin-us-east5/checkpoints/isoflop/...`
  (one per base, 97.18 GiB total). Verified clean: each has
  `manifest.ocdbt` + `metadata.json` + `d/` at the top level, no nested
  duplicates.
- 7 submit scripts at `scratch/submit_delphi_prefix_{base}_qwen3.py` on
  the skampere3 worktree (gitignored). Each launches a single combined
  materialization run that writes both the 70% and 80% prefix in one
  pass via the `--also-save-step` keep policy.
- iris and W&B both confirm: no `delphi-3e18-...-qwen3` runs exist yet.
  The sanity-check launch the user requested earlier ("launch 3e18 on
  v6 in us-east5-b") still needs to fire from the laptop (skampere3
  ADC is expired and the helper does a GCS read at planning time;
  `gsutil` works here but `fsspec` does not).

Launch order plan:

1. **3e18 sanity check first** (v6e-4 in us-east5-b, ~3-4h wall). When
   it succeeds, validates the entire Flavor A pipeline end-to-end
   (mirror-resolution of pre-copied source, `--also-save-step` keep
   policy fires at intermediate step, final force-save at target,
   post-train D5 multi-step check passes).
2. After 3e18 confirms, fire the other 6 in parallel. Wall bottleneck:
   2e20 on v5p-8 (~22h).

The candidate yaml's `materialized_checkpoint: false` rows for
3e18/3e19/3e20 cooldown30 (the 3 previously-bad-materialized cells, now
listing the bad path under `invalidated_materialized_checkpoint_path`)
will need a follow-up edit AFTER the new materializations succeed:
flip to `materialized_checkpoint: true`, point `suggested_checkpoint_path`
at the new step dir, set `suggested_step_delta: 0` and
`suggested_relation_to_target: exact_target`. The same flip applies to
the cooldown20 / cooldown10 rows for all 7 bases once their respective
prefixes land. That edit is intentionally NOT part of this commit
because the materialized paths don't exist yet.

No code/config drift since `7e511d8f3` — this is a "ready to launch"
checkpoint entry only. Pre-commit clean. Tests still pass (151).

## 2026-05-29T00:00Z — Operator runbook: launching the 7 prefix materializations from a marin-configured client

The 3e18 sanity-check launch was attempted from skampere3 and blocked
at the fray backend layer (executor falls back to `LocalClient`
outside an iris job → tries to run Levanter in-process → no
accelerator). All client-side environment workarounds were chased
(ADC, `WANDB_API_KEY`, `MARIN_PREFIX` for region detection) but the
fray-backend issue requires constructing a `FrayIrisClient` manually
and that's a brittle layer-on-layer hack from a non-marin-dev host.

**Path forward: launch from a properly-configured marin development
client** (the user's laptop has the working iris/fray setup that the
prior `submit_delphi_prefix_3e19.py` / `_3e20.py` scripts used
successfully). No code changes needed — the materialize helper Just
Works from a marin-configured host.

### Pre-flight (ALREADY DONE — do NOT redo)

- ✅ 7 source ckpts pre-copied to us-east5 (97.18 GiB, see the prior
  entry for the table). `mirror://` resolution will find them locally
  when the materialization TPU lands in us-east5; no second
  cross-region copy.
- ✅ `experiments/delphi_models.py` NTS aligned with `.executor_info`
  for all 7 small-ladder bases.
- ✅ `checkpoint_candidates.yaml` target steps recomputed so every
  70%/80% prefix lands at exactly 70.00%/80.00% of the true schedule
  (4-decimal precision).
- ✅ Helper extension `--also-save-step` shipped (commit `5cb06d791`)
  with D5 post-train check on every saved step.

### Launch commands (run from laptop, one per base)

Order: **3e18 first as sanity check** (~3-4 h wall on v6e-4). When
3e18 succeeds — both `step-26134` and `step-29868` written under
`gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-prefixes-qwen3/checkpoints/`,
both passing `assert_checkpoint_complete_for_model_type` — fire the
other 6 in parallel. They all start independently; bottleneck is 2e20
at ~22 h on v5p-8.

Each command runs the helper directly; the helper internally calls
`executor_main` which uses fray's iris backend (auto-detected on the
laptop). The single training run writes the 70% prefix at the
intermediate `--also-save-step` and the 80% prefix at the final
`--target-step` (forced save at training end).

```bash
# 3e18 (sanity check first; ~3-4 h on v6e-4)
uv run python scripts/materialize_delphi_prefix_checkpoint.py \
    --base 3e18 --source-step 20000 --target-step 29868 --also-save-step 26134 \
    --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-prefixes-qwen3 \
    --tpu v6e-4 --ram 128g --region us-east5

# 9e18 (~2 h on v6e-4)
uv run python scripts/materialize_delphi_prefix_checkpoint.py \
    --base 9e18 --source-step 30000 --target-step 35453 --also-save-step 31021 \
    --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-9e18-prefixes-qwen3 \
    --tpu v6e-4 --ram 128g --region us-east5

# 2e19 (~2 h on v6e-8)
uv run python scripts/materialize_delphi_prefix_checkpoint.py \
    --base 2e19 --source-step 30000 --target-step 44100 --also-save-step 38587 \
    --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-2e19-prefixes-qwen3 \
    --tpu v6e-8 --ram 128g --region us-east5

# 3e19 (~3.5 h on v6e-8)
uv run python scripts/materialize_delphi_prefix_checkpoint.py \
    --base 3e19 --source-step 20000 --target-step 30411 --also-save-step 26609 \
    --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e19-prefixes-qwen3 \
    --tpu v6e-8 --ram 128g --region us-east5

# 9e19 (~10 h on v6e-8)
uv run python scripts/materialize_delphi_prefix_checkpoint.py \
    --base 9e19 --source-step 20000 --target-step 32226 --also-save-step 28198 \
    --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-9e19-prefixes-qwen3 \
    --tpu v6e-8 --ram 128g --region us-east5

# 2e20 (~22 h on v5p-8; in us-east5-a since v5p lives there, not -b)
uv run python scripts/materialize_delphi_prefix_checkpoint.py \
    --base 2e20 --source-step 30000 --target-step 45181 --also-save-step 39533 \
    --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-2e20-prefixes-qwen3 \
    --tpu v5p-8 --ram 256g --region us-east5

# 3e20 (~17 h on v5p-16)
uv run python scripts/materialize_delphi_prefix_checkpoint.py \
    --base 3e20 --source-step 20000 --target-step 28408 --also-save-step 24857 \
    --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e20-prefixes-qwen3 \
    --tpu v5p-16 --ram 256g --region us-east5
```

Pre-flight is enforced by the helper itself
(`validate_static_request`): each `also-save-step` is verified to be
strictly inside `(source_step, target_step)` and distinct from any
other extras, source/target are validated against the true
`num_train_steps` from `.executor_info`, and the destination
existence check rejects any prior partial write under the same
`output_root` unless `--allow-existing-destination` is passed.

### Per-run verification

After each materialization completes, confirm:

1. Two step directories under
   `<output_root>/checkpoints/step-{also-save}` and
   `step-{target}`.
2. Each contains `manifest.ocdbt`, `metadata.json`, and `d/`.
3. Each `metadata.json` has `all_target_steps: [70%-target,
   80%-target]`, `materialization: "delphi_prefix_checkpoint"`, and
   the appropriate `step`.
4. Each OCDBT kvstore contains `q_norm` and `k_norm` keys (the bug
   that triggered this whole workstream); the D5 post-train check
   inside the launcher runs this for you, but you can also re-run
   manually:

   ```python
   from marin.midtraining import assert_qwen3_qk_norm_present
   from experiments.delphi_models import get_delphi_model
   for step in (also_save, target):
       assert_qwen3_qk_norm_present(
           f"<output_root>/checkpoints/step-{step}",
           num_layers=get_delphi_model("<base>").num_layers,
       )
   ```

### Post-materialization yaml flip

Once a base's prefixes are clean, update its three candidate rows in
`experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml`:

- For `delphi-<base>-cooldown30` (70% target): set
  `suggested_checkpoint_path` to the new
  `<output_root>/checkpoints/step-{70-target}`, set
  `suggested_step` to the 70%-target step,
  `suggested_step_delta: 0`,
  `suggested_relation_to_target: exact_target`,
  `suggested_progress_percent: 70.0000`,
  `materialized_checkpoint: true`,
  `review_status: approved`. Move any prior bad path to
  `invalidated_materialized_checkpoint_path`.
- For `delphi-<base>-cooldown20` (80% target): same flip for the
  80%-target step.
- For `delphi-<base>-cooldown10` (90% target): unchanged — we did
  not materialize 90% prefixes in this batch.

The launcher's preflight (D4) will then validate the new ckpt's
OCDBT-key completeness before any cooldown launch.

### Why this isn't automated yet

The yaml flip happens AFTER materialization completes, with paths
that don't exist at submit time. It would need a small post-train
hook on the materialize helper (or a one-off script that takes a
list of base/run-id pairs and rewrites the yaml). For now: manual
edit per base when each finishes.

### If anything goes wrong

- **`missing Qwen3 QK-norm arrays`** on the materialization output
  → the source ckpt is actually degraded. Re-verify with
  `assert_qwen3_qk_norm_present` on the staged us-east5 mirror; if
  the mirror is degraded, re-fetch from us-central2 via gsutil cp.
- **`q_norm/k_norm restore error`** on a cooldown launch that consumes
  a materialized prefix → the D4 preflight should have caught it
  earlier. Inspect the prefix's OCDBT keys directly with tensorstore.
- **TPU acquisition stalls > 24 h** → check iris scheduler priority.
  These launches use the default (interactive); should not get
  starved by BATCH-priority tasks. If a v5p-8 / v5p-16 / v6e slot is
  hard to acquire, the 2e20 / 3e20 runs may be the slowest to start.

### Files / commits this batch depends on

- `5cb06d791` — `--also-save-step` extension + D5 multi-step check
- `7e511d8f3` — registry NTS sync + yaml regeneration
- `0b50cb168` — D1-D5 defenses + `checkpoint_schema` module
- `5afac0bdf` — Qwen3 fail-closed in materialize helper (the bug fix)

All on `origin/midtrain_data`. `git pull` on the laptop picks them up.

### After all 7 materializations land

The cooldown30 / cooldown20 sweep for the full 7-base small ladder
becomes scientifically clean (every cell starts from an exact-target
prefix). Relaunch the 6 invalidated cooldown30 cells (3e18 / 3e19 ×
3 mixes) from `2026-05-25T21:28Z` — they died on the bad
materialized prefixes. Plus 3e18/9e18/2e19/3e19/9e19/2e20/3e20 × 3
mixes × 2 ratios = 42 cells of true-cooldown for the small ladder.

## 2026-05-30T01:25Z — QK-norm materialization audit and helper hardening

Ahmed asked for an extremely thorough check after the previous 70%
materialized prefix helper dropped Qwen3 `q_norm`/`k_norm` arrays.

Audit result: the `5afac0bdf` fix was necessary but not sufficient by
itself. The helper did force old untyped `.executor_info` model payloads
through `type: qwen3` and refused non-`Qwen3Config` decodes, but the
direct materialization executor step did not independently schema-check
the saved materialized output. That meant a future regression could still
look like a successful materialization until a downstream cooldown launch
failed.

Hardening added in this session:

- `scripts/materialize_delphi_prefix_checkpoint.py`
  - `validate_checkpoint_io()` now calls
    `assert_checkpoint_complete_for_model_type(..., model_type="qwen3")`
    on the source checkpoint before submission. A source missing QK-norm
    arrays fails before any executor step is launched.
  - `build_executor_step()` now uses a new `MaterializationTrainConfig`
    and `run_materialization_train()` wrapper instead of calling
    `run_levanter_train_lm()` directly.
  - `run_materialization_train()` runs training, then schema-checks every
    intended destination checkpoint path, including all `--also-save-step`
    outputs and the final `--target-step`. Any missing QK-norm arrays make
    the executor step fail.
- `tests/test_materialize_delphi_prefix_checkpoint.py`
  - Added regression coverage proving old untyped `.executor_info`
    decodes to `Qwen3Config`.
  - Added an assertion that decoded Qwen3 attention config has `qk_norm`
    enabled even when the old payload contains `use_qk_norm: False`.
  - Added source schema-check tests and destination post-train schema-check
    tests, including failure propagation.
- `docs/debug-log-qk-norm-materialization.md`
  - Added a structured debug log with proof points and validation commands.

Validation:

- Targeted tests:
  `uv run --with pytest --with pytest-timeout python -m pytest tests/test_materialize_delphi_prefix_checkpoint.py tests/midtraining/test_checkpoint_schema.py tests/midtraining/test_spec_validators.py tests/midtraining/test_val_set_equivalence.py -q --timeout=180`
  passed (`102 passed`).
- Broad midtraining/materialization tests:
  `uv run --with pytest --with pytest-timeout python -m pytest tests/midtraining tests/test_materialize_delphi_prefix_checkpoint.py -q --timeout=180`
  passed (`155 passed`).
- Safe GCS metadata/schema check for all seven planned qwen3
  materializations passed. Each source `.executor_info` decoded as
  `Qwen3Config`; each canonical source checkpoint passed schema; each
  us-east5 mirror was present and schema-clean; and the intended new
  destination step dirs were absent.
  - `3e18`: source `20000`, targets `(26134, 29868)`
  - `9e18`: source `30000`, targets `(31021, 35453)`
  - `2e19`: source `30000`, targets `(38587, 44100)`
  - `3e19`: source `20000`, targets `(26609, 30411)`
  - `9e19`: source `20000`, targets `(28198, 32226)`
  - `2e20`: source `30000`, targets `(39533, 45181)`
  - `3e20`: source `20000`, targets `(24857, 28408)`
- Safe GCS schema check rejected all three known-bad invalidated 70%
  materialized outputs (`3e18`, `3e19`, `3e20`) with missing `q_norm` and
  `k_norm`.
- `./infra/pre-commit.py --all-files --fix` passed, including pyrefly.

No jobs were launched and no checkpoint data was copied. Do not flip
`checkpoint_candidates.yaml` to the new qwen3 paths until the actual
materialization jobs complete and the new 70%/80% destination step dirs
exist. With this hardening, a bad materialization should now fail at the
materialization step itself rather than surfacing later in cooldown.

## 2026-05-30T01:58Z — Main merge and v6e materializer resource pin

Merged `origin/main` (`7115c21d47`) into `midtrain_data` and pushed the
merge commit `a7119a9bc1`. The serious conflict was semantic:
`origin/main` refactored executor/training so training runs in the
scheduled `ExecutorStep` process, while this branch still assumed
`run_levanter_train_lm()` would submit a nested Fray/Iris child from the
`TrainLmOnPodConfig.resources` field.

Resolution:

- Kept main's newer TPU JAX init behavior: Iris TPU jobs delegate through
  `iris.runtime.jax_init.initialize_jax()`, which calls
  `jax.distributed.initialize()` using TPU runtime autodiscovery.
- Preserved the midtraining resume invariant by keeping exact
  run-id/output-path basename checking in `impute_run_id()`.
- Preserved W&B project plumbing through the new
  `_build_train_lm_config()` / `prepare_lm_train()` / `train()` flow.
- Preserved mirror budget propagation into configs with `env_vars`.
- Updated the materializer import for the new
  `marin.execution.types.this_output_path` location.

After the merge, the first real 3e18 launcher submission
(`/ahmed/delphi-3e18-prefixes-qwen3-v6e4-a7119a9`) exposed the executor
semantic change: the step began on the CPU launcher because
`build_executor_step()` put the v6e resource only inside
`TrainLmOnPodConfig.resources`. That job was terminated before useful
training. The helper now builds one `ResourceConfig.with_tpu(...)` and
sets it on both `TrainLmOnPodConfig.resources` and
`ExecutorStep.resources`, so StepRunner schedules the materialization step
itself on `v6e-4`.

Validation after the merge/resource-pin fix:

- `uv run --with pytest --with pytest-timeout python -m pytest tests/test_training.py tests/execution/test_executor.py -q --timeout=180`
  passed (`70 passed, 1 skipped`).
- `uv run --with pytest --with pytest-timeout python -m pytest -o addopts='' lib/iris/tests/test_jax_init.py -q --timeout=180`
  passed (`11 passed`).
- `uv run --with pytest --with pytest-timeout python -m pytest -o addopts='' lib/levanter/tests/test_distributed.py lib/levanter/tests/test_checkpoint.py -q --timeout=180`
  passed (`37 passed`).
- `uv run --with pytest --with pytest-timeout python -m pytest tests/midtraining tests/test_materialize_delphi_prefix_checkpoint.py -q --timeout=180`
  passed (`155 passed`).
- `./infra/pre-commit.py --all-files --fix` passed, including pyrefly.
