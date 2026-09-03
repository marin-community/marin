---
name: deploy-hero-change
description: "Deploy a significant code change (backend, kernel, optimizer, data path) to the live hero run: relaunch it under a new run id from a permanent checkpoint, compare against the old run over a trial window, and roll back if the gate fails; use only when the user asks to deploy a change to the hero."
---

# Deploy a change to the live hero run

The hero is a multi-rack production run whose checkpoints, W&B history, and
dashboards are shared state. A change lands as a new run that continues the old
run's lineage from a named permanent checkpoint, is compared against the old run
on the same batches for a trial window, and is rolled back by relaunching the old
commit if the gate fails. `manage-hero-run` governs the run itself (launch record,
babysitting cadence, DRI); this skill is the swap protocol layered on top of it.

Single-rack validation (the d768 rung, a 1-rack restore smoke) cannot reach
failures that need more than one expert-parallel domain in one job: cross-domain
collectives, executable alternation across domains, save/resume of the new tree at
scale. With 11 of the 12 GB200 racks on the hero, the trial window is the first
multi-rack exercise of the change. Plan the window and the rollback for that.

## 1. Agree the plan with the user before touching anything

Write down, and get agreement on:

- The handoff checkpoint: the newest permanent checkpoint `step-N`. The trial
  starts there, so the old run must keep training past it long enough to produce
  the control window (200 steps by default, at the hero's pace about 1 hour).
- The new run id: `hero-<change>-step<N>k`, for example
  `hero-ragged_a2a-ep-step54k`. It names the W&B run, the Iris coordinator, and
  the checkpoint tree (`launch_scaling_ladder` derives the output path from
  `--run-id`, so a new id is a fresh tree by construction). Keep the W&B project;
  dashboards, the public report, and alerts key on it.
- The gate. Common criteria: loss tracks the control step for step up to bf16
  noise (or by the amount the change is meant to improve); MFU equal or better;
  token-drop rate equal or better; the metric the change targets moves as
  predicted; no unexpected movement anywhere else; no crash, restart, watchdog,
  or new alert. Write the expected direction and size of each before launch.
- What the window will not exercise (evals at the run's cadence, the new run's own
  temporary save and resume, a retry). List them in the go/no-go.
- The clock: swap early enough that a rollback lands in working hours. Report
  times in the user's zone.
- Communication: the Slack thread and the hero's status-log issue, which the
  tracker bots read. Post at each transition: kill, launch, first steps,
  go/no-go, rollback.

## 2. Land the change on main first

- Merge the code change and the launcher change together. `trigger_hero.sh` is
  the launch record: it carries `RUN_ID` and `HANDOFF_CHECKPOINT`
  (`--initialize-from-checkpoint <old>/checkpoints/step-N`). The launched command
  must be main's script at a verified SHA, from a pristine worktree.
- `--initialize-from-checkpoint` appends the named checkpoint directory to the
  resume search paths and makes a checkpoint mandatory, so the first launch
  restores the full state (params, optimizer, step, data position) from exactly
  that step and later restarts prefer the new run's own, newer checkpoints. The
  old run's tree is never written again.
- Inventory downstream reporting for the run id: the public W&B report's pinned
  run set (the Grafana bridge follows it), any tracker that hard-codes the id,
  metric keys the change renames (those need their own PR). Grafana hero-health
  enrols by the `hero-*-coord-*` job naming and needs nothing.

## 3. Build and rehearse the runbook

Keep the swap as small numbered scripts that share fail-closed helpers, and run
every guard as a dry run against the live cluster before the day:

- Every query helper returns non-zero on failure and callers stop on "unknown";
  an empty answer is never "gone" or "clean". Iris CSV output carries a header
  row and CRLF; `grep -c` exits 1 on zero matches; pod names are k8s-sanitized
  (`_` becomes `-`) and truncated; `Loaded checkpoint from` is not logged when
  the candidate is itself a search path, so key restore detection on `Loading
  checkpoint from` and the loop entering.
- Preflight (read-only): deploy worktree at `origin/main` and clean; rollback
  worktree at the old run's exact SHA and clean; handoff checkpoint complete
  (`metadata.json` present) and in the expected layout; the new run's tree empty;
  no other live coordinator, gang, or hero pod; credentials for Iris, kubectl,
  the object store, and W&B all work.
- Launch guard: refuse unless the old run's coordinator is terminal, no
  coordinator for the new run id is live, and the worktree is pristine. Capture
  the submit output to a file and verify exactly one coordinator for the new run
  id afterwards; a second submission would compete for the same tree.
- Rollback: cancel the new coordinator (this also stops the Iris retry loop)
  and confirm it is terminal; relaunch the old commit's `trigger_hero.sh` from
  the pristine rollback worktree under the old run id.
  The old tree resumes its own newest checkpoint; confirm that is the intended
  anchor before launching.
- Get an independent review of the scripts and fix or refute every finding;
  fail-open guards are the defect class to ask the reviewer for.

Submit as `IRIS_USER=marin` so the run is attributed to the project, not a person.

## 4. Execute

1. Preflight.
2. When the old run is 200 steps past `step-N`, cancel its coordinator and
   confirm the cancel took (coordinator terminal). The new gang can be submitted
   at once; Kueue admits it all-or-nothing once the racks free. A cancel that did
   not take leaves the new gang pending behind the old one indefinitely.
3. Launch from main. Expect a cold start: restore 3 to 5 minutes, then compile
   of every train-step executable (about 23 + 7 minutes at d6144 on 704 devices,
   warm cache 3 to 5 minutes), while the loader's first prefetches may take
   minutes. The startup watchdog fires at 80 minutes and the step watchdog at
   15; do not kill a compiling run.
4. Publish a W&B report before the first step: two run sets (old run id, new run
   id) over the absolute step range `N` to `N+200`, panels for loss, cross
   entropy, MFU, step time, drop fraction, routing entropy, grad norm, router
   losses, peak memory, tokens/s, plus a wider context grid. Give the user the
   URL; name it after the change.
5. Monitor from finelog, not W&B: `levanter.metrics` has every step, W&B refuses
   steps below a resumed run's counter and the public API lags. Poll job state,
   `task_attempts.attempt_id`, the last logged step and its age, watchdog and
   `JaxRuntimeError` lines, and the gate metrics against the control at the
   same step: loss, load-balancing and router losses, drop fraction, peak HBM
   against the allocator's release threshold. A corrupted restore shows within a
   few steps as loss and router losses far above the control's; judge against
   the paired control, or against fixed thresholds only when the user agreed
   them for this deployment. Emit only on change.
6. Compare per step against the control (same batches): join the two runs on
   step and report mean and max loss delta, plus the gate metrics.

## 5. Decide

Go: leave it running, post the numbers and the report, update the status issue.
Anything short of the agreed gate, a hang, a retry loop, or a signature the
change does not explain: roll back without waiting for more attempts. Each Iris
retry re-restores the handoff and burns the full cluster for the compile plus
the replay.

After a rollback, verify the old run's first steps match its own earlier
trajectory in finelog (the replay is a free determinism check), note that W&B
shows no new rows until it passes its old counter, update the status issue, and
file the failure as an issue with the evidence below.

## Localizing a silent hang

A hung collective logs nothing at any level. Use:

- NCCL RAS periodic samples in `telemetry_v1.levanter` (`collective_operations`
  per communicator, `rank_statistic` minimum and maximum). One member of every
  cross-rack communicator behind by the same count means one whole rack never
  entered those collectives.
- `iris.task` per-task `cpu_millicores` over the stall window: ranks busy-polling
  in an AllReduce burn about 2 cores each; the stuck rack's tasks sit at half
  that. Task indices map to racks in blocks of 16 at d6144.
- `task_attempts.node_name` for both attempts: the same rack twice on the same
  nodes points at hardware, a different rack points at the code.
- `iris process profile threads` shows every rank inside `train_step`; it cannot
  see below XLA. A GPU-side stack needs the CUDA core-dump arming at launch.

Record which executable alternation preceded the hang (watch step to plain step,
eval to train step); executables that share the NCCL symmetric-memory arena are a
known failure family (#8861, #8870).

## References

- `manage-hero-run` for the run record, babysitting, retention, and seal.
- `wandb-reporting` for report conventions.
- `experiments/grug/moe_hero_ep/trigger_hero.sh`, `launch_scaling_ladder.py`
  (`--initialize-from-checkpoint`, #8868).
- `docs/ops/training-stall-alert-contract.md` for the RAS query.
- The 2026-09-02 ragged all-to-all swap: #8506, #8861, #8870.
