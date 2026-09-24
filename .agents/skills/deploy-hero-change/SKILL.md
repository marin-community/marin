---
name: deploy-hero-change
description: "Deploy a significant code change (backend, kernel, optimizer, data path) to the live hero run: relaunch it under a new run id from a permanent checkpoint, compare against the old run over a trial window, and roll back if the gate fails; use only when the user asks to deploy a change to the hero."
---

# Deploy a change to the live hero run

Use `manage-hero-run` for the run record, DRI, retention, and babysitting. This
checklist governs a code cutover and its trial. Single-rack validation does not
exercise cross-domain collectives or checkpoint recovery at production scale.

## 1. Agree the plan

Record the following with the user before changing the live run:

- A complete permanent handoff checkpoint `step-N`. Use the newest scheduled
  permanent checkpoint (every 6000 steps, about 26 hours at the hero's pace), or
  request one from the old run's `training-control` endpoint with the
  `request-permanent-checkpoint` header value. Only a run whose code includes
  that action can take the request; otherwise schedule the cutover shortly after
  a scheduled permanent checkpoint. Do not hand off from an hourly checkpoint:
  it expires three days after it is written.
- A new run ID and checkpoint tree, the unchanged W&B entity/project, and the
  verified parent history boundary. Follow the
  [launcher procedure](../../../experiments/grug/moe_hero_ep/README.md#hero-cutovers-and-wb-lineage).
- A matched control window, 200 steps by default. The old run must record those
  steps after the handoff checkpoint before it is stopped.
- Explicit gates for loss, throughput/MFU, token drops, router metrics, and the
  intended effect of the change. Specify tolerances and expected directions;
  require no unexpected crash, retry, watchdog, or alert.
- Coverage gaps: evals, the child's own save/resume, and retries may fall outside
  the trial. Include these limitations in the decision.
- A schedule that leaves time for rollback during working hours, a submission
  owner, and the status issue/communication thread. Report kill, launch, first
  steps, go/no-go, and rollback transitions.

## 2. Land the launch record

Land the code change and finalized `trigger_hero.sh` values on main before
cutover. Use a pristine checkout at the verified SHA. The launcher records the
new run, retained checkpoint, and W&B fork boundary; its commands and recovery
semantics are documented in the linked launcher procedure.

Inventory downstream reports and trackers that pin the run ID or affected metric
keys. Update their selection as part of accepting the child.

## 3. Rehearse preflight and rollback

Use small scripts whose queries fail closed on errors or unrecognized output.
Dry-run every guard against the live cluster and obtain an independent review.

Verify:

- Deploy checkout is clean at fetched main; rollback checkout is clean at the
  old run's recorded SHA. Preserve its original launch command.
- Handoff `metadata.json` exists, records the expected step, and has the intended
  retention. Confirm layout and checkpoint lineage.
- Child checkpoint trees are empty for initial cutover. During later recovery,
  preserve them and verify the newest complete child checkpoint instead.
- No competing coordinator, gang, or hero pods exist apart from the old run.
  Iris, Kubernetes, object-store, and W&B credentials work.
- Launch refuses a live parent or child coordinator and a dirty checkout. One
  operator owns submission; capture its output and verify exactly one child
  coordinator afterward. Resolve an uncertain submission before retrying.
- Rollback cancels the child coordinator, confirms it is terminal, and launches
  the old revision and run ID as `IRIS_USER=marin`. Verify its intended resume
  checkpoint first. Never create another W&B fork for rollback.
- Object-storage headroom, checked again right before requesting the handoff.
  Hourly checkpoints are written to `hero-checkpoints` in US-EAST-08A (100 TiB
  quota); permanent checkpoints go to `marin-us-east-02a` in US-EAST-02A, whose
  quota all Marin work shares. At quota CoreWeave suspends writes for the whole
  zone, and the hero's next save hangs without an error (#8506, 2026-09-23).
  Require five checkpoints of free space in 08A (about 21 TB at 4.29 TB each):
  the old run's newest hourly checkpoint, a restore-smoke copy, the child's
  hourly checkpoint plus the next one being written (the older is pruned only
  after the newer commits), and one spare. Require two checkpoints plus a day of
  recent growth in 02A for the handoff and the child's next permanent
  checkpoint. Read usage and quota from the Finelog `storage.usage` namespace;
  the collector runs every few hours, so if the newest `collected_at` is more
  than an hour old, get current values with
  `python -m scripts.ops.storage.coreweave_usage --dry-run`.

Distinguish a successful query with no matching jobs from a failed query. Parse
Iris CSV headers and CRLF correctly; do not interpret `grep -c` exit status as a
query result. Match pods by task identity because Kubernetes names are sanitized
and truncated. Confirm restore from `Loading checkpoint from` and entry into the
training loop.

## 4. Execute the trial

1. Pass preflight. Verify the full matched control window is recorded and create
   the W&B fork once using the launcher procedure.
2. Cancel the old coordinator and confirm it is terminal and its training tasks
   have stopped before launching the child from the recorded SHA as
   `IRIS_USER=marin`.
3. Allow for restore, compilation, and data prefetch. Read startup and step
   watchdog limits from the resolved launch configuration; do not treat normal
   compilation as a hang.
4. Before the first child step, publish a comparison report for the 200 updates
   starting at `N` (steps `N` through `N+199`), plus wider context. Include loss,
   cross entropy, MFU, step time/tokens per second, drops, routing entropy, router
   losses, gradient norm, and peak memory. Share the URL.
5. Monitor fresh Finelog rows and execution identity, job state, task attempt IDs,
   step age, watchdogs, and errors. W&B inherited history and API lag do not prove
   new progress. Compare gate metrics against the old run at matching steps;
   emit updates on change.
6. Join the two runs by step and report mean/max loss deltas and all gate metrics.
   Judge restore correctness against the paired control or thresholds agreed
   before launch.

## 5. Decide

The trial gate overrides `manage-hero-run`'s ordinary retry policy. On a failed
gate, unexplained hang, or retry loop, cancel the child coordinator and roll back
without waiting for further attempts. Retries may restore either the handoff or
a newer child checkpoint; neither substitutes for a successful trial.

Go: leave the child running, publish the comparison and coverage gaps, update the
status issue and downstream reporting, and resume ordinary recovery policy.
The old run's leftover hourly checkpoints expire on their own within three
days, or 14 if its launcher predates the three-day TTL. Delete any restore-smoke
copy once it has served its purpose, deleting `metadata.json` first so a partial
directory never looks complete.

After rollback, compare the old run's first replayed steps with its earlier
trajectory. This checks trajectory consistency, not bitwise determinism. Its W&B
history may not advance until it passes the old counter, so verify progress in
Finelog. Update the status issue and file the failure with supporting evidence.

For collective stalls, use [silent-hang diagnostics](references/silent-hangs.md).
