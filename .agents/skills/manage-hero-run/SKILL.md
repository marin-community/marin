---
name: manage-hero-run
description: Launch, monitor, hand off, resume, rollback, or babysit expensive Marin production runs, typically at least 1e22 model flops.
---

# Manage a hero run

Use for production-critical runs or runs around `1e22` model FLOPs (6ND is a
valid heuristic), and whenever requested. Arrange babysitting unless declined;
default cadence is 15 minutes. Never stop/restart/bounce an Iris cluster without
explicit permission. Assume the user is DRI unless told otherwise; identify and
contact another DRI before launch when applicable.

## Launch gate

Before launch, print and validate the exact secrets-scrubbed command, source SHA
and dirty-tree approval, runtime/source-bundle identity, hardware, W&B
id/display name and resume policy, final step from config/code, and `initialize_from`
checkpoint (numeric step and `metadata.json`). Block unclear lineage.

Storage must be explicit: mutable/development data under caller-owned
`users/<username>/...` below `MARIN_PREFIX`; immutable datasets or the same
caller-owned path; no mutable data under shared `iris/`. Use a region-local
`marin_temp_bucket(ttl_days=30, ..., source_prefix=<output root>)` for rolling
resume checkpoints (one by default, at most two unless the DRI records deeper
rollback, and never more than five), and one durable canonical export under the
user root. Raw traces,
failed-attempt markers, rendezvous, Ray sessions, and debug uploads use a
lifecycle-managed temp prefix; Ray spill uses `/tmp/skyrl-ray-spill` or another
node-local path. Block any durable `iris/` resolution. Record checkpoint size
estimate, retention count, projected resume bytes, and all destinations.

## Durable run record

Create/link a dedicated issue and append-only `.agents/logbooks/<run>.md` before
launch; commit/push it before launch and after each material update. Include
DRI, goal/stop criteria, issue/W&B/output roots, checkpoint/trace destinations,
final step, and each instance's command, SHA/dirty status, bundle, topology,
resume source, checkpoint policy, storage report, and babysitter state. Issue
updates cover launches, milestones, failures, relaunches, rollbacks, retention,
and escalations; issue status is at least every 24 hours and logbook entries at
each check cadence. Comments start with `🤖`.

Bootstrap issue/logbook with `run-research` conventions. If needed:

```bash
gh issue create --repo marin-community/marin --title "Hero run: <run name>" \
  --label experiment --label agent-generated --body-file /tmp/<run>-hero-issue.md
git add .agents/logbooks/<run>.md && git commit -m "<run>: start hero-run logbook" && git push
```

## Babysit, recover, resume

At each cadence check job health, monitor freshness, W&B run identity/state/
timestamp/step/loss, checkpoint completion, and throughput. Report `monitor
stale` separately from `run unhealthy`; alert on sustained throughput collapse
of 20–30%, checkpoint stagnation, >30-minute capacity wedges, numerical
instability, or loss >50% above trend for roughly 10 steps. Never relaunch
blindly after repeated same failures.

Direct relaunch with the same W&B id/output root for preemption, hardware,
controller, transient cloud, and ordinary recoverable failures. Use a new id and
root with `initialize_from` for W&B corruption or semantically meaningful code
changes. Relaunch only terminal recoverable failures. Select the newest complete
checkpoint by parsed numeric step, preserve lineage, and block incomplete,
rejected, wrong-region, or unvalidated checkpoints. During resume, alert when
loss differs by >0.002 before catch-up or >1% after warmup.
Fix logging or evaluation-callback bugs in place; ask the DRI before relaunching
changes to the model, optimizer, training loop, or data pipeline because they
change the trajectory.

Checkpoint cleanup requires an explicit confirmation after listing candidates;
preserve latest requested N, final, launch, recovery, and milestone anchors.
Capacity/scheduling issues do not authorize cluster mutation. Ask the DRI for
lineage, spend, capacity, or trajectory decisions.

## Seal and handoff

Verify successful terminal status, W&B final state, final checkpoint
`metadata.json`, lifecycle cleanup for temp artifacts, final metrics/step/W&B,
output and checkpoint paths, and caveats. Stop obsolete monitors. Create/push a
seal tag; if approved dirty changes were used, seal their exact state first.
Update the issue and logbook with the command, tag, metrics, and caveats.

References: `babysit-job`, `run-research`, and `change-grug`.
