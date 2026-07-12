---
name: manage-hero-run
description: Launch, monitor, hand off, resume, rollback, or babysit expensive Marin production. Typically >=1e22 model flops.
---

# Manage Hero Run

Use this skill for expensive or production-critical runs where a bad launch, stale worktree, weak handoff, or confused resume can waste significant compute. Optimize first for deliberate launch, then reliable babysitting, then disciplined resume/recovery.

As a rule of thumb, use this skill for runs that are expected to run for 1e22 model flops or more. (6ND heuristic is fine.) You should also use this skill if asked.

## Operating Model

- If the user asks to launch a run, arrange babysitting unless they explicitly decline.
- Use a 15 minute default check cadence for babysitting unless the run's failure mode requires tighter monitoring.
- Never stop, restart, or bounce an Iris cluster without explicit user permission.
- If something needs human judgment or authorization, attempt to contact the DRI, usually the user in the chat, through available channels such as GitHub issue comments, Discord, email, or Slack.

## Launch

Before launching, print and validate:

- Exact command line, with secrets scrubbed.
- Source git SHA for this launched instance. Avoid dirty trees unless the user explicitly wants one; if dirty is approved, log the diff or patch identity.
- Pinned output root. Hero runs must use an explicit override output path; do not rely on auto-derived or ephemeral output locations.
- W&B id/name and resume policy.
- Checkpoint retention policy. Hero runs must explicitly choose permanent retention and rollback depth before launch. Suggest permanent checkpoints on the order of every 1e5 steps and temporary checkpoints every 15 minutes, keeping the previous 10 checkpoints for recovery and rollback robustness.
- `initialize_from`, parsed numeric checkpoint step, and `metadata.json` presence when starting from a checkpoint.
- Final training step resolved from the launched config/code, not from progress-bar display text.
- Runtime package, source bundle, or container identity when it differs from the local git SHA.
- A DRI. Assume the user is the DRI unless they explicitly say otherwise; if the user is not the DRI, identify and contact the DRI before launch.

If any value is inferred, label it as inferred. If code lineage, checkpoint policy, or output identity is unclear, pause before launching.

## Run Record

Maintain research-style run records (cf the run-research skill):

- Create or use a dedicated GitHub issue for important status, decisions, and escalation. Apply the appropriate experiment/run label when available.
- Keep an append-only logbook at `.agents/logbooks/<run>.md` unless the user chooses another path.
- Commit and push the logbook every time it is materially updated; git history is the durability layer, not the local Codex worktree.
- Link the issue and logbook both ways.
- Record every launched instance: timestamp, DRI, exact command line with secrets scrubbed, git SHA, dirty-tree status, source bundle identity, hardware/topology, W&B run id/display name, output root, `initialize_from`, final step, checkpoint policy, and babysitter state.
- Post issue updates for significant milestones, failures, relaunches, rollbacks, retention changes, and escalations. Keep dense logs in the logbook and concise status in the issue.

Bootstrap the issue and logbook before launch:

1. Pick a stable run slug, e.g. `grug-moe-1e23-ep8`.
2. Create `.agents/logbooks/<run>.md` from the template below.
3. If no dedicated issue exists, create one:

```bash
gh issue create \
  --repo marin-community/marin \
  --title "Hero run: <run name>" \
  --label experiment \
  --label agent-generated \
  --body-file /tmp/<run>-hero-issue.md
```

4. Put the issue URL in the logbook, and put the logbook path in the issue body.
5. Commit and push the initial logbook before launch:

```bash
git add .agents/logbooks/<run>.md
git commit -m "<run>: start hero-run logbook"
git push
```

After launch, commit and push the logbook every time it is materially updated. Use concise commit messages such as `<run>: log launch`, `<run>: log relaunch`, or `<run>: log final seal`. Issue comments should start with `🤖` unless the exact text was explicitly approved by the user.

Use this compact logbook shape:

```md
# <Run Name>: Hero Run Logbook

## Run Contract
- DRI:
- Goal:
- Stop/escalation criteria:
- Issue:
- W&B:
- Output root:
- Final step:

## Launched Instances
### YYYY-MM-DD HH:MM TZ - <instance id>
- Command: `<scrubbed command>`
- Git SHA:
- Dirty tree: no | yes, approved by <person>, patch identity <link/hash>
- Source bundle/container:
- Hardware/topology:
- `initialize_from`:
- Final step:
- Checkpoint policy:
- Babysitter/check cadence:

## Event Log
### YYYY-MM-DD HH:MM TZ - <event>
- Status:
- Evidence:
- Decision:
- Next:
```


Issue updates should happen every 24 hours unless an interesting event (crash, spike, etc) occurs. Logbook updates should happen at every check cadence and whenever an interesting event occurs. The logbook is the detailed record of what happened, while the issue is the high-level status for observers.

## Babysitting

- Check at the agreed cadence from the operating model.
- Use the babysitting workflow for job health, monitor freshness, W&B progress, checkpoint completion, loss/metric sanity, and completion checks.
- Validate the active W&B run id, display name, state, `_timestamp`, `global_step`, and key losses against the intended run id/display name recorded at launch. Do not rely only on a saved W&B URL.
- Prefer narrow Iris/orchestrator status queries, SQL checks, and targeted job inspection over broad blocking status calls.
- Classify stale diagnostic jobs separately from the current production child job.
- Escalate to the DRI when the next action requires judgment, spend/capacity tradeoffs, lineage choice, cluster intervention, or accepting a dirty/unverified state.

## What Can Go Wrong

- **Midrun crash from hardware, preemption, controller failure, or other low-level issue:** relaunch directly with the same run id/output root and see whether it makes progress past the failure. Notify the DRI immediately; escalate if the same problem repeats or progress remains blocked.
- **Code bug.** Code bugs that do not impact the training trajectory in a substantive way should just be fixed. Alert the DRI and relaunch the run. Await input if code change is likely to lead to "interesting" differences.
For instance, it's ok to fix a logging bug or misconfiguration of evaluation callbacks that led to a crash. Ask for input if the bug was in the model definition, training loop, optimizer, or data pipeline, since those could lead to a different training trajectory and require a new run with a new W&B id (using initialize_from)
- **Wrong checkpoint selected:** incomplete checkpoint, lexicographic sort bug, newer rejected lineage, wrong temporary/permanent root, or wrong region. Block launch if checkpoint step, metadata, output lineage, or source run do not match the run record.
- **Wrong code lineage:** stale worktree, dirty tree, wrong branch, unpushed commit, source bundle mismatch, or container built from a different SHA. Block launch unless explicitly approved and logged. Files that won't impact the training run (e.g. log files, markdown files, test files, unrelated experiment files) should not block a launch.
- **Wrong output path:** old output root, auto-derived path drift, wrong region, or output path mismatch between launcher and babysitter. Block launch if output root is not pinned and printed before launch.
- **W&B identity drift:** active run id differs from intended run, display name is reused ambiguously, resume policy is wrong, or state files contain a stale URL. Alert immediately because metrics can look plausible while attached to the wrong lineage.
- **Silent monitor failure:** job may be healthy while the monitor is dead, local disk is full, or screen/process is alive without fresh state updates. Report `monitor stale` separately from `run unhealthy`.
- **Throughput collapse:** run is alive but tokens/sec drops materially due to degraded hardware, input stalls, checkpoint stalls, compile churn, or retry loops. Alert if sustained throughput is more than `20-30%` (relative) below baseline for multiple cadences. Periodic dips are expected so check for sustained collapse, not single dips.
- **Checkpoint not advancing:** training steps move but complete checkpoints do not appear, `metadata.json` is missing, writes are stuck, or cleanup threatens rollback coverage. Alert before the rollback window collapses.
- **Capacity or scheduling wedge:** job remains pending for more than 30 min, partially allocated, wrong TPU type/slice count is requested, or workers never co-schedule. Notify, but do not mutate clusters without approval.
- **Repeated recoverable failures:** one crash may be preemption; repeated same-step or same-window failures suggest a deterministic bug. Escalate instead of blindly relaunching.
- **Numerical instability:** NaNs/Infs, grad norm explosion, router collapse, sudden z-loss/router metric changes, or optimizer instability. Alert immediately; do not relaunch as if it were infrastructure. Fast changes are expected during warmup, but sustained instability after warmup is a concern.
- **Config drift on relaunch:** batch size, max steps, optimizer, checkpoint interval, seed, mesh, precision, data config, or code flags differ unintentionally. Diff launched config against the prior instance before relaunch.
- **Resume loss mismatch:** Levanter is generally bitwise identical on TPU. Resumes and GPU runs can sometimes differ slightly, but should stay very close. During catch-up, alert if loss differs from the pre-resume lineage by more than `0.002`; after post-resume warmup, alert if loss differs by more than `1%`.
- **Sustained loss spike:** alert if loss is more than `50%` above the expected trend for roughly 10 or more consecutive steps.
- **Final-step misunderstanding:** progress bars may round or display a nominal max while config has extra steps. Compute final step from config/code and use that for ETA and completion.
- **Benign-looking success with missing artifacts:** orchestrator says success but final checkpoint, W&B summary, logbook update, or seal tag is missing. Do not seal until final artifacts are verified.

## Resume And Recovery

Many failures can be recoverable just by relaunching using the same id. These include hardware failures, preemptions, transient cloud issues, and some classes of code bugs. Use the launch workflow for relaunches, but with special attention to checkpoint lineage and resume policy. If the lineage is intact and the resume policy is `allow`, prefer direct relaunch with the same W&B id and output root. If the lineage is compromised or the resume policy is `never`, use a new W&B id and output root, and treat it as a new run for record-keeping purposes.

- Default to direct relaunch with the same run id, W&B identity, and pinned output root. Use this for controller job crashes, preemptions, hardware/low-level failures, and ordinary recoverable interruptions so the existing recovery mechanism keeps the run going.

### Launching with a new run id

- Use a new run id and W&B id only when the old lineage is unsafe or semantically different, such as W&B corruption or a nontrivial code change. Nontrivial code changes should have a new W&B id. Document the reason, old and new identities, source checkpoint, output root, and code SHA in the issue and logbook.
- Use `initialize_from` to have training pick up from a specific prior checkpoint.
- If the user does not specify a checkpoint to use for a resume, select the newest "complete" one. Complete checkpoints have `metadata.json`. If no complete checkpoints are available, escalate to the DRI instead of guessing. If the user specifies a checkpoint that does not have `metadata.json`, block the launch and escalate instead of guessing. Do not use incomplete checkpoints for resume or relaunch.
- Sort by parsed numeric step, not lexicographic path order.
- Do not advance to a checkpoint from a rejected or unvalidated lineage just because it is newer.
- Relaunch only on terminal recoverable failure, and record why the failure was recoverable.

## Retention

- Ordinary runs should use rolling temporary checkpoint behavior for preemption recovery and keep only the final checkpoint permanently.
- Hero runs must explicitly choose retention and rollback depth before launch.
- Never rely on permanent retention alone to protect an explicit rollback source; treat launch lineage as state.
- For any checkpoint cleanup, list deletion candidates first and get explicit user confirmation. Preserve the latest requested N, the final checkpoint, the launch checkpoint, any recovery source, and requested milestone anchors.

## Seal

When a hero run finishes or reaches a handoff milestone:

- Verify terminal orchestrator status is successful.
- Verify W&B is finished or has the expected final state and metrics.
- Verify the final checkpoint has `metadata.json`.
- Capture final metrics, final step, W&B run id/display name, output root, final checkpoint path, and any caveats.
- Stop or delete heartbeat/monitor automations that are no longer needed.
- If approved dirty-tree changes were used, create a seal commit and tag immediately so the actual operational state is recoverable.
- Create and push a seal tag for the completed run or milestone.
- Update the GitHub issue and logbook with the W&B run, checkpoint, commit/tag, final metrics, launch command, and caveats.


## References

- change-grug skill
- run-research skill
- babysit-job skill
