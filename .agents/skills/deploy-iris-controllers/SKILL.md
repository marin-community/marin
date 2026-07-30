---
name: deploy-iris-controllers
description: Deploy the Iris controller to one cluster or across the fleet, with a human gate at every step. Use when restarting, rolling out, or rolling back a controller.
---

# Skill: Deploy Iris controllers

Roll the controller in your working tree out to one cluster or to the whole
fleet. One cluster at a time, one human gate per step. The reference for every
command is `lib/iris/OPS.md` ("Controller Restart", "Rolling back a controller
deploy", "Controller Checkpoint Rollback").

**A restart deploys your working tree, not `main`.** `iris cluster controller
restart` builds images from HEAD plus staged and unstaged changes
(`get_git_sha()` hashes tree content) and pins the deploy to that hash. Update
the checkout first: a stale checkout ships stale code, which once cost ~5
red-canary days ([incident record](https://echo.oa.dev/wiki/14)).

## Rules

1. Never run `iris cluster restart` (no `controller`). It kills every worker and
   every job. `iris cluster controller restart` is the deploy — seconds of
   control-plane downtime, workers unaffected.
2. Stop at every gate. Present the evidence, then wait for the operator. Do not
   deploy the next cluster because the previous one passed.
3. Stop the whole rollout on the first failed gate. Report the failure and offer
   `--rollback` for that cluster. Do not continue to other clusters and do not
   retry a restart to "see if it sticks".
4. Never modify the controller database, and never take an action the operator
   did not approve at a gate.

## How to gate

Put every gate in front of the operator with `AskUserQuestion`. Show the evidence
in your message — the snapshot, the verify samples, the smoke result — because the
operator answers from your text, not from the raw command output. Never treat
silence, a previous approval, or a passing gate on another cluster as approval.

The options depend on what the evidence says:

- **Everything passed:** approve this cluster, stop the rollout here, or skip this
  cluster and hold at the next gate.
- **Anything failed:** stop, or roll this cluster back. Never offer to skip ahead
  after a failure — rule 3 ends the rollout there.

## Helper

`scripts/iris/rollout_controllers.py` covers the mechanical steps. It never
restarts a controller and never walks the cluster list on its own, so the gates
stay with the operator:

```bash
uv run python scripts/iris/rollout_controllers.py plan [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py preflight [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py snapshot --cluster NAME --out FILE
uv run python scripts/iris/rollout_controllers.py verify --cluster NAME --baseline FILE
uv run python scripts/iris/rollout_controllers.py smoke --cluster NAME
```

Write the snapshot files to the session scratchpad, one per cluster.

## Step 0 — Scope gate

1. Run `plan`. Without `--clusters` it prints the default order: `marin-dev`,
   `marin`, then the CoreWeave clusters smallest capacity first. With
   `--clusters` it uses the operator's list verbatim, in that order.
2. Print `git log -1 --oneline` and the tree image tag that `plan` reports.
   `preflight` reports the tree state in full at the next step.
3. Ask the operator to confirm the cluster set and the order.

Do not resolve the cluster list from memory. Cluster names come from `plan` or
from the operator.

## Step 1 — Tree and credential gate

Run `preflight` for the selected clusters *before* any restart. It reports two
things.

**What this tree would ship.** The first block prints the tree hash, the branch,
the uncommitted file count, and how far HEAD is from `origin/main` (fetched
first, so "behind" is current). Each of these raises a `[WARN]`:

- a dirty tree — the deploy ships files that are in no commit
- a tree behind `origin/main` — the deploy ships stale code, the failure mode
  that once cost ~5 red-canary days
- a tree ahead of `origin/main` — the deploy ships unmerged code

An untracked file is dirty but does **not** change the tree hash, while the Docker
build still copies it into the image — so two different images can carry one tag
and verify cannot tell them apart. Commit or remove the file before deploying.

On any warning `preflight` **exits non-zero and deploys nothing**. Show the
warnings to the operator and ask whether to deploy this exact tree. A dirty tree
is normal for a controller fix under test and reckless for a routine fleet
rollout, so let the operator decide rather than guessing. Only after they confirm,
re-run with `--accept-tree-state`. Never pass that flag on your own initiative.

**What the deploy reads from this session.** Requirements are derived from each
cluster config, so they cannot drift: `defaults.inject_env` names, the CoreWeave
S3 keys, the kube-context, the signing-key references, `git` / `gcloud` /
`kubectl` on PATH, and `docker info` plus `docker buildx version`.

If anything reports FAIL, **ask the operator for it and stop**. Do not invent a
value, mint credentials, or skip the cluster. Each FAIL line names what it needs
and why. Two causes are easy to misread:

- A `kube-context` FAIL often means an unrelated exported `KUBECONFIG`, which
  replaces the configured path exactly as the deploy resolves it.
- A signing-key FAIL on a `gcp-secret://` reference means the session lacks GCP
  credentials (`gcloud auth application-default login`).

A defined context proves the configuration, not live credentials. The snapshot in
step 2 reaches the controller through it, so expired CoreWeave credentials surface
there — still before any restart.

GCE clusters (`marin`, `marin-dev`) also need working `gcloud compute ssh` as
your local username. A failed SSH leg aborts the restart safely — the running
controller is untouched — but an agent session must not add SSH or OS Login
keys. Hand those clusters to a session that already has SSH.

## Step 2 — Per cluster

Do this loop for one cluster, then gate. Repeat for the next cluster.

1. **Snapshot.** `snapshot --cluster <name> --out <scratchpad>/<name>-before.json`
   records the controller's tree hash, its workers, rough job counts, and the tree
   this session would deploy. A count printed with `+` hit the query cap and is a
   floor. If the controller is unreachable now, stop and diagnose.
2. **Gate.** Show the snapshot. State how many running jobs ride through the
   restart, and that the restart takes seconds of control-plane downtime.
3. **Restart.** `iris --cluster=<name> cluster controller restart`. Add
   `--skip-checkpoint` only if the checkpoint step times out. On a Kubernetes dev
   cluster with amd64 nodes only, add `--image-platform linux/amd64`.
4. **Start the smoke.** `smoke --cluster <name>` submits one `echo hello world`
   job at interactive priority and waits for it. Start it **as a background task,
   right after the restart**, so the watch runs while it waits. A cluster with no
   idle worker must scale one up first, and that dominates: on `marin-dev` (0
   workers, TPU-backed) a smoke takes 15-20 minutes. In series it costs 20-25.
5. **Watch.** `verify --cluster <name> --baseline <name>-before.json` samples the
   controller every 30s for 5 minutes, then compares against the baseline. It
   fails on an unreachable controller, a tree hash that is not the one the
   baseline recorded, or lost healthy workers. Do not shorten the watch.
6. **Collect the smoke.** Read the background task. If it is still waiting, keep
   waiting — the timeout is 30 minutes and a scale-up is the expected reason.
7. **Gate.** Report the verify samples, the verdict, and the smoke job state.

The restart writes `rollout-record.json` under the cluster's
`storage.remote_state_dir` and health-checks the new controller, rolling back
automatically if it does not come up. `verify` proves which tree is running, so
read the record only when you need the rollback coordinates by hand.

## Rollback

For a cluster that failed a gate while the controller is still reachable:

```bash
iris --cluster=<name> cluster controller restart --rollback
```

This restores the previous image **and** its pre-deploy checkpoint, because
migrations run forward-only and some are destructive. Jobs created after that
checkpoint are lost. Get the operator's approval first.

For a wedged or unreachable controller, or a first deploy with no prior rollout
record, use the on-VM procedure in `lib/iris/OPS.md` ("Controller Checkpoint
Rollback"). Never recreate the controller VM.

## Notes

- `iris cluster controller serve --dry-run` is not a pre-restart gate. It boots a
  local controller that serves until killed, for interactive state inspection.
  The unit suite and CI on the tree are the gate.
- CoreWeave controllers restart over the Kubernetes API and need no SSH. The
  kubeconfig is `~/.kube/coreweave-iris`, with the context pinned per cluster
  config.
- After a fleet rollout, add one `echo-log` entry: the tree hash, the clusters
  that took it, and any cluster you left behind.
