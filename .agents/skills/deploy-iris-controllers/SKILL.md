---
name: deploy-iris-controllers
description: Deploy, restart, or roll back the Iris controller for an explicitly named cluster set.
---

# Deploy Iris controllers

Process one cluster at a time, continue after passed gates, and stop the fleet
at the first blocked gate. The command reference is
`lib/iris/OPS.md` ("Controller Restart", "Rolling back a controller deploy",
"Controller Checkpoint Rollback").

`iris cluster controller restart` deploys the working tree, including staged
and unstaged changes; it does not deploy `main`. Update the checkout first.

## Rules

1. Never run `iris cluster restart` (no `controller`). It kills every worker and
   every job. `iris cluster controller restart` is the deploy — seconds of
   control-plane downtime, workers unaffected.
   It does not reconcile Pulumi-managed Kueue charts, ResourceFlavors,
   ClusterQueues, or CoreWeave NodePools.
2. An explicit rollout request approves its named cluster set and order. Use
   `plan` order when none is given; ask only when scope is ambiguous.
3. Report each passed gate and continue without more operator input.
4. Stop the fleet at the first blocked gate. After a
   restart, offer `--rollback` for that cluster. Do not continue to other
   clusters. Do not retry a restart to "see if it sticks".
5. Never modify the controller database. Never take an action outside the
   approved rollout scope.
6. Monitor each command through exit. If it yields, resume the same task and
   read new output. Do not pipe to `tail`; it hides earlier evidence.
7. Run every controller lifecycle command through Iris's `controller` extra:
   `uv run --package marin-iris --extra controller iris ...`. The base
   environment omits the Kubernetes client and can misreport every CoreWeave
   prerequisite as missing.

For a blocked gate, show the snapshot, verify samples, smoke result, and cause.
Ask for missing input before restart or rollback approval after restart. Silence
is not approval, and skipping ahead is never an option.

## Helper

`scripts/iris/rollout_controllers.py` covers the mechanical steps. It never
restarts a controller and never walks the cluster list on its own. The agent
keeps the approved scope and continues after each passed gate:

```bash
uv run python scripts/iris/rollout_controllers.py plan [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py preflight [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py snapshot --cluster NAME --out FILE
uv run python scripts/iris/rollout_controllers.py verify --cluster NAME --baseline FILE
uv run python scripts/iris/rollout_controllers.py smoke --cluster NAME
```

Write the snapshot files to the session scratchpad, one per cluster.

## Step 0 — Scope gate

Run `plan`. Without `--clusters` it orders `marin-dev`, `marin`, then CoreWeave
clusters by increasing capacity; with `--clusters` it preserves the operator's
order. Show `git log -1 --oneline` and the reported tree image tag. Ask only if
the cluster set is ambiguous.

Do not resolve the cluster list from memory. Cluster names come from `plan` or
from the operator.

## Step 1 — Tree and credential gate

Run `preflight` before any restart. Its first block prints the tree hash, branch,
the uncommitted file count, and how far HEAD is from `origin/main` (fetched
first, so "behind" is current). Each of these raises a `[WARN]`:

- a dirty tree — the deploy ships files that are in no commit
- a tree behind `origin/main` — the deploy ships stale code
- a tree ahead of `origin/main` — the deploy ships unmerged code

An untracked file is dirty but does **not** change the tree hash, while the Docker
build still copies it into the image — so two different images can carry one tag
and verify cannot tell them apart. Commit or remove the file before deploying.

On any warning, `preflight` exits non-zero and deploys nothing. Show the warning
and ask whether to deploy that exact tree. Only after confirmation, rerun with
`--accept-tree-state`; never pass it on your own initiative.

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

Do this loop for one cluster. After a passed final gate, start the next cluster.

1. **Snapshot.** `snapshot --cluster <name> --out <scratchpad>/<name>-before.json`
   records the controller's tree hash, rough job counts, and backend health. A
   worker-daemon backend reports healthy workers. A Kubernetes backend reports
   nodes that are ready and schedulable. The snapshot also records the tree this
   session would deploy. A job count printed with `+` hit the query cap and is a
   floor. If the controller is unreachable now, stop and diagnose.
2. **Snapshot gate.** Show the snapshot. State how many running jobs ride
   through the restart. State that the restart causes seconds of control-plane
   downtime. If the gate passes, restart without waiting for operator input.
3. **Restart.**
   `uv run --package marin-iris --extra controller iris --cluster=<name> cluster controller restart`.
   Add
   `--skip-checkpoint` only if the checkpoint step times out. On a Kubernetes dev
   cluster with amd64 nodes only, add `--image-platform linux/amd64`. To reuse a
   build, pass `--prebuilt-tag <tag>`; the command requires amd64 and arm64
   manifests for both images and pins their resolved manifest digests before it
   stops the old controller.
4. **Start the smoke.** `smoke --cluster <name>` submits one `echo hello world`
   job at interactive priority and waits for it. Start it **as a background task,
   right after the restart**, so the watch runs while it waits. A cluster with no
   idle worker must scale one up first, and that dominates: on `marin-dev` (0
   workers, TPU-backed) a smoke takes 15-20 minutes. In series it costs 20-25.
5. **Watch.** `verify --cluster <name> --baseline <name>-before.json` samples the
   controller every 30s for 5 minutes, then compares against the baseline. It
   fails on an unreachable controller, a tree hash that is not the one the
   baseline recorded, a changed backend health target, or health loss above the
   churn tolerance. The tolerance applies to each backend. It is one worker or
   node, or 5% of that backend's baseline healthy count, whichever is larger. A
   loss within this tolerance is a note in the gate evidence. Do not shorten the
   watch.
   When `--prebuilt-tag` points at an image built from another working tree, add
   `--expect-tree-hash <image-tree-hash>`.
6. **Collect the smoke.** Read the background task. If it is still waiting, keep
   waiting — the timeout is 30 minutes and a scale-up is the expected reason.
7. **Final gate.** Report the verify samples, the verdict, and the smoke job
   state. If the gate passes, continue to the next cluster without operator
   input. If the gate is blocked, stop the rollout.

The restart writes `rollout-record.json` under the cluster's
`storage.remote_state_dir` and health-checks the new controller, rolling back
automatically if it does not come up. `verify` proves which tree is running, so
read the record only when you need the rollback coordinates by hand.

## Kueue changes

Controller restarts apply Iris-owned runtime objects, including the controller,
node-agent DaemonSet, PriorityClasses, ConfigMap, Secrets, and LocalQueue. Kueue
and CoreWeave NodePools remain in the Pulumi stack.

When the approved rollout includes a Kueue or NodePool change, complete the
controller verify and smoke gates first. Then inspect and apply that cluster's
stack:

```bash
KUBECONFIG=~/.kube/coreweave-iris uv run pulumi preview \
  --cwd infra/pulumi --stack <cluster> --diff --non-interactive
KUBECONFIG=~/.kube/coreweave-iris uv run pulumi up \
  --cwd infra/pulumi --stack <cluster> --yes --skip-preview --non-interactive
```

Before `pulumi up`, inspect unadmitted Workloads and reject previews that replace
or delete a NodePool unless the operator explicitly approved that action. After
the update, verify the ClusterQueue is active, the intended ResourceFlavors
exist, and the expected nodes have the new labels. Run `smoke --cluster <name>`
again so the final gate exercises the reconciled Kueue configuration.

## Rollback

For a cluster that failed a gate while the controller is still reachable:

```bash
uv run --package marin-iris --extra controller iris \
  --cluster=<name> cluster controller restart --rollback
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
