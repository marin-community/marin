---
name: deploy-iris-controllers
description: Deploy the Iris controller to one cluster or across the fleet, with automatic progress through passed gates. Use when restarting, rolling out, or rolling back a controller.
---

# Deploy Iris controllers

Follow the matching `lib/iris/OPS.md` procedures. A controller restart deploys
the current working tree (including staged/unstaged files, hashed by
`get_git_sha()`), not `main`; update the checkout first. Use
`iris cluster controller restart`, never `iris cluster restart`, which kills
workers and jobs. A controller restart does not reconcile Pulumi Kueue,
ResourceFlavor, ClusterQueue, or NodePool objects.

An explicit cluster list is approval for that set and order. Otherwise use the
order from `plan`; ask only when scope is absent/ambiguous. Process one cluster
at a time, continue automatically after passed gates, stop the whole rollout at
the first blocked gate, and never retry a restart to see whether it sticks. Do
not modify the controller DB or act outside the approved scope. Continuously
read command output; do not pipe a running command to `tail`.

## Plan and preflight

The helper performs checks but never restarts or chooses clusters on its own:

```bash
uv run python scripts/iris/rollout_controllers.py plan [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py preflight [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py snapshot --cluster NAME --out FILE
uv run python scripts/iris/rollout_controllers.py verify --cluster NAME --baseline FILE
uv run python scripts/iris/rollout_controllers.py smoke --cluster NAME
```

Run `plan`, print its tree tag and `git log -1 --oneline`, then run `preflight`
before any restart. It fetches `origin/main` and warns on dirty, behind, or ahead
trees; untracked files are dirty and may enter the image without changing its
hash. Nonzero warnings deploy nothing. Ask whether to deploy that exact tree;
only after confirmation rerun with `--accept-tree-state`, never autonomously.
Preflight also verifies config-derived env/S3 keys, kube context, signing-key
references, `git`/`gcloud`/`kubectl`, Docker, and credentials. Stop on any FAIL;
do not mint credentials or add SSH/OS Login keys. GCE clusters also require
working `gcloud compute ssh` as the local user; a failed SSH leg leaves the
controller untouched, so hand off to a session that already has SSH.

## Per-cluster gates

1. Snapshot to scratch, which records tree hash, job counts, backend health, and
   the tree to deploy. A `+` job count is a floor. Stop if unreachable.
2. Show running-job count, backend evidence, and seconds of control-plane
   downtime. On approval implied by the rollout scope, continue without asking.
3. Restart:

   ```bash
   iris --cluster=<name> cluster controller restart
   ```

   Use `--skip-checkpoint` only for a checkpoint timeout; `--image-platform
   linux/amd64` for amd64-only Kubernetes dev nodes; `--prebuilt-tag <tag>` only
   with both amd64/arm64 manifests and resolved digests.
4. Immediately start `smoke --cluster <name>` as a background task. It submits
   `echo hello world`; a scale-up can take 15–20 minutes and the timeout is 30.
5. Run `verify --cluster <name> --baseline <file>` for the full 5-minute,
   30-second watch. Do not shorten it. It checks reachability, tree hash,
   backend health, and per-backend worker/node loss (one or 5% of baseline,
   whichever is larger). Add `--expect-tree-hash` when using an external image.
6. Collect the smoke result, continuing to wait if it is still scaling.

If any final gate blocks, stop and report evidence. Restart writes
`rollout-record.json` and rolls back automatically if the controller fails to
come up. For a reachable failed cluster, rollback requires approval:

```bash
iris --cluster=<name> cluster controller restart --rollback
```

This restores the old image and checkpoint; forward-only migrations mean jobs
created after that checkpoint are lost. For an unreachable/wedged controller or
first deploy, use the on-VM `Controller Checkpoint Rollback` in `OPS.md`; never
recreate the VM.

## Pulumi and closure

If the approved rollout includes Kueue/NodePool changes, finish controller
verify/smoke first, then inspect the stack. Reject any unexpected NodePool
replace/delete unless explicitly approved:

```bash
KUBECONFIG=~/.kube/coreweave-iris uv run pulumi preview \
  --cwd infra/pulumi --stack <cluster> --diff --non-interactive
KUBECONFIG=~/.kube/coreweave-iris uv run pulumi up \
  --cwd infra/pulumi --stack <cluster> --yes --skip-preview --non-interactive
```

Verify ClusterQueue, ResourceFlavors, node labels, and run smoke again. After a
fleet rollout, record tree hash, clusters deployed, and clusters left behind in
one `echo-log` entry. `serve --dry-run` is local inspection, not a gate.
