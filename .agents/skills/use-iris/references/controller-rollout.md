# Deploy Iris controllers

An explicit rollout request authorizes only its named cluster set. Process one cluster at a time and stop the fleet at the first failed gate. Read `lib/iris/OPS.md` under `Controller Restart` and rollback before acting.

Never run `iris cluster restart`. A controller restart deploys the current working tree, including uncommitted files, and causes brief control-plane downtime without stopping workers.

Use the helper for every gate:

```bash
uv run python scripts/iris/rollout_controllers.py plan [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py preflight [--clusters a,b]
uv run python scripts/iris/rollout_controllers.py snapshot --cluster <name> --out <file>
uv run python scripts/iris/rollout_controllers.py verify --cluster <name> --baseline <file>
uv run python scripts/iris/rollout_controllers.py smoke --cluster <name>
```

`preflight` checks the tree, credentials, Docker, SSH, Kubernetes context, and injected secrets. Stop on every failure. A dirty/ahead/behind tree requires approval for that exact tree before this rerun; never add the flag automatically:

```bash
uv run python scripts/iris/rollout_controllers.py preflight \
  --clusters a,b --accept-tree-state
```

For each cluster: snapshot, report the running-job count and downtime, restart, start the smoke, run the full five-minute verify, collect the smoke, then continue only after all gates pass.

```bash
uv run --package marin-iris --extra controller iris \
  --cluster=<name> cluster controller restart
```

On a failed post-restart gate, stop the fleet and offer rollback. Rollback restores the prior image and pre-deploy checkpoint and can lose jobs created after that checkpoint, so it requires fresh approval:

```bash
uv run --package marin-iris --extra controller iris \
  --cluster=<name> cluster controller restart --rollback
```

Kueue and CoreWeave NodePool changes are Pulumi-managed and are not applied by a controller restart. Never modify the controller database, mint credentials, add SSH keys, shorten verification, or retry a failed restart speculatively.
