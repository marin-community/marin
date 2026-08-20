# Marin deploy

`deploy` is the operator entry point for application rollouts and rollbacks. Each
service owns its deployment implementation; this package provides one command tree
that can grow as services move to the shared interface.

Finelog Kubernetes deployments are the first supported service:

```bash
uv run deploy finelog rollout <cluster>
uv run deploy finelog rollback <cluster>
```

Pass `--yes` to skip confirmation. A rollback selects the next older retained
Kubernetes revision by default; use `--to-revision N` to select an exact revision.
Finelog status, logs, secret synchronization, and GCE operations remain under
`uv run finelog deploy`.
