# Marin deploy

`deploy` is the operator entry point for application rollouts and rollbacks. Each
service owns its deployment implementation; this package provides one command tree.

Pulumi-managed application services use the same rollout interface:

```bash
uv sync --all-packages --extra deploy
uv run deploy ducky rollout
uv run deploy echo rollout
uv run deploy evaldash rollout
uv run deploy grafana rollout
uv run deploy loom rollout
uv run deploy xprof rollout
```

Run these commands from the repository root with the Pulumi CLI installed. Each
command applies the production stack from its project under `infra/`. Pass
`--yes` to skip Pulumi confirmation and repeat `--config KEY=VALUE` for update-time
configuration overrides. Continue to use the Pulumi CLI directly for previews,
stack configuration, imports, and infrastructure or SaaS projects.

Finelog Kubernetes deployments add revision rollback support:

```bash
uv run deploy finelog rollout <cluster>
uv run deploy finelog rollback <cluster>
```

Pass `--yes` to skip confirmation. A rollback selects the next older retained
Kubernetes revision by default; use `--to-revision N` to select an exact revision.
Finelog status, logs, secret synchronization, and GCE operations remain under
`uv run finelog deploy`.
