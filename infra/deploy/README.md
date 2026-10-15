# Marin deploy

`marin-deploy` is the operator entry point for application rollouts and rollbacks. Each
service owns its deployment implementation; this package provides one command tree.

Pulumi-managed application services use the same rollout interface:

```bash
uv run --all-packages --extra deploy marin-deploy ducky rollout
uv run --all-packages --extra deploy marin-deploy grafana rollout
uv run --all-packages --extra deploy marin-deploy loom rollout
uv run --all-packages --extra deploy marin-deploy marina rollout
uv run --all-packages --extra deploy marin-deploy xprof rollout
```

Run these commands from the repository root with the Pulumi CLI installed. Each
command previews and applies the production stack from its project under `infra/`.
Pass `--yes` to skip Pulumi confirmation. Services that manage Cloudflare DNS load
their provider token from Secret Manager.

Repeat `--config KEY=VALUE` for an update-time configuration override. The command
applies overrides through a temporary copy of the stack config, so they do not modify
the checked-in `Pulumi.<stack>.yaml`. `marin-deploy` owns rollouts and rollbacks for
the registered application projects listed above under `infra/<service>/`. Run Pulumi
directly for shared infrastructure under `infra/pulumi/`, the bucket project under
`infra/buckets/`, SaaS resource projects, durable stack configuration, imports, and
application projects that are not registered here.

Changes on `main` deploy automatically through `ops-pulumi-rollout.yaml`. Dispatch
that workflow to redeploy the current `main` revision with the production identity
and GitHub-held service secrets. Local commands use operator credentials and
target the same production stacks. Services other than Marina also permit
deploying an unmerged checkout.

Marina production updates require a clean checkout at the current remote main;
use its main-branch workflow for a redeploy and local previews for branch work.
This also applies to direct `pulumi up` on the `marin-marina` stack. Every Marina
application uses `marina release` to publish its Postgres package independently;
a runtime rollout updates the kernel and installed dependencies.

Finelog Kubernetes deployments add revision rollback support:

```bash
uv run marin-deploy finelog rollout <cluster>
uv run marin-deploy finelog rollback <cluster>
```

Pass `--yes` to skip confirmation. A rollback selects the next older retained
Kubernetes revision by default; use `--to-revision N` to select an exact revision.
Finelog status, logs, secret synchronization, and GCE operations remain under
`uv run finelog deploy`.
