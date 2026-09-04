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
and GitHub-held service secrets. Run the command locally only to deploy an unmerged
checkout or investigate a rollout with operator credentials; it targets the same
production stack.

Finelog deployments dispatch through the backend in the named configuration:

```bash
uv run marin-deploy finelog rollout <name>
uv run marin-deploy finelog rollback <name>
uv run marin-deploy finelog status <name>
```

Kubernetes rollouts capture the active Deployment revision, apply the matching
Pulumi stack, and restore the captured ReplicaSet if update or ingest verification
fails. Pass `--yes` to skip confirmation. A rollback selects the next older retained
revision by default; use `--to-revision N` to select an exact revision.

GCE rollouts build and digest-pin the configured image, record the running digest
under `~/.cache/finelog/deploy-state/`, and activate the candidate over SSH. The
startup script becomes reboot-time metadata only after the candidate boot succeeds.
A failed candidate restores and verifies the recorded digest. Use `--no-build` to
deploy the current registry tag, `--force` to reapply a matching digest, and
`rollback --to <image@sha256:...>` to select an explicit target.

Finelog secret synchronization, logs, and one-time GCE creation or deletion remain
under `uv run finelog deploy`.

Iris controller deployments use one Pulumi stack per cluster:

```bash
uv run marin-deploy iris rollout <cluster>
uv run marin-deploy iris rollback <cluster>
```

The rollout takes a controller checkpoint before building images. Pulumi then
activates the pinned images on the existing GCE VM or Kubernetes controller. A
failed activation restores the previous controller image with that checkpoint
through the same stack and still exits with failure. Pulumi never receives
resolved controller secrets and does not own controller deletion.

GCE deployments share the typed target and noninteractive SSH runner in
`marin_deploy.gce`. Finelog persists only a startup script that completed its own
health gate. Iris persists before activation so an interrupted controller rollout
can resume its image/checkpoint transaction, and Loom leaves startup metadata under
Pulumi ownership. Service-specific health checks and rollback state remain with each
deployment.
