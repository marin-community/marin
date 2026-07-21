# Marin Ops Workflow

This directory contains the vertical slice behind `ops.oa.dev`: a read-only Grafana PostgreSQL poller, durable alert and agent state, a globally serialized ACP queue, a server-side Loom adapter, and a Vue case dashboard.

Grafana remains canonical. The backend polls firing `alert_instance` rows every minute and joins `alert_rule` metadata. Warning notifications are muted; error and critical notifications continue to Slack and email. All three severities can wake the read-only ops agent.

The full architecture, edge cases, IAP policy, and rollout order are in [`../../.agents/projects/ops_workflow/design.md`](../../.agents/projects/ops_workflow/design.md). The exact adapter and database contract is in [`../../.agents/projects/ops_workflow/spec.md`](../../.agents/projects/ops_workflow/spec.md).

## Local spike

Start the ops database and the Grafana-shaped fixture database:

```bash
docker compose -f infra/ops/compose.yaml up -d --wait postgres grafana-postgres
uv run --project infra/ops ops-workflow migrate \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops
npm --prefix infra/ops/dashboard ci
npm --prefix infra/ops/dashboard run build
```

Run the dashboard with the deterministic agent stub:

```bash
uv run --project infra/ops ops-workflow serve \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops \
  --grafana-database-url postgresql://grafana_reader:local-grafana-read@127.0.0.1:55433/grafana \
  --grafana-poll-interval 60 \
  --agent-mode stub \
  --static-dir infra/ops/dashboard/dist
```

Open <http://127.0.0.1:8088>. The first poll creates one case for the two `DNSConfigForming` instances. The case moves through `pending → investigating → waiting_human` and exposes follow-up, one-off question, Loom-link, and archive flows.

Run Playwright against the service:

```bash
OPS_BASE_URL=http://127.0.0.1:8088 npm --prefix infra/ops/dashboard run test:e2e
```

Use a local Loom ACP server instead of the stub with:

```bash
uv run --project infra/ops ops-workflow serve \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops \
  --grafana-database-url postgresql://grafana_reader:local-grafana-read@127.0.0.1:55433/grafana \
  --agent-mode loom \
  --loom-api-url "$WEAVER_API" \
  --loom-token "$LOOM_TOKEN" \
  --repo-root /path/visible/to/loom \
  --loom-base weaver/ops-workflow \
  --static-dir infra/ops/dashboard/dist
```

The base must contain `.agents/skills/ops-expert/SKILL.md`. The adapter creates ACP sessions in plan mode, and the prompt prohibits production mutation and repository changes.

## Production access

The IAP allowlist is non-secret policy checked into [`Pulumi.marin-ops.yaml`](Pulumi.marin-ops.yaml):

```yaml
config:
  marin-ops:viewers:
    - "*@openathena.ai"
    - ops@openathena.ai
```

The wildcard becomes `domain:openathena.ai`; the bare address becomes `user:ops@openathena.ai`. If the address becomes a Google Group, use `group:ops@openathena.ai`. Run `pulumi preview` after a membership change. Each entry owns one `roles/iap.httpsResourceAccessor` grant, so Pulumi changes only the added or removed member.

The service has no alert receiver. It reaches the shared Cloud SQL instance through the connector socket and uses separate credentials:

- `ops_app` reads and writes the `ops` workflow database;
- `ops_grafana_reader` can select only `grafana.public.alert_instance` and `grafana.public.alert_rule`.

The separate `ops_migrator` identity owns schema objects but is never mounted into the service. Create all three logins and their custom database roles using [`../cloudsql/README.md`](../cloudsql/README.md). The Grafana owner and migrator passwords are never mounted into the ops service.

Pulumi maps the IAP-protected service to `ops.oa.dev` with a DNS-only Cloudflare CNAME. The Cloudflare API token is a deployment credential, not a runtime secret.

## Verification

```bash
uv run --project infra/ops pytest infra/ops/tests
uv run --with pyrefly pyrefly check infra/ops/src infra/ops/tests
uv run --project infra/grafana pytest infra/grafana/tests/test_provisioning.py
npm --prefix infra/ops/dashboard run build:check
```

Repository lifecycle tests need a dedicated database whose name ends in `_test`:

```bash
OPS_TEST_DATABASE_URL=postgresql://ops:local-ops@127.0.0.1:55432/ops_test \
  uv run --project infra/ops pytest infra/ops/tests/test_repository_postgres.py
```

Production rollout applies migrations before starting the restricted runtime, then verifies that `last_poll_at` advances on an empty snapshot before enabling the warning mute policy.

The checked-in first-rollout `agent_mode` is `stub`. Change it to `loom` only after a managed Loom API is reachable from Cloud Run and `marin-ops-loom-token` has a secret version; then add the explicit Loom URL, repository root, and base to stack configuration.
