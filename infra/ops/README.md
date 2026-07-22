# Marin Ops Workflow

This directory contains the vertical slice behind `ops.oa.dev`: a read-only Grafana Alertmanager API poller, durable alert and agent state, a globally serialized ACP queue, a server-side Loom adapter, and a Vue case dashboard.

Grafana remains canonical. The backend polls Grafana's active-alert API every minute. Warning notifications are muted; error and critical notifications continue to Slack and email. All three severities can wake the read-only ops agent.

## Local development

Start the ops database and the static Grafana API fixture:

```bash
docker compose -f infra/ops/compose.yaml up -d --wait postgres grafana-api
uv run --project infra/ops ops-workflow migrate \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops
npm --prefix infra/ops/dashboard ci
npm --prefix infra/ops/dashboard run build
```

Run the dashboard with the deterministic agent stub:

```bash
uv run --project infra/ops ops-workflow serve \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops \
  --grafana-api-url http://127.0.0.1:55433 \
  --grafana-api-token local-fixture-token \
  --grafana-poll-interval 60 \
  --agent-mode stub \
  --static-dir infra/ops/dashboard/dist
```

Open <http://127.0.0.1:8088>. The first poll creates one case for the two `DNSConfigForming` instances. The case moves through `pending → investigating → waiting_human` and exposes follow-up, one-off question, Loom-link, and archive flows.

The Diagnostics page shows successful poll snapshots, the durable Slack escalation outbox, and the process-local ring buffer provided by `rigging.log_setup`. The ring is intentionally bounded and clears on service restart or rollout. Python logs continue to stderr, which Cloud Run captures in Cloud Logging as the durable service-log source. Finelog's `RemoteLogHandler` can be added later if the service is given a reachable Finelog endpoint and stable log key; it is not required for polling visibility.

Run Playwright against the service:

```bash
OPS_BASE_URL=http://127.0.0.1:8088 npm --prefix infra/ops/dashboard run test:e2e
```

Use a local Loom ACP server instead of the stub with:

```bash
uv run --project infra/ops ops-workflow serve \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops \
  --grafana-api-url http://127.0.0.1:55433 \
  --grafana-api-token local-fixture-token \
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
    - group:ops@openathena.ai
```

The wildcard becomes `domain:openathena.ai`. Google Workspace reports `ops@openathena.ai` as a Group, so the explicit entry must keep its `group:` prefix. Run `pulumi preview` after a membership change. Each entry owns one `roles/iap.httpsResourceAccessor` grant, so Pulumi changes only the added or removed member.

The service has no alert receiver. It polls Grafana's active-alert API through IAP. The ops service account has `roles/iap.httpsResourceAccessor` on the Grafana Cloud Run service and `roles/iam.serviceAccountTokenCreator` on itself. Each request carries a short-lived, URL-scoped service-account JWT in `Proxy-Authorization`. Grafana receives the IAP identity as a Viewer. No Grafana API token or database password is mounted into the service.

The service reaches the shared Cloud SQL instance through the connector socket. `ops_app` reads and writes the `ops` workflow database. The separate `ops_migrator` identity owns schema objects but is never mounted into the service. Create both logins and their custom database roles using [`../cloudsql/README.md`](../cloudsql/README.md).

[`src/ops_workflow/schema.py`](src/ops_workflow/schema.py) is the current SQLAlchemy Core model. The repository uses SQLAlchemy's async psycopg engine for queries and transactions. Files under [`migrations/`](migrations/) carry existing databases forward and run only with `ops_migrator`; the Cloud Run service cannot create or alter tables.

The service reuses the `marin-grafana-slack-webhook` Secret Manager secret for agent-requested escalations. The webhook is not passed to Loom or the agent. The backend validates the `ops-result` artifact, suppresses escalations for Grafana error/critical alerts that already notified Slack, deduplicates warning escalations by fingerprint generation, and sends through a durable retrying outbox.

Pulumi maps the IAP-protected service to `ops.oa.dev` with a DNS-only Cloudflare CNAME. The Cloudflare API token is a deployment credential, not a runtime secret.

Cloud Run keeps one service-level warm instance and sends 100% of traffic to the latest
revision. Revision templates have min 0 and max 1, so prior revisions remain cold rollback
targets. The service-level max of 1 and the database minute slot protect the poller from
duplicate work during a rollout.

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

Production rollout applies migrations before starting the restricted runtime, then verifies that `last_poll_at` advances and the Diagnostics page reports the active-alert count returned by Grafana.

The checked-in `agent_mode` is `stub`. Change it to `loom` only after a managed Loom API is reachable from Cloud Run and `marin-ops-loom-token` has a secret version; then add the explicit Loom URL, repository root, and base to stack configuration.
