# Marin Ops Workflow

This directory contains the end-to-end spike behind `ops.oa.dev`: authenticated Grafana Alerting webhooks, Grafana fingerprint/group lifecycle state in PostgreSQL, a globally serialized ACP queue, a server-side Loom adapter, and a Vue case dashboard.

Grafana remains canonical for alert rules, grouping, deduplication, repeat timing, and resolution. Warnings reach the agent-only contact point; critical/error notifications reach Slack, email, and the agent. Kubernetes and Iris are read-only evidence sources used after Grafana wakes an investigation.

The full design, edge cases, security boundary, and rollout gates are in [`../../.agents/projects/ops_workflow/`](../../.agents/projects/ops_workflow/).

## Local spike

Start PostgreSQL, apply the migration, build the dashboard, and run the deterministic agent stub:

```bash
docker compose -f infra/ops/compose.yaml up -d postgres
uv run --project infra/ops ops-workflow migrate \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops
npm --prefix infra/ops/dashboard ci
npm --prefix infra/ops/dashboard run build
uv run --project infra/ops ops-workflow serve \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops \
  --grafana-webhook-secret local-grafana-secret \
  --agent-mode stub \
  --static-dir infra/ops/dashboard/dist
```

Replay the signed firing fixture:

```bash
uv run --project infra/ops ops-workflow send-fixture \
  --secret local-grafana-secret \
  infra/ops/fixtures/dns-warning-firing.json
```

Open <http://127.0.0.1:8088>. The case moves through `pending → investigating → waiting_human`, shows the two Grafana fingerprints, and exposes follow-up, one-off question, Loom-link, and archive flows.

Run the browser test against that service:

```bash
OPS_BASE_URL=http://127.0.0.1:8088 npm --prefix infra/ops/dashboard run test:e2e
```

Use the local Loom ACP server instead of the stub with:

```bash
uv run --project infra/ops ops-workflow serve \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops \
  --grafana-webhook-secret local-grafana-secret \
  --agent-mode loom \
  --loom-api-url "$WEAVER_API" \
  --loom-token "$LOOM_TOKEN" \
  --repo-root /path/visible/to/loom \
  --loom-base weaver/ops-workflow \
  --static-dir infra/ops/dashboard/dist
```

The base must contain `.agents/skills/ops-expert/SKILL.md`. The adapter creates ACP sessions in plan mode, and the ops prompt prohibits production mutation and repository changes.

## Verification

```bash
uv run --project infra/ops pytest infra/ops/tests
uv run --with pyrefly pyrefly check infra/ops/src infra/ops/tests
uv run --project infra/grafana pytest infra/grafana/tests/test_provisioning.py
npm --prefix infra/ops/dashboard run build:check
```

`infra/ops/compose.yaml` is for local development. Pulumi declares an internal webhook-only ingest surface, a Cloud Tasks delivery queue, and an IAP-gated UI/worker. Grafana sends signed webhooks to its loopback bridge; only the Grafana service account may enqueue, and only the queue dispatcher may invoke ingest.

The IAP allowlist is non-secret policy in `Pulumi.marin-ops.yaml`:

```yaml
config:
  marin-ops:viewers:
    - "*@openathena.ai"
    - ops@openathena.ai
```

Use a `group:` member when the ops address is a Google Group. Bare addresses are individual users. Run `pulumi preview` after an allowlist change; Pulumi adds or removes only the corresponding `roles/iap.httpsResourceAccessor` grants.

Before production, finish the ingestion-only PostgreSQL function/role, crash reconciliation, queue failure metrics, retention, and negative mutation tests described in the spec.
