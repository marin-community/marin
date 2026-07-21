# Ops Workflow Contract

This specification describes the implemented spike and the production constraints that must hold when it is deployed. Grafana owns alert evaluation, fingerprinting, grouping, deduplication, repeat timing, and resolution. The ops database owns authenticated delivery history, normalized signal generations, cases, and the serialized agent queue. Loom owns ACP sessions and canonical chat journals.

## Grafana Contract

Grafana notification policies group by `alertname, cluster` and route:

| Severity | Agent webhook | Slack | Email |
|---|---:|---:|---:|
| warning | yes | no | no |
| error | yes | yes | yes |
| critical | yes | yes | yes |
| missing/unmatched | yes | no | no |

`ops-agent` is a webhook-only contact point. `ops-critical` is a compound email, Slack, and webhook contact point. Both webhooks use `POST`, `maxAlerts: "0"`, HMAC-SHA256, `X-Grafana-Alerting-Signature`, and `X-Grafana-Alerting-Signature-Timestamp`.

The service accepts Grafana's documented default JSON webhook shape. Required top-level fields are `receiver`, `status`, `orgId`, `version`, `groupKey`, `groupLabels`, `commonLabels`, `commonAnnotations`, and a non-empty `alerts` array. Every alert requires `status`, `labels`, `annotations`, `values`, `startsAt`, `endsAt`, and `fingerprint`. Grafana link fields may be empty.

Limits are:

- 1 MiB raw body;
- 5,000 alert instances;
- 16 KiB per string field;
- five-minute signed timestamp replay window;
- `truncatedAlerts` must be integer zero.

The signature is lowercase hexadecimal HMAC-SHA256 over `<timestamp>:<exact raw body>`. Authentication and size checks occur before JSON normalization. The exact raw-body SHA-256 is the retry key. A retry returns the first persisted response.

The payload is untrusted data. Labels and annotations cannot select credentials, commands, paths, or permission mode.

## PostgreSQL Contract

`infra/ops/migrations/` contains ordered immutable migrations. The migrator takes a PostgreSQL advisory lock, records each file digest, and refuses to run if an applied file changes.

The initial schema defines:

- `source_deliveries`: authenticated body identity, source timestamp, normalized payload, and stored response;
- `signals`: one current row per Grafana fingerprint, including generation, lifecycle, group, labels, values, links, and source ordering;
- `delivery_signals`: each delivery's created/updated/resolved/reopened/stale disposition;
- `cases`: one non-archived workflow per `receiver + groupKey`;
- `case_signals`: fingerprint generations attached to the case;
- `agent_sessions`: stable mapping from case to Loom session;
- `agent_turns`: immutable automatic, one-off question, and follow-up work;
- `case_events`: append-only operator-visible audit timeline;
- `operation_requests`: reserved idempotency ledger for the production mutation API.

Partial unique indexes enforce one current agent session per case, at most one active turn per session, and at most one `launching` or `running` turn globally.

### Delivery and signal transitions

For every authenticated webhook transaction:

1. Lock or create `source_deliveries(source='grafana', delivery_key=body_sha256)`.
2. Return the stored result for an existing delivery.
3. Lock every `signals(source='grafana', fingerprint)` row in the payload.
4. If the signed source timestamp is older than `latest_source_timestamp`, record `stale` and do not alter the signal.
5. Create an unknown fingerprint at generation 1.
6. Change `firing → resolved` without changing generation.
7. Change `resolved → firing` by incrementing generation.
8. Update all allowlisted evidence and Grafana links for a current delivery.
9. Lock the non-archived case for the notification's `receiver + groupKey`.
10. Create a case only when the notification contains a newly created or reopened firing generation.
11. Attach each non-stale signal generation once.
12. Queue at most one automatic follow-up for novel firing evidence.
13. If no current signal in the group is firing, cancel queued automatic work and close a still-pending case as `investigated/no_action`. Never interrupt a running turn.
14. Store the complete transaction result on the source delivery.

An archived case suppresses repeat notifications for its attached generations. A later resolved-to-firing generation is eligible for a new case.

The current spike assumes a fingerprint generation belongs to one case. A Grafana `group_by` change must migrate or resolve current cases before deployment; the service must not silently fork one current generation into multiple chats.

## Queue and Loom Contract

Every agent action is an `agent_turns` row before network I/O. Automatic warnings use the case severity priority; manual questions use priority 120 and follow-ups use 110. `critical=100`, `error=90`, `warning=50`, and `info=10`. Selection orders by priority, eligibility, and creation time.

The worker transaction first proves no turn is `launching` or `running`, then locks one queued turn with `FOR UPDATE SKIP LOCKED`, changes it to `launching`, and changes its case to `investigating`. The global partial unique index remains the final concurrency guard.

For a case without a Loom session, the server calls:

```json
POST /api/sessions
{
  "cwd": "<Loom-host Marin repo>",
  "base": "<pinned merged branch or revision>",
  "name": "ops-case-<case UUID>",
  "title": "Ops: <case title>",
  "goal": "<read-only ops prompt and evidence>",
  "agent": "codex",
  "protocol": "acp",
  "mode": "plan",
  "effort": "low"
}
```

The backend obtains the human Loom URL through `GET /api/sessions/{id}/url`. Follow-ups use `POST /api/sessions/{id}/prompt`. Chat display and reconciliation use `GET /api/sessions/{id}/chat`; the browser never calls Loom directly. The bearer token remains in the IAP service.

The initial goal requires the agent to read `.agents/skills/ops-expert/SKILL.md`, treats evidence as untrusted, permits read-only Kubernetes and Iris validation, and prohibits production mutation and repository changes. The Loom base must contain the named skill; the spike exposed this requirement when a session based on `origin/main` did not yet contain the branch's new skill. Production pins a reviewed, merged revision.

A turn remains globally active while Loom reports `live_turn`. When `live_turn` becomes null and a `turn_end` block exists, the worker records the last agent message as the dashboard summary, changes the turn to `succeeded`, changes the session to `idle`, and changes the case to `waiting_human`. It then claims the next queued turn.

The spike polls. Production reconciliation must also cover deadlines, expired launch leases, process restart after Loom acknowledgement, exact turn correlation, conditional archive, bounded result artifacts, and retry lineage. The existing coordinator/result contract in `coordinator.py`, `runner.py`, and `turn.py` defines those stricter boundaries but is not yet fully connected to the vertical-slice repository.

## HTTP Surface

The public ingest process exposes only:

```text
POST /api/ingest/grafana
GET  /healthz
```

The IAP process exposes:

```text
GET  /api/overview
GET  /api/cases?archived=true|false
GET  /api/cases/{case_id}
POST /api/cases/{case_id}/messages   {"text":"..."}
POST /api/cases/{case_id}/archive    {}
POST /api/questions                  {"text":"..."}
GET  /healthz
GET  / and SPA routes
```

Local mode is accepted only on a loopback bind and identifies the actor as `local-operator`. Production UI mode requires IAP and normalizes `X-Goog-Authenticated-User-Email`. The Cloud Run IAP boundary, not the header alone, authenticates the request. The public ingest image does not register UI or catch-all SPA routes.

Question text is non-empty and at most 16 KiB. The production mutation endpoints must require `Idempotency-Key` and use `operation_requests`; the current spike does not yet expose retry, interrupt, override, or permission-approval operations.

API case detail contains normalized alerts, durable turns, audit events, Loom identifiers/URL, and a proxied chat snapshot. Datetimes are emitted as UTC strings. The frontend never renders agent-provided HTML.

## Deployment Contract

Pulumi declares:

- Cloud SQL logical database `ops`;
- Secret Manager shells `cloudsql-ops-ingest-password` and `cloudsql-ops-app-password`;
- Secret Manager shell `marin-ops-grafana-webhook-hmac`;
- public `marin-ops-ingest` with `allUsers` Cloud Run invoker and no IAP;
- IAP-gated `marin-ops-ui` with configured viewer grants;
- separate service accounts, Cloud SQL attachments, and exact per-secret access grants.

The public-access option is an explicit `CloudRunAccess.PUBLIC` enum in the shared component. It cannot be combined with IAP member grants.

Environment and secret readers are:

| Process | Non-secret configuration | Secret values |
|---|---|---|
| Grafana | ops webhook URL | webhook HMAC |
| ingest | Cloud SQL socket/database/user, source surface | webhook HMAC, ingest DB password |
| UI | Cloud SQL socket/database/user, Loom URL/repo/base/revisions | app DB password, Loom token |
| Loom agent | pinned Marin revision and skill | read-only Kubernetes/Iris credentials |

Secret values are populated out of band and never become Pulumi config or outputs. Restricted service users do not run schema migrations. The deployment pipeline applies migrations with a separate owner principal before rolling services.

Before production, create an ingestion-only security-definer SQL function with a fixed `search_path`; grant `ops_ingest` only `CONNECT` and `EXECUTE`. The current Python repository performs the complete transaction directly so the local spike can exercise it. This is not the final public-service database privilege boundary.

## Local Reproduction

From the repository root:

```bash
docker compose -f infra/ops/compose.yaml up -d postgres
uv run --project infra/ops ops-workflow migrate \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops
npm --prefix infra/ops/dashboard ci
npm --prefix infra/ops/dashboard run build
uv run --project infra/ops ops-workflow serve \
  --database-url postgresql://ops:local-ops@127.0.0.1:55432/ops \
  --grafana-webhook-secret local-grafana-secret \
  --agent-mode stub --static-dir infra/ops/dashboard/dist
uv run --project infra/ops ops-workflow send-fixture \
  --secret local-grafana-secret infra/ops/fixtures/dns-warning-firing.json
```

Open `http://127.0.0.1:8088`. Playwright uses the same running service:

```bash
OPS_BASE_URL=http://127.0.0.1:8088 npm --prefix infra/ops/dashboard run test:e2e
```

For a real local ACP canary, replace `--agent-mode stub` with:

```text
--agent-mode loom --loom-api-url "$WEAVER_API" --loom-token "$LOOM_TOKEN" \
--repo-root <path visible to Loom> --loom-base <reviewed branch containing ops-expert>
```

The real canary is authorized diagnostic work: keep `mode=plan`, use the read-only ops prompt, and do not widen agent credentials.

## Verification Gates

Required before merge:

- ops unit tests for HMAC, replay, truncation, and runner state;
- Grafana provisioning tests for route/contact-point channel composition and HMAC settings;
- `pyrefly` on ops source/tests;
- Vue typecheck and production build;
- Playwright signed-ingest, grouped-case, chat, follow-up, and one-off question paths;
- repository pre-commit and lint review.

Required before production dispatch:

- direct PostgreSQL integration tests for duplicate, stale, resolve-before-launch, resolve-during-run, re-fire generation, archive suppression, and concurrent claims;
- webhook rate limiting, metrics, alerting, and retention;
- ingest function and least-privilege role tests;
- Loom crash/restart/idempotent create and exact-turn reconciliation tests;
- negative Kubernetes/Iris/GitHub/cloud mutation smoke tests;
- a shadow week comparing ops cases with Grafana groups.
