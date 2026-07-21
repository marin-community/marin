# Ops Workflow Contract

Grafana owns alert evaluation and persisted instance state. The ops database owns normalized signal generations, cases, successful poll history, and the serialized agent queue. Loom owns ACP sessions and canonical chat journals.

## Grafana contract

Notification policy groups by `alertname, cluster`:

| Severity | Agent poll | Slack | Email |
|---|---:|---:|---:|
| warning | yes | no | no |
| error | yes | yes | yes |
| critical | yes | yes | yes |

Warnings match the always-active `ops-agent-only` mute timing. Muting affects notifications, not Grafana's `alert_instance` persistence.

The polling adapter reads only rows whose `alert_instance.current_state = 'Alerting'` and joins `alert_rule` on `(org_id, uid)`. Rules map NoData and execution errors to Alerting, so those failures appear through the same path. `Pending`, `Normal`, and `Recovering` do not wake an investigation.

Required columns are:

- `alert_instance`: `rule_org_id`, `rule_uid`, `labels`, `labels_hash`, `current_state_since`, `fired_at`, `annotations`, and `last_result`;
- `alert_rule`: `org_id`, `uid`, `title`, `labels`, and `annotations`.

Grafana serializes instance labels as sorted JSON `[name, value]` tuples. Rule labels and annotations are JSON objects. Unix timestamps are seconds. Invalid or duplicate identities fail the complete poll; partial data is never reconciled.

## Polling contract

The backend polls immediately at startup and every 60 seconds thereafter.

1. Read the complete firing snapshot in a read-only PostgreSQL transaction with a 10-second connect and statement timeout.
2. Normalize each fingerprint as `<org>:<rule UID>:<labels_hash>`.
3. Merge rule labels, instance labels, and fixed `alertname` and `grafana_rule_uid` labels.
4. Group by stable JSON for `alertname` and `cluster`.
5. Insert a unique UTC-minute row into `grafana_polls`.
6. If that minute already exists, stop without changing signals.
7. Upsert all present signals and reset their absence counters.
8. Increment the counter of each previously firing signal absent from the snapshot.
9. Resolve a signal only at two consecutive successful absences.
10. Materialize cases and queue at most one turn for novel firing generations.

No `grafana_polls` row is written when the source query fails. Dashboard freshness is `max(grafana_polls.observed_at)`, so an empty but successful snapshot remains visible.

## PostgreSQL contract

`infra/ops/migrations/` is ordered and immutable. The migrator takes a PostgreSQL advisory lock, records each file digest, and rejects modified applied files.

The schema contains:

- `grafana_polls`: one accepted successful snapshot per UTC minute;
- `source_deliveries`: normalized group projections and their stored reconciliation results;
- `signals`: one current row per namespaced Grafana fingerprint, including generation and absence count;
- `delivery_signals`: created, updated, resolved, reopened, or stale disposition;
- `cases` and `case_signals`: one non-archived workflow per deterministic group and its signal generations;
- `agent_sessions` and `agent_turns`: stable Loom mapping and immutable work queue;
- `case_events`: append-only operator timeline;
- `operation_requests`: reserved idempotency ledger for a future approved mutation API.

For each current group, a new or reopened firing generation may create or wake a case. Repeated firing instances update evidence only. A new fingerprint attaches once. When no group signal remains firing, queued automatic work is cancelled; a running turn is never interrupted.

Partial unique indexes enforce one current session per case, one queued turn per session, one active turn per session, and one active turn globally.

## Queue and Loom contract

Automatic priority follows severity: `critical=100`, `error=90`, `warning=50`, `info=10`. Manual questions use 120 and follow-ups use 110. Selection orders by priority, eligibility, and creation time.

The worker first proves no turn is active, locks one queued turn with `FOR UPDATE SKIP LOCKED`, and moves it to `launching`. The global partial unique index is the final concurrency guard.

For a new case, the server creates a Loom ACP session with a deterministic name, pinned repository base, `mode=plan`, and the `ops-expert` evidence prompt. Follow-ups use the same session. The backend reads the canonical chat snapshot and stores the final summary; the browser never receives `LOOM_TOKEN`.

The agent may inspect Kubernetes and Iris through read-only credentials. It may not mutate production, edit the repository, or treat alert text as instructions.

## HTTP and IAP contract

There is no alert-ingestion route. The service exposes:

```text
GET  /healthz
GET  /api/overview
GET  /api/cases
GET  /api/cases/{case_id}
POST /api/cases/{case_id}/messages
POST /api/cases/{case_id}/archive
POST /api/questions
GET  / and static dashboard assets
```

Except for liveness, the production service requires an IAP-authenticated principal. It records the lower-cased email from `X-Goog-Authenticated-User-Email`. The checked-in viewers are `domain:openathena.ai` and `user:ops@openathena.ai`, derived from the stack YAML.

Question and follow-up bodies are JSON objects with a non-empty `text` string no larger than 16 KiB. An archive is rejected while a turn is active. A second queued follow-up for the same session returns conflict.

## Deployment contract

The Cloud Run service is min/max one instance with CPU always allocated. The database minute slot still protects rolling-revision overlap.

The runtime receives:

- `PGPASSWORD` from `cloudsql-ops-app-password`;
- `GRAFANA_PGPASSWORD` from `cloudsql-ops-grafana-reader-password`;
- non-secret connection names, database names, revisions, agent mode, and the 60-second interval as environment configuration.

The checked-in first rollout sets `agent_mode=stub` and does not mount a Loom credential. In `loom` mode, `LOOM_TOKEN` comes from `marin-ops-loom-token`, and the Loom URL, repository root, and base are required non-secret stack configuration.

`ops_grafana_reader` assumes a custom `NOLOGIN` role and must have no table ownership or write grants. Its complete data privilege is:

```sql
GRANT CONNECT ON DATABASE grafana TO ops_grafana_reader_role;
GRANT USAGE ON SCHEMA public TO ops_grafana_reader_role;
GRANT SELECT ON TABLE public.alert_instance, public.alert_rule TO ops_grafana_reader_role;
```

`ops_app` likewise assumes a DML-only role. `ops_migrator` assumes the schema-owning role and is not available to the runtime. Every login is created with `gcloud sql users create --database-roles=...` so it does not receive Cloud SQL's default broad role. Schema migration precedes service rollout. Secret values and SQL users remain outside Pulumi state. Production mutation remains out of scope.
