# Ops Workflow Research Notes

## Grafana as the source of truth

Grafana's alerting fundamentals distinguish alert-rule evaluation from notification delivery: rules update alert-instance state, while firing and resolved instances are later sent to contact points. This supports muting warning notifications without losing the canonical firing state ([Grafana alerting fundamentals](https://grafana.com/docs/grafana/latest/alerting/fundamentals/)).

Grafana's performance documentation states that every alert-instance state is written to the `alert_instance` table in Grafana's SQL database ([Grafana alerting performance considerations](https://grafana.com/docs/grafana/latest/alerting/set-up/performance-limitations/)). A read-only database projection is therefore a workable spike boundary. The adapter remains isolated because Grafana does not document this table as a stable public integration API.

The Grafana 13.1 source confirms the serialization details used by the adapter:

- `alert_instance.current_state` uses `Alerting`, `Normal`, `Pending`, `NoData`, `Error`, or `Recovering`;
- timestamps are persisted as Unix seconds;
- instance labels are sorted JSON `[name, value]` tuples;
- `labels_hash` is the persisted rule-instance identity;
- `alert_instance` joins `alert_rule` through organization and rule UID.

Primary source files are [`models/instance.go`](https://github.com/grafana/grafana/blob/main/pkg/services/ngalert/models/instance.go), [`models/instance_labels.go`](https://github.com/grafana/grafana/blob/main/pkg/services/ngalert/models/instance_labels.go), and [`store/instance_database.go`](https://github.com/grafana/grafana/blob/main/pkg/services/ngalert/store/instance_database.go).

The live `marin-metadata.grafana` schema was inspected read-only on 2026-07-21. It contains the expected columns in `alert_instance` and `alert_rule`; seven rules existed and no instance was firing at inspection time. No production data was changed.

## Why polling instead of an ingest service

The original spike used a signed receiver, a loopback relay, Cloud Tasks, and a second Cloud Run surface. That shape protected a push endpoint, but the endpoint itself was unnecessary: Grafana already durably stores the state the agent needs.

Polling removes:

- an internet or VPC HTTP receiver;
- webhook HMAC lifecycle and replay policy;
- a task queue and dispatcher identity;
- an ingestion-only database role and service;
- delivery retry and dead-letter operations.

It adds one narrow SQL reader. The tradeoff is up to 60 seconds of wake-up latency and reliance on a private Grafana schema. For ops warnings, that is preferable to a larger delivery system. The adapter and schema-shaped tests localize the upgrade risk.

## Grouping

Grafana notification policy already groups by `alertname, cluster`. A cheap grouping model would add cost, latency, nondeterminism, and a second source of identity without evidence that deterministic labels are insufficient. The workflow preserves exact fingerprint and group identity, while the investigation agent remains free to summarize repeated symptoms inside the case.

## IAP

The existing `CloudRunService` component normalizes `*@openathena.ai` to `domain:openathena.ai` and a bare address to `user:<address>`, then creates one `roles/iap.httpsResourceAccessor` grant per member. Keeping this non-secret list in stack YAML makes access reviewable and diffable. Secrets remain in Secret Manager and are granted only to the runtime service account.

## Agent and cluster access

Loom already provides ACP sessions, prompt continuation, and canonical chat snapshots. The ops service should proxy only the narrow APIs the Vue application needs and keep its bearer token server-side. The `ops-expert` skill requires bounded evidence and read-only Kubernetes and Iris diagnostics. Production mutation is deliberately excluded from this rollout.
