---
date: 2026-07-22
system: grafana
severity: degraded
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7460
issue: none
---

## TL;DR

- Grafana sent `[FIRING:1] ControlPlaneCrashLooping cw-rno2a`, but `ops.oa.dev` reported zero active alerts.
- Grafana 13.1 stored evaluated state in `alert_rule_state`; the ops poller still queried the empty legacy `alert_instance` table.
- The poller now reads Grafana's active-alert API through IAP and uses Grafana fingerprints for durable deduplication.
- Production completed authenticated polls at 02:04, 02:05, and 02:06 UTC. The reported alert had already resolved, so it was not replayed.

## Original problem report

Slack received `[FIRING:1] ControlPlaneCrashLooping cw-rno2a (Alerts control-plane critical)` at 01:14 UTC, while the ops dashboard showed no corresponding case. The expected value was `A=1, C=1`, with `alertname=ControlPlaneCrashLooping` and `cluster=cw-rno2a`.

## Investigation path

1. Grafana logs showed `ControlPlaneCrashLooping` active on every evaluation from 01:14:12 through 01:22:12 UTC.

2. Ops logs showed `reconciled Grafana snapshot: alerts=0 groups=0 queued=0` 15-30 seconds after each Grafana evaluation. The one-minute polling interval did not explain the miss.

3. A read-only Cloud SQL query found no rows in `grafana.public.alert_instance`. The same database contained the rule's evaluated state and fresh timestamps in `grafana.public.alert_rule_state`.

4. Commit `0fcf884289` showed that `PostgresGrafanaAlertSource` selected only `public.alert_instance` joined to `public.alert_rule` at `infra/ops/src/ops_workflow/grafana_source.py:21-40` and `infra/ops/src/ops_workflow/grafana_source.py:69-91`.

5. Querying `alert_rule_state` directly was rejected because it would couple the service to another Grafana-internal storage representation. Grafana's Alertmanager-compatible active-alert API exposed the evaluated alerts and stable fingerprints required by the workflow.

6. A local API fixture produced two `DNSConfigForming` fingerprints. The repository lifecycle tests created one grouped case, and Playwright verified the case, manual-question, and diagnostics flows.

7. The first production request at 02:03:10 UTC received `Permission 'iam.serviceAccounts.signJwt' denied on resource` while the new IAM binding propagated. The retry at 02:04:10 received `granted: true`; the ops log at 02:04:11 reported a successful snapshot.

8. After three successful production polls, Pulumi removed `ops_grafana_reader`, `ops_grafana_reader_role`, their Grafana grants, the generated password, and `cloudsql-ops-grafana-reader-password`.

## User course corrections

- An HTTP ingestion service was considered. The user required Grafana to remain canonical and chose polling, which removed a second alert-state write path.
- The empty dashboard was initially explainable as a sub-minute alert. The user pointed out that Grafana was still firing, which prompted timestamp correlation between evaluator and poller logs and ruled out poll timing.
- Reading Grafana's replacement state table was considered. The user asked to poll the API, which avoided another dependency on an internal Grafana schema.

## Root cause

The poller assumed that firing Grafana instances were materialized in `public.alert_instance`. Grafana 13.1's compressed state path instead updated `public.alert_rule_state`, leaving `alert_instance` empty while notifications and the Grafana UI correctly showed the alert. The SQL query therefore returned a valid empty snapshot; no error distinguished schema drift from a healthy zero-alert state.

This was an unsupported internal-schema dependency, not a race in the one-minute scheduler or a Grafana notification failure.

## Fix

`infra/ops/src/ops_workflow/grafana_source.py:49-89` now requests `/api/alertmanager/grafana/api/v2/alerts`, validates the response, and preserves Grafana's fingerprint, labels, annotations, start time, and generator URL.

`lib/rigging/src/rigging/auth.py:245-301` signs and caches the URL-scoped service-account JWT required by Google-managed IAP. `infra/ops/__main__.py:86-103` grants the ops runtime self-only JWT signing and Grafana IAP access.

No data migration was required. The active-alert API does not replay resolved alerts. Pulumi deleted the obsolete database roles, grants, random password, secret version, and Secret Manager secret after the API poll succeeded.

## How OPS.md could have shortened this

No Grafana OPS.md exists. Add `infra/grafana/OPS.md` with an "Alert consumer disagreement" section: correlate Grafana evaluation timestamps with consumer snapshot counts, then inspect the supported active-alert API before querying Grafana's database. State that `alert_instance`, `alert_rule_state`, and other Grafana tables are internal storage and must not be treated as stable integration APIs. This would apply to any dashboard, webhook relay, or alert-history consumer after a Grafana upgrade.

## Artifacts

- `infra/ops/src/ops_workflow/grafana_source.py`
- `infra/ops/tests/test_grafana_source.py`
- `infra/ops/fixtures/grafana-api/api/alertmanager/grafana/api/v2/alerts`
- https://ops.oa.dev
- https://grafana.oa.dev
- https://github.com/marin-community/marin/pull/7460
