# Ops Workflow Background Research

## Brief

- Effort: medium
- Date: 2026-07-21
- Marin branch: `weaver/ops-workflow`
- Question: How should Grafana alerts trigger durable, low-chatter, agent-assisted production diagnosis while keeping paging and credentials safe?
- Stop rule: stop after Grafana, Marin dashboard/Pulumi, PostgreSQL, Loom ACP, and a live read-only canary establish the integration boundaries.

## Findings

### Grafana should be canonical, but its database should not

The original Grafana panel rows are Kubernetes Warning events projected by Marin's bridge. That projection is a useful exploration surface but is bounded and omits stable Kubernetes event identity. It is not the alert lifecycle. Operators should encode actionable conditions as Grafana alert rules, then let Grafana own firing/resolved state, instance fingerprints, grouping, deduplication, and repeat timing.

Grafana's documented webhook is the supported downstream boundary. Its default payload includes `receiver`, group status, `groupKey`, alert fingerprints, labels, annotations, values, start/end times, and Grafana links. Its webhook integration supports HMAC-SHA256 over the request body, plus an optional timestamp header; when configured, the signed content is `<timestamp>:<body>`. Setting `maxAlerts=0` prevents group truncation ([Grafana webhook notifier](https://grafana.com/docs/grafana-cloud/alerting-and-irm/alerting/configure-notifications/manage-contact-points/integrations/webhook-notifier/)).

Grafana notification policies already solve the proposed cheap-model grouping problem deterministically. `group_by`, `group_wait`, `group_interval`, and `repeat_interval` control grouping and notification cadence; grouping labels should describe the operational failure domain ([Grafana notification grouping](https://grafana.com/docs/grafana/latest/alerting/fundamentals/notifications/group-alert-notifications/)). Provisioned contact points and policies remain reviewable files alongside rules ([Grafana file provisioning](https://grafana.com/docs/grafana/latest/alerting/set-up/provision-alerting-resources/file-provisioning/)).

Grafana's Cloud SQL database is private application state. Marin owns no stable alerts-table query contract, and direct writes would bypass Grafana notification semantics. The ops workflow therefore stores its own delivery and case state but never claims to be the source of truth for whether an alert is firing.

### Current Marin notification routing can express the desired tiers

Marin provisions rules, contact points, and the notification tree from YAML. Before this change, warning alerts went to Slack and critical alerts went to email plus Slack. A compound contact point can send critical/error notifications to the agent as well as the paging channels, while an agent-only contact point handles warnings. Making the agent receiver the root prevents an unlabelled rule from accidentally paging.

The motivating raw Warning rows are not all Grafana alerts. Existing rules target conditions judged close to incidents, such as cluster reachability, control-plane degradation, and stuck GPU termination. New warning-class rules must be selective; wrapping every Kubernetes Warning event in one rule would create alert and agent noise without adding intent.

### A separate ops UI should compose, not replace, Loom

`infra/evaldash` is Marin's closest application pattern: Vue 3, TypeScript, Rsbuild, Starlette, Cloud SQL, IAP, and visibility-aware refresh. Those patterns are reusable for the ops inbox and case page.

Loom already implements the hard agent runtime pieces. Its authenticated API creates ACP sessions with `cwd`, base, agent, model/effort, protocol, and permission mode; exposes canonical chat snapshots and an SSE tail; queues or steers prompts; interrupts exact sessions; archives; and returns a browser deep link. `GET /api/agents` on the local instance reported both Codex and Claude ACP runtimes available. The ops service should proxy this API and retain only session identity and workflow results. A second ACP implementation would create two transcript authorities.

The browser must not receive `LOOM_TOKEN`, launch sessions, or own the queue timer. Tabs disappear, duplicate, and sleep. A server worker plus PostgreSQL partial unique index makes the one-agent policy independent of browser state.

### Alert and workflow identities are different

The durable model needs four identities:

1. raw body SHA-256 for exact webhook retries;
2. Grafana `fingerprint` for one current alert instance;
3. fingerprint `generation` for a resolved instance that later fires again;
4. `receiver + groupKey` for the operator's investigation thread.

One `cleaned` flag cannot express source resolution, investigation completion, human follow-up, failure, archive, or re-fire. Separate signal, case, session, and turn states are required.

PostgreSQL `FOR UPDATE SKIP LOCKED` supports a non-blocking multi-worker claim, while a partial unique index remains the final guarantee that only one turn is launching/running. The queue should contain automatic alerts, one-off questions, and follow-ups so manual work cannot bypass concurrency and cost controls.

### Credentials require process isolation

Grafana signs to a loopback relay in its bridge. The bridge enqueues the exact body and signature headers in Cloud Tasks. The ingest service accepts internal traffic and the dedicated dispatcher identity, exposes no case APIs, and receives no Loom or cluster credential. The IAP UI needs an application DB user and a dedicated Loom token. Kubernetes and Iris credentials belong to the agent runtime, not the web process.

Cloud Run's `internal` ingress admits same-project Cloud Tasks while rejecting requests from the public internet. IAM still applies to every admitted network request. Google recommends a per-service account with `roles/run.invoker` and a short-lived OIDC ID token for service-to-service calls ([Cloud Run ingress](https://docs.cloud.google.com/run/docs/securing/ingress), [service-to-service authentication](https://docs.cloud.google.com/run/docs/authenticating/service-to-service)).

Prompt language and the `ops-expert` skill are not security controls. Plan-mode ACP plus read-only RBAC, no secret reads, no pod exec, no GitHub write token, and negative mutation tests form the boundary. The ingest DB role needs a fixed-`search_path` security-definer ingestion function before deployment; direct table access is acceptable only in the local vertical slice.

Pulumi's existing `CloudRunService` and `CloudSqlPostgres` components cover service accounts, IAP, Cloud SQL sockets, Secret Manager grants, logical databases, and password secret shells. The ops split adds explicit private access and internal ingress. Cloud Tasks is the only network source admitted by Cloud Run, and its dispatcher is the only principal with `roles/run.invoker`.

## Live Validation

The spike replayed a real-shaped notification group containing the two motivating `DNSConfigForming` fingerprints. The backend created one case, launched a real Codex ACP session through Loom, and exposed its chat at `https://loom.rjp.io/s/732hh1uo`. The agent used `~/.kube/coreweave-iris` and context `marin-us-east-08a_US-EAST-08A` for read-only validation.

It found:

- both alerted pods were Running/Ready with zero application-container restarts;
- a third system pod on the same node had the same warning;
- node `sg6txs64` was Ready and hosted no Iris workload;
- node-local DNS was 206/206 Ready and NVIDIA IMEX was 202/202 Ready;
- the retained resolver line was `1.1.1.1 8.8.8.8 1.1.1.1`, pointing to duplicated host resolver configuration;
- immediate impact was low, no eviction/restart was indicated, and CoreWeave should correct the managed node resolver configuration.

No production mutation occurred. This validates that Grafana evidence can identify a bounded target while live Kubernetes evidence determines current severity and next action.

The first real session based itself on `origin/main`, where the new `ops-expert` skill was not yet present. The explicit read-only prompt still kept the run safe, but the failure proved that production must pin a merged Marin revision containing the skill. The Loom adapter now accepts an explicit base.

## Negative Leads

- Polling a Grafana SQL alerts table: no supported Marin/Grafana schema contract, and it bypasses Alertmanager lifecycle semantics.
- Polling the rendered Warning-event panel: lossy projection and no rule-author intent.
- Model-based grouping before deterministic grouping is tried: adds cost and nondeterminism where Grafana already emits `groupKey`.
- Adding this UI directly to Loom: couples a general agent fleet UI to Marin-specific alert and case state.
- Letting the browser run the minute loop: loses work when hidden/closed and cannot safely serialize multiple tabs.
- Sending all warnings to Slack and relying on the agent to mute them: Slack noise has already occurred by then.
- Sharing broad production credentials with a general-purpose Loom deployment: violates least privilege and expands blast radius.

## Recommended Path

1. Keep Grafana as the only alert-definition and alert-lifecycle surface.
2. Land the signed webhook, fingerprint/group case model, global turn queue, Vue UI, and stub/real Loom adapters.
3. Split internal ingest from the IAP UI in Cloud Run and finish the database privilege boundary.
4. Run warning notifications in shadow/no-agent mode to tune rules and grouping.
5. Pin a dedicated ops Loom runtime to a reviewed revision and pass negative mutation tests.
6. Enable one cluster with a daily launch budget, then expand. Critical/error paging remains independent and continues to Slack/email.

## Stop Reason

The repository, official Grafana contracts, Loom API, browser test, and live read-only canary agree on the system boundaries. Further research would not reduce the remaining implementation risks, which are concrete production hardening tasks: least-privilege SQL, crash reconciliation, metrics/rate limits, and negative credential tests.
