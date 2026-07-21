# Ops Workflow Background Research

## Background Research Brief

- Effort: medium
- Stop rule: stop after the Marin observability, dashboard, database, Pulumi, and agent-session searches stop producing new reusable contracts
- Date: 2026-07-21
- Marin revision: `d885c9b0a40bf97de6658fa8b137e7d230d67ff6`
- Loom revision: `6f5188715d94a15dfbe10f5616a17e46ff2139ea`

### Question

How should Marin turn low-urgency operational signals into durable, agent-assisted investigations without sending every warning to Slack, duplicating agent infrastructure, or giving an unattended agent unsafe production access?

### Current Marin Context

The rows that motivated this design are Kubernetes Warning events, not Grafana alert instances. The Grafana bridge lists `type=Warning` events, truncates messages to 200 characters, keeps the newest 100, and exposes them through a current-state panel whose retention is about one hour ([source](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/grafana/src/k8s_source.py#L408-L430), [panel](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/grafana/dashboards/k8s.json#L650-L719)). It omits the Kubernetes event UID, so polling this projection cannot identify repeated observations reliably.

Grafana's provisioned alert catalog intentionally covers near-certain incidents, not all Warning events ([rules](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/grafana/provisioning/alerting/rules.yaml#L1-L13)). Notification routing currently defaults unmatched alerts to Slack, with exact `critical` and `warning` child routes ([policy](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/grafana/provisioning/alerting/policies.yaml#L4-L23)). Adding a `silent` label without a webhook-only route would therefore still notify Slack.

Grafana's private state is in the `grafana` Cloud SQL database. The repository defines no supported alert-table schema or query contract for that database. The shared Cloud SQL stack already isolates consumers with separate databases and native users ([Cloud SQL stack](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/cloudsql/__main__.py#L27-L39)). The ops workflow should own an `ops` database instead of reading or mutating Grafana internals.

### Internal Prior Work

#### Evaldash service shell

`infra/evaldash` is the closest Marin service pattern: an IAP-gated Cloud Run service, Vue SPA, Starlette API, periodic background loop, and Cloud SQL Postgres ([deployment](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/evaldash/__main__.py#L51-L130), [coordinator](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/evaldash/src/server.py#L264-L319)). Its visibility-aware 60-second Vue refresh and race-safe `fetch` wrapper are directly reusable ([refresh](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/evaldash/dashboard/src/composables/useRefresh.ts#L1-L82), [API wrapper](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/evaldash/dashboard/src/composables/useApi.ts#L17-L45)).

Two evaldash choices should not carry over. Its in-process lock relies on `min=max=1`, while ops claims must survive a restart. Its `metadata.create_all()` schema bootstrap has no ordered migration history ([schema bootstrap](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/evaldash/src/results_db.py#L147-L149)). Ops needs Postgres leases and versioned SQL migrations from the first release.

#### Pulumi and credentials

`CloudRunService` already owns the runtime service account, IAP, Direct VPC egress, Secret Manager grants, Cloud SQL attachment, one warm instance, and always-allocated CPU option ([component](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/pulumi/src/iac/gcp/cloud_run.py#L31-L89)). `CloudSqlPostgres` creates logical databases and secret shells without placing passwords in Pulumi state ([component](https://github.com/marin-community/marin/blob/d885c9b0a40bf97de6658fa8b137e7d230d67ff6/infra/pulumi/src/iac/gcp/cloud_sql.py#L33-L58)).

Grafana's CoreWeave credential is deliberately read-only. Broader Iris controller credentials can create, exec, update, and delete pods and mutate node pools. Existing recovery playbooks require explicit approval for destructive action. The v1 agent environment should therefore expose read-only cluster access and leave mutation out of scope.

#### Ops expertise

Marin already has focused operational playbooks: `debug` routes symptoms to the Iris and Zephyr runbooks, `scan-logs` bounds large-log analysis, `recover-stuck-k8s-pod` classifies recovery options and approval boundaries, and `write-ops-log` defines a durable handoff. A small `ops-expert` router should invoke these sources instead of copying them into one large personality prompt.

#### Loom session runtime

Loom already supplies the session features in the proposal. Its ACP sessions expose live conversation, steering or durable queued prompts, permission cards, archive capture, and a normalized chat journal ([conversation behavior](https://github.com/rjpower/weaver/blob/6f5188715d94a15dfbe10f5616a17e46ff2139ea/README.md#L236-L251)). Its token-authenticated API creates sessions and supports send, interrupt, archive, chat snapshot, and chat SSE ([API and auth](https://github.com/rjpower/weaver/blob/6f5188715d94a15dfbe10f5616a17e46ff2139ea/README.md#L277-L340)). The launch request accepts a managed repository, goal, agent, ACP protocol, and permission mode ([contract](https://github.com/rjpower/weaver/blob/6f5188715d94a15dfbe10f5616a17e46ff2139ea/crates/weaver-api/src/dto.rs#L531-L599)).

The ops service should use Loom as the runner, proxy its chat APIs to the IAP-authenticated browser, and store only the stable Loom session ID plus final investigation metadata. Reimplementing ACP in Marin would duplicate an actively maintained subsystem and create a second transcript authority.

### External Prior Art

Grafana supports webhook contact points, grouped notification payloads, firing and resolved deliveries, and HMAC-SHA256 signatures with optional timestamps. A timestamped HMAC lets the ops endpoint reject tampered or replayed requests without embedding credentials in the URL ([Grafana webhook documentation](https://grafana.com/docs/grafana/latest/alerting/configure-notifications/manage-contact-points/integrations/webhook-notifier/)). Notification policies route alerts to contact points by label and control grouping and repeat timing ([Grafana policy documentation](https://grafana.com/docs/grafana/latest/alerting/configure-notifications/create-notification-policy/)).

PostgreSQL `SELECT ... FOR UPDATE SKIP LOCKED` gives multiple coordinators a non-blocking claim primitive while preserving row locks inside a transaction ([PostgreSQL `SELECT`](https://www.postgresql.org/docs/current/sql-select.html)). Kubernetes recommends minimum-permission RBAC and short-lived credentials for external applications ([Service Accounts](https://kubernetes.io/docs/concepts/security/service-accounts/), [RBAC good practices](https://kubernetes.io/docs/concepts/security/rbac-good-practices/)).

### Evidence Map

#### Claim: the browser must not launch investigations

- Support: evaldash's timer pauses in hidden tabs; no open browser means no work. Multiple tabs also have no shared transactional exclusion.
- Directness to Marin: exact dashboard pattern proposed for reuse.
- Confidence: high.
- Action: run a server-side coordinator; the browser only reads state and sends authenticated user actions.

#### Claim: Grafana's database is the wrong integration boundary

- Support: no owned alert schema exists; Grafana notification webhooks are a supported, versioned boundary.
- Contradiction: using the existing database would avoid one webhook hop.
- Directness to Marin: the current Grafana database is already private application state.
- Confidence: high.
- Action: add a separate `ops` database and webhook receiver.

#### Claim: one boolean cannot represent cleanup

- Support: source resolution, agent completion, human archive, suppression, and failed execution occur independently.
- Directness to Marin: Grafana sends both firing and resolved events while Loom sessions can wait, fail, or be archived separately.
- Confidence: high.
- Action: model signals, cases, runs, and append-only case events separately.

### Negative / Failed Leads

- No supported Grafana alerts-table contract exists in Marin.
- No ACP client or persistent agent-session implementation exists in Marin; the implementation is in Loom.
- The current Kubernetes Warning-event projection is too lossy to serve as a durable source without adding stable event identity.
- Evaldash's in-process lock and schema bootstrap are insufficient for restart-safe work claiming and schema evolution.
- Browser-local chat history is not acceptable as the canonical transcript.

### Recommended First Vertical Slice

1. Add the `ops` database/user secret shell and an IAP-gated `infra/ops` service.
2. Add ordered SQL migrations, authenticated Grafana webhook ingestion, signal/case state, and a Postgres lease-based coordinator with a fake runner.
3. Add the minimal Vue inbox, case detail, status, and manual-question surfaces.
4. Add a Loom adapter and the versioned `ops-expert` skill; proxy Loom chat through the backend.
5. Add the webhook-only Grafana contact point and `handling=agent` policy after deduplication and authentication tests pass.
6. Add a stable Kubernetes Warning-event silent rule or a lossless source adapter; do not poll the existing lossy panel projection.

### Open Questions

- Should the first Kubernetes Warning-event source include provider namespaces by default, or place `kube-*` and `cw-*` events in a lower-priority queue?
- Is one automatic active turn fleet-wide the intended long-term policy, or only the safest v1 default?
- Should a session waiting for a human release the automatic slot? This design says yes.
- Which production Loom deployment and managed Marin repository should Pulumi target?
- What retention period should apply to raw deliveries and archived cases?

### Stop Reason

The service, data, session, and security boundaries are supported by implemented Marin and Loom code. Further searching did not find another queue, alert-history, or agent runtime worth reusing.
