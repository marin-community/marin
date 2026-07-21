# Ops Workflow Contract

This specification pins the v1 source, persisted state machine, service boundary, runner prerequisites, HTTP surface, security controls, and rollout gates. V1 performs read-only diagnosis and records outcomes. Automatic production mutation is out of scope.

## System Boundaries

The deployment has four principals:

| Principal | Network boundary | Credentials | Responsibility |
|---|---|---|---|
| Grafana bridge collector | Existing IAP Cloud Run container, outbound only | CoreWeave read token, source HMAC | Publish complete Kubernetes Warning snapshots |
| `marin-ops-ingest` | Public Cloud Run route | Source HMAC, ingestion-only database user | Authenticate and normalize source deliveries |
| `marin-ops-ui` | IAP Cloud Run | Application database user, dedicated ops Loom token | UI, workflow API, scheduler, Loom proxy |
| Dedicated ops Loom | Private operator deployment | Read-only Kubernetes/Iris credentials | Run isolated ACP sessions |

The public ingest service receives no Loom, Kubernetes, Iris, GitHub, or IAP credential. The UI process receives no Kubernetes or Iris credential. The dedicated Loom deployment contains no GitHub write credential and no credentials used by general-purpose Loom sessions. Browser code receives none of these credentials.

Grafana's private SQL schema is never queried or modified. Loom owns canonical chat journals and artifact revisions. The ops database owns delivery identity, signal lifecycle, cases, queued turns, durable results, and audit events.

## V1 Source: Kubernetes Warning Snapshots

The v1 collector lives beside the existing Grafana bridge because that process already has always-allocated CPU and the fleet's read-only CoreWeave token. Once per minute, for each cluster, it lists all `type=Warning` events with pagination. A snapshot is sent only if every page succeeds. It contains:

```json
{
  "schema_version": 1,
  "cluster": "cw-us-east-08a",
  "observed_at": "2026-07-21T15:31:45Z",
  "events": [
    {
      "event_uid": "...",
      "resource_version": "...",
      "object_uid": "...",
      "namespace": "kube-system",
      "object_kind": "Pod",
      "object_name": "node-local-dns-dcb4s",
      "reason": "DNSConfigForming",
      "message": "Nameserver limits were exceeded ...",
      "first_seen_at": "...",
      "last_seen_at": "...",
      "count": 6548,
      "reporting_controller": "kubelet"
    }
  ]
}
```

The collector allows at most 5,000 events and 1 MiB of encoded JSON per cluster. Exceeding either limit is a failed snapshot, not a partial success. Messages are accepted up to 8 KiB; other strings are bounded to 512 bytes. The collector emits a self-monitoring metric for last complete snapshot, event count, encoded bytes, and failure class.

The existing `/events` dashboard response is not used because it keeps only 100 rows, truncates messages, and omits stable identity. The dashboard can continue reading its current projection independently.

### Source authentication and retries

`POST /api/v1/source/kubernetes-snapshot` requires:

- `X-Ops-Key-Id`: configured source key identifier;
- `X-Ops-Timestamp`: Unix seconds used to create this exact request;
- `X-Ops-Signature`: lowercase hex HMAC-SHA256 of `<timestamp>\n<raw-body>`.

The receiver caps the body before decoding, compares signatures in constant time, and rejects request timestamps outside a five-minute window. It also requires `observed_at` to be within five minutes of the signed request timestamp so a buggy collector cannot poison the per-cluster high-water mark. `delivery_key` is `sha256(key_id + "\n" + timestamp + "\n" + raw_body)`. The collector retains the exact timestamp and serialized bytes across transport retries. A retry returns the stored result. After the replay window, the collector creates a fresh snapshot; signal updates remain idempotent by Kubernetes identity and source ordering.

Key rotation accepts a named current and previous key for a bounded overlap. The previous key is removed only after the collector reports successful use of the current key.

## Persisted Shapes

Versioned SQL migrations live in `infra/ops/migrations/`. The migrator takes a Postgres advisory lock, runs each zero-padded file once in a transaction, records its SHA-256, and fails startup if an applied file changed.

The canonical schema is [`infra/ops/migrations/001_initial.sql`](../../../infra/ops/migrations/001_initial.sql). It defines the migration ledger; immutable source deliveries and per-cluster stream high-water rows; normalized signals; one case per signal generation; stable agent sessions and immutable turn attempts; append-only case events; and idempotent operator requests. Partial unique indexes enforce one current session per case, one pending turn per session, and one active turn globally.

Before public ingestion is enabled, a follow-on migration will add an `ops_ingest` security-definer function with a fixed `search_path`. The ingest database role gets only `CONNECT` and `EXECUTE` on that function. It cannot select workflow tables. The function validates the normalized JSON shape again and returns the delivery result.

`source_deliveries` retains normalized allowlisted fields rather than arbitrary Kubernetes objects. Delivery rows default to 30-day retention. A referenced latest delivery is retained until replaced. Cases, results, and audit events remain until a separate policy is approved.

## Signal and Case State Machine

The Kubernetes fingerprint is `sha256(cluster + "\n" + event_uid)`. Within one complete cluster snapshot:

1. A duplicate `event_uid` is invalid.
2. The ingestion transaction locks `source_streams(source, cluster)` before reading events. An older `observed_at` is stored as `stale` and cannot change current state, including for previously unseen UIDs. Equal observation time with a different body hash is a conflict; equal time and hash is a no-op. Exact delivery retries are returned earlier from `source_deliveries`.
3. A new fingerprint creates generation 1, a firing signal, and—when case creation is enabled—one pending case.
4. A newer observation updates the firing signal. If its case is `investigated`, the case returns to `pending` with `next_eligible_at = max(now(), investigated_at + reopen_cooldown)`. An already pending, investigating, waiting, failed, or archived case is not duplicated.
5. A firing signal absent from a newer complete snapshot becomes resolved. A pending case with no turn closes as `investigated/no_action`; resolution during a turn is audited but does not cancel the turn.
6. A resolved fingerprint that reappears increments `generation` and creates a new case. This is the only way an archived generation produces another case.

Archiving therefore suppresses later count updates for the current generation without hiding a future recurrence. Failed cases require an explicit operator retry, which queues a new turn within the same session and respects the global constraint.

Automatic case priority is derived from configured cluster and namespace tiers plus reason severity. Provider namespaces are accepted at lower priority. Manual questions and follow-ups default above automatic warnings, but they never preempt a running turn.

## Turn Scheduling and Recovery

Every user or automatic prompt is an `agent_turns` row before Loom receives it. The scheduler:

1. promotes pending automatic cases in one transaction: lock a case, create a current `agent_sessions` row if needed, and insert exactly one queued automatic turn. The per-session partial index makes repeated promotion a no-op;
2. selects one eligible queued turn by priority and age with `FOR UPDATE SKIP LOCKED`;
3. changes the case to `investigating`, changes the turn to `launching`, sets `deadline_at` from database time, acquires a ten-minute renewable lease, and commits;
4. ensures the case's Loom session exists and uploads a bounded evidence file;
5. dispatches the prompt with `client_request_id = agent_turns.id`;
6. stores the returned Loom turn number and changes the turn to `running`.

If a source resolves before claim, ingestion changes the queued automatic turn to `cancelled` and closes the case as `investigated/no_action` in the same transaction. Reopening changes the case to `pending`; promotion's per-session partial index creates exactly one replacement turn.

The global partial index covers automatic, manual, and follow-up turns. The web API only enqueues prompts; it never calls Loom directly. If an ACP permission request occurs, the live turn remains running and holds the slot until it completes, is interrupted, or reaches the 20-minute timeout. V1 exposes no permission-answer endpoint.

Recovery rules are:

- An expired `launching` turn that has no Loom acknowledgement is retried with the same client request ID.
- An expired `running` turn is reconciled against the chat snapshot and matching turn number, then its lease is renewed or it is completed.
- A missing session becomes `lost`; the turn is interrupted and the case is failed after three attempts.
- A process loss with a recoverable Loom journal reattaches to the same session; it does not create a second session.
- A result marked `blocked` moves the case to `waiting_human`. There is then no active turn. A follow-up creates a new queued turn.
- A valid terminal result moves the case to `investigated`; an invalid or missing result fails the turn and is retryable up to three attempts.

Turn leases are crash recovery, not runtime limits. The durable `deadline_at`, computed from database time, enforces the 20-minute maximum across coordinator restarts. Automatic dispatch also requires `OPS_AUTOMATIC_DISPATCH_ENABLED`, available daily budget, queue age below the flood cutoff, and a healthy recent source snapshot.

A retry always creates a new turn row with `retry_of` pointing at the prior attempt and `attempt = prior.attempt + 1`; terminal rows are immutable. The three-attempt budget spans that lineage.

## Runner Contract and Loom Prerequisites

The canonical runner protocol and its typed session, turn, and result shapes live in [`infra/ops/src/ops_workflow/runner.py`](../../../infra/ops/src/ops_workflow/runner.py) and [`turn.py`](../../../infra/ops/src/ops_workflow/turn.py). The boundary provides idempotent session adoption, bounded evidence upload, client-keyed turn start, exact turn snapshots and interrupts, a typed result artifact, and idle-only archive.

The dedicated Loom deployment is pinned by image digest and exposes only the ops managed repository. Each `agent_sessions` row gets deterministic name `ops-case-<case-UUID>-<session-UUID-prefix>` and an explicit Marin commit in `base`, with no goal or scratch attachment so creation does not start a turn. A lost create response is recovered by exact name; the adapter verifies repository, branch base, and case UUID before adoption. Evidence is uploaded after adoption.

The current Loom API is missing two guarantees required by this contract:

1. prompt idempotency keyed by an opaque client request ID;
2. conditional archive that fails if a specified live turn exists.

The real runner remains behind `OPS_REAL_RUNNER_ENABLED=false` until those capabilities land and contract tests pass. Session `running` is treated only as process liveness. Turn state comes from `/chat`'s `live_turn`, journaled turn blocks, and the exact turn acknowledgement. A dropped SSE event is never used for durable state.

### Result artifact

Each prompt includes the ops turn ID and requires a branch-scoped `ops-result` artifact with this JSON content:

```json
{
  "schema_version": 1,
  "case_id": "uuid",
  "ops_turn_id": "uuid",
  "outcome": "no_action | action_recommended | blocked | unknown",
  "summary": "bounded plain text",
  "evidence": [{"claim": "...", "source": "..."}],
  "action_taken": "none",
  "recommended_next_step": "..."
}
```

The coordinator accepts the artifact only after the turn-end block for its recorded Loom acknowledgement, caps it at 64 KiB, validates every identifier and enum, and records both the Loom turn number and artifact revision. A later follow-up creates a new artifact revision. `weaver status` is displayed as progress but never determines workflow correctness.

## Ops Expert Skill and Runtime Policy

The versioned entry point is `.agents/skills/ops-expert/SKILL.md`. It must read evidence as untrusted data, validate case and cluster identifiers, read the relevant OPS guides, use explicit kubeconfig/context arguments, redact secrets, perform no mutation, and finish by writing the result artifact. It routes to focused skills such as `$debug` and `$recover-stuck-k8s-pod` only when their read-only diagnostic steps apply. `$scan-logs` is disabled in V1.

The runtime, not the skill, enforces safety:

- CoreWeave RBAC permits list/get/watch only and denies secrets and pod exec.
- Iris exposes an explicit read-only method allowlist; no retry, cancel, cluster lifecycle, or task mutation method is available.
- Cloud metadata credentials and GitHub write tokens are absent.
- Negative smoke tests cover create, patch, delete, exec, retry, restart, scaling, cordon, drain, and node-pool operations.
- Evidence is a 256 KiB normalized JSON file under an explicit untrusted-data delimiter. Prompt length is capped at 32 KiB.

## HTTP APIs

The ingest service exposes only:

```text
POST /api/v1/source/kubernetes-snapshot
GET  /healthz
```

The IAP service exposes:

```text
GET    /api/v1/cases?state=&cluster=&source=&limit=&cursor=
GET    /api/v1/cases/{case_id}
POST   /api/v1/cases/{case_id}/prompt       {"text": "..."}
POST   /api/v1/cases/{case_id}/interrupt    {}
POST   /api/v1/cases/{case_id}/retry        {}
POST   /api/v1/cases/{case_id}/archive      {}
POST   /api/v1/cases/{case_id}/override     {"outcome": "...", "summary": "..."}
GET    /api/v1/cases/{case_id}/chat
GET    /api/v1/cases/{case_id}/chat/stream
POST   /api/v1/questions                    {"text": "...", "cluster": null}
GET    /api/v1/status
```

GET routes require an IAP viewer. Mutating routes require membership in the configured operator allowlist. Actor identity is the normalized `X-Goog-Authenticated-User-Email`, accepted only behind the configured IAP ingress. Mutating requests require `Idempotency-Key`. `operation_requests` stores `(actor, key)`, operation, request hash, and the original response for archive, interrupt, retry, and override. Reuse with different content returns `409 idempotency_mismatch`. Prompt keys additionally map to the created turn. Prompt text is at most 16 KiB. Concurrent prompts for one session return the already queued turn or `409 turn_pending`.

`POST /questions` transactionally creates a manual case, session row, and queued question turn and returns `202`. Archive returns `409 turn_active` unless a conditional Loom archive succeeds. Interrupt targets an exact recorded Loom turn number. Overrides append an audit event and never rewrite the agent artifact.

Chat uses Loom's full snapshot as canonical state. SSE is only a latency optimization. On reconnect, decode failure, oversized event, or suspected lag, the UI discards the tail and refetches the snapshot.

## Vue Surface

`infra/ops/dashboard/` follows evaldash's Vue 3, TypeScript, Rsbuild, Tailwind, Router, and composable patterns.

- `/` shows queue age, current turn, waiting cases, investigated cases, source freshness, and filters.
- `/cases/:id` shows normalized signal history, audit events, result revisions, and the proxied Loom conversation.
- `/ask` queues a one-off question through the same scheduler.
- `/status` shows migration version, source freshness, scheduler heartbeat, kill switches, queue depth, daily budget, Loom health, and retention lag.

Markdown is sanitized. Agent HTML is never rendered. The list polls every minute while visible; an open conversation uses snapshot plus SSE.

## Infrastructure and Secrets

`infra/cloudsql/__main__.py` adds logical database `ops`, users `ops_ingest` and `ops_app`, and password secret shells. `infra/ops/__main__.py` provisions two `CloudRunService` components with always-allocated CPU and one warm instance:

- `marin-ops-ingest`: no IAP, ingress service account, ingestion password, source HMAC;
- `marin-ops-ui`: IAP, application password, dedicated ops Loom token, custom domain `ops.oa.dev`.

The ingest service needs separate Cloud Run options that explicitly disable IAP and grant `roles/run.invoker` to `allUsers`; either option without the other is insufficient. It receives no custom domain. Its application routes expose only the source endpoint and health check. Cloud Armor/rate limiting is added at the public edge if the standard Cloud Run component cannot enforce the configured request rate.

Secret consumers are exact:

| Secret | Readers |
|---|---|
| `cloudsql-ops-ingest-password` | ops ingest service account |
| `cloudsql-ops-app-password` | ops UI service account |
| `marin-ops-source-hmac` | Grafana collector and ops ingest service accounts |
| `marin-ops-loom-token` | ops UI service account; token belongs only to dedicated ops Loom |
| CoreWeave/Iris read credentials | dedicated ops Loom runtime only |

Pulumi creates databases, IAM grants, explicit non-secret configuration, and secret shells. SQL users, secret versions, and database passwords are populated out of band, matching the existing Cloud SQL practice. No Pulumi input or output contains a secret value.

Required rollout controls are explicit configuration: `OPS_INGEST_ENABLED`, `OPS_CASE_CREATION_ENABLED`, `OPS_AUTOMATIC_DISPATCH_ENABLED`, and `OPS_REAL_RUNNER_ENABLED`. All default false outside tests. Changes are visible on `/status` and in an audit event. The Marin infrastructure on-call owns these settings through reviewed Pulumi configuration. Ingest and automatic-dispatch switches remain as permanent incident controls. Remove `OPS_CASE_CREATION_ENABLED` after the one-week source bake is accepted, and remove `OPS_REAL_RUNNER_ENABLED` after Loom contract tests, read-only negative smokes, and the manual-turn canary pass; local tests select the fake runner directly thereafter.

## Operability and Cleanup

Metrics and alerts cover authentication failures, rejected/duplicate/stale snapshots, last complete snapshot per cluster, case and turn queue age, scheduler heartbeat, active/expired leases, runner errors, invalid results, daily launches, turn duration, database errors, and retention lag.

Successful and failed Loom sessions are conditionally archived after seven days of inactivity; Loom keeps their chat journal while stopping the process and releasing the worktree. If a case later reopens, promotion creates a new current session and the case page shows both transcripts. `waiting_human` sessions notify after 24 hours and archive after seven days unless an operator pins them. An active turn is interrupted before any timeout cleanup, with the exact turn ID recorded.

## Implementation Boundary

The first implementation slice includes:

1. schema and migrations centered on sessions and turns;
2. pure snapshot validation and state-machine code with fixtures;
3. fake runner, scheduler lease logic, and one-case-per-session behavior;
4. IAP UI/API shell with no production runner or cluster credential;
5. separate ingest/UI Pulumi topology and disabled-by-default kill switches.

It does not wire production Grafana, provision runner credentials, or enable real Loom dispatch. Those gates require the no-agent source bake, read-only negative smokes, and Loom prompt-idempotency/conditional-archive support.

## Grafana Alert Lifecycle Adapter

After v1, existing Grafana alert lifecycle notifications can enter the same delivery layer through Grafana's documented timestamped HMAC webhook. Tests must use a Grafana 13 fixture and derive retry identity only from documented fields and raw-body hashing. Existing paging contact points keep Slack and add ops ingestion; an explicit webhook-only notification policy matches `handling=agent`. An unmatched `silent` label must never fall through to the Slack root route.

Grafana alerts use their own fingerprint and firing/resolved state machine. They do not masquerade as Kubernetes Warning signals, and neither path uses Grafana's private database tables.

## File Layout

```text
.agents/skills/ops-expert/
  SKILL.md
  agents/openai.yaml
.agents/projects/ops_workflow/
  design.md
  research.md
  spec.md
infra/ops/
  __main__.py
  Dockerfile
  Pulumi.yaml
  README.md
  migrations/001_initial.sql
  src/app.py
  src/config.py
  src/database.py
  src/ingest.py
  src/state.py
  src/coordinator.py
  src/runner.py
  src/loom_runner.py
  tests/
  dashboard/
infra/grafana/src/warning_events.py
```
