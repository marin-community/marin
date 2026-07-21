# Ops Workflow Peer Review

The design and specification received two independent reviews before production integration work began.

## Internal production-systems review

The first pass was a no-go for the original production path. It identified path-level IAP exemptions that Cloud Run cannot provide, unscoped shared Loom credentials, conflated sessions and turns, prompts that could bypass the global scheduler, unsupported Loom idempotency claims, an unstructured completion protocol, incomplete case lifecycle rules, and an invented Grafana delivery header.

The revision addressed those findings by splitting public ingest from the IAP UI, requiring dedicated ops Loom tenancy and read-only credentials, separating sessions from turns, scheduling every prompt through Postgres, gating the real runner on missing Loom contracts, using a correlated result artifact, selecting one stable Kubernetes Warning source for V1, and defining reopen/archive/resolution behavior.

The second pass gave a conditional go for the fake-runner/no-credentials slice. Its remaining blockers and dispositions were:

| Finding | Disposition |
|---|---|
| Per-signal timestamps cannot reject a stale complete snapshot containing a previously unseen UID | Added the locked `source_streams` per-cluster high-water row and conflict rules |
| Pending cases were not transactionally materialized into sessions and turns | Pinned one promotion transaction, exact case transition, cancellation on pre-launch resolution, and unique replacement behavior |
| Disabling IAP does not grant public Cloud Run invocation | Required separate IAP and invoker-member options, including `allUsers` only for the HMAC ingest service |
| Runtime timeout was not durable | Added `deadline_at` computed from database time |
| Operation idempotency and retry lineage were incomplete | Added `operation_requests`, immutable retry rows, and `retry_of` |
| The agent could not know Loom's turn number before dispatch | Removed that value from the agent-authored result and correlate it in the coordinator |

## Claude production review

Claude reviewed the revised artifact and current first-slice code. It agreed that production integration should remain gated and identified three implementation prerequisites: the SQL migration, session/turn runner contract, and scheduler/lease reconciliation primitives. The implementation now includes all three, plus behavior tests for source high-water ordering, exact-turn correlation, lost acknowledgement recovery, the all-turn global slot, and durable timeout interruption.

Claude also recommended a small raw-SQL migration layer rather than a query-builder abstraction, strict evidence/prompt bounds, and idempotent fake-runner behavior. Those recommendations are reflected in `infra/ops`.

## Review Verdict

The reviewed boundary is approved for code review: schema, authenticated snapshot validation, pure signal/case transitions, immutable turn transitions, a contract fake, and coordinator boundaries. Production source wiring, public deployment, real Loom dispatch, and cluster credentials remain disabled until the rollout gates in the specification pass.
