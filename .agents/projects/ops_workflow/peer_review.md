# Ops Workflow Peer Review

## Review

Claude Sonnet reviewed the implementation diff, design/spec, Grafana provisioning, PostgreSQL repository, Loom adapter, Vue dashboard, Pulumi deployment, and shared Cloud Run access change on 2026-07-21. The review separated spike merge readiness from production enablement.

### Findings and disposition

| Severity | Finding | Disposition |
|---|---|---|
| Critical | A new Grafana fingerprint arriving while the case turn was running attempted to insert a queued turn, but one partial unique index allowed only one queued-or-active turn. The webhook transaction would repeatedly roll back. | Fixed. The schema now permits one queued and one active turn per session through separate partial indexes. Automatic enqueue detects any existing queued work. A real-PostgreSQL regression test holds one turn running, ingests a third group fingerprint, and proves one follow-up remains queued. |
| High | The hardened `coordinator`/`runner`/`turn` recovery contract is not yet connected to the vertical-slice service, so launch-lease recovery, result artifact correlation, and retry lineage are not production-ready. | Accepted as a production gate, not hidden. The design and spec explicitly distinguish the running spike from that hardened contract. Production dispatch stays disabled until it is wired and tested. A running-turn deadline and exact Loom interrupt are now enforced in the spike so a hung agent cannot occupy the global slot forever. |
| High | The transactional repository had no direct test coverage. | Partially fixed for merge. The new PostgreSQL test covers the highest-risk concurrent group update. HMAC/parser, runner lifecycle, Grafana provisioning, and browser flows remain covered. The spec still requires duplicate, stale, resolution, re-fire, archive, and concurrent-claim PostgreSQL cases before production. |
| Medium | Loom session creation can succeed before the database records acknowledgement. Marking the turn failed would free the global slot and permit two external agents. | Made fail-safe. Once an external turn may have started, an acknowledgement failure leaves the database turn in `launching`; it does not release the slot. Durable adoption of that state remains a production recovery gate. |
| Medium | Fire-and-forget dispatch tasks were weakly referenced and could lose exceptions before the coordinator observed them. | Fixed. HTTP handlers only persist work. The supervised coordinator loop performs all dispatch. |
| Medium | The production entrypoint's bare `postgresql://` relies on libpq environment fallback and secret arguments were copied into process argv. | Clarified and hardened. The entrypoint documents libpq fallback. Secret values stay in Secret Manager-backed environment and the CLI reads them by environment-variable name, so HMAC and Loom tokens are not placed in argv. |
| Low | Public ingest inherited a one-instance cap intended for local-state services. | Fixed. Ingest scales from zero to three instances; PostgreSQL remains the concurrency authority. |
| Low | Grafana/ops stack ordering and webhook URL handoff were manual. | Fixed. Grafana reads the ops stack's `ingest_url` through a Pulumi `StackReference`. The ops stack still deploys first because it owns the secret shell and output. |

### Areas reviewed without a finding

- Grafana routes match the intended notification tiers and configure timestamped HMAC with untruncated groups.
- The public and IAP Cloud Run branches are explicit and mutually guarded; existing IAP consumers retain their behavior.
- IAP identity is trusted only on the IAP-gated service, whose Cloud Run invoker is the IAP service agent.
- The Vue dashboard renders agent text through escaped interpolation and uses no raw HTML path.
- The split service credentials match the architecture: public ingest has no Loom or cluster credential; UI has no cluster credential; the agent runtime owns read-only diagnostics.

## Recommendation

After the critical queue-constraint fix and regression test, merge the work as an explicitly gated vertical slice. Do not enable production automatic dispatch yet. Production remains blocked on least-privilege SQL ingestion, full launch/restart reconciliation, broader repository integration tests, rate limiting/metrics, a pinned merged ops skill, and negative mutation tests for the dedicated agent credentials.
