# Daytona operations for agentic RL

Use this runbook when Harbor trials fail during sandbox setup, snapshot quota
prevents a launch, or completed trials leave provider capacity occupied. Iris
job operations remain in [Iris OPS](../../lib/iris/OPS.md).

## Credentials and account selection

Agentic RL uses the dedicated `DAYTONA_RL_API_KEY`. A generic Daytona key can
authenticate successfully against the wrong account. Do not substitute it for
the RL key. Resolve the credential through the launcher's supported secret
indirection and verify that the coordinator and trial workers receive it.

For launchers that consume only `DAYTONA_API_KEY`, map the RL key immediately
before invoking the launcher:

```bash
: "${DAYTONA_RL_API_KEY:?Load the dedicated RL key through the configured secret provider}"
export DAYTONA_API_KEY="$DAYTONA_RL_API_KEY"
```

Report presence and successful access only; never print key values. Do not copy
private secret files into launch records or source control. The legacy
OpenThoughts-Agent Jupiter launcher also uses `DC_AGENT_SECRET_ENV` to identify
its private secrets file. That is a launcher-specific setting, not an Iris
requirement. See the source Jupiter runbook below for that workflow.

## Snapshot preflight and reclamation

Read the account's current quota and the pinned launcher's snapshot naming and
cleanup behavior before submission. Snapshot quota and live sandbox capacity
are separate limits. A successful GPU allocation proves neither is available.

1. Inventory snapshots in the intended account, including every result page.
   Record exact names/IDs, state, image provenance, and available usage metadata.
   In the Daytona Python SDK, use `client.snapshot.list(page=..., limit=...)`
   through `total_pages`, then `client.snapshot.get(name)` before deletion.
   `organization_id` identifies ownership; `general=True` marks provider-owned
   general snapshots, which are outside account cleanup scope. Resolve the RL
   account through the configured RL credential and compare ownership metadata;
   never substitute another account's key to overcome a quota limit.
2. Determine the snapshots required by the dataset and pinned backend. Count
   distinct environment contents, including fixtures, rather than assuming one
   snapshot per task or counting Dockerfiles alone. Check whether required
   snapshots already exist and are usable.
3. On every snapshot check, classify a snapshot as stale when its last use was
   more than one hour ago (`now_utc - last_used_utc > 3600 seconds`). Use actual
   last-use evidence, not creation time, snapshot state, or an unfamiliar name.
   Exactly one hour does not exceed the threshold. If last-use evidence is
   missing or ambiguous, report that snapshot as unclassified; do not invent a
   timestamp or silently treat it as fresh.
   The SDK exposes the provider timestamp as `Snapshot.last_used_at`; require a
   timezone-aware value and compare it in UTC.
4. Purge **all stale snapshots** in the authorized account on every check,
   whether or not quota is exhausted. Do not stop after freeing the slots needed
   for the next launch. Enumerate exact targets and revalidate last use before
   deletion so a snapshot used since the inventory is not deleted from stale
   evidence. Record the last-use evidence and deletion result for every target.
   `client.snapshot.delete(snapshot)` can return while state is `REMOVING`.
   Confirm disappearance in a subsequent inventory before reporting deletion
   complete; do not repeat the deletion merely because removal is asynchronous.
5. Re-inventory after cleanup. Report remaining stale targets and deletion
   failures rather than claiming cleanup succeeded. Recheck quota, create any
   required snapshots that are now missing, and verify their usable state before
   launching trials. A reference in a future launch configuration alone does not
   exempt an unused snapshot from the one-hour purge rule.

Existing authorization to remove stale snapshots need not be requested again.
The default one-hour rule and full stale-set purge were explicitly supplied by
the operator on 2026-10-07. They apply within the authorized account; they do not
grant access to other accounts or authorize deleting unclassified snapshots.
Report what was removed and whether rebuilding is possible; deletion is not a
recoverable local-cache operation.

Do not assume launcher cleanup enforces this rule. MarinSkyRL campaign revision
`f8d6cf25235c20cb8365069c7561026100231ca7`, in
[`cloud/iris/iris_backend.py`](https://github.com/marin-community/MarinSkyRL/blob/f8d6cf25235c20cb8365069c7561026100231ca7/cloud/iris/iris_backend.py),
uses a two-hour cutoff, selects only names starting with `harbor__`, falls back
to creation time for never-used snapshots, and skips cleanup when the optional
Daytona SDK is unavailable. That implementation does not satisfy the newer
one-hour, all-stale-account-snapshots policy. Run the account-wide check above
with an installed SDK until the selected launcher implements the current rule.

Use the pinned provider SDK or its supported management interface for remote
snapshot operations. Harbor's `cache clean` command cleans local Docker images
and the local Harbor cache; it is not a Daytona snapshot-quota remedy.

Quota exhaustion can fail every trial before scoring. Inspect an actual trial
exception and preserve it separately from model errors. After any backend or
snapshot change, run the campaign's end-to-end smoke before scaling collection.

## Sandbox lifecycle and idle cleanup

Hard termination can orphan in-flight sandboxes. Reap only sandboxes proven idle
and covered by cleanup authority. Correlate provider IDs/labels with the owning
trial and job; a quiet agent can still be waiting for model inference or tool
execution. Check queued consumers before applying an idle-reap policy.

Cancellation during sandbox creation can leave a live provider sandbox even
when Harbor never receives its handle. In one incident, a sandbox remained
`STARTED` after an `EnvironmentStartTimeoutError`, consuming capacity until
reaping. [Harbor PR #109](https://github.com/marin-community/harbor/pull/109)
added a bounded, instance-label-scoped orphan sweep on
that cancellation path. Check the installed revision rather than assuming that
finalized trials imply successful teardown. See [the incident and fix](https://marina.oa.dev/echo/wiki/320).

Sandbox-not-found errors during queued work can indicate an overly aggressive
idle-reap policy. Uniform setup failures point toward image, dependency, account,
or capacity problems; identify the concrete exception before assigning a cause.

## Evidence and progress

Resolve the durable `trace_jobs/` location from the launched configuration or run
metadata. Agentic trials commonly retain their start configuration, final result,
agent/verifier outputs, exceptions, and timing data. Standard dataset-reward RL
has no Harbor trial artifacts; missing `trace_jobs/` is expected in that mode.

Inspect actual rewards, verifier results, stop reasons, and exceptions. All-zero
rewards from the first step can reflect sandbox, data, verifier, or weight-sync
failure. Very short or empty attempts suggest a broken request path or agent
loop. Incoherent output following weight synchronization requires correlation
with reload logs and importance-ratio metrics before concluding learning failed.

Fresh inference heartbeats, trace counts, and a running controller job can
coexist with a stalled trainer. Require advancing trainer phases/steps and
durable artifacts for training progress. Report dispatched, completed, scored,
correct, retained, and accepted-for-training counts separately.

Measure sandbox demand across all concurrent workers and replicas. Per-worker
`n_concurrent_trials` is not an account-wide concurrency limit. Compare admitted
trial demand with model-serving capacity (`max_num_seqs` times inference-engine
count), then inspect scheduler queues, KV-cache use, and request flow. Before
attributing under-feeding to Daytona, check coordinator CPU/GIL pressure and
staleness throttling.

Separate environment setup, agent setup, agent execution, and verifier timing.
Use bounded-sample medians/tails and state the sampling window. Per-request model
API timing isolates inference within agent execution; the remainder includes
tool work. Startup provisioning bursts do not establish steady-state churn.

Aggregate large trial sets near the running workload and transfer bounded
summaries. Compare timezone-aware timestamps in UTC. Resolved configurations,
logs, and trial metadata can contain capability JWTs inside model ingress URLs;
redact those as well as named API keys before publication.

## Source provenance

Transferred from MarinSkyRL revision
`544d5d6f14116a06bde0209352585903133bd618`:

- [CoreWeave operations](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/.agents/ops/coreweave.md):
  provider preflight, idle-only authorized cleanup, trial artifacts, and secrets.
- [Jupiter operations](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/.agents/ops/jupiter/README.md):
  dedicated RL credentials and legacy environment-variable mapping.
- [RL diagnostics](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/.agents/ops/rl-diagnostics.md):
  agentic failure interpretation, progress, concurrency, and timing semantics.

The snapshot-reclamation checklist incorporates the operator's 2026-10-07
instruction to purge every snapshot unused for more than one hour on every
check. This threshold was supplied separately from the source runbooks. The
cancellation incident is additional operational evidence, not cross-account
cleanup authority.
