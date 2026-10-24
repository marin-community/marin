# Snowball AgentX performance

This comparison series measures the adopted Marin vLLM GPU release on one eight-H100 node
or one four-GB200 Grace ARM node. AgentX owns session replay, warmup and measurement.
Each point and each Iris replacement starts a fresh server and native KV cache.
Compilation and CUDA graphs remain enabled. There is no host-cache offload.

## Fixed inputs

`series.json` pins the Snowball checkpoint and tokenizer, official 256K corpus, harness,
processor, resource requests and timing controls. `requirements-harness.txt` freezes the
isolated harness environment. The 393 parent sessions all fit the 262,144-token runtime
context: no extra whole-session filtering is applied. The official 256K corpus already
drops individual requests above its 256,000-token limit. The released loader reads HF
`main`, so preparation checks its revision before and after loading and verifies the
download SHA256; a moved dataset fails closed.

The grid is concurrency 8 and 32. Initial eight-H100 windows showed output throughput
increasing from about 301 to 694 tokens/s, with complete timing coverage at both points.
This provides a low-load and a busier operating point without a tuning sweep. These
exploratory local windows are not relabeled as GitHub results. The latter window had
one empty-content response among 1,037 profiling requests; its diagnostics are retained.

The production recipe explicitly limits profiling request errors to 1%, using AgentX's
`--failed-request-threshold` and a collection-time check. This is stricter than AgentX's
10% scenario default. Warmup records are excluded from this ratio. AgentX must also
stamp the full 900-second window valid and uncancelled, with its required metric coverage.
Errors remain visible in raw request timelines. No interrupted windows are stitched.

The submission snapshot resolves `marin.external_dependencies.VLLM_GPU_RELEASE` once,
including release, source, architecture-specific wheel URLs/hashes and Torch ABI.
Retries use that same checkout and manifest. A new model checkpoint starts a new series.
Serving arguments and harness pins contribute to the recorded recipe fingerprint.

## Run and collect

After human review and merge, configure `AGENTX_STATE_ROOT` to a durable prefix such as
`s3://marin-rno2a/marin/experiments/agentx-performance/v0`. Keep this prefix outside
Iris's seven-day task-output TTL. Existing CoreWeave, HF and WIF credentials are reused.
Manual runs and Sunday 10:23 UTC runs default to **batch** priority. Set
`AGENTX_WEEKLY_ENABLED=true` only after the end-to-end acceptance below succeeds.

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin \
  -f profile=h100 -f concurrency=all -f target_cluster=cw-rno2a -f priority=batch
gh workflow run agentx-collect.yaml -R marin-community/marin
# Recover or recollect a specific source attempt:
gh workflow run agentx-collect.yaml -R marin-community/marin \
  -f run_id=RUN_ID -f run_attempt=1
# Cancel a saved batch and its active native job:
gh workflow run agentx-collect.yaml -R marin-community/marin \
  -f run_id=RUN_ID -f run_attempt=1 -f operation=cancel
```

Development may select interactive priority and one concurrency from the fixed grid.
GB200 uses current Iris placement in east-08a; no historical rack is hard-coded.
Verify current batch placement excludes the Hero allocation before enabling weekly GB200.
Other serving schedules are unchanged.

Submission exits after saving the plan and submitting one job. Iris owns the queue and
replacement attempts: up to 48 hours waiting, three hours per attempt, one failure retry
and three preemption retries. The collector runs twice an hour, has a 20-minute bound,
and advances one point per invocation. Saved stable job names with KEEP recover a lost
submit response. A seven-day batch deadline cancels remaining work. Actions serializes
all control calls and the oldest active plan advances first, so overlapping source runs
do not create concurrent accelerator allocations. Rerunning a source Action creates a
new attempt and plan; cancel the old plan first when abandoning it.

Cancellation is durable before the RPC and is retried until the native job is terminal.
Collector timeouts do not cancel benchmark jobs. Collection uses the explicit federated
task API, verifies the uploader checksum and retains the archive before processing.
It waits at most ten minutes for a missing final archive. Failed/invalid attempts keep
their receipt and native attempt history but cannot produce an importable point.
Live logs include preparation, server and harness output; a pod lost before the output
uploader runs can leave only native metadata and streamed logs.

Every successful collection rebuilds the complete bundle for the original run/attempt.
The plan stays active until GitHub acknowledges the artifact upload. Recollection can
recover a lost upload and upstream database conflict keys make reimports idempotent.
The pinned upstream processor uses the small `inferencex.patch` to count DP GPUs once:
TP1/DP8/EP8 means eight physical GPUs; TP1/DP4/EP4 means four.

GitHub artifacts retain plans and bundles for 90 days. Durable object storage retains
all collected archives, hashes, Iris receipts and source IDs until an operator applies
the organization's retention policy. Never delete the only raw archive. Monitor failed
collection Actions and the active-plan backlog; no resident orchestration service runs.

## Validation and activation

Run CPU preparation before allocating GPUs, then the focused control tests and repository
affected-test runner. Validate an owned queued cancellation and preemption/replacement;
never preempt somebody else's job for a test. Run repeated real GitHub H100 windows,
collect and import through the dashboard's upstream ingestion CLI, then compare raw
counts, timing, throughput and physical GPU normalization with its APIs and rendered UI.
Reimport the same bundle and verify unchanged row counts and trace identities. Check
history across source run IDs without altering their timestamps. Finish H100 acceptance
before a normal-duration four-GB200 run.

Dashboard packaging, read-only credentials, import/cache refresh, deployment and rollback
are described in `infra/inferencex/README.md` in the companion dashboard change. The new
workflow schedules need a human-reviewed merge; no merge or permanent deployment is
part of validation. Disable weekly submission to stop new runs, then cancel active plans;
do not disable collection while a native job is still running.
