# Snowball AgentX performance

An Iris CPU coordinator prepares the workload and owns each hardware grid. GitHub submits it and observes logs
for a bounded window. Iris owns GPU queueing and whole-attempt retries. The coordinator
processes and durably retains each completed point before submitting the next one.
A separate collection Action delivers completed results to the existing dashboard importer.

**Coordinator failure can require manual recovery.** Its CPU capacity is explicitly
non-preemptible, but it can still fail or be disrupted. For v0, an operator salvages
completed children and submits a fresh run for unfinished measurements. There is no
cross-Action execution state machine or automatic coordinator recovery.

## Fixed measurement

`series.json` pins `snowball011-native-256k-1h-v1`: Snowball v0.1.1 at
`59852d6f07261722bbec1edb6ab148317156b58e`, AgentX 1.0.12, the official
`062126-256k` corpus's 393 sessions, 262,144 runtime context, one-hour measurement windows,
concurrency 8/32 and native KV cache. Each point and replacement attempt starts a new
server and KV-cache state. Compilation and CUDA graphs remain enabled. The harness alone owns
session replay, warmup and measurement. No interrupted windows are stitched.

`IsolatedCudaVllm` installs the adopted release. `VllmEnvironment` owns server startup,
readiness, native logs and process-group cleanup, using its managed compilation cache.
`collect_marin_metrics` and `collect_nccl` in `series.json` default to false. AgentX's
own measurement and server-metrics collection remain enabled. Attempts record the
effective server argv, selected nonsecret environment, compilation-cache policy and
collectors that actually started.

The pinned `agentx` preset owns agentic replay, 25–75% session starts, ten warmup
requests per lane and first-turn prefix cache-busting. Seed 42 and a 30-minute warmup
grace remain explicit. The 1% profiling-error ceiling is stricter than the preset's
10% default. Label comparisons as Snowball v0.1.1, `062126-256k`, BF16/native KV,
262,144 context and the actual TP/DP/EP/concurrency. The standalone upstream example
uses the full corpus; it is a different workload. Supported shorter windows are
validation runs. Their recipe fingerprints differ from this one-hour series and the
earlier 15-minute points.

The Iris coordinator resolves the source checkout's adopted vLLM once per run, including wheel hashes
for both architectures. Manifests retain that release, source GitHub run/attempt/commit,
recipe fingerprint, complete serving/harness argv and native Iris attempt UID. The GPU
settings retain BF16, native prefix caching and the profile's parallelism. Scheduled
Actions check out Marin main, so adoption is frozen from that source rather than
refreshed during the grid. The recipe requires an uncancelled,
valid one-hour window, required timing coverage and at most 1% profiling request
errors. Warmup records are excluded from the error ratio; diagnostics remain inspectable.

## Prepared workload and data pins

The data pins live in `series.json`; `requirements-harness.txt` freezes the separate
AgentX environment. On a miss, the CPU coordinator downloads the exact corpus and
tokenizer revisions, checks the corpus SHA256 and every source session's context fit,
then runs AgentX's native reconstruction and mmap-cache publication. No additional
whole-session filtering is applied. The official corpus already removes requests above
256,000 tokens. Preparation uses one reconstruction worker and a CPU allocation of
2 cores, 32 GB RAM and 64 GB disk. It finishes before requesting any GPU.

The bundle contains AgentX's native cache and the pinned tokenizer snapshot. Existing
regional storage and `marin_temp_bucket(30, ...)` select `tmp/ttl=30d/agentx-workload`.
The key includes source revisions/digest, AgentX version, preparation version/code,
dependency lock, seed and data settings. Updating vLLM alone reuses the workload.
An existing bundle restores offline, without HF lookups, reconstruction or repeated
Marin context scans. AgentX keeps its native format, metadata and runtime-validity
checks. Preparation and replay use the pinned preset and explicit CLI inputs; inherited
`AIPERF_*` overrides are excluded.

A verified hit republishes the same archive bytes to renew its TTL before GPU queueing.

To prepare or regenerate the pinned data on CPU in rno2a:

```sh
marin-env env MARIN_TEMP_PREFIX=s3://marin-rno2a \
  uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python -m experiments.evaluation.agentx.workload --output /tmp/agentx-preparation-RUN
```

Use a fresh output directory. For east-08a, select its existing regional scratch prefix.
Change the reviewed data or preparation pins to select a new recipe; a missing TTL
pointer or blob rebuilds the existing recipe. Each publication uses a checksum-named
archive and a `KEY/workload.json` pointer. Archive metadata and its byte digest may
change on regeneration while the preparation recipe stays the same. The saved point
manifest freezes the exact publication used by its retries.

Results durably retain source revisions and file digests, preparation code/version and
settings, recipe identity, publication URI and actual bundle digest. The resolved plan
is saved before the first child is submitted. Downloaded sources and reconstructed
bytes may expire; historical rerun capability is not required. An unavailable pinned
source fails preparation and never falls back to a newer revision. Workload data stays
outside the task image.

## Submit, observe and cancel

After human review and merge, set `AGENTX_STATE_ROOT` to a non-TTL prefix, for example
`s3://marin-rno2a/marin/experiments/agentx-performance/v0`. Existing CoreWeave, HF and
GitHub WIF credentials are reused. Sunday 10:23 UTC and manual runs default to **batch**.
Development validation may explicitly select interactive priority and one concurrency.

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin \
  -f profile=h100 -f concurrency=all -f target_cluster=cw-rno2a -f priority=batch
```

Each profile has a separate coordinator on its execution cluster. H100 uses one
8-GPU node in rno2a or east-02a; GB200 uses one four-GPU Grace ARM node in east-08a.
Only roots carry federation cluster pins. Children remain on that peer; current Iris
placement must verify spare GB200 capacity and exclude Hero's allocation before enabling
that profile. Check the GitHub submission identity's admission before using east-02a;
the validated GitHub path uses rno2a.

GPU queue allowance is 48 hours, task allowance five hours, failure retry budget one
and preemption retry budget three. Coordinators have no automatic retries and a seven-day
bound. Task and client deadlines include installation, restore, readiness, warmup,
measurement and drain. Separate source run/attempt names isolate overlapping submissions; Iris owns
admission and priority. Each coordinator submits its own points serially.

The Action's five-minute viewing window per coordinator can end while Iris keeps
running. That expiration does not cancel work. Reattach or explicitly cancel the owned
tree from the same authenticated checkout:

```sh
uv run iris --cluster=marin job logs /runner/agentx-RUN_ID-ATTEMPT-h100 --follow
uv run python -m experiments.evaluation.agentx.control cancel \
  --root "$AGENTX_STATE_ROOT" --run-id RUN_ID --run-attempt ATTEMPT --profile h100
```

Use the exact job ID printed by submission; the user prefix depends on the submitter.
Cancelling a still-running submission Action also cancels its coordinators and children.
After that Action has completed, use the explicit CLI cancellation above. Completed
points are already durable. Run the recovery command below after the tree stops when
you also want durable diagnostics for the interrupted child.

## Manual recovery and retained results

For an exceptional coordinator failure, inspect its root and children with
`iris --cluster=marin job describe JOB_ID`. If it is still live, reattach or explicitly
cancel it. Once its tree is terminal, check out the original source commit in an isolated
worktree and run this recovery command with the original source run and attempt:

```sh
uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python -m experiments.evaluation.agentx.control recover \
  --root "$AGENTX_STATE_ROOT" --run-id RUN_ID --run-attempt ATTEMPT --profile h100
```

Repeat for `gb200` if that coordinator was affected. Recovery can wait up to ten minutes for Iris output finalization. It reads the immutable
`submissions/RUN_ID-ATTEMPT-PROFILE.json` and the resolved `plan.json` under that
submission's retained run prefix, retains existing terminal children and publishes
completed output. It never submits a GPU job or restarts a measurement. Completed valid
siblings keep their original identities. Submit a fresh Action for unfinished points:

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin \
  -f profile=h100 -f concurrency=32 -f target_cluster=cw-rno2a -f priority=batch
```

That is a new source run using its newly frozen adoption; it does not fill an interrupted
window in the old run. A cancelled source is retained for inspection but excluded from
performance ingestion. If the coordinator failed before submitting any child, there is
nothing to salvage and a fresh submission is sufficient. A coordinator lost during
preparation has no resolved plan and no GPU children; recovery publishes no measurements.

Collection verifies Iris uploader hashes, retains all available attempts outside the
seven-day Iris TTL and processes only a final successful, valid window. Native metadata
and the last 20,000 streamed lines are retained when an attempt archive is unavailable.
The pinned upstream processor plus `inferencex.patch` counts TP1/DP8/EP8 as eight GPUs
and TP1/DP4/EP4 as four. No AgentX or InferenceX package enters Marin's dependency graph.

Completed immutable descriptors enter `completed/`. The default-branch collection
schedule runs twice an hour, materializes one undelivered result and uploads an
`agentx-results-*` artifact for the existing ingestion workflow. It never queries Iris,
submits children or advances execution. An `exported/` receipt acknowledges only artifact
delivery. Losing that acknowledgement can repeat delivery; upstream conflict keys and
the trace-cache fix make repeated ingestion safe.

Processed bundle names include their SHA256. Repeating manual recovery can regenerate a
bundle without replacing bytes referenced by an earlier completed-result descriptor.

```sh
# Retry delivery of an exact completed descriptor, including after manual recovery:
gh workflow run agentx-collect.yaml -R marin-community/marin \
  -f result_uri=s3://BUCKET/PREFIX/completed/RESULT.json
```

Keep collection enabled while submitted work can finish. GitHub artifacts retain results
for 90 days. Durable raw archives, receipts and processed hashes follow the organization's
retention policy; preserve a restore reference before deleting them. Monitor coordinator
failures and failed collection/import Actions. A missed result after coordinator loss is
an operator recovery case, not an automatic remediation loop.

## Validation and activation

Run preparation and CPU control checks before GPU allocation. Check native retry/attempt
identity, retained output, explicit cancellation, overlapping sources, coordinator-loss
recovery and observation expiration. Check offline native restoration, rebuilding after
TTL expiry and invalidation when preparation inputs change. Validate one one-hour H100
concurrency-8 point from GitHub through the coordinator, then existing processing and import.
Repeat GB200 measurement if its ARM/device execution changes. Reuse existing unchanged
performance sweeps and label local evidence separately from deployment acceptance.

The companion `infra/inferencex/README.md` covers reader/writer separation, app builds,
Cloud Run/IAP/SQL setup, cache refresh and rollback. Human review/merge, cloud secret/IAM
bootstrap and live IAP/TLS acceptance remain activation steps. Enable ingestion and prove
an end-to-end run before setting `AGENTX_WEEKLY_ENABLED=true`. Existing serving schedules
remain separate. No merge, permanent deployment or weekly activation is part of validation.
