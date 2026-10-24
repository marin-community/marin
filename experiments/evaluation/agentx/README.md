# Snowball AgentX performance

An Iris CPU coordinator owns each hardware grid. GitHub submits it and observes logs
for a bounded window. Iris owns GPU queueing and whole-attempt retries. The coordinator
processes and durably retains each completed point before submitting the next one.
A separate collection Action delivers completed results to the existing dashboard importer.

**Coordinator failure can require manual recovery.** Its CPU capacity is explicitly
non-preemptible, but it can still fail or be disrupted. For v0, an operator salvages
completed children and submits a fresh run for unfinished measurements. There is no
cross-Action execution state machine or automatic coordinator recovery.

## Fixed measurement

`series.json` pins `snowball011-native-256k-v0`: Snowball v0.1.1, the official
256K corpus's 393 sessions, 262,144 runtime context, supported 900-second windows,
concurrency 8/32 and native KV cache. Each point and replacement attempt starts a new
server/cache. Compilation and CUDA graphs remain enabled. The harness alone owns
session replay, warmup and measurement. No interrupted windows are stitched.

Preparation checks the corpus revision before and after the released loader, verifies
its download SHA256 and checks the pinned tokenizer. No additional whole-session
filtering is applied. The official corpus already removes requests above 256,000 tokens.
`requirements-harness.txt` freezes its isolated environment outside Marin dependencies.

The source checkout resolves Marin's adopted vLLM once per run, including wheel hashes
for both architectures. Manifests retain that release, source GitHub run/attempt/commit,
recipe fingerprint, complete serving/harness argv and native Iris attempt UID. The GPU
command is unchanged by the coordinator change. The recipe requires an uncancelled,
valid 900-second window, required timing coverage and at most 1% profiling request
errors. Warmup records are excluded from the error ratio; diagnostics remain inspectable.

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

GPU queue allowance is 48 hours, task allowance three hours, failure retry budget one
and preemption retry budget three. Coordinators have no automatic retries and a seven-day
bound. Separate source run/attempt names isolate overlapping submissions; Iris owns
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
`submissions/RUN_ID-ATTEMPT-PROFILE.json`, retains existing terminal children and publishes
completed output. It never submits a GPU job or restarts a measurement. Completed valid
siblings keep their original identities. Submit a fresh Action for unfinished points:

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin \
  -f profile=h100 -f concurrency=32 -f target_cluster=cw-rno2a -f priority=batch
```

That is a new source run using its newly frozen adoption; it does not fill an interrupted
window in the old run. A cancelled source is retained for inspection but excluded from
performance ingestion. If the coordinator failed before submitting any child, there is
nothing to salvage and a fresh submission is sufficient.

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
recovery and observation expiration. Validate one normal-duration H100 point through the
coordinator, then processing, repeated import, SQL/API values and retained UI views.
Repeat GB200 measurement if its ARM/device execution changes. Reuse existing unchanged
performance sweeps and label local evidence separately from deployment acceptance.

The companion `infra/inferencex/README.md` covers reader/writer separation, app builds,
Cloud Run/IAP/SQL setup, cache refresh and rollback. Human review/merge, cloud secret/IAM
bootstrap and live IAP/TLS acceptance remain activation steps. Enable ingestion and prove
an end-to-end run before setting `AGENTX_WEEKLY_ENABLED=true`. Existing serving schedules
remain separate. No merge, permanent deployment or weekly activation is part of validation.
