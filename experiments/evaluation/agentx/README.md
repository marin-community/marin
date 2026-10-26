# Snowball AgentX performance

GitHub submits one non-spot Iris CPU coordinator per hardware profile. It prepares
AgentX's workload, measures GPU points serially and processes each valid result with
unmodified InferenceX. A separate collection Action delivers artifacts to the
[existing importer](https://github.com/marin-community/marin/pull/10646).
Coordinator loss can require manual recovery.

[series.yaml](series.yaml) fixes Snowball v0.1.1, AgentX 1.0.12, all 393 official
`062126-256k` sessions, 262,144-token context, BF16/native KV, one-hour windows and
concurrency 8/32. The pinned preset owns replay, warmup and cache-busting; seed 42
and the 1% measured-request error ceiling remain explicit. Each point and retry
starts a fresh server through `VllmEnvironment`, with managed compilation caching
and CUDA graphs. AgentX collects request and server metrics; Marin's extra metrics
and NCCL collectors default off.

## Normal use

After human review and activation, set the GitHub repository variable
`AGENTX_STATE_ROOT` to a durable object-storage prefix. Use the same value as a
local environment variable for control commands. Manual and Sunday 10:23 UTC
submissions default to batch:

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin \
  -f profile=h100 -f concurrency=all -f target_cluster=cw-rno2a -f priority=batch
```

H100 uses one eight-GPU node in `cw-rno2a` or `cw-us-east-02a`; GB200 uses four GPUs
in `cw-us-east-08a`. Verify GB200 capacity lies outside Hero's training allocation
before using that profile. Admission and credentials must be checked before
activation. Weekly execution requires the GitHub repository variable
`AGENTX_WEEKLY_ENABLED=true`; leave it disabled until manual acceptance passes.

Iris allows 48 hours in queue and five hours per GPU attempt, including setup.
Its native budgets allow one failure retry and three preemption retries. CPU
coordinators have a seven-day bound and no retries. The Action views logs for
five minutes per coordinator; viewing expiry leaves Iris running. Use the exact
job ID printed by submission to reattach. Explicit Action cancellation cancels
the owned tree while the Action is running; after it ends, use the control CLI.
`RUN_ID` and `ATTEMPT` below mean the original GitHub submission run and attempt:

```sh
uv run iris --cluster=marin job logs /runner/agentx-RUN_ID-ATTEMPT-h100 --follow
uv run python -m experiments.evaluation.agentx.control cancel \
  --root "$AGENTX_STATE_ROOT" --run-id RUN_ID --run-attempt ATTEMPT --profile h100
```

See [Iris operations](../../../lib/iris/OPS.md#job-management) for job inspection.

## Pins and workload cache

Edit the reviewed pins in `series.yaml`. Regenerate `requirements-harness.txt`
with `uv pip compile --python 3.12` from the selected `agentx-harness` version.
Regenerate processor pins from the reviewed Git package reference:

```sh
uv run --no-sync python -c \
  'from experiments.evaluation.agentx.recipe import load_series, processor_package; print(processor_package(load_series()))' \
  | uv pip compile --python 3.12 -o experiments/evaluation/agentx/requirements-processor.txt -
```

Keep all transitive pins. AgentX and InferenceX run through `uvx` in separate
environments outside Marin's lock.

A cache miss verifies the pinned corpus digest and every session's context fit,
then uses AgentX's native reconstruction, tokenization and mmap format on CPU.
AgentX 1.0.12 has no standalone preparation command; Marin's adapter stays
in `prepare.py`. A hit restores and checks that format offline. It skips Hugging
Face lookups and reconstruction. Updating vLLM alone reuses the workload.

The regional 30-day cache holds native data and tokenizer files. Its key covers the
preparation inputs, code, version and dependency pins. A hit renews the verified
archive before queueing. This CPU command warms the shared regional cache and
writes a local receipt. Use a fresh output directory:

```sh
marin-env env MARIN_TEMP_PREFIX=s3://marin-rno2a \
  uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python -m experiments.evaluation.agentx.workload --output /tmp/agentx-preparation-RUN
```

Use the execution region's scratch prefix elsewhere. Missing pointers or blobs
rebuild on CPU. Pinned source unavailability fails preparation. Each point freezes
its publication URI and digest; results retain lineage after cached bytes expire.
The run record preserves native TP/DP/EP. Upstream result metadata uses
`TP = native TP × DP`, giving eight H100s or four GB200s for normalization.

## Recovery and delivery

If a coordinator fails, first inspect its job and children. Reattach to live work
or explicitly cancel it. Once its tree is terminal, check out the original source
commit in an isolated worktree and recover with its original run and attempt:

```sh
uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python -m experiments.evaluation.agentx.control recover \
  --root "$AGENTX_STATE_ROOT" --run-id RUN_ID --run-attempt ATTEMPT --profile h100
```

Recovery reads the saved submission and resolved plan. It can wait ten minutes
for successful output capture, validates completed children and republishes them
without submitting GPU jobs. Before preparation completes there is nothing to
salvage. Submit a fresh run for unfinished points; cancelled sources are excluded
from ingestion.

Successful measurements, raw AgentX output and lineage become durable. A final
failure retains its configuration, reason, job/attempt identity and up to 100 log
lines when available. Full failed/preempted archives stay under
[native Iris retention](../../../lib/iris/docs/task-outputs.md); interruption
capture is best effort.

Collection runs twice an hour and exports one undelivered completed descriptor.
It does not inspect Iris or submit jobs. A receipt follows successful GitHub
upload; a lost receipt can repeat delivery safely. Processed bundles include
their checksum so recovery preserves earlier descriptors. To redeliver one:

```sh
gh workflow run agentx-collect.yaml -R marin-community/marin \
  -f result_uri=s3://BUCKET/PREFIX/completed/RESULT.json
```

GitHub artifacts last 90 days. Durable results follow the organization's retention
policy. Keep collection enabled while submitted work can finish. Permanent
deployment and weekly activation require review and are separate from validation.
