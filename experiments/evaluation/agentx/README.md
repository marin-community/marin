# AgentX throughput benchmarks

GitHub submits one non-spot Iris CPU coordinator per hardware profile. It prepares
AgentX's workload, measures GPU points serially and processes each valid result with
unmodified InferenceX. A separate collection Action delivers artifacts to the
[existing importer](https://github.com/marin-community/marin/pull/10646).
Coordinator loss can require manual recovery.

[series.yaml](series.yaml) fixes Snowball v0.1.1 on eight H100s.
[hero.yaml](hero.yaml) fixes the Hero step144000 BF16 export on sixteen GB200s.
Both use AgentX 1.0.12, all 393 official `062126-256k` sessions, a
262,144-token context, BF16/native KV and one-hour profiling windows.
The pinned preset owns replay, warmup and cache-busting; AgentX seed 42
and the 1% measured-request error ceiling remain explicit. Each point and retry
starts a fresh server through `VllmEnvironment`, with managed compilation caching.
Snowball's native logs confirm CUDA graph capture. Hero's ShortConv implementation
sets `cudagraph_mode=NONE`, including with `enforce_eager=false`.
AgentX collects request and server metrics; Marin's extra metrics
and NCCL collectors default off.

| Profile | Hosts × GPUs | Native TP / DP / EP / PP | Model identity |
| --- | --- | --- | --- |
| `h100` | 1 × 8 H100 | 1 / 8 / 8 / 1 | Snowball v0.1.1 at `59852d6f07261722bbec1edb6ab148317156b58e` |
| `gb200` | 4 × 4 GB200 | 1 / 16 / 16 / 1 | Hero step144000; export manifest SHA256 `b20fdd303eb6ff8993ae5b5d7add14d65787e70981c056fea6de9dea62acfae1` |

Hero's export URI, config digest and immutable tokenizer revision are in
`hero.yaml`. Its 262K HF overrides follow the
[recorded performance-only serving recipe](https://github.com/marin-community/marin/issues/10388).
These measurements do not assess long-context answer quality. Both profiles use
published vLLM wheels. Snowball retains `01911be` with Torch 2.13.0; Hero pins
the qualified `3a054c` release with Torch 2.14.1. Both recipes pin their release
explicitly. Hero's isolated environment uses the published ARM constraints plus
fixed S3-loader dependencies, with repository UV overrides disabled. Each attempt
freezes its wheel, dependency ABI and source identity. Selecting Hero's release
does not change Marin's shared runtime default or Snowball's recipe.

## Snowball calibration

The 2026-10-28 H100 calibration selects resident-session concurrencies
`[64, 128, 256]`: below the bend, at the bend and above it. Each value below is
a separate fresh deployment with a full one-hour profiling window. The links
open the explicit collection Actions, whose artifacts retain the original
submission commit, native attempt, raw trace and processed aggregate.

| Concurrency | Output tokens/s/GPU, each run | Profiling successes / errors, each run |
| --- | --- | --- |
| 8 | [40.14996](https://github.com/marin-community/marin/actions/runs/40633833787) | 1282 / 0 |
| 16 | [68.68441](https://github.com/marin-community/marin/actions/runs/40641669311) | 2296 / 2 |
| 32 | [90.36982](https://github.com/marin-community/marin/actions/runs/40684131564) | 3025 / 1 |
| 64 | [108.29325](https://github.com/marin-community/marin/actions/runs/40684135804) | 3833 / 0 |
| 128 | [134.55371](https://github.com/marin-community/marin/actions/runs/40696525912), [134.95887](https://github.com/marin-community/marin/actions/runs/40728980006), [135.33159](https://github.com/marin-community/marin/actions/runs/40743123733) | 5406 / 0, 5419 / 0, 5446 / 0 |
| 256 | [109.69015](https://github.com/marin-community/marin/actions/runs/40712952732), [111.76721](https://github.com/marin-community/marin/actions/runs/40757943142), [110.18407](https://github.com/marin-community/marin/actions/runs/40773289445) | 4949 / 3, 4969 / 1, 4913 / 1 |

The elbow is the smallest tested load within 5% of the best repeatable mean.
This is our selection heuristic. C128 averages 134.94806 output tokens/s/GPU;
its observed range is 0.58% of that mean. C256 averages 110.54714, with a
1.88% range. C64 and C256 are 19.75% and 18.08% below the C128 mean. These
gaps support reusing C64 and stopping the doubling sweep at C256. Latency
does not constrain this throughput selection; the aggregates retain it.

Across the six C128/C256 runs, recorded serving commands, runtime versions,
parallelism, collectors and workload-cache identity match. Only the AgentX
concurrency changes. AgentX uses seed 42; Snowball's unchanged server uses
its default seed 0 and enables prefix caching. Native logs confirm graph
capture. A later explicit runtime pin changes recipe metadata while selecting
the same `01911be` wheel and Torch version; earlier identities remain intact.

The C128/C256 client samplers cover the profiling windows. The client stayed
below half a CPU core, with no observed CPU throttling. C64 has no client-load
sample. Its throughput gap is much larger than the repeated-point variation.
The one-hour protocol completes different request-length mixtures at different
loads; preserve those distributions when comparing points. Some server counters
reset during profiling. Use request-derived throughput for this selection;
counter-derived rates do not establish the cause of the decline.

The dashboard's frontier filter can hide C256 because it is dominated. Disable
the optimal/frontier filter (`i_optimal=0`) to inspect the complete curve.
Individual repeats remain in run history. Hero's retained C8/C16/C32 results
are provisional; its calibration and weekly-load selection are deferred.

## Normal use

After human review and activation, set the GitHub repository variable
`AGENTX_STATE_ROOT` to a durable object-storage prefix. Use the same value as a
local environment variable for control commands. Manual and Sunday 10:23 UTC
submissions default to batch:

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin \
  -f profile=h100 -f concurrency=all -f target_cluster=cw-rno2a -f priority=batch
```

`concurrency=all` uses that profile's configured weekly loads. A positive integer
selects one calibration load, including values outside those defaults.
H100 uses `cw-rno2a` or `cw-us-east-02a`; GB200 uses `cw-us-east-08a`.
Admission and credentials must be checked before
activation. Weekly execution requires the GitHub repository variable
`AGENTX_WEEKLY_ENABLED=true`; leave it disabled until manual acceptance passes.
Submissions default to Snowball. Scheduled execution selects Snowball's three
weekly loads. Hero calibration and weekly-load selection remain deferred.

Hero also requires `placement_uri`, an object-storage JSON receipt with the
currently spare NVLink domain and its `nodes` list. The required JSON field is
`{"nodes": ["CURRENT_NODE_1", "CURRENT_NODE_2", "CURRENT_NODE_3", "CURRENT_NODE_4"]}`;
replace all four names with the freshly verified nodes. Before each allocation,
inspect live nodes and GPU jobs, including Hero training probes. Save the check
time, cluster, domain, allowed nodes and inventory digest with the receipt.
Historical rack identity does not establish current spare capacity.

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin \
  -f profile=gb200 -f concurrency=16 -f priority=batch \
  -f placement_uri=s3://BUCKET/PREFIX/current-spare-placement.json
```

Iris places the four-host gang within one NVLink domain. Each worker rejects a
node outside the verified list before loading weights. Verify the actual four
admitted nodes against current training placement and cancel the owned job if
that check fails. The deployed controller has no named-domain selector, so a
current operator receipt remains necessary. Submitting `both` without this
receipt fails before either profile starts. Hero's retained `[8, 32]` grid is a
development default. It has no calibrated weekly selection.

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

Edit the reviewed pins in `series.yaml` or `hero.yaml`. Regenerate `requirements-harness.txt`
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
For Hero preparation, add `--series experiments/evaluation/agentx/hero.yaml` and
use `MARIN_TEMP_PREFIX=s3://marin-us-east-08a`.
The run record preserves native TP/DP/EP. Upstream result metadata uses
`TP = native TP × DP`, giving eight H100s or sixteen GB200s for normalization.
The pinned processor's `IS_MULTINODE` input selects separate prefill/decode pools.
Hero passes `IS_MULTINODE=false` for its combined pool and retains that exact
processor output beside the trace. The collector annotates the final output
`is_multinode=true` to describe Hero's physical four-host placement.

Every Hero host records its node, four GPU UUIDs and native attempt UID. The
leader records all four running UIDs before profiling and checks them afterward.
Collection requires the same four successful attempts, four distinct allowed
nodes and sixteen distinct GPUs. A replaced follower invalidates the whole
window; partial windows are never combined across retries.

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
