# AgentX throughput benchmarks

A GitHub Action submits one Iris CPU coordinator per hardware profile. The
coordinator prepares AgentX data before requesting GPUs, measures the loads
serially, and processes complete valid windows with unmodified InferenceX.
Collection uploads the retained results and can invoke the private
[knotmeter importer](https://github.com/marin-community/marin/pull/10646).
Coordinator loss uses manual recovery.

[series.yaml](series.yaml) pins Snowball v0.1.1 on eight H100s.
[hero.yaml](hero.yaml) pins the step144000 BF16 export on sixteen GB200s,
across four hosts. Both fix AgentX 1.0.12, all 393 official `062126-256k`
sessions, checkpoint/tokenizer/corpus revisions and digests, seed 42, the
262,144-token window and one-hour profiling. AgentX owns native count-driven
warmup, replay, cache busting and validity. The measured-request error ceiling
is 1%. Each load and retry starts a fresh server through `VllmEnvironment`.
Partial windows from different native attempts are never combined.

## Runtime and serving

Submission freezes Marin's `VLLM_GPU_RELEASE` into the saved plan once. All
points and retries use that release even if adoption changes during queueing.
An explicit `vllm_release` tag downloads the published, validated
`marin-vllm-gpu-manifest.json` through the shared release loader. It leaves the
shared selection unchanged. The point fingerprint includes the selected release;
historical result descriptors retain their original identities.

Snowball keeps TP1/DP8/EP8, prefix caching, 8192 batched tokens, 128 sequences
per rank and memory fraction 0.94. Hero uses TP1/PP1/DP16/EP16, BF16/native KV,
prefix caching, logical KV storage, retention interval 0, 2048 batched tokens,
64 sequences per rank, memory fraction 0.95, GPU-local NUMA binding and Triton
experts. Compilation remains enabled. Hero retains its native ShortConv
CUDA-graph guard. The [qualified recipe](https://github.com/marin-community/marin-logbooks/blob/d6f5660563606d7877e7f40b8a70a59703a35f52/logbooks/19-hero-agentx-throughput/owner-integration/hero-logical-kv-recipe.md)
is reference evidence; final validation requires the cache owner's final supported
settings and a published wheel containing [#110](https://github.com/marin-community/vllm/pull/110).
The [final cache handoff](https://github.com/marin-community/marin-logbooks/blob/2ae5ed7be2/logbooks/19-hero-agentx-throughput/owner-integration/hero-logical-kv-review-settings.json)
at source `6a64b548c67ab11e8355c372d3b252108cf8f5c1` enables logical storage
automatically for the qualified path. This benchmark keeps its supported explicit
enable so unavailable native capabilities fail startup. It pins FlashInfer,
KV dtype `auto` and the handoff's compilation configuration, including mode 3
and the accepted ShortConv graph guard.
Marin must adopt a compatible release before weekly Hero use. A manual published
release override can validate it before adoption.

Hero uses the shared released vLLM Router. AgentX sends its native stable
`X-Session-ID` to the DP cache owner, including remote ranks. AgentX scrapes the
worker's `/metrics` directly. Router metrics cannot satisfy collection admission.
Extra Marin and NCCL collectors remain explicit recipe controls, disabled by
default. Artifacts record the Marin commit, expected wheel and digest, vLLM
source, installed dependency versions, effective commands/environment, client
CPU/memory samples, physical GPU UUIDs and native attempt identities.

## Preparation and CPU compatibility

Preparation runs in a separate pinned `uvx` environment. A regional 30-day TTL
cache restores AgentX's native mmap format on a hit and rebuilds on CPU after
expiry. The cache key excludes serving releases and measurement loads. A hit
checks the exact archive and native format, then renews the archive before
queueing. The durable plan keeps lineage and digests after cached bytes expire;
durable copies of source traces and tokenizer snapshots are unnecessary.

```sh
marin-env env MARIN_TEMP_PREFIX=s3://marin-rno2a \
  uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python -m experiments.evaluation.agentx.workload --output /tmp/agentx-preparation-RUN
```

For Hero, add `--series experiments/evaluation/agentx/hero.yaml` and use
`MARIN_TEMP_PREFIX=s3://marin-us-east-08a`. Use a fresh output directory.
Update the reviewed pins beside this command's comments in each YAML file.
Regenerate AgentX requirements with `uv pip compile --python 3.12` from the pinned
`agentx-harness` version. Regenerate processor requirements from `processor_package`
in [recipe.py](recipe.py). Keep all transitive pins. InferenceX remains isolated
from Marin's dependency lock.

The CPU compatibility CI lane runs the real pinned tools on a tiny local Weka
trace and tokenizer. It prepares, publishes/restores, replays against local HTTP,
checks stable session headers, round-trips packed metrics and runs the unmodified
processor with physical eight-/sixteen-GPU normalization:

```sh
uv run --no-sync pytest -n 0 -m integration \
  experiments/evaluation/agentx/test_workload.py::test_pinned_tools_prepare_restore_replay_and_normalize_physical_gpus
```

Only this local smoke test uses AgentX's unsafe override for a short window.
Its output cannot qualify as a benchmark result. Ordinary safe tests also cover
TTL expiry, native retries, interrupted coordination and repeated delivery.

## Manual validation

Keep `AGENTX_WEEKLY_ENABLED` and `INFERENCEX_INGEST_ENABLED` disabled until
acceptance. Run committed, pushed candidate code after review. Pass explicit
state and image inputs for validation without changing shared activation variables.
Use `--ref VALIDATION_BRANCH` when invoking unmerged workflows. If GitHub has
not registered these new workflow names on main, use an isolated branch with a
known dispatch workflow calling these reusable workflows at the same commit.

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin --ref VALIDATION_BRANCH \
  -f state_root=s3://BUCKET/PREFIX/validation -f profile=h100 \
  -f concurrency=all -f target_cluster=cw-rno2a -f priority=interactive
```

`concurrency=all` selects C64/C128/C256. A positive integer selects one explicit
load. H100 supports `cw-rno2a` or `cw-us-east-02a`; Hero uses `cw-us-east-08a`.
Before Hero allocation, follow `coordinate-resources`, [Iris OPS](../../../lib/iris/OPS.md)
and the [accelerator guide](../../../.agents/skills/use-iris/references/dev-accelerators.md).
Refresh a spare-domain receipt from live provider
and Iris inventory; exclude Hero training and check quota across agents. Its
`nodes` list contains the currently allowed nodes. Historical receipts do not
establish current capacity. Verify the four admitted hosts and sixteen UUIDs.

Save a local JSON receipt with `nodes` (the allowed node names), `cluster`,
`nvlink_domain`, `verified_utc` and `inventory_sha256` from that check. Only `nodes`
is consumed by the worker guard; the other fields retain the operator's evidence.
Publish the receipt in the execution region:

```sh
marin-env uv run --no-sync python -c \
  'import json; from experiments.evaluation.agentx.artifacts import write_record; write_record("s3://BUCKET/PREFIX/current-spare-placement.json", json.load(open("placement.json")))'
```

The node list is a post-admission guard. It does not select a rack for Kueue.
Before dispatch, verify that Hero training has a higher protected priority band
than this interactive gang, and that the approved spare domain has capacity.
Check each admission and retry against the current receipt; cancel an allocation
on other nodes. A queue delay can invalidate the earlier inventory. If safe
admission cannot be established, obtain a supported restriction from the Iris
resource owner before launching. Current Iris maps only `pool` and `region` to
node selectors; an arbitrary node/domain constraint is ignored. Keep pool/label
and scheduler changes with that owner.

```sh
gh workflow run agentx-performance.yaml -R marin-community/marin --ref VALIDATION_BRANCH \
  -f state_root=s3://BUCKET/PREFIX/validation -f profile=gb200 \
  -f concurrency=all -f priority=interactive \
  -f placement_uri=s3://BUCKET/PREFIX/current-spare-placement.json \
  -f vllm_release=PUBLISHED_RELEASE_TAG
```

Freeze three comparable full-hour loads per model: below the bend, the smallest
load within roughly 5% of repeatable peak throughput, and a higher healthy load.
Snowball's retained calibration selects C64/C128/C256. Start Hero near that grid
on the final published runtime; short probes can locate the bend. Repeat only
when noise could change the choice. Throughput determines selection; retain
latency without a latency target.

One eight-H100 group and one sixteen-GB200 group may run concurrently. Use
interactive priority for validation and batch for scheduled work. Release idle
allocations and keep competing owned work off measured hosts. Check client CPU
headroom, cache pressure/reuse, effective runtime controls, and native gang
continuity. Each complete result must have successful vLLM worker metrics.

## Collection, import and recovery

Lossless zstandard packing runs after measurement and before Iris output capture.
Collection restores the exact JSON bytes and verifies both checksums before
processing. InferenceX uses `TP = native TP × DP`, yielding eight or sixteen
physical devices. Hero is one combined pool; the raw processor output remains
beside its factual four-host annotation.

Each coordinator's `Retained ...` log line reports its completed descriptor URI
under `state_root/completed/`. Pass that URI as `result_uri`, or omit the input to
select the next undelivered result. Obtain the deployed importer digest from the
[#10646 deployment handoff](https://github.com/marin-community/marin/pull/10646)
before requesting import.

```sh
gh workflow run agentx-collect.yaml -R marin-community/marin --ref VALIDATION_BRANCH \
  -f state_root=s3://BUCKET/PREFIX/validation \
  -f result_uri=s3://BUCKET/PREFIX/completed/RESULT.json \
  -f ingest_image=us-central1-docker.pkg.dev/hai-gcp-models/marin-inferencex/IMAGE@sha256:DIGEST
```

The optional explicit image invokes the hosting owner's authenticated importer,
with its existing writer role and IAP-protected cache refresh. Scheduled imports
use the hosting workflow's activation variable. Repeat explicit delivery/import
and verify unchanged source identities and counts. Inspect all six selected
points, runtime details and request timelines on `https://knotmeter.oa.dev`.
Disable the dashboard's frontier filter (`i_optimal=0`) to see dominated loads.
The final six-point campaign and hosted acceptance remain pending review, the
published Hero wheel, shared router acceptance and the deployed importer handoff.

Iris allows 48 hours in queue, five hours per GPU attempt, one failure retry and
three preemption retries. CPU coordinators are bounded to seven days, without
retries. The Action views each coordinator for five minutes. Observation expiry
leaves Iris running. Explicit Action cancellation cancels the owned tree while
the Action is active; afterward use the control command:

```sh
uv run iris --cluster=marin job logs /runner/agentx-RUN_ID-ATTEMPT-h100 --follow
uv run python -m experiments.evaluation.agentx.control cancel \
  --root "$AGENTX_STATE_ROOT" --run-id RUN_ID --run-attempt ATTEMPT --profile both
```

The submission log includes the Iris link and exact reattach command. After a
coordinator failure, inspect its children. Reattach to live work or explicitly
cancel it. Once the tree is terminal, check out the original source commit in
an isolated worktree and recover its original run and attempt:

```sh
uv run --frozen --package marin-core --extra cpu --no-default-groups \
  python -m experiments.evaluation.agentx.control recover \
  --root "$AGENTX_STATE_ROOT" --run-id RUN_ID --run-attempt ATTEMPT --profile both
```

For stop, observation and recovery, `both` processes the descriptors that exist,
including a run that submitted only one profile. An absent explicit single profile
or a run with neither descriptor reports an error.

Recovery validates completed children and republishes them without submitting
GPUs. Output finalization can take ten minutes. Before preparation completes,
there is nothing to salvage. Submit a fresh run for unfinished loads. Failed or
preempted diagnostics use Iris retention; successful raw outputs and lineage
become durable under `state_root`. GitHub artifacts last 90 days. Keep collection
available while submitted work can finish.

## Activation and stop

After acceptance, configure `AGENTX_STATE_ROOT`, a freshly verified
`AGENTX_GB200_PLACEMENT_URI` and the hosting owner's `INFERENCEX_INGEST_IMAGE`.
The Sunday 10:23 UTC schedule submits both grids at batch priority. Activate only
after Marin selects a compatible Hero release:

```sh
gh variable set AGENTX_WEEKLY_ENABLED -R marin-community/marin --body true
gh variable set INFERENCEX_INGEST_ENABLED -R marin-community/marin --body true
```

Stop new scheduled work and automatic imports with:

```sh
gh variable set AGENTX_WEEKLY_ENABLED -R marin-community/marin --body false
gh variable set INFERENCEX_INGEST_ENABLED -R marin-community/marin --body false
```

Cancel owned Iris coordinators separately and verify GPU cleanup. These variables
are currently unactivated; documentation does not enable them.

## Retained Snowball calibration

The 2026-10-28 H100 calibration on the `01911be` runtime selects resident-session concurrencies
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
Historical metadata fingerprints retain separate points in that curve even
when their effective serving controls match. Individual repeats remain in
run history. Hero's retained C8/C16/C32 results
are provisional; its calibration and weekly-load selection are deferred.

