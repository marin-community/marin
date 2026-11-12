# Knotmeter on Marina

Knotmeter browses saved AgentX inference measurements through Marina. Its React
frontend adapts InferenceX charts, diagnostics, request timelines and corpus
flamegraphs. Marina owns authentication, package releases, API routing and the
application's Postgres role and schema.

## Views and controls

| View | Controls and data |
| --- | --- |
| Performance | Model, hardware and UTC date; input/output/total throughput per physical GPU; TTFT, interactivity, E2E and normalized axes; percentiles; Horace He accounting; measured Pareto curves and Perf Ruler; numeric table; labels, legend, zoom, linear/log scales; PNG, CSV and shared selections. |
| Runs and submissions | Search, sort, pagination, expandable configuration, physical topology and source/runtime/recipe pins; selected-run overlays; weekly and cumulative imported submissions. |
| History | Throughput at matched P90 interactivity, dated source observations and record-improvement waterfall. Model, corpus and prepared workload identities remain separate. |
| Run diagnostics | Token distributions and in-flight estimates; rolling/cumulative latency and interactivity; per-engine KV, queues, completed requests, prefix hits, input/decode throughput and prompt-token sources. |
| Request timeline | Warmup/profiling, conversation/worker grouping, replay and subagent hierarchy, queue waits, idle/cursor counts, zoom/pan and virtual scrolling. A source link opens the exact nested corpus request; return restores the timeline view. |
| Cache and aggregates | Measured HBM/host/not-reused shares, corpus theoretical ceiling, profiling totals and token/cache distributions across configurations. |
| Corpus | Dataset/model/cache/subagent summaries and distributions; searchable, sortable, paginated conversations; original timing/token structure and expandable token flamegraphs. |

Selectors list imported data. Performance defaults to Snowball when present,
output tokens/s/GPU versus P90 TTFT, and the latest available UTC date. Choose
**All dates** or select runs in the inventory for comparisons. Shared performance
links freeze the selected run identities and effective date. Labels, high contrast
and Perf Ruler are visible controls.

A single configuration is shown as a point. Curves require measured loads from the
same submission and configuration identity, or matching private calibration controls.
Matched history and rulers also require
the same saved workload key, covering subset, context and tokenizer inputs. Reimport
older bundles without that identity to enable matched comparisons. History uses the latest submission per
UTC date and the pinned upstream Pareto/Hermite interpolation. It does not extrapolate.
A submission is an original GitHub run/attempt; it can contain several loads
(concurrency settings). Curves group loads within that attempt by hardware, framework,
precision, parallelism/offload settings and model/corpus revisions. Dates use the
original GitHub attempt’s `created_at` in UTC. Private calibrations use their native
benchmark start time. Neither uses collection/import time. Measurement selectors
and ruler scopes separate 15-minute calibrations from one-hour baselines.

Interactivity is the inverse of full-response inter-token latency, in tokens/s/user.
E2E latency includes the whole request; E2E-normalized interactivity also includes
first-token delay per generated token. Perf Ruler shows a throughput ratio at the
same selected horizontal-axis value.

One dated observation has no historical trend. Perf Ruler compares overlapping
measured curve ranges and keeps model/corpus revisions and precision compatible.

## Start and load a local preview

From the repository root, start a package release and disposable Postgres stack.
Docker, Node 24, npm, uv and the Marina Python environment are required. Compressed
calibration server exports also require `zstd`:

```bash
uv run --package marin-marina marina release infra/marina/apps/knotmeter --local --json
```

Leave this terminal running. It prints a loopback origin and a revision URL such as
`http://127.0.0.1:PORT/a/fe24b7ec-33fa-4edb-8787-2df5b9091a75/v/1/`.
In another terminal, load saved inputs into that explicit origin:

```bash
preview_origin=http://127.0.0.1:PORT
uv run --package marin-marina python -m infra.marina.apps.knotmeter.ingest \
  --url "$preview_origin" \
  --app-id fe24b7ec-33fa-4edb-8787-2df5b9091a75 \
  --bundle /path/to/snowball-collector \
  --bundle /path/to/hero-collector \
  --corpus /path/to/corpus.jsonl \
  --source-attempts /path/to/source-attempts.json \
  --series-cache /path/to/local-series-cache
```

`--url` and `--series-cache` are required. Store the cache outside the app package.
The build creates `.build/reduce-series.mjs`; loading invokes it through an isolated
`uv run --script` parser pinned to `ijson==3.6.0`. Large raw server exports are streamed.
The cache key covers the export, profile and reducer/parser hashes. Node is needed
for loading, not for serving. Without `--source-attempts`, the loader reads original
attempts with `gh api`; a saved response file is a JSON object keyed by run ID.

Load retained private calibrations into the same explicit preview:

```bash
uv run --package marin-marina python -m infra.marina.apps.knotmeter.ingest \
  --url "$preview_origin" \
  --app-id fe24b7ec-33fa-4edb-8787-2df5b9091a75 \
  --calibration /path/to/hero-bf16-tp-calibration \
  --series-cache /path/to/local-series-cache
```

Add `--calibration-job /romain/EXACT_JOB` to import one observation first. The loader
reads `curve-summary.json` and `window-effects.json`, then checks native validity,
900-second coverage, attempt identity, topology, request accounting and saved
aggregate normalization. Invalid screens are excluded. Points load sequentially;
adjacent server exports can be JSON or `.json.zst`. Original artifacts stay unchanged.
Calibration diagnostics select the receipt's worker metrics endpoint before
reduction because the router mirrors the same engine counters. The cache key
includes that endpoint. Profiling token totals sum the saved per-engine worker
aggregates; the original combined router/worker aggregates remain in provenance.
Server output can include tokens from unfinished requests, while benchmark
throughput uses completed client requests.

The inventory's calibration link selects all dates and includes dominated points.
For the retained Hero data, its chart fragment is
`#/chart?model=marin-community%2FHero-535B-step144000&window=900&date=all&optimal=0`.
Inspect a series to see every load, date and repeat, then open its diagnostics.

The preview used artifacts `agentx-results-40625266678-1` and
`agentx-results-40705471793-1`. Download and unpack them with:

```bash
gh run download 40633833787 --repo marin-community/marin \
  --name agentx-results-40625266678-1 --dir /path/to/snowball-collector
gh run download 40723697736 --repo marin-community/marin \
  --name agentx-results-40705471793-1 --dir /path/to/hero-collector
```

The retained corpus JSONL was exported from the standalone dashboard’s loaded
original conversation structures. It must be supplied separately in the contract
below; there is no automatic corpus download in this loader.

Keep the collector bundles, corpus and cache outside the application package. The
corpus is a separate input, absent from the collector bundles. No benchmark jobs
are submitted by these commands.

On a Mac, forward the printed server port over the existing SSH route:

```bash
ssh -N -L 18097:127.0.0.1:PORT devbox-tailnet
# http://localhost:18097/a/fe24b7ec-33fa-4edb-8787-2df5b9091a75/
```

Ctrl-C in the first terminal stops this preview and removes its Postgres container.
Repeat the release and import commands to recreate it. A restart may choose a new
port; update the forward. The standalone preview on 18096 has independent containers.

For an edit without restarting, inspect the current revision and use its value:

```bash
uv run --package marin-marina marina apps versions fe24b7ec-33fa-4edb-8787-2df5b9091a75 --url "$preview_origin" --json
uv run --package marin-marina marina release infra/marina/apps/knotmeter \
  --url "$preview_origin" --base-version CURRENT_VERSION --allow-dirty --json
```

`--allow-dirty` permits local branch source and omits verified main-branch provenance.
Open the printed revision URL or the stable UUID URL without `/v/NUMBER/`.

## Data contracts and definitions

The loader accepts `agentx-source.json`, `bmk_*/agg.json` from the pinned processor,
and matching raw `attempt.json` and `aiperf_artifacts/profile_export.jsonl` files.
Multihost runs also require `gang.json`. An adjacent `server_metrics_export.json`
supplies collected server series when present. It validates original attempt,
recipe, receipts, request counts, the full-hour window and physical-GPU normalization.
Multihost normalization counts unique devices across successful hosts.

Private calibration inputs retain the original `private-calibration` source,
Iris job/attempt, runtime and workload receipts, processor command, artifact archive
and file hashes. Their keys use the native attempt UID. Curve identities cover
the actual server command/environment, runtime pin, source files, workload,
topology and duration. Listening addresses/ports, placement, startup timeout and
local cache paths are omitted. Independent jobs and dates can therefore contribute loads to
one curve. Repeats stay separate; no means replace them. History keeps each profile
and date separate and uses the existing frontier interpolation.

The processor's aggregate `tp` is its physical-GPU normalization input. Private
calibrations display actual TP/DP/EP from the native receipt and retain the original
aggregate. Tail cancellation counts come from hash-checked native phase logs.
When a calibration ends, the harness cancels unfinished requests. These tail
cancellations are absent from the request exports, so their individual
timelines are unavailable. They are distinct from profiling errors. Short windows
can change throughput and rankings; use one-hour measurements for final baselines
and operating-load decisions.

The saved bundles were produced with AgentX `1.0.12` and the isolated processor
revision `80dff120d27fd6092210f4546b8faa5e6c9723ac`. This producer pin is separate
from the frontend/reducer source pin below.

Aggregate throughput and latency come from `agg.json`. Throughput is divided by
all allocated physical GPUs, across all hosts. Aggregate latencies are in seconds;
request TTFT/TPOT values are in milliseconds. Successful profiling requests supply
the benchmark aggregates; the timeline also retains warmup and error records.
Warmup is capped at one output token, so its decode/interactivity can be unavailable.

Horace He accounting subtracts corpus-cacheable input from input/total throughput;
output throughput is unchanged. Measured HBM/host cache shares instead use the saved
processor's profiling counter deltas. Per-interval charts retain the upstream
server-series calculation. These definitions can differ: the retained Snowball
profiling share is 70.342%, while the standalone UI's chart-window calculation is
67.697%. Corpus cacheability is a theoretical ceiling, not measured server reuse.
In-flight views estimate tokens from request lifetimes and shared conversation
prefixes; they are distinct from measured KV gauges. Missing measurements remain
unavailable, including percentile markers for absent series.

Epoch nanoseconds travel as a decimal string plus exact integer offsets. Requests
retain worker, replay, ancestry, credit/start/ack/end timing and original outer/inner
corpus indices. The timeline's waiting-for-first-token bar includes transport/client
time; it is not a server-prefill measurement. Source timing in the corpus is separate
from measured replay timing.

A run key combines original run, attempt, hardware profile and concurrency. Content
hashes cover metadata, requests and source files. Loading stages requests and server
series, then atomically selects a complete import. Reads of a run's metadata,
requests and series use one database snapshot. Unchanged imports are no-ops;
competing loaders cannot silently overwrite a newer selection. Failed staging data
remain available for inspection.

The corpus JSONL contains one object per original conversation:

```json
{"dataset":"owner/dataset","revision":"commit","conversation_id":"original-id","structure":{"blockSize":64,"nodes":[],"totals":{"in":0,"out":0,"cached":0,"uncached":0,"numTurns":0,"numSubagentGroups":0}}}
```

`structure` retains upstream main turns and subagent groups, model names, source
indices, timing and token counts. This is a token/timing corpus, not answer text.
Repeated loads preserve immutable structures and refresh derived summaries.
Load the full dataset revision for its summary and distributions. Missing conversations
return an unavailable response.

## Retained preview inputs and ownership

The authentic inputs are [Snowball collector 40633833787](https://github.com/marin-community/marin/actions/runs/40633833787)
and [Hero collector 40723697736](https://github.com/marin-community/marin/actions/runs/40723697736).
Their original measurements are 40625266678/1 and 40705471793/1 on October 28, 2026.
Snowball v0.1.1 uses 8 H100s; Hero step144000 uses 16 GB200s across four hosts. These
are different models. Hero is provisional: prefix caching disabled, max batched
tokens 512 and max sequences 64. The preview labels those settings.

The retained corpus has 393 conversations from
`semianalysisai/cc-traces-weka-062126-256k` at
`8fecd2fc56694469f758f0afbbb6335ad3043740`. The one-hour results each have one measured
load/date. The [retained Hero calibration](https://echo.oa.dev/wiki/945) adds 13 valid
900-second observations: TP1/DP16/EP16 on 16 GB200s (four points), TP2/DP8/EP16
with NCCL on 16 GB200s (five), and TP6/DP2/EP12 on 12 GB200s (four). These include
three repeated loads and 11,673 successful profiling requests with zero profiling
errors. Three preliminary 180-second trials failed native validity checks and
are excluded. Separate synthetic
browser fixtures exercise larger histories, sweeps,
rulers and hierarchy; they are never imported as authentic preview results.

Knotmeter adapts InferenceX revision `315128ce4e49c8a1ec1aa67eafa8f2de26e4c1f9`;
source and GPL v3 attribution ship with the browser package. See the app's
`THIRD_PARTY.md`. It owns the adapted components, API contracts and ingestion.
Upstream UI improvements require deliberate integration. The app adds local React,
D3, Radix controls, Tailwind and build/test dependencies, with its own lockfile.
There is no Next.js runtime, Node service, shared Python dependency change or upstream
patch overlay. The [standalone draft](https://github.com/marin-community/marin/pull/10646)
retains the upstream application, database/ETL and its 40-file patch overlay.

Production use still requires a reviewed Marina release, authenticated loading,
production authentication checks and data parity verification. A named host needs
operator registration and approved DNS/runtime work. The runner owner must connect
loading before weekly automation is enabled. This app does not activate scheduled
workflows or change cloud resources. Pricing, power, accuracy suites, OperatorX
and new server-log ingestion remain outside scope.
