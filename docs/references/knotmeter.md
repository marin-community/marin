# Knotmeter on Marina

Knotmeter browses saved AgentX inference measurements in Marina. It provides a
run inventory, model/hardware/date filters, throughput and first-token latency
plots, dated history, runtime/recipe provenance, request timelines and navigation
to retained corpus conversations. Selectors list imported data. Every measured
load is shown, with output tokens/s per physical GPU as the default metric.

This is a local comparison with the [standalone dashboard draft](https://github.com/marin-community/marin/pull/10646),
at `a411962ba9e63ceb3c27c762da08991710dbaeb8`. It does not select a production approach.
Pricing, power, accuracy evaluation, OperatorX and server-log ingestion are outside
this application. A hardware comparison needs the same model measured on both chips.

## Preview and load

From the repository root, start the actual package release and disposable Postgres
stack. Docker, Node 24, npm and the Marina Python environment are required:

```bash
uv run --package marin-marina marina release infra/marina/apps/knotmeter --local --json
```

Leave this terminal running. It prints a loopback origin and an immutable revision
URL such as `http://127.0.0.1:PORT/a/fe24b7ec-33fa-4edb-8787-2df5b9091a75/v/1/`.
In another terminal, load saved inputs into that explicit origin:

```bash
preview_origin=http://127.0.0.1:PORT
uv run --package marin-marina python -m infra.marina.apps.knotmeter.ingest \
  --url "$preview_origin" \
  --app-id fe24b7ec-33fa-4edb-8787-2df5b9091a75 \
  --bundle /path/to/snowball-collector \
  --bundle /path/to/hero-collector \
  --corpus /path/to/corpus.jsonl \
  --source-attempts /path/to/source-attempts.json
```

`--url` is mandatory. Omitting it cannot target production accidentally. Without
`--source-attempts`, the loader reads the original attempt through `gh api`; saved
responses are a JSON object keyed by run ID. No benchmark runs are submitted.

The retained collectors measure Snowball v0.1.1
(`open-athena/Marin-Snowball-67B-A2B-v0.1.1`) and Hero step144000
(`marin-community/Hero-535B-step144000`): [Snowball 40633833787](https://github.com/marin-community/marin/actions/runs/40633833787)
and [Hero 40723697736](https://github.com/marin-community/marin/actions/runs/40723697736).
Their source measurements are 40625266678/1 and 40705471793/1, both on October 28,
2026. Collector IDs must never replace source IDs or dates. The retained local
corpus snapshot contains 393 upstream conversation structures for
`semianalysisai/cc-traces-weka-062126-256k` at revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`. Keep that JSONL and saved attempt responses
beside the collector bundles, outside the application package. The corpus snapshot
is a separate input; it is not included in the collector bundles.

On a Mac, forward the printed server port, then open the same application path:

```bash
ssh -N -L 18097:127.0.0.1:PORT devbox
# http://localhost:18097/a/fe24b7ec-33fa-4edb-8787-2df5b9091a75/
```

Stop only this preview with Ctrl-C in its first terminal. Marina removes its
Postgres container. Repeating the release and import commands recreates the app
from the saved inputs; a new preview may choose another port. The standalone
preview on port 18096 has independent containers and is unaffected.

For an edit while the preview remains running, inspect its current revision,
then release with the inspected value:

```bash
uv run --package marin-marina marina apps versions fe24b7ec-33fa-4edb-8787-2df5b9091a75 --url "$preview_origin" --json
uv run --package marin-marina marina release infra/marina/apps/knotmeter \
  --url "$preview_origin" --base-version CURRENT_VERSION --allow-dirty --json
```

`--allow-dirty` permits branch source for this local comparison and omits verified
main-branch provenance. Open the newly printed revision URL or the stable UUID URL.

## Import contract

The adapter accepts the runner's collector-bundle layout:

- `agentx-source.json`: original source attempt, point manifests and valid receipts.
- `bmk_*/agg.json`: pinned upstream processor outputs, including metrics and physical GPUs.
- Matching raw directory: `attempt.json`, `aiperf_artifacts/profile_export.jsonl`,
  and `gang.json` for a multihost run.

It validates source/attempt/recipe identity, receipt request counts, the full-hour
window and physical-GPU normalization. For multiple hosts it counts unique devices
across successful hosts. It stores the upstream aggregate and declared settings;
original model, runtime, workload and recipe pins remain visible.

Throughput and latency statistics come directly from `agg.json`. Output throughput
is output tokens/s divided by all allocated physical GPUs, including all hosts.
TTFT and inter-token latency aggregates are in seconds; the UI labels TTFT in
seconds and converts inter-token latency to milliseconds. Statistics refer to
successful profiling requests. The timeline retains warmup and error records too.
Epoch nanoseconds travel as a decimal string plus exact integer offsets, avoiding
JavaScript's loss of precision for absolute nanoseconds. Its waiting-for-first-token
bar includes client/transport time and is not a server-prefill measurement.

A run key is the original run, attempt, hardware profile and concurrency. A content
hash covers the adapted payload, requests and source-file hashes. Loading stages
request batches, then atomically selects the complete import. An unchanged import
is a no-op. Competing writers cannot overwrite a newer selection silently. Old
selected data is removed after replacement; failed staging rows remain for inspection.

The corpus JSONL contract is one object per original conversation:

```json
{"dataset":"owner/dataset","revision":"commit","conversation_id":"original-id","structure":{"nodes":[],"totals":{}}}
```

`structure` is the upstream conversation structure: main turns and subagent groups,
with original model names, source indices, timing and token counts. A replay's
`source_trace_id` selects the conversation; `source_outer_idx` highlights its source
group. Corpus timing is distinct from measured replay timing. This retained corpus
contains token/timing traces, not generated answer transcripts. Missing corpus data
produces an explicit unavailable response. Dataset revisions are immutable identities;
repeat corpus loads insert only missing records.

Raw profiles, server-metric dumps, corpus JSONL and GitHub responses stay outside the
release archive. This comparison stores compact request records and 393 conversation
structures in Postgres. A larger corpus or high-volume campaign should move bulk
payloads to versioned object storage and retain a small serving catalog. That requires
an explicit Marina data-prefix grant; no such infrastructure change is included here.

## Comparison and later hosting

The standalone implementation reuses the broad InferenceX UI, its Node server,
schema and ETL. Its patch touches 40 upstream files, including tests, model catalogs,
provenance and correctness fixes. It already supports richer trace/cache exploration.
Updating it requires rebasing those changes and validating the upstream stack.

The Marina implementation owns a smaller Vue/SVG interface, Python read API, schema
and collector adapter. It reuses Marina's authentication, package lifecycle, SQL
loading, role-scoped database and browser kit. There is no upstream UI patch, copied
Next.js component tree, dedicated Node service or standalone image build. It also
owns its charts and timeline interactions; upstream UI improvements will not arrive
automatically. Current production Marina dependencies are reused without additions.
The app has its own frontend development lockfile.

One measured concurrency setting and one date per model can verify navigation and metrics. They cannot show
a throughput elbow or performance trend. Hero remains the provisional October 28 run
with prefix caching disabled, max batched tokens 512 and max sequences 64.

If Marina is selected, release the reviewed package, explicitly load the corpus and
measurements, and verify production authentication and source/data parity. A named
host needs operator registration and any approved DNS/runtime changes. The runner
owner must connect authenticated loading before enabling weekly automation. Existing
AgentX and processor execution can remain pinned `uvx` commands in the collector;
this loader consumes their saved outputs. No scheduler or runner change is included.
