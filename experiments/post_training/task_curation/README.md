# RL data curation

[Dataset declarations](datasets/README.md) feed TaskSpec conversion and the RL
Data Atlas through [all_sources()](sources.py). See the
[reference](../../../docs/references/task-curation.md) for source fields, grading,
reviewed SAMPLE/FULL campaigns and artifact outputs.

## Local conversion loop

From the repository root:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver --mode quick --run \
  --source tasktrove-calendar --source tasktrove-math_prism \
  --output-root /tmp/task-curation-pass-1
```

QUICK uses local Zephyr, downloads pinned inputs once, and converts every selected
row. It preserves converter rejections and known reviewed defects; it skips
review, grader execution, resource admission and deduplication. Outputs are
`normalize/*.parquet`, `manifest.json` and `campaign.json`, with no admitted
`final/` view. Failed sources are recorded while the rest continue.

Select `--mode quick`, `sample` or `full`, or resume with `--corpus` as described
below. Omit `--run` to print a plan without downloading, converting or starting
workers. QUICK preserves named sources in request order, ignoring repeated names.
SAMPLE/FULL select sources in catalog order. Omitting `--source` selects every
runnable source. Local QUICK needs no review model, controller, worker image or
coordinator settings.

Repeat `--source` for a cohort and use a fresh output root each pass. Downloads
are cached under `~/.cache/marin` (`--cache-dir` overrides this).
`--input-root PATH` uses already staged files from the declared revision;
`--input NAME PATH` overrides an auxiliary input. `--output-root` selects QUICK
outputs; reviewed local runs use `--report-path` for their report location.

For a local fixture, replace `--input-root` with
`--input-file LOGICAL_PATH LOCAL_FILE`, repeated as needed. The logical path must
match the source declaration. These files become the selected primary inputs;
the manifest records their physical paths and SHA-256 while preserving logical
row IDs. Auxiliary inputs still use overrides or the download cache.

### Reviewed local loop on real sources

Use `--cache-dir` with SAMPLE or FULL to run the same reviewed pipelines on a
local Zephyr pool. Pinned downloads and source artifacts live in that directory;
unchanged graphs are reused on the next invocation. SAMPLE processes each
source's panel, while FULL expands sources accepted by its panel review.

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver \
  --mode sample --cache-dir ~/.cache/marin/rl-curation \
  --source aime24 --source svamp --quality-sample-size 5 \
  --assess --assessment-sample-size 3 --excluded-sample-size 2 \
  --max-workers 2 --concurrent-sources 1 --review-concurrency 1 \
  --review-mode chat --model-revision YOUR_GLM_REVISION \
  --base-url YOUR_GLM_ENDPOINT --run
```

Local reviewed runs need no worker image or controller for in-process graders.
The default report is `CACHE_DIR/campaigns/campaign.json`, with its retained run
catalog beside it. `--report-path` chooses another report
location. Review requests are cached under `CACHE_DIR/review-cache` unless
`--review-cache` selects another location. `GLM_BULK_TOKEN` supplies credentials.
Package-backed and image graders retain their normal environment prerequisites.
`--image-backend` and `--controller-url` can explicitly select a remote image
grader while curation runs locally; worker-image and coordinator flags apply only
to Iris execution.
Use `--input-root`, `--input-file` or `--input` only when deliberately replacing
the pinned inputs; the local rollout can use real sources without overrides.
Keep the same `--quality-sample-size` and `--seed` when moving from SAMPLE to
FULL to reuse identical panel review requests. Changing either selects a new
panel; the plan prints both values before execution.

FULL appends assessment automatically; SAMPLE uses `--assess`. Use
`--skip-assessment` to record a curation-only run. The assessment scans admitted
IDs, samples excluded rows by primary reason, and prepares bounded packets from
the selected TaskSpecs and curation sidecars. GLM gives each packet one judgment.
The report separates admitted-task opinions from excluded-row diagnostics and
executed controls. Assessment never changes row admission.

Incomplete quality or control outcomes are retained as curation artifacts so
assessment can diagnose them. The campaign still fails after saving its reports.
Use a new `--curation-attempt` to rerun incomplete controls under a new source key;
pinned downloads and exact successful model requests remain cached. An assessment
failure retries only the failed grading stage.

The default assessment sample is 100 admitted tasks and up to 20 excluded rows.
`--assessment-rubric FILE` adds criteria to the shared rubric. Changing rubric
or model configuration reuses prepared packets; source changes use the ordinary
upstream pipeline key. Implementation changes require explicit stage version
bumps. Python source files do not enter assessment identity.

The grading pass has two artifact stages: Zephyr prepares bounded packets from
selected TaskSpecs and curation sidecars, then GLM emits one structured judgment
per packet. Publication joins these outputs by task ID, checks coverage and
archives packets and request/response traces in FineStore. Atlas loads the typed
records and packets into Postgres; SQL computes cohort labels and counts.

Malformed or unavailable responses use finite retries. Remaining failures retain
partial grades and fail the stage. Repeated runs reuse successful exact requests.
Changing the rubric or `--assessment-model-revision` reuses curation and prepared
packets. Ordinary source changes invalidate their downstream artifacts; shared
stage implementation changes require an explicit version bump.

Chat transport allows up to 64 requests per worker process, selected with
`--review-concurrency`. Eight concurrently grading sources at 64 requests each
can issue up to 512 requests in flight. Local reviewers share one process limit;
remote worker processes have separate limits. Byte groups do not impose a
completion barrier. Each task still has its own request, judgment and trace;
`--review-mode batch` selects the provider's separate asynchronous batch API.
Tune concurrency from measured throughput and errors for the chosen endpoint.

The cohort label is unknown for an empty sample, unavailable judgments or a
missing TaskSpec. Otherwise bad takes precedence over some_issues, unknown,
and good. Read the verdict counts alongside this conservative label. Missing
prior reviews are optional context; executed control coverage is reported
separately. Excluded counts cover processed non-admitted verification rows;
unprocessed input rows stay in the curation summary.

Each terminal campaign publishes the complete inventory through the typed
`CurationCatalog` reader. Assessments, populations, items and packet references
retain their executed pipeline key. A failed attempt under a changed key has no
current grade; an exact-key retry can retain a previously completed grade.
Sources without pipelines remain inventory entries with a skip reason.
Publication archives selected evidence in the data's storage region. Atlas data
updates use the [explicit loader](../../../docs/references/rl-data-atlas.md#updating-atlas).

### Spiral rollout

Start with AIME24 and SVAMP and a small SAMPLE panel in the local cache above.
Inspect task packets, rejected rows, provider failures and runtime. Repeat the
same command to check cache reuse; change only the grading rubric to check that
curation stays cached. Fix failures before increasing either the source set or
sample size. Move to FULL on these sources, then expand by source family and
measure throughput, retries, packet sizes and publication time at each step.
Run large remote campaigns in the data's storage region. Load Atlas after the
campaign and catalog are complete.

### Harbor compatibility view

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.tasktrove.export \
  --input-root /tmp/task-curation-pass-1/tasktrove-calendar \
  --output-root /tmp/curation-harbor/calendar \
  --grader-image '<registry/image>@sha256:<digest>'
```

The exporter writes the legacy 12-column `tasks.parquet` and `manifest.json`,
preserving source, original archive path, task ID and rejection accounting.
Hidden tests stay private; oracles go in `solution_binary`. Unsupported contracts
and reference leaks become explicit lowering rejections.

TaskTrove tasks retain the actor's shared Harbor environment and can omit
`--grader-image`. Other supported tasks use a separate verifier image.
Export restores source actor recipes and bundles current Verifyit; use
`--verifyit-package-root` to select its source package.

Export builds no images and runs no graders. Comparisons cover the emitted
recipes, files and execution settings; image tags and build-time downloads
retain the source's reproducibility limits.
The exporter does not install Harbor or validate every generated `task.toml`
against Harbor's schema. The manifest records this limit. After extracting a
`task.toml` from `task_binary`, check it with the pinned external Harbor runtime:

```bash
uv run --project config/external/harbor --frozen python -c \
  'from pathlib import Path; from harbor_config.models.task.config import TaskConfig; TaskConfig.model_validate_toml(Path("task.toml").read_text())'
```

[`harbor_export_step`](tasktrove/export.py) binds this export to a normalized
artifact. It records a hash of verifier files and configuration in `manifest.json`.

Artifact reuse follows explicit versions. Bump the source recipe version when
conversion or bundled grader code changes, the affected pipeline stage revision
for shared processing changes, and the export version for Harbor lowering or
bundled verifier changes. Download identities still follow pinned source bytes.

The nl2bash RL smoke is unsupported. Its export uses shared grading and Docker
build recipes. The grader runs on the same machine as the agent.
SkyRL Harbor tasks require prebuilt images and a separate verifier machine.

The [TaskSession validation recipe](../task_sessions/validation.py) uses tasks
that count `cat` words and two-turn math tasks. It does not exercise Harbor tasks.

### Full TaskTrove content comparison

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.tasktrove.compare \
  --normalized-root /tmp/task-curation-pass-1 \
  --source tasktrove-calendar --source tasktrove-math_prism \
  --baseline-repository . \
  --baseline-revision 61bb85cc5231d8ac9344696ef51766257940538d \
  --grader-image '<registry/image>@sha256:<digest>' \
  --output-root /tmp/tasktrove-content-report
```

Omit `--source` to require every registry pipeline reading
`open-thoughts/TaskTrove`. Repeat `--normalized-root` to combine runs; later roots
win. QUICK manifests locate the original input bytes; `--raw-root` supplies a
staging root for older manifests. The output directory must be new.

The isolated reference loads the pinned legacy converter and libraries from Git.
It compares every source/path, instructions, task and oracle files, permissions,
recipes, parsed settings and row metadata. Legacy static checks are recorded
separately from conversion outcomes. It skips grading, review, deduplication and
release caps; archive compression, ownership, ordering and timestamps are ignored.

Each source gets `summary.json`, `provenance.json` and `parity.sqlite`; `run.json`
records overall provenance. Review differences and record accepted changes and
unresolved gaps separately. Operational
failures make the command exit unsuccessfully after trying the remaining sources.

For readable file diffs, repeat with one `--source` and
`--inspect-path '<original archive path>'` in a new output directory. `task.diff`
includes file metadata and text changes, capped at 64 KiB per side per file with
explicit truncation; binary files retain hashes. Full runs discard archive bytes.

## Task and curriculum embeddings

The curation driver embeds public task instructions and curriculum entries with
Harrier, computes exact cosine matches, and exports an Atlas release through
Zephyr. Both sides use the same retrieval rubric. Optional model rewrites support
bounded comparisons. The curriculum has no separate embedding CLI or database.

Start with a local QUICK conversion and a seeded sample:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' \
  --with-editable './lib/marin[cpu]' python \
  -m experiments.post_training.task_curation.driver \
  --mode quick --source aime24 \
  --output-root /path/to/quick --cache-dir /path/to/persistent/cache \
  --embed coverage --embedding-sample-size 20
```

Add `--run` to execute. AIME24 downloads its pinned inputs automatically. Direct
embedding needs no inference-service credential.
`--embed STAGE` runs that embedding stage and its dependencies after conversion
or curation. QUICK embeds converted tasks; SAMPLE/FULL embeds admitted tasks.
Stages are `inputs`, `descriptions`, `task_embeddings`, `curriculum_embeddings`,
`assignments` and `publication`. Omit `--embedding-sample-size` for all selected
tasks. `--catalog-root` selects an immutable directory containing `curriculum.json`;
otherwise the graph uses `TASK_CURRICULUM`.

`--cache-dir` selects local workers and the local artifact cache. `--review-cache`
holds GLM request/description caches and Harrier vectors in its `descriptions` and
`vectors` subdirectories; it defaults to `<cache-dir>/review-cache` locally. Keep
this inference cache when changing worker counts or resuming a corpus. Remote
artifact outputs use the Iris job’s regional Marin prefix.

Direct embedding uses instructions and the public answer/tool contract. It opens
no task files, private verifier resources or solutions. Inputs are bounded to
8 KiB of UTF-8, with beginning/middle/end excerpts and omission markers when
needed. Public-content hashes cover the complete input before sampling, and
identical rendered texts share vectors. Harrier accepts up to 16,384 tokens for
direct inputs; this is independent of task instruction length.

For a rewrite comparison, select `--representation rewrite --model MODEL
--model-revision REVISION --base-url ENDPOINT` and supply `GLM_BULK_TOKEN`.
The model receives at most 32 KiB of public input with the same excerpt strategy.
It returns roughly 15–40 words naming the required knowledge or environment and
core operation. Harrier's rewrite input limit is 512 tokens. A response carries
no task hash: the transport identifies its request. Failed rewrites exhaust three
attempts, retain a `skipped` row and increment `semantics/skipped_embeddings`;
the pass fails only above a 5% failure rate across distinct public inputs.
`insufficient_context` separately records that the shown input did not establish
the requested operation. Skipped rows remain in Atlas denominators with their
failure reason; no vector or match is guessed.

Tasks and curriculum anchors share the pinned Harrier checkpoint, embedding
instruction, tokenizer, bfloat16 CPU inference, last-token pooling and L2
normalization. FineStore stores vectors by rendered-text/specification hash in
1,024 fixed partitions, independent of worker count. Zephyr routes each embedding
partition to one writer, avoiding shared-manifest contention. Reads and writes
are batched. A curriculum edit reuses task descriptions and vectors.
Assignment workers load the curriculum matrix once per shard and score batches
streamed from the task-vector Parquet outputs. Scoring uses the embedding keys
deduplicated by the embedding pass. Zephyr joins those scores to descriptions and
task instances, producing assignment records for the Atlas export. Alignment
does not read the inference cache.
Alignment and export shuffles use `--max-workers` for their output partition
count, keeping the number of output files independent of the source-file count.

All stages use the driver's shared `--max-workers`, `--worker-cpu`,
`--worker-memory` and `--coordinator-memory` options. Embedding defaults to four
CPU cores and 8 GB per worker. Inference batches contain at most 16 texts and
16,384 padded tokens, sorted by token length. Vectors are checkpointed every 256
texts so retries reuse completed work. After writing vectors, each partition
compacts its checkpoint files for subsequent reads. PyTorch threads follow the
requested CPU count. Description and vector deduplication distribute work across
the pool independently of source-file count.

The catalog declares subject-domain and task-mechanic routing facets. Matching
selects one primary subject in each facet. It retains the three highest-ranked
subjects per facet, three capabilities within each primary subject, and three
capabilities across the whole catalog. The coverage export applies fixed
maximum cosine distances of 0.45 for subjects and 0.25 for capabilities.
Alignment does not change curation admission or train/eval use.

The `publication` stage checks that every selected source/task ID has exactly one
alignment row with matching lineage. It writes full task Parquet and
`release.json`, preserving source inventory, population counts and upstream
artifact roots. Complete converted releases cover each conversion manifest's
converted-task count.

The `coverage` stage reads those completed task outputs through Zephyr. Each
shard counts task instances by source, outcome and curriculum node, then a
small reduction combines the counts. A separate bounded sample keeps up to
100 tasks per source and, independently, 10 per curriculum node (subject, group
or capability). Their union is deduplicated by source/task identity. It writes statistics, sampled task Parquet
and `coverage.json`. Atlas loads this compact dataset. The full task corpus,
embeddings and assignments remain in object storage. Dataset and train/eval
views sum the precomputed counts; browsing the heatmap does not aggregate task
rows. Example lists show samples and retain full-population cell counts.

### Complete converted corpus

Run the existing QUICK mode on Iris with region-local outputs:

```bash
uv run iris --cluster=marin job run --target-cluster YOUR-CW-REGION \
  --extra cpu --cpu 0.5 --memory 2GB --timeout 43200 \
  -- python -m experiments.post_training.task_curation.driver \
  --mode quick --output-root s3://YOUR-REGIONAL-BUCKET/corpus \
  --max-workers 8 --worker-cpu 2 --worker-memory 16g \
  --coordinator-memory 8g --concurrent-sources 4
```

Add `--run` to execute. List runnable source names with:

```bash
uv run python -c 'from experiments.post_training.task_curation.sources import runnable_sources; print("\n".join(runnable_sources()))'
```

Repeat `--source NAME` to select sources; omit it for every runnable source. QUICK skips grading and admission, preserves
converter rejections, and writes `corpus.json` after all selected sources finish.
Conversion artifacts retain their pinned acquisition graph and recipe identities.
The manifest includes completed source roots, immutable source-manifest hashes,
conversion counts and the full registry inventory. Inventory entries without
converters remain visible but have no converted tasks.

Resume from that corpus with the same driver and shared model/cache options.
On CoreWeave, disable Xet downloads to avoid cold-cache download stalls:

```bash
uv run iris --cluster=marin job run --target-cluster YOUR-CW-REGION \
  --extra cpu --cpu 0.5 --memory 2GB --timeout 43200 \
  -e HF_HUB_DISABLE_XET 1 \
  -- python -m experiments.post_training.task_curation.driver \
  --corpus s3://YOUR-REGIONAL-BUCKET/corpus/corpus.json \
  --embed coverage --review-cache s3://YOUR-REGIONAL-BUCKET/semantic-cache \
  --max-workers 128 --worker-cpu 4 --worker-memory 8g --coordinator-memory 8g
```

For optional rewrites, pass the credential to the Iris job with
`-e GLM_BULK_TOKEN VALUE` before `--`.
The driver prints the final artifact `root` and `paths`; the Atlas manifest is
`<root>/release.json` after publication. It checks source-manifest hashes before
processing. The default `--representation direct` skips model rewrites. For rewrite
comparisons, `--embed descriptions` stops before Harrier inference. Local resumes
use `--cache-dir` to select a local artifact cache and local Zephyr workers.
Converter rejections never enter the embedding population. Converted counts and
embedding coverage do not establish curation admission.

## Declaring a dataset

Follow [Adding a dataset](datasets/README.md#adding-a-dataset). Edit declarations,
and publish the catalog; [Atlas publishing](../../../docs/references/rl-data-atlas.md)
reads the published curation catalog. Run the affected sources to publish their
new declarations and assessments. Recount pinned whole-file inputs offline:

```bash
uv run python -m experiments.post_training.task_curation.count_inputs \
  --snapshot /path/to/tasktrove/repo \
  --revision 02923004846e4e73862c20962f823a6d05100e7a \
  --files '*/tasks.parquet'
```

The command verifies HF download revision metadata and reads parquet footers.
Sum only declared files; row selectors require their own complete count.
