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

The driver requires `--mode quick`, `sample` or `full`. Omit `--run` to print a
plan without downloading, converting or starting workers. QUICK requires named
sources and preserves their request order, ignoring repeated names. SAMPLE/FULL
select sources in catalog order; omitting `--source` selects the full catalog.
QUICK needs no review model, controller, worker image or coordinator settings.

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

The canonical embedding graph consumes existing curation outputs and the immutable
task curriculum. It runs every data pass through Zephyr: instruction projection,
cached semantic descriptions, task vectors, curriculum anchor vectors and exact
faceted alignment. The old curriculum OpenAI embedding CLI and SQLite cache are
removed.

Start with a seeded sample from a local QUICK conversion. QUICK does not admit
tasks, so select the `converted` population explicitly:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' \
  --with-editable './lib/marin[cpu]' python \
  -m experiments.post_training.task_curation.embed \
  --curated 'Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2' \
    /path/to/quick/tasktrove-nl2bash \
  --cache-root /path/to/persistent/embedding-cache \
  --population converted --sample-size 20 \
  --model-revision YOUR_GLM_REVISION --base-url YOUR_GLM_ENDPOINT \
  --embedding-backend cpu --compute-dtype float32
```

The command prints a plan. Add `--run` to execute; `GLM_BULK_TOKEN` supplies the
description-model credential. Repeat `--curated SOURCE_ID ROOT` for more sources.
For reviewed outputs, select `--population admitted`. Omit `--sample-size` to
process complete selected populations. `--catalog-root` selects another immutable
directory containing `curriculum.json`; otherwise the graph uses `TASK_CURRICULUM`.
Use `--stage descriptions` or `--stage curriculum_embeddings` to run one branch
and its dependencies while measuring the local loop.

The annotator sees instructions and the public tool/answer contract. It does not
read task files, source labels, private verifier resources or solutions. Tasks
with the same public input share one description. Missing information about the
requested operation produces `insufficient_context`.

The description names the required knowledge or tool environment and the core
operation in roughly 15–40 words. Routine setup, output capture and submission
details are omitted unless they are the main problem. Each ready task produces
one vector. Full instructions go to the describer; `--max-tokens` limits the
rendered embedding input, including its instruction prefix. It does not limit
task instruction length. Overlong embedding inputs fail without truncation.

Tasks and curriculum anchors share one embedding instruction and the full Harrier
embedding specification. FineStore stores vectors by rendered-text/specification
hash, with batched reads and writes. A curriculum-only change reuses task
descriptions and vectors. Keep the vector archive under a persistent retention
policy: assignment workers read it in bounded batches. Independent Parquet vector
outputs preserve each embedding stage's data and provenance.

For an existing Iris-managed TEI Harrier pool, use `--embedding-backend tei
--tei-endpoint NAME` and declare the pool's actual `--compute-dtype`. Configure
the same checkpoint, tokenizer and token limit. Requests reject truncation.
`--max-workers` and `--worker-memory` select the Zephyr worker pool; run the same
entrypoint inside Iris with a storage-region cache root.

For a direct single-GPU baseline, select `--embedding-backend gpu --max-workers 1`
inside a GPU-equipped Iris job. This runs local Zephyr workers in that job and
keeps Harrier resident on its GPU. Compare against CPU using identical inputs and
precision in separate cache roots before sharing vectors across backends.

Alignment selects the nearest subject within each declared curriculum routing
facet. `MAX_SUBJECT_DISTANCE` and `MAX_CAPABILITY_DISTANCE` in
`taskcompendium.pipeline.curriculum_alignment` control accepted cosine distance
at each level. Changing either constant invalidates alignment without recomputing
task or curriculum embeddings. The three highest-ranked subject and capability candidates, including the nearest
match, and capability score margins remain available for inspection. Out-of-scope tasks and
subject matches with weak capability matches remain explicit outcomes. This pass
does not change curation admission or train/eval use.

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
