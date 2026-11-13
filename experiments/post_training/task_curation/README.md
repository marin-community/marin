# Task data curation

[Dataset declarations](datasets/README.md) feed TaskSpec conversion and the RL
Data Atlas through [all_sources()](sources.py). See the
[reference](../../../docs/references/task-curation.md) for source fields, grading,
reviewed SAMPLE/FULL campaigns and artifact outputs.

## Pinned registry receipts

Generated census, lexical, retention, deduplication and fingerprint-index JSON is
archived in the private `open-athena/marin-rl-data-atlas-registry` dataset. The
repository keeps a reference at each original path with `repo`, exact commit
`revision`, archive `path` and `sha256`. The archive preserves the original bytes
and layout; it does not publish or modify a retained TaskCompendium release.
Human-reviewed policy, including duplicate acknowledgements, stays in Git.

`registry.read_registry_json()` uses the Hugging Face client's authenticated
cache and verifies the downloaded bytes before decoding. Missing revisions,
unavailable credentials and checksum mismatches fail the operation; there is no
fallback to a different revision or inline data. Catalog construction, admission,
selection and the source-overlap gate use this reader. Importing catalog modules
does not download the registry; receipt accessors load it when called.

For new measurements, follow the [registry update procedure](../../../docs/references/task-curation.md#updating-pinned-registry-receipts): stage raw JSON in a fresh tree, append to the private registry, verify readback, then update Git references. The overlap workflow uses the read-only `ATLAS_REGISTRY_HF_TOKEN` secret, exposed as `HF_TOKEN` in that job. Local readers use normal Hugging Face credentials; archive updates need write access to this private registry only.

Unit tests inject small documents at the download boundary and reject registry
downloads. Tests that inspect the complete private registry use the
`data_integration` marker and are excluded from the default safe suite. The
source-overlap ranking fixture contains two small contracts rather than an
archived production population.

## Final task selection

Pass `--selection RECIPE.json` to a FULL campaign to append a
`rl-data/task-selection/<name>/<version>` artifact after source admission.
The recipe names the source IDs, assigns them to selection families, sets
family quotas and pins the complete heldout source in retained storage.
The step filters heldout overlap and cross-source duplicates before sampling,
then writes canonical `task_spec` Parquet for training and evaluation.
See the [quick RL loop](../../../docs/recipes/quick-rl-loop.md) for the current
selection recipe and the training/evaluation handoff.
See [task selection](../../../docs/references/task-curation.md#final-task-selection)
for the recipe and resume commands.

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

Evaluation sources also use QUICK, with stricter publication checks that preserve
the complete benchmark population. They publish canonical TaskSpec rows and
evaluation protocol sidecars. See [evaluation tasks](../../../docs/references/evaluation-tasks.md)
for declarations, outputs and the standard evaluation driver.
`evaluation.py` binds `EvalDataset` recipes to
`taskcompendium.pipeline.evaluation.publish_evaluation_corpus`. Evaluation
protocols use schema version 3 from `taskcompendium.evaluation.spec`; source
adapters live under `taskcompendium.importers` and `taskcompendium.convert`,
including reusable benchmark modules in `taskcompendium.importers.evalchemy`.

Select `--mode quick`, `sample` or `full`, or resume with `--corpus` as described
below. Omit `--run` to print a plan without downloading, converting or starting
workers. QUICK preserves named sources in request order, ignoring repeated names.
SAMPLE/FULL select sources in catalog order. Omitting `--source` selects runnable
training sources. Select evaluation sources explicitly with `--mode quick`.
Local QUICK needs no review model, controller, worker image or
coordinator settings.

Remote artifacts default to `s3://marin-us-east-02a/marin` on CW, independently
of the ambient `MARIN_PREFIX`. Use `--artifact-root URL` for an explicit override;
local `--cache-dir` takes precedence. Keep each retained campaign's
`--output-root` or `--report-path` in a new versioned directory in that bucket.

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
  --reviewer-tier HEAVY \
  --run
```

Local reviewed runs need no worker image or controller for in-process graders.
The default report is `CACHE_DIR/campaigns/campaign.json`, with its retained run
catalog beside it. `--report-path` chooses another report
location. Review requests are cached under `CACHE_DIR/review-cache` unless
`--review-cache` selects another location. `GLM_BULK_TOKEN` supplies credentials.
Package-backed and image graders retain their normal environment prerequisites.
`--image-backend daytona` selects a remote image grader for local curation.
`--image-backend gvisor` requires an authenticated Iris driver inside the
execution cluster with Pod networking; a controller URL alone is insufficient.
Worker-image and coordinator flags apply only to Iris execution.
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
Changing the assessment rubric reuses curation and prepared packets. Changing the
reviewer spec selects new review and assessment cache identities. Ordinary source
changes invalidate their downstream artifacts; shared stage implementation changes require an explicit version bump.

Chat transport allows up to 64 requests per worker process, selected with
`--review-concurrency`. Eight concurrently grading sources at 64 requests each
can issue up to 512 requests in flight. Local reviewers share one process limit;
remote worker processes have separate limits. Byte groups do not impose a
completion barrier. Each task still has its own request, judgment and trace.
Programmatic callers can select the provider's asynchronous batch API with
`ReviewConfig(spec, mode=ReviewMode.BATCH)`.
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

For a production FULL campaign, add
`--publish-catalog s3://marin-us-east-02a/marin/rl-data/catalog/production`.
Enable assessment and omit `--cache-dir`, which selects local execution and a
local artifact root. Production reuses the remote artifact cache under
`--artifact-root` (default `s3://marin-us-east-02a/marin`) and model requests under
the explicit `--review-cache`. Reuse those locations, pinned inputs, recipe
versions, rubric and model revisions across campaigns. A new report directory or
enabling assessment does not force recomputation of completed work. Switching
artifact roots does not move cached work; verify artifact and model-cache reuse
before executing a campaign from a new root. `--publish-catalog` alone does not
change either cache location. The shared catalog retains run history
and accumulates assessments. Omit the option for local previews and SAMPLE campaigns.
Use `--source` to grade only new or changed sources; other production sources keep
their declarations and grades without executing their pipelines. Inventory-only
sources take their declarations from the current code at every publication. Older attempts
are ignored per source, and removing a declaration does not retire its published entry.
After publication, run the Atlas loader to refresh Postgres; no inference is repeated.

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
`task.toml` from `task_binary`, check it with the vendored Harbor runtime:

```bash
uv run --project vendor/harbor --extra runtime --frozen python -c \
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

Start with a local conversion and a seeded sample from a selected source:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' \
  --with-editable './lib/marin[cpu]' python \
  -m experiments.post_training.task_curation.driver \
  --mode quick --source SOURCE \
  --output-root /path/to/quick --cache-dir /path/to/persistent/cache \
  --embed coverage --embedding-sample-size 20
```

Add `--run` to execute. Omit `--embedding-sample-size` to embed every selected
task. QUICK embeds converted tasks; SAMPLE/FULL embeds admitted tasks. Neither
matching nor Atlas changes admission or train/eval use. `--embed coverage` runs
through the compact export; other `--embed` values stop at intermediate stages.
`--catalog-root` selects a pinned directory containing `curriculum.json`;
otherwise the graph uses `TASK_CURRICULUM`.

Direct embedding needs no inference-service credential. It uses bounded public
instructions and answer/tool requirements, excluding task files, solutions and
private verifier resources. Task and curriculum text use the same Harrier
checkpoint, retrieval instruction and normalization. `--worker-cpu` and
`--worker-memory` size each CPU worker; `--max-workers` bounds the shared pool.
The model loads inside workers when an embedding batch has cache misses.
Model initialization is serialized within each process because Transformers
temporarily changes Torch's default dtype. Concurrent workers reuse the cached
backend after initialization.

`--cache-dir` selects local workers and artifacts. `--review-cache` selects the
shared inference cache, defaulting locally to `<cache-dir>/review-cache`.
Keep it across campaigns and worker-count changes. FineStore caches vectors by
rendered text and embedding specification; Zephyr writes vector Parquet.
Alignment reads that Parquet and keeps the curriculum matrix in worker memory.
A curriculum edit reuses task vectors.

Optional `--representation rewrite --description-model MODEL --description-model-revision REVISION
--description-base-url ENDPOINT` uses an inference service to rewrite public inputs before
embedding. Supply its credential through `GLM_BULK_TOKEN`. Failed rewrites retain
`skipped` rows and counters after bounded retries; the pass fails above 5%
failures across distinct public inputs. `insufficient_context` records an input
that does not establish the requested operation. Both remain in Atlas counts.

Publication references the assignment Parquet and records source/population
counts in `release.json`. Complete converted releases reconcile those counts
against conversion manifests. Coverage exports full-population statistics and
independent samples of up to 100 tasks per source and 10 per curriculum area.
The [Atlas loader](../../../docs/references/rl-data-atlas.md#curriculum-coverage)
imports those statistics, samples and curriculum entries into Postgres.
Full task and vector outputs stay in object storage.

See the [pipeline diagram and module links](../../../docs/references/task-curation.md#pipeline-overview)
for the complete graph and implementation details.

### Complete converted corpus

Run the existing QUICK mode on Iris with CW us-east02a outputs:

```bash
uv run iris --cluster=marin job run --target-cluster YOUR-CW-REGION \
  --extra cpu --cpu 0.5 --memory 2GB --timeout 43200 \
  -- python -m experiments.post_training.task_curation.driver \
  --mode quick --output-root s3://marin-us-east-02a/marin/rl-data/campaigns/YYYY-MM-DD-N \
  --max-workers 8 --worker-cpu 2 --worker-memory 16g \
  --coordinator-memory 8g --concurrent-sources 4
```

Add `--run` to execute. List runnable source names with:

```bash
uv run python -c 'from experiments.post_training.task_curation.sources import runnable_sources; print("\n".join(runnable_sources()))'
```

Repeat `--source NAME` to select sources; omit it for runnable training sources. QUICK skips grading and admission, preserves
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
  --corpus s3://marin-us-east-02a/marin/rl-data/campaigns/YYYY-MM-DD-N/corpus.json \
  --embed coverage --review-cache s3://marin-us-east-02a/marin/rl-data/semantic-cache \
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

## Complete retained-population review

`full_qc_inputs` indexes the combined deduplication and lexical remainder against
the frozen `SourceTextPopulation`. It records task and runtime hashes separately,
and reports missing execution requirements. Indexing does not qualify execution
or establish FULL quality.

`full_qc` reviews content independently from execution qualification. Its
artifacts record `qualification_status=pending`, so complete content coverage
does not establish execution admission. Qualification and repairs use the
[full-population oracle procedure](../../../docs/references/task-curation.md#full-population-oracle-filter).

Choose `--reviewer-version v3` for future FULL-QC source starts. The ongoing
48-worker campaign, Gym repair and Prism finalization remain on their frozen v2
specifications. Resuming or assembling that campaign requires explicit `v2`;
never replay its completed observations under v3. New v3 source and campaign
outputs use separate prefixes and an exact-request cache. The reviewer identity
and source assembly bind the v3 model-YAML digest and generation budgets, so v2
and v3 observations cannot be silently combined.

Reviewer-spec changelog (2026-11-13):

| Version | Serving context (tokens) | Initial / retry generation (tokens) | Status |
| --- | ---: | ---: | --- |
| v2 | 262,144 | 4,096 / 8,192 | Existing main pass, preserved |
| v2 repair | 262,144 | 8,192 / 16,384 | Existing incomplete-only repair policy |
| v3 | 524,288 | 8,192 / 16,384 | Future FULL-QC starts |

V3 constructs a literal ReviewerSpec: GLM-5.3 revision
`aca966e4e02791568aa6a4ced368624b3d897f42`, API model `glm-5.3`, FP8 precision,
temperature 0.7, top_p 0.95, top_k 0 (disabled), low reasoning effort, three
attempts, payload revision 4 and the task-quality-v2 rubric with its content hash.
Its explicit serving pin checks the checkpoint, API name and context independently
of shared relay presets. Top_p matches the pinned model's generation configuration;
top_k disables top-k filtering of candidate next tokens in vLLM 0.28.0. Optional sampling overrides are
absent from existing v2 requests and identity, preserving its frozen caches. Prompt character budgets remain explicit and unchanged (512,000 by
default); they are separate from the serving token context. Compilation and CUDA
graphs remain enabled. The v3 YAML has not yet been exercised by a live GPU job.

Provide a JSON list with `inputs_path` and `output_path` for each source. Run
inside a CPU Iris job with batch priority, a three-day timeout, and an explicit
target cluster. Launch this coordinator with --target-cluster cw-us-east-08a.
Local GLM children inherit its regional controller and use the ANY_REGION marker
to suppress inherited region selectors, with batch priority and a three-day
timeout. Iris removes this marker before scheduling; CoreWeave nodes need not
advertise an iris.region label. A cluster target on these local children would
request another federation handoff. The immutable public checkpoint, serving image, precision, model
YAML digest, explicit worker setup scripts, sampling settings, and endpoint identity participate in reviewer
evidence and request caches. The existing relay retains its separate unverified
checkpoint identity. The upstream vLLM image needs Git installed before Marin
worker imports; the explicit setup command is part of local endpoint identity.

Payload revision 4 omits top-level execution requirements and actor/grader lock
artifact locations. Deployment runtime is retained in audit hashes rather than
appended to the content rubric. Calibration-only changes to these fields reuse
identical reviewer requests. Public instructions, grader settings, images and
resource files remain visible: changes to those inputs select new requests.
Native resource files are preserved, including any embedded resource settings;
rewriting a native task.toml therefore invalidates its content request.

Batch preemption can move a serving task to a new address. Full-QC reviewers wait
for serving readiness and resolve the current endpoint before each uncached
batch or outer retry. Valid observations stay in immutable batch receipts and
are reused after recovery. When replacing an interrupted worker, stop the old
writer before resuming the same shard assignment and output prefix; keep its
reviewer identity and caches when the task and reviewer contracts are unchanged.

```bash
uv run --no-sync --package marin-core --extra dedup python -m \
  experiments.post_training.task_curation.full_qc --reviewer-version v3 \
  --work s3://bucket/campaign/work.json \
  --campaign-output s3://bucket/campaign/review-v3 \
  --review-cache s3://bucket/campaign/request-cache-v3 --run
```

Omit `--run` to inspect the source plan and pinned reviewer identity
without reserving GPUs. Successful batches are reused by exact request identity;
failed and preempted attempts remain in versioned output paths. Each source
completion writes task-level observations, computed Result and repair guidance,
and a campaign record identifying serving jobs and pending qualification. This
command does not publish datasets or alter Atlas admission.

### Audit reviewer errors

`audit_harness` freezes a stratified sample of Prism issue-positive reviews and
two differently ordered blinded review packets. It verifies native task/runtime
hashes, separates original GLM verdicts from reviewer inputs, and records the
sample seed, strata and packet hashes. `audit_analysis` reconciles the two review
files with separately recorded claim causes and estimates the population-weighted
harness false-discovery fraction. Its interval conditions on the human/model
review labels; two model reviewers can share blind spots. It measures errors
among issue verdicts, not overall reviewer accuracy.

`audit_review` repeats the frozen 100 tasks through an existing owned local GLM
endpoint. Supply its scoped capability URL as the `QA_ENDPOINT_URL` secret file;
never put the URL in command arguments, artifacts or logs. Run the entrypoint in
a CPU Iris job at batch priority with a 72-hour timeout and the data's target
cluster. It starts no model worker. Each prompt version needs a distinct output
prefix, including a fresh cache for an original-prompt rerun. Preserve baseline
code before changing the rubric and compare the actual stored reviewer identities.
The sample digest and native task hashes are checked before requests are sent.
Exact batch receipts preserve valid observations across retries and preemption.

Before relaunching a corrected campaign, compare original, fresh-baseline and
revised observations against the frozen independent labels. Require valid paired
coverage, lower harness error and retained detection of confirmed content defects.
Keep uncertain or disputed labels separate. A factual transport correction does
not establish that other types of reviewer error are resolved. Preserve all
original task artifacts, review judgments, annotations and output prefixes.

### Parallel retained-batch review

Use `full_qc_shards` when a source needs more than one model replica. A worker
owns original 32-task batches whose `(batch_offset + batch_index) % shard_count`
equals its `shard_index`. All workers use the same immutable work manifest and
shard count. Set each source's `batch_offset` to the cumulative number of batches
in earlier manifest rows. The last batch may contain fewer than 32 tasks.

Each manifest row has these fields:

```json
{
  "inputs_path": "s3://bucket/contracts/source/v1",
  "batch_output_path": "s3://bucket/reviews/source/v1",
  "shard_output_path": "s3://bucket/shards/source/v1",
  "assembly_output_path": "s3://bucket/assembled/source/v1",
  "batch_offset": 0,
  "prompt_budget": 512000
}
```

Wrap the rows in a JSON list. `batch_output_path` may reuse an existing full-QC
campaign: source sharding preserves the batch and exact-request identities.
`prompt_budget` sets both the initial and retry character budgets for that source;
changing it changes the reviewer identity. Keep the pinned model, rubric and
sampling settings unchanged when resuming.

Run one batch-priority CPU coordinator on cw-us-east-08a for every shard index
from zero through `shard_count - 1`, with a three-day timeout. Its local GLM
children also use batch priority and three-day timeouts:

```bash
uv run --no-sync --package marin-core --extra dedup python -m \
  experiments.post_training.task_curation.full_qc_shards review --reviewer-version v3 \
  --work s3://bucket/campaign/shard-work.json --shard-index 0 --shard-count 48 \
  --max-passes 3 --review-cache s3://bucket/campaign/request-cache-v3 \
  --campaign-output s3://bucket/campaign/worker-000 --run
```

Each pass retries invalid or unavailable observations under the pinned review
protocol. Valid observations in partially completed batches are retained,
successful batches are reused, and every attempt remains available. Run exactly
one current-version writer per batch cache: do not overlap a whole-source
review driver with shard workers, or two shard campaigns sharing that
cache. The batch `latest.json` pointer has no compare-and-swap protection.
Assignments are disjoint within the shard set. Pre-sharding drivers that write
only immutable attempts and all-valid
`complete.json` receipts may finish alongside a shard campaign; they never
replace its partial-progress pointer, but overlapping coverage can duplicate
requests. A shard
record reports its attempted and valid counts; it makes no source-wide FULL claim.
The assembly campaign reports final status counts and the invalid rate per
source separately from its issue rate.
The `latest.json` index pins an immutable shard record by SHA-256.

Assemble completed sources on CPU in the data's cluster, also at batch priority
with a three-day timeout. Assembly makes no model calls and skips sources still
missing shard records:

```bash
uv run --no-sync --package marin-core --extra dedup python -m \
  experiments.post_training.task_curation.full_qc_shards assemble --reviewer-version v3 \
  --work s3://bucket/campaign/shard-work.json --shard-count 48 \
  --campaign-output s3://bucket/campaign/assembly --run
```

Assembly requires evidence for every distinct retained task and verifies every
shard, receipt digest, reviewer identity, task/runtime hash and batch assignment
before writing the source result. Missing, overlapping or foreign evidence is
rejected. Invalid observations
remain unavailable in the computed Result and leave `coverage_complete=false`.
Execution qualification remains pending.

### Repair incomplete retained reviews

After a source's primary pass finishes, freeze its strict assembled artifact and
the SHA-256 of its `.artifact.json`. `full_qc_repair` preserves every completed
primary observation, including uncertain valid verdicts, and retries only INVALID
(malformed response) or UNAVAILABLE (no valid response) observations. Identity is
checked against the retained TaskSpec and runtime hashes. The repair uses the same
pinned model, rubric, payload and temperature with explicit 8,192-token initial
and 16,384-token retry budgets.
These budgets have a separate ReviewerSpec and exact-request cache identity.

Run a batch-priority CPU job on the serving workers' cluster with a three-day
timeout. Repeat `--serving-root` to borrow existing workers from the same pinned
primary campaign. Each request selects from their current registrations and
checks the advertised model; an unregistered worker is skipped until it returns.
The command neither starts nor stops serving jobs. Keep the primary 48 workers
running and use a separate output prefix and cache.

Set `--batch-workers` explicitly, from one to 16. Inference groups contain only
original 32-task batches with incomplete primary observations, so intervening
valid batches do not consume worker slots. Only one group of TaskSpecs is queued
at a time. Completed repair results remain in memory until a second source pass
writes every retained batch in its original order. Each batch admits at most four
requests, so the CLI adds at most 64 concurrent repair requests across the
borrowed serving pool. Physical routing and concurrency do not change the
reviewer, request-cache or composite identity. Only incomplete primary contracts
are submitted, and completed repair receipts are reused on resume:

```bash
uv run --no-sync --package marin-core --extra dedup python -m \
  experiments.post_training.task_curation.full_qc_repair \
  --primary s3://bucket/assembled/source/v1/attempts/PRIMARY \
  --primary-sha256 PRIMARY_ARTIFACT_SHA256 \
  --serving-root /owner/primary-worker-000 \
  --serving-root /owner/primary-worker-001 --batch-workers 2 \
  --output s3://bucket/repair-overlay/source/v1 \
  --review-cache s3://bucket/repair-overlay/request-cache --run
```

Omit `--run` to validate the frozen observations and print the primary, repair
and composite identity hashes without model requests. The repair output contains
every retained exact contract once. Each row records the selected review's
reviewer identity and the SHA-256 of its primary records or repair receipt.
Primary review JSON is copied byte for byte. The source identity lists both specs
and their selection policy. Source issue counts, rates and verdicts use both sets
of observations, retaining uncertainty separately; failed repairs
leave coverage incomplete. Rerunning the same frozen primary and repair prefix
reuses successful repair receipts without resampling them. Existing primary
records and pointers are never changed. Oracle filtering and resource calibration
remain separate requirements for execution qualification.

Prism's v2 content review is finalized with 2,220 valid reviews of 2,221 retained
contracts. The Result is SOME_ISSUES: 286 issues among 2,213 conclusive reviews
(12.9236%); seven valid uncertain reviews are reported separately. One task stays
in the population as “QC unreviewed: review exceeded 16,384 tokens.” Its strict
coverage remains incomplete, and execution qualification is unchanged.
[Final source evidence](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Fatlas%2Ffull-qc%2Ffinal-content%2F2026.11.13.1%2Fprism.json)
is stored at
`s3://marin-us-east-02a/marin/rl-data/atlas/full-qc/final-content/2026.11.13.1/prism.json`.

### CrossCode Python qualification candidate

CrossCodeEval Python, TypeScript, Java and C# are evaluation sources. Python's
native `tests/test_state.py` contains the Terminal-Bench canary; the evaluation
policy applies to all four derived cohorts. Their explicit `intended_use` is
`eval`, and TRAIN selection rejects even older artifacts that claim training
eligibility. Within-source deduplication still applies; lexical overlap is
measured without dropping evaluation tasks.

All four frozen duplicate and combined-drop manifests are empty, so the private
pins and all 7,763 task contracts are unchanged. [Drop-set equality evidence](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Fatlas%2Ffull-qc%2Fcrosscode-eval%2F2026.11.13.1%2Fretention-proof.json)
is also available at
`s3://marin-us-east-02a/marin/rl-data/atlas/full-qc/crosscode-eval/2026.11.13.1/retention-proof.json`.
Python's original pin retains FULL content QA: 500 valid reviews, eight issues
(1.60%, GOOD), with zero unavailable or uncertain observations. That exact-pin
content evidence does not establish execution qualification of Python's four-tier
grading repair candidate.

`crosscode_python` stages a new candidate for the frozen 500-task CrossCode Python
population. Build the archived public actor context with the image builder first,
using explicit batch priority and three-day deadlines for its build and push.
Keep the returned image digest and context hash. The staging step checks that
every retained task names that exact archived context.

Prepare the separate grader's hashed dependency lock on a batch CPU worker:

```bash
uv run --no-sync --package marin-core --extra dedup python -m \
  experiments.post_training.task_curation.crosscode_python prepare-grader \
  --output s3://bucket/crosscode/grader-build/v1 --run
```

The resulting artifact pins a Python 3.12 base and a hash-locked `tomli-w`
dependency in `context.tar.gz`. Build that context with the same explicit image
job policy, then stage the new TaskSpecs and faithful Harbor packages on CPU:

```bash
uv run --no-sync --package marin-core --extra dedup python -m \
  experiments.post_training.task_curation.crosscode_python stage \
  --inputs s3://bucket/crosscode/retained-inputs/v1 \
  --actor-image registry/actor@sha256:ACTOR_DIGEST \
  --actor-context-sha256 ARCHIVED_CONTEXT_SHA256 \
  --grader-image registry/grader@sha256:GRADER_DIGEST \
  --output s3://bucket/crosscode/candidate/v1 --run
```

The original `solution/solution.py` is a completion fragment. Its new `solve.sh`
copies those original bytes to `/app/solution.txt`; it does not execute the
fragment or generate a reference. The private grader retains the native
normalization and all four scoring tiers, including leading-identifier full
credit and non-empty partial credit. It emits an authoritative structured verdict
directly from those comparisons. A missing submission now produces a completed
scored zero instead of a failed producer; a missing reference is an unscored
invalid task. The new candidate records these reporting changes and preserves
the archived native files.

The candidate uses SEPARATE grading and explicit file capture. It is unqualified:
run full-population original-oracle calibration and independent negative controls,
then review all new exact contracts. Preserve native 900-second actor and
720-second verifier deadlines while calibrating missing machine capacities.
Use the [oracle filter procedure](../../../docs/references/task-curation.md#full-population-oracle-filter)
for production Harbor trials and receipts. Staging neither publishes a dataset
nor changes Atlas admission.
