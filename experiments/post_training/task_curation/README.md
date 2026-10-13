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

Campaign evidence links point to `assessment_inputs`, `assessment_judgments`
and `assessment_report`. Inputs contain a sample manifest, curation summary and
packet parquet. Judgments contain grade parquet and paired request/response
JSONL traces. A malformed or unavailable response is retried twice; remaining
failures retain partial grades and fail the stage. Batch transport separately
retries a failed HTTP request up to three times. Reruns reuse successful
request-cache entries. Missing required evidence keeps the source's sampled
quality unknown even when the model supplies a favorable opinion.
Invalid structured replies, including token truncation, use the existing curation
review retry budget of 8,192 tokens. First attempts use 4,096; transport failures
do not increase the current budget. Valid cached judgments keep their original
request budget.
Retries constrain the tool's task ID to the selected item, preventing a copied
hash typo from misattributing a judgment. Identity validation still rejects any
mismatch; the publisher never repairs model IDs after the fact.

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

`--curated-inventory FILE_OR_URL` reassesses retained FULL outputs without
repeating conversion, panel review or controls. The inventory contains `sources`
entries with `source` and `manifest` fields. Each adopted manifest must match
the current source, dataset pin and recipe revision. Incomplete or gated
curation remains visible in the assessment; model opinions cannot override it.
Run large GCS-backed inputs in their storage region.
Remote Zephyr scratch uses the current artifact prefix, so GCS campaigns keep
their intermediate data in GCS too.
It lives at `ARTIFACT_PREFIX/.zephyr/rl-data-campaign` remotely and
`CACHE_DIR/.zephyr` locally. Zephyr gives each execution its own directory and
removes it when releasing that execution.
Selected evidence reads decode complete Parquet row groups, including task JSON
for unselected rows in those groups. ID scans project only the required columns.
The bounded packet set is written as one Parquet file after evidence projection.
Packets include the runtime submission contract: conversation answers may be
extracted and written to grader files by the adapter. Duplicate TaskSpec JSON
and normalization text are omitted from review/verification sidecars; oversized
source rows and normalization changes have explicit hashed previews. Full audit
data remains in the curation output.

Each campaign appends its reports and compact evidence to the FineStore catalog
beside the campaign report. `assessment_runs` records ordinary pipeline keys,
complete reports, failed attempts with partial evidence, and disabled assessments
with a skip reason. Catalog source counts retain their full or SAMPLE scope.

SAMPLE/FULL also publishes a complete inventory snapshot from these catalog
records. It archives sampled packets, judgments and inference traces before
changing the current snapshot pointer. Publication compacts blob descriptors, so evidence
reads do not reopen every shard written by earlier publications. Evidence URIs
and older read views remain valid.
The inventory keeps control counts and policies; detailed control evidence and
model examples remain in the linked archived report.

A failed retry keeps a previous grade only when its report pipeline key matches.
A changed key with a failed assessment
has no current grade. Sources without conversion pipelines remain inventory
entries with an explicit skip reason. Authored legacy ratings are excluded.
Repeated successful invocations with the same pipeline key reuse the canonical
grade. Changes to the rubric, model revision, sample size or seed change that key.

Read the current snapshot without rereading source parquet:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.catalog_snapshot \
  --catalog-root /path/to/cache/campaigns/catalog --output /path/to/catalog.json
```

`publish_catalog(root, run_id, sources)` publishes a retained terminal invocation
from the same catalog. Run publication in the data's storage region: it reads
sampled packets and traces to preserve them. SAMPLE and overridden input counts
never replace a declaration's full-source count. Curation failures and executed
verification coverage remain separate from assessment opinions.

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

## Declaring a dataset

Follow [Adding a dataset](datasets/README.md#adding-a-dataset). Edit declarations,
not generated Atlas JSON; [Atlas publishing](../../../docs/references/rl-data-atlas.md)
rebuilds and bundles the catalog. For an export-only check:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.export_catalog \
  --output infra/marina/applets/rl_data_catalog/dist/catalog.json
```

Recount pinned whole-file inputs offline:

```bash
uv run python -m experiments.post_training.task_curation.count_inputs \
  --snapshot /path/to/tasktrove/repo \
  --revision 02923004846e4e73862c20962f823a6d05100e7a \
  --files '*/tasks.parquet'
```

The command verifies HF download revision metadata and reads parquet footers.
Sum only declared files; row selectors require their own complete count.
