# Task curation

Task curation turns pinned RL datasets into TaskSpec parquet files. Each dataset
is declared once as an `RlDataPipeline` under
[`experiments/post_training/task_curation/datasets/`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/datasets/README.md),
and the catalog
[`sources.py`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/sources.py)
lists every declaration:

```python
from experiments.post_training.task_curation.sources import all_pipelines

pipelines = all_pipelines()  # name -> RlDataPipeline
```

Building the catalog performs no downloads, inference or job submission.

## Package organization

| Package | Responsibility |
|---|---|
| `experiments/post_training/task_curation/datasets/` | Dataset declarations, converters, rubrics, controls and `*_grade.py` grader scripts |
| `experiments/post_training/task_curation/images/` | Grader image recipes and their digest-pinned `Image` constants |
| `experiments/post_training/task_curation/sources.py` | The catalog, `all_pipelines()` |
| `experiments/post_training/task_curation/pipeline.py` | `RlDataPipeline` and its `data/rl/<name>-<hash>` artifact |
| `experiments/post_training/task_curation/driver.py`, `campaign.py` | Campaign options, grading machines, shared pool and full-mode admission |
| `taskcompendium.convert` | Conversion techniques shared by declarations |
| `taskcompendium.pipeline` | Sampling, review, filtering, verification and outputs |
| `taskcompendium.runtime` | Grading in fresh Shellbox machines |
| `verifyit` | Stock graders and the bridge to image-installed source scorers |
| `shellbox` | Isolated machines and their backends |

TaskCompendium does not import experiments, name datasets, or construct
ArtifactSteps.

## Declarations

An `RlDataPipeline` has these fields:

| Field | Meaning |
|---|---|
| `name` | Catalog key and artifact name. |
| `source` | `HfSource(repo, revision, files, format, select, decode, read)` or `UrlSource(url, sha256, filename, format, ...)`. |
| `convert` | `RawRow -> TaskSpec | NormalizedTask | ImportRejection`; it fixes the task's grader. |
| `version` | Converter revision; bump it when conversion changes outside the converter module. |
| `environment` | `ShellSim()` for conversation tasks, or an `Image` for agentic tasks. |
| `intended_use` | `train` or `eval`. |
| `rubric` | Optional review rubric string, one criterion per paragraph. |
| `controls` | Optional `Controls(golden, negative, memory_mb)` for grader verification. |
| `inputs` | Auxiliary pinned sources, passed by name to `select`, `decode` and `read`. |
| `atlas_id` | Join key into `atlas_catalog.json`; metadata only. |

Families whose members differ only by data are tables: one module builds every
declaration of the family in a loop.

## Graders and controls

A task's grader is one of four kinds:

- `VerifyitGrader` names a stock verifyit mode. Without an environment it grades
  in process; with `environment=IMAGE.requirements()` it grades in a fresh
  machine of that image.
- `ScriptGrader` runs a command in a fresh machine of a pinned image and reads
  its reward from stdout, its exit code, or a reward file. A dataset-specific
  script is a `<name>_grade.py` file next to its declaration, shipped in the
  task's verifier resources under `/tests`. Scorers installed in a grader image
  are called through `verifyit/execution/source_callable.py` with an
  `invocation.json`, built by `taskcompendium.convert.source_scorer`.
- `SessionGrader` marks a task graded by its registered interactive session.
- `NoGrader` records a source evaluator this repository cannot run, with the
  source contract. Its rows never reach `final/`.

Conversion preserves the source's grading semantics. It does not repair
comparators or rewrite tests to accept a reference.

Controls check a grader before its tasks are admitted. For each sampled task the
pipeline grades an empty submission (must score 0), `golden(task)` (must score 1)
and `negative(task)` (must score 0). A golden is a `Reply`, `WorkspaceFiles`, or
an `OracleCommand` run in a fresh machine of the grader image with the task's
oracle files, such as a TaskTrove `solution/solve.sh`. A task without a known
answer returns `None`, which records a skipped golden. In-process numeric, MCQ,
exact and action graders are also checked per task during preparation.

## Source procedure

1. Download the pinned files once per distinct pin
   (`task-curation/download/<hash>`).
2. Draw at most 100 raw rows with a seeded sample, convert them and run cheap
   checks.
3. With a rubric, send bounded batches to the GLM reviewer. More than 90% known
   good judgments accepts the source without reviewing the rest; more than 50%
   known defects rejects it. Both use the whole panel as the denominator.
   Without a rubric, rows are kept as `unreviewed`.
4. In full mode, convert and audit every row of an accepted source.
5. Filter rows into kept, rejected and deferred.
6. With controls, verify a seeded sample of kept rows: every control runs twice
   in fresh machines, and the source passes at a 95% pass fraction.
7. Admit rows and write the outputs.

The campaign runs source procedures in threads over one Zephyr context and
worker pool. `--concurrent-sources` limits whole-source admission and
`--max-workers` the shared worker count. A failed source does not stop the
others; the campaign report records it. Review requests are cached by the
complete request and declared model revision, so rebuilt artifacts do not repeat
identical inference.

## Run a campaign

Run the driver inside an Iris job whose `EnvironmentSpec` includes
`pip_packages=["./lib/taskcompendium[pipeline]"]`. Plan first:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m \
  experiments.post_training.task_curation.driver \
  --review-transport direct-chat --model-revision YOUR_GLM_REVISION \
  --review-cache CACHE_PREFIX --mode sample \
  --max-workers 64 --coordinator-memory 16g --concurrent-sources 10 \
  --normalized-shards 32 \
  --worker-image ghcr.io/marin-community/iris-task@sha256:DIGEST \
  --verification-backend qemu \
  --report-path CAMPAIGN_PREFIX/sample.json
```

`--verification-backend` is `qemu`, `gvisor` or `iris` (with
`--controller-url`). QEMU boots the guest bundle that the worker image carries
for each grader image, as recorded in `images/__init__.py`; given
`--controller-url`, it schedules images without a bundle on Iris instead of
leaving their sources inconclusive. Grading machines never have network
access. Add `--run --base-url PROVIDER_URL` to execute, with
`GLM_BULK_TOKEN` in the driver environment. Full execution requires
`--mode full --sample-report CAMPAIGN_PREFIX/sample.json`; the sample must match
the current source graph and worker image, and only sources whose sample ended
`sampled` or `completed` are processed.

## Outputs

Each source artifact `data/rl/<name>-<hash>` contains:

| Relative path | Contents |
|---|---|
| `download/locators/` | Row locators of the staged source files |
| `normalize/part-*.parquet` | Every converted row: TaskSpec JSON or rejection, and normalization changes |
| `review/part-*.parquet` | Review evidence and filter decisions |
| `review/unprocessed-*.parquet` | Rows outside an unexpanded sample, with the gate's reason |
| `review/report.json` | The source quality decision |
| `verify/part-*.parquet` | Per-row checks, grader readiness and admission |
| `verify/report.json` | Sampled controls, trials and the source verification decision |
| `final/part-*.parquet` | Admitted rows only |
| `manifest.json` | Status, counts, revisions, reports and the source admission |
| `telemetry.json` | Zephyr execution IDs, counters and phase wall times |

Every row's `admission` is `admitted`, `rejected`, `deferred`, `no_grader`,
`deferred:judge` or `unverified`; `final/` holds the admitted rows. Sidecars
join on `task_id`, `source_locator`, `raw_input_sha256` and decoded
`raw_sha256`. The artifact name's hash covers the source pins, inputs, version,
converter module bytes, grader scripts, referenced images, rubric, controls and
pipeline settings, so changing any of them produces a new artifact.

The
[pipeline contract](https://github.com/marin-community/marin/blob/main/lib/taskcompendium/src/taskcompendium/pipeline/README.md)
describes each stage in detail.
