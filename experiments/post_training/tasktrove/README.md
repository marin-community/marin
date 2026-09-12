# TaskTrove conversion

This pipeline converts the pinned
[open-thoughts/TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove) revision into
Harbor tasks with explicit grader contracts. It retains sources and rows that can be normalized
deterministically. `source_verdicts.json` records each source decision, and the release ledger
records every rejected row. `reviewed_defects.json` contains the small set of source/path pairs
whose task contract, golden, or grader failed manual review.

Each retained task contains:

- `instruction.md` and `task.toml`;
- `environment/Dockerfile` with the pinned verifier installed;
- `tests/test.sh`, which invokes `tasktrove-verify`;
- `tests/verifier.toml`, which declares one grader mode; and
- mode-specific hidden data under `tests/`.

`task_format.py` defines this layout. [`tasktrove-verify`](../../../lib/tasktrove-verify/README.md)
defines and executes the grader contract.

## Run

The release version and TaskTrove revision are constants in `pipeline.py`. The verifier commit
comes from Marin launch provenance. Commit and push converter and verifier changes before starting
a release; the pipeline rejects a dirty launch because task Dockerfiles fetch that commit from
GitHub.

```bash
# Print the pinned build plan.
uv run python -m experiments.post_training.tasktrove.pipeline

# Build the release or reuse its cached artifacts.
uv run python -m experiments.post_training.tasktrove.pipeline --run

# Build through one stage.
uv run python -m experiments.post_training.tasktrove.pipeline --stage templates --run
```

Update `PIPELINE_VERSION` for a new conversion release. Update `TASKTROVE_REVISION` and
`RAW_VERSION` together when the input revision changes. The generated Dockerfiles and release
manifest record the clean launch commit used to build the pipeline.

## Pipeline

| stage | module | result |
|---|---|---|
| `raw` | `dataset.py` | pinned source Parquet files, reshuffled into 64 working shards |
| `summaries` | `task_templates.py` | counts and file shapes grouped by normalized task template |
| `templates` | `task_templates.py` | exemplars plus converter coverage for retained sources |
| `converted` | `convert.py` | normalized task binaries, optional solution archives, metadata, and row status |
| `filtered` | `verify.py` | within-source exact deduplication and fail-closed verifier checks |
| `release` | `publish.py` | one task Parquet plus the ledger, manifest, and report |

## Release layout

The release deliberately uses one Parquet file rather than Hive-style source partitions. Each row
is one retained task, and consumers can select a source, family, or other cohort from ordinary
columns before reading the packed task payloads.

| column | meaning |
|---|---|
| `path` | stable task identifier from the source dataset |
| `source` | original TaskTrove source name |
| `family` | broad conversion family assigned by `source_verdicts.json` |
| `template_id` | normalized source template identity |
| `converter` | converter that produced the task |
| `mode` | declared `tasktrove-verify` grader mode |
| `dockerfile_id` | normalized environment/Dockerfile identity |
| `language` | task language when the converter can determine it |
| `tags` | list of selection labels preserved or added during conversion |
| `has_solution` | whether the release includes a shipped oracle solution |
| `task_binary` | gzip-compressed Harbor task archive |
| `solution_binary` | optional gzip-compressed oracle solution archive |

The physical path is `tasks/part-00000.parquet`. Source-specific files can be materialized from
the `source` column when needed; the canonical release stays single-shard so it has one immutable
object, one footer, and one row-count contract.

The current release is under
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9/`:

| path | contents |
|---|---|
| `tasks/part-00000.parquet` | retained tasks and selection columns |
| `ledger.parquet` | rejected source and row decisions |
| `manifest.json` | counts by source, status, converter, grader, tag, and environment |
| `report.md` | tables generated from the manifest |

The authenticated browser at <https://marina.oa.dev/tasktrove/> uses a paginated Marina API. The
server reads the Parquet with its existing S3 credentials. Exact source-only pages use the
manifest's released count and scan row groups only until the requested page is full, avoiding a
cold full-column read. Other filters cache their columns, and every process keeps bounded
row-group, filter-result, and rendered-page caches. Task archives are read one at a time and are
not cached.

## Add a converter

1. Build the `templates` stage. Inspect `coverage.json` and its exemplar under
   `<Marin prefix>/tasktrove/templates/2026.09.10.9/`; the production prefix is
   `s3://marin-us-east-02a/marin`.
2. Add a converter under `converters/` that returns `ConvertedTask` or a specific `Rejected`
   status. Use deterministic parsing; reject rows that need heuristic recovery.
3. Register it in `converters/registry.py`.
4. Add one representative archive under `fixtures/` and a behavior test under `tests/`.
5. Run a Docker audit against source rows:

   ```bash
   uv run python -m experiments.post_training.tasktrove.docker_audit \
     --source <source> --parquet /local/path/to/tasks.parquet \
     --count 20 --out /tmp/tasktrove-audit
   ```

Download the selected source's `tasks.parquet` from the pinned Hugging Face revision or copy that
single file from the `raw` artifact before running the audit. The optional `solution_binary`
contains `solution/solve.sh`; the audit applies it and requires the resulting workspace to score
one. An empty workspace must score zero. SWE solutions that install dependencies require
`--network bridge`.

Rows confirmed broken after release sampling belong in `reviewed_defects.json`, with a reason that
can stand alone in `ledger.parquet`. Use a source-level drop only when the sampled defect is shared
by the source template. Explicit acceptance clauses can be normalized without recovering an
answer; for example, stdin/stdout converters use numeric comparison only when the instruction
states a numeric error tolerance.

## Validate and inspect

```bash
uv run pytest experiments/post_training/tasktrove/tests lib/tasktrove-verify/tests
./infra/pre-commit.py --changed-files --fix

# Export one Parquet row as a Harbor task directory.
uv run python -m experiments.post_training.tasktrove.publish export \
  s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9/tasks <task-path> --dest /tmp/tasktrove-task
```
