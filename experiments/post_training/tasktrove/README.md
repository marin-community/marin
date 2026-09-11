# TaskTrove conversion

This pipeline converts the pinned
[open-thoughts/TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove) revision into
Harbor tasks with explicit grader contracts. It retains sources and rows that can be normalized
deterministically. `source_verdicts.json` records each source decision, and the release ledger
records every rejected row.

Each retained task contains:

- `instruction.md` and `task.toml`;
- `environment/Dockerfile` with the pinned verifier installed;
- `tests/test.sh`, which invokes `tasktrove-verify`;
- `tests/verifier.toml`, which declares one grader mode; and
- mode-specific hidden data under `tests/`.

`task_format.py` defines this layout. [`tasktrove-verify`](../../../lib/tasktrove-verify/README.md)
defines and executes the grader contract.

## Run

The release version, TaskTrove revision, and verifier commit are constants in `pipeline.py`.

```bash
# Print the pinned build plan.
uv run python -m experiments.post_training.tasktrove.pipeline

# Build the release or reuse its cached artifacts.
uv run python -m experiments.post_training.tasktrove.pipeline --run

# Build through one stage.
uv run python -m experiments.post_training.tasktrove.pipeline --stage templates --run
```

Update `PIPELINE_VERSION` for a new conversion release. Update `TASKTROVE_REVISION` and
`RAW_VERSION` together when the input revision changes. Update `VERIFY_TOOL_REF` when generated
Dockerfiles must install a new verifier commit.

## Pipeline

| stage | module | result |
|---|---|---|
| `raw` | `dataset.py` | pinned source Parquet files, reshuffled into 64 working shards |
| `summaries` | `task_templates.py` | counts and file shapes grouped by normalized task template |
| `templates` | `task_templates.py` | exemplars plus converter coverage for retained sources |
| `converted` | `convert.py` | normalized task binaries, optional solution archives, metadata, and row status |
| `filtered` | `verify.py` | within-source exact deduplication and fail-closed verifier checks |
| `release` | `publish.py` | one task Parquet plus the ledger, manifest, and report |

The current release is under
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.7/`:

| path | contents |
|---|---|
| `tasks/part-00000.parquet` | retained tasks and selection columns |
| `ledger.parquet` | rejected source and row decisions |
| `manifest.json` | counts by source, status, converter, grader, tag, and environment |
| `report.md` | tables generated from the manifest |

The authenticated browser at <https://marina.oa.dev/tasktrove/> reads these files directly with
footer and byte-range requests.

## Add a converter

1. Build the `templates` stage. Inspect `coverage.json` and its exemplar under
   `<Marin prefix>/tasktrove/templates/2026.09.10.7/`; the production prefix is
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

## Validate and inspect

```bash
uv run pytest experiments/post_training/tasktrove/tests lib/tasktrove-verify/tests
./infra/pre-commit.py --changed-files --fix

# Export one Parquet row as a Harbor task directory.
uv run python -m experiments.post_training.tasktrove.publish export \
  s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.7/tasks <task-path> --dest /tmp/tasktrove-task
```
