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
are cached under `~/.cache/marin` (`--download-cache` overrides this).
`--input-root PATH` uses already staged files from the declared revision;
`--input NAME PATH` overrides an auxiliary input. These local input options,
`--output-root` and `--download-cache` are accepted only in QUICK mode.

For a local fixture, replace `--input-root` with
`--input-file LOGICAL_PATH LOCAL_FILE`, repeated as needed. The logical path must
match the source declaration. These files become the selected primary inputs;
the manifest records their physical paths and SHA-256 while preserving logical
row IDs. Auxiliary inputs still use overrides or the download cache.

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
