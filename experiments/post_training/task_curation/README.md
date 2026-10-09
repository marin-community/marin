# RL data curation

[Dataset declarations](datasets/README.md) feed TaskSpec conversion and the RL
Data Atlas through [all_sources()](sources.py). See the
[reference](../../../docs/references/task-curation.md) for source fields, grading,
reviewed SAMPLE/FULL campaigns and artifact outputs.

## Local conversion loop

From the repository root:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.quick \
  --source tasktrove-calendar --source tasktrove-math_prism \
  --output-root /tmp/task-curation-pass-1
```

QUICK uses local Zephyr, downloads pinned inputs once, and converts every selected
row. It preserves converter rejections and known reviewed defects; it skips
review, grader execution, resource admission and deduplication. Outputs are
`normalize/*.parquet`, `manifest.json` and `campaign.json`, with no admitted
`final/` view. Failed sources are recorded while the rest continue.

Repeat `--source` for a cohort and use a fresh output root each pass. Downloads
are cached under `~/.cache/marin` (`--download-cache` overrides this).
`--input-root PATH` uses already staged files from the declared revision;
`--input NAME PATH` overrides an auxiliary input.

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

SWEsmith, SWE-rebench, Code Contests and TACO use the actor's shared Harbor
environment and can omit `--grader-image`. Other supported tasks use a separate
verifier image with the declared grader dependencies. Actor-installed packages
and changes outside submission files do not transfer to separate verifiers.
Export restores source actor recipes and bundles current Verifyit; use
`--verifyit-package-root` to select its source package.

Export builds no images and runs no graders. Native configuration parsing and
content comparisons do not establish runtime equivalence or make mutable image
tags and build-time downloads reproducible.

[`harbor_export_step`](tasktrove/export.py) binds this export to a normalized
artifact and records verifier payload identity. For the smoke input, first run
QUICK with `--source tasktrove-nl2bash --output-root /tmp/curation-quick`. Then
adopt that output and plan export plus training:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.rl_smoke --version 2026.10.09 \
  --normalized-name data/rl/tasktrove-nl2bash --normalized-version 2026.10.09 \
  --normalized-source /tmp/curation-quick/tasktrove-nl2bash \
  --grader-image '<registry/image>@sha256:<digest>'
```

The default only prints the graph. `--run` executes it; the normalized input must
be accessible to the coordinator. See [images/](images/README.md) for grader builds.

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
