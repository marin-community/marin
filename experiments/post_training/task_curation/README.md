# RL data curation

This experiment turns pinned RL datasets into TaskSpec parquet files. Every
dataset is declared once, in [datasets/](datasets/README.md), and listed in the
catalog [sources.py](sources.py):

```python
from experiments.post_training.task_curation.sources import all_pipelines

pipelines = all_pipelines()  # name -> RlDataPipeline
```

## Declaring a dataset

A declaration is an `RlDataPipeline` ([pipeline.py](pipeline.py)):

```python
RlDataPipeline(
    name="math500",                      # catalog key and artifact name
    source=HfSource("HuggingFaceH4/MATH-500", "6e4ed1a2...", ("test.jsonl",), SourceFormat.JSONL),
    convert=convert_math500,             # (RawRow, ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection
    version="1",                         # bump when conversion changes outside the hashed files
    environment=ShellSim(),              # or AgentImage("repo@sha256:...") for agentic tasks
    intended_use=IntendedUse.EVAL,
    rubric=MATH500_RUBRIC,               # optional model review; one criterion per paragraph
    controls=MATH_CONTROLS,              # optional grader verification
    atlas_id="MarinSkyRL:math500",       # join key into atlas_catalog.json
    grader_image=None,                   # GRADER when a grader runs in a sandbox
    ships=(),                            # directories whose files the converter packages into tasks
    resource_budget_bytes=1_000_000,     # tasks carrying more resource bytes are deferred
)
```

- **Source.** `HfSource(repo, revision, files, format)` or
  `UrlSource(url, sha256, filename, format)`. `select` drops rows, `decode`
  rewrites a row before conversion (for example, unpacking a TaskTrove archive),
  and `read` replaces the format reader. `inputs` names auxiliary pinned
  sources. Each callable receives a `ConversionContext`, whose `inputs` holds
  the staged auxiliary sources by name.
- **Converter.** A module-level function `convert(row, context)` that builds the
  task and fixes its grader. `context.grader_environment` is the built grader
  image when the declaration sets `grader_image`, else `None`;
  `required_grader_environment(context)` returns it or raises. Shared
  techniques live in
  [`taskcompendium.convert`](../../../lib/taskcompendium/src/taskcompendium/convert/):
  `math_answer_task`, `numeric_answer_task`, `mcq_task`, `exact_answer_task`,
  `ifeval_task` and `json_schema_task` build conversation tasks graded in
  process by verifyit; `script_grader` packages a grade script with the row's
  `config.json` and the vendored scorer files it imports; the TaskTrove
  helpers unpack archives.
- **Graders.** A task's grader is one of:
  - a `VerifyitGrader` with no environment, graded in process;
  - a `ScriptGrader` or `VerifyitGrader` with
    `environment=required_grader_environment(context)`, graded in a fresh
    machine of the grader image. A dataset-specific script is a `<name>_grade.py`
    file next to the declaration, and vendored upstream scorers live under
    `datasets/<family>/scorers/` and are listed in `ships`. The converter ships
    the script as `/tests/grade.py`, the row's hidden data as
    `/tests/config.json` and the scorer files at their package paths under
    `/tests`; the script prints its reward as the last nonempty stdout line
    (`StdoutReward`) and exits nonzero when it cannot score;
  - `NoGrader`, when no runnable grader exists. Such rows never reach `final/`.
- **Environment.** `ShellSim()` for conversation tasks; an `AgentImage` pinned
  by digest when the agent works in a container. Agent images are separate from
  the grader image, which [images/](images/README.md) builds from a recipe.
- **Resource budget.** A task whose decoded resources exceed
  `resource_budget_bytes` is deferred with reason `resources_over_budget`, and
  the manifest counts it.
- **Rubric.** Optional. Without one, rows skip model review and are kept as
  `unreviewed`.
- **Controls.** `Controls(golden)` grades one submission per sampled task: the
  known-correct `golden(task)`, which must score 1, or an empty submission,
  which must score 0, when the task has no golden. Sandbox graders grade the
  empty submission in their own image, so it also shows the grader runs.
  Without controls,
  verification is skipped and sandbox-graded rows stay out of `final/`. Rows
  graded by an LLM judge are never sampled and reach `final/` without controls.

To add a dataset, copy the closest declaration, set its source, converter,
environment and rubric, add a fixture row to the family test's `ROWS`, and add
the module's `pipelines()` to [sources.py](sources.py).

## Outputs

Each declaration becomes one cached artifact, `data/rl/<name>-<hash>`. The hash
covers the source pins, auxiliary inputs, `version`, every `*.py` file in the
converter module's directory, every file below `ships`, the digest of the built
grader image, the agent image, the resource budget, the rubric, the controls
code, and the review and verification settings. A declaration with a
`grader_image` needs that image's artifact first; building its source without
one raises `MissingImageArtifact` with the build command (see
[images/](images/README.md)). Downloads are
shared artifacts, `task-curation/download/<hash>`, keyed by the pinned files.

```
download/   normalize/   review/   verify/   final/   manifest.json   telemetry.json
```

`final/` holds rows that passed filtering and have a ready grader: an in-process
verifyit grader, a verifyit judge, or a sandbox grader whose source verification
passed. `manifest.json` records counts, the quality and verification reports,
and the source's `admission` (`admitted` or `none`). The [pipeline contract](../../../lib/taskcompendium/src/taskcompendium/pipeline/README.md)
describes each stage.

## Running a campaign

Run [driver.py](driver.py) inside one Iris driver job whose `EnvironmentSpec`
includes `pip_packages=["./lib/taskcompendium[pipeline]"]`. The campaign keeps
one Zephyr worker pool across all sources; `--concurrent-sources` (at least 10)
limits how many sources run at once. Plan first:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m \
  experiments.post_training.task_curation.driver \
  --review-mode chat --model-revision YOUR_GLM_REVISION \
  --review-cache CACHE_PREFIX --max-workers 64 --coordinator-memory 16g \
  --concurrent-sources 10 --normalized-shards 32 \
  --worker-image ghcr.io/marin-community/iris-task@sha256:DIGEST \
  --mode sample --report-path CAMPAIGN_PREFIX/sample.json
```

`--verification-backend` selects how sandbox graders run. `iris`, the default,
schedules each grader machine as an Iris task; inside an Iris job the driver uses
the job's controller, and elsewhere it requires `--controller-url`. `gvisor` runs
the image on the worker's Docker daemon. The artifact identity records the
backend and whether a controller is present, not the controller's address. Add `--run` to
execute, with `GLM_BULK_TOKEN` set; the driver resolves the review endpoint
from the Iris GLM relay job (`--relay-job`, default
`DEFAULT_GLM_RELAY_JOB` in `experiments/post_training/glm.py`) unless `--base-url` is given.
For full execution, pass `--mode full`, a new `--report-path` and
`--sample-report CAMPAIGN_PREFIX/sample.json`; only sources whose sample ended
`sampled` or `completed` are processed, and they reuse their sample's control
trials. Repeat `--source NAME` to run a subset.

Keep `--review-cache` stable across campaigns: reviews are cached by the
complete request and the declared model revision, so a changed artifact can be
rebuilt without repeating identical inference.
