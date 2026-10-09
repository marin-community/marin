# RL data curation

This experiment turns pinned RL datasets into TaskSpec parquet files. Every
dataset is declared once, in [datasets/](datasets/README.md), and listed in the
catalog [sources.py](sources.py):

```python
from experiments.post_training.task_curation.sources import all_sources

sources = all_sources()  # stable Atlas ID -> RlDataSource
```

## Atlas export

Build the Atlas input from the same declarations without downloading data or
running converters:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m experiments.post_training.task_curation.export_catalog \
  --output infra/marina/applets/rl_data_catalog/dist/catalog.json
```

The applet build runs this command and bundles the resulting JSON. Its digest
covers the complete inventory. The Atlas does not scrape upstream catalogs on refresh.

## Declaring a dataset

An `RlDataSource` ([source.py](source.py)) combines source information, an optional
review and a conversion recipe:

```python
RlDataSource(
    info=SourceInfo(
        id="MarinSkyRL:math500", title="MATH-500", origin="MarinSkyRL",
        family="math-answer", tags=("rlvr", "single-turn", "benchmark"),
    ),
    pipeline=math500_pipeline,
    review=DataSourceReview(),
)
```

`info.id` preserves Atlas review links; `info.title` is a display label, while
`source.name` identifies the pipeline. `family` groups related tasks. Free-form
tags describe task type, interaction, benchmark or excluded status, licenses,
and source-specific search aliases. `SourceReference(name, revision, url)`
identifies a verifier, or the dataset of an inventory entry without a pipeline.
Runnable sources define their dataset only in `pipeline.source`.

`info.count` counts selected input rows at that dataset revision, before
conversion or curation. Use `None` when the pinned population is unknown.
The Atlas shows **Input rows** beside links to the same pinned conversion input.
TaskTrove archive counts therefore refer to `open-thoughts/TaskTrove`, rather
than counts of successful conversions in a different release. Revisions covered
by authored or saved reviews are compared with these displayed dataset and
verifier revisions; a mismatch hides the rating but preserves review history.

Reproduce whole-file counts from a local Hugging Face download with:

```bash
uv run python -m experiments.post_training.task_curation.count_inputs \
  --snapshot /path/to/tasktrove/repo \
  --revision 02923004846e4e73862c20962f823a6d05100e7a \
  --files '*/tasks.parquet'
```

This verifies each file's HF download commit metadata and reads parquet footers
without network access or data-page reads. Sum only the files selected by the
recipe. A row selector requires a complete count of that selection; do not use
whole-file or QUICK sample counts. Reset or recount counts whenever the input
pin or selection changes. The Nemotron blend counts retain the complete pinned
selection audit, including SWE-Gym membership; card estimates are omitted.

A review remains unrated until an assessment is supplied. Executed reviews and
difficulty measurements remain in the Atlas database. Sources without a recipe
remain in [datasets/unconverted.py](datasets/unconverted.py).
`all_pipelines()` projects runnable recipes for the campaign driver.

A conversion recipe is an `RlDataPipeline` ([pipeline.py](pipeline.py)):

```python
RlDataPipeline(
    name="math500",                      # catalog key and artifact name
    source=HfSource("HuggingFaceH4/MATH-500", "6e4ed1a2...", ("test.jsonl",), SourceFormat.JSONL),
    convert=convert_math500,             # (RawRow, ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection
    version="1",                         # bump when conversion changes outside the hashed files
    environment=ShellSim(),              # or Environment(image="repo@sha256:...") for agentic tasks
    intended_use=IntendedUse.EVAL,
    rubric=MATH500_RUBRIC,               # optional model review; one criterion per paragraph
    controls=MATH_CONTROLS,              # optional grader verification
    grader=None,                         # GRADER_PACKAGES, or another Environment, when a grade script needs packages
    ships=(),                            # directories whose files the converter packages into tasks
    resource_budget_bytes=1_000_000,     # tasks carrying more resource bytes are deferred
)
```

- **Source.** `HfSource(repo, revision, files, format)` or
  `UrlSource(url, sha256, filename, format)`. `select` drops rows, `decode`
  rewrites a row before conversion (for example, unpacking a TaskTrove archive),
  and `read` replaces the format reader. `parts` replaces it for a file that
  several workers read in parts, each yielding its rows with their indices in
  the whole file. Parts report the file's row count, so a sample produces only
  its sampled rows; the Reasoning Gym generator runs this way. `inputs` names
  auxiliary pinned sources. Each callable receives a `ConversionContext`, whose `inputs` holds
  the staged auxiliary sources by name.
- **Converter.** A module-level function `convert(row, context)` that builds the
  task and fixes its grader. `context.grader_environment` is the grader
  environment when the declaration sets `grader`, else `None`;
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
    machine of the grader's environment. A dataset-specific script is a `<name>_grade.py`
    file next to the declaration, and vendored upstream scorers live under
    `datasets/<family>/scorers/` and are listed in `ships`. The converter ships
    the script as `/tests/grade.py`, the row's hidden data as
    `/tests/config.json` and the scorer files at their package paths under
    `/tests`; the script prints its reward as the last nonempty stdout line
    (`StdoutReward`) and exits nonzero when it cannot score;
  - `NoGrader`, when no runnable grader exists. Such rows never reach `final/`.
- **Environment.** `ShellSim()` for conversation tasks; `Environment(image=...)`
  pinned by digest when the agent works in a container. The grader's
  environment is separate: an `Environment` stating packages, which
  [images/](images/README.md) builds.
- **Resource budget.** A task whose decoded resources exceed
  `resource_budget_bytes` is deferred with reason `resources_over_budget`, and
  the manifest counts it.
- **Rubric.** Optional. Without one, rows skip model review and are kept as
  `unreviewed`.
- **Controls.** `Controls(golden)` grades one submission per sampled task: the
  known-correct `golden(task)`, which must score 1, or an empty submission,
  which must score 0, when the task has no golden. Graders that run in a
  machine grade the empty submission in a fresh machine of their environment,
  so it also shows the grader runs. Without controls, verification is skipped
  and rows those graders grade stay out of `final/`. Rows
  graded by an LLM judge are never sampled and reach `final/` without controls.

To add a dataset, copy the closest declaration, set its source, converter,
environment and rubric, add a fixture row to the family test's `ROWS`, and add
the module's `sources()` to [sources.py](sources.py).

## Outputs

Each declaration becomes one cached artifact, `data/rl/<name>-<hash>`. The hash
covers the source pins, auxiliary inputs, `version`, every `*.py` file in the
converter module's directory, every file below `ships`, the grader's built
environment, the agent image, the resource budget, the rubric, the controls
code, and the review and verification settings. A declaration whose `grader`
names no image needs that environment's artifact first; building its source
without one raises `MissingEnvironmentArtifact` with the build command (see
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

A declaration's `grader` states what its scripts need, and the pipeline places
it ([environment.py](environment.py)). A digest-pinned `image` runs in a sandbox
of that image. `apt` packages that the worker image lacks (`WORKER_IMAGE_APT`)
run in a sandbox of an image built for the environment. Every other environment,
including `GRADER_PACKAGES`, runs in a bubblewrap sandbox on the Zephyr
worker (which needs the privileged container profile, `--container-profile`),
in a self-contained Python environment (a uv-managed CPython and a venv) the
worker builds once from the environment's lock and mounts read-only
([environment_runtime.py](environment_runtime.py)).
`--verification-backend` selects how sandbox graders run. `iris`, the default,
schedules each grader machine as an Iris task; inside an Iris job the driver uses
the job's controller, and elsewhere it requires `--controller-url`. `gvisor` runs
the image on the worker's Docker daemon. The artifact identity records the
backend and whether a controller is present, not the controller's address. Add `--run` to
execute, with `GLM_BULK_TOKEN` set; the driver resolves the review endpoint
from the Iris GLM relay job (`--relay-job`, default
`DEFAULT_GLM_RELAY_JOB` in `experiments/post_training/glm.py`) unless `--base-url` is given.
The default, `--mode sample`, is a test run that converts and gates only each
source's panel. For a full run, pass `--mode full` and a new `--report-path`.
A full run processes each source on its own: it makes its own panel quality
decision, converts every row of an accepted source, draws its control sample
from all kept rows, and ends `gated` when its quality or verification gate
rejects the source. Repeat `--source NAME` to run a subset.

Keep `--review-cache` stable across campaigns: reviews are cached by the
complete request and the declared model revision, so a changed artifact can be
rebuilt without repeating identical inference.
