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
covers the complete source inventory. Metadata is reviewed in the repository;
`recorded_at` identifies the metadata capture date. The Atlas no longer scrapes
upstream catalogs on refresh.

Inventory counts and reviews refer to the declared release population. For
TaskTrove, that release can differ from the native archive used for conversion.
The export retains both the inventory revision and the recipe's input revision
under `pipeline`.

## Declaring a dataset

An `RlDataSource` ([source.py](source.py)) combines inventory metadata, an optional
review and a conversion recipe:

```python
RlDataSource(
    metadata=DataSourceMetadata(id="MarinSkyRL:math500", name="math500", origin="MarinSkyRL"),
    pipeline=math500_pipeline,
    review=DataSourceReview(),
)
```

The metadata ID preserves Atlas review links. Optional fields record counts,
classifications and pinned evidence. Presentation defaults come from the recipe's
source when omitted. A review remains unrated until an assessment is supplied;
executed reviews and difficulty measurements remain in the Atlas database.
Sources without a recipe remain in [datasets/unconverted.py](datasets/unconverted.py).
`all_pipelines()` projects the runnable recipes for the campaign driver.

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

## Local conversion loop

Quick mode converts every selected row from staged files into TaskSpec parquet.
It skips model review, grader controls, resource admission, mechanical checks
and deduplication. Explicit converter rejections remain. The output includes every
input row in `normalize/`, with either `task_json` or a typed rejection, and
counts and elapsed time in `manifest.json`. It has no admitted `final/` view.

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' \
  python -m experiments.post_training.task_curation.quick \
  --source tasktrove-calendar \
  --source tasktrove-math_prism \
  --input-root /path/to/staged/tasktrove \
  --output-root /tmp/task-curation-pass-1
```

The staged root contains the source's declared relative paths, for example
`laion__nemotron-gym-agent-calendar-v2/tasks.parquet`. Use bytes from the
source's pinned revision. The command runs a local Zephyr pool and keeps its
scratch files under the output root. Repeat `--source` to reuse the pool across
sources. Supply auxiliary inputs with `--input NAME /path/to/staged/input`.
Choose a fresh output root for each pass; existing source outputs are refused.
Failures are recorded in `campaign.json` while the remaining sources continue;
the command exits unsuccessfully if any source failed.

`conversions.convert_source(source, mode=SourceProcessingMode.QUICK, ...)`
accepts an `RlDataSource` from `all_sources()` and runs its bound conversion recipe.
`SAMPLE` and `FULL` use the same function with a matching `SourcePipelineConfig`
and the resolved grader environment. This entry point and reviewed campaign
artifacts share the `pipeline.convert_pipeline` dispatcher. Quick mode needs no review config or
credentials. It records a declared grader image or local dependency lock
without building or executing the environment. Sources that declare only PyPI
pins need an explicit resolved grader environment before quick conversion.

Compare row counts by source and retain the rejection categories. Quick counts
can exceed the published TaskTrove release because that release also removes
duplicates, applies reviewed defects and routes some MCQA rows away from RL.
`original_path` preserves TaskTrove's archive key for later comparisons.

### Harbor compatibility view

Lower QUICK output to Task Trove's 12-column parquet format:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.harbor \
  --input-root /tmp/curation-quick/tasktrove-calendar \
  --output-root /tmp/curation-harbor/calendar \
  --grader-image '<registry/image>@sha256:<digest>'
```

The source name in the QUICK manifest selects the registry declaration, which
supplies the exported family and Atlas ID. The verifier image is an explicit
runtime input. It must contain the dependencies required by the task's grader
package lock. Export does not build or run images;
its manifest records that dependency parity and runtime behavior remain
unverified. Current verifyit code is bundled under the hidden `tests/` directory.
The exporter supports plain-text answers with recorded original file-delivery
instructions, and file submissions graded by verifyit or an archived Harbor
`test.sh` emitting `reward.txt`. Other contracts produce explicit rejection
records. Rejections from normalization remain in the export manifest.

Each task keeps its source, original archive path, and TaskSpec ID. The agent image
contains only public resources; the verifier runs separately and receives the
declared output files. Oracle files are stored in `solution_binary`. Generated
`task.toml` files are parsed with Harbor's native configuration model.

Compare the output with a downloaded, pinned release manifest:

Download `manifest.json` from
`https://huggingface.co/datasets/open-athena/task-trove/resolve/<release-commit>/manifest.json`
and pass that same commit to `--golden-revision`.

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.compare_harbor \
  --tasks /tmp/curation-harbor/calendar/tasks.parquet \
  --source laion__nemotron-gym-agent-calendar-v2 \
  --golden-manifest /path/to/tasktrove-manifest.json \
  --golden-revision '<release-commit>' \
  --output /tmp/curation-harbor/calendar/comparison.json
```

Repeat `--source` for every expected source, including sources with no generated
rows. The comparison records the golden manifest's hash and supplied release
revision. It checks every generated archive, source identity, schema, and source
count. Count differences remain visible: QUICK skips release deduplication and
verification, so its output may include rows absent from the released dataset.
The comparison does not establish grader equivalence or compare golden task
binaries.
