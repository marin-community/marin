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
    convert=convert_math500,             # RawRow -> TaskSpec | NormalizedTask | ImportRejection
    version="1",                         # bump when conversion changes outside the converter module
    environment=ShellSim(),              # or an Image from images/ for agentic tasks
    intended_use=IntendedUse.EVAL,
    rubric=MATH500_RUBRIC,               # optional model review; one criterion per paragraph
    controls=MATH_CONTROLS,              # optional grader verification
    atlas_id="MarinSkyRL:math500",       # join key into atlas_catalog.json
)
```

- **Source.** `HfSource(repo, revision, files, format)` or
  `UrlSource(url, sha256, filename, format)`. `select` drops rows, `decode`
  rewrites a row before conversion (for example, unpacking a TaskTrove archive),
  and `read` replaces the format reader. `inputs` names auxiliary pinned
  sources passed to those callables.
- **Converter.** A module-level function that builds the task and fixes its
  grader. Shared techniques live in
  [`taskcompendium.convert`](../../../lib/taskcompendium/src/taskcompendium/convert/):
  `math_answer_task`, `numeric_answer_task`, `mcq_task`, `exact_answer_task`,
  `ifeval_task` and `json_schema_task` build conversation tasks graded in
  process by verifyit; `source_scorer_package` calls a scorer installed in a
  grader image; the TaskTrove helpers unpack archives.
- **Graders.** A task's grader is one of:
  - a `VerifyitGrader` with no environment, graded in process;
  - a `ScriptGrader` or `VerifyitGrader` with `environment=IMAGE.requirements()`,
    graded in a fresh machine of that image. A dataset-specific script is a
    `<name>_grade.py` file next to the declaration, read as bytes by the
    converter and shipped in the task's verifier resources (mounted at `/tests`);
  - `NoGrader`, when no runnable grader exists. Such rows never reach `final/`.
- **Environment.** `ShellSim()` for conversation tasks; an `Image` from
  [images/](images/README.md) when the agent works in a container.
- **Rubric.** Optional. Without one, rows skip model review and are kept as
  `unreviewed`.
- **Controls.** `Controls(golden, negative)` grades a known-correct and a
  known-wrong submission per sampled task, plus an empty one. Without controls,
  verification is skipped and sandbox-graded rows stay out of `final/`.

To add a dataset, copy the closest declaration, set its source, converter,
environment and rubric, add a fixture row to the family test's `ROWS`, and add
the module's `pipelines()` to [sources.py](sources.py).

## Outputs

Each declaration becomes one cached artifact, `data/rl/<name>-<hash>`. The hash
covers the source pins, auxiliary inputs, `version`, the converter module's
bytes, the `*_grade.py` scripts beside it, the images it references, the rubric,
the controls code, and the review and verification settings. Downloads are
shared artifacts, `task-curation/download/<hash>`, keyed by the pinned files.

```
download/   normalize/   review/   verify/   final/   manifest.json   telemetry.json
```

`final/` holds rows that passed filtering and have a ready grader: an in-process
verifyit grader, or a sandbox grader whose source verification passed.
`manifest.json` records counts, the quality and verification reports, and the
source's `admission` (`admitted`, `deferred:judge` for judge-graded sources, or
`none`). The [pipeline contract](../../../lib/taskcompendium/src/taskcompendium/pipeline/README.md)
describes each stage.

## Running a campaign

Run [driver.py](driver.py) inside one Iris driver job whose `EnvironmentSpec`
includes `pip_packages=["./lib/taskcompendium[pipeline]"]`. The campaign keeps
one Zephyr worker pool across all sources; `--concurrent-sources` (at least 10)
limits how many sources run at once. Plan first:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m \
  experiments.post_training.task_curation.driver \
  --review-transport direct-chat --model-revision YOUR_GLM_REVISION \
  --review-cache CACHE_PREFIX --max-workers 64 --coordinator-memory 16g \
  --concurrent-sources 10 --normalized-shards 32 \
  --worker-image ghcr.io/marin-community/iris-task@sha256:DIGEST \
  --verification-backend qemu \
  --mode sample --report-path CAMPAIGN_PREFIX/sample.json
```

`--verification-backend` selects how sandbox graders run: `qemu` boots the guest
bundle that the worker image carries for each grader image, `gvisor` runs the
image on the worker's Docker daemon, and `iris` schedules it on
`--controller-url`. An image the backend cannot run leaves its source
inconclusive. Add `--run --base-url URL` to execute, with `GLM_BULK_TOKEN` set.
For full execution, pass `--mode full`, a new `--report-path` and
`--sample-report CAMPAIGN_PREFIX/sample.json`; only sources whose sample ended
`sampled` or `completed` are processed, and they reuse their sample's control
trials. Repeat `--source NAME` to run a subset.

Keep `--review-cache` stable across campaigns: reviews are cached by the
complete request and the declared model revision, so a changed artifact can be
rebuilt without repeating identical inference.
