# Task curation

Task curation turns pinned RL datasets into TaskSpec parquet files. Each dataset
is declared once as an `RlDataSource` under
[`experiments/post_training/task_curation/datasets/`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/datasets/README.md),
and the catalog
[`sources.py`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/sources.py)
lists every declaration:

```python
from experiments.post_training.task_curation.sources import all_sources

sources = all_sources()  # stable Atlas ID -> RlDataSource
```

Building the catalog performs no downloads, inference or job submission.

## Dataset-owned invocation

`RlDataSource.pipeline` is one graph-building callable:

```python
def curate(source: RlDataSource[MyConfig], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    ...
```

`PipelineOptions` contains an explicit QUICK, SAMPLE or FULL mode, the campaign
runtime, input overrides and optional concrete recipe settings. The executor
receives output/cache paths separately. Each dataset constructs its own dependencies and chooses its
conversion, validation, review and control stages. Its run closure accesses the
active Zephyr context through the runtime. A pipeline can generate inputs or
read local files without reviewer credentials, grader builds or sandbox setup.

`CampaignArtifact.result` carries a `PipelineResult`: the existing `SourceStatus`,
named output and evidence paths, and the stages that actually ran. It requires no
`final` or admitted view. Exceptions enter the campaign's failure report while
peer results remain available.

`process_rows(source: RlDataSource[CurationRecipe], options)` is a reusable callable. It
chooses pinned downloads and grader dependencies, then invokes the existing
source processor. Review endpoints and credentials resolve only during its
chosen review path; controller lookup occurs when its grader machine is used.
QUICK executes the complete declared dependency graph through StepRunner. Its
terminal uses the executor's mutable `dev` rerun policy at the explicit output
root; ingestion dependencies retain their versions and caches. Recipe conversion
refuses existing `normalize/` or `manifest.json` payloads, including partial
outputs. Other datasets own their overwrite behavior.

`RlDataSource` stores its typed dataset-owned `config` and one plain callable.
Configurations expose catalog facts (`name`, `version`, `dataset`, `files`) without
converter, rubric or service requirements. Pinned inputs expose their reference
directly. Atlas reads these facts without invoking or inspecting the callable.

## Package organization

| Package | Responsibility |
|---|---|
| `experiments/post_training/task_curation/datasets/` | Dataset declarations, converters, rubrics, controls, `*_grade.py` grader scripts and vendored `scorers/` |
| `experiments/post_training/task_curation/environment.py` | `Environment`, what a machine must provide, and `placement`, which decides where it runs |
| `experiments/post_training/task_curation/images/` | `build.py`, which builds declared environments and records each as an artifact, and the build CLI |
| `experiments/post_training/task_curation/environment_runtime.py` | The uv environments that local graders run in on the Zephyr worker |
| `experiments/post_training/task_curation/sources.py` | The source registry, `all_sources()` and `runnable_sources()` |
| `experiments/post_training/task_curation/source.py`, `export_catalog.py` | Source metadata, authored reviews and generated Atlas JSON |
| `experiments/post_training/task_curation/config.py` | Invocation options and concrete recipe settings; service binding belongs to `process_rows` |
| `experiments/post_training/task_curation/pipeline.py` | `CurationRecipe`, `process_rows` and its `data/rl/<name>-<hash>` artifact |
| `experiments/post_training/task_curation/pipeline.py`, `local.py` | Source-level mode dispatch and local mechanical conversion |
| `experiments/post_training/task_curation/driver.py`, `campaign.py` | Campaign options, shared pool and per-source results |
| `taskcompendium.convert` | Conversion techniques shared by declarations |
| `taskcompendium.pipeline` | Sampling, review, filtering, verification and outputs |
| `taskcompendium.runtime` | Grading in fresh Shellbox machines |
| `verifyit` | Stock grading modes, in process or in a grader machine |
| `shellbox` | Isolated machines and their backends |

TaskCompendium does not import experiments, name datasets, or construct
ArtifactSteps.

## Declarations

An `RlDataSource` has `info`, `review`, a typed `config` and an optional `pipeline`.
`SourceInfo` requires a stable `id`, display `title` and `origin`; it adds family,
search tags, notes and an optional input-row count. A `SourceReference` groups a
name, revision and URL for the verifier or inventory dataset. Runnable sources
expose their pinned input through `config.dataset`. `DataSourceReview` records an authored grade, evidence URL, date and
the dataset/verifier revisions it covers. Its default is unrated. Executed
reviews and difficulty measurements remain in the Atlas database.

The applet build exports this registry to `dist/catalog.json`:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m experiments.post_training.task_curation.export_catalog \
  --output infra/marina/applets/rl_data_catalog/dist/catalog.json
```

The exporter alone maps Python declarations to Atlas fields. Dataset links and
**Input rows** describe the same pinned conversion input, before conversion or
curation. Unknown counts remain unknown. A changed dataset or verifier revision
makes reviews covering a different revision stale while preserving their history.
See [RL Data Atlas](rl-data-atlas.md) for publishing and saved reviews, and the
[experiment README](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/README.md)
for offline count regeneration.

The `CurationRecipe` configuration has these fields:

| Field | Meaning |
|---|---|
| `name` | Catalog key and artifact name. |
| `source` | `HfSource(repo, revision, files, format, select, decode, read)` or `UrlSource(url, sha256, filename, format, ...)`. |
| `convert` | `(RawRow, ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection`; it fixes the task's grader. |
| `version` | Converter revision; bump it when conversion or bundled grader code changes. |
| `intended_use` | `train` or `eval`. |
| `rubric` | Optional review rubric string, one criterion per paragraph. |
| `controls` | Optional `Controls(golden, memory_mb)` for grader verification. |
| `inputs` | Auxiliary pinned sources, staged by name in `ConversionContext.inputs`. |
| `grader` | The `Environment` that grade scripts need, such as `GRADER_PACKAGES` from `datasets/environments.py`. The pipeline builds it and decides where it runs (see [Environments](#environments)); the converter reads the result as `ConversionContext.grader_environment`. |
| `resource_budget_bytes` | Decoded resource bytes admitted by reviewed modes, default 1,000,000; larger tasks are deferred as `resources_over_budget`. Quick mode skips this budget. |

Families whose members differ only by data are tables: one module builds every
declaration of the family in a loop.

`convert`, `select`, `decode`, `read` and `parts` each receive a `ConversionContext` with
two fields: `inputs`, the staged auxiliary sources, and `grader_environment`, the
`EnvironmentRequirements` of the declared `grader` as the pipeline placed it, or
`None` when the declaration names no `grader`.
`required_grader_environment(context)` returns the environment or raises.

## Environments

An `Environment` declares grader build inputs. The converter records agent and
grader requirements on each TaskSpec; execution selects suitable Shellbox machines.

| Field | Meaning |
|---|---|
| `pypi` | Exact `name==version` pins, compiled at build time with `uv pip compile --generate-hashes` for Python 3.12 on `x86_64-unknown-linux-gnu`. |
| `lock` | A uv-compiled requirements lock with hashes, used verbatim. It excludes `pypi`. |
| `apt` | Debian package names. |
| `data` | Downloads; only `nltk:<package>` is supported. |
| `image` | A digest-pinned image used as-is. It excludes every other field. |

| Declaration | Placement | Recorded `EnvironmentRequirements` |
|---|---|---|
| `image` set | A sandbox of that image | `command_semantics=LINUX_PROCESS`, `docker_image=image` |
| `apt` names a package outside `WORKER_IMAGE_APT` | A sandbox of an image built for the environment | `command_semantics=LINUX_PROCESS`, `docker_image=<built digest>` |
| Anything else | A bubblewrap sandbox on the Zephyr worker | `command_semantics=LINUX_PROCESS`, `packages_lock=<lock URL>` |

`WORKER_IMAGE_APT` in `environment.py` lists the Debian packages the `task`
stage of `lib/iris/Dockerfile` installs, such as `build-essential` and `git`.
An environment that needs only those runs in the worker. `GRADER_PACKAGES`, the
lock every grade script in the catalog imports with the NLTK `punkt_tab` and
`wordnet` data, runs in the worker. The TaskTrove competitive-programming
sources declare `COMPILER_GRADER_PACKAGES`, which adds `build-essential` for
C++ submissions; the worker image provides it, so they also run in the worker.
Converters declare agent requirements independently. A simulated shell requires
explicit `SHELL_SIMULATOR` semantics; an absent image does not select simulation.
Conversation-only tasks need no machine. Memory settings are advisory sizing hints.

## Graders and controls

A task's grader is one of four kinds:

- `VerifyitGrader` names a stock verifyit mode. Without an environment it grades
  in process; with `environment=required_grader_environment(context)` it grades
  in a fresh machine of the grader's environment.
- `ScriptGrader` runs a command in a fresh machine of the grader's environment. Archived
  TaskTrove graders keep the archive's `tests/test.sh`, which writes its reward
  to a file (`FileReward`).
- `SessionGrader` marks a task graded by its registered interactive session.
- `NoGrader` records a source evaluator this repository cannot run, with the
  source contract. Its rows never reach `final/`.

TaskTrove math uses verifyit's `math` mode in the grader sandbox. It compares the
last boxed answer, falling back to the last nonempty line, with the archive's
typed reference; tuples and lists use ordered member comparison. Archived
scorer and runner code stays hidden as source evidence and is never executed on
model answers. Parsing and unit handling follow verifyit; source scorer parity
is not guaranteed. Archived oracle scripts still run in a sandbox to produce
golden control answers when present.

A source whose scorer is upstream code grades with a script. The script is a
`<name>_grade.py` file next to the declaration, and the upstream scorer is
vendored under `datasets/<family>/scorers/`. `taskcompendium.convert.script_grader` builds the
package:

- `grade_script` installs the script as `/tests/grade.py`, with the files it
  imports; `shipped_files` places vendored scorer files at their package paths
  under `/tests`, so the script imports them as upstream does;
- `script_package` adds the row's hidden data as `/tests/config.json` (sorted
  keys) and grades with `ScriptGrader(argv=("python3", "/tests/grade.py"),
  cwd="/", reward=StdoutReward())` in the grader's environment.

The script puts `/tests` on its import path, reads `config.json` and the reply
at `/app/answer.txt` (or the conversation at `/tests/conversation.json`), and
prints the reward, fractional when the scorer is, as its last nonempty stdout
line. The runtime keeps the first 16 KiB of stdout, so the script keeps its
output below that. It exits nonzero when it cannot import its scorer or a
dependency, which the runtime reports as an infrastructure error, never a zero
reward. The grader's environment supplies third-party dependencies only; scorer code
always ships with the task.

Conversion preserves the source's grading semantics. It does not repair
comparators or rewrite tests to accept a reference.

Controls check a grader before its tasks are admitted. For each sampled task the
pipeline grades exactly one submission: `golden(task)`, which must score 1, or,
when the declaration has no `golden` or it returns `None` because the task has no
known answer, an empty submission, which shows that the grader runs. A grader that
runs in a machine grades the empty submission in a fresh machine of its environment,
staged as for a rollout whose agent replied with empty text and wrote nothing: an
empty answer file where the grader reads one, a conversation ending in the empty
reply, and an empty workspace. The empty control passes when the grader runs and
scores 0 or rejects the submission. When the grader runs and gives the empty
submission a positive reward, the task is defective: an empty reply satisfies it,
as it satisfies an instruction-following constraint such as "use no commas". The
control records `defect`, which rejects the row with reason `check:empty`; the
trial still counts as checked and passed toward the source's pass fraction, and
the verification report counts it under `defective`. A golden that does not score
1 fails, which rejects the row and counts against the source. A grader that
crashes or writes no reward is an infrastructure error. An in-process grader
scores the empty reply directly. A golden is a `Reply`, `WorkspaceFiles`, or an
`OracleCommand`, such as a TaskTrove `solution/solve.sh`,
run with the task's worker and oracle files in a fresh machine of the task's agent
image, whose tools and directories the oracle expects. A task without an agent
image, such as a conversation task, runs its oracle in the grader's machine. The
oracle's output is then graded like any other submission. In-process numeric,
MCQ, exact and action graders are also checked per task during preparation.

Sources graded by an LLM judge (verifyit's judge mode), such as the TaskTrove
judged, open-QA and MultiChallenge sources, have no control path yet. When every
task the panel converts is judge-graded, verification samples nothing and
records `skipped` with reason `judge grader; no control path yet`, and kept rows
are admitted like rows of in-process graders.

## Conversion modes

Each dataset's callable receives `PipelineOptions` with an explicit mode. The
single `driver` CLI requires `--mode quick|sample|full` and
prints a plan unless `--run` is supplied. QUICK runs locally without review or
controller settings. See the campaign quickstart below for input overrides.
The recipe implementation shares conversion and the normalized schema across
its modes. QUICK retains TaskSpecs
and typed rejections, skipping admission, fingerprints, review, deduplication
and grading. It records declared images or locks without building environments.

Recipe SAMPLE and FULL bind `SourcePipelineConfig` and a resolved grader
environment. Other callables choose their own configuration. The recipe's FULL
mode reviews a bounded panel, then
reuses those conversions when its quality gate allows expansion. The procedure
below describes those reviewed stages. See the
[campaign quickstart](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/README.md)
for local overrides, Harbor export and pinned content comparison commands.

## Source procedure

Sample and full modes run the reviewed procedure below.

1. Download the pinned files once per distinct pin
   (`task-curation/download/<hash>`).
2. Draw at most 100 raw rows with a seeded sample, convert them and run cheap
   checks.
3. With a rubric, send the panel in bounded batches to the GLM reviewer. More
   than 50% known defects, over the whole panel, rejects the source; otherwise
   it is accepted. Uncertain judgments, unsupported conversions and duplicate
   rows are not defects. Rows outside the panel are never reviewed individually.
   Without a rubric, rows are kept as `unreviewed`.
4. In full mode, convert and audit every row of an accepted source.
5. Filter rows into kept, rejected and deferred.
6. With controls, verify a seeded sample of at most 20 kept rows
   (`--verification-sample-size`): each task's one control runs once in a fresh
   machine, rerun up to twice more after an infrastructure error, and the source
   passes at a 95% pass fraction. Judge-graded sources skip this step.
7. Admit rows and write the outputs.

For a standalone task batch, call
`review_tasks(tasks, rubric, reviewer, cached=None)` from
`taskcompendium.pipeline.review`. It returns a `ReviewBatchResult` with `.reviews`
and `.attempts`. The caller's sink owns serialization and output. Pipeline sinks
write one evidence record per batch alongside audit rows in the same execution,
without repeating provider calls.

Verification telemetry records `select`, `trials`, and `row_gate` executions.
The gate writes the complete audit and accepted rows together and returns
manifest counts, avoiding separate export and count executions. Control trials
remain parallel on workers. Review-cache telemetry separates descriptor lookup
from payload reads; descriptor shards are read with at most four threads per
lookup, with whole-fragment prefetch disabled.

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
  --review-mode chat --model-revision YOUR_GLM_REVISION \
  --review-cache CACHE_PREFIX --mode sample \
  --max-workers 64 --coordinator-memory 16g --concurrent-sources 10 \
  --normalized-shards 32 \
  --image-backend gvisor \
  --worker-image ghcr.io/marin-community/iris-task@sha256:DIGEST \
  --report-path CAMPAIGN_PREFIX/sample.json
```

Build the [registered environments](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/images/README.md#building)
first, with the same `MARIN_PREFIX`:

```bash
uv run python -m experiments.post_training.task_curation.images --all
uv run python -m experiments.post_training.task_curation.images --identity IDENTITY_PREFIX
```

Each environment without an `image` becomes the artifact
`images/env-<identity[:16]>`, which stores its hash lock as `requirements.lock`.
The identity hashes the `pypi` pins or the lock's bytes, `apt`, `data`, Python
3.12, the platform, the digest-pinned base image, the files of `verifyit`
(which every built environment puts on the grader's import path) and the
placement, so a rerun with an unchanged declaration does nothing. An
environment placed in a built image also gets a generated Dockerfile: the
pinned `iris-task` base, `apt-get install` of `apt`, `uv pip sync
--require-hashes` of the lock, the NLTK data and `verifyit`. The build pushes it
as `ghcr.io/marin-community/iris-task:task-curation-env-<identity[:16]>` and
records the digest. An image build needs a `docker login` for ghcr.io; every
build needs, for a CoreWeave `MARIN_PREFIX`, the `CW_KEY_ID` and
`CW_KEY_SECRET` pair in the environment. Planning a source whose environment
has no artifact raises `MissingEnvironmentArtifact` with the `--identity`
command that builds it.

Local graders run in bubblewrap sandboxes on the Zephyr worker, each over a
private root, so the worker pods need Iris's privileged container profile
(`--container-profile CONTAINER_PROFILE_PRIVILEGED`). On first use, each worker downloads the environment's lock from its artifact and builds a
self-contained Python environment (a uv-managed CPython 3.12 and a venv) under
`/tmp/task-curation-env-<identity>`, with the NLTK data and `verifyit`; the
sandbox mounts only that directory and the system directories, and concurrent
graders on one host build it once.
Image grading requires `--image-backend gvisor` or `--image-backend daytona`.
The gVisor launcher uses the enclosing Iris job's controller, or `--controller-url`
outside a job, to create sandbox jobs. Daytona creates remote sandboxes through its API.
Neither path requires a Docker daemon in the Zephyr worker. Without an image backend,
controls can still use package locks through Local or explicit simulator requirements
through ShellSim. Programmatic callers pass grading machines through `RecipeSettings.machines`.
Grading machines never have network access. Add `--run` to execute, with `GLM_BULK_TOKEN` in the driver environment;
the review endpoint is resolved from the Iris GLM relay job (`--relay-job`) unless
`--base-url` overrides it. `--mode sample` is a test run that
converts and gates only each source's panel. `--mode full` runs each source on
its own: it makes its own panel quality decision, converts every row of an
accepted source, draws its control sample from all kept rows, and ends `gated`
when its quality or verification gate rejects the source. Repeat
`--source NAME` to run a subset.

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

Every row's `admission` is `admitted`, `rejected`, `deferred`, `no_grader` or
`unverified`; `final/` holds the admitted rows. Sidecars
join on `task_id`, `source_locator`, `raw_input_sha256` and decoded
`raw_sha256`. Artifact identities cover source pins, inputs, explicit code versions,
grader environments, resource budgets, rubrics, controls and pipeline settings.
Python source files are not hashed. Bump the recipe version for converter or bundled
grader changes, the corresponding stage/control revision for shared pipeline changes,
and the export version for Harbor lowering changes. Download versions are independent.
The manifest counts rows deferred for `resources_over_budget` with the other
normalization reasons.

The
[pipeline contract](https://github.com/marin-community/marin/blob/main/lib/taskcompendium/src/taskcompendium/pipeline/README.md)
describes each stage in detail.
