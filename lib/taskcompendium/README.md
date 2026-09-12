# TaskCompendium

TaskCompendium separates a task's meaning from its agent interface and submission
format. It imports bounded samples from cleaned TaskTrove, R2E-Gym, and GSM8K into a
versioned `TaskSpecification`, stores one specification per Parquet row, and exports
Harbor packages with an execution configuration.

This is a spike. Training, marinskyrl integration, Snowball serving, and difficulty
measurement are deferred. Source acceptance and execution validation are separate:
an importer returning a specification does not establish that its runtime works.

## Install and run

Run these commands from the repository root. The package has an independent uv
workspace so Harbor and verifier dependencies do not enter Marin's training stack.
Python 3.12 and Cargo are required; Docker is needed for executable graders.

```bash
uv sync --project lib/taskcompendium --frozen --extra harbor --group test
cargo build --locked --manifest-path lib/taskcompendium/shellsim-bridge/Cargo.toml
export PATH="$PWD/lib/taskcompendium/shellsim-bridge/target/debug:$PATH"
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests
```

The default test selection excludes Docker, external integrations, and manual tests.
The dedicated CI workflow runs that safe selection with the actual Rust ShellSim
bridge. The root affected-test selector does not include this independent package.

To exercise local Docker validation, including the native terminal agents:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest \
  lib/taskcompendium/tests -m 'docker and not integration and not manual'
```

These trials use scripted responses or known validation programs. They do not call
Snowball or any external model. Terminus-2 runs unchanged. The mini-SWE-agent adapter
checks that version 2.4.6 is preinstalled and inherits Harbor's native setup, run,
and trajectory handling. Its model client runs inside the container; disconnected
exports need an endpoint reachable there. The acceptance fixture uses container
loopback. A live model connection needs an explicitly configured deployment.

## Build the sample dataset

The builder reads checked-in source archives and records their checksums. The `.8`
release manifest pins the verifier; its Iris producer job
`/power/iris-run-job-20260911-210153` records a clean checkout of
[`ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2`](https://github.com/marin-community/marin/commit/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2).
The TaskTrove converter and verifier trees match the separately recorded inspected
revision. A later [release note for `.9`](https://github.com/marin-community/marin/pull/9061#issuecomment-5641882077)
supersedes `.8`; this spike retains the requested `.8` data. All 20 sampled archives
declare its verifier pin. Imports reject a different or missing verifier pin when an
archive contains a Dockerfile. The builder writes canonical JSON, nested Parquet, a
rejection ledger, and Harbor exports into a new directory, then verifies that Parquet
round-trips to the same specification hashes.

```bash
docker build -t taskcompendium-runtime:validation \
  -f lib/taskcompendium/src/taskcompendium/harbor/runtime.Dockerfile \
  lib/taskcompendium/src/taskcompendium/harbor
runtime_image=$(docker image inspect taskcompendium-runtime:validation --format '{{.Id}}')
uv run --project lib/taskcompendium --extra harbor \
  lib/taskcompendium/examples/build_poc.py \
  --runtime-image "$runtime_image" --output /tmp/taskcompendium-poc
```

Build R2E verifier images for all 20 sample indices, then pass their runtime mapping
to the dataset builder. Use `--rows 0 1` for only the two tasks exercised live:

```bash
uv run --project lib/taskcompendium --extra harbor \
  lib/taskcompendium/examples/build_r2e_runtimes.py \
  --rows {0..19} --output /tmp/r2e-runtimes.json
uv run --project lib/taskcompendium --extra harbor \
  lib/taskcompendium/examples/build_poc.py \
  --runtime-image "$runtime_image" --r2e-runtimes /tmp/r2e-runtimes.json \
  --output /tmp/taskcompendium-poc-with-r2e
```

The mapping keys select R2E source commits; each value is a `ContainerRuntime` for
that commit's source image. Building all 20 rows downloads about 18.8 GB of unique
compressed source layers; extracted images and build cache need substantially more
space. The full local build used a 128 GiB Docker disk allocation. No training or
model endpoint is needed.

Omit `--runtime-image` to build only answer tasks. The checked-in sample under
[`examples/poc`](examples/poc/manifest.json) records the local validation image ID.
Its manifest records 41 specifications, 81 generated exports, and two rejections,
including 20 R2E tasks with built verifier runtimes. The Parquet dataset and manifest are checked in; the
builder recreates the per-task JSON and Harbor directories. Local image IDs are usable on the machine that built them. Portable exports require
a registry image pinned with `@sha256:`; rebuild and export with the resolved image
when using another machine. Build recipes alone do not preserve every transitive
OS package version.

Each export contains `instruction.md`, `task.toml`, `execution.json`, a manifest,
and trusted specification/protocol files. The sample includes replay templates with no submission configured and Terminus-2
templates requiring a model and provider configuration. Set `agent.kwargs.response` for chat, or
`agent.kwargs.commands` for terminal replay. Terminus-2 templates need
`agent.model_name` and provider configuration. Then run:

```bash
uv run --project lib/taskcompendium --extra harbor \
  python -m taskcompendium.harbor.runner /tmp/taskcompendium-poc/harbor/gsm8k-train-0-plain \
  --trials-dir /tmp/taskcompendium-trials --trial-name example
```

The judge sample policy points to an unreachable validation-only endpoint and has
no configured client. For live judging, choose a real `JudgeConfig`, re-export the
specification, and set `verifier.kwargs.judge_api_key_env` to the name of an environment
variable holding the credential. An exact reference match may pass the source exact
gate without a judge call; that does not demonstrate a working judge endpoint.

Use `--help` on the runner for its explicit execution-file option. Keep full task
packages trusted: they contain hidden grading information. The lowering exposes
only the rendered instruction and resources marked `agent` to the agent.

For individual specifications, the `taskcompendium` command supports `schema`,
`validate`, `pack`, `list`, and `export`. `export` takes a specification JSON,
protocol JSON, and `ExecutionConfig` JSON, for example
`{"agent":"replay","environment":{"kind":"none"}}`. This input is a different
schema from the generated Harbor `execution.json`; do not pass a generated execution
file back as `export --execution`. It checks compatibility before writing:
chat with a file submission, intrinsic JSON with an XML extractor, and a repository
task with no filesystem are rejected.

## Contracts

`TaskSpecification` records instructions, intrinsic answer requirements, the required
agent environment, role-scoped resources, verifier semantics, verifier runtime, and
source provenance. `Protocol` selects chat or chat with terminal tools and an
explicit submission extractor. Agent-facing instructions state the task and output
requirements directly; they never describe judges, verifiers, rewards, hidden tests,
or grading procedures. Importers rewrite source evaluation boilerplate while keeping
the actual behavior and format requirements. Verifier and oracle data remain separate. `ExecutionConfig` selects an agent and environment.

The agent environment can be absent, ShellSim, or Docker. A protocol can add ShellSim
for file submission to a task that has no intrinsic environment requirement.
Verifier execution is independent: the shell example uses ShellSim for agent actions
and a separate Docker container for its original script checker.

Resources have `agent`, `verifier`, and/or `oracle` roles. Small resources are embedded;
external resources have a URI and required SHA256. Oracle material cannot be marked
agent-visible. Container images require a registry digest or local content ID.

Canonical hashing uses sorted JSON with an explicit schema version, independently
of Parquet encoding. Arrow uses nested structures for tasks, environments, resources,
protocol-independent requirements, judge configuration, and provenance. Arbitrary
upstream verifier parameter values are individually encoded as JSON in an Arrow
map; the whole task is not an opaque JSON column.

Extraction errors, invalid tasks, and infrastructure errors carry no reward. A
successfully graded incorrect answer carries reward zero. Judge policy records the
provider, endpoint, model, declared size, sampling count, and temperature; results
record the returned model identity and fingerprint when supplied. Judge credentials
are injected separately, never serialized into a specification.

## Validation record

[`examples/poc/validation.json`](examples/poc/validation.json) records the local
commands and outcomes: 184 safe Python tests, 36 Docker tests, and one Rust test
passed, alongside type checking and required formatting checks. All 41 Parquet rows
match their canonical JSON hashes, and all 81 export manifests were verified. Review findings for answer, math, and
structured importers were addressed before these checks. This is local evidence;
the new CI workflow has not yet run on GitHub.

## Coverage and limits

| Source family | Agent world | Verification |
| --- | --- | --- |
| TaskTrove MCQ and exact answers | None | Pinned source comparisons after explicit extraction |
| TaskTrove numeric/coordinate puzzles | None; optional ShellSim file protocol | Pinned math verifier |
| GSM8K | None; optional file protocol | Delimited gold and pinned math verifier; rationale stays private |
| TaskTrove JSON/XML | None | Original schema or element checker; format constraints retained |
| TaskTrove reference-answer judge | None | Original reference contract, explicit judge policy |
| TaskTrove nl2bash | ShellSim | Original private script in Docker |
| TaskTrove Python/C++ | Docker | Original pytest tests or stdio cases |
| R2E-Gym | Docker required | Original private tests; good/bad/empty Harbor trials on two pinned rows |

The judge examples establish plumbing only; live grading quality is unvalidated.
Structured-output graders check shape, not whether every value faithfully summarizes
the source text. Validation-only structured answers demonstrate that distinction.

The pinned ShellSim bridge is a real persistent VFS/interpreter with resource limits,
not a host shell. It does not implement all GNU behavior: `cp source/. destination/`
copies differently from GNU `cp`. The retained shell validation uses an explicit
seed-file copy permitted by the task, then runs the requested pipeline. Do not infer
support for arbitrary source shell scripts from this example.

Executable source graders run in fresh containers with disconnected networking and
separate candidate, test, and supervisor state. Candidate processes cannot access
the supervisor's specification/result files. The inherited pytest and arbitrary
script reporting mechanisms are not a complete defense against adversarial code
that manipulates its own test process. This spike does not claim general hostile-code
benchmark integrity.

R2E source images retain their original repository dependencies alongside a separate
Python 3.12 supervisor. The checked-in recipe relocates the original Python 3.7
interpreter so the unprivileged candidate can execute it; it does not reinstall the
repository dependencies. R2E imports require
`ContainerRuntime(workspace=ImageOverlay((".venv",)), ...)`; use a protocol with
`FinalState((".",), excluded_paths=(".venv", "datasets"))`. The grading container retains
its image's dependency directory and replaces the remaining workspace with the
submitted snapshot, including deletions. `supervisor_python` must be an absolute
path to the separately installed Python 3.12 interpreter; the source image PATH
selects Python 3.7. The selected repairs do not change the source image's dangling
`datasets` alias, which is excluded from snapshots without replacing it with guessed
data. All 20 pinned source rows have observed-log parity checks. Rows 0 and 1 also
pass actual Harbor trials: the original repair earns 1, and the unrepaired and empty
submissions earn 0. The other 18 rows have not been graded live.

Native Docker workspace snapshots reject links and are limited to 256 MiB, 100,000
archive entries, and 120 seconds. Excluded dependency directories are never downloaded
from the candidate container. Overlay grading seeds a fresh local Docker volume from
the image, so large dependency directories still require local copying and disk space.
The volume is removed with its grading container. This path has been exercised through
Harbor with both an internal-symlink fixture and the original R2E environments.

The rejection ledger covers validity defects and unsupported contracts. Difficulty
for a roughly 2B-active policy remains a separate, deferred rollout measurement.

## Published examples

The [Hugging Face dataset](https://huggingface.co/datasets/open-athena/taskcompendium-spike)
contains separate task and lowering tables. Lowering rows expose the selected agent,
required and selected environment, submission kind, and verifier information.
MCQA includes plain/JSON file submissions through ShellSim and Docker/Terminus-2.
The Docker-required Python/C++ tasks include replay and Terminus-2 workspace exports;
R2E workspace exports use replay. Live-model configuration is separate from the
scripted model responses used by local tests.
