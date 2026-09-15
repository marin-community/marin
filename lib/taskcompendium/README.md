# TaskCompendium

Read [the TaskCompendium specification](SPECIFICATION.md) for the semantic schema, renderings, Harbor lowering, SkyRL integration path, and planned environment extensions.

TaskCompendium stores task requirements independently of execution choices. It
imports bounded samples from TaskTrove, R2E-Gym, GSM8K, and NeMo Gym into pinned
`TaskSpec` records and exports Harbor lowerings with
separate execution configuration and private verification information.

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
  lib/taskcompendium/tests -m 'docker and not slow and not integration and not data_integration and not cluster and not requires_cluster and not manual'
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

Build R2E verifier images for the 20 Orange3 sample indices and SymPy row 500,
then pass their runtime mapping to the dataset builder. Use `--rows 0 1 500`
for the three source tasks exercised live:

```bash
uv run --project lib/taskcompendium --extra harbor \
  lib/taskcompendium/examples/build_r2e_runtimes.py \
  --rows {0..19} 500 --output /tmp/r2e-runtimes.json
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
Its manifest records 44 specifications, 88 generated exports, and two rejections,
including 21 R2E tasks with built verifier runtimes and two two-step tasks: repository modification and conversational revision. The Parquet dataset and manifest are checked in; the
builder recreates the per-task JSON and Harbor directories. Local image IDs are usable on the machine that built them. Portable exports require
a registry image pinned with `@sha256:`; rebuild and export with the resolved image
when using another machine. Build recipes alone do not preserve every transitive
OS package version.

Each export contains `task.toml`, `binding.json`, a manifest, private
`specification.json`, and `renderings.json`. Single-step instructions live in
`instruction.md`; multi-step instructions live under `steps/step-N/instruction.md`.
`binding.json` records the task-owned environment shape, declared tool interface, and
conversation requirement. It never selects a model or a harness.

Harbor resolves a compatible agent, model endpoint, timeout, retry policy, and launch
configuration when it starts a rollout. A checked-in `reference-execution.json` may
be supplied for a reproducible example, but it is not a canonical task artifact. Pass
an explicit resolved configuration to the runner:

```bash
uv run --project lib/taskcompendium --extra harbor \
  python -m taskcompendium.harbor.runner /tmp/taskcompendium-poc/harbor/gsm8k-train-0-plain \
  --execution /tmp/harbor-launch.json --trials-dir /tmp/taskcompendium-trials --trial-name example
```

Full task packages are trusted because they contain hidden grading information. The
lowering exposes only rendered instructions and resources marked `agent` to the
model.

For individual specifications, `taskcompendium export` takes a specification JSON,
a `--renderings` JSON array with one rendering per step, and a `--binding` JSON file.
For an answer-only chat task that file is
`{"environment":{"kind":"none"},"interaction":{"kind":"chat"}}`. It rejects
incompatible task and binding combinations before writing a task package.

## NeMo first-wave examples

The first wave adds four pinned source cases: instruction following with binary
and fractional constraint scores, competitive code submitted as an answer,
prediction of a native tool call, and Workplace domain actions. It tests distinct
execution requirements with one source seed per case; it is not a representative
dataset sample or a difficulty measurement for Snowball.

Rebuild the local artifact after building the private runtime image above:

```bash
uv run --project lib/taskcompendium --extra harbor \
  lib/taskcompendium/examples/build_first_wave.py \
  --runtime-image "$runtime_image" --output /tmp/taskcompendium-first-wave
```

The builder emits six semantic instances and twelve Harbor exports, including
plain/JSON/XML variants of instruction and code answers, plus a derived
three-request Workplace workflow. The requests reply to Carlos and create a
follow-up task, move it to In Progress, then move it to In Review and send a
specified notification. Harbor retains provider state and conversation between
requests; each private step check includes earlier mutations. See the
[full requests](examples/first_wave/workplace-multistep.md). It writes semantic
JSON/Parquet and lowerings alongside those exports. Model
endpoints remain explicit execution inputs. The tests use private scripted
attempts and local HTTP fixtures; agent-visible lowering artifacts contain neither those
attempts nor the source verification payloads. Source provenance distinguishes
verified Gym fixtures from their associated Hugging Face dataset revisions.

See the [coverage manifest](../../.agents/projects/taskcompendium-coverage/suite.json)
for the TaskTrove intersection and deferred cases. Existing MCQA, code, and R2E
fixtures remain the controls; no duplicate TaskTrove import is needed merely to
exercise another output convention.

## Harbor conformance suite

The Harbor conformance selector is the integration acceptance gate for this package. It lowers representative canonical specifications, runs their exported packages through Harbor’s actual `Trial` lifecycle, and checks the observable verdict for known-good, known-wrong, malformed, and missing submissions. It uses replay agents or local scripted OpenAI-compatible endpoints, so it validates the integration rather than model capability.

```bash
uv run --project lib/taskcompendium --extra harbor --group test \
  pytest -m harbor_conformance \
  lib/taskcompendium/tests
```

The selector includes answer-only and rendered-answer tasks; ShellSim file tasks; source-backed Docker code and R2E-Gym final-state repairs; native action prediction; and stateful, multi-step Workplace workflows. The thin native-action and stateful-workflow cells include known-good, known-wrong, malformed-submission, and verifier-outage trials. A verifier outage is retained as an `infra_error` with no reward; it is never normalized to score zero. The selector also covers private-resource isolation, extraction failures, and retained conversation. Docker is intentionally included in this explicit command. The ShellSim fixture builds its bridge with Cargo. A Qwen or other live-model smoke run may complement it, but a model’s reward is not the conformance oracle.

## Contracts

Schema 0.8 stores ordered `StepSpecification` entries in a pinned semantic
`TaskSpec`. `taskcompendium.rendering.TaskFamily` defines the source/family
interface `instantiate(key) -> TaskSpec | Rejected`. Instantiating one source row
fixes its identity before rendering. Each step records
instructions, intrinsic answer requirements, verifier, resources, and context
requirements. Shared requirements, resources, source provenance, and the success
policy belong to the concrete specification.

`TaskSpec.coverage_tags` records reviewed semantic coverage labels:
competency, shape, domain, artifact, interaction, state, context, and one
Snowball-calibrated `difficulty:easy`, `difficulty:medium`, or `difficulty:hard`.
Each lowering carries those labels forward and adds result-encoding labels from its
rendering: `result:json`, `result:xml`, or `result:file`. Result tags
describe only how the answer is submitted. A task that substantively produces or
consumes structured data uses semantic competency or artifact tags as well.

`TaskRequirements` declares `Capability.FILESYSTEM`, `SHELL`, and/or `PROCESS`, plus
`WorkspaceState`: workdir, setup commands, additional directories, and an optional
immutable image identity. Empty requirements describe answer-only tasks. The image
identifies required starting state; it does not select an environment provider.
Repository repairs preserve their exact image and dependency state.

`Rendering` selects submission location and extraction. A lowering renders each
instruction, materializes only agent resources, exposes the selected tools, and
records the submission contracts. Its `specification_sha256` links back to the
pinned instance. It excludes private verifiers and oracle material. File submission
may add capability requirements without changing the pinned semantic instance.

Environment implementations and task bindings live in `taskcompendium.execution`.
`HarborTaskBinding` declares the required environment shape and public tools. An
empty tool list is chat; nonempty bindings expose the declared tools. `HarborLaunchConfig` selects a compatible agent only
when Harbor starts a rollout. `HarborExecutionConfig` is their resolved pair and is
kept solely for explicit reference executions. The semantic record names none of them.

`lower_to_harbor(spec, renderings, binding, destination)` writes a task package from
one compatible submission convention per step. It writes `binding.json`, never an
agent-specific `execution.json`. A package may be launched by any Harbor configuration
that satisfies the binding.

Agent-facing instructions state the task and output requirements directly. They never
describe judges, verifiers, rewards, hidden tests, or grading procedures. Importers
rewrite known source evaluation boilerplate while keeping actual behavior and format
requirements. Verifier and oracle resources remain private.

An empty `tools` tuple exposes no tools even when the binding uses Docker. A nonempty
tuple is an explicit tool binding, and the supported adapters currently accept exactly one:

- `ShellToolBinding(name="shell", backend="shellsim")` exposes a model function named
  `shell` with one required string argument, `command`. Use backend `docker` for a
  real shell.
- `HarnessToolBinding(interface="terminal", backend="docker")` requires a terminal
  capability. Harbor may satisfy it with Terminus, mini-SWE-agent, or replay; replay
  can also use `backend="shellsim"`. None of those harness names appear in the task
  package.
- `ProviderToolBinding(interface="workplace_assistant")` binds a declared domain action
  interface to its provider adapter.

Tool availability is independent of submission: `AssistantFinal` checks a response,
while `FinalState` checks the declared workspace paths and does not require a
substantive final message. `FinalActionSubmission` carries source-native function
schemas for a prediction; the action-output agent records calls without executing
them. Its expected action and comparison rules stay in `PredictedActionVerifier`.

`ProviderToolBinding(interface="workplace_assistant")` instead exposes actual domain actions
through the selected provider. `TaskRequirements.action_interfaces` pins the
required interface version and seed identity; `ProviderEnvironment` holds the
adapter and private seed configuration. The generic `provider_chat` loop correlates
calls and results, while the adapter owns tool behavior and state. Its private
`ProviderStateVerifier` reads authoritative state after execution. Arbitrary MCP
bindings and reactive user conversations remain deferred.

The supported providers are no environment, ShellSim, Docker, and Workplace. ShellSim supplies
filesystem and supported shell operations; Docker also supplies native processes.
Capability compatibility includes the actions actually exposed to the model, so an
empty tool list in Docker does not grant filesystem access. A provider must also preserve
the required workspace and immutable image state.

Executable verifier modes (`stdio`, `pytest`, `script`, `junit`, `gotest`) require
`TaskTroveVerifier.runtime: ContainerRuntime`. Trusted answer checks, including exact
match, MCQA, and math, run in the evaluator process and reject runtime configuration.
Judges use `TaskTroveVerifier.judge` for model and service configuration. Extraction belongs
to the rendering. Verification executes privately, independently of the model's
action interface; switching environment implementation cannot redefine correctness.
The shell example uses ShellSim actions and a separate container for its checker.

Other source semantics use tagged verifier variants. `ConstraintVerifier` records
binary or fractional constraint aggregation. `CodeAnswerVerifier` materializes an
extracted code answer inside an isolated checker container, without adding tools
to the agent-visible lowering. `PredictedActionVerifier` compares submitted native calls;
`ProviderStateVerifier` checks mutations in a domain provider. These private
contracts are independent of the selected answer rendering.

Harbor records trajectories and execution results. Model wire formats do not define
the task schema. Fixed ordered requests are supported; reactive user simulators and
an additional conversational environment interface remain deferred.

Harbor owns the multi-step lifecycle and persistent environment. Shared inputs
are available initially; step inputs are uploaded at their step. Agent step
resources named `setup.sh` are rejected because Harbor reserves that file for
executable setup. Resource placement cannot override preserved image dependencies.

The pinned Harbor version supports `mean` and `final` aggregation. Multi-step
`all_required_steps` is rejected because the backend cannot express it exactly.
A missing step or an extraction/infrastructure error suppresses the task aggregate
while retaining per-step results. A graded incorrect answer remains a valid zero
under the selected aggregation policy. The sequential greeting example uses `final`; its last private test suite
checks both the original API and the requested extension.

Every TaskCompendium Harbor agent retains the full agent-visible exchange across
ordered steps. `prior_conversation` marks a semantic dependency on that history;
`instruction_and_workspace` means the task does not rely on it. Native Harbor agents
without resumable conversation remain unsupported for ordered tasks.

The examples contrast these context requirements:

| Task | Step sequence | Required context | Exported execution |
| --- | --- | --- | --- |
| `synthetic/conversational-revision` | Produce a specified sentence, then change its meeting day without repeating the sentence in the follow-up | Prior conversation for step two | Chat with retained history, plain-text or JSON answers |
| `synthetic/sequential-greeting` | Implement `greet(name)`, then add an uppercase option preserving existing behavior | Current instruction and workspace | Retained conversation, persistent Docker workspace |

Each step has one instruction string. Later instructions and step resources are
released only when that step starts. Both examples use final-step scoring. The
revision's fixed initial sentence makes its revised answer deterministic; the
repository's final tests cover both the original behavior and the extension.
Local HTTP-fixture tests inspect the actual model requests to check history
retention and absence of future instructions. A Docker tool-chat trial checks that
the retained second step can read and extend the first step's file. These checks use
scripted responses, not live model rollouts.


Resources have `agent`, `verifier`, and/or `oracle` roles. Small resources are embedded;
external resources have a URI and required SHA256. Oracle material cannot be marked
agent-visible. Container images require a registry digest or local content ID.

Canonical hashing uses sorted JSON with an explicit schema version, independently
of Parquet encoding. Arrow uses nested structures for tasks, capability/state requirements, resources,
rendering-independent requirements, judge configuration, and provenance. Arbitrary
upstream verifier parameter values are individually encoded as JSON in an Arrow
map; the whole task is not an opaque JSON column.

Extraction errors, invalid tasks, and infrastructure errors carry no reward. A
successfully graded incorrect answer carries reward zero. Extraction failure is an
unsuccessful agent attempt when an explicit output convention was violated. A future
training consumer must apply an explicit penalty policy rather than silently omit
those attempts; infrastructure failures remain a separate category. Judge policy records the
provider, endpoint, model, declared size, sampling count, and temperature; results
record the returned model identity and fingerprint when supplied. Judge credentials
are injected separately, never serialized into a specification.

## Validation record

[`examples/poc/validation.json`](examples/poc/validation.json) records local commands,
results, and export/hash checks, with the schema and scope recorded per validation entry. This includes native Harbor
multi-step chat and Docker trials, extraction-error handling, conversation
continuity, and staged ShellSim inputs. Source evaluation boilerplate has been removed from the affected TaskTrove prompts;
boxed-answer instructions are added only by the matching rendering. The new source conversions
received an independent fidelity review. CI has not run on GitHub for this branch.

## Coverage and limits

| Source family | Task requirements | Verification |
| --- | --- | --- |
| TaskTrove MCQ and exact answers | None | Pinned source comparisons after explicit extraction |
| TaskTrove numeric/coordinate puzzles | None; optional ShellSim file protocol | Pinned math verifier |
| GSM8K | None; optional file protocol | Delimited gold and pinned math verifier; rationale stays private |
| TaskTrove JSON/XML | None | Original schema or element checker; format constraints retained |
| TaskTrove reference-answer judge | None | Original reference contract, explicit judge policy |
| TaskTrove nl2bash | Filesystem and supported shell | Original private script in Docker |
| TaskTrove Python/C++ | Filesystem and native processes | Original pytest tests or stdio cases |
| R2E-Gym | Filesystem, native processes, pinned repository/dependencies | Original private tests; Orange3 good/bad/empty and SymPy good/bad Harbor trials |
| Synthetic conversational revision | None | Exact sentence checks; second step requires prior conversation |
| Synthetic sequential greeting | Filesystem and native processes | Two ordered feature requests with cumulative final pytest requirements |

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
Python 3.12 supervisor. The checked-in recipe relocates the source image
interpreter so the unprivileged candidate can execute it; it does not reinstall the
repository dependencies. R2E imports require
`ContainerRuntime(workspace=ImageOverlay((".venv",)), ...)`; use a protocol with
`FinalState((".",), excluded_paths=(".venv", "datasets"))`. The grading container retains
its image's dependency directory and replaces the remaining workspace with the
submitted snapshot, including deletions. `supervisor_python` must be an absolute
path to the separately installed Python 3.12 interpreter; the source image PATH
selects the source interpreter. The selected repairs do not change the source image's dangling
`datasets` alias, which is excluded from snapshots without replacing it with guessed
data. The 20 Orange3 source rows have observed-log parity checks. Rows 0 and 1 also
pass actual Harbor trials: the original repair earns 1, and the unrepaired and empty
submissions earn 0. The other 18 Orange3 rows have not been graded live. Complete
SymPy rows 500 and 550 have source checksum and status-map parity evidence; row 500
is included with a built runtime and good/bad Harbor execution. Row 550 is retained
as an import fixture. Orange3 explicitly requires Xvfb; SymPy runs without it.

Native Docker workspace snapshots reject links and are limited to 256 MiB, 100,000
archive entries, and 120 seconds. Excluded dependency directories are never downloaded
from the candidate container. Overlay grading seeds a fresh local Docker volume from
the image, so large dependency directories still require local copying and disk space.
The volume is removed with its grading container. This path has been exercised through
Harbor with both an internal-symlink fixture and the original R2E environments.

The rejection ledger covers validity defects and unsupported contracts. Difficulty
for a roughly 2B-active policy remains a separate, deferred rollout measurement.

## Published examples

The [published schema 0.1 checkpoint](https://huggingface.co/datasets/open-athena/taskcompendium-spike/tree/e59893fb44145196bf620fbecbd235d3d4ee8e04)
contains separate task and lowering tables. It corresponds to commit `051edff52c`
and has 41 specifications and 81 exports. The [published schema 0.2 checkpoint](https://huggingface.co/datasets/open-athena/taskcompendium-spike/tree/8300827d24b4d1963f4ff642884d20a2a2cf6501)
has 43 specifications and 84 exports. The [schema 0.4 capabilities checkpoint](https://huggingface.co/datasets/open-athena/taskcompendium-spike/tree/666fd6c4ec77d3d342f6f438f6d292253889dfe3)
has 44 specifications and 88 exports. The [schema 0.5 first-wave publication](https://huggingface.co/datasets/open-athena/taskcompendium-spike/tree/18ad898b8fe5b50043af9d843209d755a4fbc00c)
contains 49 specifications and 99 exports. Its 14 payload checksums and both
dataset configurations were verified after upload.
The [multi-step Workplace update](https://huggingface.co/datasets/open-athena/taskcompendium-spike/tree/b79b2c023a4a890046cd2e1c1badb0de37b68a7d)
adds the derived three-request workflow and its scripted rollout: 50 tasks and
100 lowerings, with all 16 payload checksums and both table loads verified.
Lowering rows distinguish task capability/state requirements from the
selected agent and environment, and preserve submission and private verifier
information separately.
MCQA includes plain/JSON file submissions through ShellSim and Docker/Terminus-2.
The native-process Python/C++ tasks include replay and Terminus-2 workspace exports;
R2E workspace exports use replay. Live-model configuration is separate from the
scripted model responses used by local tests.

The [Clean `.9` review and pruning update](https://huggingface.co/datasets/open-athena/taskcompendium-spike/tree/31e3c89fdd6dd0ade69145d7f21d8ed02a787021)
attaches the release-balanced 86-archive review and removes four confirmed
bad or unsupported examples. It contains 46 task rows and 95 lowerings; the
remaining examples retain their original fixture provenance.
