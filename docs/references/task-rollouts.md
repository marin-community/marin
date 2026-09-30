# Task rollouts

TaskCompendium defines executable tasks and the common rollout loop in
`lib/taskcompendium/src/taskcompendium/rollout.py`.
`ShellboxRolloutEngine` calls a model, executes task operations, and grades the result.
`RolloutSink` accepts the resulting records.

The SkyRL worker is in `MarinSkyRL/skyrl-train/skyrl_train/rollouts/task_worker.py`.
`TaskRolloutWorker` supplies model inference, completes group grading, and projects
rollouts into training data. It submits completed prompt groups through
`BufferRolloutSink`. The engine has no buffer dependency.
The explicit SkyRL entrypoint is `skyrl_train.entrypoints.taskcompendium`.
The SWE examples use this entrypoint. The default training entrypoint prepares
Gym source rows as task Parquet and uses the same worker. The Harbor entrypoint
also uses this worker. Nemotron prepares its Gym and terminal rows in one task
file and uses the same worker pool.
`skyrl_train.entrypoints.terminal_bench_generate` also uses that worker and the
same task dataset. It collates dataset rows before it creates evaluation requests.
All entrypoints require the shared vLLM structured-chat transport before Ray
initialization. The deleted Harbor agents no longer select a separate capability
profile, OpenCode continuation adapter, or literal-log reader.
The Iris launch document has no agent ingress section. The driver does not start
a recording proxy or forward controller credentials to rollout workers. Model
requests use the injected SkyRL inference client. Shellbox handles sandbox
operations independently of model inference.

## Task format

`taskcompendium.parquet.write_tasks(path, tasks)` writes a Parquet file with one
`task_spec` string column. Each value is a serialized `TaskSpec`.
`read_tasks(path)` reads bounded batches and validates the schema and verifier.
The path can use an installed fsspec filesystem.

The Parquet file contains private grading inputs. The model request contains the
public conversation, submission instructions, and tool definitions.

| Field | Execution contract |
| --- | --- |
| `context` | The public conversation before the first model request. |
| `environment.kind` | `null`, `shellsim`, or `docker`. |
| `environment.image` | Docker source: `LocalImage`, `RegistryImage`, or `DockerBuild`. |
| `environment.workdir` | Working directory for commands. The default is `/workspace`. An empty value uses the Docker image's working directory. |
| `environment.files` | Files that the engine installs before inference. JSON uses base64 content and retains permission bits. |
| `environment.env` | Environment variables for task commands. `${VAR}` and `${VAR:-default}` resolve at execution. |
| `environment.setup` | Commands that prepare a fresh task machine. |
| `environment.healthcheck` | Readiness command, startup grace period, interval, and retry limit. |
| `environment.network` | Network access. The default is disabled. |
| `environment.startup_timeout` | Optional deadline for machine creation, file upload, setup, and health checks. |
| `attempt_timeout` | Optional deadline for one complete attempt, from machine creation through grading. |
| `environment.memory_mb` | Optional memory limit for the machine factory. |
| `environment.cpus`, `environment.storage_mb`, `environment.gpus` | CPU, storage, and GPU settings. Backends reject settings they cannot apply. |
| `environment.interaction` | Optional application-supplied task session. |
| `verifier` | Private verifier kind and serialized parameters. |
| `agent_timeout` | Optional elapsed-time limit for model requests and task transitions. Grading has its own timeout. |
| `agent_user` | Optional execution user for agent shell commands. Docker accepts a username or numeric UID as a string. |
| `stages` | Ordered phases with separate instructions, setup, graders, and minimum reward requirements. All phases use the same machine. |
| `metadata.teacher_route` | Optional SkyRL teacher route for this task. |

`null` creates no machine. `shellsim` uses ShellSim's virtual filesystem and
built-in commands. It does not load a Docker image.
`DockerBuild` stores its context files, binary content, and permission bits in the
task row. Its `dockerfile` path starts at the context root. Private verifier files
remain in `verifier` and do not enter the agent's build context.
`LocalImage` requires an installed image. `RegistryImage` and `DockerBuild`
require Skopeo and an image cache in the Docker factory.
SkyRL configures these through `trajectory_runner.skopeo` and
`trajectory_runner.image_cache`.
Daytona also executes tasks with `environment.kind: docker`. The caller's machine
factory selects the backend. Daytona accepts registry images or a Dockerfile at
the build context root. It does not accept nested Dockerfile paths or local images.
An `environment.interaction` value selects a factory from the engine's `sessions`
mapping. The factory receives the task, machine, and submission convention.
Without that value, the engine uses its shell-tool session.

The caller supplies a `MachineFactory` for each executable environment kind.
Each task gets a fresh machine. The engine closes the machine after completion,
failure, or cancellation.
The default session exposes `shell(command: string)` for executable environments.
Files persist between commands. Each command starts a new shell process.

## Grading

Text, numeric, multiple-choice, and final-action tasks use the shared verifier registry.
A final-action task submits a function call as its answer. A null environment
records that call without execution.
An incorrect answer has a numeric grade. A verifier failure has no grade.

Grading starts when the session reports completion, the model reaches its token
limit, or the engine reaches `max_turns`. Execution failures raise
`RolloutInterrupted`, with the failed operation, the last completed rollout,
and the original exception as the cause. The engine releases task resources
before the caller receives that exception. Token-contract violations propagate
as `RolloutContractError`.
Machine startup and setup failures use the `start` operation and retain an empty
rollout record. The engine releases resources before it returns that failure.

`ShellVerifierSpec` defines a command, private files, a timeout, environment
variables, and a reward source. The engine installs private files after the last
model response. The command receives the conversation as JSON on standard input.
`ShellVerifierSpec.user` selects the verifier user separately from `agent_user`.
Setup and collect commands can also select an execution user.

The default `StdoutReward` requires a zero exit code and one finite numeric value
on standard output. Truncated output is a verifier failure.

`FileReward` defines reward files in priority order. Each file contains a number
or a JSON object with a configured numeric key. The first existing file controls
the result. A malformed file is a verifier failure. The engine does not use a
lower-priority file after a parse failure.
An optional `pass_above` threshold supplies the separate pass/fail result.
Harbor conversion sets this threshold to zero, so a positive grade is a pass.
For a JSON object, the configured key supplies the scalar grade. Other finite
numeric keys remain in `grade.diagnostics["rewards"]` for stage gates and aggregation.
`GradeResult.failure` identifies a verifier timeout, missing reward, empty reward,
invalid reward, or execution failure. Error policies use this field without parsing
the error message.

Before grading, the engine creates the reward directories and removes existing
reward files. A valid reward file can supply a grade after a nonzero command exit.
A command timeout has no grade. This contract supports Harbor reward files with
`reward.json` before `reward.txt`.

`ExitCodeReward` gives reward `1` for exit code zero and reward `0` for a nonzero
exit code. A command timeout has no grade.

The optional `ShellVerifierSpec.environment` defines a fresh grading machine.
The engine runs `collect` commands in the agent machine, then copies the declared
`artifacts` to that grading machine. Each artifact identifies its source, target,
and kind: file, directory, or automatic detection. Directory exclusions use
`tar --exclude` in the agent machine. A declared missing-file policy selects
an error or a skipped artifact. The engine installs private verifier files only
in the grading machine. It closes the two machines after execution.

`ExternalVerifierSpec` stores private parameters for an application-supplied
`TaskSession`. A session prepares the conversation, executes transitions, grades
the result, and releases its resources. It does not call the model.

Tasks with `stages` use a `staged` verifier. Each `TaskStage` defines its own
verifier and can install files relative to the machine's working directory.
The first stage uses the task's public conversation. Later stages append their
instructions to the conversation and retain the exact token prefix.
The turn limit and agent deadline apply separately to each stage.
The engine removes shared private grader files before the next stage starts.

`StageVerifierSpec.strategy` selects `mean` or `final`. The mean includes only
stages with valid grades. Missing reward keys count as zero in that mean.
JSON reward files can supply multiple numeric keys, such as `reward` and `safety`.
The final strategy uses the last attempted stage, including a failed grader.
A stage's `minimum_rewards` maps each key to a minimum value. A missing key or
a value below its minimum stops execution before the next stage.
The engine assigns the aggregate reward to the last action with a valid grade.
All other actions receive zero optimization reward and retain their stage grades.
The engine masks tokens from a stage without a valid grade.
If no stage has a valid grade, the result retains the last stage's failure.
When the aggregate has no grade, SkyRL excludes the entire rollout from loss
and baseline calculations, including earlier valid stages under the final strategy.

A model failure after a completed turn triggers grading of the completed state.
`RolloutInterrupted` retains that grade and the exact token evidence.
Stage setup failures also retain earlier completed stages. The caller's error
policy determines whether the interrupted record enters training.

## Model and token contract

`RolloutModel.complete(ModelRequest)` returns a `ModelTurn` with the parsed
assistant message and exact prompt and response token IDs.
Optional log probabilities must align with the response tokens.
For continuation, the next prompt must preserve the complete served token prefix.
The engine rejects changed prefixes and empty token evidence.
For example, prompt tokens `[1, 2]` and response tokens `[3, 4]` require the next
prompt to start with `[1, 2, 3, 4]`. Observation tokens follow that prefix.

A model adapter raises `GenerationLimitReached` when a rendered prompt exceeds
its configured limit. The exception contains the rendered prompt tokens.
The engine retains completed turns and grades their result with stop reason `length`.
If no turn completed, it returns an empty response with no grade.
It does not include the observation that exceeded the limit in a retained response.

`RolloutData` contains the conversation, grade, token IDs, loss mask, optional log
probabilities, and per-step records. Model tokens have mask value `1`.
Observation tokens have mask value `0` and log probability `0`.
A session can request a conversation reset through `Transition.reset_conversation`.
When another turn is available, that reset discards earlier attempts from the
training record. Lean refinement uses this operation after a failed proof attempt.

`RolloutEngine.generate(Iterator[TaskSpec])` returns an asynchronous iterator of
`RolloutData`. `RolloutSink.consume` consumes that iterator.

This example connects a task, caller-supplied model, and caller-supplied sink.
The model must implement the token contract above.

```python
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import TaskSpec
from taskcompendium.rollout import RolloutModel, RolloutSink, ShellboxRolloutEngine
from taskcompendium.submission import AnswerFormat, SubmissionConvention


async def run_task(task: TaskSpec, model: RolloutModel, sink: RolloutSink) -> None:
    engine = ShellboxRolloutEngine(
        model,
        {
            EnvironmentKind.DOCKER: DockerMachineFactory(),
            EnvironmentKind.SHELLSIM: ShellSimMachineFactory(),
        },
        max_turns=20,
        command_timeout=120,
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    )
    await sink.consume(engine.generate(iter([task])))
```

Application-supplied sessions require an additional `sessions` mapping.

## SkyRL integration

`taskcompendium.importers.skyrl.read_gym_tasks` converts source Parquet rows with
`prompt` and `env_class` fields. Its caller supplies the dataset revision and
environment configurations. The importer stores grading inputs in the private
verifier and selects the `skyrl_gym` task session.

The default SkyRL entrypoint uses `GymTaskDataset` to prepare Gym sources before
rollout execution. It writes task Parquet into `data.task_cache_dir`, with a
filename derived from the file content. Each task records a hash of its source
row. Training can then use the materialized file through `TaskDataset` without
the original source dataset. The loader retains trainer metadata separately
from the public model prompt. Worker specifications carry runtime environment
registrations from the trainer process.
The worker limits Gym environment threads through
`environment.skyrl_gym.max_env_workers`. Cancellation waits for an active
environment operation before the session closes its resources.

`taskcompendium.importers.swe.swe_task` converts a SWE source instance with an
explicit environment and verifier timeout. It collects a binary Git patch,
applies that patch in a fresh environment, and runs the private evaluation script.
The SWE examples materialize these tasks before training. They use the common
shell tool and do not contain a separate MiniSWE inference loop.

`taskcompendium.importers.harbor.harbor_task` packages a Harbor task
directory, including its build context, setup files, and private test files.
It preserves reward JSON priority, collect commands, separate verifier images,
execution users, resource settings, health checks, and artifact exclusions.
The converter rejects task features that it cannot yet represent, including
MCP services, skills directories, GPU type selection, and TPUs.
Tasks with multiple steps use `TaskStage` and preserve shared machine state,
stage setup, minimum reward requirements, and the configured reward strategy.
SkyRL supplies `HarborTaskDataset` in `skyrl_train.dataset.harbor` for directory
and packed TaskTrove conversion. Packed selection compares the task count,
selection digest, and distinct environment count with the launch snapshot.
A mismatch stops conversion. The resulting task file includes executable and verifier files,
so rollout workers do not require the source directories or packed archive.
The Harbor entrypoint selects this dataset and the common worker. Its launch
settings select Docker or Daytona, resource overrides, agent and verifier
deadlines, stage limits, error treatment, and reward shaping.
Use `skyrl_train.entrypoints.terminal_bench` with
`terminal_bench_config.harbor.environment_type: docker` or `daytona`.
The same `harbor` mapping contains `enable_reward_shaping`, `reward_shaper`,
`reward_parser`, and `override_timeout_sec`.
Unknown task settings stop the launch before Ray initialization. The Harbor
mapping no longer accepts installed-agent names, versions, logging options,
snapshot options, or agent-specific request settings.
Set chat-template options through `generator.chat_template_kwargs`. Tokenization
and generation receive the same options. Task-specific options override these
generator defaults.
Verifier grades remain separate from optimization rewards. Group-based test
shaping and truncation penalties affect optimization rewards. Span tags and
token credit align with the exact response tokens and exclude observation tokens.
Custom backend classes still require migration.

`harbor.verifier_disable: true` selects an explicit skipped verifier during
task import and execution. Task packages can omit grader files in this mode.
The engine executes all stages and bypasses their minimum-reward gates. It
retains valid tokens and reports `SKIPPED` with no verifier score. Training
uses zero optimization reward for these rollouts. A verifier failure still
produces an error outcome and follows the configured error policy.

Packed task archives require the instruction files selected by `task.toml`.
A staged task supplies `steps/<name>/instruction.md` for each stage. A task
with `environment.docker_image` can omit `environment/Dockerfile`.

Harbor tasks use the configured `max_retries`, `include_exceptions`, and
`exclude_exceptions` for training and evaluation. Exclusions take precedence.
Passthrough failures remain terminal so a retry cannot discard their retained grade.
Each retry starts a fresh machine and token history. Backoff uses `min_wait_sec`
times `wait_multiplier` to the retry index, capped at `max_wait_sec`.
The worker releases concurrency slots before the wait. Cancellation stops retries
and prevents a partial buffer write. `rollout_retries` counts completed retries.
Harbor's `environment.build_timeout_sec` sets the environment startup deadline.
`harbor.timeout_multiplier` scales that deadline, including separate verifier
environments. The engine releases the machine after startup failure.
`harbor.trial_attempt_timeout_sec` sets a separate deadline for each attempt.
It excludes queue and retry-backoff time. Expiry produces `TrialTimeoutError`
without tokens or a grade from the unfinished attempt. The configured retry
and error policies apply to that result. Cleanup completes before a retry starts.
Cleanup is outside the execution deadline so expiry cannot interrupt resource release.
With `preserve_logprobs_on_timeout`, a verifier timeout retains completed tokens
and log probabilities while the verifier score remains unavailable.
A strict reward-parser failure retains the raw verifier score and applies the
configured `VerifierOutputParseError` treatment to that response.

With `data.terminal_bench_data`, the default entrypoint uses `NemotronTaskDataset`.
It resolves terminal instance IDs against Harbor directory names and the IDs in
`tests/config.json` or `tests/test_info.json`. Matching ignores letter case.
Missing or ambiguous IDs stop dataset preparation. The task file retains source
labels, teacher routes, and Nemotron metadata. Harbor resource settings, turn
limits, error policies, and reward shaping apply only to Harbor tasks in the batch.
The worker retains request order and publishes coverage counts for each blend and agent.
Harbor training concurrency divides `harbor.n_concurrent_trials` across workers.
The reserved evaluation worker uses the full value. Harbor's limit does not queue
Gym tasks. `trajectory_runner.max_concurrent_tasks` optionally limits all tasks
within each worker. Its default is `null`, with no common task limit.

The SkyRL adapter uses exact structured-chat inference through vLLM.
It completes GenRM comparison cohorts before it emits rollout records.
Each cohort contains repeated responses to one prompt. The GenRM judge compares
those responses to assign grades. The private Gym configuration sets
`genrm.num_rollouts_per_prompt` and `genrm.judge`.
GenRM evaluation without a comparison cohort has no grade.
Explicitly skipped grading retains trainable tokens with zero reward.
The adapter distinguishes `skipped` from `unavailable` and verifier errors.
It supplies the served generation budget to Gym graders, including AIME length
penalties. Native Gym rewards retain their token rewards, token credit, and
reward components separately from the verifier grade.

`generator.engine_init_kwargs.max_model_len` sets the context window when configured.
The adapter limits each response to the space after the exact rendered prompt.
Without that setting, `generator.max_input_length` limits each prompt.
The default context window is the sum of `max_input_length` and `max_generate_length`.
Environment action rewrites are rejected because training must retain sampled tokens.

For Gym tasks, the worker can retain verified turns after a timeout or server
context overflow. It discards an incomplete turn and its token evidence.
`generator.error_handling` controls the resulting training disposition:
`mask` excludes loss and baseline calculations, `zero` uses zero optimization
reward, and `passthrough` retains the available reward.
Recovery requires behavior log probabilities when the request requires them.
`preserve_logprobs_on_timeout=false` disables retention after a timeout.
Other model-server failures discard completed evidence and have no grade.
Unclassified model-transport exceptions and token-contract violations abort the
group. They cannot produce a partial buffer write.

`WholeTaskProjection` emits one training row per rollout.
`StepTaskProjection` emits one row per retained model turn, with the exact served
prompt for that turn. The projections preserve token rewards, log probabilities,
top-K candidates, expert routes, and teacher routes.
`unshaped_rewards` records verifier scores, with zero as the placeholder for a missing
score. `verification_results` retains score availability. `rewards` supplies scalar
optimization rewards or token rewards at action positions.
`token_level_shaping` supplies separate token credit.
`response_span_tags` identifies generated token spans: `0` for none, `1` for thought,
`2` for action, and `3` for edit. Observation tokens receive zero tags and credit.
Unavailable terminal grades and verifier errors exclude the rollout from loss and
baseline calculations. Explicitly skipped grading remains eligible.
An intermediate tool operation can remain trainable when the terminal grade is valid.

`BufferRolloutSink` projects and finalizes a completed prompt group before one
lease-aware buffer write. A failed group produces no partial buffer commit.
A prompt group contains the rollouts in one leased `RolloutTask` request.
The sink passes the lease to the buffer writer so the buffer can identify the
worker assignment and policy step for the result.

## Local checks

From the Marin repository root:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests -q
```

From the MarinSkyRL repository root, with TaskCompendium and Shellbox installed:

```bash
uv run --no-sync pytest skyrl-train/tests/cpu/rollouts/test_engine.py -q
```

The CPU tests use ShellSim and model or HTTP fixtures. They do not validate a live
vLLM service or a Docker rollout.
