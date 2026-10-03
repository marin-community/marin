# Task rollouts

TaskCompendium defines executable tasks. The `marin-rolloutengine` package owns the rollout loop in
`lib/rolloutengine/src/rolloutengine/engine.py`.
`ShellboxRolloutEngine` calls a model, executes task operations, and grades each task.

## Task format

`taskcompendium.parquet.write_tasks(path, tasks)` writes a Parquet file with one
`task_spec` string column. Each value is a serialized `TaskSpec`.
`read_tasks(path)` reads bounded batches and validates the task schema.
Paths use Rigging's guarded filesystem access, including transfer budgets and
backend timeouts.

The Parquet file contains private grading inputs. The model request contains the
public conversation, submission instructions, and tool definitions.

| Field | Execution contract |
| --- | --- |
| `context` | The public conversation before the first model request. |
| `environment.kind` | `null`, `shellsim`, or `docker`. |
| `environment.image` | Docker source: `RegistryImage` or `DockerBuild`. |
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
| `metadata` | Application data that does not change the execution contract. |

`null` creates no machine. `shellsim` uses ShellSim's virtual filesystem and
built-in commands. It does not load a Docker image.
`DockerBuild` stores its context files, binary content, and permission bits in the
task row. Its `dockerfile` path starts at the context root. Private verifier files
remain in `verifier` and do not enter the agent's build context.
`RegistryImage` and `DockerBuild` require Skopeo and an image cache in the Docker
factory.
The caller's machine factory selects the Docker backend and its image cache.
An `environment.interaction` value selects a factory from the engine's `sessions`
mapping. The callable receives the task and returns a fresh session.
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
`GradeResult.score_min` and `score_max` retain the verifier's native score range.
SkyRL uses those bounds for normalized score metrics. Score normalization leaves
optimization rewards and reward shaping unchanged.

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
`TaskSession.prepare` returns a `SessionStart` with the initial messages and
model options. The engine owns conversation and token accumulation after that
point so all session implementations use the same exact-token checks.

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
When the aggregate has no grade, the caller decides if earlier valid stages can enter training.

A model failure after a completed turn triggers grading of the completed state.
`RolloutInterrupted` retains that grade and the exact token evidence.
Stage setup failures also retain earlier completed stages. The caller's error
policy determines whether the interrupted record enters training.

## Model and token contract

The model callable accepts `ModelRequest` and asynchronously returns `ModelTurn` with the parsed
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

`ShellboxRolloutEngine.run(TaskSpec)` asynchronously returns one `RolloutData`.
The caller starts one coroutine for each active task and controls concurrency.
Model, machine, and session operations run on the caller's event loop. A caller
cancels the task that awaits `run`; the coroutine does not finish until session
and machine cleanup finishes. Cleanup errors propagate.

The caller controls storage of completed records.

This example connects one task and a caller-supplied model.
The model must implement the token contract above.

```python
from collections.abc import Awaitable, Callable

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import TaskSpec
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutData
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.submission import AnswerFormat, SubmissionConvention


async def run_task(
    task: TaskSpec, model: Callable[[ModelRequest], Awaitable[ModelTurn]]
) -> RolloutData:
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
    return await engine.run(task)
```

Application-supplied sessions require an additional `sessions` mapping.

## Integrations

TaskCompendium contains importers for Harbor, SWE, and SkyRL Gym tasks. The
rollout engine does not own batching, group grading, retry policy, or training
projection. Applications implement those policies around `ShellboxRolloutEngine`.
See the MarinSkyRL rollout modules for the SkyRL integration.

## Local checks

From the Marin repository root:

```bash
uv run --project lib/taskcompendium --extra harbor --group test pytest lib/taskcompendium/tests -q
uv run --project lib/rolloutengine --group test pytest lib/rolloutengine/tests -q
```

From the MarinSkyRL repository root, with rolloutengine, TaskCompendium, and Shellbox installed:

```bash
uv run --no-sync pytest skyrl-train/tests/cpu/rollouts/test_engine.py -q
```

The CPU tests use ShellSim and model or HTTP fixtures. They do not validate a live
vLLM service or a Docker rollout.
