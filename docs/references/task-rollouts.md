# Task rollouts

TaskCompendium defines tasks. `ShellboxRolloutEngine` calls the model, executes
task operations through a `TaskSession`, and returns `RolloutData`.
Shellbox supplies the task's execution machine. The caller controls concurrency,
retries, group grading, and training projections.

## Task format

`TaskSpec.model_dump_json()` serializes a task.
`TaskSpec.model_validate_json()` validates a serialized task.
Applications own dataset file formats. SkyRL converts source rows with Hugging
Face `Dataset.map` and retains prepared tasks in memory.
Its task Parquet exports and private Harbor caches store one serialized task per
row in `task_spec`.
The optional `task_execution` column stores separate execution settings.

The serialized task contains private grading inputs. The model request contains
the public conversation, submission instructions, and tool definitions.

| Field | Execution contract |
| --- | --- |
| `context` | The public conversation before the first model request. |
| `environment.kind` | `null`, `shellsim`, or `docker`. |
| `environment.image` | Docker source: `RegistryImage` or `DockerBuild`. |
| `environment.workdir` | Working directory for commands. The default is `/workspace`. An empty value uses the Docker image's working directory. |
| `environment.files` | Files that the engine installs before inference. JSON uses base64 content and retains permission bits. Explicit `mtime_ns` values require Docker. |
| `environment.env` | Environment variables for task commands. Entire values `${VAR}` and `${VAR:-default}` resolve from the rollout process environment. |
| `environment.setup` | Commands that prepare a fresh task machine. |
| `environment.healthcheck` | Readiness command, startup grace period, interval, and retry limit. |
| `environment.network` | Network access. The default is disabled. |
| `environment.startup_timeout` | Optional deadline for machine creation, file upload, setup, and health checks. |
| `environment.memory_mb` | Optional memory limit for the machine factory. |
| `environment.cpus`, `environment.storage_mb`, `environment.gpus` | CPU, storage, and GPU settings. Backends reject settings they cannot apply. |
| `environment.interaction` | Optional application-supplied task session. |
| `verifier` | Private verifier kind, serialized parameters, files, and an optional isolated grading environment. |
| `oracle_files` | Private control files for curation checks. The agent does not receive them. |
| `stages` | Ordered phases with instructions, graders, and minimum reward requirements. All phases use the same machine. |
| `metadata` | Application data that does not change the execution contract. |

`environment` describes the task machine. `environment_requirements` declares
task capabilities.
Iris does not provide per-job network denial. Tasks on Iris must explicitly set `environment.network=True`.
The engine does not change a task's network policy to match its backend.

`null` creates no machine. `shellsim` uses ShellSim's virtual filesystem and
built-in commands. It does not load a Docker image.
`DockerBuild` stores its context files, binary content, and permission bits in the
task row. Its `dockerfile` path starts at the context root. Private verifier files
remain in `verifier` and do not enter the agent's build context.
`RegistryImage` and `DockerBuild` require Skopeo and an image cache in the Docker
factory.
The caller's machine factory selects the Docker backend and its image cache.
An `environment.interaction` value selects a factory from the engine's `sessions`
mapping. The callable receives the task and its Shellbox machine, then returns a fresh session.
Null environments pass no machine. The engine closes the session before it closes the machine.
Without that value, the engine uses its shell-tool session.
Native curation providers and workspace graders require an application-supplied
`TaskSession`. The default session rejects those contracts before inference.
TaskCompendium's curation runtime supplies its native calendar and shell providers.

The caller supplies a `MachineFactory` for each executable environment kind.
Each task gets a fresh machine. The engine attempts machine cleanup after completion,
failure, or cancellation.
The default session exposes `shell(command: string)` for executable environments.
Files persist between commands. Each command starts a new shell process.
`rolloutengine.shell_tool` defines that contract: `SHELL_TOOL_NAME`, the function
definition from `shell_tool_definition()`, and the tool message content from
`shell_observation(result)`. The content is a JSON object with `stdout`, `stderr`,
`exit_code`, `reason` (the Shellbox `ExitReason` value), and `truncated`.

`ShellboxRolloutEngine.run(task, execution=...)` accepts a `TaskSpec` and separate
`TaskExecution` settings from `taskcompendium.execution`.

| Execution field | Contract |
| --- | --- |
| `attempt_timeout` | Optional deadline from machine creation through grading. |
| `agent_timeout` | Optional elapsed-time limit for model requests and task transitions. Grading has its own timeout. |
| `agent_user` | Optional user for agent shell commands. Docker accepts a username or numeric UID as a string. |
| `stages` | A mapping with one `StageExecution` for every task stage. Missing or unknown names cause rejection. |

`StageExecution` supplies files relative to the machine workdir, setup commands,
readiness checks, and optional agent deadline and user overrides.
The Harbor importer exposes `harbor_task` and `harbor_execution` separately.
SkyRL applies launch-time deadline overrides to the execution settings.

## Grading

Text, numeric, multiple-choice, and final-action tasks use the shared verifier registry.
A final-action task submits a function call as its answer. A null environment
records that call without execution.
An incorrect answer has a numeric grade. A verifier failure has no grade.
`GradeResult.score_min` and `score_max` retain the verifier's native score range.
SkyRL uses those bounds for normalized score metrics. Score normalization leaves
optimization rewards and reward shaping unchanged.

Grading starts when the session reports completion, the model reaches its token
limit, or the engine reaches `max_turns`. An agent deadline also ends the turn
loop and starts grading. It applies to model calls and task transitions.
The rollout retains generated tokens and reports `stop_reason="agent_timeout"`.
If no model response exists for the current stage, its grade is unavailable.
An interrupted task transition records `advance_incomplete=1` in its step metrics.
That terminal step has no task observations or intermediate grade.
The verifier has its own deadline. Docker task images must supply `setsid`.
Docker command interruption stops its process group and retains task files and
other services for grading. If Docker cannot identify or stop the command, it
disposes the task container and reports the original interruption.
One rollout step contains one model response and the following task transition,
including tool calls.
Execution failures raise `RolloutInterrupted`, with the failed operation, the completed steps of the current task,
and the original exception as the cause. The engine attempts resource cleanup
before the caller receives that exception. Token-contract violations propagate
as `RolloutContractError`.
Machine startup and setup failures use the `start` operation and retain an empty
rollout record. After cancellation, a background thread can return a machine.
The engine retains the build context until the factory finishes, then closes
that machine within the cleanup deadline.

`ShellVerifierSpec` defines a command, a timeout, environment variables, and a reward source.
`VerifierSpec.files` holds private files. The engine installs them after the last
model response. The command receives the conversation as JSON on standard input.
`ShellVerifierSpec.user` selects the verifier user separately from `TaskExecution.agent_user`.
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
numeric keys remain in `grade.rewards` for stage gates and aggregation.
`GradeResult.failure` identifies a verifier timeout, missing reward, empty reward,
invalid reward, or execution failure. Error policies use this field without parsing
the error message.

Before grading, the engine creates the reward directories and removes existing
reward files. A valid reward file can supply a grade after a nonzero command exit.
A command timeout has no grade. This contract supports Harbor reward files with
`reward.json` before `reward.txt`.

`ExitCodeReward` gives reward `1` for exit code zero and reward `0` for a nonzero
exit code. A command timeout has no grade.

The optional `VerifierSpec.environment` defines a fresh grading machine.
The engine runs `collect` commands in the agent machine, then copies the declared
`artifacts` to that grading machine. Each artifact identifies its source, target,
and kind: file, directory, or automatic detection. Directory exclusions use
`tar --exclude` in the agent machine. A declared missing-file policy selects
an error or a skipped artifact. The engine installs private verifier files only
in the grading machine. It closes the two machines after execution.

The engine runs some grading commands as user `0` in the agent machine. It
inspects an artifact of kind `auto` or with missing-file policy `skip`, archives
a directory artifact with exclusions and removes that archive, and removes a
stage's private verifier and reward files before the next stage.

`ExternalVerifierSpec` stores private parameters for an application-supplied
`TaskSession`. A session prepares the conversation, executes transitions, grades
the result, and releases its resources. It does not call the model.
`TaskSession.prepare` returns a `SessionStart` with the initial messages and
model options. The engine owns conversation and token accumulation after that
point so all session implementations use the same exact-token checks.

Tasks with `stages` use a `staged` verifier. Each `TaskStage` defines its instructions,
verifier, and reward gates. Its `StageExecution` supplies stage preparation.
The first stage uses the task's public conversation. Later stages append their
instructions to the conversation and retain the exact token prefix.
The turn limit and agent deadline apply separately to each stage.
An agent deadline ends the stage chain after grading the current stage.
The engine removes shared private grader files before the next stage starts.
If removal fails before another stage can run, the engine stops the chain with a `cleanup` interruption.
The last stage retains its grade and records removal failures as cleanup errors.
An earlier execution failure retains its original operation and cause.

`StageVerifierSpec.strategy` selects `mean` or `final`. The mean includes only
stages with valid grades. Missing reward keys count as zero in that mean.
JSON reward files can supply multiple numeric keys, such as `reward` and `safety`.
The final strategy uses the last attempted stage.

If a stage cannot produce a grade, the aggregate retains that stage's outcome,
failure details, and diagnostics. A skipped grader is not a failure.
A stage's `minimum_rewards` maps each key to a minimum value. A missing key or
a value below its minimum stops execution before the next stage.
The gate controls stage progression. It does not change the grade of the attempted stage.
When the aggregate is graded, the engine assigns its reward to the last model turn
with a valid grade.

Other turns receive zero optimization reward and retain their stage grades.
The engine masks tokens from a stage without a valid grade, except for explicitly skipped grading.
An explicitly skipped stage retains its token masks and supplies no score.
Earlier valid stages keep their exact tokens and masks. When the aggregate has
no grade, all turns receive zero optimization reward.
The caller decides whether earlier graded stages can enter training.

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
probabilities, and per-step records. Model tokens initially have mask value `1`.
Stage grading can set that value to `0` when the stage has no valid grade.
Observation tokens have mask value `0` and log probability `0`.
A session can request a conversation reset through `Transition.reset_conversation`.
When another turn is available, that reset removes all earlier turns from the
training record. Lean refinement uses this operation after a failed proof attempt.

`ShellboxRolloutEngine.run(task, execution=...)` asynchronously returns one `RolloutData`.
The caller starts one coroutine for each active task and controls concurrency.
Model, machine, and session operations run on the caller's event loop. A caller
cancels the task that awaits `run`. Each cleanup action has the caller-supplied
`cleanup_timeout` deadline. Repeated cancellation cannot extend that deadline.
A cleanup error does not remove a completed grade or replace an execution failure.
`grade.diagnostics.cleanup_errors` records the cleanup operation and exception type.
`metrics.cleanup_error_count` records the number of cleanup errors.
A cleanup deadline cancels the cleanup action. If that action ignores cancellation,
the engine retains it until completion and reports the timeout without an unbounded wait.

The caller controls storage of completed records. This example connects a task
to a model that supplies exact tokens:

```python
from collections.abc import Awaitable, Callable

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.execution import TaskExecution
from taskcompendium.models import TaskSpec
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutData
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.submission import PlainText


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
        cleanup_timeout=30,
        convention=PlainText(id="plain"),
    )
    return await engine.run(task, execution=TaskExecution())
```

Application-supplied sessions require an additional `sessions` mapping.

## Task machines and shell grading

Applications that prepare a task machine or grade a shell verifier outside a
rollout use the engine's own operations:

- `rolloutengine.machines.task_machine(environment, factories, cleanup)` is an async
  context manager. It creates a fresh machine for an `EnvironmentSpec`, installs
  its files, and runs its setup commands and healthcheck within
  `environment.startup_timeout`. It yields `None` for a null environment. On exit
  it closes the machine through `cleanup`.
- `rolloutengine.cleanup.Cleanup(timeout)` runs each cleanup action within
  `timeout` seconds and records failures in `errors` as `CleanupError` values.
- `rolloutengine.grading.shell_grade(verifier, messages, machine, files)` installs
  the private verifier files in `machine`, runs the `ShellVerifierSpec` command
  with `messages` as JSON on standard input, and returns its `GradeResult`. It
  does not run `collect` commands or copy grading artifacts.

```python
from rolloutengine.cleanup import Cleanup
from rolloutengine.grading import shell_grade
from rolloutengine.machines import task_machine
from taskcompendium.environment import ShellVerifierSpec


async def grade_reply(task, factories, reply):
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    messages = ({"role": "assistant", "content": reply},)
    async with task_machine(task.environment, factories, Cleanup(30)) as machine:
        return await shell_grade(verifier, messages, machine, task.verifier.files)
```

## Integrations

TaskCompendium contains importers for Harbor, SWE, and SkyRL source rows.
See the MarinSkyRL [rollout modules](https://github.com/marin-community/MarinSkyRL/tree/rollout-engine/skyrl-train/skyrl_train/rollouts)
for the SkyRL integration. SkyRL sets `trajectory_runner.cleanup_timeout` in
[its base configuration](https://github.com/marin-community/MarinSkyRL/blob/rollout-engine/skyrl-train/skyrl_train/config/ppo_base_config.yaml).

SWE task exports must initialize `refs/taskcompendium/base` before inference.
The SWE importer saves the initial Git revision there during machine setup.
Patch collection compares the final index with that revision, including agent
commits and new files.

## Local checks

From the Marin repository root:

```bash
uv run --project lib/taskcompendium --extra pipeline --group test pytest lib/taskcompendium/tests -q
uv run --project lib/rolloutengine --group test pytest lib/rolloutengine/tests -q
```

From the MarinSkyRL repository root, with rolloutengine, TaskCompendium, and Shellbox installed:

```bash
uv run --no-sync pytest skyrl-train/tests/cpu/rollouts/test_engine.py -q
```

The CPU tests use ShellSim and model or HTTP fixtures. They do not validate a live
vLLM service or a Docker rollout.
