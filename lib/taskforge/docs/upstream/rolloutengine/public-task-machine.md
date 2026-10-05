# [rolloutengine] Export the task-machine lifecycle and in-machine shell grading

A pipeline that builds tasks needs a machine prepared exactly as a rollout prepares one, and needs to run a task's shell verifier in it, before any model rollout exists. A task builder uses both to prototype fixtures and to check that a draft grader gives a reference answer full credit. Today both entry points are private, so a caller imports underscore names that can change without notice.

Today, at the head of #9623 (`lib/rolloutengine`, `7eb1dff26f`, not on `origin/main`):

- `rolloutengine.machines._task_machine(environment, factories)` is the async context manager that creates the machine from the `EnvironmentKind` factory, installs `environment.files`, runs setup and the healthcheck, and closes the machine on every exit path. `engine.py` and `grading.py` both import it.
- `rolloutengine.grading._shell_grade(verifier, messages, machine)` installs a `ShellVerifierSpec`'s private files, runs its command with the conversation on stdin, and reads the reward. `_grade_rollout` calls it for both the in-machine and the separate-environment case.

Proposed: export both under public names with the current signatures and behavior:

```python
# rolloutengine/machines.py
@asynccontextmanager
async def task_machine(
    environment: EnvironmentSpec, factories: Mapping[EnvironmentKind, MachineFactory]
) -> AsyncIterator[Machine | None]: ...

# rolloutengine/grading.py
async def shell_grade(
    verifier: ShellVerifierSpec, messages: Sequence[Mapping[str, Any]], machine: Machine
) -> GradeResult: ...
```

`engine.py` and `grading.py` switch to the public names. No behavior change. If #9763 (`grade-supplied-state.md`) lands as `ShellboxRolloutEngine.grade_state`, it covers the grading half, and only `task_machine` needs exporting.

Usage, in Taskforge's builder SDK (`lib/taskforge/src/taskforge/build/sdk.py`, `Build.machine` and `Build.try_grader`):

```python
async with task_machine(environment, factories) as machine:
    grade = await shell_grade(shell_verifier, messages, machine)
```

Not in scope: machine reuse across several gradings in one build session, which is a separate request.
