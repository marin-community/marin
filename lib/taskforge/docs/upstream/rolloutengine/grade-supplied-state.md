# [rolloutengine] Grade a supplied final state without model inference

The rollout engine can grade a task only at the end of a model rollout. A task author cannot check a task's verifier against a known final state before training on it: for example, that a reference solution scores 1, that a plausible wrong solution scores 0, or that a partly correct solution earns the expected partial credit. Shell-graded tasks need these checks most, because their verifier reads the final machine state, and the only way to produce that state today is to run a model.

Today, at the head of #9623 (`lib/rolloutengine`, not on `origin/main`):

- `ShellboxRolloutEngine.run(task)` creates the machine, runs the model loop, and then grades through the private `rolloutengine.grading._grade_rollout(task, convention, messages, machine, factories)`.
- `_grade_rollout` already handles each verifier kind: pure answer verifiers through `taskcompendium.grading.grade_answer`, `ShellVerifierSpec` in the task machine or in a separate grading machine with `collect` and `artifacts`, and `skipped`.
- `_validate_task` rejects any nonempty `TaskSpec.resources` group, including `oracle`. Schema `0.22` therefore has no executable path for oracle material.
- The machine lifecycle (`machines._task_machine`, file installation, setup, healthcheck) is private.

Proposed: a public entry point that builds the task machine, applies a supplied final state, and runs the task's verifier with the same code path as a rollout:

```python
@dataclass(frozen=True)
class SuppliedState:
    messages: tuple[dict[str, Any], ...]          # final conversation; the last item is the assistant submission
    files: tuple[EnvironmentFile, ...] = ()       # installed after environment setup, before grading
    commands: tuple[EnvironmentCommand, ...] = () # run after files, for state that is not a file

async def grade_state(self, task: TaskSpec, state: SuppliedState) -> GradeResult: ...
```

on `ShellboxRolloutEngine`, so it uses the engine's machine factories and convention. For a staged task, it grades only the first stage, or accepts one `SuppliedState` per stage. Machines are closed on every exit path, as in `run`.

Usage: before admitting a task, the author grades a reference solution and labeled wrong solutions, then compares each `GradeResult` with its label:

```python
grade = await engine.grade_state(task, SuppliedState(messages=final_messages, files=reference_files))
assert grade.status == Outcome.GRADED and grade.reward == 1.0
```

Evidence:

- `experiments/post_training/capability_env_gen/capability_pipeline/synthesis.py` (`_validate_controls`, which accepts `workspace` candidates) and `docs/partial_controls.md` on branch `mark/autoenv`. That pipeline graded authored workspaces through the same verifier as trials, using its own sandbox wrapper.
- 15 of the 16 tasks accepted in the construct-003 production run used a verifier that ran in a container. Each needed a reference-solution check before acceptance.

Not in scope for rolloutengine: which reference and wrong solutions a task has, and how their labels are checked (the task author's pipeline). The `GradeResult` fields used for partial-credit labels are covered by the TaskCompendium issue on verifier status and detail.
