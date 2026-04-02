---
name: marin-executor
description: Build or debug Marin executor DAGs. Use when asked how to define ExecutorSteps, compose multi-step experiments, run subsets with --run_only, or launch executor-backed code manually outside executor_main.
---

# Skill: Marin Executor

Use this skill when the task is about authoring, extending, or debugging Marin executor experiments.

## Core Model

- A Marin experiment is a DAG of `ExecutorStep`s.
- Each step function should accept a plain dataclass config and operate on concrete paths and values.
- The executor is responsible for turning config markers into concrete values before it calls the step function.

The config markers are:
- `output_path_of(step)` / `InputName`: depend on another step's output.
- `this_output_path()` / `OutputName`: refer to this step's output path.
- `versioned(value)` / `VersionedValue`: include a value in the step hash.

Treat those markers as executor-only. They are valid inside `ExecutorStep.config`, but they are not valid inputs to manual `fn(config)` calls.

## Recommended Pattern

Keep two layers separate:

1. Plain config builders for manual or non-executor launches.
2. `ExecutorStep(...)` construction that injects `output_path_of(...)`, `this_output_path()`, and `versioned(...)`.

```python
from dataclasses import dataclass

from marin.execution import ExecutorStep, executor_main, output_path_of, this_output_path, versioned


@dataclass(frozen=True)
class TrialConfig:
    input_path: str
    tokenizer: str
    output_path: str


def build_trial_config(*, input_path: str, tokenizer: str, output_path: str) -> TrialConfig:
    return TrialConfig(
        input_path=input_path,
        tokenizer=tokenizer,
        output_path=output_path,
    )


def run_trial(config: TrialConfig) -> None:
    ...


def tokenizer_trial_step(raw_step: ExecutorStep) -> ExecutorStep[TrialConfig]:
    return ExecutorStep(
        name="tokenizer/trial",
        fn=run_trial,
        config=TrialConfig(
            input_path=output_path_of(raw_step),
            tokenizer=versioned("meta-llama/Llama-3.1-8B"),
            output_path=this_output_path(),
        ),
    )


if __name__ == "__main__":
    raw = ...
    trial = tokenizer_trial_step(raw)
    report = ...
    executor_main(steps=[trial, report], description="Tokenizer trial")
```

Manual launch uses the plain builder:

```python
config = build_trial_config(
    input_path="/tmp/raw",
    tokenizer="meta-llama/Llama-3.1-8B",
    output_path="/tmp/trial",
)
run_trial(config)
```

Never do this outside the executor:

```python
run_trial(trial.config)
```

`trial.config` may still contain `InputName`, `OutputName`, or `VersionedValue`.

## Multi-Step Experiments

- Put all top-level outputs from one experiment file into the same `executor_main(...)` call.
- The executor resolves the full DAG once, computes shared dependencies once, and reuses identical ancestors.
- Do not launch each leaf separately if they share expensive upstream work.

Typical shape:

```python
executor_main(
    steps=[train_step, *eval_steps, report_step],
    description="Train once, then fan out to evals and reporting.",
)
```

## `--run_only`

- `--run_only` matches step names by `regex.search`.
- The executor runs the matched step or steps and all of their transitive dependencies.
- Quote the regex in the shell.

Example:

```bash
uv run python experiments/my_experiment.py --run_only 'train|evals/core'
```

This is the default way to rerun one leaf of a larger DAG without hand-editing the experiment file.

## Launch Boundary: Iris `job run`

Iris stores and executes argv. It does not receive your local shell's heredoc body.

Good:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run -- \
  python -c 'from my_pkg.entrypoint import main; main()'
```

Bad:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run -- python - <<'PY'
from my_pkg.entrypoint import main
main()
PY
```

In the bad form, the local shell consumes the heredoc before Iris sees the command, so the remote job only receives `python -`.

If you truly need shell syntax, invoke a remote shell explicitly. Prefer `python -c ...` for small inline programs.

## Common Pitfalls

- Do not pass `ExecutorStep.config` directly into the underlying function outside executor resolution.
- Keep `versioned(...)` at the `ExecutorStep(..., config=...)` site, not in generic manual-launch helpers.
- Use `output_path_of(dep, "subdir/file")` when you need a concrete child path from another step.
- Use `override_output_path` only when you intentionally want to pin or reuse a heavy artifact path.

## See Also

- `docs/explanations/executor.md`
- `docs/tutorials/executor-101.md`
- `lib/marin/src/marin/execution/executor.py`
