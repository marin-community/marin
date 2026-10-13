# Taskforge

Taskforge turns task ideas into TaskCompendium `TaskSpec`s with a model's help. It is the
`marin-taskforge` member of the repository's uv workspace.

## Package layout

```
src/taskforge/
  content_hash.py  canonical JSON and its sha256 digests
  llm/             the ModelClient protocol, the GLM client, sampling policy, structured output
  stage30_author/  program.py: build steps and the DAG they form;
                   author.py: Revision, the Author protocol, `static_author`, the program manifest
  stage40_build/   context.py: RLEnvBuildContext, BuildFailure;
                   step_store.py: step keys, memos, the memo store and `memoized`;
                   run.py: `run_program`, `run_build`, the draft and `ItemBuildProvenance`
```

Each pipeline stage is a package named `stageNN_<name>`, numbered by tens in pipeline order:
proposal 10, triage 20, author 30, build 40. The gaps leave room to insert a stage later without
renumbering. A stage imports from lower-numbered stages and from the shared modules
(`content_hash`, `llm`), and the shared modules import no stage.
`tests/test_imports.py` checks both directions.

## The build DAG

An author writes a build program as `async def` steps decorated with `@rl_env_build_step`, each
returning a value of its annotated type. The final step returns a `TaskSpec` and carries
`@rl_env_build_sink` in place of `@rl_env_build_step`. The author passes the steps, and the sink as
the keyword `sink`, to `rl_env_build_program`. That call orders the steps so each runs after its
inputs, puts the sink last, and refuses a step whose output reaches neither another step nor the
sink. A step receives an `RLEnvBuildContext` (item id, model, sampling policy) as its first
parameter. The decorator reads the step's inputs from the parameters after it, and the runner
passes each one by name: another step's output, or the caller's proposal for a parameter called
`proposal`. `tests/stage40_build/test_run.py` builds this toy program, abridged here:

```python
@rl_env_build_step
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return inline_file("workspace/numbers.txt", b"6 7\n")


@rl_env_build_step
async def grader(context: RLEnvBuildContext) -> GraderPackage:
    return verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


@rl_env_build_sink
async def task(context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, proposal: dict) -> TaskSpec:
    return TaskSpec(...)  # the prompt from proposal["task"], numbers as a worker resource, grader.grader


PROGRAM = rl_env_build_program(numbers, grader, sink=task)
```

`run_build` runs the program and writes `draft/task.json` and `draft/provenance.json`. The second
file lists every step, the sink last, with the stored memo of its output and whether this build hit
or missed the cache for it. The memo files in the store carry no status, because the store writes
a memo only on a miss.

Each step's output is memoized under a key built from `SDK_VERSION`, a source digest, the content
digest of each input, and a digest of the model and its sampling policy. The decorator computes the
source digest over the step's own source text, followed by the source of its output type when the
program defines that class beside the step. An output type imported from a library, such as
`TaskSpec`, falls under `SDK_VERSION`. A step should therefore read everything it depends on from
its inputs, its own body or an output type defined beside it. Editing a helper defined outside the
body leaves the key unchanged, so a library change of that kind comes with an `SDK_VERSION` bump.
Because inputs enter the key by content, a step that reruns and returns the same output leaves its
consumers cached. The runner memoizes each step through `step_store.memoized`, and a driver
memoizing a helper calls it the same way with a key of its own.

## Testing

```bash
uv run --package marin-taskforge --frozen --group test pytest -c lib/taskforge/pyproject.toml \
    lib/taskforge/tests experiments/post_training/capability_driven_envs/tests
cd lib/taskforge && uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```
