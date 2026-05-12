# Downstream Scaling Evals Guide

This package evaluates a model on a task while leaving completion generation
open-ended. The framework fixes the artifact contract and the top-level graph:

```text
task -> prompts.jsonl.gz
model + prompts.jsonl.gz + completion algorithm -> completions.jsonl.gz
task + prompts.jsonl.gz + completions.jsonl.gz -> grades.jsonl.gz
```

Completion algorithms own their internal execution topology. They may run one
remote step, a Zephyr pipeline inside a step, or a larger executor subgraph. The
framework only requires that the final completion step writes
`completions.jsonl.gz`.

## Layout

- `framework/core.py`: top-level composition API and `Protocol` contracts.
- `framework/schema.py`: artifact filenames, row types, and JSONL readers.
- `tasks/`: task implementations that write prompts and grades.
- `algorithms/`: completion-generation implementations.

## Top-Level API

Use `make_eval_step` to compose one configured task, one model, and one
configured completion algorithm:

```python
from experiments.downstream_scaling.evals.framework.core import make_eval_step

grade_step = make_eval_step(
    name="downstream-scaling/evals/gsm8k/model-123",
    model_path=model_path,
    task=gsm8k_task,
    alg=iid_algorithm,
)
```

`make_eval_step` returns the final grade step. The prompt and completion steps
are dependencies reachable through executor inputs.

## Task Contract

A task is an already-configured object responsible for prompt materialization
and grading:

```python
class EvalTask(Protocol):
    def make_prompts_step(self) -> ExecutorStep:
        """Return the stable prompt step for this configured task."""

    def make_grade_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        completions_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        """Return a step whose output directory contains grades.jsonl.gz."""
```

Rules:

- `make_prompts_step()` takes no eval-specific name. The same task config should
  reuse the same prompt artifact across model evaluations.
- Prompt-defining task config must be included in the prompt step config with
  `versioned(...)`.
- Operational grading settings, such as the number of Zephyr grade workers,
  should not be versioned unless they change the logical grade output.
- Graders receive both `prompts_path` and `completions_path`. Completion
  algorithms are not responsible for copying task fields into completions.

## Completion Algorithm Contract

A completion algorithm is an already-configured object with this shape:

```python
class CompletionAlgorithm(Protocol):
    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        """Return the final step that writes completions.jsonl.gz."""
```

Rules:

- The algorithm owns the full completion-generation topology.
- Do not return `(intermediate_steps, final_step)`. Encode dependencies through
  `ExecutorStep` inputs.
- Algorithm-specific config belongs on the configured algorithm object.
- `model_path` is an input to the builder, not part of the algorithm config.
- The framework does not impose an `n_samples` invariant. Completion count may
  vary by prompt.
- Raw string paths are wrapped with `version_path(...)` directly when building
  `ExecutorStep` configs. `InputName` and `MirroredValue` inputs pass through
  unchanged.

## Artifact Formats

All artifacts are gzip-compressed JSONL files. `id` is the stable join key
across prompts, completions, and grades. Readers in `framework/schema.py`
validate row shape and reject duplicate ids.

### `prompts.jsonl.gz`

Required fields:

```json
{
  "id": "gsm8k/test/42",
  "prompt": "..."
}
```

Optional fields:

```json
{
  "ground_truth": "...",
  "metadata": {
    "problem": "...",
    "source": "gsm8k",
    "split": "test"
  }
}
```

`prompt` is the exact model input. `ground_truth` and `metadata` are task-owned.
There are no configurable field-name knobs.

### `completions.jsonl.gz`

Required shape:

```json
{
  "id": "gsm8k/test/42",
  "completions": [
    {
      "text": "..."
    }
  ]
}
```

Optional metadata:

```json
{
  "id": "gsm8k/test/42",
  "completions": [
    {
      "text": "...",
      "metadata": {
        "finish_reason": "stop"
      }
    }
  ],
  "metadata": {
    "completion_algorithm": "iid"
  }
}
```

Per-completion metadata and top-level completion metadata are algorithm-owned.
Do not copy prompt rows or grader inputs through this artifact.

### `grades.jsonl.gz`

Required shape:

```json
{
  "id": "gsm8k/test/42",
  "grades": [
    {
      "score": 1.0
    }
  ]
}
```

Optional metadata:

```json
{
  "id": "gsm8k/test/42",
  "grades": [
    {
      "score": 1.0,
      "metadata": {
        "correct": true,
        "extraction": "42"
      }
    }
  ]
}
```

`grades[i]` corresponds to `completions[i]`. Task-specific grading details
belong in grade metadata. Tasks may write additional summary artifacts in their
grade output directory, but the framework only requires `grades.jsonl.gz`.

## Versioning

Version semantic inputs. Do not version pure execution topology.

Examples:

- Task prompt config such as split, few-shot count, few-shot seed, and debug
  subset size should be versioned in the prompt step.
- Completion sampling config should be versioned.
- IID `chunk_size` is versioned because it defines the chunk plan and
  internal chunk output namespace.
- Worker counts and worker resources are operational settings unless changing
  them changes the logical artifact.

If an internal output layout depends on a parameter, include that parameter in
the layout or version it into the executor step config. Prefer doing both for
chunk-plan parameters that affect resume behavior.

## Zephyr And Ordering

Zephyr `reshard` does not preserve global input order. Any pipeline that shards
per-completion records must carry enough information to restore per-prompt
completion order before writing `completions.jsonl.gz` or `grades.jsonl.gz`.

The current grading and IID paths use a `completion_index` in intermediate
records, then group by prompt id and sort by `completion_index` before writing
the final row. `completion_index` is an internal pipeline field, not part of the
public completion or grade artifact schema.

## IID Algorithm

`algorithms/iid.py` implements IID sampling with vLLM as one Marin step that
runs Zephyr internally:

1. Read `prompts.jsonl.gz`.
2. Build flat `(prompt, sample)` chunks with durable JSONL chunk outputs and
   `.SUCCESS` markers.
3. Run chunk processing across Zephyr workers with vLLM on TPU resources.
4. Aggregate chunk records into `completions.jsonl.gz`.

Flat request chunking is intentional in this implementation. It gives finer
resume granularity than making each prompt the unit of work. The public
framework does not require this chunk shape; other completion algorithms may
choose a different topology.

The IID config separates semantic sampling config from operational execution
config:

```python
@dataclass(frozen=True)
class IIDConfig:
    sampling: IIDSamplingConfig
    execution: IIDExecutionConfig
```

`model_path` is passed to `make_completions_step`; it is not stored in
`IIDConfig`.

## Adding A Task

1. Add a task config dataclass with prompt-defining fields and grading execution
   fields.
2. Implement `make_prompts_step()` with a stable task-family step name and
   versioned prompt config.
3. Write `prompts.jsonl.gz` with `id` and `prompt`.
4. Implement `make_grade_step(...)`.
5. In the remote grade function, read prompts and completions with the schema
   readers, grade each completion, and write `grades.jsonl.gz`.

Task-specific extraction and scoring logic stays in the task module.

## Adding A Completion Algorithm

1. Add semantic config dataclasses for algorithm behavior.
2. Add execution config fields only for runtime topology and resources.
3. Implement a configured object satisfying `CompletionAlgorithm`.
4. Return the final `ExecutorStep` that writes `completions.jsonl.gz`.
5. If the algorithm shards intermediate per-completion records, carry an
   internal ordering key and restore per-prompt completion order before writing
   the public artifact.

No framework-level Zephyr helper is required. Reuse existing algorithm code when
the topology matches; otherwise the algorithm owns its distributed execution.
