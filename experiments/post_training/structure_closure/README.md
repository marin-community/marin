# Structural completion probes

These probes test whether a model completes an assistant prefix while preserving
required content. They cover JSON objects, arrays, nesting, escapes, tool and
reasoning delimiters, code fences and repeated tool openers. Continuation controls
distinguish closing a structure from dropping required content. Some instructions
include the reference answer; these measure conditional completion. They do not
establish independent task-solving or downstream tool-use improvement.

## Build cases and tasks

The benchmark reads BFCL reference answers and an explicit split manifest with
`train_task_ids`, `diagnostic_reserved_task_ids` (validation) and `parity_task_ids`. It keeps
training and validation parent tasks disjoint and excludes parity tasks from both.

```bash
uv run python -m experiments.post_training.structure_closure.benchmark \
  --source /path/to/bfcl --split /path/to/split.json --output /path/to/cases

uv run python -m experiments.post_training.structure_closure.task_data \
  --cases /path/to/cases --tool-prompt compact --output /path/to/tasks
```

The task exporter writes ordinary TaskSpecs with a user instruction, assistant
text prefix and `StructureClosureSpec` grader. Run them through RolloutEngine's
`PrefillHarness`: register it as `"prefill"` in the engine's harness mapping and
set `TaskRolloutSpec.harness="prefill"`. Ordinary chat rendering treats assistant
messages as completed history. The model must return the raw continuation as assistant content. Parsed
reasoning or tool actions are unscored by the harness.

Use `--tool-prompt native-declared --initializer /path/to/local/tokenizer` to
serialize tool calls with the tokenizer's chat template and advertise their tool
schema. These probes do not execute the synthetic `write_result` function.
`task_data.paired_task_probes` produces four diagnostic TaskSpecs, crossing
declared/undeclared tools with compact/native serialization while keeping the
call content fixed.

`training_prefixes` advances training continuation prefixes to a point where
closing immediately would lose required content. For example, the reference
`["first", "second"]` may have prefix `["first", "secon`: appending `"]` gives valid
JSON but loses the final `d`. Validation bytes remain unchanged.

```bash
uv run python -m experiments.post_training.structure_closure.training_prefixes \
  --source /path/to/cases --output /path/to/near-boundary-cases
```

Pass the resulting cases to the task or token exporter.

## Exact-token reproduction

The token exporter preserves assistant prefixes and records tokenizer/template
fingerprints, context budgets and conditional BPE round trips:

```bash
uv run python -m experiments.post_training.structure_closure.token_data \
  --cases /path/to/cases --initializer /path/to/local/tokenizer \
  --context-tokens 16384 --tool-prompt compact --output /path/to/tokens
```

`tool_prompt_probe` exports the four paired tool-probe arms as token records.
`eval_audit` scores historical session-format evaluation dumps against their
validation token cohort. It checks recorded prompt tokens, reports syntax and
content-preserving success by family/control, and reports missing, duplicate or
unverified responses as incomplete coverage:

```bash
uv run python -m experiments.post_training.structure_closure.eval_audit \
  --tokens /path/to/tokens --eval-dump /path/to/eval.jsonl \
  --uid-source row_index --output /path/to/report.json
```

Use `--uid-source case_id` for dumps keyed by stable task IDs. Row-index dumps
must use the original unfiltered cohort order with closure rows first. Generated
TaskSpecs use the generic grading and rollout APIs described in
[task rollouts](../../../docs/references/task-rollouts.md).
