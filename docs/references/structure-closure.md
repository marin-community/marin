# Structural completion probes

These probes measure whether a model finishes a supplied assistant prefix while
preserving required content. They cover JSON objects, arrays, nesting, escaped
strings, tool delimiters, reasoning delimiters, code fences and repeated tool
openers. Continuing required content and merely appending closing delimiters are
separate controls. Some controls deliberately include the reference in the user
instruction. They measure conditional completion. Independent task solving and
downstream tool-use improvement require separate evaluations.

The implementation lives under
[`experiments/post_training/structure_closure/`](https://github.com/marin-community/marin/tree/main/experiments/post_training/structure_closure).
It uses the existing [task grading and rollout APIs](task-rollouts.md).

## Build tasks

`benchmark` reads the BFCL complement manifest, its oracle answer scripts and an
explicit split manifest. The split manifest supplies `train_task_ids`,
`diagnostic_reserved_task_ids` and `parity_task_ids`. Train and validation parent
IDs must be disjoint and together cover the approved complement; parity IDs must
be outside both sets.

```bash
uv run python -m experiments.post_training.structure_closure.benchmark \
  --source /path/to/bfcl --split /path/to/split.json --output /path/to/cases

uv run python -m experiments.post_training.structure_closure.task_data \
  --cases /path/to/cases --tool-prompt compact --output /path/to/tasks
```

The second command writes `train.jsonl` and `validation.jsonl` as ordinary
TaskSpecs, with a manifest of source hashes, counts and output hashes. Every row
must convert before either split is published. Each task has a user instruction,
an assistant text prefill and a private `StructureClosureSpec`. No historical
token IDs are needed. Select `PrefillHarness` explicitly when creating the
`TaskRolloutSpec`; ordinary chat rendering treats assistant events as completed
history. The evaluator selects this harness and retains actual model token traces
as rollout evidence.

For the native declared-tool variant, use `--tool-prompt native-declared` and
`--initializer /path/to/local/tokenizer`. This records the tokenizer and template
fingerprints used to serialize the tool body, while keeping the task's input as
text. Tool declarations are advertised schemas; these probes do not execute the
synthetic `write_result` function. The serving backend must return the missing
continuation as text. Parsed reasoning and tool actions are unscored transport
failures. Wrong raw-text answers remain scored failures.

`training_prefixes` creates a separate near-boundary training variant. It advances
ordinary continuation prefixes to a verified premature-closing boundary and
copies validation bytes unchanged:

```bash
uv run python -m experiments.post_training.structure_closure.training_prefixes \
  --source /path/to/cases --output /path/to/near-boundary-cases
```

Pass that output to `task_data` to build the corresponding tasks.
`task_data.paired_task_probes(case, source, tokenizer)` also returns four
ordinary TaskSpecs with fixed content: declared/undeclared tools crossed with
compact/native serialization. These diagnostic arms have distinct task IDs and
`probe:` tags and can use the same evaluation and audit functions as other tasks.
Quoted tool delimiter strings remain JSON content during serialization.

## Evaluate and compare

`evaluate` runs the entire manifest-verified validation cohort exactly once,
with bounded concurrency. It does not launch training or provision a model.
Provide an already running OpenAI-compatible `/v1/completions` endpoint, its
local tokenizer and chat template, an API key in `OPENAI_API_KEY`, a sampling
JSON file and explicit rollout limits. The evaluator renders messages locally
and sends prompt text. It retains the endpoint's raw completion, so reasoning
and tool parsers cannot remove structural delimiters before grading. The server
tokenizes that text using the serving model's own tokenizer.

For example, `sampling.json` may contain:

```json
{"temperature": 0, "max_tokens": 256, "seed": 42}
```

`limits.json` must provide the fields of `RolloutLimits`; evaluation requires
`max_turns` to equal one. One bounded configuration is:

```json
{
  "max_turns": 1,
  "max_total_tokens": 16384,
  "model_turn_timeout": 120,
  "tool_turn_timeout": null,
  "total_turn_timeout": 150,
  "attempt_timeout": 180,
  "verifier_timeout": 10,
  "cleanup_timeout": 10
}
```

The experiment requests thinking for the reasoning family and disables it for
the other families using the template's `enable_thinking` parameter. Prefill flags
come from `PrefillHarness`; tool declarations come from the task. These are part of the captured request
and are checked by the auditor. Verify support for these template controls on
the chosen backend; CPU tests do not validate a deployed model's rendering.

```bash
uv run python -m experiments.post_training.structure_closure.evaluate \
  --tasks /path/to/tasks --endpoint "$MODEL_ENDPOINT" --model "$MODEL_NAME" \
  --initializer /path/to/local/tokenizer \
  --sampling /path/to/sampling.json --limits /path/to/limits.json \
  --concurrency 4 --output /path/to/baseline-rollouts.jsonl

uv run python -m experiments.post_training.structure_closure.rollout_audit \
  --tasks /path/to/tasks --rollouts /path/to/baseline-rollouts.jsonl \
  --sampling /path/to/sampling.json --output /path/to/baseline-audit.json
```

For a later checkpoint, repeat evaluation against that checkpoint's endpoint
with the same tasks and sampling settings. Add
`--baseline /path/to/baseline-audit.json` to its audit command for per-family and
per-control success-rate changes. Both audits must be complete and must match
the ordered task cohort and sampling settings.

Each evaluation record contains the task identity and fingerprint, the actual
model request and a generic `GradedRollout`. The audit regrades the captured
conversation. It accepts the model's own tokenization, semantic JSON variations
and output after the first complete outer structure. It separately reports
trailing output, whole-response success, premature closure and tool protocol
status. Duplicate, missing, interrupted or inconsistent evidence invalidates a
complete comparison. Family denominators remain the full expected cohort;
unscored cases are reported explicitly. A valid model failure stays a scored
failure rather than disappearing from coverage.

API callers can supply their own `Model` to `evaluate_task`. It must expose the
raw continuation as assistant content. A parsed reasoning-channel response is
rejected as unscored evidence rather than recorded as a failed model answer.

Evaluation records include public prompts and generated content. Audit reports
include case identities and results. Keep source tasks and private references
under the same access controls as the input dataset. No evaluation here proves
that a historical development panel is an independent final benchmark.

## Historical reproduction

The model-specific reproduction tools remain available:

| Module | Functionality |
| --- | --- |
| `token_data` | Pinned-template token targets, conditional BPE round trips, context budgets and tokenizer/template fingerprints |
| `tool_prompt_probe` | Four paired tool-probe token exports with a matching historical control arm |
| `eval_audit` | Exact prompt-token checks and reward/metric checks for historical session-format evaluation dumps |

These tools retain their command-line interfaces. Their manifests record failed
conversions, and consumers reject failed or modified cohorts. Exact historical
conditioning checks apply to these reproduction artifacts; ordinary TaskSpecs
and generic rollout audits use the text completion contract.
