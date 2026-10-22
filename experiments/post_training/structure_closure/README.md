# Structural completion reproduction

The [BFCL curation source](../task_curation/datasets/bfcl_structure_closure/README.md)
creates training and validation TaskSpecs during `process_rows` for RolloutEngine's
`PrefillHarness`. These commands retain the case and token exports needed to
reproduce historical prompts and evaluation audits.

## Exact-token export

Create the case files and the training-only near-boundary variant:

```bash
uv run python -m experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases \
  --source /path/to/bfcl --split /path/to/split.json --output /path/to/cases

uv run python -m experiments.post_training.task_curation.datasets.bfcl_structure_closure.prefixes \
  --source /path/to/cases --output /path/to/near-boundary-cases
```

The token exporter preserves assistant prefixes and records tokenizer/template
fingerprints, context budgets, and conditional BPE round trips:

```bash
uv run python -m experiments.post_training.structure_closure.token_data \
  --cases /path/to/near-boundary-cases --initializer /path/to/local/tokenizer \
  --context-tokens 16384 --tool-prompt compact --output /path/to/tokens
```

`tool_prompt_probe` exports four paired tool-prompt arms as token records.
`eval_audit` scores historical session-format evaluation dumps against their
validation token cohort. It reports syntax and content-preserving success by
family and control, and incomplete coverage for missing, duplicate, or
unverified responses:

```bash
uv run python -m experiments.post_training.structure_closure.eval_audit \
  --tokens /path/to/tokens --eval-dump /path/to/eval.jsonl \
  --uid-source row_index --output /path/to/report.json
```

Use `--uid-source case_id` for dumps keyed by stable task IDs. Row-index dumps
must use the original unfiltered cohort order with closure rows first. Generated
TaskSpecs use the generic grading and rollout APIs described in
[task rollouts](../../../docs/references/task-rollouts.md).
