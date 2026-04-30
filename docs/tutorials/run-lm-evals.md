# Running Evaluations with Marin

This guide shows the current evaluation entrypoints in Marin. For a high-level overview of the evaluation stack, see [Evaluation Overview](../explanations/evaluation.md).

## Prerequisites

- A trained model checkpoint in Hugging Face format, or an existing `ExecutorStep` from training.
- Access to the TPU or GPU resources required by the evaluator you choose.

## Core APIs

The canonical helpers live in `experiments/evals/evals.py`:

```python
from experiments.evals.evals import (
    default_eval,
    default_key_evals,
    evaluate_lm_evaluation_harness,
    evaluate_levanter_lm_evaluation_harness,
)
```

- `default_eval` runs `CORE_TASKS` through the Levanter LM evaluation harness by default.
- `default_key_evals` returns the current "key evals" bundle: one generation step over `KEY_GENERATION_TASKS` and one multiple-choice step over `KEY_MULTIPLE_CHOICE_TASKS`.
- `evaluate_lm_evaluation_harness` is the lower-level helper for custom vLLM-backed LM-eval runs.
- `evaluate_levanter_lm_evaluation_harness` is the lower-level helper for Levanter-backed evaluation runs.

Task sets are defined in `experiments/evals/task_configs.py`. The most commonly used ones are:

- `CORE_TASKS`
- `CORE_TASKS_PLUS_MMLU`
- `KEY_GENERATION_TASKS`
- `KEY_MULTIPLE_CHOICE_TASKS`

## 1. Run `CORE_TASKS`

Use `default_eval` when you want the default multiple-choice evaluation suite:

```python
from fray.cluster import ResourceConfig
from experiments.evals.evals import default_eval
from marin.execution.executor import executor_main

model_path = "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m"

core_eval_step = default_eval(
    step=model_path,
    resource_config=ResourceConfig.with_tpu("v4-8"),
    # Optional overrides:
    # evals=CORE_TASKS_PLUS_MMLU,
    # max_eval_instances=100,
)

if __name__ == "__main__":
    executor_main(steps=[core_eval_step])
```

- `default_eval` accepts a checkpoint path, an `ExecutorStep`, or an `InputName`.
- To include MMLU in this path, pass `evals=CORE_TASKS_PLUS_MMLU`.

## 2. Run the Current Key-Evals Bundle

Use `default_key_evals` for the repository's current key-eval bundle:

```python
from fray.cluster import ResourceConfig
from experiments.evals.evals import default_key_evals
from marin.execution.executor import executor_main

model_path = "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m"

key_steps = default_key_evals(
    step=model_path,
    resource_config=ResourceConfig.with_tpu("v6e-8"),
    model_name="my_key_evals",
    # max_eval_instances=50,
)

if __name__ == "__main__":
    executor_main(steps=key_steps)
```

Today, `default_key_evals` returns two `ExecutorStep`s:

1. A generation run over `KEY_GENERATION_TASKS` using `evaluate_lm_evaluation_harness`.
2. A multiple-choice run over `KEY_MULTIPLE_CHOICE_TASKS` using `evaluate_levanter_lm_evaluation_harness`.

At the time of writing, `KEY_GENERATION_TASKS` includes:

- `ifeval`
- `gsm8k_cot`
- `drop`
- `humaneval`
- `bbh_cot_fewshot`
- `minerva_math`

`KEY_MULTIPLE_CHOICE_TASKS` currently includes:

- `mmlu` 0-shot
- `mmlu` 5-shot
- `truthfulqa_mc2`

## 3. Build a Custom Eval Step

Use the lower-level helpers when you want a custom task list or evaluator:

```python
from fray.cluster import ResourceConfig
from experiments.evals.evals import evaluate_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

custom_tasks = [
    EvalTaskConfig(name="commonsense_qa", num_fewshot=5),
    EvalTaskConfig(name="openbookqa", num_fewshot=0),
]

custom_step = evaluate_lm_evaluation_harness(
    model_name="custom_eval",
    model_path="gs://path/to/model",
    evals=custom_tasks,
    resource_config=ResourceConfig.with_tpu("v4-8"),
    max_eval_instances=200,
)

if __name__ == "__main__":
    executor_main(steps=[custom_step])
```

Use `evaluate_levanter_lm_evaluation_harness` instead when you specifically want the Levanter-backed evaluator path used by `default_eval`.

## 4. Run the Repository Example Scripts

The checked-in examples under `experiments/evals/` are the safest starting points because they track real repository usage:

```bash
uv run python experiments/evals/run_key_evals.py
uv run python experiments/evals/run_base_model_evals.py
uv run python experiments/evals/run_sft_model_evals.py
uv run python experiments/evals/run_on_gpu.py
```

These scripts launch the requested hardware, load the selected checkpoint or model definition, run the configured eval tasks, and log results to W&B.

## Parameter Reference

### `default_eval`

- `step`: checkpoint path, `ExecutorStep`, or `InputName` to evaluate.
- `resource_config`: hardware configuration for the evaluator.
- `evals`: optional override for the task list. Defaults to `CORE_TASKS`.
- `max_eval_instances`: optional cap on evaluated examples.
- `apply_chat_template`: whether to apply the model chat template before evaluation.
- `discover_latest_checkpoint`: whether to resolve the latest checkpoint under the provided path.

### `default_key_evals`

- `step`: checkpoint path, `ExecutorStep`, or `InputName` to evaluate.
- `resource_config`: hardware configuration for both returned steps.
- `model_name`: optional override for the logged model name.
- `max_eval_instances`: optional cap on evaluated examples.
- `engine_kwargs`: optional vLLM engine overrides for the generation step.

### `evaluate_lm_evaluation_harness`

- `model_name`: run name for tracking.
- `model_path`: checkpoint path to evaluate.
- `evals`: list of `EvalTaskConfig` entries to run.
- `max_eval_instances`: optional cap on evaluated examples.
- `engine_kwargs`: optional vLLM engine overrides.
- `resource_config`: optional hardware configuration.
- `apply_chat_template`: whether to apply the chat template before evaluation.
- `wandb_tags`: optional W&B tags.
- `discover_latest_checkpoint`: whether to resolve the latest checkpoint under the provided path.

### `evaluate_levanter_lm_evaluation_harness`

- `model_name`: run name used to construct the executor step.
- `model_path`: checkpoint path to evaluate.
- `evals`: list of `EvalTaskConfig` entries to run.
- `resource_config`: hardware configuration.
- `max_eval_instances`: optional cap on evaluated examples.
- `apply_chat_template`: whether to apply the chat template before evaluation.
- `discover_latest_checkpoint`: whether to resolve the latest checkpoint under the provided path.

For deeper dives, see:

- `docs/explanations/evaluation.md`
- `experiments/evals/task_configs.py`
- `experiments/evals/evals.py`

## Raw Perplexity Gap Datasets

The raw perplexity-gap workflow uses `default_raw_validation_sets()` from `experiments/defaults.py`. That bundle now includes:

- Paloma
- Uncheatable Eval
- Curated capability-family slices for:
  - `chat/wildchat`
  - `agent_traces/openhands_swe_rebench`
  - `reasoning_qa/gsm8k_main`
  - `reasoning_qa/global_mgsm_en`

These capability datasets are first normalized into reusable OpenAI-chat JSONL artifacts under each step's `oai/` output. Consumers that want Levanter chat tokenization can use `capability_chat_validation_components()`, which wraps those rows in `ChatLmDatasetFormat` with `MARIN_CHAT_TEMPLATE`. The raw gap finder still consumes plain `text`, so the same step also writes a derived `raw_text/` projection using Marin's chat-token surface. OpenHands traces keep the full system/user/tool conversation in the OAI artifact, while the raw-text projection scores only assistant-generated trace targets and final patches.

The curated default uses modest, reproducible slices for the larger structured corpora rather than mirroring whole Hugging Face datasets into GCS. That keeps cost and executor output size bounded while still giving useful coverage for base-model PPL comparisons.

If you want the gated chat sources as well, use `extended_raw_validation_sets()` instead of `default_raw_validation_sets()`. That currently adds:

- `chat/lima_train`
- `chat/lmsys_chat_1m`

Those opt-in datasets stay out of the default bundle because access and licensing are more restrictive.
