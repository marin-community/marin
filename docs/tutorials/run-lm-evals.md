# Running LM Evaluations with Marin

This guide shows how to add and run lm-eval harness evaluation tasks in Mari For a high‐level overview of evaluation concepts, see [Evaluation Overview](../explanations/evaluation.md).

## Prerequisites

- A trained model checkpoint in Hugging Face (HF) format or an existing `ExecutorStep` from training.

## Required Imports

```python
# Core evaluation functions
from experiments.evals.evals import (
    default_eval,            # Runs CORE_TASKS via LM Evaluation Harness
    default_key_evals,       # Runs KEY_GENERATION_TASKS, KEY_MULTIPLE_CHOICE_TASKS, and Alpaca
    evaluate_lm_evaluation_harness,  # Custom MCQA eval step
)

# Task configuration constants
from experiments.evals.task_configs import (
    CORE_TASKS,             # Default multiple‐choice tasks
    CORE_TASKS_PLUS_MMLU,   # CORE + MMLU
    KEY_GENERATION_TASKS,   # Generation tasks (e.g. GSM8K, Alpaca)
    KEY_MULTIPLE_CHOICE_TASKS,  # MMLU few‐shot
)

# Hardware / executor
from experiments.evals.resource_configs import SINGLE_TPU_V4_8, SINGLE_TPU_V6E_8
from marin.execution.executor import executor_main
from marin.execution.executor import ExecutorMainConfig  # for retry logic
```

## 1. Multiple‐Choice Eval: CORE_TASKS

Run the canonical CORE_TASKS (subset of DCLM tasks) via LM Evaluation Harness:

```python
# run_mcqa_eval.py
from experiments.evals.evals import default_eval
from experiments.evals.resource_configs import SINGLE_TPU_V4_8
from marin.execution.executor import executor_main

# Example: evaluate a standalone checkpoint
model_path = "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m"

# This creates an ExecutorStep that runs CORE_TASKS
core_evals_step = default_eval(
    step=model_path,
    resource_config=SINGLE_TPU_V4_8,
    # Optional: override the task set:
    # evals=CORE_TASKS_PLUS_MMLU,
    # max_eval_instances=100,
)

if __name__ == "__main__":
    executor_main(steps=[core_evals_step])
```

- `default_eval` wraps `evaluate_lm_evaluation_harness` with `CORE_TASKS` by default.
- To include MMLU, pass `evals=CORE_TASKS_PLUS_MMLU`.

## 2. Key Evals: Generation + MMLU + Alpaca

Use `default_key_evals` to run a collection of generation tasks (`KEY_GENERATION_TASKS`), MMLU (`KEY_MULTIPLE_CHOICE_TASKS`), and an Alpaca‐style generation eval:

```python
# run_key_evals.py  (see 1:18:experiments/evals/run_key_evals.py)
from experiments.evals.evals import default_key_evals
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import executor_main

# Point to your checkpoint or a training ExecutorStep
model_path = "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m"

# This returns a list of three ExecutorSteps:
#  1) generation tasks (e.g. gsm8k, humaneval)
#  2) MMLU few-shot
#  3) Alpaca eval
key_steps = default_key_evals(
    step=model_path,
    resource_config=SINGLE_TPU_V6E_8,
    model_name="my_key_evals",
    # max_eval_instances=50,
)

if __name__ == "__main__":
    executor_main(steps=key_steps)
```

- `KEY_GENERATION_TASKS` is defined here: 1:35:experiments/evals/task_configs.py
- `KEY_MULTIPLE_CHOICE_TASKS` (MMLU) is defined alongside.

### Alpaca Eval in Key Evals

`default_key_evals` automatically calls `evaluate_alpaca_eval` with a `stop_token_ids` list inferred from the model name:
- If `"llama3"` in `model_name`, it uses `[128009]`.
- If `"olmo"` in `model_name`, it uses `[100257]`.

To customize stop tokens manually, call `evaluate_alpaca_eval` directly:

```python
from experiments.evals.evals import evaluate_alpaca_eval

alpaca_step = evaluate_alpaca_eval(
    model_name="my_model",
    model_path="...",
    resource_config=SINGLE_TPU_V6E_8,
    engine_kwargs=DEFAULT_VLLM_ENGINE_KWARGS,
    stop_token_ids=[<YOUR_EOS_TOKEN_ID>],  # must match your HF model's eos_token_id
)
```

*Find your `eos_token_id` with:*  `AutoConfig.from_pretrained(<model>).eos_token_id`.

## 3. Custom Eval Harness (Advanced)

If you want fine‐grained control over which tasks to run:

```python
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.evals.resource_configs import SINGLE_TPU_V4_8
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# Define a custom list of EvalTaskConfig
custom_tasks = [
    EvalTaskConfig(name="commonsense_qa", num_fewshot=5),
    EvalTaskConfig(name="openbookqa", num_fewshot=0),
]

custom_step = evaluate_lm_evaluation_harness(
    model_name="custom_eval",
    model_path="...",
    evals=custom_tasks,
    resource_config=SINGLE_TPU_V4_8,
    max_eval_instances=200,
)

if __name__ == "__main__":
    executor_main(steps=[custom_step])
```

## 4. Launching Evaluation

From the workspace root, pick your script and run:

```bash
python experiments/evals/run_mcqa_eval.py
python experiments/evals/run_key_evals.py
python experiments/evals/run_on_gpu.py
```
Each will:
1. Launch Ray + specified TPU/GPU hardware.
2. Load the model checkpoint.
3. Run the chosen eval tasks.
4. Log results (accuracy, generation metrics) to W&B under your `model_name`.

## Parameter Reference

### evaluate_lm_evaluation_harness (multiple‐choice)
- `model_name: str` — run name for tracking.
- `model_path: str` — HF checkpoint path or pipeline input.
- `evals: list[EvalTaskConfig]` — which tasks to run (e.g. `CORE_TASKS`).
- `resource_config: ResourceConfig` — hardware spec.
- `max_eval_instances: int  None` — limit number of examples.
- `engine_kwargs: dict  None` — passes through to vLLM if launching with Ray.

### evaluate_alpaca_eval (generation)
- `model_name`, `model_path`, `resource_config`, `engine_kwargs` — same as above.
- `temperature: float` — sampling randomness.
- `presence_penalty, frequency_penalty, repetition_penalty: float` — control repetition.
- `top_p: float`, `top_k: int` — nucleus/top‐k sampling.
- `stop_token_ids: list[int]` — **must include your HF model's `eos_token_id`.**
Please look at the [Running Alpaca Eval tutorial](run-alpaca-eval.md).

### default_eval
- Wraps `evaluate_lm_evaluation_harness` with `CORE_TASKS` (or custom).

### default_key_evals
- Runs three steps: generation tasks, MMLU, Alpaca eval (with inferred `stop_token_ids`).

---

For deeper dives, see:
- `docs/tutorials/train-an-lm.md` for training.
- `docs/explanations/evaluation.md` for evaluation concepts.
- `experiments/evals/task_configs.py` for full task lists.
- `experiments/evals/evals.py` for implementation details.
