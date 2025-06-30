# Running Alpaca Eval with Marin

This tutorial shows how to configure and launch the Alpaca evaluation pipeline in Marin, using the `evaluate_alpaca_eval` helper from `experiments/evals/evals.py`. For an overview of evaluation concepts, see [Evaluation Overview](../explanations/evaluation.md).

## Prerequisites

- A trained model checkpoint in Hugging Face (HF) format.
- Access to TPUs. AlpacaEval via GPU is in progress and we recommend using the main [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval/tree/main) library on GPU at present.

## Required Imports

The default evaluation script for alpaca is `experiments/evals/run_alpaca_eval.py`), if for some reason you want to make your own script import:

```python
from experiments.evals.engine_configs import DEFAULT_VLLM_ENGINE_KWARGS
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.evals.evals          import evaluate_alpaca_eval
from marin.execution.executor         import ExecutorMainConfig, executor_main
```

## Example Eval Script

```python
# nodryrun
from experiments.evals.engine_configs import DEFAULT_VLLM_ENGINE_KWARGS
from experiments.evals.evals          import evaluate_alpaca_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor         import ExecutorMainConfig, executor_main

# Retry any failed steps by default
executor_main_config = ExecutorMainConfig(force_run_failed=True)

steps = [
    evaluate_alpaca_eval(
        model_name="my_alpaca_model_eval",              # Name for logging / W&B
        model_path="path/to/your/model/checkpoint/hf/",  # HF checkpoint directory
        resource_config=SINGLE_TPU_V6E_8,                 # E.g., TPU v6e-8; choose GPU/TPU config
        engine_kwargs=DEFAULT_VLLM_ENGINE_KWARGS,         # vLLM backend parameters

        # IMPORTANT: stop_token_ids must include the eos_token_id of your HF model.
        # You can fetch it via:
        #   from transformers import AutoConfig
        #   AutoConfig.from_pretrained(<your_model>).eos_token_id
        stop_token_ids=[<YOUR_EOS_TOKEN_ID>],

        # Optional overrides (defaults shown):
        # max_eval_instances: int | None     = None
        # temperature:          float         = 0.7
        # presence_penalty:     float         = 0.0
        # frequency_penalty:    float         = 0.0
        # repetition_penalty:   float         = 1.0
        # top_p:                float         = 1.0
        # top_k:                int           = -1
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
```


## Parameter Reference

| Argument             | Type                 | Description |
|----------------------|----------------------|-------------|
| model_name           | `str`                | Name for experiment tracking through executor framework. |
| model_path           | `str`                | Path on GCP or URL to HF-format model checkpoint. |
| resource_config      | `ResourceConfig`     | Hardware spec (e.g. `SINGLE_TPU_V6E_8`). |
| engine_kwargs        | `dict  None`   | vLLM engine settings (e.g. batch size, sequence length). |
| max_eval_instances   | `int  None`    | Limits the number of examples to evaluate; `None` = all. |
| temperature          | `float`              | Sampling temperature. |
| presence_penalty     | `float`              | Penalize new token presence. |
| frequency_penalty    | `float`              | Penalize repeated tokens. |
| repetition_penalty   | `float`              | Another repetition control. |
| top_p                | `float`              | Nucleus sampling threshold. Fixed to 1.0 for TPU |
| top_k                | `int`                | Top-K sampling; `-1` disables. |
| stop_token_ids       | `List[int]`  None | Token IDs at which generation halts—**must include your model's `eos_token_id`.** |

## Launching the Eval

From your workspace root, run:

```bash
python experiments/evals/run_alpaca_eval.py
```

This will:
1. Spin up the specified hardware (Ray + TPU).
2. Load your model checkpoint.
3. Run the Alpaca prompts through a vLLM-powered loop.
4. Save and log results (accuracy, generation samples) to W&B under `model_name`.

---

## Common pitfals

If there is a discrepancy between the `eos_token_id` between the tokenizer_config.json and model_config.json then
***make sure to pass in correct stop_token_id*** otherwise the generation will continue until the max model length.
If you don't specify this the alpaca_evaluator will guess what those tokens are from the HF config but that is often wrong :)

 To find your model's `eos_token_id`, do:
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("path/to/your/model")
print(config.eos_token_id)
```

If that fails check the generation_config.json
