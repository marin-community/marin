# Evaluation Guide

## Overview

### 1. Evaluator summary
By default, we use `lm-evaluation-harness` in Levanter for running all our evaluations; this runs on TPU using the executor framework.
- Training a model using `default_train` automatically runs in-the-loop evaluations every 10,000 steps and logs it to WandB.
- If you want to evaluate a model step after it has been trained, you can use `default_eval` from [`evals.py`](evals.py) as a a step in your experiment script.

### 2. Evaluation tasks
The list of tasks we evaluate models on is configured in [`task_configs.py`](../../experiments/evals/task_configs.py).
- By default, we run `CORE_TASKS` for both in-the-loop and external evaluation. This is a subset of the [DCLM CORE tasks](https://arxiv.org/html/2406.11794v3#A7). Note that this does not include MMLU by default.
- If you want to include MMLU in your evaluation run, you can use `CORE_TASKS_PLUS_MMLU` within `default_train` or `default_eval`.
- To define a new set of evals, you can also just create a new set of tasks within [`task_configs.py`](task_configs.py) and pass in that list instead.

NOTE: See [`marin/evaluation/evaluators/levanter_lm_eval_evaluator.py`](../../marin/evaluation/evaluators/levanter_lm_eval_evaluator.py) for the default evaluator code. For other evaluators, including running `lm-evaluation-harness` on GPU, HELM, and Alpaca, see [`marin/evaluation/evaluators`](../../marin/evaluation/evaluators/).

### 3. Generation-based tasks
There are a few generation-based tasks that require a bit more setup since they require a fast inference backend (e.g. vLLM). This include but are not limited to Alpaca Eval, HumanEval, GSM8K, MATH, and many more. See the `task_configs.py` file for more details for tasks that we currently support. Because vLLM is not on the original cluster, you will need to use the `Dockerfile.vllm` to run these tasks or use the separate vLLM ray cluster (e.g. infra/marin-us-east5-b-vllm.yaml). The entrypoint to the key evals that we seek to run is in `experiments/evals/run_key_evals.py`. We currently support models up to around 100B parameters. For an 8B model, you can run the model using a single TPU v6e-8 chip, but for larger models like 70B, you may need to specify to use tensor parallelism. You can also specify the node type in the `resource_configs.py` file.

#### Evaluation Metrics
We use a fork of `lm-evaluation-harness` (https://github.com/stanford-crfm/lm-evaluation-harness); in addition to the accuracies already tracked by `lm-evaluation-harness`, we also compute the following metrics:

1. **Bits per Byte (bpb)**: `bpb = -log_prob / byte_length * ln(2)`

2. **Log Probability (logprob)**: Raw log probability of the correct answer.

3. **Choice Log Probability (choice_logprob)**: `log_prob_correct - log(∑exp(log_prob_i))`

4. **Choice Probability Normalized (choice_prob_norm)**: `exp(log_prob_correct/(byte_length_correct * ln(2))) / ∑exp(log_prob_i/(byte_length_i * ln(2)))`
