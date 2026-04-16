# Evaluation Overview

This document explains how Marin evaluates models and where to find runnable workflows.

For step-by-step usage, start with:
- [Running Evaluations with Marin](../tutorials/run-lm-evals.md) for multiple-choice, generation, and key eval suites.
- [Harbor Framework Integration](../harbor-integration.md) for Harbor-backed agent and benchmark evaluation.

## Evaluation modes

Marin supports three primary evaluation paths:

- **Multiple-choice tasks**: run through `lm-evaluation-harness` (during training or as standalone eval jobs).
- **Generation tasks**: run through a vLLM-backed evaluator pipeline.
- **Harbor tasks**: run through Marin's Harbor integration for containerized agent benchmarks and registry datasets.

## Multiple-choice evaluation (LM Evaluation Harness)

For multiple-choice tasks, Marin uses a fork of `lm-evaluation-harness`:
https://github.com/stanford-crfm/lm-evaluation-harness

Key integration points:
- [`default_train`][experiments.defaults.default_train] runs in-loop evaluations periodically and logs to W&B.
- [`default_eval`][experiments.evals.evals.default_eval] runs standalone harness evaluation after training (or on an existing checkpoint).

### Task sets

Task sets are configured in [`task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py).

- `CORE_TASKS` is the default for in-loop and standalone harness evals.
- `CORE_TASKS_PLUS_MMLU` extends `CORE_TASKS` with MMLU.
- You can define custom task lists in `task_configs.py` and pass them to `default_eval`.

!!! note

    See [`levanter_lm_eval_evaluator.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/evaluation/evaluators/levanter_lm_eval_evaluator.py) for the default evaluator implementation.
    Additional evaluators (including HELM, Alpaca, and other backends) live in [`lib/marin/src/marin/evaluation/evaluators`](https://github.com/marin-community/marin/tree/main/lib/marin/src/marin/evaluation/evaluators).

### Reported metrics

Beyond task accuracy, Marin tracks these multiple-choice metrics:

1. **Bits per byte (`bpb`)**: `bpb = -log_prob / byte_length * ln(2)`
2. **Log probability (`logprob`)**: raw log probability of the correct answer.
3. **Choice log probability (`choice_logprob`)**: `log_prob_correct - log(sum(exp(log_prob_i)))`
4. **Length-normalized choice probability (`choice_prob_norm`)**:
   `exp(log_prob_correct / (byte_length_correct * ln(2))) / sum(exp(log_prob_i / (byte_length_i * ln(2))))`

## Generation-based evaluation

Generation tasks (for example AlpacaEval, HumanEval, GSM8K, and MATH) use a fast inference backend, typically vLLM.

- Task and suite definitions are in [`task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py).
- A common entrypoint is [`run_key_evals.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/run_key_evals.py).
- Current generation-eval setup is documented in [Running Evaluations with Marin](../tutorials/run-lm-evals.md).

In current Marin workflows, generation evals are commonly run with `Dockerfile.vllm` or on a dedicated vLLM Ray cluster (for example [`marin-us-east5-b-vllm.yaml`](https://github.com/marin-community/marin/blob/main/infra/marin-us-east5-b-vllm.yaml)).

## Harbor-based evaluation

Harbor tasks use [`evaluate_harbor`](https://github.com/marin-community/marin/blob/main/experiments/evals/evals.py) and the Harbor evaluator integration to run registry datasets in containerized environments.

- Harbor supports agent-style benchmarks such as AIME, Terminal-Bench, SWE-bench Verified, and other registry datasets.
- Marin's Harbor integration supports local Docker and hosted environments such as Daytona, E2B, and Modal.
- Setup, examples, and environment requirements are documented in [Harbor Framework Integration](../harbor-integration.md).

## Where to go next

- [Running Evaluations with Marin](../tutorials/run-lm-evals.md)
- [Harbor Framework Integration](../harbor-integration.md)
- [`experiments/evals/evals.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/evals.py)
- [`experiments/evals/task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py)
