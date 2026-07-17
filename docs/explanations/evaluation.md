# Evaluation Overview

This document explains how Marin evaluates models and where to find runnable workflows.

For step-by-step usage, start with:
- [Running Evaluations with Marin](../tutorials/run-lm-evals.md) for multiple-choice, generation, and key eval suites.
- [Harbor Framework Integration](../harbor-integration.md) for Harbor-backed agent and benchmark evaluation.

## Evaluation modes

Marin supports three evaluation paths:

- **In-loop training evals**: [`train_lm`][marin.experiment.train.train_lm] runs periodic multiple-choice evaluations through Levanter's lm-evaluation-harness integration and logs to W&B when an `EvalSuite` is provided.
- **Post-hoc evals**: composable `EvalGroup`s evaluated against a served model with the evalchemy fork — multiple-choice and generation both.
- **Harbor tasks**: run through Marin's Harbor integration for containerized agent benchmarks and registry datasets.

## Post-hoc evaluation (evalchemy over a served endpoint)

A post-hoc eval is decoupled from the model backend by an OpenAI-compatible URL. Each `EvalGroup` is
run by serving the model once as an OpenAI endpoint (marin-serve: vLLM or Levanter), evaluating its
tasks with the [evalchemy](https://github.com/marin-community/evalchemy) fork
(`eval.eval --model local-completions`), and tearing the server down. Multiple-choice tasks use the
served backend's logprob API, so they run the same way as generation — no separate JAX-logprob
backend. See [`serve_and_eval.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/evalchemy/serve_and_eval.py).

- [`eval_step`][experiments.evals.evals.eval_step] builds one post-hoc eval artifact from an `EvalGroup`; combine groups and aggregate them with [`eval_report`][experiments.evals.evals.eval_report]. See [Running Evaluations with Marin](../tutorials/run-lm-evals.md).

One `EvalGroup` (a task set) becomes one `EvalchemyResult` artifact addressed by
`evaluation/evalchemy/{model}/{group_id}`, so a pipeline picks up exactly the evals it needs and each
is cached and reused. The in-loop `EvalSuite` and the post-hoc `EvalGroup`s draw from the same task
menu.

### Task sets

Task sets are configured in [`task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py).

- `CORE_TASKS` is the default for in-loop and post-hoc multiple-choice evals.
- `CORE_TASKS_PLUS_MMLU` extends `CORE_TASKS` with MMLU.
- Named menus (`core_evals`, `key_evals`, `base_model_evals`) bundle task sets into `EvalGroup`s; you can also define custom task lists in `task_configs.py` and pass them to your own `EvalGroup`s.

### In-loop metrics

Beyond task accuracy, the in-loop Levanter evaluator tracks these multiple-choice metrics:

1. **Bits per byte (`bpb`)**: `bpb = -log_prob / byte_length * ln(2)`
2. **Log probability (`logprob`)**: raw log probability of the correct answer.
3. **Choice log probability (`choice_logprob`)**: `log_prob_correct - log(sum(exp(log_prob_i)))`
4. **Length-normalized choice probability (`choice_prob_norm`)**:
   `exp(log_prob_correct / (byte_length_correct * ln(2))) / sum(exp(log_prob_i / (byte_length_i * ln(2))))`

## Generation tasks

Generation tasks (for example HumanEval, GSM8K, and MATH) run through the same post-hoc evalchemy
path as multiple-choice tasks: the served endpoint answers completion requests, and evalchemy scores
them.

- Task and suite definitions are in [`task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py).
- A common entrypoint is [`run_key_evals.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/run_key_evals.py).

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
