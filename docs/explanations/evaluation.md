# Evaluation Overview

This document explains how Marin evaluates models and where to find runnable workflows.

For step-by-step usage, start with:

- [Running Evaluations with Marin](../tutorials/run-lm-evals.md) for command lines covering
  Evalchemy and Harbor across registered models.
- [Harbor Framework Integration](../harbor-integration.md) for Harbor dataset, agent, endpoint, and
  result details.

## Evaluation modes

Marin supports three evaluation paths:

- **In-loop training evals**: [`train_lm`][marin.experiment.train.train_lm] runs periodic multiple-choice evaluations through Levanter's lm-evaluation-harness integration and logs to W&B when an `EvalSuite` is provided.
- **Post-hoc evals**: the shared launcher or composable `EvalGroup`s evaluate multiple-choice and
  generation tasks with the Evalchemy fork.
- **Harbor tasks**: the shared launcher runs containerized agent benchmarks and registry datasets
  through Marin's Harbor integration.

## Post-hoc evaluation (evalchemy over a served endpoint)

A post-hoc eval is decoupled from the model backend by an OpenAI-compatible URL. Each `EvalGroup` is
run inside one `remote_inference` context. The generic group runner passes its `RunningModel` to
`run_evalchemy`, then inference is torn down.
Multiple-choice tasks use the served backend's logprob API, so they run the same way as generation —
no separate JAX-logprob backend.

The lifecycle and the evaluator are separate APIs:

```python
with remote_inference(model, engine, iris) as session:
    run_evalchemy(session.model, eval_config, output_path, env_vars=env)
```

`remote_inference` owns startup, liveness, Iris link registration, and teardown. Endpoint-oriented
mechanisms such as `run_evalchemy`, `run_lm_eval`, and `run_harbor_driver` own only their native task and
result contracts. This keeps the lifecycle common without hiding mechanism-specific configuration
or outcomes behind a universal `do_eval` interface.

- [`eval_step`][marin.experiment.evaluation.eval_step] builds one post-hoc eval artifact from an
  `EvalGroup`; combine groups and aggregate them with
  [`eval_report`][marin.experiment.evaluation.eval_report]. Concrete task menus remain in
  `experiments/evals/evals.py`. See [Running Evaluations with Marin](../tutorials/run-lm-evals.md).

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

Levanter uses the pinned upstream EleutherAI harness with the `lm_eval` extra;
model inference remains in Levanter/JAX, including on TPU. Its Python task
extension supplies the core-task smooth metrics below. An explicit task
`metric_list` overrides these defaults. Missing or incompatible harness imports
raise before evaluation loads model weights.

Packed examples and background batches stay on CPU until dispatch transfers
each complete batch to the evaluation mesh.

Beyond task accuracy, the in-loop Levanter evaluator tracks these multiple-choice metrics:

1. **Bits per byte (`bpb`)**: `bpb = -log_prob / (byte_length * ln(2))`
2. **Log probability (`logprob`)**: raw log probability of the correct answer.
3. **Choice log probability (`choice_logprob`)**: `log_prob_correct - log(sum(exp(log_prob_i)))`
4. **Length-normalized choice probability (`choice_prob_norm`)**:
   `exp(log_prob_correct / (byte_length_correct * ln(2))) / sum(exp(log_prob_i / (byte_length_i * ln(2))))`
5. **Log normalized choice probability (`choice_logprob_norm`)**: `ln(choice_prob_norm + 1e-30)`.

Byte lengths count UTF-8 bytes, with a minimum length of one. For MMLU, group
accuracy is document-weighted; group smooth metrics average subjects equally.

## Generation tasks

Generation tasks (for example HumanEval, GSM8K, and MATH) run through the same post-hoc evalchemy
path as multiple-choice tasks: the served endpoint answers completion requests, and evalchemy scores
them.

- Task and suite definitions are in [`task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py).
- A common entrypoint is [`run_key_evals.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/run_key_evals.py).

## Table-9 accuracy companions for the Qwen3 scaling ladder

The frozen Table-9 objective has 51 BPB components. Accuracy coverage is tracked
separately: scoring a gold continuation does not establish choice accuracy or
math/code correctness. The ladder workflow currently targets 34 components:

| Evaluation | Components | Runner |
| --- | ---: | --- |
| Existing QA/MMLU overlap | 14 | `experiments.domain_phase_mix.evaluate_mariner_ladder_accuracy` |
| Basic Skills and gen2mc QA | 11 | `experiments.domain_phase_mix.evaluate_table9_accuracy --mode choices` |
| Seven Minerva subjects, HumanEval, Python MBPP | 9 | `experiments.domain_phase_mix.evaluate_table9_accuracy --mode generation` |
| Multilingual MBPP without execution tests | 17 | Deferred; not counted as scored |

The 20 backfill tasks preserve the frozen native BPB document identities,
contexts, and gold references. An east5 CPU parent verifies that parity before
releasing TPU work. The existing 14 overlap counterparts use their own pinned
lm-eval prompts, not exact native Table-9 prompts. Reports retain that distinction.

Inference uses each checkpoint's Qwen3 architecture and tokenizer on a
single-host v5p-8 in us-east5-a. Each mode requires a two-document-per-task canary
before full release. Task outputs are resumable and include every choice score
or generated completion. Canary outputs cannot satisfy full-run completion.

`experiments.domain_phase_mix.grade_table9_accuracy` grades saved generations
without rerunning inference. Minerva retains the harness's `exact_match` and
`math_verify`; the coverage report uses `math_verify` for correctness. HumanEval
and MBPP use greedy pass@1 in disposable, network-disabled Docker containers
without host mounts or credentials. Docker must be running; infrastructure
failures are errors, not incorrect answers. Run grading and reporting with
`uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1`.

For a newly completed ladder checkpoint, use
`experiments.domain_phase_mix.prepare_table9_checkpoint --help`. It checks the
explicit permanent training step and regional HF export, pins object identities,
and creates both `overlap_plan.json` and `backfill_plan.json` from frozen protocol
templates. It does not submit jobs or attach an automatic training callback.
Submit the overlap suite and both backfill modes using those plans, with the
normal east5 placement validation and Fieldbook registration. Do not substitute
the preparation smoke-test plans for existing evaluations.

After grading, run `experiments.domain_phase_mix.report_table9_accuracy` with
`--plan`, `--overlap-plan`, and `--output`. It writes a per-component CSV and a
provenance-bearing JSON report. Ungraded generations remain missing accuracy;
deferred languages remain visible. It never calls 34/51 complete Table-9 coverage
or publishes a partial 51-component accuracy macro.

## Harbor-based evaluation

Agentic launcher runs pass the same `RunningModel` boundary to
`run_harbor_driver`. Off-cluster sandboxes receive a scoped capability route resolved by the
inference session.

- Harbor supports agent-style benchmarks such as AIME, Terminal-Bench, SWE-bench Verified, and other registry datasets.
- Marin's Harbor integration supports local Docker and hosted environments such as Daytona, E2B, and Modal.
- Setup, examples, and environment requirements are documented in [Harbor Framework Integration](../harbor-integration.md).

## Where to go next

- [Running Evaluations with Marin](../tutorials/run-lm-evals.md)
- [Harbor Framework Integration](../harbor-integration.md)
- [`experiments/evals/evals.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/evals.py)
- [`experiments/evals/task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py)
