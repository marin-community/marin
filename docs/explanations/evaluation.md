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

Evalchemy and Harbor write their results directly into FineStore. Each evaluator preserves its native
files as source objects and writes the normalized `samples` and `steps` tables consumed by Evaldash.
After the evaluator seals those tables, Marin derives the shared `rollouts_v1` table from `samples` and
`steps`, then records provenance and aggregate results.

### Normalized rollout table

The versioned `rollouts_v1` table stores one ordered conversation part per row. Its primary key is
`(task, doc_id, trial_id, turn_id, part_id)`. `task`, `doc_id`, and `trial_id` identify one model
attempt; `turn_id` orders turns; and `part_id` orders content within a turn.

The remaining columns have the same meaning for both evaluators:

- `conversation_type` is `chat` when an Evalchemy sample has structured `prompt_messages`,
  `completion` when it has `prompt_text`, and `agentic` for Harbor steps.
- `participant_type` is the normalized role: `system`, `user`, `assistant`, `tool`, `environment`,
  or `other`. Evalchemy uses the message role as `participant_id`. Harbor uses an assistant step's
  model name when available and otherwise uses the ATIF source role.
- `content_type` is `message`, `reasoning`, `tool_call`, or `tool_result`. Structured tool content is
  JSON in `content`; `metadata_json` holds small type-specific annotations.
- Token counts, token IDs, log probabilities, and cost are attached to the first model-produced part
  of a Harbor turn when the trajectory provides them.

Evalchemy prompt messages and model responses become `completion` or `chat` rows. Multiple grading
filters for one Evalchemy response share one rollout. Harbor's normalized `steps` rows supply the
`agentic` messages, reasoning, tool calls, and observations. The evaluator-native JSON/JSONL and
complete Harbor trajectory remain source objects and blobs, so normalization does not replace the
raw record.

### Rollout run catalog

Completed and failed attempts are discoverable in Finelog's `marin.rollout_runs` table. Each row
identifies the logical run and concrete attempt, its producer and terminal status, the model and Iris
job when known, and the URI and format of the retained rollouts. Harbor and Evalchemy rows point to
the FineStore archive that contains `rollouts_v1` alongside the evaluator-owned tables and raw source
objects. SkyRL rows point to the retained trajectory directory and terminal model manifest; the
catalog contract does not depend on the trajectory storage format.

For example, list the newest successful rollout attempts with:

```sql
SELECT timestamp_ms, run_kind, producer, run_id, attempt_id, model,
       rollout_uri, storage_format
FROM "marin.rollout_runs"
WHERE status = 'succeeded'
ORDER BY timestamp_ms DESC
```

The table is an append-only discovery index, not the source of truth for rollout contents. Writers
emit one terminal row per attempt from Marin's evaluation orchestrator or SkyRL adapter. Evalchemy
and Harbor do not need to depend on Marin's Finelog schema: they continue to own their native
FineStore output, and Marin catalogs and normalizes it after the evaluator finishes.
Regional writers register the object-native table with their local Finelog server. The forwarder
then registers it on the hub before forwarding its first row, so operators do not pre-register the
table when enabling it on a new cluster.

- [`eval_step`][marin.experiment.evaluation.eval_step] builds one post-hoc eval artifact from an
  `EvalGroup`; combine groups and aggregate them with
  [`eval_report`][marin.experiment.evaluation.eval_report]. Concrete task menus remain in
  `experiments/evals/evals.py`. See [Running Evaluations with Marin](../tutorials/run-lm-evals.md).

One `EvalGroup` (a task set) becomes one `FineStoreEvalchemyResult` artifact addressed by
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
