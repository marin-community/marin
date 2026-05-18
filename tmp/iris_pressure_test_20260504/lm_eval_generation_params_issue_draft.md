Title: [evals] Pass generation_params to lm-eval harness

Labels: bug, agent-generated

Posting notes:
- Duplicate check: `gh issue list --repo marin-community/marin --state open --search "lm-eval generation_params max_gen_toks"` returned #4827, a broader served-lm-eval tracker. I did not find an exact open bug for this failure.
- Evidence artifacts:
  - failing log: `tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1739.txt`
  - successful fixed run log: `tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1744.txt`
  - successful result JSON: `tmp/iris_pressure_test_20260504/rebased_standard_smoke_results_20260514T003544Z.json`

Issue body:

**Describe the bug**
`EvaluationConfig.generation_params` is accepted and copied into `ModelConfig`, but the `lm_evaluation_harness` evaluator on `main` does not pass it to `lm_eval.simple_evaluate`. Generation tasks with task-level `generation_kwargs` therefore ignore the caller override.

This breaks small-context vLLM evals. Stock `humaneval` sets `max_gen_toks: 1024`; a caller setting `generation_params={"max_gen_toks": 128}` still sends `max_tokens=1024` to `/v1/chat/completions`.

**To Reproduce**
1. On `main` at `592ca9a60`, submit an Iris job that calls `marin.evaluation.run.evaluate` with:
   - `evaluator="lm_evaluation_harness"`
   - `model_path="Qwen/Qwen3-0.6B"`
   - `evals=[EvalTaskConfig("humaneval", 0, task_alias="humaneval_0shot")]`
   - `apply_chat_template=True`
   - `engine_kwargs={"tokenizer": "Qwen/Qwen3-0.6B", "max_model_len": 1024, "max_num_batched_tokens": 1024}`
   - `generation_params={"max_gen_toks": 128}`
2. Run it on Iris with `--extra eval --extra tpu --extra vllm` and `--tpu v6e-4`.
3. The job reaches vLLM, then lm-eval retries `/v1/chat/completions` and fails:

```text
This model's maximum context length is 1024 tokens. However, you requested 1024 output tokens and your prompt contains 398 characters ...
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: http://127.0.0.1:8000/v1/chat/completions
```

**Expected behavior**
`generation_params` should be forwarded to lm-eval as `gen_kwargs`, so task YAML defaults can be overridden by Marin eval config. In the repro above, vLLM should receive `max_tokens=128`, not `max_tokens=1024`.

**Additional context**
The missing handoff is in `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`: the `simple_evaluate(...)` call does not include `gen_kwargs`. `EvaluationConfig.generation_params` is already defined in `lib/marin/src/marin/evaluation/evaluation_config.py` and copied into `ModelConfig` by `marin.evaluation.run`.

The one-line fix is to pass:

```python
gen_kwargs=model.generation_params or None
```

to `simple_evaluate`. A follow-up Iris smoke with that change succeeded as `/romain/iris-inference-standard-smoke-20260514T003544Z`; the saved lm-eval result records `generation_kwargs.max_gen_toks: 128` and `base_url: http://127.0.0.1:8000/v1/chat/completions`.
