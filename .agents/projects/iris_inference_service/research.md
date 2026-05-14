# Research — iris_inference_service

The design is vLLM-only. Earlier drafts considered both vLLM and Levanter/JAX
behind OpenAI-compatible HTTP, but current team direction is to serve this path
through vLLM and keep Levanter on existing non-served eval paths.

## In-Repo Findings

- Served evals already have the right boundary:
  [`OpenAIEndpoint`, `RunningModel`, `ModelDeployment`, and `ModelLauncher`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/types.py#L10-L45).
  Evaluators should consume `RunningModel`; launchers should own model
  lifecycle and cleanup.
- `run_lm_eval` converts `RunningModel` into `local-completions` or
  `local-chat-completions` args and writes results with `EvaluationTracker`:
  [`run_lm_eval`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/evaluation/lm_eval.py#L42-L70)
  and [`build_lm_eval_model_args`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/evaluation/lm_eval.py#L73-L92).
- `LMEvaluationHarnessEvaluator` already starts local vLLM and points lm-eval
  at `/v1/completions` or `/v1/chat/completions`, but it couples evaluator
  code to serving lifecycle:
  [`LMEvaluationHarnessEvaluator.evaluate`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py#L86-L190).
- `VllmEnvironment` owns native vLLM startup, readiness, diagnostics, and the
  `/v1` API root:
  [`VllmEnvironment`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L360-L430).
  Iris rejects Docker sidecar mode:
  [`resolve_vllm_mode`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L241-L252).
- vLLM readiness is `GET {server_url}/models == 200`, where `server_url` ends
  in `/v1`:
  [`_poll_until_ready`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L306-L344)
  and [`_get_first_model_id`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L347-L357).
- Fray has short `.remote()` calls and long-running `.submit()` calls. Proxy
  waits should use the long-running path:
  [`ActorMethod.submit`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/fray/src/fray/actor.py#L97-L112).
- Fray actor context lives in `ContextVar`s, and child threads do not inherit
  it automatically:
  [`current_actor`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/fray/src/fray/actor.py#L62-L76).
  This matters for any threaded eval-side proxy.
- `FrayIrisClient` already maps Fray jobs and actors to Iris jobs and
  replicas:
  [`FrayIrisClient.submit`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/fray/src/fray/iris_backend.py#L550-L576).

## Prototype Findings

Closed prototype PR [#5351](https://github.com/marin-community/marin/pull/5351)
validated the shape locally with `fray.LocalClient`.

- eval client -> OpenAI proxy -> broker actor -> worker actor ->
  deterministic OpenAI-compatible server -> client response worked locally.
- The proxy should expose `/v1/completions` and `/v1/chat/completions`.
  Workers should receive an engine API root ending in `/v1`.
- Broker submit must be idempotent for identical `(request_id, envelope)` and
  reject conflicting reuse of a request id.
- The broker should keep the first valid current-lease terminal result and
  ignore stale leases.
- The first proxy was single-threaded to preserve Iris context. That is enough
  for the MVP, but not throughput evidence.
- CodeQL flagged echoing arbitrary request-id headers. The proxy should
  validate request ids before reflecting them.

## Pressure-Test Findings

The 2026-05 pressure test used existing Marin `VllmEnvironment`, lm-eval on
Iris, and a manual runner because the service implementation does not exist
yet.

- PR #5712 is the implementation base for new pressure-test work. It moves the
  serving lane to `vllm-tpu==0.19.0`, `tpu-inference==0.19.0`, and
  `libtpu==0.0.39`.
- The explicit 1e22 MoE artifact found in project artifacts is
  `gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/checkpoints/step-77725/`.
  It is Orbax/OCDBT-shaped and not directly loadable by current vLLM.
- The closest vLLM-loadable 1e22 stand-in found in repo-configured eval suites
  is
  `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234/`.
  A patched MMLU submission against it was accepted on `v5p-8`, then remained
  pending on capacity and was stopped before runtime evidence.
- Issue #5672 tracks the Delphi/1e22 RPA scoped-VMEM startup failure. Treat it
  as target-model follow-up work, not as a blocker for the standard-model
  service path.
- Small-model HumanEval on v6e-4 succeeded through
  `local-chat-completions` and `/v1/chat/completions`, with saved lm-eval
  result and sample artifacts.
- The saved HumanEval config still reported `num_fewshot: 0` and `n-shot: 0`
  even when the runner passed
  `EvalTaskConfig("humaneval", 5, task_alias="humaneval_5shot")`. This proves
  the generation endpoint, not exact 5-shot semantics.
- Small-model MMLU reached `local-completions` and `/v1/completions`, but TPU
  vLLM 0.18.0 returned HTTP 500 on prompt-logprob scoring. A direct
  non-lm-eval request reproduced the failure with `echo=true`, `logprobs=1`,
  and `max_tokens=1`; server logs raised `KeyError: 425` in
  `_create_completion_logprobs`.
- lm-eval and the Iris-installed vLLM package were skewed:
  `lm_eval.models.vllm_causallms` imports `vllm.utils.get_open_port`, which
  was missing in that installed vLLM package. The manual runner used a
  scratch-only shim. Production code should fix dependency versions or avoid
  that import path.
- The standard-model runner now defaults to `standard_humaneval_smoke` with
  `Qwen/Qwen3-0.6B`, matching the MVP proof: model readiness, OpenAI
  chat-completions plumbing, result upload, and cleanup.

## Conclusion

`RunningModel` plus `OpenAIEndpoint` is the right evaluator boundary.
Generation over `/v1/chat/completions` is viable enough for the MVP. Prompt
logprob scoring over `/v1/completions` and target-model readiness remain
follow-up compatibility gates.
