# RFC 2: Levanter HTTP Parity For Evals

This doc is intentionally narrow. It covers the changes needed in Levanter's OpenAI-compatible HTTP layer so that Levanter can satisfy [`01-rfc-evals-over-openai-http.md`](./01-rfc-evals-over-openai-http.md).

It does not cover the Iris proxy / broker / worker system. That lives in [`03-rfc-iris-inference-service.md`](./03-rfc-iris-inference-service.md).

## Raison d'être

Levanter already has an OpenAI-compatible HTTP server. The point of this RFC is to close whatever gap remains between:
- what our eval clients need
- what Levanter's HTTP layer currently does

Once that gap is closed, `levanter_lm_evaluation_harness` should disappear and Levanter should run through the same eval path as vLLM.

## Goals

- Use [`levanter.inference.openai.InferenceServer`](../../../lib/levanter/src/levanter/inference/openai.py) as the single Levanter serving surface for evals.
- Bring that server up to the parity needed by [`01-rfc-evals-over-openai-http.md`](./01-rfc-evals-over-openai-http.md).
- Delete [`levanter_lm_eval_evaluator.py`](../../../lib/marin/src/marin/evaluation/evaluators/levanter_lm_eval_evaluator.py) once the shared HTTP path works.
- Keep the scope tight: implement the HTTP behavior current evals need.

## Current state

- Levanter already exposes:
  - [`/v1/chat/completions`](../../../lib/levanter/src/levanter/inference/openai.py#L740)
  - [`/v1/completions`](../../../lib/levanter/src/levanter/inference/openai.py#L744)
  - [`/v1/tokens`](../../../lib/levanter/src/levanter/inference/openai.py#L748)
- [`CompletionRequest`](../../../lib/levanter/src/levanter/inference/openai.py#L88) already includes fields such as:
  - `echo`
  - `logprobs`
  - `max_tokens`
  - `stop`
  - `temperature`
- [`_create_completion()`](../../../lib/levanter/src/levanter/inference/openai.py#L515) currently formats logprobs for generated tokens.
- The direct Levanter evaluator still bypasses HTTP entirely:
  - it constructs `TrainerConfig(..., ray=RayConfig(auto_start_cluster=False))` in [`levanter_lm_eval_evaluator.py`](../../../lib/marin/src/marin/evaluation/evaluators/levanter_lm_eval_evaluator.py#L72)
  - it builds an `EvalHarnessMainConfig` and runs Levanter's eval harness directly in [`levanter_lm_eval_evaluator.py`](../../../lib/marin/src/marin/evaluation/evaluators/levanter_lm_eval_evaluator.py#L93)

## Likely gap

Levanter already exposes `/v1/completions` and `/v1/chat/completions`, but `/v1/completions` does not yet implement the echoed prompt-logprob response shape that `lm_evaluation_harness` needs for loglikelihood scoring.

More concretely:
- `lm_eval local-completions` sends `echo=true` and `logprobs=1` for scoring requests.
- It expects `choices[].logprobs.token_logprobs` and `choices[].logprobs.top_logprobs` in the echoed response.
- Levanter's current completion handler only formats generated-token logprobs and sets `top_logprobs=None`.

So the main work in this RFC is not “make Levanter speak OpenAI better” in the abstract. It is: make `/v1/completions` satisfy the `lm_eval` scoring contract.

## Capability checklist

- Required now:
  - `/v1/completions` scoring parity for `lm_eval`
  - `/v1/chat/completions` generation parity
  - Evalchemy on the same lm-eval OpenAI path
  - Harbor using the same `api_base`
- Not required by current eval code:
  - `/v1/tokens`
  - tokenizer HTTP endpoints
  - prompt-side scoring on chat completions
  - chat batching
  - Responses API / streaming / tools / full OpenAI parity

<details>
<summary>Required now</summary>

- `/v1/completions` accepts:
  - `prompt`
  - `model`
  - `max_tokens`
  - `temperature`
  - `stop`
  - `seed`
  - `logprobs`
  - `echo`
- `/v1/completions` returns:
  - `choices[].text`
  - `choices[].index`
  - `choices[].logprobs.token_logprobs`
  - `choices[].logprobs.top_logprobs`
- Prompt-side scoring works over `/v1/completions`.
- `/v1/chat/completions` accepts generation requests and returns `choices[].message.content`.
- Evalchemy uses the same lm-eval OpenAI path rather than a separate Levanter-specific adapter.
- Harbor can target the same `api_base`; any required `model_info` stays in Harbor-side config.

</details>

<details>
<summary>Not required by current eval code</summary>

- `/v1/tokens`
- Tokenizer HTTP endpoints in general
- Prompt-side scoring on `/v1/chat/completions`
- Chat batching
- OpenAI Responses API
- Streaming
- Tools / tool calls
- Full OpenAI parity

</details>

## Future work

- The more likely follow-up is to keep pushing on vLLM so both GPU and TPU paths support our custom JAX architectures and the eval features we need, such as prompt scoring.
