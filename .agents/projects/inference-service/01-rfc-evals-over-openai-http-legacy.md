# Legacy RFC 1: Evals Over OpenAI-Compatible HTTP

> [!NOTE]
> Superseded by [`01-rfc-served-model-lm-eval.md`](./01-rfc-served-model-lm-eval.md).
> This document is kept as historical context for the broader eval migration attempt and closed draft PR [#4841](https://github.com/marin-community/marin/pull/4841).

This doc only covers the eval-side contract.

Engine-specific HTTP parity lives in [`02-rfc-levanter-http-parity.md`](./02-rfc-levanter-http-parity.md). Iris deployment and queueing live in [`03-rfc-iris-inference-service.md`](./03-rfc-iris-inference-service.md).

## Raison d'être

Allow our current eval entrypoints to run against an OpenAI-compatible HTTP server, so getting off Ray is mostly a question of:
- which eval clients we support
- what HTTP surface they require
- how we decouple the evaluator from the inference engine

## Goals

### The system

- Support the `run.py` evaluators we care about for Ray migration:
  - `lm_evaluation_harness`
  - `harbor`
  - `evalchemy`
- Decouple the evaluator from the inference engine. vLLM vs Levanter should be deployment config, not evaluator identity.
- Run the same eval flow against either vLLM or Levanter, as long as the backend exposes the required OpenAI-compatible HTTP surface.
- Delete `levanter_lm_evaluation_harness` once Levanter satisfies that same surface.

### Non-goals

- How the Iris inference service is deployed.
- How Levanter implements any missing HTTP behavior internally.
- Direct Levanter analysis jobs such as [`log_probs.py`](../../../lib/marin/src/marin/evaluation/log_probs.py), [`save_logprobs.py`](../../../lib/marin/src/marin/evaluation/save_logprobs.py), and [`visualize.py`](../../../lib/marin/src/marin/evaluation/visualize.py). Those are tracked separately in [#4640](https://github.com/marin-community/marin/issues/4640#issuecomment-4256093134).

## Current state

- [`run.py`](../../../lib/marin/src/marin/evaluation/run.py#L35) currently exposes:
  - `lm_evaluation_harness`
  - `levanter_lm_evaluation_harness`
  - `evalchemy`
  - `harbor`
- [`lm_evaluation_harness_evaluator.py`](../../../lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py#L188) already talks to a local OpenAI-compatible server via `local-completions` or `local-chat-completions`, and passes:
  - `model`
  - `base_url`
  - tokenizer config
  - extra engine args
- [`harbor_evaluator.py`](../../../lib/marin/src/marin/evaluation/evaluators/harbor_evaluator.py#L194) already uses `api_base`, but also needs Harbor-specific config such as:
  - dataset
  - version
  - agent
  - concurrency
  - optional `model_info`
- [`evalchemy_evaluator.py`](../../../lib/marin/src/marin/evaluation/evaluators/evalchemy_evaluator.py#L928) still shells out with `--model vllm`, so it is not on the shared HTTP path yet.
- [`levanter_lm_eval_evaluator.py`](../../../lib/marin/src/marin/evaluation/evaluators/levanter_lm_eval_evaluator.py#L72) still loads Levanter directly and uses `TrainerConfig(..., ray=RayConfig(auto_start_cluster=False))`, so it also bypasses the shared HTTP path.

## Proposed direction

- The key move in this RFC is to decouple the eval from the deployed model server. The evaluator should target an OpenAI-compatible contract. Which engine happens to implement that contract is deployment config.
- `lm_evaluation_harness` remains the shared path for lm-eval style workloads.
- `harbor` remains its own evaluator, but should point at the same OpenAI-compatible backend.
- `evalchemy` should stop using its direct vLLM backend and instead target the same lm-eval OpenAI-compatible model path.
- `levanter_lm_evaluation_harness` should disappear. Levanter should run through the same eval path as vLLM.

This RFC should define the eval-side contract, not the engine-specific or service-specific plumbing.

## Required HTTP surface

We do not need general OpenAI API parity. We need the subset required by the evals above.[^openai-responses]

At minimum:
- `/v1/completions`
  - still needed for `lm_eval` scoring / loglikelihood-style tasks
  - request side: `prompt`, `model`, `max_tokens`, `temperature`, `stop`, `seed`
  - scoring requests also require `echo=True` and `logprobs=1`
  - response side must include:
    - `choices[*].text`
    - `choices[*].index`
    - `choices[*].logprobs.token_logprobs`
    - `choices[*].logprobs.top_logprobs`
- `/v1/chat/completions`
  - needed for chat-template / conversation-style generation flows
  - request side: `messages`, `model`, generation params, and `seed`
  - response side must include:
    - `choices[*].message.content`
    - `choices[*].index`
- Stable model id / model name wiring.
- Tokenizer wiring where the evaluator library needs it.
- Prompt-side scoring on the completions surface.
- Generated-token logprobs on the completions surface.

<details>
<summary>Details: why <code>/v1/completions</code> is still required</summary>

`lm-evaluation-harness` uses `local-completions` for scoring tasks. Its scoring payload sets `echo=True` and `logprobs=1`, and its parser reads both `token_logprobs` and `top_logprobs` from the echoed response.[^lm-eval-local-completions] That means the required contract is stricter than "return generated text and some logprobs." The backend has to support echoed prompt scoring on the completions endpoint.

`local-chat-completions` is generation-only. It explicitly does not support `loglikelihood`, so it is not a substitute for the completions path on scoring tasks.[^lm-eval-local-chat]

</details>

<details>
<summary>Details: tokenizer expectations</summary>

Current Marin usage does not require lm-eval's remote tokenizer endpoints. Marin currently passes `tokenizer_backend=huggingface` and `tokenized_requests=False`, and when needed it stages or injects a Hugging Face tokenizer path/id itself.[^marin-lm-eval-wrapper] So the current contract is "the evaluator has access to a tokenizer", not "the server must implement `/tokenizer_info`, `/tokenize`, and `/detokenize`." Those remote-tokenizer endpoints exist in lm-eval, but they are not part of the MVP contract we need today.[^lm-eval-remote-tokenizer]

</details>

### Per-evaluator view

`lm_evaluation_harness`
- Uses `local-completions` when `apply_chat_template=False`
- Uses `local-chat-completions` when `apply_chat_template=True`
- Needs more than just a URL: model id, tokenizer handling, and optional extra args.
- For scoring tasks, it needs completions with echoed prompt logprobs and top logprobs.

`harbor`
- Needs an OpenAI-compatible `api_base`
- Also needs Harbor-specific config which is not inference-server-specific:
  - dataset
  - version
  - agent
  - concurrency
  - optional `model_info`

`evalchemy`
- Today it is wired to `--model vllm`.
- It already initializes models through the lm-eval registry, and its README already documents `openai-chat-completions` support.[^evalchemy-readme]
- The cleanest migration is therefore to stop hardcoding `--model vllm` and instead pass an lm-eval OpenAI-compatible model name plus the right `model_args`.[^evalchemy-init][^marin-evalchemy-wrapper]
- No Evalchemy-specific OpenAI adapter looks necessary.

<details>
<summary>Details: what the library source says</summary>

`lm_evaluation_harness`

- `LocalCompletionsAPI` builds scoring requests with `echo=True` and `logprobs=1`, and its parser expects both `token_logprobs` and `top_logprobs` in the response.[^lm-eval-local-completions]
- `LocalChatCompletion` forces batch size 1 for chat completions and explicitly disables `loglikelihood`.[^lm-eval-local-chat]

`evalchemy`

- `initialize_model()` delegates model construction to `lm_eval.api.registry.get_model(...).create_from_arg_string(...)`, so it can already reuse lm-eval's API-backed model classes.[^evalchemy-init]
- Evalchemy task code already special-cases `OpenAIChatCompletion` / `OpenAICompletionsAPI`, which is a strong sign that the path already exists.[^evalchemy-openai-task]

`harbor`

- Harbor's generic LiteLLM wrapper takes `model_name`, optional `api_base`, optional `model_info`, and an optional `use_responses_api` flag.[^harbor-litellm]
- The hosted-vLLM path validates `model_info` and requires token-limit / cost metadata.[^harbor-hosted-vllm]
- So the right summary is: Harbor is mostly backend-agnostic behind `api_base`, but some agent paths still need model metadata.

</details>

## Client config boundary

The job that launches the eval may know a lot:
- which eval to run
- which model to serve
- whether the workers are running vLLM or Levanter

The evaluator should not know all of that. It should just get the HTTP endpoint and the extra fields its library needs.

One reasonable shape is:

```python
@dataclass(frozen=True)
class OpenAIEndpointConfig:
    url: str
    model: str


@dataclass(frozen=True)
class LmEvalClientConfig:
    endpoint: OpenAIEndpointConfig
    tokenizer: str | None = None
    extra_model_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarborClientConfig:
    api_base: str
    model: str
    dataset: str
    version: str
    agent: str
    n_concurrent: int
    model_info: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvalchemyClientConfig:
    endpoint: OpenAIEndpointConfig
    tokenizer: str | None = None
    extra_model_args: tuple[str, ...] = ()
```

Here `url` means the exact endpoint URL, e.g. `.../v1/completions` or `.../v1/chat/completions`.

The important bit is:
- the evaluator should not decide whether the backend is vLLM or Levanter
- the evaluator should get the client config it needs
- once the client is configured, requests should just go to the OpenAI-compatible endpoint

## Out of scope (initially)

- OpenAI Responses API
- OpenAI Batch API
- OpenAI streaming
- Full OpenAI parity beyond what the current evals actually require

## Future work

- Once this contract is stable, other offline inference clients can target it too.
- If we later want an always-on inference system, this doc can still stay the contract doc while the serving system changes underneath it.

[^openai-responses]: OpenAI recommends moving new work from Chat Completions to Responses, but our current eval libraries still target the Completions / Chat Completions surface. Source: OpenAI, “Migrate to the Responses API”, https://developers.openai.com/api/docs/guides/migrate-to-responses
[^marin-lm-eval-wrapper]: Marin's current lm-eval wrapper passes `local-completions` / `local-chat-completions`, `base_url`, `tokenizer_backend=huggingface`, `tokenized_requests=False`, and an explicit tokenizer when needed in [lm_evaluation_harness_evaluator.py](../../../lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py#L188).
[^marin-evalchemy-wrapper]: Marin's current Evalchemy wrapper still shells out with `--model vllm` in [evalchemy_evaluator.py](../../../lib/marin/src/marin/evaluation/evaluators/evalchemy_evaluator.py#L928).
[^lm-eval-local-completions]: EleutherAI, `lm_eval/models/openai_completions.py` at commit `d5e3391f22cde186c827674d5c3ec7c5f4fe0cab`, https://github.com/EleutherAI/lm-evaluation-harness/blob/d5e3391f22cde186c827674d5c3ec7c5f4fe0cab/lm_eval/models/openai_completions.py#L15-L122
[^lm-eval-local-chat]: EleutherAI, `lm_eval/models/openai_completions.py` at commit `d5e3391f22cde186c827674d5c3ec7c5f4fe0cab`, https://github.com/EleutherAI/lm-evaluation-harness/blob/d5e3391f22cde186c827674d5c3ec7c5f4fe0cab/lm_eval/models/openai_completions.py#L141-L241
[^lm-eval-remote-tokenizer]: EleutherAI, `lm_eval/utils.py` remote tokenizer support at commit `d5e3391f22cde186c827674d5c3ec7c5f4fe0cab`, https://github.com/EleutherAI/lm-evaluation-harness/blob/d5e3391f22cde186c827674d5c3ec7c5f4fe0cab/lm_eval/utils.py#L702-L890
[^evalchemy-readme]: Evalchemy README, API-backed model support examples at commit `010412ccda0de14491f78119c05a3045f6ab6c33`, https://github.com/mlfoundations/evalchemy/blob/010412ccda0de14491f78119c05a3045f6ab6c33/README.md#L16-L48
[^evalchemy-init]: Evalchemy `initialize_model()` delegates model creation through lm-eval's registry at commit `010412ccda0de14491f78119c05a3045f6ab6c33`, https://github.com/mlfoundations/evalchemy/blob/010412ccda0de14491f78119c05a3045f6ab6c33/eval/eval.py#L466-L510
[^evalchemy-openai-task]: Evalchemy task code already special-cases OpenAI-backed lm-eval model classes at commit `010412ccda0de14491f78119c05a3045f6ab6c33`, https://github.com/mlfoundations/evalchemy/blob/010412ccda0de14491f78119c05a3045f6ab6c33/eval/task.py#L26-L59
[^harbor-litellm]: Harbor LiteLLM wrapper constructor at commit `1ae29a3390b6e60480acc27a62bfbfb4ff370e03`, https://github.com/harbor-framework/harbor/blob/1ae29a3390b6e60480acc27a62bfbfb4ff370e03/src/harbor/llms/lite_llm.py#L61-L132
[^harbor-hosted-vllm]: Harbor hosted-vLLM model validation at commit `1ae29a3390b6e60480acc27a62bfbfb4ff370e03`, https://github.com/harbor-framework/harbor/blob/1ae29a3390b6e60480acc27a62bfbfb4ff370e03/src/harbor/llms/utils.py#L75-L145
