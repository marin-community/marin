# Served-Model LM Eval Spike Report

## Recommended minimal RFC stance

- Keep RFC 1 scoped to the launcher-neutral served-model boundary plus the real `lm_eval` adapter contract.
- Do not require served vLLM to support multiple-choice/loglikelihood scoring in the initial implementation. Existing Marin usage already keeps generation on vLLM and multiple-choice/loglikelihood on Levanter.
- Keep real vLLM validation manual and generation-focused for now. Treat the prompt-logprob failure as a documented follow-up risk, not an implementation blocker.
- Use the deterministic fake OpenAI-compatible server plus real `lm_eval` as the normal-CI contract test.

## What works

- Added the RFC 1 boundary in code:
  - `OpenAIEndpoint`, `RunningModel`, `ModelDeployment`, `ModelLauncher`, and `VllmModelLauncher`.
  - `LmEvalRun`, `build_lm_eval_model_args`, and `run_lm_eval`.
- `run_lm_eval` consumes only `RunningModel` plus `LmEvalRun`. It does not launch vLLM, stage artifacts, dispatch through the legacy evaluator registry, or clean up backend resources.
- Added lightweight OpenAI-compatible protocol request models in `lib/levanter/src/levanter/inference/openai_protocol.py`.
  - Levanter's runtime `openai.py` now imports and re-exports those models.
  - The deterministic test server validates incoming request bodies with those shared protocol models.
- Added a reusable deterministic fake HTTP server helper for the Marin eval OpenAI subset:
  - `/v1/completions` with `echo=True` and `logprobs`
  - `/v1/chat/completions` generation
- Added a real `lm_eval` test that points `local-completions` at the deterministic server and runs a one-document scoring task through real `simple_evaluate`.
- Added a manual smoke CLI for real vLLM plus real lm-eval:
  - `python -m marin.evaluation.served_lm_eval_vllm_smoke`
  - It is not a pytest test and is not part of normal CI.
  - It now defaults to native TPU mode, pins `vllm-tpu==0.13.2.post6`, stages tokenizer files from remote object-store model paths when `--tokenizer` is omitted, and prints lm-eval result/sample file summaries into job logs.
  - `VllmModelLauncher` now emits vLLM backend diagnostics if an exception occurs while the served model context is active.
- Extended the existing `marin.inference.vllm_smoke_test` manual CLI with `--regions` so generation-only vLLM smokes can target a compatible Iris TPU region explicitly.

## Fake-server contract findings

- The deterministic OpenAI-compatible server now lives in `tests/evals/openai_stub.py` and validates request bodies with the shared Levanter protocol schemas before returning responses.
- The focused tests exercise the server over HTTP through `/v1/completions` and `/v1/chat/completions`; there is no fake `lm_eval` module or `simple_evaluate` call-echo test.
- The contract intentionally covers only Marin's current RFC 1 subset: text-completion scoring with `echo=True` and `logprobs`, plus basic chat generation. It is not a complete OpenAI API conformance suite.

## Real lm-eval findings

- Confirmed: `local-completions` wants endpoint-specific `base_url`, not the `/v1` API root. The runner should derive `{root}/completions`.
- Confirmed: non-chat scoring needs a tokenizer for context length calculation. The real test used `tokenizer_backend=tiktoken` and `tokenized_requests=False`.
- Confirmed: `extra_model_args` must be able to override runner defaults such as `tokenizer_backend`; the runner now merges defaults first and then applies overrides.
- Falsified RFC prose detail: real `local-completions` scoring sends `max_tokens=1`, not `max_tokens=0`.
- Falsified RFC prose detail: real `local-completions` scoring did not send a `stop` field in the tested path.
- Confirmed: prompt-logprob responses must include an echoed prompt plus one generated token. lm-eval computes the score from `token_logprobs[ctxlen:-1]`, so it slices away context tokens and the final generated token.
- Confirmed result layout for `EvaluationTracker(output_path=<dir>)`: files are written under `<output_path>/<model_name_sanitized>/results_<date>.json` and `samples_<task>_<date>.jsonl`.

## Real vLLM status

- Local macOS vLLM import is not feasible with the repo's `vllm` extra because `nixl==0.3.0` has no macOS arm64 wheel/source.
- A manual smoke CLI has been added for Iris-capable infrastructure. It launches real vLLM through `VllmModelLauncher`, receives `RunningModel`, and calls `run_lm_eval`.
- The manual smoke should run through an Iris CPU parent job so the CLI's Fray submission uses an Iris-backed `current_client()`. A direct laptop invocation resolves to Fray's local backend and fails before reaching Iris with `FileNotFoundError: [Errno 2] No such file or directory: 'vllm'`.
- The stale quickstart checkpoint path from `experiments/evals/run_on_gpu.py` did not exist from this environment. The representative smoke was switched to an existing small model used by current eval experiments.
- Representative smoke workload:
  - model: `gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m`
  - eval: `arc_easy`, `num_fewshot=10`, `limit=4`
  - first target: GCP TPU/vLLM native, because the model artifact is available on GCS and small enough for a representative smoke.
- First Iris attempt `/romain/served-lm-eval-vllm-smoke-tpu-20260428-165214` failed before vLLM launch: the child Fray `JobRequest.name` included the raw model URI, and Iris rejected `/` in the job name. The manual smoke now sanitizes child job names and has a focused regression test for this case.
- Second Iris attempt `/romain/served-lm-eval-vllm-smoke-tpu-20260428-175455` failed before vLLM launch: the child requested `v6e-8` in inherited/targeted region `us-central2`, where the marin Iris config has no `v6e-8` scale group.
- Third Iris attempt `/romain/served-lm-eval-vllm-smoke-tpu-20260428-181011` used `v5p-8` with explicit `us-east5` constraints and successfully launched the child job. Native vLLM then failed before serving:
  - `ImportError: cannot import name 'get_open_port' from 'vllm.utils'`
  - this is a vLLM environment/package import failure, not an lm-eval contract failure. The most likely cause is version skew inside the Iris child vLLM environment: the smoke launcher was asking Iris to install unpinned `vllm-tpu`, while Marin's package extra pins `vllm-tpu==0.13.2.post6`. The smoke launcher now uses the pinned package.
- Fourth Iris attempt `/romain/served-lm-eval-vllm-smoke-tpu-docker-20260428-181546` used the same workload with `--mode docker` as an exploratory fallback after the native import failure. Docker is not a desired path for this spike and should not be pursued for RFC validation. It failed before serving:
  - `RuntimeError: Docker socket not available at /var/run/docker.sock`
  - normal Iris child tasks in this path do not expose docker-alongside-docker, but this blocker is not relevant to the native Iris/vLLM validation path.
- Fifth Iris attempt `/romain/served-lm-eval-vllm-smoke-native-pinned-20260428-182512` used pinned native `vllm-tpu==0.13.2.post6` and got past the prior import failure:
  - vLLM started and `/v1/models` became ready.
  - lm-eval then failed before sending scoring requests because the smoke CLI passed the raw `gs://.../llama-200m` URI as the Hugging Face tokenizer. The CLI now preserves `--tokenizer=None` so remote checkpoint tokenizer files are staged locally.
- Sixth Iris attempt `/romain/served-lm-eval-vllm-smoke-native-tokenizer-20260428-183110` staged tokenizer files and reached real lm-eval scoring:
  - vLLM started, `/v1/models` returned 200, lm-eval loaded `arc_easy`, built 4 contexts with 10 fewshot examples, and began 16 loglikelihood requests.
  - the first `POST /v1/completions` returned HTTP 500 with response body `{"error":{"message":"14924","type":"Internal Server Error","param":null,"code":500}}`.
- Seventh Iris attempt `/romain/served-lm-eval-vllm-smoke-native-diag-20260428-183642` was only a diagnostic rerun and failed during child environment build due to a transient GitHub 502 downloading `kitoken==0.10.2`.
- Eighth Iris attempt `/romain/served-lm-eval-vllm-smoke-native-diag2-20260428-183801` reproduced the same HTTP 500 with vLLM diagnostics:
  - vLLM API server version: `0.13.2.post6`
  - non-default args included `load_format='runai_streamer'`, `max_model_len=2048`, and the GCS model path.
  - RunAI streamer loaded `model.safetensors` from GCS, vLLM initialized `LlamaForCausalLM`, and `/v1/models` returned 200.
  - vLLM listed `/v1/completions` as an available POST route, then logged `POST /v1/completions HTTP/1.1" 500 Internal Server Error`.
- Existing Marin eval wiring uses vLLM for generation tasks and Levanter for multiple-choice/loglikelihood tasks. That means the failed `arc_easy` prompt-logprob smoke is exercising a less-paved path, not the normal vLLM generation path.
- Generation-only Iris smoke through the existing vLLM CLI succeeded for `/v1/chat/completions`:
  - parent job: `/romain/served-lm-eval-vllm-chat-parent-20260428-1855b`
  - child job: `/romain/served-lm-eval-vllm-chat-parent-20260428-1855b/vllm-smoke:v5p-8`
  - command shape: CPU-only Iris parent job, native vLLM child on `v5p-8`, `--regions us-east5`, `--load-format runai_streamer`, `--max-model-len 2048`
  - both parent and child finished `JOB_STATE_SUCCEEDED`
- Generation-only Iris smoke through `/v1/completions` also succeeded with the same model and serving configuration:
  - parent job: `/romain/served-lm-eval-vllm-completions-parent-20260428-1858`
  - child job: `/romain/served-lm-eval-vllm-completions-parent-20260428-1858/vllm-smoke:v5p-8`
  - both parent and child finished `JOB_STATE_SUCCEEDED`
- Iris reported zero captured log lines for the successful generation jobs, so the generated text was not preserved in the logs. The smoke script still raises on HTTP errors and on missing response fields, so the successes prove the served routes returned parseable OpenAI-compatible generation responses.
- Real vLLM serving is now proven for native TPU generation through both `/v1/chat/completions` and `/v1/completions`. Real vLLM prompt-logprob scoring for lm-eval is still not proven and currently fails on the representative native TPU smoke.

## What remains unproven

- Real vLLM prompt-logprob scoring for lm-eval is not proven; the native TPU smoke currently fails with HTTP 500 on the first `/v1/completions` loglikelihood request.
- Direct vLLM + `run_lm_eval` reaches vLLM readiness and real lm-eval task construction on Iris TPU, but does not complete an eval result.
- Successful generation smokes prove one request each for chat and text completions, not throughput, batching, long prompts, or production generation task quality.
- Remote checkpoint tokenizer staging is implemented only in the manual smoke CLI. The library runner still expects `RunningModel.tokenizer` to already be a usable tokenizer name/path.
- Batched scoring prompts, concurrent API requests, retries, and timeout behavior are not covered by the fake server test.
- Iris compatibility for RFC 1 remains partial: manual Iris parent/child jobs work for generation, but an Iris launcher returning `RunningModel` has not been implemented, and proxy auth, request ids, retries, and lifecycle wiring are not exercised.

## Biggest RFC holes, ordered by severity

1. The lm-eval scoring request shape is wrong/underspecified.
   RFC 1 should say real `local-completions` scoring sends `echo=True`, `logprobs=1`, `max_tokens=1`, and may omit `stop`. It should also specify the echoed prompt plus generated-token logprob shape that lm-eval slices with `ctxlen:-1`.

2. Tokenizer ownership is unclear.
   `local-completions` requires a tokenizer even when requests are sent as strings. The smoke CLI can stage tokenizer files from remote checkpoints, but the RFC should say whether that staging belongs in launchers, `run_lm_eval`, or pipeline orchestration.

3. Real vLLM prompt-logprob support is not established.
   Native TPU vLLM can serve the representative model and complete generation through `/v1/chat/completions` and `/v1/completions`, but the representative lm-eval scoring request fails with HTTP 500. RFC 1 should not assume OpenAI generation compatibility implies lm-eval scoring compatibility.

4. Result layout is not defined.
   The real tracker writes under `<output_path>/<model_name_sanitized>/`. RFC 1 should define whether `LmEvalRun.output_path` is a directory, a JSON file path, or a per-task path convention.

5. Shared conformance tests need a home.
   The protocol models can be shared now, but the RFC should define where reusable conformance assertions live so Levanter, vLLM, and Iris can run the same contract tests.

6. Iris compatibility is too implicit.
   RFC 1 should include an Iris launcher sketch that returns the exact `RunningModel` shape and states how auth, model id, retry policy, local proxy URL, request identity, region constraints, CPU parent jobs, accelerator child jobs, and runtime packaging interact with `run_lm_eval`.

7. `extra_model_args` serialization is fragile.
   lm-eval model args remain comma-delimited strings. The RFC should restrict values to simple scalars or define an escaping/encoding rule.

8. vLLM runtime package ownership is unspecified.
   The manual smoke exposed that the child runtime can drift from Marin's declared extras unless the Iris environment pins the same package set. RFC 1 should state who owns vLLM package versions for launcher jobs and how smoke jobs prove the installed runtime.

## Recommended minimal RFC edits

- Add an explicit scope line:
  - initial RFC 1 implementation covers the `RunningModel`/`run_lm_eval` boundary and the real `lm_eval` adapter contract
  - served vLLM prompt-logprob scoring for MCQ/loglikelihood is deferred
  - existing Levanter paths remain the supported path for MCQ/loglikelihood until a follow-up RFC or implementation validates served scoring
- Replace the `/v1/completions` scoring prose with the real lm-eval behavior:
  - required request fields for the tested scoring path: `model`, `prompt`, `temperature`, `max_tokens=1`, `logprobs=1`, `seed`, `echo=True`
  - `stop` is optional, not required for scoring
  - response must include `choices[*].logprobs.token_logprobs` and `top_logprobs` covering echoed prompt tokens and the generated token
- Add an adapter-args subsection:
  - `OpenAIEndpoint.base_url` is the API root, for example `http://host:port/v1`
  - `local-completions` receives `base_url={root}/completions`
  - `local-chat-completions` receives `base_url={root}/chat/completions`
  - defaults are `tokenizer_backend=huggingface` and `tokenized_requests=False`
  - `extra_model_args` may override defaults
- Define tokenizer requirements before implementation PRs migrate call sites.
- Define output layout and whether it should match lm-eval `EvaluationTracker` directly.
- Move protocol request schemas into a lightweight shared module and keep deterministic contract tests backend-independent.
- Add the manual vLLM generation smoke command to implementation notes, but keep real vLLM excluded from normal CI.
- Add explicit Iris smoke notes:
  - generation serving should be validated separately from prompt-logprob scoring
  - the manual command should run under a CPU-only Iris parent job, not directly from a laptop-local Fray backend
  - child job names must not include raw model URIs
  - child accelerator region must be explicit when the parent is CPU-only
  - native TPU vLLM package versions must be pinned and logged before treating the smoke as a contract result
  - Docker sidecar validation is out of scope for this RFC spike unless production explicitly chooses that runtime shape
- Move these to deferred follow-up work, not the minimal RFC 1 acceptance criteria:
  - representative served scoring validation with a production-style MCQ task such as `arc_easy` 10-shot
  - a single-request prompt-logprob probe that logs exact request shape, prompt token length, vLLM response body, and backend diagnostics
  - remote checkpoint tokenizer staging ownership beyond the manual smoke CLI
