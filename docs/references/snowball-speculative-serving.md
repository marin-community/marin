# Snowball speculative serving

`experiments.evaluation.snowball_speculative` serves the frozen Snowball 67B-A2B
step-1903 target in three modes. It uses Marin's evaluation configuration and
direct Iris serving path, with OpenAI streaming and Prometheus metrics.

| Mode | Draft | Verification |
| --- | --- | --- |
| `baseline` | None | Ordinary target decoding |
| `eagle3` | EAGLE3, depth 3 | Native standard rejection sampling |
| `synthetic` | EAGLE3, depth 3 | Native synthetic rejection sampling; explicit mean required |

The target is
`s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/`.
The draft is `laion/snowball-64k-eagle3-draft-r2egym` at
`4bdb47c08e5b5190bea3c7a93c3e14470230e469`. Marin resolves that immutable HF
snapshot through its regional model cache. The tokenizer is
`marin-community/marin-tokenizer` at `a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`;
its `tokenizer.json` matches the target export byte for byte (SHA256
`881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2`).
The served model alias is `snowball-67b-a2b-step1903`.

## Launch on Iris

Read [Iris operations](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md#submitting-a-gpu-gang-from-a-workstation)
and check shared capacity before submitting. Submit through the hub to
`cw-us-east-02a`, whose H100 backend uses the `h100-8x` pool. Each command below
requests one node of eight H100s. The recipe creates no child jobs. It uses
TP1, DP8 and expert parallelism, BF16, 65,536 total context tokens, a 7,168-token
prefill budget and 32 concurrent sequence slots. Compilation and CUDA graphs
retain the engine defaults.

Choose a unique job name and endpoint prefix. This example serves the baseline:

```bash
uv run iris --cluster marin job run --no-wait \
  --target-cluster cw-us-east-02a --priority interactive \
  --enable-extra-resources --sync-package marin-core \
  --gpu H100x8 --cpu 16 --memory 512GB --disk 200GB \
  --timeout 21600 --job-name snowball-baseline \
  -- python -m experiments.evaluation.snowball_speculative \
  --mode baseline --endpoint-prefix /serve/my-snowball --timeout-hours 1
```

Use `--mode eagle3` for verified speculation. For synthetic mode, use
`--mode synthetic --acceptance-length "$CODING_ACCEPTANCE_LENGTH"`. Supply a
separately measured coding-prompt acceptance value for this target, draft,
depth and thinking mode. The allowed interval is `[1, 4]`: mean acceptance
length includes one verifier token. The recipe forwards the value unchanged as
vLLM's `synthetic_acceptance_length` with `rejection_sample_method=synthetic`.
It does not implement an acceptance sampler or calibrate a value.

Obtain `CODING_ACCEPTANCE_LENGTH` from a separate standard `eagle3` run on coding
prompts (for AgentX comparisons, its specified SPEED-Bench coding calibration).
Use the counter formula below. Thinking mode means the chat-template/request
setting that enables or suppresses reasoning. This recipe uses the pinned
tokenizer template without request overrides; if a consumer changes
`chat_template_kwargs`, its calibration must match that setting.

The persistent service stops after its requested lifetime; the outer Iris
timeout also includes queue and startup time. Preserve needed evidence and
cancel the exact owned job when finished.

After submission, `uv run iris --cluster marin job describe /USER/snowball-baseline`
shows its task and state. Check readiness with
`uv run iris --cluster marin endpoints list /serve/my-snowball-baseline --exact`.
Startup logs are available with `uv run iris --cluster marin job logs /USER/snowball-baseline`.
To release it, run `uv run iris --cluster marin job cancel /USER/snowball-baseline`.
Replace `USER` with the owner printed by the submit command.

## Consumer interface

The registered endpoint is `<endpoint-prefix>-<mode>`. Obtain an access URL with
`uv run iris --cluster marin endpoints mint /serve/my-snowball-baseline --ttl-hours 1`;
treat that URL as a credential. The controller's authenticated
proxy route replaces endpoint slashes with periods: `/serve/my-snowball-baseline`
becomes `/proxy/serve.my-snowball-baseline`. A client needs hub authentication or an
endpoint-scoped capability; a worker's private address is not a public endpoint.

The minted URL is the complete endpoint root, including routing and access.
Use it as printed; do not append another proxy path. For example:

```bash
BENCHMARK_ROOT='<complete URL printed by endpoints mint, without a trailing slash>'
curl --fail "$BENCHMARK_ROOT/v1/models"
```

Relative to that endpoint root:

- `POST /v1/chat/completions` accepts `model: snowball-67b-a2b-step1903`,
  `messages`, sampling settings and `stream: true` (SSE, ending in `[DONE]`).
- `GET /v1/models` reports the served alias and context length.
- `POST /tokenize` exposes the engine's tokenizer.
- `GET /metrics` exposes the existing vLLM counters for the harness's
  server-scrape path. Use an exclusive serving allocation for run-scoped deltas.

The public consumer interface was assessed at AgentX harness revision
[`e4bb2f835ffedbbe6b76252ea5c93caec68c5faf`](https://github.com/SemiAnalysisAI/agentx-harness/tree/e4bb2f835ffedbbe6b76252ea5c93caec68c5faf).
Its [metrics guide](https://github.com/SemiAnalysisAI/agentx-harness/blob/e4bb2f835ffedbbe6b76252ea5c93caec68c5faf/docs/tutorials/spec-decode-metrics.md)
supports aggregate server scraping without per-request telemetry. Select the
matching pinned tokenizer separately from the model alias. The
[quickstart](https://inferencex.semianalysis.com/agentx/quickstart) uses the
server root as `--url` and `/v1/chat/completions` as the endpoint. Point its
server metrics setting at `<root>/metrics` if auto-discovery cannot reach it.
This recipe does not run AgentX.

Use deltas of `vllm:spec_decode_num_drafts_total` (verification steps),
`vllm:spec_decode_num_draft_tokens_total` and
`vllm:spec_decode_num_accepted_tokens_total`. Sum matching engine labels before
computing `1 + accepted_draft_tokens / verification_steps`. With zero steps,
the mean is unavailable, not zero. Zero accepted drafts with positive steps
means a length of one. Draft acceptance rate is accepted drafts divided by
proposed drafts when that denominator is positive. Baseline has no speculative
measurement. Synthetic counters describe forced acceptance and must be kept
separate from measured standard acceptance and quality results.

The pinned harness's SPEED report reads the first series for each counter. With
DP8, that summary covers one engine. Its raw server scrape retains the series;
use all matching engine labels for a whole-server mean as Marin's metric window
does.

## Bounded serving check and provenance

Run a configuration-only check locally:

```bash
uv run python -m experiments.evaluation.snowball_speculative \
  --smoke-all --acceptance-length 2.5 --endpoint-prefix /serve/my-check --dry-run
```

On the same eight-GPU Iris command above, replace the final arguments with:

```bash
--smoke-all --acceptance-length 2.5 --endpoint-prefix /serve/my-check
```

Add `-e VLLM_SERVER_DEV_MODE 1` to the Iris job options for this check. The pinned
engine gates `/server_info` behind that diagnostic setting. The smoke runner
reads it on the engine's loopback listener to verify resolved sampler settings;
the Iris dashboard exposes only its ordinary serving routes. Persistent serves
use the ordinary engine defaults and do not need this setting.

Here `2.5` is an arbitrary control probe, with no calibration claim. All three
engines run sequentially in the same allocation. Each receives three short
coding requests, including streaming. Iris captures `IRIS_OUTPUT_DIR` with
requested and resolved settings, model metadata, HTTP responses, counter deltas,
raw metrics, runtime version, source/bundle identity and GPU/driver information.
The record keeps the wheel verifier's distribution version separately from
`/version`, whose source version can omit the wheel's CUDA packaging suffix.
The worker checks that baseline has no speculative config, verified mode drafts,
and synthetic mode resolves to the requested native sampler and rates. Find
the output archive with `iris attempt describe <task>:<attempt>`. For a single
task the ID is `/USER/JOB/0`; `iris task describe /USER/JOB/0` shows the current
attempt number, starting at zero. Preserve
startup logs, especially `MARIN_VLLM_WHEEL_VERIFIED`, alongside the archive;
that existing verifier identifies the actual installed build and GPU ABI.

The supported build at introduction is Marin vLLM source
`01911be34fac8715347962a8d791ca34a879e01e`, release
`marin-vllm-gpu-20260924-01911be34fac`, version
`0.0.0.dev20260924+marin.01911be34fac.cu132`. The recipe uses the repository's
promoted wheel and startup verifier. Retain the actual runtime identity on each
run because that promotion can change. Evaluation records also preserve the
requested rejection sampler and synthetic acceptance length.

## Context and benchmark limits

Both the target's `max_position_embeddings` and the draft's
`transformer_layer_config.max_position_embeddings` are 65,536. The tokenizer
advertises 131,072; that does not increase either model's limit. The draft also
uses 2,048-token sliding attention. Requests must fit prompt plus output within
65,536. No truncation or context extension is applied.

The pinned vLLM runner masks speculation when a sequence plus the proposal
budget exceeds the effective draft context. Near that boundary, ordinary
decoding can replace drafting; a request beyond the target context is rejected.
Keep this fallback distinct from rejected draft tokens. This readiness check
does not exercise long traces or establish a speedup or output quality.

Published AgentX traces include contexts beyond 64k, including its 256k
variant. A private consumer must define and disclose a workload that fits this
pair, preserve input/output lengths and replay settings, supply the separately
measured acceptance assumption, and handle context errors explicitly. See the
[methodology](https://inferencex.semianalysis.com/agentx/methodology) for warmup,
measurement windows and workload controls. This check establishes serving
readiness; stock-workload and submission conformance remain unverified.
