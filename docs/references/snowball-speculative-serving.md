# Snowball speculative serving

`experiments.evaluation.snowball_speculative` serves Snowball 67B-A2B step1903
through Marin's direct Iris endpoint, with OpenAI streaming and Prometheus metrics.
It loads the existing model catalog and overrides the tokenizer, alias, host
resources and speculative settings for this frozen pair.

| Mode arguments | Behavior |
| --- | --- |
| `--mode baseline` | Ordinary target decoding |
| `--mode eagle3` | EAGLE3, depth 3, native standard rejection sampling |
| `--mode synthetic --acceptance-length "$CODING_ACCEPTANCE_LENGTH"` | EAGLE3, depth 3, native synthetic rejection sampling |

The fixed identities are:

- Target: `s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/`.
- Draft: `laion/snowball-64k-eagle3-draft-r2egym` at
  `4bdb47c08e5b5190bea3c7a93c3e14470230e469`, resolved through Marin's regional HF cache.
- Tokenizer: `marin-community/marin-tokenizer` at
  `a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`. Its `tokenizer.json` matches the target export
  (SHA256 `881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2`).
- Served alias: `snowball-67b-a2b-step1903`.

## Launch and access

Follow [Iris operations](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md#submitting-a-gpu-gang-from-a-workstation)
for capacity checks, job inspection, logs and cancellation. Submit through the
hub to `cw-us-east-02a`, pool `h100-8x`, with a unique job name and endpoint prefix:

```bash
uv run iris --cluster marin job run --no-wait \
  --target-cluster cw-us-east-02a --priority interactive \
  --enable-extra-resources --sync-package marin-core \
  --gpu H100x8 --cpu 16 --memory 512GB --disk 200GB \
  --timeout 21600 --job-name snowball-baseline \
  -- python -m experiments.evaluation.snowball_speculative \
  --mode baseline --endpoint-prefix /serve/my-snowball --timeout-hours 1
```

Select the other modes with the arguments in the table. Each launch uses one
eight-H100 node, TP1/DP8 with expert parallelism, BF16, a 7,168-token prefill
budget and 32 sequence slots. Compilation and CUDA graphs keep their ordinary
defaults. The service stops after its requested lifetime; the outer job timeout
also covers queue and startup time. Cancel the owned job when finished.
For a local configuration check, add `--dry-run` to the Python command.

`CODING_ACCEPTANCE_LENGTH` must come from a separate standard `eagle3` run on
coding prompts, using the counter formula below. For AgentX comparisons, follow
the pinned [SPEED-Bench coding procedure](https://github.com/SemiAnalysisAI/agentx-harness/blob/e4bb2f835ffedbbe6b76252ea5c93caec68c5faf/docs/tutorials/speed-bench.md#single-category).
Match the target, draft, depth, sampling and thinking settings, and disclose any
context filtering needed to fit this pair. Without request overrides, the pinned
template emits no explicit `Reasoning` header. Setting `enable_thinking` to true
or false in `chat_template_kwargs` adds `/think` or `/nothink`; calibration and
benchmark requests must use the same setting. The allowed mean is `[1, 4]`, including one
verifier bonus token. The value is forwarded unchanged to vLLM's native
`synthetic_acceptance_length` with `rejection_sample_method=synthetic`.

The registered endpoint is `<endpoint-prefix>-<mode>`. Check readiness and mint access:

```bash
uv run iris --cluster marin endpoints list /serve/my-snowball-baseline --exact
uv run iris --cluster marin endpoints mint /serve/my-snowball-baseline --ttl-hours 1
```

The minted URL is a credential and the complete endpoint root. Use it as printed:

```bash
BENCHMARK_ROOT='<complete minted URL, without a trailing slash>'
curl --fail "$BENCHMARK_ROOT/v1/models"
```

Relative to that root, `POST /v1/chat/completions` accepts the served alias,
`messages`, sampling settings and `stream: true` (SSE ending in `[DONE]`).
`GET /v1/models` reports the alias and context; `POST /tokenize` exposes the
engine tokenizer; `GET /metrics` exposes the vLLM counters.

The consumer interface was assessed at AgentX harness revision
[`e4bb2f835ffedbbe6b76252ea5c93caec68c5faf`](https://github.com/SemiAnalysisAI/agentx-harness/tree/e4bb2f835ffedbbe6b76252ea5c93caec68c5faf).
Its [quickstart](https://inferencex.semianalysis.com/agentx/quickstart) uses the
server root as `--url` and `/v1/chat/completions` as the endpoint. Select the
pinned tokenizer separately from the served alias. If the client cannot pin a
tokenizer revision, materialize it and pass the local directory as `--tokenizer`:

```bash
hf download marin-community/marin-tokenizer \
  --revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2 --local-dir ./snowball-tokenizer
```

## Metrics and recorded settings

Use an exclusive allocation for run-scoped counter deltas. Sum all matching
data-parallel engine series for `vllm:spec_decode_num_drafts_total` (verification
steps), `vllm:spec_decode_num_draft_tokens_total` and
`vllm:spec_decode_num_accepted_tokens_total` before computing
`1 + accepted_draft_tokens / verification_steps`. With zero steps the mean is
unavailable. Zero accepted drafts with positive steps gives a mean of one.
Baseline has no speculative measurement. Keep synthetic results separate from
standard acceptance and quality measurements.

The pinned harness's [metrics guide](https://github.com/SemiAnalysisAI/agentx-harness/blob/e4bb2f835ffedbbe6b76252ea5c93caec68c5faf/docs/tutorials/spec-decode-metrics.md)
describes server scraping. Its SPEED summary reads only the first counter series;
use its raw scrape for a whole-server mean with DP8. Point server metrics at
`<root>/metrics` if auto-discovery fails. Federated reads can exceed five seconds:
use a 30-second read deadline for probes and
`AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT=30` for harness startup.

Retain the launch configuration, request sampling settings, Iris source/bundle
and hardware allocation, startup logs and raw counter deltas. Evaluation records
preserve the requested sampler and synthetic mean. The repository's promoted
wheel and startup verifier supply the engine build identity; at introduction:
source `01911be34fac8715347962a8d791ca34a879e01e`, version
`0.0.0.dev20260924+marin.01911be34fac.cu132`. Capture the actual `source_commit`
and `version` fields from the startup log's `MARIN_VLLM_WHEEL_VERIFIED=` JSON.

## Context and workload limits

Both target and draft have a 65,536-token context. The tokenizer's advertised
131,072 does not extend it. The draft uses 2,048-token sliding attention. Prompt
plus output must fit 65,536 tokens; the launcher applies no truncation or context
extension. The pinned engine can fall back to ordinary decoding when the sequence
plus proposal budget exceeds the draft context. Requests beyond target context
are rejected; fallback is distinct from rejected draft tokens.

Published AgentX traces include contexts beyond 64k. A private consumer must
define and disclose a fitting workload, preserve input/output lengths and replay
settings, supply measured acceptance, and handle context errors. Follow its
[methodology](https://inferencex.semianalysis.com/agentx/methodology) for warmup and
measurement windows, and account for client/proxy transit in latency results.
The serving checks establish neither speedup, quality nor stock-workload or
submission conformance. No AgentX workload runs in this launcher.
