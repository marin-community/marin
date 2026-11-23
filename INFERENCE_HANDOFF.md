# Native Snowball inference and learner handoff

The inference stack on `codex/dlwh-levanter-rl` uses Snowball's paged prefill
and cached decode through a shared runtime. The Marin backend owns checkpoint
and mesh setup. The OpenAI and tensor APIs share request admission, stopping,
sampling, and policy installation. See [native inference](lib/levanter/docs/design/inference.md).

| Gate | Integrated path |
| --- | --- |
| Paged inference | Snowball initializes KV pages and runs prefill and decode with GPU and TPU implementations. Tests cover mixed requests, compact short-window KV, and reuse of retained prompt pages. |
| Versioned installation | The runtime stages and checks candidate arrays before swapping model and version. Pause and version fences invalidate earlier admission and cache state. Both endpoints return the emitting policy identity and reject stale expectations. |
| OpenAI and binary serving | Both endpoints accept exact prompt IDs and expose generated IDs, finish reasons, policy version, and separate policy and behavior scores. Each can return the Snowball route payload. |
| Router replay | Capture uses the optimized forward and keeps route rows alongside retained prefixes. The learner can validate the emitted positions, align routes against padded rows, move them during packing, and consume them in differentiable policy scoring. |
| Fixed-input qualification | The benchmark compares learner scoring, prefill and cached decode with fixed weights and input tokens. Report numerical KL separately from the reference-KL penalty used in the RL objective. |

## Benchmarks

The pinned AgentX workload uses the 67B-A2B v0.1 revision, 64K context, 128
offered clients, and a 180-second measurement. The published inference source
at `15bc5e751c` measured 575.55 versus 549.46 output tokens/s on v5p-8,
583.52 versus 554.00 on v6e-8, and 621.18 versus 590.24 on 8× H100. These
are the pre-integration native and compiled vLLM baselines. The integrated
replay-on/off overhead and vLLM gap have their own measurements.
[AgentX results and launch settings](docs/reports/snowball-inference-status-2026-11-20.md)

The four-chip TPUv4 fixed-schedule fixture matched all evaluated FP32 logits
between the learner, prefill, and cached decode. Max logprob error and
full-vocabulary forward KL were both zero. The native Splash control had
maximum absolute logit error 0.0004404 and maximum mean forward KL
`7.56e-9` nats across the reported comparisons.

The production-engine replay A/B used the same requests and seeds at 32
decode rounds. Generated tokens, sampled logprobs, and finish reasons
matched bit-for-bit. Timing includes the serialized route payloads.

| TPUv4 fixture | Replay off (output tok/s) | Replay on (output tok/s) | Throughput change |
| --- | ---: | ---: | ---: |
| Homogeneous burst | 5,701 | 5,027 | −11.8% |
| Mixed requests with slot reuse | 948 | 739 | −22.1% |

[TPUv4 qualification](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-qualification-20261122-v3)

The full-model v5p-8 job runs both replay arms with raw policy scores enabled.
Both use the same worker and pinned 64K AgentX traffic. The measured native
and compiled vLLM gap is a separate acceptance gate.
[Paired v5p-8 AgentX runs](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-replay-pair-20261122)

## Distributed rollout and publication

The SkyRL quick-loop recipe uses its vLLM engine group and expert-block
NCCL publication. Run `2026.11.22.3` at `baa10702ae` completed one finite
native learner update on 32 H100s. The initial publication verified
19,687,358,464 bytes per receiver on eight inference H100s and returned
generation stamp 1 for weight version 0. The run committed step-1/version-1
checkpoint metadata and reached Python shutdown. It failed the validator's
five-second natural exit deadline.

Commit `768ea25030` bounds stage-transform caches to 64 entries per publisher
and releases them with their owner. Fresh-process CPU tests verified release
after normal and aborted publication, and kept exact publication bytes.
Run `2026.11.22.4` is testing the repair from `768ea25030` with two updates
on 40 H100s. At 17:09 PST, all five GPU tasks were running and both staged
bundles matched their frozen source and runtime inventories. Process exit
and restart qualification remain open.

The native server stages and installs weights inside one runtime. The worker
bridge accepts a version-bound native route payload per trajectory and
preserves that binding through batching and policy scoring. The distributed
driver still needs to launch native serving replicas and attach their payloads
during trajectory assembly. Updated-policy publication, full-model restart,
and a matched 25-update run remain open.
[Quick-loop run status](experiments/post_training/inventory/levanter-quick-loop-status.md)
