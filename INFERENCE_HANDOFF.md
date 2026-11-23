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

The binary serving endpoint is `/v1/levanter/generate`: a versioned frame
containing JSON metadata and safetensors tensors. It is separate from the
learner's local pickle-protocol-5 IPC and the protobuf worker-pipe proofs.

Penalty-free training is supported by disabling both `use_kl_loss` and
`use_kl_in_reward`. The quick-loop bridge requires both off, supplies no
reference scores, and uses zero KL weights. A zero coefficient alone does
not remove the reference-actor requirement when a KL flag remains enabled.
Numerical zero KL between training and serving is a separate qualification;
the exact result below applies to the stated TPUv4 fixture.

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
`7.55e-9` nats across the reported comparisons.

The production-engine replay A/B used the same requests and seeds at 32
decode rounds. Generated tokens, sampled logprobs, and finish reasons
matched bit-for-bit. Timing includes the serialized route payloads.

| TPUv4 fixture | Replay off (output tok/s) | Replay on (output tok/s) | Throughput change |
| --- | ---: | ---: | ---: |
| Homogeneous burst | 5,701 | 5,027 | −11.8% |
| Mixed requests with slot reuse | 948 | 739 | −22.1% |

[TPUv4 qualification](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-qualification-20261122-v3)

Packing route IDs and thresholds into contiguous device buffers reduced
transfer overhead. The repeat on the same TPUv4 worker reached 5,134
versus 5,709 output tok/s for the homogeneous burst, a 10.1% slowdown.
The mixed batch reached 754 versus 963 output tok/s, a 21.6% slowdown.
All compared outputs still matched bit-for-bit.
[Compact route buffer repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-compact-20261122)

A separate terminal executable added a device dispatch for short scanned
requests. It left throughput 11.4% lower for the homogeneous burst and 27.7%
lower for mixed requests with capture enabled. Capturing terminal rows
inline on scanned models reached 5,189 versus 5,739 output tokens/s
(−9.6%) on the homogeneous burst and 761 versus 964 (−21.1%) on mixed
requests. Tokens, scores, and finish reasons still matched exactly.
Unrolled models share one terminal executable across prefill and decode
buckets. The first cold v5p-8 prefill and extraction fell from 133 s to 61 s
after this change.
[Separate terminal capture](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-terminal-20261122)
· [Inline terminal capture](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-scan-inline-20261122)

The initial full-model replay arm on v5p-8 generated 400.97 output tokens/s.
That snapshot buffered text until completion for requests with raw policy
scores, so its first-token latency includes generation time. The second arm
hit the one-hour job cap during compilation.
[Paired v5p-8 AgentX runs](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-replay-pair-20261122)

The follow-up runs use the shared terminal kernel and incremental chat
streaming. They retain the same context, offered concurrency, checkpoint,
and frozen prompts. Both native arms request raw policy scores. The
compiled vLLM result is the previous same-platform AgentX baseline, using
its standard response fields.

| v5p-8 run | Output tok/s | P90 first token (s) |
| --- | ---: | ---: |
| Compiled vLLM baseline | 549.46 | 1.339 |
| Native, replay off | 570.53 | 1.340 |
| Native replay, original archive and framing | 442.62 | 2.153 |
| Native replay, original archive and byte-oriented framing | 463.50 | 1.577 |
| Native compact replay, four prefill slots | 518.90 | 1.619 |
| Native compact replay, eight prefill slots | 522.62 | 1.687 |

The initial replay lowered throughput by 22.4% against the native control and
19.4% against compiled vLLM. Both jobs reported zero profiling errors.
The replay arm cancelled one outstanding request at the drain deadline.
Its median HTTP response was 33.4 MiB, compared with 15.3 KiB for the control.
[Optimized replay](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-optimized-replay-20261122)
· [Replay-off control](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-optimized-control-20261122)

The streaming adapter now writes a route event as JSON bytes and sends the
SSE framing separately. A synthetic 46 MiB route event took 56 ms to format,
compared with 118 ms for the old formatter, and produced identical bytes.
The normal OpenAI stream and tensor endpoint still agree on route identity
and policy scores.
[Byte-oriented replay run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-byte-stream-replay-20261122)

Route archive version 2 uses one byte per expert ID when all IDs in the
layer fit. The decoder widens IDs before validating the unchanged evidence
digest and still reads version 1 archives. A synthetic 64K capture with
the 67B model's route dimensions shrank from 34.63 MiB to 15.13 MiB before
base64 encoding. The fresh-process score and gradient tests and the OpenAI
stream and binary endpoint checks passed.

The compact AgentX run's median HTTP response dropped to 14.5 MiB and the
median interval from final text to HTTP completion fell from 420 to 179 ms.
All 366 measured requests completed with zero errors. Compact replay was
5.6% behind compiled vLLM on output throughput and 9.0% behind native with
replay off. Its p90 first token was 21.0% above compiled vLLM, 13 ms above
the 20% limit. Increasing the prefill batch limit to eight added 3.72 output
tokens/s and 68 ms of p90 first-token latency. All 370 requests completed
with zero errors. The two-slot run tests the smaller batch limit while
holding the other settings fixed.
[Four prefill slots](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-compact-wire-replay-20261122)
· [Eight prefill slots](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-eight-prefill-replay-20261122)
· [Two prefill slots](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-two-prefill-replay-20261122)

On CoreWeave, a four-GB200 smoke matched packed native and captured
generation, including scores and terminal routes. Both full AgentX runs
used the existing regional 67B checkpoint mirror at 64K and concurrency 128,
with route capture off. Native generated 571.43 output tokens/s versus
654.80 for compiled vLLM, a 12.7% throughput gap. Native p90 first-token
latency was 0.665 s versus 1.047 s for compiled vLLM.
[GB200 smoke](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-cw-gb200-smoke-20261122-v3)
· [Native AgentX](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-cw-gb200-c128-agentx-20261122)
· [Compiled vLLM AgentX](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-vllm-cw-gb200-c128-agentx-20261122)

## Distributed rollout and publication

Current distributed publication copies JAX shards to host bytes, stages them
as Torch GPU tensors, and uses the existing SkyRL expert-block/NCCL receivers.
Native `reload` stages and atomically installs weights within one serving
runtime. Distributed learner-to-native transport and coordinated replica
installation/admission still need wiring and qualification, including transfer
latency and staging-memory measurements.

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
Run `2026.11.22.4` tested the repair from `768ea25030` on 40 H100s. After
one coordinator preemption, retried child `2026.11.22.4-a7ebaaff0409` passed
initial and step-one finite-state checks on all four learners. Model and Adam
fingerprints changed, and step-1/version-1 checkpoint metadata was committed.
The learners completed all seven cleanup phases, acknowledged shutdown and
entered Python `atexit`, but again exceeded the five-second natural-exit limit.
The driver failed at 18:04 PST; the cache fix did not resolve GPU exit.
All five current GPU pods and the five interrupted-attempt GPU pods are gone.
The accepted bundles match all 4,590 frozen source entries and 49 required
runtime files; all 670 materialized configuration fields match the frozen recipe.
Progress from the interrupted attempt is kept separate. Fresh-process restore,
step two, updated-policy publication and generation remain unverified.
This run covers source `768ea25030`; it does not qualify the newer native
inference integration now on this branch.

The native server stages and installs weights inside one runtime. The worker
bridge accepts a version-bound native route payload per trajectory and
preserves that binding through batching and policy scoring. The distributed
driver still needs to launch native serving replicas and attach their payloads
during trajectory assembly. Updated-policy publication, full-model restart,
and a matched 25-update run on the maintained branch remain open. The
quick-loop design records the separate protobuf worker-pipe proofs and the
held H100 comparisons. Those worker responses carry the same replay payload
used by the native HTTP endpoints.
[Quick-loop migration design](https://github.com/marin-community/marin/pull/11934)
· [Quick-loop run status](experiments/post_training/inventory/levanter-quick-loop-status.md)
