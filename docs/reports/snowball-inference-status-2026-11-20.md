# Snowball inference status: 2026-11-20

The bounded Snowball inference-to-learner path now reproduces sampled policy
logprobs bit-for-bit on four TPUv4 devices. The production engine retains its
device-side decode loop. On TPUv4, 32-token chunks measured 3.54× faster than
one-token chunks on the long-request fixture and 1.79× on the mixed-request
fixture. The full 67B checkpoint now runs AgentX on v5p-8. Full-model learner
replay parity and external rollout/learner integration remain open.

This publication snapshots accepted code at
`fcc3167fef231ed2dacdd9e21cd48a564668e7ee`. The publication commit adds documentation
and recorded results. It contains 42 inference commits after the shared ancestor
`7405c96fe6ea3675d5a1bf7977d406cfc3887243`. At publication preflight, upstream
`main` was `eecb397a0c29ef3cf0c123028a53df304a249b77`, 756 commits beyond that
ancestor. This branch has not been rebased onto or qualified against that newer
`main`.

## AgentX on v5p-8, 2026-11-21

Native Levanter served the BF16 Snowball 67B-A2B v0.1 checkpoint and completed
the 64K AgentX diagnostic on one v5p-8. The comparison used four chips in
`us-central1`, concurrency 1, a 180-second sending window, and a one-hour Iris
job cap. Both runs used checkpoint and tokenizer revision
`adc0078a19a4383bc998fcdb43867b3360baf5f3`, AgentX 1.0.12, and the same eight
eligible sessions from the pinned Weka corpus. The frozen workload SHA256 was
`16b554c33632fe702852ccc2d4382f34733fad3fae394179b3e0932eec0b13f2`.

| Run | Completed requests | Output tokens | AgentX output tok/s | Active decode tok/s/user |
| --- | ---: | ---: | ---: | ---: |
| [Native, bounded attention tiles](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v5) | 2 | 1,008 | 4.80 | 9.17 |
| [Native, vectorized prefill and clean shutdown](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v6) | 3 | 1,602 | 7.75 | 9.11 |
| [Native, compact sliding-window reads](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v7) | 3 | 1,602 | 8.78 | 10.50 |
| [Native prototype, stacked KV and fused prefill](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v8) | 4 | 1,917 | 9.13 | 12.10 |
| [Native prototype, unrolled decode and small KV updates](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v9) | 8 | 2,905 | 16.14 | 26.85 |
| [Native prototype, prompt-page reuse and fused full-context attention](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v10) | 12 | 4,133 | 21.83 | 31.46 |
| [Native, expert-sharded KV heads](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v11) | 18 | 6,461 | 33.22 | 61.13 |
| [Native prototype, split expert weights with replicated KV](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v12) | 13 | 4,505 | 24.33 | 36.53 |
| [Native, expert-sharded KV and split expert weights](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v13) | 22 | 7,853 | 42.13 | 84.50 |
| [Native, full-distribution sampler bypass](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v14) | 23 | 7,947 | 43.90 | 104.60 |
| [Native, fused decode KV append](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v15) | 25 | 8,311 | 46.17 | 114.22 |
| [Native, larger prefix cache with per-token layer copies](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v17) | 18 | 6,461 | 33.31 | 59.04 |
| [Native, per-layer buffers across each decode round](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v22) | 26 | 9,092 | 50.51 | 157.99 |
| [Native, per-round buffers and three recent prompt prefixes](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v21) | 26 | 9,092 | 50.51 | 158.99 |
| [Native, clean checkout of `5580503d9f`](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v23) | 26 | 9,092 | 50.51 | 165.68 |
| [vLLM TPU, prefix caching enabled](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-vllm-v5p8-v01-64k-c1-20261121) | 26 | 9,092 | 50.47 | 116.28 |

Reported throughput includes drain time. The second native run completed one
request during drain and exited successfully. The first exported its metrics
before an active batch triggered a libtpu shutdown abort. The
`--unsafe-override` setting makes these diagnostics ineligible for leaderboard
submission. Each run recorded two warmup errors and zero profiling errors.

The native rerun's three completed turns appear in the vLLM export with the same
output token counts. The longest had 45,502 input tokens on native and 45,500 on
vLLM. It took 88.16 seconds end to end on native and 6.29 seconds on vLLM. Native
used expert parallelism across four chips. vLLM used tensor parallelism across
four chips and reported an 88.8% prefix-cache token hit rate. The latency
percentiles summarize different numbers of completed requests.

The initial 64K compile exhausted HBM in attention. The tiled FP32 attention
path fit after bounding query and KV tile sizes. The original larger tiles also
compiled with `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=60000`, and
took 1.04 seconds on the synthetic long prefill. The bounded tiles took about
0.82 seconds at 50,000 KiB. The native runs in the table use bounded tiles at
50,000 KiB. [VMEM comparison](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rpa-v5p-vmem-20261121-v1)

The serving fixes cover tiled attention, streamed usage, graceful shutdown,
vectorized KV writes, and projection of sampled rows. [`2dd5eafffc`](https://github.com/marin-community/marin/commit/2dd5eafffc)

Full-model learner replay parity remains open. Time to first token accounts for
more of the remaining latency gap after the decode optimizations.

The short-window kernel now discards fully masked tiles while preserving the
softmax tile boundaries. It matched the untrimmed kernel bit for bit in the
packed-window test on v5p. With one decode query after a 51,870-token prefix,
the compact 2,048-token window took 0.38 ms and full-context attention took
1.35 ms on the same synthetic shape. [v5p kernel test](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rpa-v5p-window-20261121-v3)

Warmed decode traces at 512 and 51,870 prefix tokens took about 3.1 seconds
per 32-token round on the device. On one TPU TensorCore, three expert-weight
slice operations consumed about 870 ms and gate/up packing about 580 ms per
round. SparseCore scatter activity occupied about another second and
overlapped TensorCore execution. [In-region device trace analysis](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-v5p-decode-profile-analysis-20261121)

The stacked-cache prototype also used the pinned vLLM TPU fork's RPA v3 kernel
for short-window prefill with FP32 value accumulation and BF16 KV storage.
On the synthetic 51,870-token-input case, short-window prefill per layer dropped
from 822 ms to 24.8 ms. On the matched AgentX turn with roughly 45.5K input
tokens and 594 output tokens, time to first token dropped from 21.77 to 9.70
seconds, and request latency from 77.56 to 59.05 seconds.
[Kernel precision test](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rpa-v3-v5p-20261121-v2)

Unrolling the decode layers gave the compiler direct access to each layer's
expert weights. Dynamic slice updates for up to eight KV writes also removed the
small-batch scatter path. Together these changes raised active decode from 12.10
to 26.85 tokens/s/user. The 594-output-token turn completed in 30.81 seconds
with an 8.13-second time to first token. The full 67B checkpoint completed the
diagnostic on the same v5p-8 configuration. [Unrolled decode job](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v9)

An opt-in single-sequence cache reuses whole matching pages from the previous
prompt. With this cache, prefill buckets, and fused full-context attention, the
594-output-token turn took 24.13 seconds with a 4.73-second time to first token.
Three later turns reused 47,488 to 50,560 prompt tokens. Their server-side
prefill and extraction took 0.29 to 0.46 seconds. Warming the buckets took 118
seconds before the AgentX measurement. [Prefix reuse job](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v10)

Padding five KV heads to eight makes two head slots per expert shard. The
v5p-8 job moved attention off the replicated cache, raising native decode to
61.13 tokens/s/user. The same 594-output-token turn took 13.05 seconds with
a 3.23-second time to first token. The trace measured about 88 ms per
32-token round in full-vocabulary sort and gather operations.
[KV sharding and trace](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-v5p-decode-profile-analysis-20261121-v11)

The split-expert path reads gate and up matrices only for active assignments
in small expert-parallel decode batches. The synthetic balanced-route layer
took 0.32 ms, compared with 1.22 ms for grouped dispatch with in-step weight
packing. The full-checkpoint prototype used this path with sharded KV and
completed the same 594-output-token turn in 9.97 seconds.
[Expert microbenchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-small-ep4-v5p-20261121-v2) · [Matched AgentX run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v13)

AgentX used the full sampling distribution. Skipping the vocabulary sort for
those requests preserved seeded draws and log probabilities in the sampler
tests. The trace confirmed that the sort left the warmed decode loop. Fusing KV
append with paged attention brought active decode to 98.2% of the vLLM result.
The 594-output-token turn took 8.14 seconds. Across the measured requests,
native p90 time to first token was 2.37 seconds and vLLM was 0.56 seconds.
[Sampler trace](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-v5p-decode-profile-analysis-20261121-v14) · [Fused KV check](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-fused-kv-decode-v5p-20261121-v2)

The fused append is scoped to small decode batches. A synthetic long prefill
matched the cache contents for full-context attention. The sliding-window
case differed from the separate cache update. Prefill uses the separate
update for both attention schedules. [Prefill experiment](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-fused-kv-prefill-v5p-20261121-v1)

The decode trace showed roughly 57 µs per layer per token to copy the layer's
64 MiB KV shard out of the stacked buffer. The per-round layout splits the
cache once, carries each layer's physical shard through the 32-token device
loop, and repacks at the boundary. With 512 physical pages, native and vLLM
each completed the same 26 turns and emitted the same output-token count on
every turn. Native matched end-to-end
throughput and measured 36% higher active decode throughput. The p90 time
to first token was 2.14 seconds on native and 0.56 seconds on vLLM. The
594-output-token turn took 6.40 seconds on native and 6.29 seconds on vLLM.
[Matched native run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v22)

Allocating 2,048 physical pages let the engine retain full pages from three
recent prompts and reserve enough space for the active request. The per-round
layout sustained 50.51 output tokens/s with this cache. The run reused up to
63,232 prompt tokens on a turn and brought p90 time to first token to 1.24
seconds. [Three-prefix run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v21)

### Reproduce the native diagnostic

Run from the inference branch and point `iris_client` and `iris_config` to
the current Marin Iris client and its cluster config. Set `weights` to the
regional checkpoint mirror and `workload` to the frozen workload receipt.
Keep the weights and TPU worker in the same GCS region.

```bash
workload=gs://marin-us-central1/tmp/ttl=30d/agentx-workload/95f345f596f40f84b64cf57ad70cf53ea6493004a5ac71d6c24448da770b9fe2/workload.json
"$iris_client" --config "$iris_config" job run --no-wait \
  --job-name snowball-agentx-native-v5p8-64k-c1 \
  --priority batch --region us-central1 --tpu v5p-8 --enable-extra-resources \
  --cpu 32 --memory 340g --disk 80g --timeout 3600 --sync-package marin-core --extra tpu \
  -e JAX_PLATFORMS tpu -e LIBTPU_INIT_ARGS '--xla_tpu_scoped_vmem_limit_kib=50000' \
  -- uv run --no-sync python -m experiments.evaluation.knotmeter.native_snowball_probe \
  --weights "$weights" --workload "$workload" --attention tpu_v3 --decode-layers unrolled_split_cache \
  --prefill-buckets 4096 8192 16384 32768 --prefix-cache last_prompt \
  --kv-sharding expert --decode-experts split --kv-append fused
```

The `tpu_v3` mode uses the pinned vLLM TPU fork's paged attention kernel with
FP32 value accumulation and BF16 inputs and outputs on v5p. It passes
packed-prefill and decode comparisons against a dense reference.
[Kernel source](https://github.com/marin-community/tpu-inference/commit/2f327d385a90347681a7de86472b4bace6e940b2) · [v5p correctness job](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rpa-v3-v5p-integrated-tests-20261121-v2)

Add `--profile-decode` to capture short and long warmed decode after the
measurement. Use `--attention tpu_fp32` to select the earlier bounded
attention kernel. Set `--prefix-cache recent_prompts --kv-cache-contexts 4`
to select the larger prefix cache with the per-round layout. To rebuild an
expired workload receipt on a CPU worker, run
`python -m experiments.evaluation.knotmeter.workload` with the
`experiments/evaluation/knotmeter/snowball-v01-tpu-64k.yaml` series, a fresh
local output directory, and a region-local `--cache-root`. It writes the
receipt to `workload.json` in that directory.

## Prefill performance on TPUv5p

Native and vLLM ran the same BF16 Snowball v0.1 checkpoint on one v5p-8
each, with a 65,536-token context limit and one client request at a time.
The prefill probe sent identical raw token IDs to `/v1/completions` and
requested one output token. It discarded one warmup per case and reported
the median of five trials. Every fresh trial started with a different
prefix. Every primed trial was preceded by a request for the same prompt
without its final 256 tokens. The measurement includes HTTP and one token
of generation. [Native baseline](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-prefill-native-v5p8-20261121-v1) · [vLLM baseline](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-prefill-vllm-v5p8-20261121-v1)

| Input tokens | Native original, fresh / primed (s) | Native active expert rows, fresh / primed (s) | Native adaptive dispatch, fresh / primed (s) | Native finer buckets, fresh / primed (s) | vLLM, fresh / primed (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.374 / 0.316 | 0.210 / 0.167 | 0.206 / 0.100 | 0.140 / 0.100 | 0.076 / 0.043 |
| 8,192 | 0.563 / 0.329 | 0.369 / 0.173 | 0.338 / 0.107 | 0.267 / 0.106 | 0.209 / 0.052 |
| 32,768 | 1.491 / 0.378 | 1.046 / 0.198 | 0.923 / 0.131 | 0.828 / 0.128 | 0.889 / 0.087 |
| 51,840 | 2.778 / 0.395 | 1.883 / 0.215 | 1.622 / 0.149 | 1.510 / 0.145 | 1.560 / 0.112 |
| 64,512 | 3.175 / 0.395 | 2.149 / 0.227 | 1.907 / 0.164 | 1.902 / 0.156 | 1.966 / 0.129 |

The tuned run used two BF16 products with FP32 accumulation for the
prefill attention value multiply, smaller buffers for cached suffixes,
immediate admission when the single request slot filled, and a six-context
KV cache. The value-multiply change alone brought the fresh 51,840-token
measurement to 2.372 s. The active-expert run skipped unused matmul rows
and returned to four KV contexts. Fresh throughput rose from the original
18.7k to 27.5k input tokens/s. vLLM reached 33.2k. [Value-multiply trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-prefill-native-v5p8-20261121-v3) · [Six-context run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v5p8-20261121-v4) · [Active-expert run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v5p8-20261121-v5)

On the 26 common AgentX turns, every run emitted 9,092 output tokens.
Streamed p90 time to first token was 1.235 s for the earlier native
four-context run, 1.127 s with the six-context cache, 0.922 s for the
active-expert run, and 0.561 s for vLLM.
The six-context cache reduced native active decode on those turns from
160.34 to 154.67 tokens/s. The active-expert run recovered to 159.65
tokens/s. vLLM measured 116.28 tokens/s. Active decode here divides tokens
after the first output by the sum of per-request decode durations.
The active-expert run completed two additional turns and measured 58.22
output tokens/s overall. [Four-context run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-v01-64k-c1-20261121-v21) · [vLLM AgentX run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-vllm-v5p8-v01-64k-c1-20261121)

The adaptive-dispatch run also used static layers for buffers up to 512
tokens and tighter intermediate buckets. At 64,512 tokens it processed
33.8k input tokens/s on the fresh prompt, compared with 32.8k for vLLM.
On 26 common AgentX turns, native measured 159.56 active decode tokens/s
and 759 ms p90 TTFT. The two full runs measured 58.36 output tokens/s over
28 turns on native and 50.47 over 26 turns on vLLM. Medium-length fresh
prompts accounted for much of the remaining p90 gap, including native
TTFT of 860 and 892 ms for two prompts near 19K and 21K tokens. vLLM
measured 600 and 684 ms for the corresponding turns. [Adaptive-dispatch run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v5p8-20261121-v8)

The finer-bucket run added intermediate sizes and extended static layers
through 32,768 tokens. Fresh prefill was faster than vLLM at 32,768,
51,840, and 64,512 tokens. Primed prefill remained slower at each tested
length. On the same 26 AgentX turns, native active decode fell to
153.95 tokens/s and p90 TTFT rose to 918 ms. The full run measured
53.86 output tokens/s over 27 turns. [Finer-bucket run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v5p8-20261121-v9)

The fresh prefill trace spent most of its device time in routed expert
matmuls and long attention. In the primed trace, async copies and packing
expert weights were among the largest costs. A BF16 expert microbenchmark
with 51,840 active tokens in a 65,536-token buffer cut median layer time
from 57.74 to 37.11 ms by skipping unused matmul rows. Outputs were bitwise
equal with zero dropped routes, including a separate concentrated-routing
case. [Profile summary](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-prefill-profile-summary-v5p8-20261121-v1) · [Expert microbenchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-ep-active-prefill-v5p8-20261121-v1)

### Reproduce the prefill comparison

Use `--prefill-only` in either probe for the one-token grid, or pass
`--workload "$workload" --measure-prefill` to run AgentX before the grid.
The native probe also accepts `--profile-prefill` to capture two warmed
requests after measurement.

For the compiled vLLM run, use the Knotmeter experiment directory from the
inference branch and Marin serving revision `2bf3f721c9`. That revision
pins the TPU adapter used in the comparison. Submit with the same Iris
hardware, region, resource, and timeout settings as the native command
above. Use `JAX_PLATFORMS=tpu,cpu` and this entry point. [Serving revision](https://github.com/marin-community/marin/commit/2bf3f721c9946150a02c3417c4c93e14b273a5e8)

```bash
uv run --no-sync python -m experiments.evaluation.knotmeter.tpu_probe \
  --series experiments/evaluation/knotmeter/snowball-v01-tpu-64k.yaml \
  --weights "$weights" --chips 4 --tensor-parallel 4 --concurrency 1 \
  --max-seqs 8 --prefill-only
```

## TPUv6e validation

The native paged-attention suite passed all 17 tests on both one v5p-8 and
one v6e-8, including the packed prefill and fused decode KV append checks.
The v5p job used 50,000 KiB of scoped VMEM and the v6e job used 98,304 KiB.
[v5p tests](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rpa-v3-v5p8-parity-20261121-v3) · [v6e tests](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rpa-v3-v6e8-parity-20261121-v4)

The v6e full-model runs use one eight-chip host each in `us-east5`, a
regional mirror of the pinned checkpoint, and the same 64K, C1,
180-second AgentX replay and prefill grid. Native uses EP8. vLLM uses TP4
and EP2 because its tensor-parallel size must divide the model's 20 query
heads. The pinned vLLM adapter needs `NEW_MODEL_DESIGN=1` to build its
full tensor and expert mesh. Its legacy mesh rejected eight devices with
TP4. [Legacy mesh failure](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-vllm-v6e8-20261121-v1) · [vLLM mesh retry](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-vllm-v6e8-20261121-v2)

Compiled vLLM completed 25 AgentX turns at 46.17 output tokens/s with
518 ms streamed p90 TTFT. Its one-token prefill grid completed all 60
requests, including warmups. [vLLM v6e-8 measurements](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-vllm-v6e8-20261121-v2)

| Input tokens | vLLM fresh (s) | vLLM primed (s) |
| ---: | ---: | ---: |
| 2,048 | 0.058 | 0.032 |
| 8,192 | 0.184 | 0.039 |
| 32,768 | 0.734 | 0.064 |
| 51,840 | 1.226 | 0.081 |
| 64,512 | 1.539 | 0.093 |

Native checkpoint conversion initially requested 36.40 GiB of HBM
temporaries per chip with 31.24 GiB available. The retry uses
`--layerwise-load`, which invokes the same Snowball conversion for each
layer and releases its source tensors before converting the next one.
On a four-device CPU mesh, all 46 arrays in a two-layer BF16 fixture
matched the existing conversion in values, dtype, and sharding, with and
without a checkpoint prefix. [Whole-checkpoint failure](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v6e8-20261121-v1) · [Layerwise retry](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v6e8-20261121-v2)

Layerwise conversion loaded the 67B model in 284 seconds and reached a
successful two-token completion. The next prefill bucket used a scanned
layer loop and requested 39.56 GiB of HBM temporaries per chip. The
full-context retry sets `--max-unrolled-prefill-tokens 65536` to avoid the
runtime stack of layer weights. [Static-layer retry](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v6e8-20261121-v3)

## H100 AgentX comparison

Each run used one eight-H100 host, the pinned BF16 checkpoint, and the same
checksum-verified Weka replay at 64K and C1 for 180 seconds. Both runs
completed with zero profiling errors and two warmup errors. Replay pacing
produced different completed turn sets.

| Engine | Completed requests | Output tokens | AgentX output tok/s | Active decode tok/s/user | p90 TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Native CUDA attention](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v1) | 22 | 7,853 | 43.63 | 85.04 | 997.65 |
| [vLLM with CUDA graphs](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-vllm-64k-c1-20261121-v1) | 28 | 11,014 | 58.65 | 164.23 | 940.69 |

The 22 common turns produced the same output token count on every turn.
On those turns, active decode measured 85.77 tok/s for native and 164.35
tok/s for vLLM. Their streamed p90 TTFT was 997.65 ms and 946.45 ms.

The selected-route CUDA kernel and static single-token KV writes brought
native throughput to 48.73 output tokens/s overall, 17% below vLLM's
58.65. On 26 common turns, both engines emitted 9,092 tokens with the
same count on every turn. Native and vLLM active decode were 117.00 and
164.79 tokens/s, respectively. Streamed p90 TTFT was 914.53 ms on native
and 942.61 ms on vLLM. [Selected expert kernel run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v7-static-experts) · [Static KV write run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v8-static-kv)

## What works

- **Paged generation.** The real `InferenceEngine` handles heterogeneous causal
  prompts, prefill, queued admission, slot reuse, and STOP/LENGTH completion.
  After prefill, `max_rounds` controls how many decode steps execute in the
  device `while_loop` before host extraction. One step generates one token per
  active sequence. Sampled outputs and optional route-capture buffers stay on
  device through the chunk.
- **Portable serving-route replay.** `generate_snowball_replay` uses the
  production scheduler and captures actual per-layer expert choices and row
  validity. Each request produces a versioned JSON/safetensors payload binding
  tokens, positions, prompt boundary, sampling metadata, policy identity and
  routes. It carries no KV cache or model weights. The caller must bind the
  policy identity to the exact weights; a payload checksum does not establish
  sender authenticity or hash the weights.
- **Learner forward, loss and gradients.** The learner consumes those routes
  through the public Snowball model and next-token loss. Replay fixes expert
  choices while current parameters still determine combine weights and
  gradients. The native qualification reloads the exact source weights in a
  fresh process and packs transported prefixes into the supported four-slot,
  256-token learner layout. It creates no persistent optimizer state.
- **Shared arithmetic.** Opt-in fixed-schedule projections, gates, weighted
  RMSNorm, Q/K RMSNorm and shared attention align teacher forcing with paged
  decoding in the bounded fixture. The sampler and unfused policy loss also
  share FP32 log-normalization. Fused projection/loss kernels retain their own
  normalization schedule.

The branch also contains native Hero model loading and paged short-convolution
state, with CPU parity coverage. The TPU results below qualify Snowball only.
See the [inference design](https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/docs/design/inference.md)
for API and backend contracts.

## Latest native evidence

The final run used four TPUv4 chips in one process, with a row-sharded explicit
mesh: data size four; context, expert and model sizes one. The model had two
layers, hidden width 128, two experts with top-1 routing, intermediate and shared
widths 32, two query heads, one KV head, head dimension 128, vocabulary 23 and
sequence length 256. Short attention layers used half-RoPE and a 128-token
window; long layers used full-causal NoPE. KV storage was BF16, with highest
matmul precision. These are small seeded weights, not a production checkpoint.

The test generated 15 requests and 555 targets across long continuations,
heterogeneous queued requests, and first-token STOP. It compared public learner
outputs with an independent cached-serving reference, then checked the actual
engine's sampled raw logprobs. Raw-policy equality was tested at temperature
one and top-p one; other sampling settings transform the distribution.

| Check | Actual result |
| --- | --- |
| Full-vocabulary logits | 56,235 FP32 values bitwise equal |
| Full-vocabulary raw logprobs | 12,765 FP32 values bitwise equal |
| Actual engine sampled raw logprobs | All 555 bitwise equal |
| Weighted next-token loss | Maximum absolute difference `2.3841858e-7` |
| Parameter gradients | Four batches, 46 leaves / 474,628 elements per batch; maximum absolute difference `4.1987733e-8` |
| Loss / gradient tolerances | Existing `1e-5` / `1e-4`; unchanged |
| Latest normalizer CPU checks | 30 tests pass; separate four-device CPU sharding and gradient probes pass |
| Latest changed-file types | Zero scoped errors |

Whole-repository typing remains unclean: the latest full check reported 23
missing optional-module imports and one existing Snowball type diagnostic in
unchanged files. These focused checks do not establish a green full-branch CI
run for all 42 implementation commits.

An earlier one-layer NoPE fixture also passed 555 native targets before the
final normalization repair. Its evidence is historical; the table above is the
final two-layer result on the repaired implementation.

### The last logprob mismatch

Identical predictor logits originally normalized one FP32 ULP differently in
one-row STOP and 129-row continuation contexts. A fixed-row normalizer alone
still differed inside the engine because the engine vmaps the sampler over
sequences, adding an outer dimension to its tile layout. The accepted fix
absorbs mapped dimensions before making eight-row blocks, gathers vocabulary
shards, uses highest-accuracy exp/log, and preserves differentiation with an
analytic JVP. Direct, mapped and nested-mapped native checks now agree bitwise.

The loss check uses an independent stable highest-accuracy exp/log expression.
An FP64 calculation from retained native logits showed that ordinary TPU
`logsumexp` was `4.81e-5` from the oracle in the first batch; the new public loss
was `1.59e-7` away. The final table compares loss and gradients with a separately
coded stable max/exp/sum/log reference using highest-accuracy exp/log, replacing
ordinary TPU `logsumexp`. Neither loss nor gradient tolerances changed.

## Device-loop performance

The same final two-layer fixture compared `max_rounds=1` and `max_rounds=32`,
with one first call (including compilation), two warmups and five measured
calls per arm per workload, alternating arm order: 16 calls per workload and
48 total. All complete calls retained identical tokens, sampled logprobs and
finish reasons between chunk sizes. Tokens and finish reasons also matched the
pre-repair benchmark.

| Workload | Requests / generated tokens per call | 1-step mean | 32-step mean | Speedup |
| --- | --- | --- | --- | --- |
| Long continuations | 4 / 516 | 582.44 ms | 164.76 ms | 3.54× |
| Mixed lengths, queued admission and slot reuse | 7 / 35 | 125.85 ms | 70.31 ms | 1.79× |
| First-token STOP | 4 / 4 | 29.09 ms | 29.33 ms | 0.99× |

The pre-repair 32-step means were 165.99, 70.21 and 29.16 ms, respectively.
The normalization fix left these measured latencies essentially unchanged.
These are finite completed-batch measurements on a small model. They do not
measure sustained service throughput, time to first token, or a production-size
model. First-token STOP provides no multi-step work to amortize.

## Remaining work

1. Integrate this snapshot with current `main` and run the affected suite and CI.
2. Generalize and qualify the `fixed_schedule` Snowball projection/attention
   path and its Pallas RMSNorm kernels at production layer counts,
   dimensions, expert counts, vocabulary sizes and context lengths. Genuine
   tensor/expert/context parallelism and multi-host execution are outside the
   qualified four-device row-sharded layout.
3. Add a public learner batch adapter for the fixed four-by-256 attention
   layout. The single-payload `replay_snowball_logits` / `replay_snowball_loss`
   convenience APIs are not qualified for that fixed layout; the native gate
   explicitly constructs the supported batch.
4. Connect replay payloads and immutable weight identities to the external
   rollout/learner boundary. vLLM/SkyRL integration, arbitrary packed segments,
   capture-time clones/callbacks/abort, and raw-policy comparisons under changed
   temperature/top-p remain outside the qualified route-replay contract.
5. Measure full-checkpoint latency, memory, compile time and sustained
   throughput on intended TPU/GPU targets. The production-shape unfused loss
   materializes full-vocabulary logits; the fused-loss path has a separate
   normalization schedule and is not covered by this bitwise claim.

All accepted implementation outputs had been integrated at this snapshot. No
native job remained active. Publication does not launch another experiment.

## Code, commands and evidence

- [Production engine](https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/src/levanter/inference/engine.py)
- [Engine replay API](https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/src/levanter/inference/snowball_engine_replay.py)
- [Portable payload](https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/src/levanter/inference/snowball_replay_payload.py)
- [Shared normalizer](https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/src/levanter/kernels/logsumexp.py)
- [Machine-readable qualification and timing data](data/snowball-inference-2026-11-20.json)

The native qualification executed implementation commit
`cca88c3d87e3b97a2774a6ecef329e8214c2bc9f`; accepted commit `fcc3167` has the same
tree. The native output archive SHA256 is
`e54492c7df61cd6ae1c2267bad449d4dd6f1c7ceed93eb655fdf7cd4e01fa839`.
This branch includes the audited numerical and timing summary. The exact
one-off qualification harness and raw outputs remain in the internal validation
record.

The checked-in commands below are local regressions and the engine benchmark;
they do not reproduce the full 555-target native gate by themselves. Run TPU
commands only in an allocated TPU environment.

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/kernels/test_batch_invariant_logsumexp.py \
  lib/levanter/tests/test_sampler.py \
  lib/levanter/tests/test_loss.py \
  lib/levanter/tests/test_lm_model_loss.py

uv run --package marin-levanter python \
  lib/levanter/benchmarks/snowball_engine_performance.py \
  --backend cpu --output /tmp/snowball-engine-cpu.json
```

The checked-in engine benchmark defaults to its one-layer fixture. The final
native measurement above uses the retained two-layer/window-128 configuration
override. Use the configuration recorded with the results when reproducing
those numbers.
