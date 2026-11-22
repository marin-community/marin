# Snowball inference status: 2026-11-20

The bounded Snowball inference-to-learner path now reproduces sampled policy
logprobs bit-for-bit on four TPUv4 devices. The production engine retains its
device-side decode loop. On TPUv4, 32-token chunks measured 3.54× faster than
one-token chunks on the long-request fixture and 1.79× on the mixed-request
fixture. Full-model learner replay parity and external rollout/learner
integration remain open.

The full BF16 67B checkpoint completed the matched 64K AgentX replay at
128 offered clients on v5p-8, v6e-8, and eight H100s. The best native runs
reached 80% and 96% of compiled vLLM's aggregate output throughput on
v5p-8 and v6e-8. On eight H100s, two native replicas with singleton decode
reached 616.68–621.18 output tok/s versus 590.24–593.09 for compiled vLLM.
Native p90 time to first token was 0.96–1.01 s versus 1.43–1.44 s. At
64,512 fresh prompt tokens, native HTTP prefill reached 35.5k prompt tok/s on v5p-8 and 42.2k on v6e-8
versus 33.1k and 42.4k for vLLM.

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

| Input tokens | Native original, fresh / primed (s) | Native active expert rows, fresh / primed (s) | Native adaptive dispatch, fresh / primed (s) | Native finer buckets, fresh / primed (s) | Native smaller expert tiles, fresh / primed (s) | vLLM, fresh / primed (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.374 / 0.316 | 0.210 / 0.167 | 0.206 / 0.100 | 0.140 / 0.100 | 0.088 / 0.058 | 0.076 / 0.043 |
| 8,192 | 0.563 / 0.329 | 0.369 / 0.173 | 0.338 / 0.107 | 0.267 / 0.106 | 0.219 / 0.064 | 0.209 / 0.052 |
| 32,768 | 1.491 / 0.378 | 1.046 / 0.198 | 0.923 / 0.131 | 0.828 / 0.128 | 0.780 / 0.085 | 0.889 / 0.087 |
| 51,840 | 2.778 / 0.395 | 1.883 / 0.215 | 1.622 / 0.149 | 1.510 / 0.145 | 1.464 / 0.102 | 1.560 / 0.112 |
| 64,512 | 3.175 / 0.395 | 2.149 / 0.227 | 1.907 / 0.164 | 1.902 / 0.156 | 1.858 / 0.113 | 1.966 / 0.129 |

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

Computing the gate and up projections separately eliminated expert weight
packing on TPU and lowered all ten prefill medians. At 2,048 tokens
native fresh and primed latency were 0.123 and 0.082 s, down from 0.140
and 0.100 s with finer buckets alone. The 64,512-token medians were
1.875 and 0.142 s. On the same 26 AgentX turns, active decode reached
159.78 tokens/s with 746 ms p90 TTFT. The full run measured 58.37
output tokens/s over 28 turns. [Separate projections](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v5p8-20261121-v10)

Expert matmul tiles of 128 rows through 8,192 padded prompt tokens and
256 rows for larger buffers brought native ahead of vLLM on fresh and
primed prefill at 32,768 tokens and above. The tile sweep preserved exact
BF16 outputs and zero drops for all five tested sizes. The full native
run measured 58.38 output tokens/s. On 26 matched turns, active decode
was unchanged at 159.65 tokens/s and p90 TTFT fell to 711 ms. vLLM
measured 116.28 active tokens/s and 561 ms. [Tile sweep](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-ep-prefill-tiles-v5p8-20261121-v1) · [Tiled v5p-8 run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v5p8-20261121-v11)

The timed repeat measured 633 ms p90 TTFT and 160.15 active decode
tokens/s on the same 26 turns. On 20 of 28 turns, the first arriving
text followed the first sampled-token callback by more than 120 ms
after accounting for chat encoding and admission. The median residual
was 212 ms. A trial with a dynamic four-token first decode round lowered
matched median TTFT from 533 to 390 ms. Matched active decode fell to
142.10 tokens/s and full throughput fell from 58.59 to 50.51 output
tokens/s. The trial is outside the serving branch. [Timed repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-host-v5p8-20261121-v1) · [Dynamic round trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-host-v5p8-20261121-v2-first-round)

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

With static layers, native completed both full benchmarks on v6e-8.
Native completed 29 AgentX turns and produced 64.14 output tokens/s.
vLLM completed 25 turns at 46.17 output tokens/s. On the 25 common
turns, both emitted 8,311 output tokens with the same count for every
turn. Native active decode was 184.28 tokens/s and streamed p90 TTFT was
603 ms. vLLM active decode was 98.01 tokens/s and p90 TTFT was 518 ms.
[Native v6e-8 measurements](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v6e8-20261121-v3) · [vLLM v6e-8 measurements](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-vllm-v6e8-20261121-v2)

Both prefill grids completed all 60 requests, including warmups. Fresh
native prefill was within 4% of vLLM at 32,768 tokens and above.

| Input tokens | Native baseline, fresh / primed (s) | Native smaller expert tiles, fresh / primed (s) | vLLM fresh / primed (s) |
| ---: | ---: | ---: | ---: |
| 2,048 | 0.102 / 0.070 | 0.088 / 0.052 | 0.058 / 0.032 |
| 8,192 | 0.204 / 0.076 | 0.208 / 0.059 | 0.184 / 0.039 |
| 32,768 | 0.761 / 0.094 | 0.768 / 0.076 | 0.734 / 0.064 |
| 51,840 | 1.196 / 0.106 | 1.221 / 0.088 | 1.226 / 0.081 |
| 64,512 | 1.552 / 0.113 | 1.575 / 0.095 | 1.539 / 0.093 |

A trial with separate projections and smaller expert matmul tiles improved
all five primed medians. At 64,512 tokens it measured 96 ms, compared
with 113 ms in the earlier native run and 93 ms for vLLM. The trial
measured 59.62 output tokens/s overall, 163.69 active decode tokens/s
and 830 ms p90 TTFT on the same 25 matched turns. Most of the decode
slowdown came from four consecutive long turns. [v6e expert tile trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v6e8-20261121-v4)

An exact repeat on the same worker measured 64.33 output tokens/s overall.
On the same 25 matched turns its active decode of 183.86 tokens/s
was within 0.3% of native's earlier 184.28 tokens/s. Paired p90 TTFT fell
to 558 ms, against 518 ms for vLLM. The repeat reproduced the primed
prefill gains shown in the table. [Tiled v6e-8 repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v6e8-20261121-v5)

A 1,280-column expert output tile reduced the 51,840-token medians to
1.175 s fresh and 0.084 s primed, and the 64,512-token medians to 1.523
and 0.092 s. It preserved bitwise BF16 outputs and zero dropped routes
in the synthetic tile sweep. The full run measured 64.19 output tokens/s
and retained 184.13 active decode tokens/s on the same 25 paired turns.
Paired p90 TTFT was 610 ms. [Output tile sweep](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-ep-prefill-mxu-tiles-v6e8-20261121-v1) · [Wider tile AgentX run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-prefill-native-v6e8-20261121-v6)

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

A clean rerun with the selected expert kernel and static KV writes measured
48.61 output tokens/s overall, 116.58 active decode tokens/s, and
906.13 ms streamed p90 TTFT on the same 26 turns. It again matched all
9,092 output token counts. [Clean H100 rerun](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v12-clean)

Replicating the single decode activation while retaining sharded expert
weights and KV pages improved native to 50.51 output tokens/s overall,
14% below vLLM. On the same 26 turns and 9,092 outputs, active decode
reached 137.29 tokens/s against 164.79 for vLLM. Streamed p90 TTFT was
877.80 ms against 942.61 ms. [Replicated activation trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v19-replicated-activation)

The integrated branch reproduced 50.51 output tokens/s overall, and
measured 139.12 active decode tokens/s and 632.47 ms p90 TTFT on the
same 26 turns. [Integrated H100 run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v28-integrated)

The next trial kept per-layer KV buffers between decode rounds and
replicated KV for sliding-window layers. It produced 53.48 output tokens/s
overall against 58.65 for vLLM. Both engines emitted 9,967 tokens on the
27 common turns with identical counts for each turn. Native measured
151.93 active decode tokens/s and 891.50 ms p90 TTFT. vLLM measured
164.06 tokens/s and 941.65 ms. [Sliding-window KV trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v29-window-kv)

Split-K CUDA matvecs for replicated single-token attention and shared
expert projections raised native throughput to 65.16 output tokens/s.
The integrated repeat measured 65.34 output tokens/s against 58.65 for
compiled vLLM on eight H100 80 GB GPUs. Native completed 29 turns and
vLLM completed 28. On the 28 common turns, both emitted 11,014 output
tokens with identical counts for every turn.

| Engine | AgentX output tok/s | Paired active decode tok/s | Paired median TTFT (ms) | Paired p90 TTFT (ms) |
| --- | ---: | ---: | ---: | ---: |
| Native, integrated | 65.34 | 206.65 | 509 | 642 |
| Compiled vLLM | 58.65 | 164.16 | 375 | 941 |

The native run retained four KV contexts and used the same 64K,
concurrency-one, 180-second replay. [Projection trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v31-dense-matvec) · [Integrated repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-64k-c1-20261121-v33-integrated-dense)

Run the native H100 probe with the regional checkpoint and replay receipt.
These settings preserve the compiled prefill path and four KV contexts
used in the H100 comparison.

```bash
uv run --no-sync python -m experiments.evaluation.knotmeter.native_snowball_probe \
  --accelerator h100x8 --attention gpu --weights "$weights" --workload "$workload" \
  --decode-layers unrolled_split_cache --decode-experts split --decode-token-layout replicated \
  --kv-sharding expert --replicate-window-kv --kv-cache-contexts 4 --prefix-cache recent_prompts \
  --max-unrolled-prefill-tokens 0 --prefill-buckets 4096 8192 16384 32768
```

### AgentX at concurrency 128

Both engines used one eight-H100 80 GB host, the pinned BF16 checkpoint,
the frozen 64K Weka replay, and a 180-second profiling window. vLLM used
compiled CUDA graphs, eight data-parallel ranks with expert parallelism,
and up to 16 sequences per rank. Native used a single eight-way expert
mesh. Both engines enabled prefix caching.

| Engine | Completed requests | Output tokens | Output tok/s | p50 TTFT (s) | p90 TTFT (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Compiled vLLM](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-vllm-c128-20261122-v2) | 423 | 120,615 | 593.09 | 0.70 | 1.44 |
| [Compiled vLLM repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-vllm-c128-20261122-v3-repeat) | 419 | 120,006 | 590.24 | 0.73 | 1.43 |
| [Native, fixed request batches](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v2) | 47 | 10,043 | 47.82 | 55.89 | 127.39 |
| [Native, continuous admission](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v3-continuous) | 190 | 34,494 | 164.26 | 10.91 | 21.81 |
| [Native, per-layer KV and decode buckets](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v4-layer-kv) | 214 | 46,412 | 221.01 | 15.85 | 20.70 |
| [Native, adaptive prefix retention](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v5-adaptive-prefix) | 260 | 58,491 | 278.53 | 9.09 | 12.42 |
| [Native, separate expert projections](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v7-separate-moe) | 335 | 87,435 | 416.35 | 1.61 | 6.68 |
| [Native, shard-local sampling and 32 KV contexts](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v8-shard-sampling) | 378 | 100,674 | 479.40 | 0.94 | 2.02 |
| [Native, shared page reservations and 1 ms batching](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v9-shared-pages) | 385 | 101,743 | 484.49 | 0.76 | 1.45 |
| [Native, tighter prefill and decode buckets](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v10-tight-buckets) | 383 | 101,412 | 482.91 | 0.81 | 1.49 |
| [Native, compact decode and smaller expert tiles](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v11-expert-tiles) | 387 | 106,114 | 505.30 | 0.75 | 1.56 |
| [Native, 48 KV contexts and host-side completion](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v12-larger-kv) | 400 | 109,781 | 522.76 | 0.63 | 1.29 |
| [Native, packed expert routing](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v13-packed-routing) | 403 | 110,891 | 528.05 | 0.67 | 1.33 |
| [Native, 32 decode steps per device round](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v14-longer-rounds) | 412 | 114,335 | 544.45 | 0.99 | 1.94 |
| [Native, 48 KV contexts and H100 attention split tuning](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v19-48-context-tuned) | 413 | 114,525 | 545.35 | 1.11 | 2.15 |
| [Native, 56 KV contexts with bounded prefill scratch](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v20-bounded-prefill) | 415 | 114,820 | 546.76 | 0.85 | 1.45 |
| [Native, two four-GPU replicas](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v21-bounded-replicas) | 421 | 117,409 | 559.09 | 0.48 | 0.96 |
| [Native, two replicas and 24-step rounds](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v23-replicas-24-rounds) | 421 | 117,409 | 559.09 | 0.66 | 1.25 |
| [Native, two replicas and batched KV release](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v25-replicas-batched-release) | 433 | 124,472 | 593.47 | 0.44 | 0.95 |
| [Native, batched release repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v28-parity-repeat) | 431 | 120,142 | 572.10 | 0.43 | 0.90 |
| [Native, two replicas with singleton decode](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v27-replicas-singleton-fastpath) | 432 | 124,385 | 616.68 | 0.45 | 0.96 |
| [Native, singleton decode repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v29-singleton-repeat) | 431 | 124,171 | 621.18 | 0.44 | 1.01 |

The fixed-batch run had 12 full contexts of KV capacity. It cancelled 48
requests at the drain deadline. On 47 common turns, both engines emitted
10,043 tokens with identical counts, and active decode was 58.26 tok/s
for native and 45.46 tok/s for vLLM.

The continuous run increased KV capacity to 16 full contexts and admitted
new requests between eight-step device rounds. It cancelled 20 requests
at the drain deadline. Pairing by trace and branch coordinates matched
all 190 completed turns and all 34,494 output tokens. Paired active decode
was 11.93 tok/s for native and 45.48 tok/s for vLLM. The continuous run's
active decode interval includes pauses to process other requests' prompts.
Both native runs and vLLM reported zero profiling errors.

Keeping per-layer KV buffers throughout serving, using decode buckets,
and switching to 16-step rounds raised throughput to 221.01 tok/s. Both
engines returned identical output counts on all 214 paired turns. Native
active decode reached 18.68 tok/s on those turns and vLLM reached
48.35 tok/s. Native cancelled three requests after the drain deadline
and reported zero profiling errors.

Letting retained prefixes occupy unused KV pages and increasing capacity
to 24 full contexts raised native throughput by 26% from the per-layer
run. All 260 paired turns matched replay output lengths. Paired active
decode measured 18.19 tok/s for native and 49.27 tok/s for vLLM. vLLM
reused 41.9% of queried prompt tokens across its eight ranks during
profiling.

An 8,192-token chunked-prefill trial measured 119.65 output tok/s and
48.77 s p90 TTFT on the same eight-H100 workload. The branch retains
whole-prompt admission. [Chunked-prefill trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v6-chunked-prefill)

On one H100, bulk page allocation took 0.81 ms for one 65,536-token
prompt and 0.80 ms for eight 8,192-token prompts. The sequential
allocator took 23.97 and 24.46 ms respectively. The page mappings,
reference counts, and token destinations matched exactly across five
prefill and decode cases. [KV allocator benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-kv-allocator-h100-20261122-v1)

On eight H100s, separating the expert gate and up projections reduced
the expert MLP time from 0.86 to 0.50 ms for 32 padded decode rows and
from 2.34 to 1.96 ms for 8,192 prefill rows. Both configurations produced
bitwise-equal results for those inputs. [Expert projection benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-moe-h100-packed-20261122-v1)

The full run combined separate expert projections with bulk KV page
allocation and smaller prefill metadata. All 335 paired turns matched
the replay output lengths. On those turns, native active decode was
25.63 tok/s and vLLM was 52.51 tok/s. Native reported zero profiling
errors.

The shard-local sampler exchanges sampled tokens and scores while keeping
vocabulary rows on their original expert shards. With 32 full KV contexts,
native reached 80.8% of vLLM's aggregate throughput. All 378 paired turns
matched output lengths. Active decode reached 34.52 tok/s on those turns,
compared with 53.81 tok/s for vLLM. Both runs reported zero profiling errors.

The shared-page follow-up charges each physical prefix page once and
reduces the native HTTP admission wait from 100 ms to 1 ms. All 385 paired
turns matched output length. Paired active decode was 35.41 tok/s for
native and 53.95 tok/s for vLLM. Paired p90 TTFT was 1.453 s for native and
1.467 s for vLLM. The native run reported zero profiling errors.

Finer bucket boundaries left throughput within 0.4% of the shared-page
run and raised p90 TTFT by 33 ms. Later runs use the original buckets.

On one H100, smaller grouped matmul tiles reduced the three expert
projections from 0.213 to 0.175 ms with 128 padded rows and 12 active
routes. The 32-row case improved from 0.171 to 0.136 ms. Outputs matched
bitwise on all six tested route distributions, including a skewed batch
with 62 routes assigned to one expert. [H100 tile benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-h100-small-expert-tiles-20261122-v1)

With compact decode metadata and smaller expert tiles, the full H100 run
reached 505.30 output tok/s. It matched output lengths on all 387 paired
turns and reported zero profiling errors. Paired active decode was 36.38
tok/s for native and 54.53 tok/s for vLLM.

The 48-context follow-up reached 88.1% of vLLM throughput. Native used
a 64-row decode cap and kept output handling on the host. All 400
paired turns matched output lengths. Paired active decode was 40.94
tok/s for native and 54.86 tok/s for vLLM. Overall p90 TTFT measured
1.29 s for native and 1.44 s for vLLM.

Packing token activations, expert IDs, route weights, and validity into one
collective raised the same H100 configuration to 528.05 output tok/s.
All 403 paired turns matched output lengths. Paired active decode reached
41.61 tok/s for native and 54.97 tok/s for vLLM. The isolated v6e-4 test
preserved outputs bitwise and improved timings by 0.6% to 2.3% across
four packed shapes. [v6e routing benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-packed-routing-v6e-20261122-v1)

Increasing the 48-context baseline from 16 to 32 steps per device round
raised throughput to 91.8% of vLLM. All 412 paired turns matched output
lengths. Native active decode on those turns reached 49.44 tok/s, compared
with 55.33 tok/s for vLLM. Overall p90 TTFT increased from 1.29 to 1.94 s.

On one H100, using 64 key splits for full-context decode reduced attention
time from 0.319 to 0.292 ms at 32 padded rows and from 0.544 to 0.511 ms at
64 rows. The maximum absolute difference was below 5e-7. The 48-context
run used this split and 40-step device rounds. Paired active decode reached
55.61 tok/s for native and 55.35 tok/s for vLLM across 413 matching turns.
The higher overall p90 TTFT limited aggregate throughput to 92.0% of vLLM.
[Attention split benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-h100-attention-splits-20261122-v1)

The 52- and 56-context attempts hit memory limits during large prefill
buckets. Bounding the separate expert projections' scratch rows and
processing skewed routes in chunks let the 56-context configuration
complete AgentX with 32-step rounds. It reached 92.2% of vLLM throughput
and p90 TTFT was 1.453 s versus 1.443 s. All 415 paired turns matched
output lengths. Paired active decode was 53.72 tok/s for native and
55.36 tok/s for vLLM. Both runs reported zero profiling errors.
[52-context memory limit](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-h100-native-c128-20261122-v17-52-contexts)

The two-replica trial ran one expert mesh on each group of four H100s.
Each replica used 16 KV contexts, 16 decode steps per round, and up to
32 decode rows. A local streaming proxy routed requests by a hash of the
first user message to retain prefix affinity. The replicas reached
94.3% of vLLM's aggregate throughput with 0.96 s p90 TTFT. Across 420
paired turns, output lengths matched and active decode was 58.37 tok/s
for native versus 55.72 tok/s for vLLM. Both runs reported zero profiling
errors.

Increasing the replica decode round from 16 to 24 steps produced the same
aggregate output count and throughput, with 1.25 s p90 TTFT. The
batched-release trial returned to 16 steps and reached 593.47 output
tok/s versus 593.09 for vLLM. All 423 vLLM turns matched native output
lengths, and native completed 10 additional turns. Paired active decode
was 73.98 versus 56.91 tok/s, and paired p90 TTFT was 0.947 versus 1.443 s.
Both backends reported zero profiling errors. The native artifacts include
each replica's config and serving log.

The batched-release repeat reached 572.10 output tok/s and 0.898 s p90
TTFT. It matched all 422 paired output lengths. A 4,169-token turn from
the vLLM run was absent from the completed native turns. Paired active
decode remained faster at 75.58 versus 55.75 tok/s.

The singleton path replicates the activation when an H100 expert mesh
serves one live sequence. On four H100s, an isolated two-layer
model matched all 129 generated tokens at both tested prompt lengths.
At 29,000 input tokens, a 16-step decode round took 8.43 ms versus 9.09 ms
with packing. The 67B AgentX trial completed 124,385 output tokens with
a shorter drain and reached 616.68 output tok/s at 0.962 s p90 TTFT.
All 423 paired output lengths matched. Paired active decode was 72.94
tok/s versus 56.91 for vLLM.
[Singleton kernel test](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-replicated-singleton-h100-20261122-v2)

The singleton repeat reached 621.18 output tok/s. The pinned vLLM repeat
reached 590.24 output tok/s. Both completed with zero profiling errors.
All 419 paired turns matched output lengths, with active decode of
72.91 versus 53.97 tok/s and overall p90 TTFT of 1.006 s versus 1.430 s.
Across the two repeated configurations, native throughput exceeded vLLM
by 4.0% to 5.2%.

Reproduce the native run in an Iris job on one eight-H100 80 GB host.
Set the weights and replay receipt to the local CoreWeave region.

```bash
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_MEM_FRACTION=.95 \
  uv run --no-sync python -m experiments.evaluation.knotmeter.replicated_h100_probe \
  --weights "$weights" --workload "$workload" --concurrency 128 --max-rounds 16
```

## AgentX at concurrency 128 on TPU

The first native v5p-8 attempt with 16 full KV contexts reached decode
compilation. XLA reported 106.34 GiB of temporaries per chip against
95.73 GiB of available HBM. The lower-memory follow-up retains per-layer
KV buffers through both prefill and decode and sizes decode rounds to
the live batch. It served its preflight completion with eight full KV
contexts. Repeated vLLM worker losses interrupted compilation in
`us-central1`. The retry uses four precompile workers and a persistent
XLA cache in the worker's region. [Native v5p-8 failure](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v1) · [Native v5p-8 retry](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v3-layer-kv)

The first native v6e-8 attempt failed to allocate a 6.50 GiB device
buffer at preflight. With six full KV contexts, stacked decode required
38.55 GiB of temporaries per chip against 31.24 GiB available. The
per-layer retry uses four full KV contexts and selects the checkpoint
in the worker's region. It completed the full prefill warmup and reached
32.05 GiB of HBM temporaries during wider decode warmup. The follow-up
keeps 128 offered clients and caps each device decode step at eight tokens.
[Stacked decode failure](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v2-continuous) · [Per-layer decode failure](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v3-layer-kv) · [Capped decode trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v4-decode-cap)

Both compiled vLLM baselines enabled the prefix cache and configured
128 scheduler sequences and 8,192 batched tokens.

| Hardware and engine | Completed requests | Output tokens | Output tok/s | p50 TTFT (ms) | p90 TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [v5p-8, vLLM TP4](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-vllm-v5p8-c128-20261122-v3-persistent) | 385 | 107,895 | 538.33 | 286 | 1,539 |
| [v5p-8, native EP4](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v3-layer-kv) | 90 | 16,526 | 78.69 | 34,453 | 65,072 |
| [v5p-8, native EP4 with shared reservations and 12 KV contexts](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v4-shared-pages) | 118 | 20,338 | 96.85 | 23,806 | 41,521 |
| [v5p-8, native EP4 with compact attention metadata](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v5-compact-decode) | 139 | 23,367 | 106.21 | 23,908 | 34,497 |
| [v5p-8, native EP4 with fused append through 32 rows](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v6-fused-decode) | 157 | 26,337 | 121.82 | 20,917 | 33,037 |
| [v5p-8, native EP4 with 24 KV contexts](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v7-24-contexts) | 304 | 75,891 | 361.38 | 1,698 | 4,856 |
| [v5p-8, native EP4 with 32 KV contexts and bounded prefill scratch](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v10-bounded-32-contexts) | 325 | 84,402 | 401.91 | 1,548 | 3,932 |
| [v5p-8, 48 logical contexts and a FIFO window pool](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v11-hybrid-48-contexts) | 151 | 25,648 | 122.13 | 21,675 | 33,494 |
| [v5p-8, 60 logical contexts with retained window tails](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v12-hybrid-pinned-60-contexts) | 336 | 90,975 | 433.21 | 1,445 | 2,632 |
| [v6e-8, vLLM TP4 × EP2](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-vllm-v6e8-c128-20261122-v1) | 385 | 108,246 | 531.50 | 284 | 1,274 |
| [v6e-8, native EP8 with eight decode rows](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v4-decode-cap) | 102 | 18,619 | 88.66 | 41,158 | 64,947 |
| [v6e-8, native EP8 with shared reservations and 16 decode rows](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v5-shared-pages) | 111 | 19,864 | 94.59 | 36,227 | 53,402 |
| [v6e-8, native EP8 with compact attention metadata](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v6-compact-decode) | 145 | 29,477 | 133.99 | 31,722 | 40,447 |
| [v6e-8, native EP8 with eight KV contexts and separate append](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v7-eight-contexts) | 173 | 35,534 | 169.21 | 20,156 | 28,726 |
| [v6e-8, native EP8 with fused append through 16 rows](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v8-fused-decode) | 171 | 37,727 | 179.65 | 25,885 | 38,854 |
| [v6e-8, native EP8 with eight KV contexts and fused append](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v9-fused-live-prefix) | 180 | 37,969 | 180.80 | 24,634 | 35,626 |
| [v6e-8, 24 logical contexts with retained window tails](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v13-hybrid-terminal-pins) | 359 | 102,751 | 509.62 | 1,379 | 3,375 |

The v5p-8 baseline ran in `us-central1` with zero profiling errors. The
v6e-8 baseline ran in `us-east1` with one disconnected request. Native
v6e-8 ran in `us-east5` using the same checkpoint and replay hash. All
90 and 118 paired v5p turns and all 102 and 111 paired v6e turns matched
output lengths. Native paired active decode on v5p-8 was 9.95 tok/s
against 35.70 tok/s in the earlier run and 9.27 tok/s against 37.06 tok/s
with 12 KV contexts. The eight-row v6e run reached 25.24 tok/s against
33.56 tok/s. The shared-page v6e run reached 20.82 tok/s against 34.13
tok/s. All four native runs reported zero profiling errors. The first
v5p and both v6e runs showed no prefix reuse in their profiling logs.
The 12-context v5p run reused 746,880 prompt tokens across 18 requests.
The H100 32-context run logged 11.15 million reused prompt tokens across
252 requests.

vLLM's profiling counters showed cache hits for 88.6% of queried
prompt tokens on v5p-8 and 81.6% on v6e-8. It averaged 16.0 and 16.5
running requests in those runs. The compiled TPU backend reported four
hybrid KV cache groups with 61.15 and 26.79 full 65,536-token contexts of
capacity, respectively. Native used full-length BF16 pages for all layers
in the matching runs above.

On one v6e chip, trimming the attention metadata to the packed decode
width reduced sliding-window attention from 0.813 to 0.242 ms for eight
padded rows and from 0.880 to 0.295 ms for 16 rows. Each result matched
the padded-metadata output bitwise. The 32-row case measured 0.510 ms
with compact FP32 attention and 0.381 ms with paged BF16 attention.
[v6e window benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-short-window-attention-v6e-20261122-v1) · [Compact decode trial](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v6-compact-decode)

Compact attention metadata raised v6e AgentX throughput by 41.6% over the
same four-context configuration. All 145 paired turns matched output
lengths, with zero native profiling errors. Paired active decode was
28.13 tok/s for native and 34.20 tok/s for vLLM.

Eight KV contexts raised separate-append v6e throughput by 26.3% over
four contexts. All 173 paired turns matched output lengths. Paired
active decode was 16.44 tok/s for native and 33.10 tok/s for vLLM.

On v5p, compact metadata raised AgentX throughput by 9.7% over the
same 12-context configuration. All 139 paired turns matched output
lengths, with zero profiling errors. Paired active decode was
10.92 tok/s for native and 37.80 tok/s for vLLM.

Wider fused append raised v5p AgentX throughput by a further 14.7%.
All 157 paired turns matched output lengths. Native paired active decode
rose to 12.75 tok/s, against vLLM's 36.07 tok/s.

With 24 KV contexts, v5p-8 throughput rose by 197%. The native run reused
8.76 million prompt tokens across 192 admissions during the profiling
window. All 304 paired turns matched output lengths and paired active
decode reached 18.27 tok/s for native versus 33.27 tok/s for vLLM.

The bounded-prefill run with 32 KV contexts reused 10.22 million prompt
tokens across 227 admissions during profiling. It reached 401.91 output
tok/s with zero errors and matching output lengths for all 325 paired
turns. Paired active decode reached 21.32 versus 33.34 tok/s. Serial
fresh 64,512-token prefill took 1.846 s, or 35.0k prompt tok/s. Primed
prefill with a 256-token suffix took 0.168 s versus vLLM's 0.115 s.

The hybrid BF16 cache keeps full pages for seven full-attention layers
and shares a smaller page map across the 19 sliding-window layers. The
48-context FIFO trial retained just 1.22 million reused prompt tokens
over 29 admissions and measured 122.13 output tok/s. Preserving the
window tails of recent prompts with 60 logical contexts and 4,096
window pages raised reuse to 10.79 million tokens over 241 admissions.
Throughput reached 433.21 output tok/s with p90 TTFT of 2.63 s. All 336
paired output lengths matched and native reported zero profiling errors.
Native paired active decode reached 24.11 versus 33.34 tok/s.
The 64,512-token fresh and primed measurements were 1.836 s and 0.157 s.

Run the hybrid configuration inside a v5p-8 Iris job with local weights
and the frozen replay receipt.

```bash
export JAX_PLATFORMS=tpu
export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=50000 --xla_tpu_use_enhanced_launch_barrier=true'
uv run --no-sync python -m experiments.evaluation.knotmeter.native_snowball_probe \
  --weights "$weights" --workload "$workload" --chips 4 --concurrency 128 \
  --attention tpu_v3 --decode-layers unrolled_split_cache --kv-sharding expert \
  --kv-append fused --decode-experts separate --prefix-cache recent_prompts \
  --max-unrolled-prefill-tokens 65536 --max-seqs-in-prefill 4 \
  --kv-cache-contexts 60 --window-cache-pages 4096 --max-tokens-per-round 32 \
  --max-rounds 16 --decode-buckets 4 8 16 \
  --prefill-buckets 512 2048 8192 16384 32768 49152 --measure-prefill
```

On one v6e chip, fusing KV append at 16 padded rows cut full-context
attention plus cache update from 0.727 to 0.520 ms and sliding-window
attention plus cache update from 0.567 to 0.416 ms. The 32-row case
measured 0.579 versus 0.796 ms for full context and 0.430 versus 0.780 ms
for sliding-window attention. Cache contents matched bitwise at all
tested widths. The extended cache-update and dense-reference tests passed
all six cases on both TPU generations. [Packed append benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-wide-kv-append-v6e-20261122-v1) · [v5p append tests](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-wide-kv-append-tests-v5p-20261122-v2) · [v6e append tests](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-wide-kv-append-tests-v6e-20261122-v2)

The v6e-8 fused run raised throughput by 34.1% over compact decode
with four KV contexts. All 171 paired turns matched output lengths.
Native paired active decode reached 35.70 tok/s, compared with vLLM's
33.03 tok/s. The native run logged one reused prefix with 39,808 tokens.

With eight KV contexts, fused append, and retention of live prefixes, v6e-8
reused 168,192 prompt tokens across four admissions during profiling.
Throughput reached 180.80 output tok/s. All 180 paired turns matched output
lengths and paired active decode was 28.55 tok/s versus 32.49 tok/s.
Both of the higher-capacity native TPU runs reported zero profiling errors.

The 12- and 14-context v6e-8 attempts ran into the 31.24 GiB per-chip
HBM limit during preflight compilation. XLA reported 31.90 GiB and
31.58 GiB respectively. The 14-context run used bounded prefill scratch.
[12-context attempt](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v10-12-contexts) · [14-context attempt](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v11-bounded-prefill)

The compact BF16 cache fit 24 logical contexts and 3,072 sliding-window
pages on v6e-8. It reused 9.93 million prompt tokens across 223 admissions
during the 180-second send phase. The run reached 509.62 output tok/s
against 531.50 for compiled vLLM. It completed 359 requests with zero
errors. All 359 paired output lengths matched, and paired active decode
was 30.61 tok/s versus 32.91. The raw 64,512-token fresh prefill took
1.343 s versus 1.522 s, or 48.0k versus 42.4k prompt tok/s. With a primed
prefix and a 256-token suffix, native took 103 ms versus 84 ms.

On one v6e chip, batched KV page release took 0.46 ms versus 12.83 ms for
one newly finished slot following 20 stale finish flags. With four newly
finished slots it took 0.45 ms versus 18.26 ms. Both cases used a 128-slot
page table with shared prefix pages. Page mappings and reference counts
matched exactly.
[KV release benchmark](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-batched-kv-release-v6e-20261122-v1)

### Serial raw prefill with the concurrency-128 server configurations

Each sample issued one raw request and generated one token. Primed requests
reused a 256-token-shorter prefix. The table shows median HTTP TTFT in
seconds over five recorded trials per case after warmup. Effective
fresh prefill throughput at 64,512 tokens was 35.5k versus 33.1k prompt
tok/s on v5p-8 and 42.2k versus 42.4k prompt tok/s on v6e-8.

| Prompt tokens | v5p native fresh / primed | v5p vLLM fresh / primed | v6e native fresh / primed | v6e vLLM fresh / primed |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.0942 / 0.0619 | 0.0756 / 0.0432 | 0.0725 / 0.0429 | 0.0579 / 0.0325 |
| 8,192 | 0.2238 / 0.0715 | 0.2066 / 0.0506 | 0.1832 / 0.0488 | 0.1820 / 0.0383 |
| 32,768 | 0.7969 / 0.0985 | 0.8820 / 0.0796 | 0.7345 / 0.0686 | 0.7296 / 0.0619 |
| 51,840 | 1.5316 / 0.1191 | 1.5416 / 0.1012 | 1.1838 / 0.0831 | 1.2104 / 0.0739 |
| 64,512 | 1.8164 / 0.1287 | 1.9487 / 0.1153 | 1.5294 / 0.0895 | 1.5221 / 0.0838 |

Both native runners paused and resumed serving before these raw grids to
clear outstanding replay work. The v6e 64,512-token primed result improved
from 155 to 90 ms across the capped-decode and fused runs.
[v5p native grid](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v5p8-c128-20261122-v6-fused-decode) · [v5p vLLM grid](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-vllm-v5p8-c128-20261122-v3-persistent) · [v6e native grid](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-agentx-native-v6e8-c128-20261122-v8-fused-decode) · [v6e vLLM grid](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-prefill-vllm-v6e8-20261122-v2-packed-config)

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
