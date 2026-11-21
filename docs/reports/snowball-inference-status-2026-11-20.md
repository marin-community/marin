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

Full-model learner replay parity remains open. Native active decode is the
largest gap in the comparison.

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
  --weights "$weights" --workload "$workload" --attention tpu_v3 --decode-layers unrolled \
  --prefill-buckets 4096 8192 16384 32768 --prefix-cache last_prompt
```

The `tpu_v3` mode uses the pinned vLLM TPU fork's paged attention kernel with
FP32 value accumulation and BF16 inputs and outputs on v5p. It passes
packed-prefill and decode comparisons against a dense reference.
[Kernel source](https://github.com/marin-community/tpu-inference/commit/2f327d385a90347681a7de86472b4bace6e940b2) · [v5p correctness job](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rpa-v3-v5p-integrated-tests-20261121-v2)

Add `--profile-decode` to capture short and long warmed decode after the
measurement. Use `--attention tpu_fp32` to select the earlier bounded
attention kernel. To rebuild an expired workload receipt on a CPU worker, run
`python -m experiments.evaluation.knotmeter.workload` with the
`experiments/evaluation/knotmeter/snowball-v01-tpu-64k.yaml` series, a fresh
local output directory, and a region-local `--cache-root`. It writes the
receipt to `workload.json` in that directory.

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
