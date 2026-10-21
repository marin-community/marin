# Tune vLLM for Snowball and Hero

This guide preserves the dated measurements in [MarinSkyRL #873](https://github.com/marin-community/MarinSkyRL/pull/873). Rates and qualification limits describe those source revisions. Use the current [Marin SkyRL artifact launcher](skyrl-migration.md) and validate its resolved recipe before running a historical layout.

[Choose the task](skyrl-performance.md), then use the closest measured BF16 layout for standalone serving or SkyRL rollouts on H100 or GB200/B200. Check queues, KV pressure and useful output before tuning. Historical measurements were checked on 30 September 2026.

**Keep Hero's qualified 4K settings:** non-eager, batch-invariant serving and captured expert routes for learner replay. We lack matched numerical and online-performance measurements for invariance OFF, eager mode or another attention backend. Longer contexts have performance-only evidence. [Qualification](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5835032707); [numerical tradeoffs](skyrl-performance-megatron.md#performance-and-numerical-alignment).

## Standalone serving

Use an identified Marin wheel and staged BF16 export. Check the [Marin GPU artifact](https://github.com/marin-community/marin/blob/main/config/external/vllm/gpu.toml), installed version and supported arguments. Trials used vLLM source `25f0fc1a` over precompiled kernels; validate any newer wheel.

**Hero release gap:** on 30 September, Marin's promoted artifact and draft [runtime #799](https://github.com/marin-community/MarinSkyRL/pull/799) pinned `01911be34fac`. It lacks the qualified split-expert loading and bounded reload. Use the identified trial assembly or a later qualified immutable wheel; those old pins cannot reproduce qualification.

Short-context trials used TP1/PP1 with data and expert parallelism across the serving GPUs. Snowball H100 fits one eight-GPU node; Hero and the 16-GB200 layouts require the tested placement and network across nodes. Stage checkpoints once per node and check memory before loading. See [the Iris operations guide](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/.agents/ops/coreweave.md).

For latency, start at expected live concurrency below the throughput point. Measure p95 first-token and response time at the expected burst; the capacity rows do not establish a latency limit. For bulk throughput, replace completed clients throughout the window and also measure whole-bank completion, including tails.

<details>
<summary>Snowball one-node server argument sketch</summary>

Server arguments for eight H100s, a supported installed fork and a staged export. Add deployment credentials and networking through the normal launch path:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve /path/to/staged-snowball \
  --dtype bfloat16 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 8 --enable-expert-parallel \
  --max-model-len 4096 --max-num-seqs 1024 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching
```

For RL, configure route capture, processed token log-probabilities and weight transfer in SkyRL. Match the sampling/stop policy; forced-length outputs cannot measure answer quality.

</details>

## Measured serving examples

Measured with a forced-length 4K GSM8K bank, BF16, non-eager batch-invariant serving and prefix caching off. Sequence caps are per engine; clients and **supplied decode tok/s** cover the fleet. Below, cap A/B means `max_num_seqs` / `max_num_batched_tokens`. Whole-bank and RL rates use different windows and tokens.

| Model / hardware | Serving GPUs; engine sequence / batched-token caps | Tested clients → decode tok/s | What the row supports |
| --- | --- | --- | --- |
| Snowball / H100, 26 Sept | 8; 1,024 / 4,096 | 4,096 → 32,967, zero KV preemptions | Short bulk/rollout capacity reference; 8,192 clients overload KV. [H100 evidence](#serving-evidence) |
| Hero / H100, 28 Sept | 32; 64 / 512 | 768 → 3,558, zero preemptions; 1,280 → 5,243, 120 preemptions | 768 is the lower-pressure measured point. At 1,536, 5,690 tok/s costs 448 preemptions for only 1.7% more whole-bank rate than 1,280. [Evidence](#serving-evidence) |
| Snowball / GB200, 28 Sept | 16; 512 / 4,096 | 4,096 → 29,354, zero preemptions | Curve still rising. Smaller fleet and interactive limit unknown. [Evidence](#serving-evidence) |
| Hero / GB200, 29 Sept | 16; 256 / 512 | 3,072 → 9,142, zero preemptions, peak sampled KV 88.4% | At 4,096 clients: 9,960 tok/s, 100% KV, 218 preemptions. Eight-GPU isolated sizing remains unknown. [Evidence](#serving-evidence) |

Caps and supply differ across rows, preventing a direct model-speed comparison. The window covers 20–80% of completions with at least one full client wave remaining. Zero preemptions requires a fresh, complete series.

## SkyRL rollout serving

Use the [standard](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/.agents/skills/rl-standard-launch-iris/SKILL.md) or [agentic](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/.agents/skills/rl-agentic-launch-iris/SKILL.md) Iris launch path. Inspect resolved `generator` fields: `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `enable_prefix_caching`, `enforce_eager` and `engine_init_kwargs`. Rollout admission controls supply. See [ppo_base_config.yaml](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/skyrl-train/skyrl_train/config/ppo_base_config.yaml).

Use 64/512 caps for the tested Hero H100 online path and 16/512 for Hero GB200. Raising the H100 sequence cap 64→128 reduced accepted RL throughput 8.7% with 90 versus zero KV preemptions. GB200's 16→32 change gave 89.56→88.39 accepted response tok/s with zero preemptions in both arms. These four-step samples generated different outputs, so they do not isolate the cap's cost. [Exact online runs](skyrl-performance-megatron.md#online-evidence).

Snowball's [32+8 H100 example](skyrl-performance-megatron.md#worked-task-short-snowball-math-rl-on-h100) used 1,024/4,096 caps with little warm learner wait or queueing. That window gives no reason to add capacity.

## Read the signals together

Open [RL Post-training (async)](https://grafana.oa.dev/d/marin-async-rl), select the cluster/run and check fresh samples per engine. SkyRL exports `service=marinskyrl`, `metric_source=vllm`; standalone `/metrics` uses `service=vllm`. See [run selection](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/docs/grafana-rl-runs.md), the [metrics contract](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/docs/design/vllm-metrics-contract.md) and [FineLog access/SQL](skyrl-performance-megatron.md#telemetry-you-can-use).

| Signal and source | Unit / window | Supports the hypothesis | Contrary evidence and next action |
| --- | --- | --- | --- |
| Native `generation_tokens_total`, `prompt_tokens_total`; `/metrics` or FineLog | Counter delta / elapsed seconds, per physical engine then fleet, 5 min or exact ordinary phase | Decode rises with more supply while latency stays acceptable | Flat decode with a growing prefill load: try the batched-token budget; low running counts: check upstream supply |
| `num_requests_running`, `num_requests_waiting` | Requests per engine, median and maximum over the same window | High waiting with busy engines suggests admission/server pressure | Idle engines, low queue and low KV during learner wait: inspect groups, tools, admission or pauses |
| `kv_cache_usage_perc`, `num_preemptions_total` | Native KV fraction 0–1; reset-aware preemption delta per window | High KV plus preemptions supports cache pressure | Low KV and zero preemptions weaken it; inspect CPU dispatch, kernels and communication instead |
| `time_to_first_token_seconds`, `request_queue_time_seconds`, `e2e_request_latency_seconds` | Histogram interval p95 and count, seconds, 5 min at expected burst | Lower queue/TTFT meets a latency goal | Decode gain with p95 above the limit is a failed latency tradeoff |
| `request_time_per_output_token_seconds` | Histogram interval, seconds per output token per completed request | Better sustained response speed | It is a request average, not a native inter-token histogram or aggregate fleet capacity |
| `request_success_total` by finish reason; bridge `request_outcome_count` | Reset-aware native completions and bridge outcomes per window | Complete useful answers without length/error growth | More truncated, abort/error or timed-out output reduces useful work; inspect the retained trajectory and first error |

Missing, stale or dropped telemetry leaves the hypothesis **unknown**. SkyRL `vllm/peak_gpu_cache_usage_perc` reports percentages; native KV uses fractions. Do not average engine p95s or difference gauges. Compute interval quantiles from bucket deltas per complete engine series, handling resets. [Typed source](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/skyrl-train/skyrl_train/inference_engines/vllm/stats.py).

Standalone Prometheus adds `vllm:` to these exported names. The snapshot API uses base names such as `generation_tokens`, `num_preemptions` and `request_success`; SkyRL adds `_total`. Check the exported schema.

<details>
<summary>One adjustment after identifying the limit</summary>

| Knob | When to try it | Performance and correctness boundary |
| --- | --- | --- |
| Client/admission concurrency | Server idle and useful groups scarce | Increase supply within the current policy-age contract. More workers alone may not expand the admission window |
| `max_num_seqs` | Waiting requests, memory headroom, a server cap reached | Can raise batch throughput and KV pressure together. Check tails/preemptions and whole-RL acceptance |
| `max_num_batched_tokens` / chunked prefill | Prefill displaces decode or TTFT is poor | Lower budget favors decode; larger favors prefill throughput. Test the task's latency and throughput together. [vLLM tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) |
| KV allocation / replicas / EP layout | Confirmed KV pressure | Raising memory utilization trades headroom for cache; another replica costs GPUs. Recheck startup, longest requests and publication peaks |
| Prefix caching | Repeated prompt prefixes | Historical Hero hits were 30–34%, but after-pause rate was 755.78 OFF versus 755.51 ON. The probe paused/resumed without a weight transfer. Keep off for the tested online path; reuse across actual publication is unmeasured. [Evidence](#serving-evidence) |
| CUDA graphs / eager mode | Startup cost dominates a short standalone job | Eager skips graph capture but its steady-state cost is workload-dependent. Hero's non-eager RL path is qualified; eager RL has no same-token pair |
| Attention / MoE backend, batch invariance | A supplied plateau without cache pressure, or numerical mismatch | Keep Hero's qualified FA2/replay/invariance package. A kernel change needs fixed-token numerical checks and online clipping/accepted-rate evidence |
| CPU scheduling / communication | Low KV, supplied workload, flat rate | Inspect event-loop/dispatch spans and rank or host traces. A CPU span includes waits; it does not prove GPU kernel utilization |

</details>

## Context and qualification

Hero's permanent export remains 4K. Probes raised HF, rotary and engine limits together and served through 256K on 64 H100/PP2 or 16 GB200/PP1. Sparse clients generated one token or selected 64-token continuations. Long-response capacity, RL correctness and multi-turn quality remain untested. The 32-H100/PP1 layout had roughly 208K KV slots per engine, too few for 256K. [Probe evidence](#serving-evidence).

Snowball longer-context rates are **unknown** here. Check peak input + output, longest requests, KV reservation and all position limits. Raising the configured maximum alone does not measure a longer-context workload.

### Serving evidence

<details>
<summary>Exact historical runs, sources and durable reports</summary>

The `s3://` reports require CoreWeave access. Check their configs/revisions before reproducing a run; fetch small reductions where possible. H100 results are recorded in the [26 September sizing comment](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5848726769).

| Result | Iris job; source revision | Report locator |
| --- | --- | --- |
| Snowball H1008 | `/romain/snowball-capacity-h100-01a0dca4-a8`; `b8c37cc90111795c3cccb355b0de163796280b8f` | `s3://marin-us-east-02a/marin/users/romain/hero-learning-01a0dca4/snowball-capacity-h100-a8/analysis/report-bb47d02f3c5f23179dbcf6b58aa58939f6aeb837e31ba0abe875d40df2f1e761.json`; SHA256 `bb47d02f3c5f23179dbcf6b58aa58939f6aeb837e31ba0abe875d40df2f1e761` (211 MiB raw bank) |
| Hero H10032 knee | `/romain/hero-h100-serve-32g-knee-01a0e406-a1`; `9ffb4919322b4bd67cb51b05c98332f506e644c1` | `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-serve-32g-knee-a1/report.json`; SHA256 `3093ac5959d84e2f2be00fd0e5d657481e184bd53b9b392e8e364f997d159dab` |
| Snowball GB20016 | `/romain/gb200-snowball-serve-16g-01a0e406-a1`; same source as knee | `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/gb200-snowball-serve-16g-a1/report.json`; SHA256 `f9b73d0d02244dbc07ea27aa36b6d7b1a2c7a1c0cc41212da5206fea1df0fe40` |
| Hero GB20016 cap256 | `/romain/hero-gb200-serve-16g-cap256-01a0e406-a1`; `dfdc80be6643f49af8d22ab9adda5319ae34befa` | `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/gb200-serve-16g-cap256-a1/report.json`; SHA256 `04b728f72b5dd8a67b19efb4b4c884d5c525da531685e3bcc2f4cd47012c289a` |
| Hero H100 prefix pair | `/romain/hero-h100-prefix-on-32g-01a0e406-a2` / `/romain/hero-h100-prefix-off-32g-01a0e406-a1`; source `577a5db16b3b971a96ba7254db2e89f60dd83cbb` | `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-prefix-on-32g-a2/report.json` and `h100-prefix-off-32g-a1/report.json` under the same parent |
| Hero GB200256K / H100256K | `/romain/hero-gb200-long-16g-256k-01a0e406-a1` / `/romain/hero-h100-long-64g-256k-pp2-01a0e406-a2` | `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/gb200-long-16g-256k-a1/report.json` / `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-long-64g-256k-pp2-a2/report.json` |

All short supplied curves use vLLM `25f0fc1aae71cccb6da65046cff13145512a6946`, TP1/PP1, memory utilization 0.90 and bank SHA256 `b15aeac7c3eb714fe45fc6b8e274342ad26f00970d84f2b7985b42c422b4e905`. Hero online H100 used 0.92. For implementation details, see [vLLM Grug](https://github.com/marin-community/vllm/blob/39e62869693c/vllm/model_executor/models/grugmoe.py) and [NVIDIA parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).

</details>
