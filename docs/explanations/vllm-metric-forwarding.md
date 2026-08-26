# vLLM metric forwarding contract

Marin forwards a small, explicit subset of the Prometheus families exposed by
vLLM's `/metrics` endpoint. The checked-in contract is
[`vllm_metric_families.toml`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/inference/vllm_metric_families.toml).
It covers request volume, scheduler pressure, KV-cache use, preemption, and the
latencies used by Marin's inference dashboard.

## Selection

The standard list preserves signals consumed by the current inference dashboard,
then checks their forwarding cost. These are the primary names emitted by
Marin's pinned vLLM builds; query-side compatibility aliases do not expand the
forwarding contract.

The representative costs below assume eight `(model_name, engine)` label sets
and five `finished_reason` values. A histogram contributes
`(finite buckets + 1 + count + sum + _created) * 8` samples, where `1` is the
implicit `+Inf` bucket. A counter contributes its value and `_created` sample per
label set. A gauge contributes one sample per label set. These are sample counts
for one scrape, not measured bytes or storage cost.

| Family | Dashboard signal | Representative samples per scrape |
| --- | --- | ---: |
| `vllm:e2e_request_latency_seconds` | End-to-end latency mean and quantiles | 200 |
| `vllm:generation_tokens` | Generated-token rate and window total | 16 |
| `vllm:inter_token_latency_seconds` | Inter-token latency mean and quantiles | 184 |
| `vllm:kv_cache_usage_perc` | KV-cache use over time, average, and peak | 8 |
| `vllm:num_preemptions` | Preemption window total | 16 |
| `vllm:num_requests_running` | Running requests over time, average, and peak | 8 |
| `vllm:num_requests_waiting` | Waiting requests over time, average, and peak | 8 |
| `vllm:prompt_tokens` | Prompt-token rate and window total | 16 |
| `vllm:request_queue_time_seconds` | Queue latency mean and quantiles | 200 |
| `vllm:request_success` | Request outcomes by finish reason | 80 |
| `vllm:request_time_per_output_token_seconds` | Request TPOT mean and quantiles | 184 |
| `vllm:time_to_first_token_seconds` | TTFT mean and quantiles | 208 |

The five histograms contribute 976 samples, `request_success` contributes 80,
the other three counters contribute 48, and the three gauges contribute 24:
1,128 samples in total. The counts already include all 104 `_created` samples:
40 from histograms, 40 from `request_success`, and 24 from the other counters.

The family names are the names returned by the Prometheus parser. Counter names
therefore omit the exposition suffix `_total`. `request_success` keeps every
`finished_reason` label. Each selected histogram keeps every bucket, its count,
its sum, and the parser's generated `_created` series. A family absent from one
vLLM build is simply absent from that scrape.

`request_time_per_output_token_seconds` measures per-request time divided by
output tokens; `inter_token_latency_seconds` measures gaps between tokens. They
remain separate families.

## Optional additions

Both local and Iris submission accept `--vllm-metrics-config PATH`. The file is
TOML with the same one-field schema:

```toml
families = ["vllm:one_additional_family"]
```

The additions are unioned with the entire standard list; they cannot remove a
standard family. Marin validates the file before starting or submitting vLLM,
then passes the normalized family names—not the submitter's path—to the worker.

Other vLLM families are feature-specific, high-cardinality, or have no baseline
dashboard or alert consumer. Add them when a concrete consumer justifies their
Finelog cost.

## Safety limit

Marin's pinned [GPU](https://github.com/marin-community/vllm/blob/76c6650513a97507a485e142d48b4ce50c7fd0e0/vllm/v1/metrics/loggers.py)
and [TPU](https://github.com/marin-community/vllm/blob/3d26773be1d7aa7361a542943e3ef14f023d6f3a/vllm/v1/metrics/loggers.py)
metric definitions are byte-identical. A representative exposition using their
labels and histogram buckets, five finish reasons, and eight `engine` values
contains 1,128 samples across the twelve standard families, including 104
parser-created `_created` samples.

The post-selection limit is 2,048 samples, leaving 920 samples of headroom for
that envelope and any optional additions. An oversized scrape is rejected as a
whole and reported through Prometheus `sample_limit` health while inference
keeps serving. Forwarding resumes after the next under-limit scrape.

The upstream [production metrics guide](https://docs.vllm.ai/en/latest/usage/metrics/)
documents the broader family inventory and deprecation policy. No complete raw
GPU or TPU scrape was retained for sizing; the evidence is the pinned
definitions, representative parsed exposition, and bounded post-admission
Finelog inventory.
