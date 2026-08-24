# vLLM metric forwarding contract

Marin forwards a small, explicit subset of the Prometheus families exposed by
vLLM's `/metrics` endpoint. The checked-in contract is
[`vllm_metric_families.toml`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/inference/vllm_metric_families.toml).
It covers request volume, scheduler pressure, KV-cache use, preemption, and the
latencies used by Marin's inference dashboard.

The family names are the names returned by the Prometheus parser. Counter names
therefore omit the exposition suffix `_total`. `request_success` keeps every
`finished_reason` label. Each selected histogram keeps every bucket, its count,
its sum, and the parser's generated `_created` series. A family absent from one
vLLM build is simply absent from that scrape.

`request_time_per_output_token_seconds` is per-request time divided by that
request's output-token count. `inter_token_latency_seconds` measures the gaps
between successive output tokens. They answer different questions and remain
separate families.

## Optional additions

Both local and Iris submission accept `--vllm-metrics-config PATH`. The file is
TOML with the same one-field schema:

```toml
families = ["vllm:one_additional_family"]
```

The additions are unioned with the entire standard list; they cannot remove a
standard family. Marin rejects unreadable TOML, extra or missing top-level
fields, non-array or non-string values, and names without `vllm:` before it
starts or submits vLLM. The submitter carries only the sorted immutable family
names into `VllmEngineConfig`, so a worker never reopens the submitter's path.

No extra family is in the standard contract today. The remaining official
families are request-shape or scheduler histograms, feature-specific metrics
such as speculative decoding and KV connectors, or high-cardinality config
information. None is needed by the baseline Marin dashboard or an existing
alert. They can be added through the submission option when a concrete consumer
justifies their Finelog cost.

## Size and provenance

The contract was checked against three bounded sources:

- Marin's GPU pin is vLLM commit
  [`76c6650513a97507a485e142d48b4ce50c7fd0e0`](https://github.com/marin-community/vllm/blob/76c6650513a97507a485e142d48b4ce50c7fd0e0/vllm/v1/metrics/loggers.py).
  The TPU pin is
  [`3d26773be1d7aa7361a542943e3ef14f023d6f3a`](https://github.com/marin-community/vllm/blob/3d26773be1d7aa7361a542943e3ef14f023d6f3a/vllm/v1/metrics/loggers.py).
  Their `vllm/v1/metrics/loggers.py` files are byte-identical.
- Those definitions use `model_name` and `engine` on the selected families,
  plus `finished_reason` on `request_success` and `le` on histograms. The five
  explicit histogram bucket counts are 22 for TTFT, 19 for request TPOT, 19 for
  inter-token latency, 21 for queue time, and 21 for end-to-end latency.
- A generated exposition using those definitions and eight `engine` values
  parses to 1,024 base-family samples plus 104 `_created` samples: 1,128 total.
  This is complete for all twelve standard families.
- A bounded Finelog inventory from 2026-08-20 04:22 UTC through 2026-08-21
  00:51 UTC covered 10,397,394 rows, 4 runs, 14 jobs, 2 clusters, 9 nodes, and
  8 distinct `engine` values. It contained every expected selected counter and
  histogram component, including `_created`. This inventory is flattened
  post-admission telemetry, not a complete raw production scrape.

The post-selection safety limit is 2,048 samples. This names and covers the
observed eight-engine standard envelope with 920 samples, or 81.6%, of
headroom. Optional additions share that headroom. If the selected batch exceeds
the limit, Marin rejects the whole batch, reports zero enqueued samples and the
full selected count under Prometheus health's `sample_limit` drop reason, and
keeps serving and polling. The next under-limit scrape clears that health value
and resumes forwarding.

The upstream [production metrics guide](https://docs.vllm.ai/en/latest/usage/metrics/)
documents `/metrics`, the broader family inventory, and vLLM's metric
deprecation policy. No complete raw GPU or TPU scrape was available for this
change; the pinned definitions, parsed generated exposition, and bounded
Finelog inventory are the evidence used instead.
