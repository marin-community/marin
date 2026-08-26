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
