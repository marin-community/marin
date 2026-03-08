Training chunk-kernel focus for this session:

- Optimize end-to-end training performance first, specifically the chunked flash/train path (`chunk_gated_delta_rule` and its Pallas kernels).
- Do not spend an iteration on decode/recurrent-only improvements unless they also reduce the training-step hotspot cost in profile traces.

Scope priorities:

1) Reduce total train-path budget, not just inner `custom-call` time.
2) Track both kernel-local cost and control-flow cost (`while`, `conditional`) in every profile comparison.
3) Prefer structural train-path changes over scalar tuning and config-only tweaks.
4) Keep correctness and deployability for the target train config unless this run is explicitly marked as a probe/ablation.

Probe policy:

- Probe/ablation runs may intentionally relax triangular solve correctness to measure bottleneck sensitivity.
- Such runs must be labeled in the log as probe-only and must not be promoted as champion commits.
