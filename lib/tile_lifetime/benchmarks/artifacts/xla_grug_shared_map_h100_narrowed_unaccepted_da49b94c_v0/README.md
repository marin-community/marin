# Unaccepted narrowed twelve-call H100 Grug replay

This artifact records the only physical-H100 replay of the natural one-layer
Grug train-step boundary after generic demand-driven Contract row narrowing.
The executed Shuttle source revision is
`da49b94c359104690c2b8f98192300605cbd292e`; it contains the narrowing
checkpoint `1ee45b825d`.

All execution guards passed:

- all twelve selected custom-call targets occur exactly once in transformed
  HLO;
- every handler executed 35 times (one correctness call, four warmups, and 30
  measurements);
- all 30 measurements used a balanced counterbalanced order;
- generated and XLA outputs each produced one stable whole-tree hash;
- ordered-FP correctness passed over all 53 result leaves;
- runtime dependency and generated-source audits found no Torch or Triton
  dependency.

The result remains unaccepted for performance. The generated median is
`0.647561999 ms` versus `0.521619004 ms` for XLA, or `1.241446333x`. This
exceeds the `1.20x` ceiling. Maximum absolute error is `9.760260582e-7`, mean
absolute error is `7.977501935e-11`, and 38 of 53 leaves are bitwise equal.

Compared with the prior full-domain twelve-call replay at `2732ef51a9`, row
narrowing reduces generated median latency by `0.010726002 ms` (`1.63%`) and
the generated/XLA ratio from `1.247987838x` to `1.241446333x`. The absolute
whole-step gap falls by `0.004865505 ms`. This is useful but insufficient; it
does not justify accepting the twelve-call boundary.

The benchmark was invoked once and made no replay retry. One CPU and one H100
were requested at batch priority; Kubernetes normalized the CPU request to its
four-CPU minimum. The physical device was an NVIDIA H100 80GB HBM3 with compute
capability 9.0, driver 595.71.05, and a 700 W power limit. Clocks were not
pinned. The allocation was released immediately after copying the raw result,
and both local session status and pod lookup verified it inactive.

`execution-evidence.json.gz` preserves every timing pair, launch order,
whole-tree hash, per-leaf hash, target occurrence, handler count, and semantic
comparison. `summary.json.gz` preserves the full benchmark result. The eleven
`generated_*.cu` files plus `generated_attention_backward.cu` are the exact
generated handler sources. Build caches, cubins, DSOs, duplicate handler
copies, and attention AOT intermediates are excluded.

The exact command was:

```bash
/app/.venv/bin/python \
  lib/tile_lifetime/benchmarks/xla_grug_routed_combined_gpu_custom_call.py \
  --nvcc /app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --architecture sm_90a \
  --repository /tmp/shuttle-da49b94c35 \
  --artifact-directory /tmp/shuttle-narrowed-da49-raw \
  --output /tmp/shuttle-narrowed-da49-raw/summary.json \
  --composition-mode shared_map_xla_remainder \
  --warmup 4 \
  --repeats 30
```
