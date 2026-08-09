# Generated row-normalization backward on GB200

This checkpoint evaluates the generic axis-Fold lowering for RMS and centered
LayerNorm backward at Shuttle revision
`5509e444487df21aceb64c2266419ba92c3bff8e`. No schedule candidate was tuned.

Primary shape: 2,048 rows, hidden size 4,096, 256 threads, BF16 standardized
input and feature scale, FP32 projected input/inverse scale and FP32 outputs.
Each primary capture has 30 counterbalanced raw samples with 10 iterations per
sample after 10 warmups.

| Statistic | Generated median | torch.compile median | Ratio |
| --- | ---: | ---: | ---: |
| RMS | 0.087160 ms | 0.054582 ms | 1.597x |
| Centered LayerNorm | 0.088042 ms | 0.053051 ms | 1.660x |

Both generated variants are deterministic and numerically close to the
reference. Maximum input-cotangent error is 1.90735e-6 for both; maximum
feature-scale-cotangent error is 3.05176e-5. Both fail the frozen 1.2x
performance threshold, so this is a correctness proof of life and a measured
backend gap, not a performance acceptance result.

The measured feature-scale Fold assigns one block to each feature, producing
hidden-size-strided row loads. Commit `82bedcd6b4` adds a generic 32-feature
coalesced schedule; that follow-up is not part of this frozen artifact.

Commands:

```text
TORCH_CUDA_ARCH_LIST=10.0a uv run --frozen --package marin-tile-lifetime \
  python lib/tile_lifetime/benchmarks/h100_generated_row_normalization_backward.py \
  --rows 2048 --hidden 4096 --threads 256 --statistic rms \
  --warmups 10 --repeats 30 --iterations 10 \
  --source-directory /tmp/shuttle-row-norm-rms-source \
  --json-output /tmp/shuttle-row-norm-rms-gb200.json \
  --shuttle-revision 5509e444487df21aceb64c2266419ba92c3bff8e

TORCH_CUDA_ARCH_LIST=10.0a uv run --frozen --package marin-tile-lifetime \
  python lib/tile_lifetime/benchmarks/h100_generated_row_normalization_backward.py \
  --rows 2048 --hidden 4096 --threads 256 --statistic layer \
  --warmups 10 --repeats 30 --iterations 10 \
  --source-directory /tmp/shuttle-row-norm-layer-source \
  --json-output /tmp/shuttle-row-norm-layer-gb200.json \
  --shuttle-revision 5509e444487df21aceb64c2266419ba92c3bff8e
```

Environment: one low-priority NVIDIA GB200, driver 595.71.05, Torch
2.10.0+cu130, CUDA 13.0, 1200 W power limit, observed 1950 MHz SM and 3996 MHz
memory clocks. The JSON files preserve raw samples, order, deterministic
hashes, errors, environment, and exact revision.
