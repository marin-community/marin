# Deterministic KV-major slot-wave candidate

Date: 2026-08-07

This checkpoint records Shuttle's first bounded KV-major candidate that executes
the generic block-shared `RelationPlan` directly. The physical schedule advances
one selected-slot wave at a time. Within a wave, edges are grouped by KV block,
and each Triton program is the sole writer of one FP32 online state tile for a
query block, query head, and query-row range. There are no atomics and no
per-edge partial-state tensors. Final output is BF16.

## Environment and workload

- GPU: one NVIDIA H100 80GB HBM3 on an eight-GPU holder; only GPU 0 used.
- Driver: 595.71.05.
- Maximum SM clock and power limit: 1980 MHz and 700 W; clocks were not pinned.
- PyTorch: 2.8.0+cu128.
- PyTorch CUDA runtime: 12.8.
- Triton: 3.4.0.
- Shape: BF16 causal GQA, `Hq=32`, `Hkv=8`, `D=128`, block size 128,
  selected-block limit 8.
- Relation: deterministic historical blocks including block zero and the
  current block.

The 16K Boolean relation SHA-256 is
`b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`,
identical to the Seer and FSA checkpoints. Each final JSON includes raw CUDA
event samples, Q/K/V and output hashes, sampled clock telemetry, correctness,
planning times, and the complete physical-plan dump.

The implementation was uncommitted on branch
`research/shuttle-routed-sparse-attention` over base commit
`9ba3888cb0f91e2cf54f2a182927f13e769be2c6`; the base commit does not contain
the benchmark files. `manifest.json` records exact SHA-256 identities for the
three executed scripts at each source stage and every raw result JSON.
`source_evolution.patch` records the source-order and M64 changes made after the
selected M16/M32 runs, so the earlier sources can be reconstructed rather than
incorrectly attributed to the base revision.

## Candidate selection

| Sequence | Query tile / warps | Edge order | Median (ms) | Range (ms) | Selected TFLOP/s | Decision |
|---:|---:|---|---:|---:|---:|---|
| 2K | 16 / 4 | KV-major | 0.5691 | 0.5672–0.6460 | 47.17 | Rejected |
| 2K | 32 / 4 | KV-major | 0.5020 | 0.4984–0.5174 | 53.47 | Selected |
| 2K | 64 / 8 | KV-major | 0.6601 | 0.6572–0.6723 | 40.67 | Rejected |
| 16K | 32 / 4 | KV-major | 4.0173 | 4.0119–4.0278 | 66.55 | Selected |
| 16K | 32 / 4 | source | 4.0189 | 4.0136–4.2785 | 66.53 | No-op ablation |

The final 2K and 16K records use 30 measured repetitions after ten warm-ups.
The 16K candidate computes 267,361,714,176 selected QK/PV FLOPs and holds
272,629,760 bytes of global FP32 online state. It materializes neither
sequence-squared scores nor edge-local partial attention states.

The requested source-order/no-sort ablation does not isolate destination
locality for this fixture. For every selected slot, the canonical relation is
already monotonically ordered by KV block in query/source order. The source and
KV-major schedules therefore have identical edge arrays and output hashes; the
0.04% median difference is measurement noise rather than evidence for sorting.

## Correctness and determinism

At 16K, eight sampled query blocks match an independent source-ordered FP32
selected-attention reference with maximum/mean/p99 absolute errors of
0.0078306/0.0001243/0.0008653. There are no NaNs or infinities. First-run and
timed-run BF16 outputs have the same SHA-256
`7fee4b9c61ea72736f203fad5ab212f1f31d9178f750bc967f8c8db2eeb66917`.

At 2K, the selected M32 candidate reports maximum/mean/p99 errors of
0.0079474/0.0001363/0.0008793 and is also bitwise deterministic. The M64
candidate is numerically valid but has a different deterministic output hash
because its row tiling changes finite-precision reduction order.

## Cross-oracle interpretation

At 16K, this candidate is 1.68x slower than the query-major Seer smoke
(2.3882 ms) but 3.12x faster than the FSA public-call adapter (12.5392 ms).
Those are architectural references rather than perfectly matched kernel
comparisons: Seer scans a dense causal block mask and expands GQA outside the
timed kernel, while FSA rebuilds its own inverse relation and materializes
partial state internally.

The result validates deterministic bounded state consumption and direct use of
Shuttle's relation plan. It does not yet validate actual cross-query KV reuse:
grouped edges remain independent CTAs, so grouping can improve cache locality
but does not stage one KV tile for several query CTAs.

## Current physical limitations

- Eight slot waves mean eight kernel launches and global synchronization points.
- The full FP32 online state is stored globally between waves.
- KV grouping does not yet use a CTA cluster or explicit shared staging.
- State allocation/initialization and finalization are included in the region
  timing, but relation planning is excluded.
- Only BF16 causal self-attention with equal Q/K/V feature dimensions and head
  dimensions 64 or 128 is supported.

## Reproduction

From the repository root on an H100 environment with PyTorch and Triton:

```bash
export PYTHONPATH="$PWD/lib/tile_lifetime/src:$PWD/lib/tile_lifetime/benchmarks"
python lib/tile_lifetime/benchmarks/h100_kv_major_slot_waves.py \
  --gpu --sequence-length 2048 --block-size 128 --selected-blocks 8 \
  --query-tile-size 32 --warmups 10 --repeats 30 --correctness-blocks 8
python lib/tile_lifetime/benchmarks/h100_kv_major_slot_waves.py \
  --gpu --sequence-length 16384 --block-size 128 --selected-blocks 8 \
  --query-tile-size 32 --warmups 10 --repeats 30 --correctness-blocks 8
```
