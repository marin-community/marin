# Seer Baseline Delta Analysis

Date: 2026-08-07

## Result

At the frozen 16K workload, the generated KV-major slot-wave schedule is
1.682 times the Seer query-major kernel:

| Implementation | Latency | Selected TFLOP/s |
|---|---:|---:|
| Seer query-major | 2.388208 ms | 111.951 |
| Shuttle KV-major slot waves | 4.017344 ms | 66.552 |
| Delta | 1.629136 ms | |

Both runs use the same 996-edge relation and report the same
267,361,714,176 selected QK/PV FLOPs. Both record NVIDIA driver 595.71.05 and
1830 MHz active SM clocks. The software stacks differ: Shuttle uses Torch
2.8/Triton 3.4 and Seer uses Torch 2.7.1/Triton 3.3.1.

The gap is principally explained by global online-state traffic and repeated Q
loads, not relation metadata or raw launch overhead.

## State traffic

The slot-wave implementation materializes FP32 state for every query row and
head:

```text
(row_max, row_sum_exp, weighted_value[128])
```

Every selected edge reads and writes that state. For sequence 16384, 32 query
heads, head dimension 128, query/KV blocks 128, and 996 selected edges:

```text
per-row/head state              520 B
eight-wave state read/write     4,242,800,640 B
state initialization              272,629,760 B
final-state read                  270,532,608 B
BF16 output write                 134,217,728 B
minimum state lifecycle         4,920,180,736 B
```

The 272.6-MB state is much larger than H100 L2, so launch boundaries force most
of this state through HBM rather than preserving it on chip.

## Repeated query traffic

The KV-major slot-wave kernel reloads Q for every selected edge:

```text
edge-wise Q reads               1,044,381,696 B
one query-major Q read            134,217,728 B
extra Q traffic                   910,163,968 B
```

The minimum state lifecycle plus extra Q traffic is about 5.83 GB. At
2.5--3.35 TB/s effective bandwidth, this costs roughly 1.74--2.33 ms. That is
already the scale of the measured 1.629-ms gap before accounting for smaller
M32 work units or other implementation differences.

## Kernel structure

One timed Shuttle call performs approximately:

```text
three FP32 state fills
eight slot-wave update launches
one finalize launch
```

The update grid contains 127,488 Triton programs and uses M32 QK/PV work units.
Seer uses query-major M128 work units and can retain Q and online-softmax state
while traversing selected KV blocks. Launch overhead is real, but the more
important effect of the launch decomposition is forced state spill/reload.

## What does not explain the gap

Relation metadata is small. Shuttle performs roughly two scalar edge-index
loads per program. Seer instead checks a dense causal block mask. The operation
counts are comparable and negligible next to contraction and state traffic.

Neither measured implementation proves cross-query shared-memory KV reuse.
Shuttle orders work by KV destination but each Triton program loads its own K/V
tile. Seer is query-major and may receive only ordinary cache reuse. The
canonical relation was already KV-monotone, so the prior sort/no-sort comparison
did not test locality.

## GQA comparison caveat

Seer expands K/V from 8 to 32 heads outside the timed region. The artifact
records 52.05 ms for the cold expansion and 201,326,592 additional resident
bytes at 16K. Shuttle performs native GQA head indexing in the timed kernel.

The 2.388-ms Seer result is therefore a useful physical-kernel target, but not
an end-to-end native-GQA result.

## Decision

Do not spend another iteration tuning M32/M64 tile sizes. The current schedule
has exposed the abstraction's cost clearly: destination-oriented ordering does
not help when online state is materialized at every edge and K/V is not actually
shared.

The next sparse-attention experiment, if revisited, should combine:

1. a deliberately non-monotone relation that distinguishes source and
   destination traversal;
2. real cluster/shared-memory staging of one K/V block across incident query
   work; and
3. a fused state lifetime that processes multiple edges before writing FP32
   partial state.

That experiment would test relation reorientation and physical KV reuse. A
further tile-size sweep would not.

## Evidence

- Seer artifact: `lib/tile_lifetime/benchmarks/artifacts/routed_sparse_attention_h100_v0/seer_16k.json`
- Shuttle artifact: `lib/tile_lifetime/benchmarks/artifacts/routed_sparse_attention_h100_v0/slot_waves/slot-wave-16k-b128-k8-final.json`
- Shuttle kernel: `lib/tile_lifetime/benchmarks/triton_kv_major_slot_waves.py`
- Seer adapter: `lib/tile_lifetime/benchmarks/artifacts/routed_sparse_attention_h100_v0/benchmark_seer_query_major.py`
