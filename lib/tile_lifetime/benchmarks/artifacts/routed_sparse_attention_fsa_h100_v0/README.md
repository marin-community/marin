# FSA KV-major adapter H100 checkpoint

Date: 2026-08-07

## Shuttle source identity

The benchmark ran from branch `research/shuttle-routed-sparse-attention`, based
on commit `9ba3888cb0f91e2cf54f2a182927f13e769be2c6`. The adapter was uncommitted and
is not contained in that commit. Consequently, the `shuttle_revision` field in
the raw JSON identifies only the branch base.

The exact executed `lib/tile_lifetime/benchmarks/h100_fsa_kv_major.py` has
SHA256 `bdcbf4b2f12c00c2047459b69ed4aa28b1df1f5c64620450e67cfb522d08ec45`.
It was copied unchanged to the H100 holder. `manifest.json` records that hash,
the uncommitted status, and SHA256/size pairs for both raw JSON files and the
pristine-source failure log.

This checkpoint runs Shuttle's generic block-shared `RelationPlan` through the
public selected-attention entry point from Flash Sparse Attention (FSA) at
`7ff144fd7ff485dc4220d439f31cc1708b64fef3`. The adapter expands each
query-block/KV-block edge across its query tokens and eight KV heads to produce
FSA's int32 `[Hkv, T, topk]` input. FSA then rebuilds its private block-to-token
orientation inside every timed call. It is therefore a KV-major expert oracle,
not an execution of Shuttle's grouped offsets or inverse map.

## Upstream compatibility patch

The pristine pinned source fails during Triton 3.4 JIT compilation because
`reduce_kernel` constructs `lse_ptrs` as an accidental singleton tuple:

```text
AttributeError: 'tuple_type' object has no attribute 'is_ptr'
```

The executable runs use one compatibility edit in the ephemeral FSA checkout:

```diff
-    lse_ptrs = (lse_ptr + pid_q_j * stride_lse_n,)
+    lse_ptrs = lse_ptr + pid_q_j * stride_lse_n
```

This removes the tuple wrapper without changing pointer arithmetic, the
physical schedule, or attention math. Each JSON file records the pinned Git
head, dirty status, and complete diff. The pristine traceback is preserved in
`fsa_2k_pristine_failure.log`.

## Environment

- GPU: one NVIDIA H100 80GB HBM3 on an eight-GPU holder; only GPU 0 used.
- Driver: 595.71.05.
- PyTorch: 2.8.0+cu128.
- Triton: 3.4.0.
- Shape: BF16 causal GQA, 32 query heads, 8 KV heads, head/value dimension 128.
- Relation: deterministic historical blocks including block zero and the
  current block.
- Clocks: cluster default, unpinned; per-run telemetry is in the JSON.

The environment intentionally omits the full FSA requirements, FlashAttention,
and CUDA extension builds. Only PyTorch, Triton, NumPy, Einops, and Shuttle's
declared JAX dependency were installed. The image has no `nvcc`; FSA uses
Triton JIT and does not require it.

## 2K correctness smoke

At sequence length 2,048, block size 128, and top-k 8, the generic relation has
100 block edges. The combined public FSA call has a 20-sample selected-work
throughput of 2.193 TFLOP/s. Against an independent source-ordered FP32
selected-attention reference over eight query blocks:

- maximum absolute error: 0.0185196;
- mean absolute error: 0.000180095;
- p99 absolute error: 0.00122058;
- no NaN or infinity;
- first and final outputs are bitwise identical.

## 16K cross-oracle run

At sequence length 16,384, block size 128, and top-k 8, the raw Boolean relation
hash is
`b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`,
identical to the existing Seer query-major artifact. The relation has 996 block
edges and 267,361,714,176 selected QK+PV FLOPs.

The combined FSA public call, including its private relation inversion,
allocations, partial computation, and reduction, has a 30-sample median of
12.5392 ms (12.3622–13.0573 ms), or 21.322 selected-work TFLOP/s. The sampled
reference comparison reports maximum/mean/p99 absolute errors of
0.0207922/0.000164022/0.00120181, no NaN or infinity, and bitwise-identical
repeated output hash
`0d711cf008f91f857a2241737ebb122b2336c5c1d128e885f2db3b6b47ae53f5`.

Source-visible FSA buffers account for 111,225,856 bytes of partial state and
statistics plus 20,865,024 bytes of internal inverse-index storage. The measured
peak allocator increment is 431,091,712 bytes. Shuttle's current eager physical
plan declares 2,121,400,320 bytes because it materializes one FP32
`(max, denominator, weighted value)` state for every edge, query token, and
query head. This is a schedule-search pressure point: partial states must be
consumed in bounded head/group waves rather than allocated for all edges at
once.

The generic relation-plan median is 0.6411 ms. The FSA adapter median is
0.6226 ms, but its first four raw samples are 0.93–1.07 seconds in the holder;
the JSON preserves these cold/outlier samples rather than hiding them.

## Semantic seam

The adapter is exact for this block-shared causal workload, subject to FSA's
ascending-block merge order. It deliberately rejects a relation whose selected
slots are not strictly increasing instead of silently changing Shuttle's
source-order floating-point contract.

FSA cannot accept Shuttle's destination-major edge arrays. Its public call also
does not expose separate inverse-plan, QK/PV, state-materialization, or merge
timings. Reusing FSA as a true backend primitive would require extracting an
entry point after block-to-token inversion that accepts compact selected-token
ranges and caller-provided bounded partial buffers. Copying FSA's complete
global schedule into Shuttle would not test the compiler abstraction.
