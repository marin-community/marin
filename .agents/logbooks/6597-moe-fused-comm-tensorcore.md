---
topic: Fused communication and tensor-core expert-parallel MoE
issue: https://github.com/marin-community/marin/issues/6597
pull_request: https://github.com/marin-community/marin/pull/6841
description: Persistent fused Hopper kernels built from the JAX-native semantic source-push plan.
author: David Hall
---

# 6597 Fused Communication and Tensor-Core MoE Logbook

Architecture spec: `.agents/projects/20260710_moe_fused_comm_tensorcore.md`

Prior logbook: `.agents/logbooks/6597-moe-mgpu-forward.md`

## Fixed Target

```text
EP ranks:               8
tokens/rank:            32768
top-k:                  4
experts/rank:           32
hidden dimension:       2560
intermediate dimension: 1280
routing:                roughly balanced
target:                 250 useful TFLOP/s/rank aggregate forward + backward
```

## Hypothesis Queue

| ID | Hypothesis | Status | Decisive evidence |
| --- | --- | --- | --- |
| FUSED-MOE-001 | B256 sends feeding B64 WGMMA consumers overlap semantic permutation with W13. | Running | Target H100 fused, copy-only, compute-only, and split medians. |
| FUSED-MOE-002 | Streaming W2 output directly to source-owned return storage removes the dense semantic return tax. | Proposed | Fused W2+return+combine target median and parity. |
| FUSED-MOE-003 | Source-owned dcombine producers can feed W2 backward without replicated `dy`. | Proposed | No all-gather in lowered graph plus target timing and parity. |
| FUSED-MOE-004 | W13 backward can stream source x and return dX while accumulating dW13. | Proposed | Fused dX/dW13 target timing and parity. |

## 2026-07-10 FUSED-MOE-001 - Semantic fused permute + W13 implementation snapshot

Implemented the first persistent fused stage from the architecture spec. The
new semantic kernel lowers four B64 expert-row blocks into one B256 send
generation, partitions each generation into parallel B64xK256 producer tiles,
and runs Lane-lowered WGMMA consumers directly from the rolling destination
inbox. Producer completion, full publication, consumer completion, and slot
release use cumulative generation semaphore targets.

The package-private semantic API builds all route/chunk metadata with JAX and
returns source-padded expert-major W13 preactivation and validity. The benchmark
alias `semantic_permute_w13` includes the optimized fused mode, the existing
split inbox path, copy-only pack, and compute-only prepacked W13 baseline. No
new production tuning flags were added.

Local verification:

```text
semantic raw-token adapter tests:       7 passed
semantic decomposition harness tests: 32 passed
parallel fused W13 kernel tests:        4 passed
optimized fused benchmark smoke:       1 passed
combined focused suite:                43 passed
scoped pre-commit:                     passed
```

No GPU claim is attached to this snapshot. Next run: one H100x8 target-shape
compile/correctness/decomposition comparison of the optimized parallel producer,
the older raw-token fused adapter, and the split copy/compute baselines.
