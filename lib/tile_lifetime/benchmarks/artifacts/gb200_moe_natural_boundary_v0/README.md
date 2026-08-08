# GB200 natural-boundary MoE clean synthesis

This artifact closes the distributed BF16 MoE clean-synthesis row for one
four-GB200 configuration. The accepted path starts from ordinary JAX-exported
StableHLO, executes router logits and top-k at runtime, builds the receiver
`RelationPlan` on device, uses DeepEP only for payload dispatch, executes
generic segmented contractions and generated Maps, returns payloads with
`all_to_all_single`, and performs a generated deterministic Fold and shared
Map. It does not use a saved route, MoK forward, or DeepEP semantic combine.

## Configuration

```text
GPU:               4 x NVIDIA GB200
tokens per rank:   2048
hidden size:       7168
intermediate size: 3072
global experts:    384
top-k:             6
dtype:             BF16, with FP32 route weights and Fold accumulation
DeepEP SMs:        56
W13 layout:        concatenated [E, 2I, K]
clock policy:      cluster default, unpinned
observed clocks:   1950 MHz SM, 3996 MHz memory
power limit:       1200 W per GPU
```

The matched MoK oracle executes the identical router Contract, top-k, and
normalized route-weight Maps/Fold before MoK schedule construction and the
complete MoK BF16 forward.

## Accepted result

Two independent runs contain 30 steady-state rank-maximum samples per
implementation. Run 1 times Shuttle before the oracle; run 2 reverses that
order. The acceptance ratio pools all 60 samples.

| Capture | Order | Shuttle | MoK | Ratio |
|---|---|---:|---:|---:|
| `x2-final-run1` | Shuttle first | 4.126384 ms | 3.645056 ms | 1.132050x |
| `x2-final-run2` | Oracle first | 4.140336 ms | 3.642048 ms | 1.136815x |
| Pooled | Counterbalanced | 4.137120 ms | 3.645056 ms | 1.134995x |

The pooled result passes the 1.20x completion target and misses the 1.10x
stretch target. The remaining pooled latency gap is 0.492064 ms.

## Candidate history

All captures remain under `raw/`; none were deleted after target selection.

| Capture | Change | Shuttle / MoK |
|---|---|---:|
| `smoke` | Serial deterministic receiver RelationPlan | 2.032554x |
| `optimized-smoke` | Warp-prefix receiver RelationPlan, 56 communication SMs | 1.210696x |
| `candidate-sms48` | Same candidate with 48 communication SMs | 1.215089x |
| `final-run1` + `final-run2` | 56 SMs, scalar source-ordered Folds, 60 pooled samples | 1.201725x |
| `x2-smoke` | BF16x2 generic Fold smoke | 1.149247x |
| `x2-final-run1` + `x2-final-run2` | BF16x2 generic Folds, 60 pooled samples | 1.134995x |

The accepted optimization vectorizes two generic deterministic Fold kernels
over BF16 pairs. Each component still uses explicit FP32 round-to-nearest
multiply and add in route-slot and rank order. It reduced pooled Shuttle
latency by 0.227104 ms relative to the scalar-Fold captures.

## Correctness and synthesis boundary

Across both accepted captures and all four ranks:

- device-generated group counts, padded rows, edge rows, and edge weights match
  the independent relation exactly;
- relation overflow is zero;
- repeated Shuttle outputs are bitwise equal;
- Shuttle versus MoK maximum absolute error is `0.0001220703125`;
- the largest per-rank mean absolute error is `2.667012722668005e-06`;
- the independent small semantic reference passes.

The generated receiver RelationPlan, route-slot Fold, and rank Fold contain no
CUDA atomic operation. The grouped GEMM primitive may use queue or semaphore
control, but every semantic output tile has one owner. DeepEP may use internal
readiness counters for transport; it does not accumulate semantic values or
choose their order. Reverse transport is a payload-only
`all_to_all_single`. The generated rank Fold combines owner ranks in fixed
order and then applies the shared-output Map.

The only expert semantic kernel in the artifact is complete MoK forward, and
it is loaded by the oracle phase only. The accepted Shuttle call graph lists no
expert/oracle-only semantic kernel.

## Revisions

```text
Shuttle base:       4fba36752bdbfd28ad9a0ea8dee121bb382b21c9
MoK:                3e1cf43ab93ad040afed52a45ab03cb490ffe4be
ThunderKittens:     1c3920d993404dd49a6d4c7267ea11d583bd5c68
DeepEP:             7febc6e25660af0f54d95dd781ecdcd62265ecca
CUDA toolkit:       13.0, V13.0.88
NVIDIA driver:      595.71.05
PyTorch:            2.10.0+cu130
NCCL:               2.28.9
```

`raw/deepep-torch-intranode.patch` records the local build-only DeepEP changes.
`source/` contains the exact benchmark and generated primitive source used by
the accepted captures. `fixtures/` contains the StableHLO artifact and the two
sets of per-rank semantic fixtures. `SHA256SUMS` covers every artifact file
except itself. Repository linting removed trailing spaces from
`raw/gpu-final.txt` after capture; the telemetry fields and line order are
unchanged, and `SHA256SUMS` records the normalized bytes.
