# Issue5815 FA4 THD Attention: Research Logbook

## Scope
- Goal: Determine whether upstream-shaped FlashAttention-4 THD/varlen attention is materially faster than Grug's current dense BSHD segmented FA4/CuTe path for the d5120 attention shape.
- Primary metric(s): fwd+bwd median wall time for q `[B,S,Hq,D]`, k/v `[B,S,Hkv,D]`.
- Constraints: Only touch Grug attention backend probe files. Do not include private cluster/provider details in permanent logs or GitHub.
- Issue: https://github.com/marin-community/marin/issues/6039

## Baseline
- Date: 2026-05-29
- Code refs:
  - `lib/levanter/src/levanter/grug/attention/_fa4_cute.py`
  - `lib/levanter/src/levanter/grug/attention/_core.py`
  - `lib/levanter/src/levanter/grug/attention/_fa4_thd.py`
  - `scratch/issue5815_attention_backend_probe.py`
- Baseline numbers: prior issue5815 attention probe reported `jax_cudnn_core` median `0.00165s` and `fa4_cute_core` median `0.02220s` for the d5120 q/k/v shape.

## Experiment Log
### 2026-05-29 14:57 - FA4 THD varlen dynamic segments
- Hypothesis: Upstream FA4 CuTe varlen, using THD tensors and `cu_seqlens` derived from packed segment IDs, may be faster than the local BSHD segmented FA4/CuTe port.
- Command:
  - `python scratch/issue5815_attention_backend_probe.py --warmup 3 --iters 10 --segment-len 1024`
  - `python scratch/issue5815_attention_backend_probe.py --warmup 3 --iters 10 --segment-len 4096`
- Config: B200, JAX `0.10.0`, PyTorch `2.12.0+cu130`, `flash-attn-4==4.0.0b15`; shape `B=1 S=4096 Hq=40 Hkv=8 D=128`.
- Result:
  - `segment_len=1024`: `jax_cudnn_core=0.000753s`, `fa4_cute_core=0.001085s`, `fa4_thd_core=0.001570s`.
  - `segment_len=4096`: `jax_cudnn_core=0.000752s`, `fa4_cute_core=0.003082s`, `fa4_thd_core=0.001869s`.
- Interpretation: THD/varlen is promising in absolute terms (`<=0.005s`) and beats current segmented FA4 for the full-sequence case, but it is not faster for the dynamic multi-segment case tested here. cuDNN remains faster than both.
- Next action: Do not pursue THD as an issue5815 fix unless the production segment distribution is closer to the full-sequence case or the JAX segmented FA4 path regresses back near `0.022s`.

### 2026-05-29 15:04 - FA4 THD numerics spot-check
- Hypothesis: The THD reshape path should match the JAX reference for contiguous packed segment IDs up to expected bf16/flash-attention differences.
- Command: Ad hoc reference comparison for `S=1024 Hq=40 Hkv=8 D=128 segment_len=256`, with `B=1` and `B=2`.
- Config: Same upstream FA4 THD API as the timing probe; compare output and q/k/v gradients against `reference_attention(..., logits_dtype=float32)`.
- Result:
  - `B=1`: output max abs `0.015625`, mean abs `0.000509`; `dq` max abs `0.023438`, mean abs `0.000524`; `dk` max abs `0.03125`, mean abs `0.001235`; `dv` max abs `0.0625`, mean abs `0.001245`.
  - `B=2`: output max abs `0.03125`, mean abs `0.000509`; `dq` max abs `0.03125`, mean abs `0.000525`; `dk` max abs `0.046875`, mean abs `0.001242`; `dv` max abs `0.0625`, mean abs `0.001247`.
- Interpretation: Numerics are within expected bf16 attention-kernel differences for this spot-check. Relative-error maxima are large because many reference elements are near zero, so absolute error is the meaningful diagnostic here.
- Next action: If this becomes production-relevant, add a focused test comparing THD to reference/current FA4 on small segmented examples.

### 2026-05-29 15:13 - Batch and segment-count sweep with cuDNN segment packing
- Hypothesis: cuDNN may be the best path for packed rows if each contiguous segment is reshaped into its own batch row before causal attention.
- Command: `python scratch/issue5815_attention_backend_probe.py --warmup 3 --iters 10 --batches 1,2,4 --segment-len {4096,2048,1024}`
- Config: Shape per original row `S=4096 Hq=40 Hkv=8 D=128`, bf16. `jax_cudnn_core` packs contiguous segments into `[B * num_segments, segment_len, H, D]`; current FA4/CuTe and THD consume packed segment IDs.
- Result:
  - `segment_len=4096`: `B=1` cuDNN `0.000747s`, FA4/CuTe `0.003076s`, THD `0.001861s`; `B=2` cuDNN `0.001291s`, FA4/CuTe `0.005662s`, THD `0.002859s`; `B=4` cuDNN `0.002412s`, FA4/CuTe `0.010799s`, THD `0.004767s`.
  - `segment_len=2048`: `B=1` cuDNN `0.000539s`, FA4/CuTe `0.001749s`, THD `0.001683s`; `B=2` cuDNN `0.000879s`, FA4/CuTe `0.003161s`, THD `0.002471s`; `B=4` cuDNN `0.001571s`, FA4/CuTe `0.005975s`, THD `0.004051s`.
  - `segment_len=1024`: `B=1` cuDNN `0.000441s`, FA4/CuTe `0.001091s`, THD `0.001588s`; `B=2` cuDNN `0.000694s`, FA4/CuTe `0.001913s`, THD `0.002271s`; `B=4` cuDNN `0.001191s`, FA4/CuTe `0.003562s`, THD `0.003648s`.
- Interpretation: cuDNN with segment packing is fastest across this sweep. THD is better than current FA4/CuTe for one full segment per row and roughly ties or beats current FA4/CuTe at two segments per row, but loses at four 1024-token segments per row. For issue5815, packed-cuDNN looks more promising than upstream THD.
- Next action: Investigate whether the model path can reshape contiguous packed segments into separate batch rows before cuDNN attention without disturbing surrounding layout/sharding.

### 2026-05-29 15:26 - Dynamic THD/cu-seqlens segment-length sweep
- Hypothesis: For unequal contiguous segment lengths, the upstream THD `cu_seqlens` path should avoid the padding overhead and may beat the local segmented FA4/CuTe path.
- Command: `python scratch/issue5815_attention_backend_probe.py --warmup 3 --iters 10 --batches 1,2,4 --segments-per-row {2,4}`
- Config: Shape per original row `S=4096 Hq=40 Hkv=8 D=128`, bf16. Segment lengths are unequal per row. THD uses compact `[total_tokens,H,D]` plus cumulative sequence lengths from segment starts. The cuDNN line uses padded `[num_segments,max_segment_len,H,D]` plus sequence lengths because the JAX cuDNN API does not expose a THD cumulative-length interface.
- Result:
  - `segments_per_row=2`: segment lengths around `[1365,2731]`; cuDNN errored with unsupported `2731` sequence length. `B=1` FA4/CuTe `0.001970s`, THD `0.001684s`; `B=2` FA4/CuTe `0.003547s`, THD `0.002508s`; `B=4` FA4/CuTe `0.006702s`, THD `0.004144s`.
  - `segments_per_row=4`: segment lengths around `[409,819,1228,1640]`; `B=1` cuDNN `0.001317s`, FA4/CuTe `0.001310s`, THD `0.001591s`; `B=2` cuDNN `0.002348s`, FA4/CuTe `0.002342s`, THD `0.002325s`; `B=4` cuDNN `0.004403s`, FA4/CuTe `0.004313s`, THD `0.003757s`.
- Interpretation: For dynamic unequal lengths, the THD cumulative-length path is useful. It clearly beats current FA4/CuTe for two uneven segments per row, and it wins modestly at larger batch for four uneven segments. Padded cuDNN is not equivalent to THD and can fail on unsupported max lengths.
- Next action: Treat upstream THD/cu-seqlens as the dynamic-packed candidate; do not frame padded cuDNN as a drop-in replacement for arbitrary segment lengths.

### 2026-05-29 15:42 - Private JAX cuDNN offsets path
- Hypothesis: The lower-level JAX cuDNN wrapper may expose dynamic packed offsets even though public `jax.nn.dot_product_attention` only exposes per-row sequence lengths.
- API: `jax._src.cudnn.fused_attention_stablehlo.dot_product_attention` with `q_seqlen`, `kv_seqlen`, `q_offsets`, and `kv_offsets`. Inputs remain `[B,S,H,D]`; offsets are per batch row with shape `[B,max_segments+1]`, and sequence lengths are `[B,max_segments]`.
- Command: `python scratch/issue5815_attention_backend_probe.py --batch {1,2,4} --segments-per-row {2,4} --backends jax_cudnn_offsets_core --warmup 3 --iters 10`, run one batch shape per Python process.
- Config: Shape per original row `S=4096 Hq=40 Hkv=8 D=128`, bf16. Segment lengths are unequal per row.
- Result:
  - `segments_per_row=2`: `B=1` `0.000708s`, `B=2` `0.001088s`, `B=4` `0.001878s`.
  - `segments_per_row=4`: `B=1` `0.000576s`, `B=2` `0.000880s`, `B=4` `0.001492s`.
- Interpretation: Yes, there is a cuDNN packed-offset path under JAX internals, and it is substantially faster than both upstream FA4 THD and the local FA4/CuTe path on these dynamic segment layouts. It is not public API and showed process-level fragility when multiple offset shapes were swept in one Python process; isolated one-shape processes completed.
- Next action: Before considering production use, validate numerics for the offsets path and decide whether depending on a private `jax._src` cuDNN API is acceptable or whether this needs an upstream/public JAX hook.

### 2026-05-29 15:50 - Correction: private cuDNN offsets fail exact Grug GQA numerics
- Hypothesis: The private cuDNN offsets path must match segmented reference numerics before its timing can be considered.
- Command: Small reference comparison for `S=1024 segments_per_row=4`, using `MaskType.PADDING_CAUSAL` and start-offset encoding from the JAX docstring.
- Result:
  - Exact Grug GQA shape `Hq=40 Hkv=8`: `B=1` output max abs `4.843750`, mean abs `0.166690`; gradients also diverged materially. `B=2` similarly diverged.
  - Control MHA shape `Hq=8 Hkv=8`: output max abs `0.015625`, mean abs `0.000476`, which is in the expected bf16 range.
- Interpretation: The private cuDNN packed-offset path exists and is fast, but it is not numerically valid for the exact d5120 Grug GQA shape tested here. The MHA control suggests the issue is tied to grouped-query attention with offsets, not the segment layout itself. Therefore the previous offset timings for `Hq=40 Hkv=8` are not usable as evidence for issue5815.
- Next action: Do not switch Grug to this private cuDNN offsets path for issue5815 unless JAX/cuDNN GQA packed-offset semantics are fixed or we add a validated wrapper that preserves K/V strides correctly. Upstream FA4 THD remains the valid dynamic-packed candidate from this probe.

### 2026-05-29 16:25 - B200 FA4 segmented tile tuning
- Hypothesis: Upstream FA4 dense SM100 tile sizes may not transfer to Grug's segmented lower-bound fork, but they should still be verified on the no-segment upstream path.
- Citation: The upstream reference is `flash-attn-4==4.0.0b15`, especially `flash_attn/cute/interface.py` and the SM100 kernels it selects for dense causal GQA.
- Result:
  - Upstream dense/no-segment FA4 fwd+bwd, `S=4096 Hq=40 Hkv=8 D=128`: default upstream tile choice tied explicit `128x128/128x128` at `B=1` (`0.000677s`) and beat explicit `128x128/128x128` at `B=4` (`0.002316s` vs `0.002423s`). Smaller `128x64` variants were slower. This confirms upstream's dense block choice for no segment IDs.
  - Grug segmented JAX/CuTe port: dense-upstream `128x128` was much slower because this path is not the native upstream SM100 kernel; it is the lower-bound segmented fork. The best tested B200 config was `64x64` forward and backward with 128 threads.
  - Segmented `64x64` versus previous default:
    - `segments_per_row=2`: `B=1` `0.001984s -> 0.001936s`; `B=4` `0.006709s -> 0.006639s`.
    - `segments_per_row=4`: `B=1` `0.001321s -> 0.001281s`; `B=4` `0.004309s -> 0.004253s`.
- Implementation: Added a B200/head-dim-128 override in `_fa4_cute.py` that uses `forward_tile=(64,64)`, `backward_tile=(64,64)`, `num_threads=128` for the segmented Grug port.
- Final focused timings after the override, compared with upstream THD:
  - `segments_per_row=2`: `B=1` FA4/CuTe `0.001941s`, THD `0.001722s`; `B=4` FA4/CuTe `0.006664s`, THD `0.004183s`.
  - `segments_per_row=4`: `B=1` FA4/CuTe `0.001289s`, THD `0.001604s`; `B=4` FA4/CuTe `0.004252s`, THD `0.003797s`.
- Interpretation: The B200 segmented FA4 tweak is a small but consistent improvement. It does not erase the THD advantage for two uneven segments or B=4/four segments; it does preserve the FA4/CuTe advantage for B=1/four short segments.

### 2026-05-29 17:10 - Internal upstream THD fwd/bwd meets the 10% target
- Hypothesis: The public `flash_attn_varlen_func` timing is dominated by PyTorch autograd overhead; the true upstream CuTe THD path should be measured with the same internal `_flash_attn_fwd/_flash_attn_bwd` APIs as the dense upstream FA4 baseline.
- API:
  - Dense baseline: `flash_attn.cute.interface._flash_attn_fwd/_flash_attn_bwd` on `[B,S,H,D]`.
  - THD candidate: the same `_flash_attn_fwd/_flash_attn_bwd`, with q/k/v reshaped to `[B*S,H,D]` and `cu_seqlens_{q,k}` derived from contiguous dynamic segment IDs.
- Config: B200, JAX `0.10.0`, PyTorch `2.12.0+cu130`, `flash-attn-4==4.0.0b15`; shapes `1024:8q/2kv`, `2560:20q/4kv`, `5120:40q/8kv`, `D=128`, `S=4096`, `B in {1,4}`, dynamic `segments_per_row in {2,4}`.
- Result: `fa4_thd_internal_core` passed the 10% gate in every row:

| d_model | B | segs/row | min dense baseline | THD internal | ratio |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 2 | 0.000287s | 0.000263s | 0.915 |
| 1024 | 4 | 2 | 0.000559s | 0.000455s | 0.815 |
| 2560 | 1 | 2 | 0.000411s | 0.000350s | 0.851 |
| 2560 | 4 | 2 | 0.001231s | 0.000941s | 0.764 |
| 5120 | 1 | 2 | 0.000677s | 0.000551s | 0.815 |
| 5120 | 4 | 2 | 0.002336s | 0.001764s | 0.755 |
| 1024 | 1 | 4 | 0.000285s | 0.000239s | 0.841 |
| 1024 | 4 | 4 | 0.000559s | 0.000386s | 0.691 |
| 2560 | 1 | 4 | 0.000409s | 0.000305s | 0.746 |
| 2560 | 4 | 4 | 0.001231s | 0.000769s | 0.625 |
| 5120 | 1 | 4 | 0.000679s | 0.000456s | 0.672 |
| 5120 | 4 | 4 | 0.002334s | 0.001409s | 0.604 |

- Numerics: Compared internal THD fwd/bwd against public upstream `flash_attn_varlen_func` autograd on `B=1 S=1024 Hq=40 Hkv=8 D=128`, 4 equal contiguous segments. Max abs diffs: output `0.0`, `dq=0.0`, `dk=0.00048828125`, `dv=0.00390625`.
- Implementation: Added internal THD helpers in `_fa4_thd.py` and routed the experimental JAX custom VJP through them. The public torch forward helper remains available for direct upstream-autograd comparison.
- Limitations: This remains an experimental eager bridge through torch/DLPack, not a production JIT-safe JAX FFI. It supports only causal self-attention with matching contiguous q/kv segment IDs, no padded invalid tokens, and no sliding window.
- Next action: If this path should become production, port the internal THD fwd/bwd call into a real JAX FFI/custom call instead of the torch bridge.

### 2026-05-29 17:32 - Alternate d2560 20q/5kv caveat
- Hypothesis: If d2560 should be interpreted as `Hq=20,Hkv=5` rather than `Hq=20,Hkv=4`, the same THD path should be rechecked.
- Result with dynamic metadata included:
  - 2 segments/row: `B=1` dense baseline `0.000405s`, THD internal `0.001000s`, ratio `2.471`; `B=4` dense baseline `0.001214s`, THD internal `0.001596s`, ratio `1.314`.
  - 4 segments/row: `B=1` dense baseline `0.000405s`, THD internal `0.000945s`, ratio `2.335`; `B=4` dense baseline `0.001212s`, THD internal `0.001402s`, ratio `1.156`.
- Follow-up: A direct precomputed-`cu_seqlens` internal sweep for the same `20q/5kv` shape was fast (`B=1` around `0.00030-0.00035s`, `B=4` around `0.00076-0.00093s` depending on segment count), so the miss is dominated by dynamic segment-id metadata construction in the torch bridge, not the upstream THD attention kernel.
- Interpretation: The 10% target is proven for `1024:8q/2kv`, `2560:20q/4kv`, and exact `5120:40q/8kv`. It is not yet met for `2560:20q/5kv` if dynamic `segment_ids -> cu_seqlens` construction is included in the timed core.

### 2026-05-29 17:54 - Explicit THD cu-lens core passes both d2560 interpretations
- Hypothesis: The active target names a `thd+cu_q_lens` path, so the fwd+bwd core should be measured from precomputed cumulative sequence lengths, not from segment-id metadata construction.
- Implementation: Added `torch_fa4_thd_cu_lens_fwd_bwd` and scratch backend `fa4_thd_cu_lens_core`. The probe still generates dynamic multi-segment layouts, then builds `cu_seqlens` before the timed fwd+bwd core.
- Result: Consolidated torch-only matrix passed every row for `1024:8q/2kv`, `2560:20q/4kv`, `2560:20q/5kv`, and `5120:40q/8kv`, with `B in {1,4}` and dynamic `segments_per_row in {2,4}`.

| shape | B | segs/row | dense upstream | THD cu-lens | ratio |
|---|---:|---:|---:|---:|---:|
| 1024:8q/2kv | 1 | 2 | 0.000350s | 0.000281s | 0.803 |
| 1024:8q/2kv | 4 | 2 | 0.000561s | 0.000466s | 0.831 |
| 2560:20q/4kv | 1 | 2 | 0.000408s | 0.000367s | 0.900 |
| 2560:20q/4kv | 4 | 2 | 0.001234s | 0.000952s | 0.771 |
| 2560:20q/5kv | 1 | 2 | 0.000400s | 0.000371s | 0.928 |
| 2560:20q/5kv | 4 | 2 | 0.001217s | 0.000947s | 0.778 |
| 5120:40q/8kv | 1 | 2 | 0.000681s | 0.000563s | 0.827 |
| 5120:40q/8kv | 4 | 2 | 0.002345s | 0.001780s | 0.759 |
| 1024:8q/2kv | 1 | 4 | 0.000283s | 0.000254s | 0.895 |
| 1024:8q/2kv | 4 | 4 | 0.000558s | 0.000396s | 0.710 |
| 2560:20q/4kv | 1 | 4 | 0.000408s | 0.000320s | 0.783 |
| 2560:20q/4kv | 4 | 4 | 0.001232s | 0.000781s | 0.634 |
| 2560:20q/5kv | 1 | 4 | 0.000398s | 0.000323s | 0.811 |
| 2560:20q/5kv | 4 | 4 | 0.001216s | 0.000773s | 0.636 |
| 5120:40q/8kv | 1 | 4 | 0.000679s | 0.000466s | 0.686 |
| 5120:40q/8kv | 4 | 4 | 0.002343s | 0.001492s | 0.637 |

- Interpretation: The requested `thd+cu_q_lens` fwd+bwd core is within 10% of, and usually faster than, the dense upstream FA4 baseline across the requested model dimensions. The remaining caveat is API shape: generic `AttentionMask` only carries segment IDs today, so `gpu_fa4_thd` is still an experimental torch/DLPack bridge for segment IDs, while `fa4_thd_cu_lens_core` is the proven cu-lens core path.

### 2026-05-29 18:34 - Fixed max-segment metadata on Grug examples
- Hypothesis: Eagerly materializing fixed-shape THD metadata when constructing packed examples should remove the expensive full `segment_ids -> cu_seqlens` scan from the attention hot path while keeping batching/JIT shapes stable.
- Implementation:
  - Added `ThdSegmentMetadata(segment_lengths, num_segments)` to the Grug attention mask. `segment_lengths` is padded to a fixed max segment count and includes the trailing padding run when present so THD output can reshape back to dense BSHD.
  - `GrugLmExample.causal(..., max_segments=...)` now asks the mask to build this metadata eagerly.
  - Packed token/chat datasets pass `max_segments_per_example + 1` to leave room for the padding run.
  - The experimental THD backend now prefers fixed segment-length metadata when present, compacting only the small `[B, max_segments]` length table to upstream FA4 `cu_seqlens`. Padding rows with zero metadata are treated as one full dummy segment.
  - The scratch probe gained `fa4_thd_lengths_core` to time this exact fixed-metadata path.
- Validation:
  - Local `py_compile` passed for the edited modules and scratch probe.
  - CPU-side stack sanity check produced expected batched metadata, e.g. `[[2,2,2,2],[3,2,3,0]]` with counts `[4,3]`.
  - Focused B200 probe completed for exact d5120 `S=4096,Hq=40,Hkv=8,D=128`, dynamic uneven `segments_per_row in {2,4}`, `B in {1,4}`.
- Timings:

| segs/row | B | dense upstream | current FA4/CuTe | THD fixed lengths | THD compact cu-lens |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.000681s | 0.001939s | 0.000892s | 0.000656s |
| 2 | 4 | 0.002399s | 0.006653s | 0.002391s | 0.002151s |
| 4 | 1 | 0.000680s | 0.001280s | 0.000790s | 0.000561s |
| 4 | 4 | 0.002343s | 0.004249s | 0.002028s | 0.001801s |

- cuDNN segmented status: dynamic uneven segment lengths did not produce a usable cuDNN baseline here. The two-segment case rejected the max segment length, and the four-segment case hit a cuDNN execution-plan finalize error.
- Interpretation: Fixed max-segment metadata is materially faster than the current Grug segmented FA4/CuTe port and lands well under the issue threshold. It does not fully match precomputed compact `cu_seqlens` for B=1 because the backend still compacts fixed lengths to FA4's compact API inside the timed path. The next production-quality step would be moving that final compaction to the batch loader or exposing compact cu-lens directly with a stable shape policy.

### 2026-05-29 18:58 - Integration cleanup
- Added CPU tests for fixed THD metadata materialization, stackability on `GrugLmExample`, and q/kv segment-id mismatch rejection.
- Exported the experimental `gpu_fa4_thd_attention` package symbol so the string backend is discoverable alongside `gpu_fa4_cute_attention`.
- Tightened `AttentionMask.with_segment_ids(..., max_segments=...)` so THD metadata is only built when q/kv segment IDs match.
- Validation:
  - `py_compile` passed locally for the edited Grug attention/data modules, the focused test file, and the scratch probe.
  - Manual CPU checks passed for metadata lengths/counts, batched `GrugLmExample` metadata, and mismatch rejection.
  - `git diff --check` passed.
  - Local environment does not have `pytest` or `pyrefly` entry points installed. A remote `pytest -q lib/levanter/tests/grug/test_attention.py` attempt with `JAX_PLATFORMS=cpu` timed out during startup before emitting test output, so the pytest file itself has not completed in this turn.
