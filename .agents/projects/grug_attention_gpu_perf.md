# Grug GPU Attention Perf Report (2026-02-21)

## Snapshot
This report summarizes expanded GPU perf testing for `lib/levanter/src/levanter/grug/attention.py` on a DGX Spark (`NVIDIA GB10`) using the local benchmark harness:
- `lib/levanter/scripts/bench/bench_grug_attention.py`

Scope:
- Compare `reference` vs `pallas_gpu`
- Evaluate current `auto` block-size dispatch
- Compare GB10-explicit block sizes vs `auto`
- Evaluate tokamax-like block sizes on GB10
- Include sliding-window + segment-id cases
- Include forward-only and forward+backward

## Environment
- Branch: `feat/gpu-attention-block-size-defaults`
- Backend: `jax.default_backend() == "gpu"`
- Device: `CudaDevice(id=0)` with device kind `NVIDIA GB10`
- Dtype in all runs: `bf16`
- For `--benchmark-backward`: harness upcasts to `f32` before grad capture

## Commands
Representative command forms used:

```bash
uv run python lib/levanter/scripts/bench/bench_grug_attention.py \
  --impls reference,pallas_gpu \
  --batch <B> --seq <S> --heads <H> --kv_heads <HKV> --head_dim <D> \
  --steps <N> --dtype bf16

uv run python lib/levanter/scripts/bench/bench_grug_attention.py \
  --impls reference,pallas_gpu --benchmark-backward \
  --batch <B> --seq <S> --heads <H> --kv_heads <HKV> --head_dim <D> \
  --steps 1 --dtype bf16
```

Override tests used:
- Tokamax-like: `--gpu-block-q 128 --gpu-block-k 128 --gpu-block-q-dkv 32 --gpu-block-kv-dkv 32 --gpu-block-q-dq 32 --gpu-block-kv-dq 32`
- GB10 explicit (llama/qwen-like): `--gpu-block-q 32 --gpu-block-k 32 --gpu-block-q-dkv 16 --gpu-block-kv-dkv 32 --gpu-block-q-dq 32 --gpu-block-kv-dq 16`
- GB10 explicit (125m high-batch): `--gpu-block-q 32 --gpu-block-k 32 --gpu-block-q-dkv 16 --gpu-block-kv-dkv 64 --gpu-block-q-dq 64 --gpu-block-kv-dq 16`
- HD256 safety probe: all 16s for fwd/bwd block sizes

Raw run logs saved under:
- `/tmp/grug_attn_perf/*.log`

## Forward Results (auto)
| Shape | Mask | Reference tok/s | Pallas auto tok/s | Speedup |
|---|---|---:|---:|---:|
| B=1 S=4096 H=32 HKV=8 D=128 (llama-8b-ish) | causal | 33,023.95 | 698,292.77 | 21.15x |
| B=1 S=8192 H=32 HKV=8 D=128 (llama-8b-ish long) | causal | 16,445.61 | 416,590.26 | 25.33x |
| B=4 S=4096 H=16 HKV=8 D=128 (qwen-ish) | causal | 66,270.30 | 1,472,412.51 | 22.22x |
| B=16 S=2048 H=8 HKV=8 D=128 (125m high-batch) | causal | 262,587.53 | 6,723,430.49 | 25.61x |
| B=8 S=2048 H=4 HKV=4 D=256 | causal | 477,472.89 | 5,690,771.98 | 11.92x |
| B=4 S=4096 H=16 HKV=8 D=128 | causal + SWA(1024) + segment_ids[2d] | 66,299.57 | 1,333,758.71 | 20.12x |

## Forward+Backward Results (auto)
| Shape | Mask | Reference tok/s | Pallas auto tok/s | Speedup |
|---|---|---:|---:|---:|
| B=1 S=4096 H=32 HKV=8 D=128 | causal | 39,112.40 | 91,348.44 | 2.34x |
| B=1 S=8192 H=32 HKV=8 D=128 | causal | 20,077.70 | 52,903.28 | 2.64x |
| B=4 S=4096 H=16 HKV=8 D=128 | causal | 77,042.60 | 127,420.24 | 1.65x |
| B=16 S=2048 H=8 HKV=8 D=128 | causal | 284,352.38 | 557,914.10 | 1.96x |
| B=8 S=2048 H=4 HKV=4 D=256 | causal | 478,039.78 | **FAIL** | n/a |
| B=4 S=4096 H=16 HKV=8 D=128 | causal + SWA(1024) + segment_ids[2d] | 76,391.52 | **FAIL** | n/a |

Failure details:
- HD256 auto backward failed with:
  - `RESOURCE_EXHAUSTED: Shared memory size limit exceeded: requested 167936, available: 101376`
- SWA+segment-id backward failed with:
  - `TypeError: eq got incompatible shapes for broadcasting: (16, 1), (32, 1)`

## GB10 Explicit vs Auto
### llama-8b-ish (B=1, S=4096, H=32, HKV=8, D=128)
- Forward:
  - auto: `698,292.77 tok/s`
  - GB10 explicit: `720,153.86 tok/s`
  - delta: GB10 explicit +3.1%
- Forward+backward:
  - auto: `91,348.44 tok/s`
  - GB10 explicit: `94,953.78 tok/s`
  - delta: GB10 explicit +3.9%

### 125m high-batch (B=16, S=2048, H=8, HKV=8, D=128)
- Forward:
  - auto: `6,723,430.49 tok/s`
  - GB10 explicit: `6,463,201.15 tok/s`
  - delta: auto +4.0%
- Forward+backward:
  - auto: `557,914.10 tok/s`
  - GB10 explicit: `563,115.33 tok/s`
  - delta: GB10 explicit +0.9%

Takeaway:
- `auto` and GB10-explicit are effectively equivalent for tested D=128 regimes.

## Tokamax-Like Blocks on GB10
Tokamax-like blocks (`128/128` forward, `32` backward tiles) failed on every tested GB10 case.

Observed failures:
- Forward runs (llama-8b-ish/qwen-ish/125m/hd256):
  - `RESOURCE_EXHAUSTED: Shared memory size limit exceeded`
  - requests were `163840` to `360448` bytes vs `101376` available
- Backward run (llama-8b-ish):
  - `RESOURCE_EXHAUSTED: ... requested 327680, available: 101376`

Takeaway:
- Tokamax-like defaults are not viable on GB10 without major retuning.

## HD256 Backward Safety Probe
For `B=8, S=2048, H=4, HKV=4, D=256`:
- Auto (current table-derived backward tiles) failed shared-memory limits.
- A conservative explicit config (`16/16/16/16/16/16`) succeeds, but is slower than reference:
  - reference: `452,187.29 tok/s`
  - pallas_gpu explicit-small: `252,772.01 tok/s` (`0.559x` vs reference)

Takeaway:
- HD256 backward path currently needs either:
  - safer default tile selection on GB10, or
  - explicit fallback to reference for this regime.

## Interpretation: Non-GB10 Defaults
Question: should we keep tokamax-like defaults for non-GB10 devices?

Answer from this data:
- Yes, keeping non-GB10 defaults separate from GB10 is still the correct policy.
- GB10 behavior is materially different (strict shared-memory envelope) and fails with tokamax-like tiles.
- We did not run on server GPUs in this sweep, so we cannot make new claims against A100/H100 behavior here.
- Practical policy remains:
  - non-GB10: tokamax-like or JAX-default-like larger tiles
  - GB10: dedicated smaller tiles and separate tuning table

## Open Issues Found During Perf Sweep
1. Segment-id backward bug with mixed q/kv block sizes
- Repro: SWA + segment ids + auto backward on qwen-ish shape
- Error: broadcasting mismatch in equality compare (`(16,1)` vs `(32,1)`)
- Impact: segment-id backward can fail when q and kv backward tiles differ.

2. HD256 GB10 backward default instability
- Repro: D=256 backward auto profile
- Error: shared memory exhausted
- Impact: default GB10 tuned entry is not robust for tested D=256 case.

## Suggested Next Steps
1. Fix segment-id backward mask shape bug in `pallas_mosaic` and rerun SWA+segment-id backward matrix.
2. Adjust GB10 `llama2-hd256` backward defaults (or force safe fallback) to avoid runtime OOM.
3. Add a benchmark matrix script to automate this report generation and prevent regression.
