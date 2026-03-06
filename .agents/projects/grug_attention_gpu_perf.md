# Grug GPU Attention Pallas Kernel - Running Report

## Scope
This running log tracks GPU Pallas attention tuning and validation for:
- `lib/levanter/src/levanter/grug/attention.py`
- `lib/levanter/src/levanter/kernels/pallas/attention/pallas_mosaic.py`
- `lib/levanter/src/levanter/kernels/pallas/attention/tuned_block_sizes.py`
- `lib/levanter/scripts/bench/bench_grug_attention.py`

Primary goal:
- Set robust GB10-friendly defaults for `pallas_gpu` while preserving device-specific fallback behavior and reproducible benchmarking.

## Current Status
- `auto` block-size dispatch gives strong forward gains and moderate forward+backward gains on tested GB10 `head_dim=128` shapes.
- Previously failing GB10 cases (HD256 backward OOM and SWA+segment-id backward shape mismatch) were fixed and revalidated.
- GB10 needs dedicated small-tile defaults; tokamax-like large tiles repeatedly exceed GB10 shared-memory limits.

## Environment
- Date: 2026-02-21
- Device: `NVIDIA GB10`
- Backend: `jax.default_backend() == "gpu"`
- Dtype: `bf16` for benchmark inputs
- Backward benchmark path: harness upcasts to `f32` for grad capture
- Harness: `lib/levanter/scripts/bench/bench_grug_attention.py`
- Raw logs: `/tmp/grug_attn_perf/*.log`

## Iteration Log

### 2026-02-21 Initial matrix sweep

#### Commands used
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

#### Shape/dtype grid
- `B=1, S=4096, H=32, HKV=8, D=128` (llama-8b-ish)
- `B=1, S=8192, H=32, HKV=8, D=128` (long-context llama-8b-ish)
- `B=4, S=4096, H=16, HKV=8, D=128` (qwen-ish)
- `B=16, S=2048, H=8, HKV=8, D=128` (125m high-batch)
- `B=8, S=2048, H=4, HKV=4, D=256`
- Sliding-window + segment-id case: `B=4, S=4096, H=16, HKV=8, D=128`, `sliding_window=1024`, `segment_id_mode=2d`

#### Forward-only performance (`auto`)
| Shape | Mask | Reference tok/s | Pallas auto tok/s | Speedup |
|---|---|---:|---:|---:|
| B=1 S=4096 H=32 HKV=8 D=128 | causal | 33,023.95 | 698,292.77 | 21.15x |
| B=1 S=8192 H=32 HKV=8 D=128 | causal | 16,445.61 | 416,590.26 | 25.33x |
| B=4 S=4096 H=16 HKV=8 D=128 | causal | 66,270.30 | 1,472,412.51 | 22.22x |
| B=16 S=2048 H=8 HKV=8 D=128 | causal | 262,587.53 | 6,723,430.49 | 25.61x |
| B=8 S=2048 H=4 HKV=4 D=256 | causal | 477,472.89 | 5,690,771.98 | 11.92x |
| B=4 S=4096 H=16 HKV=8 D=128 | causal + SWA(1024) + segment_ids[2d] | 66,299.57 | 1,333,758.71 | 20.12x |

#### Forward+backward performance (`auto`)
| Shape | Mask | Reference tok/s | Pallas auto tok/s | Speedup |
|---|---|---:|---:|---:|
| B=1 S=4096 H=32 HKV=8 D=128 | causal | 39,112.40 | 91,348.44 | 2.34x |
| B=1 S=8192 H=32 HKV=8 D=128 | causal | 20,077.70 | 52,903.28 | 2.64x |
| B=4 S=4096 H=16 HKV=8 D=128 | causal | 77,042.60 | 127,420.24 | 1.65x |
| B=16 S=2048 H=8 HKV=8 D=128 | causal | 284,352.38 | 557,914.10 | 1.96x |
| B=8 S=2048 H=4 HKV=4 D=256 | causal | 478,039.78 | FAIL | n/a |
| B=4 S=4096 H=16 HKV=8 D=128 | causal + SWA(1024) + segment_ids[2d] | 76,391.52 | FAIL | n/a |

#### Pitfalls observed
- HD256 backward failed with `RESOURCE_EXHAUSTED` shared memory (`requested 167936`, `available 101376`).
- SWA + segment-id backward failed with broadcast mismatch (`(16,1)` vs `(32,1)`).

#### Targeted override comparisons
- Tokamax-like blocks on GB10 failed across tested cases with shared-memory overflow (`163840` to `360448` bytes requested vs `101376` available).
- GB10-explicit block sizes vs `auto` (D=128 regimes): effectively equivalent; differences were small (roughly +/-4%).

#### Conclusion
- Dedicated GB10-specific defaults are required.
- Non-GB10 defaults should remain separate.

### 2026-02-21 Follow-up fixes + revalidation

#### Code changes landed
1. Normalized `segment_mask` handling in `pallas_mosaic` for singleton-column views (`[N,1]`) in backward paths.
2. Updated GB10 `llama2-hd256` tuned block sizes to a stable/faster config:
   - `block_q=32`, `block_k=16`, `block_q_dkv=16`, `block_kv_dkv=32`, `block_q_dq=32`, `block_kv_dq=16`, `num_warps=4`.
3. Adjusted shape-bucket precedence so exact `head_dim=256` cases match `llama2-hd256` before `qwen3-ish`.
4. Updated fallback selection to prefer device-specific keys (`NVIDIA GB10`) before generic defaults.
5. Added benchmark regression targets:
   - `--target gb10_segment_swa_backward`
   - `--target gb10_hd256_backward_auto`

#### Revalidation evidence
- HD256 backward auto case now passes (previously OOM):
  - reference: `490,341.10 tok/s`
  - pallas auto: `429,395.66 tok/s`
  - status: PASS, no shared-memory OOM.
- SWA + segment-id backward auto case now passes (previously shape error):
  - reference: `77,816.44 tok/s`
  - pallas auto: `128,008.90 tok/s`
  - status: PASS, speedup `1.645x`.

#### Focused HD256 tuning notes
- Conservative all-16s config was stable but slow (`~252,772 tok/s`, `0.559x` vs reference).
- Best stable tested config (listed above) reached `~445,607 tok/s`, much closer to reference.
- Larger KV tiles (`bk=32`, `bk_dkv=64`, or `q_dkv=32`) repeatedly exceeded GB10 shared-memory limits.

## Known Limitations
- GB10 HD256 backward is stable after retuning, but still slightly below reference throughput in tested shape(s).
- This report is GB10-focused; it does not provide new server-GPU (A100/H100) claims.

## Next Steps
1. Keep GB10-specific defaults for `pallas_gpu` and separate non-GB10 defaults.
2. Continue targeted backward tuning for HD256 regimes if additional gains are needed.
3. Keep regression targets in benchmark harness as guardrails for segment-id/SWA and HD256 paths.

## Repro Commands (regression checks)
```bash
uv run python lib/levanter/scripts/bench/bench_grug_attention.py \
  --impls reference,pallas_gpu --benchmark-backward \
  --batch 8 --seq 2048 --heads 4 --kv_heads 4 --head_dim 256 \
  --steps 1 --dtype bf16

uv run python lib/levanter/scripts/bench/bench_grug_attention.py \
  --impls reference,pallas_gpu --benchmark-backward \
  --batch 4 --seq 4096 --heads 16 --kv_heads 8 --head_dim 128 \
  --steps 1 --dtype bf16 --sliding-window 1024 --segment-id-mode 2d
```
