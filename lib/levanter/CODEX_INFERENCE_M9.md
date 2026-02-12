# CODEX Inference M9: `max_seqs` Scaling Beyond 10

Status: IN PROGRESS (started 2026-02-07; length-fixed sweep now validated through `max_seqs=125` configs, with current stable ceiling `max_seqs=124`)

## Goal

Find the practical limit for increasing concurrent sequences (`engine.max_seqs`) beyond 10 on real TPU (`v5p-16`) for:
- `max_new_tokens: 2048`
- `n_rounds: 1`
- Kernel-on inference path from M8 (`q32`, `kv16`)

## Fixed Baseline (from M8 final)

Starting point:
- `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m8_final.yaml`
- `round_total_s ~ 75s` for 10 prompts.

M9 keeps these invariants unless explicitly varied:
- `engine.max_seq_len: 2560`
- `engine.page_size: 64`
- `engine.reset_mode: physical`
- `engine.cleanup_mode: none`
- `model.use_tpu_ragged_paged_attention: true`
- `model.ragged_paged_q_block_size: 32`
- `model.ragged_paged_kv_block_pages: 16`
- `defer_tracker_logs_until_end: true`
- `skip_samples_table: true`
- `trainer.tracker.type: noop`

## Experiment Design

Phase A (fixed pages):
- Increase `max_seqs` while keeping `max_pages=480` to locate immediate page-capacity boundary.

Phase B (scaled pages):
- Increase both `max_seqs` and `max_pages` to test higher concurrency and determine practical limits.

Phase C (extended scaled pages):
- Continue scaled-page sweep beyond 28 to verify practical boundary under current scheduler defaults.

Phase D (boundary isolation):
- Probe `56+` with scaled pages to identify the first unstable concurrency point.

## Experiment Matrix

| ID | Config | max_seqs | max_pages | prompts | status | round_total_s | notes |
|---|---|---:|---:|---:|---|---:|---|
| A1 | `...m9_s12_p480.yaml` | 12 | 480 | 12 | PASS | 79.591s | `DecodeStats[after_decode]: pages_in_use=364 free=116`; `round_total_generated=22536` |
| A2 | `...m9_s14_p480.yaml` | 14 | 480 | 14 | PASS | 80.428s | `DecodeStats[after_decode]: pages_in_use=398 free=82`; `round_total_generated=24590` |
| A3 | `...m9_s16_p480.yaml` | 16 | 480 | 16 | FAIL (fast-reject) | - | Config validation: `engine.max_pages=480` too small (`est. 528 pages needed`) |
| B1 | `...m9_s20_p720.yaml` | 20 | 720 | 20 | PASS | 80.842s | `DecodeStats[after_decode]: pages_in_use=420 free=300`; `round_total_generated=25620` |
| B2 | `...m9_s24_p864.yaml` | 24 | 864 | 24 | PASS | 80.970s | `DecodeStats[after_decode]: pages_in_use=420 free=444`; `round_total_generated=25624` |
| B3 | `...m9_s28_p1008.yaml` | 28 | 1008 | 28 | PASS | 80.375s | `DecodeStats[after_decode]: pages_in_use=420 free=588`; `round_total_generated=25628` |
| C1 | `...m9_c1_s32_p1152.yaml` | 32 | 1152 | 32 | PASS | 80.981s | `DecodeStats[after_decode]: pages_in_use=424 free=728`; `round_total_generated=25632` |
| C2 | `...m9_c2_s36_p1296.yaml` | 36 | 1296 | 36 | PASS | 83.452s | `DecodeStats[after_decode]: pages_in_use=432 free=864`; `round_total_generated=25636` |
| C3 | `...m9_c3_s40_p1440.yaml` | 40 | 1440 | 40 | PASS | 81.022s | `DecodeStats[after_decode]: pages_in_use=440 free=1000`; `round_total_generated=25640` |
| C4 | `...m9_c4_s44_p1584.yaml` | 44 | 1584 | 44 | PASS | 82.091s | `DecodeStats[after_decode]: pages_in_use=444 free=1140`; `round_total_generated=25644` |
| C5 | `...m9_c5_s48_p1728.yaml` | 48 | 1728 | 48 | PASS | 83.818s | `DecodeStats[after_decode]: pages_in_use=448 free=1280`; `round_total_generated=25648` |
| C6 | `...m9_c6_s52_p1872.yaml` | 52 | 1872 | 52 | PASS | 83.082s | `DecodeStats[after_decode]: pages_in_use=444 free=1428`; `round_total_generated=25652` |
| C7 | `...m9_c7_s56_p2016.yaml` | 56 | 2016 | 56 | PASS | 82.265s | `DecodeStats[after_decode]: pages_in_use=432 free=1584`; `round_total_generated=25656` |
| D1 | `...m9_c7a_s57_p2052.yaml` | 57 | 2052 | 57 | FAIL (runtime) | - | Reproducible first-decode failure: `Decode iter ~18.3s` then `Anomalies` and both workers exit status 1 |
| D1P | `...m9_d1p_s57_p2052_pref56.yaml` | 57 | 2052 | 57 | FAIL (fast-reject) | - | Config validation: `engine.max_seqs_in_prefill must cover all prompts per round` |
| D1S | `...m9_d1s_s57_p2052_tpr8_r80.yaml` | 57 | 2052 | 57 | FAIL (runtime) | - | Scheduler probe (`max_tokens_per_round=8`, `max_rounds=80`) still hits the same first-decode `Anomalies` failure |
| D2 | `...m9_c7b_s58_p2088.yaml` | 58 | 2088 | 58 | FAIL (runtime) | - | Same first-decode failure signature as D1 |
| D3 | `...m9_c8_s60_p2160.yaml` | 60 | 2160 | 60 | FAIL (runtime, repro x2) | - | Same first-decode failure signature (`Anomalies`, both workers exit status 1) |
| D4 | `...m9_d2a_s60_p2160_pref56.yaml` | 60 | 2160 | 56 | FAIL (runtime) | - | Isolation probe: `after_prefill active=56` still fails with same first-decode `Anomalies` signature |

## Logs

- A1: `/tmp/levanter_run_m9_a1_s12_p480.log`
- A2: `/tmp/levanter_run_m9_a2_s14_p480.log`
- A3: `/tmp/levanter_run_m9_a3_s16_p480.log`
- B1: `/tmp/levanter_run_m9_b1_s20_p720.log`
- B2: `/tmp/levanter_run_m9_b2_s24_p864.log`
- B3: `/tmp/levanter_run_m9_b3_s28_p1008.log`
- C1: `/tmp/levanter_run_m9_c1_s32_p1152.log`
- C2: `/tmp/levanter_run_m9_c2_s36_p1296.log`
- C3: `/tmp/levanter_run_m9_c3_s40_p1440.log`
- C4: `/tmp/levanter_run_m9_c4_s44_p1584.log`
- C5: `/tmp/levanter_run_m9_c5_s48_p1728.log`
- C6: `/tmp/levanter_run_m9_c6_s52_p1872.log`
- C7: `/tmp/levanter_run_m9_c7_s56_p2016.log`
- C7 revalidation: `/tmp/levanter_run_m9_c7_s56_p2016_rerun2.log`
- D1: `/tmp/levanter_run_m9_c7a_s57_p2052.log`
- D1 prefill probe (fast-reject): `/tmp/levanter_run_m9_d1p_s57_p2052_pref56.log`
- D1 scheduler probe: `/tmp/levanter_run_m9_d1s_s57_p2052_tpr8_r80.log`
- D2: `/tmp/levanter_run_m9_c7b_s58_p2088.log`
- D3: `/tmp/levanter_run_m9_c8_s60_p2160.log`
- D3 rerun: `/tmp/levanter_run_m9_c8_s60_p2160_rerun1.log`
- D4 isolation probe: `/tmp/levanter_run_m9_d2a_s60_p2160_pref56.log`

## Notes

- For fair capacity comparison, each config sets prompt count equal to `max_seqs`.
- Phase A boundary: `max_pages=480` supports `max_seqs=14`; `max_seqs=16` is rejected by startup validation.
- Phase B/C/D stability boundary: scaled-page configs are stable through `max_seqs=56` on `v5p-16`.
- Runtime failure boundary: `max_seqs>=57` fails reproducibly at the first decode iteration with an `Anomalies` crash signature (worker exit status 1), not with page-capacity validation failure and not with container OOM.
- Isolation result: failure tracks configured `max_seqs` shape, not only active prompt count (`max_seqs=60` with `56` active prompts still fails in D4).
- Scheduler sensitivity probe at `max_seqs=57` (`max_tokens_per_round=8`, `max_rounds=80`) does not recover stability; failure signature is unchanged.
- Throughput trend: round wall time remains in the same band (`~80-84s`) from `20` to `48` prompts.
- Workload saturation note: with M6 scheduler knobs (`max_rounds=64`, `max_tokens_per_round=10`, `max_queued_tokens=64`), decode workload plateaus near `~25.6k` generated tokens/round for B1-C7; per-prompt generated tokens decrease as `max_seqs` rises.
- Practical interpretation:
  - We have page-capacity stability and substantial free-page headroom through `max_seqs=56`.
  - Under current scheduler settings, higher `max_seqs` does not increase total generated tokens per round proportionally.
  - Practical stable ceiling for this setup is currently `max_seqs=56`.
- Next M9 step: investigate the first-decode `Anomalies` failure at `57+` (runtime/collective behavior) and test whether scheduler knob changes can recover stability above `56`.

## Scheduler Explainer (What It Is, Why It Matters)

The scheduler here is a bounded decode work planner:
- `max_tokens_per_round`: how many queued tokens can be packed per decode step.
- `max_rounds`: max decode steps per device-side generation loop invocation.
- `max_queued_tokens`: queue capacity between token updates and packing.

This is not the mechanism that creates "variable length" outputs. Variable-length behavior comes from per-sequence finish checks (`max_new_tokens` and optional stop sequences).

Useful decode-budget approximation per `generate()` call:
- `decode_budget ~= floor(max_seq_len / max_rounds) * max_rounds * min(max_tokens_per_round, max_seqs)`

For the M9 scaled-page default regime (`max_seq_len=2560`, `max_rounds=64`, `max_tokens_per_round=10`):
- `decode_budget ~= 2560 * 10 = 25,600` decode-generated tokens.
- Reported totals around `~25.6k` are therefore expected.
- Round totals can be slightly above this because prefill can contribute up to ~1 sampled token per active prompt.

Practical consequence:
- If your target is full `2048` outputs for many prompts in one round, page capacity alone is insufficient.
- You also need scheduler budget high enough for total target decode tokens.
- Necessary (not sufficient) check:
  - `decode_budget >= num_prompts * target_new_tokens`
- In practice, this is why the length-fixed path uses much larger scheduler knobs than the scaled-page default path.

## Phase E (Length-Fixed 2048 Sweep)

Objective:
- Measure max concurrent prompts when each prompt still receives full `2048` new tokens (`round_total_generated == prompts * 2048`).

Key scheduler settings used:
- `max_tokens_per_round = max_seqs`
- `max_queued_tokens = max_seqs`
- `max_rounds = 64`

### E1: Initial scale-up (`max_pages = 36 * max_seqs`)

Passes observed:
- `56,57,60,64,68,69,70,80,96,100,110,112,113` all PASS with exact `prompts*2048`.

Failures observed:
- `114,115,120` (and `114` rerun) FAIL with runtime `Anomalies` / TPU halt signatures.

Important pattern:
- All failures in this set used `max_pages > 4096`.
- All passes in this set used `max_pages <= 4068`.

### E2: 4096-page cap hypothesis

We then capped `max_pages` to `4096` and re-tested:
- `114 @ max_pages=4096`: PASS
- `120 @ max_pages=4096`: PASS
- `124 @ max_pages=4096`: PASS (`DecodeStats[after_decode]: pages_in_use=4092 free=4`)
- `125 @ max_pages=4096`: FAIL fast with explicit validation:
  - `engine.max_pages=4096 is too small ... est. 4125 pages needed`

### Current M9 length-fixed result

- Stable max prompt count at `max_new_tokens=2048` is currently **`124`** (single round, this kernel/config family).
- The `125` boundary is rejected deterministically by page-capacity estimation.

Representative logs:
- `/tmp/levanter_run_m9_lenfix_s113_p4068_tpr113.log`
- `/tmp/levanter_run_m9_lenfix_s114_p4104_tpr114.log` (fail)
- `/tmp/levanter_run_m9_lenfix_s114_p4104_tpr114_rerun.log` (fail)
- `/tmp/levanter_run_m9_lenfix_s114_p4096_tpr114.log` (pass)
- `/tmp/levanter_run_m9_lenfix_s120_p4096_tpr120.log` (pass)
- `/tmp/levanter_run_m9_lenfix_s124_p4096_tpr124.log` (pass)
- `/tmp/levanter_run_m9_lenfix_s125_p4096_tpr125.log` (fast-reject)

## Phase F (128 Prompts x 2048 Optimization)

Objective:
- Force `128` prompts in one round and optimize throughput while preserving fixed-length generation (`round_total_generated == 128 * 2048`).

### F1: Bring-up at 128

Attempts with `page_size=128` (`max_seqs=128`) consistently failed at prefill with scoped-vmem OOM:
- Signature: `ragged_paged_attention ... Scoped allocation ... 16.79M > 16.00M`.
- This persisted across `max_prefill_size`, `max_seq_len`, `q_block`, `kv_block`, and page-geometry tweaks.

Switching to `page_size=64` with exact page budget (`max_pages=4224`) was the first setup to decode at `128`.

### F2: 128 Scheduler Sweep

| Variant | Core knobs | Outcome | Fixed 2048/prompt | Key metric |
|---|---|---|---|---|
| Baseline-like | `ps=64,p=4224,tpr=128,r=64,mqt=128` | FAIL (`Anomalies`) | yes until crash | mid-decode crash, `real 249.88` |
| OptA | `tpr=64,r=80,mqt=128` | PASS | no | `round_avg_tok_s ~= 1363.9`, `real 230.20` |
| OptB | `tpr=96,r=80,mqt=128` | PASS | no | `round_avg_tok_s ~= 1601.9`, `real 257.23` |
| OptC | `tpr=256,r=32,mqt=256` | PASS | **yes** (`262,144`) | **`round_avg_tok_s ~= 1684.0`, `decode_avg_tok_s ~= 2223.9`, `real 275.72`** |
| OptD | `tpr=128,r=64,mqt=256` | FAIL (`Anomalies`) | yes until crash | `real 254.33` |
| OptF | `tpr=128,r=64,cleanup=end` | FAIL (`Anomalies`) | yes until crash | `real 253.07` |
| OptG | `tpr=256,r=64,mqt=256` | PASS | **yes** (`262,144`) | `round_avg_tok_s ~= 1686.4`, `real 269.17` |
| OptI | `tpr=384,r=32,mqt=384` | PASS | **yes** (`262,144`) | `round_avg_tok_s ~= 1618.1`, `real 268.03` |

### F3: Extended 128x2048 Throughput Sweep

Goal:
- Improve stable full-length throughput beyond OptG while keeping `round_total_generated=262144`.

| Variant | Core knobs | Outcome | Key metric |
|---|---|---|---|
| OptG rerun2 | `tpr=256,r=64,mqt=256` | PASS | `round_avg_tok_s ~= 1693.2`, `real 266.29` |
| OptJ | `tpr=320,r=64,mqt=320` | PASS | `round_avg_tok_s ~= 1630.9`, `real 264.65` |
| OptL | `tpr=384,r=64,mqt=384` | PASS | `round_avg_tok_s ~= 1613.5`, `real 269.60` |
| OptM | `tpr=256,r=64,mqt=320` | PASS | `round_avg_tok_s ~= 1697.7`, `real 258.28` |
| OptN | `tpr=256,r=64,mqt=384` | PASS but unstable perf | `round_avg_tok_s ~= 1669.2`, `real 276.32` |
| OptN rerun2 | `tpr=256,r=64,mqt=384` | PASS | `round_avg_tok_s ~= 1699.3`, `real 259.14` |
| OptO | `tpr=256,r=64,mqt=512` | PASS | **`round_avg_tok_s ~= 1702.4`**, `real 261.61` |
| OptO rerun2 | `tpr=256,r=64,mqt=512` | PASS | `round_avg_tok_s ~= 1699.1`, **`real 255.93`** |
| OptP | `seq=2112,tpr=256,r=64,mqt=320` | PASS | `round_avg_tok_s ~= 1691.7`, `real 273.03` |
| OptQ | `tpr=256,r=64,mqt=640` | PASS | `round_avg_tok_s ~= 1675.8`, `real 267.84` |
| OptR | `tpr=256,r=64,mqt=768` | PASS | `round_avg_tok_s ~= 1649.5`, `real 266.99` |

Observations:
- All successful high-throughput `128x2048` runs in this band used `decode_iters=32` with `8192 new` tokens/iter.
- `max_queued_tokens` helped up to `512`; `640` and `768` both regressed.
- `mqt>=384` and `mqt=512` commonly showed a large first decode iteration (`iter_total_s_max ~27-30s`, high `submit_s_max`), but still delivered high aggregate throughput when steady-state remained fast.

### Current 128x2048 Best Stable Result

- Current best-throughput stable fixed-length configuration:
  - `page_size=64`, `max_pages=4224`, `max_tokens_per_round=256`, `max_rounds=64`, `max_queued_tokens=512`
  - `round_total_generated=262144` (full `128 * 2048`)
  - Best observed throughput: `round_avg_tok_s ~= 1702.4` (`OptO`)
  - Fastest observed wall clock for this family: `real 255.93` (`OptO rerun2`)

Representative logs:
- `/tmp/levanter_run_m9_lenfix_s128_baseline_ps64_p4224_tpr128_r64_mqt128.log` (decode + `Anomalies` fail)
- `/tmp/levanter_run_m9_lenfix_optA_s128_ps64_p4224_tpr64_r80_mqt128.log` (pass, non-fixed-length)
- `/tmp/levanter_run_m9_lenfix_optB_s128_ps64_p4224_tpr96_r80_mqt128.log` (pass, non-fixed-length)
- `/tmp/levanter_run_m9_lenfix_optC_s128_ps64_p4224_tpr256_r32_mqt256.log` (pass, early fixed-length reference)
- `/tmp/levanter_run_m9_lenfix_optG_s128_ps64_p4224_tpr256_r64_mqt256.log` (pass)
- `/tmp/levanter_run_m9_lenfix_optG_s128_ps64_p4224_tpr256_r64_mqt256_rerun2.log` (pass)
- `/tmp/levanter_run_m9_lenfix_optM_s128_ps64_p4224_tpr256_r64_mqt320.log` (pass)
- `/tmp/levanter_run_m9_lenfix_optN_s128_ps64_p4224_tpr256_r64_mqt384_rerun2.log` (pass)
- `/tmp/levanter_run_m9_lenfix_optO_s128_ps64_p4224_tpr256_r64_mqt512.log` (pass, best throughput)
- `/tmp/levanter_run_m9_lenfix_optO_s128_ps64_p4224_tpr256_r64_mqt512_rerun2.log` (pass, fastest wall clock)
- `/tmp/levanter_run_m9_lenfix_optQ_s128_ps64_p4224_tpr256_r64_mqt640.log` (pass, regression)
- `/tmp/levanter_run_m9_lenfix_optR_s128_ps64_p4224_tpr256_r64_mqt768.log` (pass, stronger regression)
- `/tmp/levanter_run_m9_lenfix_optD_s128_ps64_p4224_tpr128_r64_mqt256.log` (fail)
- `/tmp/levanter_run_m9_lenfix_optF_s128_ps64_p4224_tpr128_r64_mqt128.log` (fail)
