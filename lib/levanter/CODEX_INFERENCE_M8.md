# CODEX Inference M8: TPU Kernel Tuning (10x2048)

Status: COMPLETED (2026-02-07)

## Goal

Tune TPU ragged paged attention kernel settings for the M6 production workload:
- 10 prompts
- 2048 max_new_tokens
- 1 round

## Baseline Workload Definition

Scheduler/engine settings held constant from M6 final:
- `engine.max_seq_len: 2560`
- `engine.max_pages: 480`
- `engine.page_size: 64`
- `engine.max_seqs: 10`
- `engine.max_seqs_in_prefill: 10`
- `engine.max_prefill_size: 4096`
- `engine.max_queued_tokens: 64`
- `engine.max_tokens_per_round: 10`
- `engine.max_rounds: 64`
- `engine.reset_mode: physical`
- `engine.cleanup_mode: none`

## M8 Experiment Matrix

### B0 - Kernel OFF baseline

Config:
- `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m8_kernel_off.yaml`

Log:
- `/tmp/levanter_run_m8_b0_kernel_off_10prompts_2048.log`

Outcome:
- PASS
- `round_total_s=161.4312`
- `round_avg_tok_s=126.8652`
- `decode_avg_tok_s=177.0592`
- `prefill_extract_s=45.7971`

### K1 - Kernel ON (`q16`, `kv16`)

Config:
- `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m8_kernel_on_q16_kv16.yaml`

Log:
- `/tmp/levanter_run_m8_k1_kernel_on_q16_kv16_10prompts_2048.log`

Outcome:
- PASS
- `round_total_s=75.5820` (`2.136x`, `53.18%` faster than B0)
- `round_avg_tok_s=270.9639`
- `decode_avg_tok_s=521.0893`
- `prefill_extract_s=36.2829`

### K2 - Kernel ON (`q32`, `kv16`)

Config:
- `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m8_kernel_on_q32_kv16.yaml`

Log:
- `/tmp/levanter_run_m8_k2_kernel_on_q32_kv16_10prompts_2048.log`

Outcome:
- PASS
- `round_total_s=74.7381` (`2.160x`, `53.70%` faster than B0)
- `round_avg_tok_s=274.0234`
- `decode_avg_tok_s=521.8573`
- `prefill_extract_s=35.4927`

### K3 - Kernel ON (`q16`, `kv32`)

Config:
- `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m8_kernel_on_q16_kv32.yaml`

Log:
- `/tmp/levanter_run_m8_k3_kernel_on_q16_kv32_10prompts_2048.log`

Outcome:
- PASS
- `round_total_s=75.2413` (`2.146x`, `53.39%` faster than B0)
- `round_avg_tok_s=272.1910`
- `decode_avg_tok_s=519.7447`
- `prefill_extract_s=35.8401`

### K4 - Kernel ON (`q32`, `kv32`)

Config:
- `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m8_kernel_on_q32_kv32.yaml`

Log:
- `/tmp/levanter_run_m8_k4_kernel_on_q32_kv32_10prompts_2048.log`

Outcome:
- PASS
- `round_total_s=74.7579` (`2.159x`, `53.69%` faster than B0)
- `round_avg_tok_s=273.9511`
- `decode_avg_tok_s=520.3374`
- `prefill_extract_s=35.4017`

## Summary Table

| Run | Kernel | q_block | kv_pages_block | round_total_s | round_avg_tok_s | decode_avg_tok_s | vs B0 |
|---|---|---:|---:|---:|---:|---:|---|
| B0 | off | n/a | n/a | 161.4312 | 126.8652 | 177.0592 | baseline |
| K1 | on | 16 | 16 | 75.5820 | 270.9639 | 521.0893 | `2.136x`, `53.18%` faster |
| K2 | on | 32 | 16 | 74.7381 | 274.0234 | 521.8573 | `2.160x`, `53.70%` faster |
| K3 | on | 16 | 32 | 75.2413 | 272.1910 | 519.7447 | `2.146x`, `53.39%` faster |
| K4 | on | 32 | 32 | 74.7579 | 273.9511 | 520.3374 | `2.159x`, `53.69%` faster |

## M8 Final Recommendation

Selected final config:
- `config/sampler/sample_llama8b_multihost_real_10prompts_2048_reset_physical_round1_cleanup_none_noop_m8_final.yaml`

Rationale:
- Uses kernel-on (`ragged paged attention`) and captures the best measured setting from the sweep (`q32`, `kv16`).
- K2 and K4 are effectively tied; K2 is chosen as canonical because it is marginally fastest and keeps `kv_block_pages` smaller.

Validation run for canonical config:
- Log: `/tmp/levanter_run_m8_final_q32_kv16_10prompts_2048.log`
- Outcome: PASS
- `round_total_s=75.3241`, `round_avg_tok_s=271.8918`, `decode_avg_tok_s=518.2397`
- Throughput remains in the same band as K1/K2/K3/K4, confirming no regression from selecting the canonical file.

## Notes

- All experiments were run on real TPU (`v5p-16`) with foreground launcher.
- Tracker remained `noop` to isolate kernel effects and avoid logging-side noise seen in earlier milestones.
