# CODEX Inference M9: `max_seqs` Scaling Beyond 10

Status: IN PROGRESS (started 2026-02-07)

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

## Experiment Matrix

| ID | Config | max_seqs | max_pages | prompts | status | round_total_s | notes |
|---|---|---:|---:|---:|---|---:|---|
| A1 | `...m9_s12_p480.yaml` | 12 | 480 | 12 | pending | - | |
| A2 | `...m9_s14_p480.yaml` | 14 | 480 | 14 | pending | - | |
| A3 | `...m9_s16_p480.yaml` | 16 | 480 | 16 | pending | - | |
| B1 | `...m9_s20_p720.yaml` | 20 | 720 | 20 | pending | - | |
| B2 | `...m9_s24_p864.yaml` | 24 | 864 | 24 | pending | - | |
| B3 | `...m9_s28_p1008.yaml` | 28 | 1008 | 28 | pending | - | |

## Logs

- A1: `/tmp/levanter_run_m9_a1_s12_p480.log`
- A2: `/tmp/levanter_run_m9_a2_s14_p480.log`
- A3: `/tmp/levanter_run_m9_a3_s16_p480.log`
- B1: `/tmp/levanter_run_m9_b1_s20_p720.log`
- B2: `/tmp/levanter_run_m9_b2_s24_p864.log`
- B3: `/tmp/levanter_run_m9_b3_s28_p1008.log`

## Notes

- For fair capacity comparison, each config sets prompt count equal to `max_seqs`.
- If a run fails, capture the first hard failure signature (`Out of free pages`, VMEM OOM, launch-group failure, etc.) and move to the next hypothesis.
