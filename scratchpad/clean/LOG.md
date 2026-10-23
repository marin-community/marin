# marin_belay_v1_clean log

- 2026-10-23 14:25:26 worktree created from marin_belay_v1_w2026-10-19 (87dfa66f24)
- 2026-10-23 14:34:19 baselines: hlo_before/hashes.txt (6 cases), loss2_before.txt (tiny baseline/kma/belay, pooled-wave MoE on CPU)
- 2026-10-23 14:41:21 INVENTORY.md written (decisions.py over inventory.json); recent usage from 1029 Iris launches since 2026-10-17
- 2026-10-23 14:45:08 baseline tests: 5 failed (4 stale _apply_runtime_defaults calls in tests/grug/test_fast_track_ragged_moe.py; test_latent_out_tail worker crash), 59 passed, 1 skipped
- 2026-10-23 14:55:42 model.py rewritten (scratchpad/clean/model_new.py, 2407 lines); commit order: GB200 removal (launcher/trainer/recipe) first, then dead flags incl. now-unreachable GB200 model fields
- 2026-10-23 15:00:52 pass 1 (GB200/MFU launcher+trainer+recipe removal): 20 HLO cases identical to hlo_before
- 2026-10-23 15:06:58 pass 2 model rewrite: all 20 HLO cases identical (fixed: final-gate z order, per-pass stack order)
- 2026-10-23 15:15:08 pass 2 (dead flags): 19/19 HLO cases identical, tiny 2-step losses identical, fast_track+grug tests 46 passed 1 skipped (test_kda_local erase-gate options dropped)
- 2026-10-23 15:22:38 phase B (HLO no-op residue: x*1.0, +0/-0, ones-multiplied head gate + zero head_skip_frac metric, leaky_relu(0)->relu, size-1 pass stack/reduce, unused entropy-scan carries): 2-step tiny losses AND full train state bitwise identical; op diff in phaseB_opdiff.txt; tests 46 passed
- 2026-10-23 15:23:21 smoke submitted: /larry/lc1-clean-smoke-d512-coord (cw-us-east-02a, interactive, --stop-after-steps 5 --temporary-checkpoints-only) at 72cc4e61d9
- 2026-10-23 15:29:23 smoke 1 failed: StateCallbackRunner needs eval_model_getter (dropped with the EMA getter); fixed
- 2026-10-23 15:34:42 reference worktree /Users/larry/marin_wt_clean_ref (detached: w2026-10-19 + CW forwarding) for an A/B smoke
- 2026-10-23 15:45 smoke 2/3 (clean) and ref (w2026-10-19 + CW forwarding) on cw-us-east-02a, d512 Ckpt 1, 5 steps + final eval + TTL checkpoint, all succeeded.
  train/loss steps 0-4: clean2 9.8450861 9.8372431 9.8392935 9.8392620 9.8240891; clean3 9.8450861 9.8372431 9.8393068 9.8392725 9.8240862;
  ref 9.8450871 9.8372450 9.8393021 9.8392744 9.8240919 (spread = run-to-run GPU nondeterminism). Paloma macro 9.80804 / 9.80804 / 9.80805.
