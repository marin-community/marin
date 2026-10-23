# marin_belay_v1_clean log

- 2026-10-23 14:25:26 worktree created from marin_belay_v1_w2026-10-19 (87dfa66f24)
- 2026-10-23 14:34:19 baselines: hlo_before/hashes.txt (6 cases), loss2_before.txt (tiny baseline/kma/belay, pooled-wave MoE on CPU)
- 2026-10-23 14:41:21 INVENTORY.md written (decisions.py over inventory.json); recent usage from 1029 Iris launches since 2026-10-17
- 2026-10-23 14:45:08 baseline tests: 5 failed (4 stale _apply_runtime_defaults calls in tests/grug/test_fast_track_ragged_moe.py; test_latent_out_tail worker crash), 59 passed, 1 skipped
- 2026-10-23 14:55:42 model.py rewritten (scratchpad/clean/model_new.py, 2407 lines); commit order: GB200 removal (launcher/trainer/recipe) first, then dead flags incl. now-unreachable GB200 model fields
- 2026-10-23 15:00:52 pass 1 (GB200/MFU launcher+trainer+recipe removal): 20 HLO cases identical to hlo_before
