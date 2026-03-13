You are running one unattended hill-climb iteration for fixed-`3/4` GDN TPU training.

Iteration metadata:
- Iteration: {{ITERATION}} / {{TOTAL_ITERATIONS}}
- Starting commit: {{HEAD_SHA}}

Read first:
1. `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb_summary.md`
2. latest entries in `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
3. `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/docs/recipes/optimize_gdn_pallas_tpu.md`

Primary objective:
- reduce full-step TPU training time for the fixed `3/4` GDN regime
- optimize against the real critical path, not against old bucket names

Current regime:
- same-boundary GDN kernel hillclimbing is demoted
- `dispatch_shard_shell_delta_ms` is the mainline budget
- `ad_wrapper_shell_delta_ms` is the second budget
- `interaction_remainder_ms` and `xprof_idle_attributed_ms` are safety checks
- `S3` is complete
- `A3` is complete and rejected
- the next required optimization slot is `P3`

Repo context:
- GDN implementation: `lib/levanter/src/levanter/layers/gated_deltanet.py`
- benchmark entrypoint: `experiments/speedrun/hackable_transformer_gdn/tiny_profile.py`
- benchmark model: `experiments/speedrun/hackable_transformer_gdn/hackable_transformer_gdn.py`
- correctness tests: `lib/levanter/tests/test_gdn_kernels.py`, `lib/levanter/tests/test_gdn_layer.py`
- harness: `scripts/gdn/gdnctl.py`

Required behavior:
1. Read the current summary and latest log entries.
2. Generate a shortlist of 2-3 candidates with upside and risk.
3. Pick exactly one slot:
   - `P3` fixed `3 GDN + 1 attention` block with bespoke backward + explicit sharding/layout contract
   - `A3-diagnostic` only if you can explain why a materially different AD boundary is now higher information than `P3`
   - `S3-diagnostic` only if you changed attribution/xprof plumbing in this iteration
   - `U` bounded CE side-arm only if fresh attribution re-implicates CE
4. Classify the change as exactly one of:
   - `whole-layer boundary`
   - `diagnostic side-arm`
   - `CE backend`
5. Keep CE fixed unless this is an explicit CE side-arm:
   - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
   - CE backward mode `pallas`
6. Before writing code, state the expected direction of:
   - `step_duration_ms`
   - `dispatch_shard_shell_delta_ms`
   - `ad_wrapper_shell_delta_ms`
   - `hybrid_generic_shell_delta_budget_ms`
   - `interaction_remainder_ms`
   - `xprof_idle_attributed_ms`
7. Reject the candidate unless it improves the step and the shell budgets, unless the run is explicitly diagnostic.
8. Implement the smallest change needed for the chosen slot.
9. Run TPU correctness if code changed.
10. Run the required profile(s).
11. Update the hillclimb log and commit exactly one validated result commit.

Slots:

### P3) Mainline prototype
- Optimize the fixed `3 GDN + 1 attention` block as one unit.
- The block must own:
  - forward boundary
  - backward/custom VJP contract
  - sharding contract
  - layout contract
- Reuse current leaf kernels first.
- Do not let generic JAX AD or generic sharding rebuild the same shell outside the new boundary.
- Reject the prototype if:
  - `dispatch_shard_shell_delta_ms` stays flat/up,
  - or `ad_wrapper_shell_delta_ms` grows,
  - or `interaction_remainder_ms` grows,
  - or `xprof_idle_attributed_ms` stays flat/up when available,
  - even if old train-path buckets disappear.

### A3-diagnostic
- Only if the backward/sharding contract is materially different from the rejected A3 attempt.
- If it is just another outward layer-level manual-VJP boundary, do not do it.

### S3-diagnostic
- Only if the iteration changes xprof extraction, matched-pair attribution, or shell grouping logic.
- Do not spend another mainline iteration here just to reconfirm the shell ranking.

### U) CE side-arm
- Only if fresh evidence points back to CE.

Required metrics in the writeup:
- `CE backend selected`
- `CE bwd mode`
- `gdn_layer_fraction`
- `step_duration_ms`
- `forward_closed_call_ms`
- `backward_closed_call_ms`
- `train_path_budget_ms`
- `decoder_layer_shell_budget_ms`
- `hybrid_generic_shell_delta_budget_ms`
- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `layout_shell_delta_ms`
- `residual_add_shell_delta_ms`
- `interaction_remainder_ms`
- `upper_bound_gap_ms`
- `gap_explained_by_train_path`
- `gap_explained_by_decoder_layer_shell`
- `gap_explained_by_hybrid_generic_shell_delta`
- `hybrid_generic_shell_delta_topk`
- `remainder_topk`
- `throughput/mfu`
- `throughput/tokens_per_second`
- `throughput/duration`
- xprof metrics when a matched XPlane pair is available:
  - `xprof_dispatch_shard_shell_delta_ms`
  - `xprof_ad_wrapper_shell_delta_ms`
  - `xprof_layout_shell_delta_ms`
  - `xprof_residual_add_shell_delta_ms`
  - `xprof_idle_attributed_ms`

Correctness gate:
- TPU correctness is only considered complete when run through the `gdnctl` remote TPU wrapper with `torch` and `transformers` installed.
- Expected full parity slice is `88 passed, 2 skipped` on the current inventory.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --all-transformer --no-sync`
- `uv run python scripts/gdn/gdnctl.py xprof-compare-runs --cluster us-east5-a --tpu-name "$USER-gdn" --before-run-target <attn_run_url_or_id> --after-run-target <hybrid_run_url_or_id> --normalize-positive-deltas-ms <interaction_remainder_ms> --output <xprof_json>`
- `uv run python scripts/gdn/gdnctl.py summary-attribution ...`
- `uv run python scripts/gdn/gdnctl.py lint-log`

Definition of done:
- one validated `P3` result, or
- one explicitly diagnostic run that changes the shell attribution/tooling enough to justify itself,
- plus TPU correctness when code changed,
- plus profiling,
- plus a log entry with canonical shell-delta metrics.
