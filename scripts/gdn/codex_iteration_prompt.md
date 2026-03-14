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
- the obvious outward `HackableDecoderLayer` / `HackableDecoderBlock` boundaries are demoted
- the first broad lower `G1` branch-wrapper family is also demoted
- `dispatch_shard_shell_delta_ms` is the mainline budget
- `ad_wrapper_shell_delta_ms` is the second budget
- `interaction_remainder_ms` and `xprof_idle_attributed_ms` are safety checks
- `S3` is complete
- `A3` is complete and rejected
- outward `P3` block-boundary variants are complete and rejected
- the first broad `G1` family is complete and rejected
- `D1` is complete as a partial diagnostic lead only
- the next required optimization slot is `D2`

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
   - `D2` branch-core sharding diagnostic with no new AD boundary
   - `A2` branch-core AD-boundary diagnostic only if there is already a positive `D2` on the same cut
   - `G2` lower primitive / `custom_partitioning` branch-core attempt only if the chosen `D2` cut is const-clean enough and a prior `D2` proved the sharding cut helps
   - `U` bounded CE side-arm only if fresh attribution re-implicates CE
4. Classify the change as exactly one of:
   - `branch-core sharding diagnostic`
   - `branch-core AD diagnostic`
   - `branch-core primitive`
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

### D2) Mainline sharding diagnostic
- Optimize a smaller branch-core island inside the hybrid-specific GDN branch.
- This cut must be smaller than the rejected broad `G1` wrappers.
- It should own only:
  - branch-core sharding contract
  - branch-core layout contract
  - the existing GDN leaf-kernel island or the smallest deterministic subgraph around it
- Do not introduce a new custom VJP or broad branch wrapper in `D2`.
- Carry forward the `D1` head-first layout idea where applicable.
- Reject the prototype if:
  - `dispatch_shard_shell_delta_ms` stays flat/up,
  - or `interaction_remainder_ms` grows,
  - or `xprof_idle_attributed_ms` stays flat/up when available,
  - even if `ad_wrapper_shell_delta_ms` improves.

### A2) AD-boundary diagnostic
- Only after a positive `D2` on the same branch-core cut.
- Keep the `D2` forward/sharding cut fixed.
- Change only the AD/manual-backward ownership on that already-proven cut.
- Reject if:
  - `dispatch_shard_shell_delta_ms` regresses versus the winning `D2`,
  - or `ad_wrapper_shell_delta_ms` stays flat/up,
  - or `xprof_idle_attributed_ms` grows.

### G2) Lower primitive / custom-partitioned branch-core cut
- Only after `D2` proves the smaller sharding cut helps and the chosen cut is const-clean enough.
- Do not spend the iteration here if the cut still closes over const arrays or would simply rebuild wrapper shell under a new name.

### U) CE side-arm
- Only if fresh evidence points back to CE.

Hard anti-goals:
- do not propose another outward `HackableDecoderLayer` boundary
- do not propose another outward `HackableDecoderBlock` wrapper
- do not propose another broad `G1` branch wrapper that owns forward + backward + sharding all at once
- do not spend a mainline pass on attribution refresh unless tooling changed
- do not spend mainline budget on checkpoint/remat toggles

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
- Expected full parity slice is `88 passed, 2 skipped` on the current inventory, or `90 passed, 2 skipped` when a new branch-layout test is intentionally added.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --all-transformer --no-sync`
- `uv run python scripts/gdn/gdnctl.py xprof-compare-runs --cluster us-east5-a --tpu-name "$USER-gdn" --before-run-target <attn_run_url_or_id> --after-run-target <hybrid_run_url_or_id> --normalize-positive-deltas-ms <interaction_remainder_ms> --output <xprof_json>`
- `uv run python scripts/gdn/gdnctl.py summary-attribution ...`
- `uv run python scripts/gdn/gdnctl.py lint-log`

Definition of done:
- one validated `D2` result, or
- one explicitly diagnostic `A2` / `G2` / `U` run that is justified by the current shell evidence,
- plus TPU correctness when code changed,
- plus profiling,
- plus a log entry with canonical shell-delta metrics.
