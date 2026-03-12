You are running one unattended hill-climb iteration for Gated DeltaNet TPU kernels.

Iteration metadata:
- Iteration: {{ITERATION}} / {{TOTAL_ITERATIONS}}
- Starting commit: {{HEAD_SHA}}

Primary objective:
- Improve end-to-end TPU training throughput for the fixed `3/4` GDN regime.
- `3/4` GDN is non-negotiable for this benchmark/model family.
- Do not optimize toward smaller GDN closed-call buckets alone.
- Optimize toward a shorter full-step critical path.

Repo context:
- GDN implementation: `lib/levanter/src/levanter/layers/gated_deltanet.py`
- Benchmark entrypoint: `experiments/speedrun/hackable_transformer_gdn/tiny_profile.py`
- Benchmark model: `experiments/speedrun/hackable_transformer_gdn/hackable_transformer_gdn.py`
- Correctness tests: `lib/levanter/tests/test_gdn_kernels.py`, `lib/levanter/tests/test_gdn_layer.py`
- Optimization recipe: `docs/recipes/optimize_gdn_pallas_tpu.md`
- Running log: `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Harness: `scripts/gdn/gdnctl.py`

Current diagnosis to optimize against:
- The deployable hybrid champion is still only about:
  - `throughput/mfu ~= 6.09`
  - `step_duration ~= 166.3 ms`
- The fresh attention-only control on the same benchmark family is about:
  - `throughput/mfu ~= 21.36`
  - `step_duration ~= 57.13 ms`
- The matched hybrid-vs-attention gap is about `110 ms`.
- The tracked train path explains only about `39%` of that gap.
- The broad `HackableDecoderLayer/*` family is too coarse to use as the main optimization target because the attention-only control also carries substantial normal layer-body compute there.
- The next actionable target is a **hybrid-specific generic shell delta budget** computed from matched hybrid vs attention-only attribution.
- The dominant hybrid-only generic shell buckets are categories such as:
  - `shard_map/pallas_call`
  - `closed_call/shard_map`
  - `transpose(jvp(...))/closed_call/shard_map`
  - `reshape`
  - `add_any`
  - `select_n`
  - `scatter-add`
- Therefore:
  - same-boundary GDN Pallas hillclimbing is demoted from the mainline,
  - broad `HackableDecoderLayer/*` attribution is only a coarse upper bound,
  - the next serious systems bet is a fixed 4-layer block with bespoke backward and explicit sharding,
  - kernel-local train-path wins are secondary unless they reduce the new hybrid-specific shell delta and the full step.

Required behavior for this iteration:
1. Read the latest log entries and identify the current validated baseline.
2. Generate a shortlist of 3 candidates with upside and risk.
3. Pick exactly one coverage slot:
   - `S3` hybrid-specific generic shell delta attribution,
   - `L3` fixed-4-layer block design / skeleton with manual VJP + explicit sharding contract,
   - `P3` first fixed-4-layer block prototype with manual VJP + explicit sharding contract,
   - `U` bounded CE side-arm only if CE is again implicated,
   - `diagnostic` only if you can explain why `S3/L3/P3/U` are lower information.
4. Classify the change as exactly one of:
   - `decoder shell attribution`
   - `whole-layer boundary`
   - `CE backend`
   - `diagnostic side-arm`
   - `inner kernel math`
5. Unless the run is an explicit CE side-arm, keep CE fixed at:
   - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
   - CE backward mode `pallas`
6. State what you expect to happen to:
   - `step_duration_ms`
   - `upper_bound_gap_ms`
   - `hybrid_generic_shell_delta_budget_ms`
   - `gap_explained_by_hybrid_generic_shell_delta`
   - `train_path_budget_ms`
   - `interaction_remainder_ms`
   - `remainder_budget_ms`
7. State whether the candidate should be rejected if:
   - `step_duration_ms` does not improve,
   - or `hybrid_generic_shell_delta_budget_ms` stays flat / grows,
   - or `interaction_remainder_ms` grows.
8. Implement the smallest code/config change needed to execute the chosen slot.
9. Validate correctness on TPU when code changed.
10. Run the required profile(s).
11. Update the log with measured metrics and commit exactly one validated result commit.

Coverage slots:

### S3) Hybrid-specific generic shell delta attribution (highest priority)
- No kernel changes are required.
- Refresh hybrid vs attention-only comparison under the same:
  - TPU family,
  - CE settings,
  - benchmark family,
  - step window,
  - fixed `3/4` GDN hybrid config.
- Prefer xprof-backed shell accounting for this slot:
  - compare matched XPlane artifacts with `gdnctl xprof-compare-runs`
  - feed that JSON into `gdnctl summary-attribution --xprof-compare-json ...`
- Record:
  - `step_duration_ms`
  - `train_path_budget_ms`
  - `decoder_layer_shell_budget_ms`
  - `hybrid_generic_shell_delta_budget_ms`
  - `dispatch_shard_shell_delta_ms`
  - `ad_wrapper_shell_delta_ms`
  - `layout_shell_delta_ms`
  - `residual_add_shell_delta_ms`
  - `remainder_budget_ms`
  - `interaction_remainder_ms`
  - `upper_bound_gap_ms`
  - `gap_explained_by_train_path`
  - `gap_explained_by_decoder_layer_shell`
  - `gap_explained_by_hybrid_generic_shell_delta`
  - `hybrid_generic_shell_delta_topk`
  - `remainder_topk`
  - `xprof_hybrid_generic_shell_delta_budget_ms`
  - `xprof_dispatch_shard_shell_delta_ms`
  - `xprof_ad_wrapper_shell_delta_ms`
  - `xprof_layout_shell_delta_ms`
  - `xprof_residual_add_shell_delta_ms`
  - `xprof_idle_attributed_ms`

### L3) Fixed-4-layer block design / skeleton
- This is a design-and-scaffold iteration, not a promotion candidate unless it also improves step time.
- Keep the exact `3 GDN + 1 attention` block pattern and benchmark config.
- Build the minimal specialized 4-layer block boundary with manual/custom VJP and an explicit sharding contract.
- Target the hybrid-specific generic shell families, not another chunk-kernel/tape tweak.

### P3) First fixed-4-layer block prototype
- One serious systems prototype only.
- Optimize the fixed `3 GDN + 1 attention` block as a unit.
- Own all three of:
  - the forward block boundary,
  - the backward / AD strategy,
  - and the sharding/layout contract.
- Prefer XLA-visible shell + Pallas leaf kernels initially, but do not leave backward to generic JAX AD.
- Do not combine this with unrelated kernel-local math changes.

### U) Bounded CE side-arm
- Only if the new attribution clearly points back to CE.
- Promotion bar is strict:
  - `ce_attributed_while_ms` must drop materially,
  - and `step_duration_ms` must improve,
  - with no shell-delta or interaction-remainder regression.

### Diagnostic only
- Use only if you can justify why `S3/L3/P3/U` would be lower information.
- Same-boundary GDN shell/tape/kernel work belongs here now unless it directly attacks the hybrid-specific shell delta or the interaction remainder.

Deprioritized unless explicitly justified:
- same-boundary GDN shell/tape rewrites,
- forward-only GDN improvements,
- kernel-local matmul/tiling/exp-diff work,
- more closed-call-only wins,
- broad `HackableDecoderLayer/*` accounting as the main shell target,
- CE micro-tuning without a clear end-to-end path.

Acceptance gate checklist (must appear in the iteration writeup):
- Correctness:
  - TPU tests command + result.
  - Treat TPU correctness as complete only when it runs through `gdnctl`'s remote TPU test wrapper
    with both `torch` and `transformers` installed. `87 passed, 2 skipped` or `88 passed, 2 skipped`
    is the expected full parity slice depending on the current test inventory.
- Perf:
  - `CE backend selected: ...`
  - `CE bwd mode: ...`
  - `gdn_layer_fraction: ...`
  - `Forward closed-call: ...`
  - `Backward closed-call: ...`
  - `Kernel budget: ...`
  - `Control budget: ...`
  - `Train-path budget: ...`
  - `Decoder-layer shell budget: ...`
  - `Hybrid generic shell delta budget: ...`
  - `Dispatch/shard shell delta budget: ...`
  - `AD/wrapper shell delta budget: ...`
  - `AD shell budget: ...`
  - `Sharding shell budget: ...`
  - `Layout shell budget: ...`
  - `Residual/add shell budget: ...`
  - `xprof hybrid generic shell delta budget: ...` when a matched XPlane pair is available
  - `xprof dispatch/shard shell delta budget: ...` when a matched XPlane pair is available
  - `xprof AD/wrapper shell delta budget: ...` when a matched XPlane pair is available
  - `xprof layout shell delta budget: ...` when a matched XPlane pair is available
  - `xprof residual/add shell delta budget: ...` when a matched XPlane pair is available
  - `xprof IDLE attributed remainder: ...` when a matched XPlane pair is available
  - `Step duration: ...`
  - `Remainder budget: ...`
  - `Interaction remainder: ...`
  - `Upper-bound gap: ...`
  - `Gap explained by train-path: ...`
  - `Gap explained by decoder-layer shell: ...`
  - `Gap explained by hybrid generic shell delta: ...`
  - `hybrid_generic_shell_delta_topk: ...`
  - `remainder_topk: ...`
  - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration`
- Governance:
  - reject any candidate where `step_duration_ms` does not improve unless it is explicitly diagnostic,
  - reject any candidate where `hybrid_generic_shell_delta_budget_ms` is flat/up unless it is explicitly diagnostic,
  - reject any candidate where `interaction_remainder_ms` grows unless it is explicitly diagnostic,
  - classify `train_path_budget_ms down, generic shell delta flat/up` as `namespace-only / renamed-bucket progress`,
  - do not leave `Commit: (pending)` or `Commit: this commit` in the log.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --all-transformer --no-sync`
- `uv run python scripts/gdn/gdnctl.py xprof-compare-runs --cluster us-east5-a --tpu-name "$USER-gdn" --before-run-target <attn_run_url_or_id> --after-run-target <hybrid_run_url_or_id> --normalize-positive-deltas-ms <interaction_remainder_ms> --output <xprof_json>`
- `uv run python scripts/gdn/gdnctl.py summary-attribution ...`
- `uv run python scripts/gdn/gdnctl.py lint-log`

Definition of done:
- either:
  - one current-baseline attribution result that materially improves understanding of hybrid-specific shell delta,
  - or one fixed-4-layer block design/skeleton result that makes the new boundary concrete,
  - or one fixed-4-layer block prototype result that improves the full step,
  - or one bounded CE side-arm that improves the full step,
- plus TPU correctness when code changed,
- plus one profiled run (or comparison) completed,
- plus a log entry with explicit hybrid-specific shell-delta metrics, not only coarse decoder-layer or train-path metrics.
