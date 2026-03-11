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
- The attention-only control on the same benchmark family is about:
  - `throughput/mfu ~= 21.09`
  - `step_duration ~= 57.86 ms`
- With `3/4` GDN fixed, the main question is no longer GDN fraction.
- The main question is why the hybrid GDN-bearing decoder layers carry such a large shell/scaffolding tax.
- The tracked train path explains only about `39%` of the hybrid-vs-attention gap.
- The dominant unexplained buckets are decoder-layer shell categories such as:
  - `HackableDecoderLayer/shard_map/pallas_call`
  - `HackableDecoderLayer/closed_call/shard_map`
  - `transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map`
  - `HackableDecoderLayer/reshape`
  - `transpose(jvp(HackableTransformer))/HackableDecoderLayer/add_any`
- Therefore:
  - same-boundary GDN Pallas hillclimbing is demoted from the mainline,
  - decoder-layer-shell attribution is the mainline,
  - whole-layer boundary prototypes are the next serious systems bet,
  - kernel-local train-path wins are secondary unless they reduce the whole-layer shell budget and step time.

Required behavior for this iteration:
1. Read the latest log entries and identify the current validated baseline.
2. Generate a shortlist of 3 candidates with upside and risk.
3. Pick exactly one coverage slot:
   - `S2` decoder-layer-shell attribution widening,
   - `L2` specialized whole-layer design/skeleton work,
   - `P2` first whole-layer prototype,
   - `U` bounded CE side-arm only if CE is again implicated,
   - `diagnostic` only if you can explain why `S2/L2/P2/U` are lower information.
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
   - `decoder_layer_shell_budget_ms`
   - `gap_explained_by_decoder_layer_shell`
   - `train_path_budget_ms`
   - `remainder_budget_ms`
7. State whether the candidate should be rejected if:
   - `step_duration_ms` does not improve,
   - or `decoder_layer_shell_budget_ms` stays flat / grows,
   - or `remainder_budget_ms` grows.
8. Implement the smallest code/config change needed to execute the chosen slot.
9. Validate correctness on TPU when code changed.
10. Run the required profile(s).
11. Update the log with measured metrics and commit exactly one validated result commit.

Coverage slots:

### S2) Decoder-layer-shell attribution (highest priority)
- No kernel changes are required.
- Refresh hybrid vs attention-only comparison under the same:
  - TPU family,
  - CE settings,
  - benchmark family,
  - step window,
  - fixed `3/4` GDN hybrid config.
- Record:
  - `step_duration_ms`
  - `train_path_budget_ms`
  - `decoder_layer_shell_budget_ms`
  - `ad_shell_budget_ms`
  - `sharding_shell_budget_ms`
  - `layout_shell_budget_ms`
  - `remainder_budget_ms`
  - `upper_bound_gap_ms`
  - `gap_explained_by_train_path`
  - `gap_explained_by_decoder_layer_shell`
  - `decoder_layer_shell_topk`
  - `remainder_topk`

### L2) Specialized whole-layer design / skeleton
- This is a design-and-scaffold iteration, not a promotion candidate unless it also improves step time.
- Keep the same `3/4` GDN math and benchmark config.
- Build the minimal specialized GDN-bearing decoder-layer boundary with manual/custom VJP at the layer boundary.
- Target the whole `HackableDecoderLayer` shell, not another chunk-kernel/tape tweak.

### P2) First whole-layer prototype
- One serious systems prototype only.
- Optimize the entire GDN-bearing decoder-layer boundary:
  - QKV / gate projections,
  - conv / RMSNorm / gating path,
  - chunked GDN primitive,
  - output projection,
  - residual add / layer output,
  - backward boundary.
- Prefer XLA-first shell + Pallas leaf kernels initially.
- Do not combine this with unrelated kernel-local math changes.

### U) Bounded CE side-arm
- Only if the new attribution clearly points back to CE.
- Promotion bar is strict:
  - `ce_attributed_while_ms` must drop materially,
  - and `step_duration_ms` must improve,
  - with no decoder-layer-shell or remainder regression.

### Diagnostic only
- Use only if you can justify why `S2/L2/P2/U` would be lower information.
- Same-boundary GDN shell/tape/kernel work belongs here now unless it directly attacks the decoder-layer shell budget.

Deprioritized unless explicitly justified:
- same-boundary GDN shell/tape rewrites,
- forward-only GDN improvements,
- kernel-local matmul/tiling/exp-diff work,
- more closed-call-only wins,
- model-boundary/GDN-fraction sweeps as a product recommendation,
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
  - `AD shell budget: ...`
  - `Sharding shell budget: ...`
  - `Layout shell budget: ...`
  - `Step duration: ...`
  - `Remainder budget: ...`
  - `Upper-bound gap: ...`
  - `Gap explained by train-path: ...`
  - `Gap explained by decoder-layer shell: ...`
  - `decoder_layer_shell_topk: ...`
  - `remainder_topk: ...`
  - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration`
- Governance:
  - reject any candidate where `step_duration_ms` does not improve unless it is explicitly diagnostic,
  - reject any candidate where `decoder_layer_shell_budget_ms` is flat/up unless it is explicitly diagnostic,
  - reject any candidate where `remainder_budget_ms` grows unless it is explicitly diagnostic,
  - classify `train_path_budget_ms down, shell flat/up` as `wrong-boundary progress`,
  - do not leave `Commit: (pending)` or `Commit: this commit` in the log.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --all-transformer --no-sync`
- `uv run python scripts/gdn/gdnctl.py summary-attribution ...`
- `uv run python scripts/gdn/gdnctl.py lint-log`

Definition of done:
- either:
  - one current-baseline attribution result that materially improves understanding of decoder-layer shell cost,
  - or one whole-layer design/skeleton result that makes the new boundary concrete,
  - or one whole-layer prototype result that improves the full step,
  - or one bounded CE side-arm that improves the full step,
- plus TPU correctness when code changed,
- plus one profiled run (or comparison) completed,
- plus a log entry with explicit whole-layer shell metrics, not only train-path metrics.
