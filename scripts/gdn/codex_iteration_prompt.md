You are running one unattended hill-climb iteration for Gated DeltaNet TPU kernels.

Iteration metadata:
- Iteration: {{ITERATION}} / {{TOTAL_ITERATIONS}}
- Starting commit: {{HEAD_SHA}}

Primary objective:
- Improve end-to-end training throughput on TPU.
- Do not confuse lower GDN closed-call time with a faster step.
- The practical ceiling on this benchmark family is now the attention-only control:
  - `throughput/mfu ~= 21.09`
  - `step_duration ~= 57.86 ms`
- The current validated hybrid regime is only about:
  - `throughput/mfu ~= 6.09`
  - `step_duration ~= 166.3 ms`

Repo context:
- Kernel implementation: `lib/levanter/src/levanter/layers/gated_deltanet.py`
- Profile entrypoint: `experiments/speedrun/hackable_transformer_gdn/tiny_profile.py`
- Benchmark model config: `experiments/speedrun/hackable_transformer_gdn/hackable_transformer_gdn.py`
- Correctness tests: `lib/levanter/tests/test_gdn_kernels.py`, `lib/levanter/tests/test_gdn_layer.py`
- Optimization recipe: `docs/recipes/optimize_gdn_pallas_tpu.md`
- Running log: `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Infra CLI: `scripts/gdn/gdnctl.py`

Current diagnosis to optimize against:
- CE backend selection was the last giant false wall and is already mostly fixed.
- The remaining hybrid-vs-attention gap is the dominant fact:
  - same benchmark family,
  - same TPU family,
  - same CE settings,
  - but attention-only is still about `3.4x` MFU and about `108 ms` faster per step.
- More than half of that gap is outside the currently tracked train-path budget.
- Therefore:
  - same-boundary GDN Pallas hillclimbing is no longer the mainline,
  - remainder attribution and model-boundary measurement are the mainline,
  - CE is a bounded side-arm,
  - same-boundary GDN shell/tape moves are research branches only.

Required behavior for this iteration:
1. Read the latest log entries and identify the current validated baseline.
2. Generate a shortlist of 3 candidates with upside and risk.
3. Pick exactly one coverage slot:
   - `S` remainder attribution / hybrid-vs-attention accounting,
   - `T` model-boundary sweep (`gdn_layers_per_block`),
   - `U` bounded CE side-arm,
   - `diagnostic` only if you can explain why `S/T/U` are not the best use of budget.
4. Classify the change as exactly one of:
   - `attribution`
   - `model boundary`
   - `CE backend`
   - `outer control structure`
   - `inner kernel math`
5. Unless the point of the run is explicit CE A/B, keep CE fixed at:
   - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
   - CE backward mode `pallas`
6. State what you expect to happen to:
   - `step_duration_ms`
   - `train_path_budget_ms`
   - `remainder_budget_ms`
   - `upper_bound_gap_ms`
7. State whether the candidate should be rejected if:
   - `step_duration_ms` does not improve,
   - or `remainder_budget_ms` grows.
8. Implement the smallest code/config change needed to execute the chosen coverage slot.
9. Validate correctness on TPU when code changed.
10. Run the required profile(s).
11. Update the log with the measured metrics and commit exactly one validated result commit.

Coverage slots:

### S) Remainder attribution (highest priority)
- No kernel changes are required.
- Run the current hybrid head and the attention-only control back-to-back under the same:
  - TPU family,
  - CE settings,
  - benchmark family,
  - step window.
- Record:
  - `step_duration_ms`
  - `train_path_budget_ms`
  - `remainder_budget_ms`
  - `upper_bound_gap_ms`
  - `gap_explained_by_train_path`
  - `remainder_topk`
  - `gdn_layer_fraction`

### T) Model-boundary sweep
- No GDN kernel changes.
- Keep `gdn_block_size = 4`.
- Sweep `gdn_layers_per_block in {0, 1, 2, 3}`.
- Keep CE fixed at `pallas_tpu` + `pallas`.
- Goal: measure throughput penalty per GDN fraction.

### U) Bounded CE side-arm
- Only if you have a specific, narrow CE hypothesis.
- Promotion bar is strict:
  - `ce_attributed_while_ms` must drop materially,
  - and `step_duration_ms` must improve,
  - with no remainder-budget regression.

### Diagnostic only
- Use only if you can explain why `S/T/U` would be lower information than the diagnostic branch.
- Same-boundary Macro M/N/O/R work belongs here now unless it clearly attacks the measured remainder.

Deprioritized unless explicitly justified:
- same-boundary GDN shell/tape rewrites,
- forward-only GDN improvements,
- kernel-local matmul/tiling/exp-diff work,
- more closed-call-only wins,
- CE micro-tuning without a clear end-to-end path.

Acceptance gate checklist (must appear in the iteration writeup):
- Correctness:
  - TPU tests command + result.
  - Treat TPU correctness as complete only when it runs through `gdnctl`'s remote TPU test wrapper
    with both `torch` and `transformers` installed. `87 passed, 2 skipped` is the expected full parity slice.
- Perf:
  - `CE backend selected: ...`
  - `CE bwd mode: ...`
  - `gdn_layer_fraction: ...`
  - `Forward closed-call: ...`
  - `Backward closed-call: ...`
  - `while: ...`
  - `conditional: ...`
  - `Kernel budget: ...`
  - `Control budget: ...`
  - `Train-path budget: ...`
  - `Step duration: ...`
  - `Remainder budget: ...`
  - `Upper-bound gap: ...`
  - `Gap explained by train-path: ...`
  - `Remainder top-k: ...`
  - `throughput/mfu`, `throughput/tokens_per_second`, `throughput/duration`
- Governance:
  - reject any candidate where `step_duration_ms` does not improve unless it is explicitly a diagnostic slot,
  - reject any candidate where `remainder_budget_ms` grows unless it is a pure diagnostic slot,
  - classify `train_path_budget_ms down, step not faster` as `off-critical-path`,
  - do not leave `Commit: (pending)` or `Commit: this commit` in the log.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name "$USER-gdn" --tests both`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --all-transformer --no-sync`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-east5-a --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --ce-implementation pallas_tpu --ce-bwd-mode pallas --gdn-block-size 4 --gdn-layers-per-block 1 --no-sync`
- `uv run python scripts/gdn/gdnctl.py lint-log`

Definition of done:
- either:
  - one current-baseline attribution result that materially improves understanding of the unexplained gap,
  - or one model-boundary sweep result that quantifies throughput penalty per GDN fraction,
  - or one bounded CE result that improves end-to-end step time,
- plus TPU correctness when code changed,
- plus one profiled run (or sweep) completed,
- plus a log entry with explicit full-step metrics, not only GDN-local buckets.
