# Recipe: Optimize Gated DeltaNet TPU Kernels

## Current Regime

Use this recipe for the current fixed-`3/4` GDN TPU hillclimb.

Read these in order before changing code:
1. `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb_summary.md`
2. `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md` (latest entries only)
3. `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/scripts/gdn/codex_iteration_prompt.md`

This recipe is intentionally current-state-only. Historical regimes and dead ends belong in the hillclimb log, not in the main control recipe.

## Fixed Constraints

- Benchmark/model family keeps `3/4` GDN. Reducing GDN fraction is not allowed.
- The attention-only run is a ceiling/diagnostic control, not a product option.
- CE stays fixed unless an iteration is explicitly a CE side-arm:
  - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
  - CE backward mode `pallas`

## Current Diagnosis

Matched hybrid vs attention-only evidence now splits the gap into:

- `train_path_budget_ms`
- `hybrid_generic_shell_delta_budget_ms`
- `interaction_remainder_ms`

Current best accounting says:

- tracked train path is real, but not dominant enough to be the mainline target
- the hybrid-specific shell tax is the next actionable target
- inside that shell tax:
  - `dispatch_shard_shell_delta_ms` is the largest family
  - `ad_wrapper_shell_delta_ms` is the second family
- xprof shows the remaining manifestation is mostly `IDLE`, which should be interpreted as waiting / serialization / shell tax, not hidden useful compute

Therefore:

- same-boundary GDN Pallas kernel hillclimbing stays demoted
- broad `HackableDecoderLayer/*` accounting is too coarse to be the main optimization target
- the obvious outward `HackableDecoderLayer` / `HackableDecoderBlock` boundaries are demoted after the `A3/P3` failures
- the first broad `G1` branch-wrapper family is also demoted after repeated regressions
- the next serious mainline move is a smaller branch-core sharding diagnostic, not another broad wrapper or manual-backward move

## Mainline Metrics

Primary decision metrics:

- `step_duration_ms`
- `upper_bound_gap_ms`
- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `hybrid_generic_shell_delta_budget_ms`
- `interaction_remainder_ms`
- `xprof_idle_attributed_ms` when a matched XPlane pair is available

Secondary diagnostics:

- `train_path_budget_ms`
- `decoder_layer_shell_budget_ms`
- `forward_closed_call_ms`
- `backward_closed_call_ms`

Shell sub-budgets to record:

- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `layout_shell_delta_ms`
- `residual_add_shell_delta_ms`

If xprof is available, also record:

- `xprof_dispatch_shard_shell_delta_ms`
- `xprof_ad_wrapper_shell_delta_ms`
- `xprof_layout_shell_delta_ms`
- `xprof_residual_add_shell_delta_ms`
- `xprof_idle_attributed_ms`

## Promotion Rules

Reject a candidate unless it improves the real step.

Hard rejection rules:

- reject if `step_duration_ms` does not improve, unless the run is explicitly diagnostic
- reject if `dispatch_shard_shell_delta_ms` stays flat or grows, unless the run is explicitly diagnostic
- reject if `ad_wrapper_shell_delta_ms` grows, unless the run is explicitly diagnostic
- reject if `hybrid_generic_shell_delta_budget_ms` grows, unless the run is explicitly diagnostic
- reject if `interaction_remainder_ms` grows, unless the run is explicitly diagnostic
- reject if `xprof_idle_attributed_ms` grows when an XPlane pair is available, unless the run is explicitly diagnostic

Classification rules:

- `train_path_budget_ms down, shell delta flat/up` => wrong-boundary / renamed-bucket progress
- `shell delta down, xprof_idle_attributed_ms flat/up` => waiting/serialization still dominant
- `old bucket names disappear but canonical shell delta grows` => not a win
- `AD/layout improves but dispatch/shard stays flat/up` => diagnostic lead only, not a mainline win

## Coverage Sequencing

Current coverage status:

- `S3`: complete
  - hybrid-specific shell delta attribution is established
  - use only when attribution tooling itself changes
- `A3`: complete and rejected
  - outward layer-level manual-backward boundary increased shell tax badly
- outward `P3` block-boundary family: complete and rejected
  - outward block custom-VJP / scan-switch / shard-map / custom-partitioning / no-checkpoint variants all failed
- broad `G1` family: complete and rejected
  - broad branch wrappers re-emitted shell under new branch-local names and slowed the step
- `D1`: partial positive lead only
  - it improved AD/layout behavior somewhat, but not the main dispatch/shard budget

Next required optimization slot:

- `D2`

`D2` definition:

- smaller branch-core sharding/layout ownership cut inside the hybrid-specific GDN branch
- reuse current GDN leaf kernels first
- no new custom VJP on the first `D2` pass
- carry forward head-first layout when it helps
- the goal is to move `dispatch_shard_shell_delta_ms` first

`A2` definition:

- only after a positive `D2` on the same cut
- keep the `D2` forward/sharding cut fixed
- change only AD/manual-backward ownership on that proven cut

Do not spend mainline budget on:

- another `S3` refresh just to reconfirm the same ranking
- another outward `A3` or `P3` retry
- another broad `G1` wrapper
- same-boundary GDN tape/kernel tweaks
- checkpoint/remat toggles as a mainline strategy
- CE micro-tuning unless fresh attribution points back to CE

## Required Validation

When code changes:

1. TPU correctness through the `gdnctl` remote wrapper
2. one profiled hybrid run
3. one matched attention-only control run when shell-delta attribution is part of the decision
4. xprof compare when a matched XPlane pair is available
5. one log entry with explicit shell-delta metrics

Full TPU parity expectation:

- `88 passed, 2 skipped` on the current inventory
- `90 passed, 2 skipped` if the iteration intentionally adds one new branch-layout parity test

Treat the old incomplete remote setup as invalid evidence if it silently skipped HF parity because `torch` or `transformers` were missing.

## Preferred Commands

```bash
uv run python scripts/gdn/gdnctl.py dev-tpu-test \
  --cluster us-east5-a \
  --tpu-name "$USER-gdn" \
  --tests both
```

```bash
uv run python scripts/gdn/gdnctl.py dev-tpu-profile \
  --cluster us-east5-a \
  --tpu-name "$USER-gdn" \
  --tpu v5p-8 \
  --size 130m \
  --num-steps 20 \
  --profile-start-step 2 \
  --profile-num-steps 6 \
  --batch-size 8 \
  --ce-implementation pallas_tpu \
  --ce-bwd-mode pallas \
  --no-sync
```

```bash
uv run python scripts/gdn/gdnctl.py dev-tpu-profile \
  --cluster us-east5-a \
  --tpu-name "$USER-gdn" \
  --tpu v5p-8 \
  --size 130m \
  --num-steps 20 \
  --profile-start-step 2 \
  --profile-num-steps 6 \
  --batch-size 8 \
  --ce-implementation pallas_tpu \
  --ce-bwd-mode pallas \
  --all-transformer \
  --no-sync
```

```bash
uv run python scripts/gdn/gdnctl.py xprof-compare-runs \
  --cluster us-east5-a \
  --tpu-name "$USER-gdn" \
  --before-run-target <attn_run_url_or_id> \
  --after-run-target <hybrid_run_url_or_id> \
  --normalize-positive-deltas-ms <interaction_remainder_ms> \
  --output <xprof_json>
```

## Definition Of Done

A good iteration is one of:

- a `D2` diagnostic that materially reduces `dispatch_shard_shell_delta_ms` and improves the step
- an `A2` follow-up on a winning `D2` cut that reduces `ad_wrapper_shell_delta_ms` without giving back dispatch/shard wins
- a lower primitive experiment justified by a prior positive `D2` on the same const-clean cut
- a tooling-attribution change that materially improves understanding and is explicitly marked diagnostic

Everything else should be logged and rejected cleanly.
