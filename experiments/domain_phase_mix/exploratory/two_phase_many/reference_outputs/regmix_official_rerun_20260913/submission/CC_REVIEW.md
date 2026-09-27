## Verdict

**Acceptable. No blocking issues.** The launcher is a faithful clone of the completed midpoint template with only mixture, identity, seed-grouping, and count changed, and every guard that made the midpoint batch safe is present and correctly re-parameterized.

## Blocking issues

None found.

## What I verified in scope

**Count and uniqueness.** 4 runs, hard-enforced three ways: `expected_run_count=len(candidate_ids)` per group (`launch_delphi_regmix_reference_3e18.py:64`), `len(all_specs) != MAX_CONCURRENT` (`:128`), and `--max-concurrent` must equal 4 exactly (`:159`). The 4 mixtures are provably runtime-distinct — `load_candidate_mixtures` builds `alias_map` from the count vectors and rejects anything that isn't the identity map (`launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18.py:304-311`), and `expected_alias_map` here is identity (`:63`). All four blocks sum to exactly 2048 with `weight == count/2048` enforced per bucket (`:279-284`).

**Seeds and identity.** Trainer seed 0 on all four; data seed 666200 on both Uncheatable arms, 662009 on both suite arms — the same two constants every sibling objective-matched launcher uses. `run_id` 7470000/7470001 and 7470100/7470101 (`:44-45`) occupy a fresh block: 7_430 (matched_olmix), 7_440 (comparator), 7_450 (olmix_scaling), 7_460 (path_midpoints) are taken, 7_470 is not. Grep for `rgref` / `delphi_regmix_reference` returns only the new launcher, so run names, eval step names (`evaluation/olmo_base_eval_table9/t9rgref_*`), and the experiment tree are globally new. `WeightConfig.run_id` is inert metadata (`config.py:289-291`), so run_id does not perturb sampling.

**Blend identity — the decisive check.** I reconstructed the second parent from `2·midpoint − endpoint` and compared it to `policy_weights.csv`. Both arms pair each objective's RegMix endpoint with **that same objective's** MARINER endpoint (no cross-objective swap): e.g. Uncheatable `dolma3_cc/literature_high` (70 + 251)/2 = 160.5 → 160, suite `dolma3_stack_edu` (421 + 293)/2 = 357 exact. Sixteen half-integer ties occur; 8 break up and 8 break down, which is exactly why both blends still total 2048 with no re-rounding — corroborating `candidate_receipt.json:14`.

**cap64 inactive.** `f"cap{cap:02d}"` renders "cap64" so the naming assert passes (`:274`), and the cap assert compares against 64 while the largest actual epochs are 9.93 / 9.70 / 8.22 / 7.58 (`validation_review_plan.json:22,32,42`). Confirmed loader metadata, not a training cap.

**Region/runtime.** Parent interactive, 1 CPU / 4 GB, us-east5-a (`submit.sh:4-6`); children v6e-8 in us-east5-b for both training and Table-9 (`launch_delphi_augmented_swarm_3e18.py:110`). `marin_prefix_for_region("us-east5")` → `gs://marin-us-east5`, matching the `-e MARIN_PREFIX`, and the launcher hard-fails on any hardware, candidate-table, sha, analysis-path, or block-size override (`:157-174`). `preflight_graph.py:91-93` asserts `regions == ["us-east5"]` and `zone == "us-east5-b"` on both train and eval resources, and `:128` asserts every resolved GCS URI is under the east5 prefix (40 URIs, all east5).

**Evaluation dependencies.** All present: 16 Paloma + 7 `uncheatable_eval` inline sets and the Table-9 request set resolve to east5 paths (`full_graph_regional_preflight.json:217-257`); `inline_uncheatable_count: 7` on every run, matching the seven frozen objective weights; eval checkpoint is `…/hf/step-3006` with `expected_checkpoint_step == 3006`, `train_steps == 3007`, and the training output path present in the eval's `dep_paths` (`preflight_graph.py:94-96`).

**Resumability / duplicate outputs.** Output paths are content-versioned and recorded (`…-b5a5dc`, `-80405e`, `-02614e`, `-330587`; evals `-cef482`, `-c0cb54`, `-355ead`, `-9f20a2`; manifests `-d6571c`, `-8ab24f`). All ten step names are distinct, and none collide with an existing tree. The resume path is sound as long as nothing in the hash inputs moves — see item 4 below.

**Launch scope.** `submit.sh` is one `iris job run`; the graph is exactly 10 steps (2 manifests, 4 trainings, 4 evals), 1.2e19 nominal training FLOPs. No argument path can widen it without editing the launcher, which changes `launcher_sha256`.

## Non-blocking observations

- `region_validation.json:1` reports `"child_table9_tpu_zone": null`. That is a reporting gap in that validator, not an unset zone — `TABLE9_EVAL_RESOURCES` pins us-east5-b at `launch_delphi_augmented_swarm_3e18.py:110`, and the preflight asserts it directly. No action needed.
- The tightest invariant in the launch is the materialization-accounting assert (`launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18.py:340`), whose tolerance is absolute (1e-5). The panel/runtime offset is structurally ~5.0e-7 *relative* in both batches, so the 9.93-epoch endpoint consumes ~50% of the budget vs ~20% for the midpoint batch's largest row. It passes, and it fails closed at launch before any TPU is acquired — just noting this is the assert that would fire first if the bucket token-count table were ever regenerated.
- The seven aggregation weights in `prediction_freeze.json:9-17` are byte-identical to the long-frozen vector in `prepare_single_phase_heldout_benchmark_20260902.py:103-109`; they sum to 1 + 2.05e-8, which is pre-existing, shared with the audited midpoint batch, and cancels across arms.

## Required before submission

1. Re-run `preflight_graph.py` immediately before submit and require `launcher_sha256 == 856769a8cca7117e8f6b2753165a499dcdfdf320c55647eb218bc85045e5e90d`, `candidate_sha256 == 7c4b8b00…c3899`, and the eight output-path suffixes unchanged. The recorded preflight is only as good as those three hashes.
2. Confirm `candidate_weights.csv` survives bundling — `--bundle-include` and the negative-lookahead exclude (`submit.sh:8,12`) both permit it, but the launcher hard-fails at `load_candidate_mixtures` if the file is absent or altered, so verify rather than rely on precedence between the two flags.
3. Export `WANDB_API_KEY` in the submitting shell; `submit.sh:18` fails closed without it, before the job is created.
4. Do not touch `launch_delphi_regmix_reference_3e18.py`, `launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18.py`, `launch_delphi_augmented_swarm_3e18.py`, or `candidate_weights.csv` between the preflight and any resubmit. Those recorded `-hash` suffixes are the resume keys; any edit forks a new output path and silently retrains instead of resuming.

Optional housekeeping: `validation_review_plan.json:2` still reads `"status": "not_submitted"`, which will be stale once the job is created.
