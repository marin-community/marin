I read the round-2 handoff, then the current files directly. Note up front: this session has no shell, so I verified by reading the exact current sources — I did not re-run the 71 tests, `pyrefly`, or recompute the SHA-256 digests the handoff cites.

# Verdict: **GO**

Both blockers are fixed. The strengthened idempotence checks are internally consistent between collector and analyzer. No new production blockers.

## Blocker 1 — component ordering: FIXED

`analyze_delphi_tpp40_bridge_acceptance.py:490-498` now does exactly what the handoff claims: an exact name-set comparison, then canonical reconstruction through `bridge_eval._uncheatable_metrics`, which iterates `EXPECTED_UNCHEATABLE_NAMES` (`launch_delphi_tpp40_bridge_uncheatable_eval.py:330-341`) and builds `component_metrics` in producer order before the float32 `np.mean` at line 344-346. So the writer's `sort_keys=True` (`launch_...:370`) can no longer perturb the reduction order. `_validate_uncheatable_payload` returns the *reconstructed* dict (`analyze_...:505`), so the downstream component deltas at `analyze_...:843-846` are canonical too.

The regression test is load-bearing, not decorative. `test_uncheatable_payload_round_trip_accepts_sorted_json_keys` (`tests/test_analyze_delphi_tpp40_bridge_acceptance.py:235-242`) goes through the real `bridge_eval._write_json` → `acceptance._read_json` path, and `UNCHEATABLE_SUBSETS` (`experiments/datasets/uncheatable.py:23-31`) is in near-reverse alphabetical order, so `sort_keys=True` genuinely permutes all seven keys. Under the old code the float32 mean would be computed in sorted order and the macro equality at `launch_...:356` would fail.

I also checked the same bug class on the Table-9 side, since `analyze_...:537` requires insertion order. It's safe: `assemble_table9` (`lib/marin/src/marin/evaluation/olmo_base_eval/aggregate.py:51-54`) builds in `table9_components()` order, `_write_results` (`.../olmo_base_eval/run.py:310`) does *not* sort keys, and `table9_macro` looks up by name so its sum order is canonical regardless of input dict order.

## Blocker 2 — `all([])`: FIXED as reported

`analyze_...:932-942`. The vacuous-truth path is closed: `loading_is_only_incomplete` now requires `bool(loading_errors)`, and the new `missing_idempotence_only` branch requires `numerical_acceptance_passed`. The exact reported failure — everything materialized, thresholds violated, `--allow-incomplete` passed — now raises. Covered by `test_allow_incomplete_does_not_mask_complete_numerical_failure` and `test_allow_incomplete_accepts_only_missing_outputs_or_idempotence` (`tests/test_analyze...:245-266`).

One residual case survives, and it is **not** a production blocker but you should know about it. When *some* results are still missing, a threshold failure on a pair that is already fully materialized is still suppressed. Concretely: East@21855 and Europe@21855 both land with |Δ| = 0.01, Europe@27335 and Europe Table-9 not yet written → every loading error is `"result is missing"` → `loading_is_only_incomplete` is True → `analyze_...:997` does not raise, exit 0. Phase 0 is a decided coordinate at that point per the contract's `uncheatable_any_row_definition`, so this is a real result being reported as "still waiting."

Production is not at risk: the report records `uncheatable.phase_0.threshold.passed: false` and `production_launch_authorized: false`, and the final gate (no missing results) fails correctly. The cost is operational — the poll loop doesn't stop and Europe v6e keeps burning. If you want it tightened, gate `loading_is_only_incomplete` at `analyze_...:938-940` on every threshold whose `observed_pair_count == expected_pair_count` having passed.

## Idempotence — internally consistent

Field-by-field, collector emission (`collect_delphi_tpp40_bridge_idempotence.py:219-236`) against analyzer consumption (`analyze_...:663-711`): all five top-level keys and all six per-side keys line up, and `completed_output_unit_counts` is deliberately ignored — that is the round-1 "self-asserted counts" fix landing correctly.

The substantive checks hold up:

- **Three-way inventory equality** (`analyze_...:694-699`) is real. `current` is recomputed live from GCS via `result_inventory` → `tree_payload_identity`, which hashes relative path + size + crc32c of every object under each frozen root and raises on an empty tree (`delphi_tpp40_evaluation_identity.py:106-107`).
- **Zero child jobs** is proven from the Iris `parent_job_id` join (`collect_...:136-138`), and is structurally guaranteed by the launcher returning before `executor_main` when `pending_steps` is empty (`launch_...:758-759`).
- **Command identity** rebuilds the exact frozen argv with only the `--job-name` value substituted, then compares outer envelope, inner argv, and the stored entrypoint argv (`collect_...:102-110, 148-159`).

I also checked whether the byte-identical-inventory assertion is *achievable*, since a strict check that can never pass is as bad as a missing one. It is: a cached-success step returns at `lib/marin/src/marin/execution/step_status.py:190-192` before acquiring the lock or writing status, so it writes nothing; `create_lock` (`lib/rigging/src/rigging/filesystem/distributed_lock.py:340-349`) only constructs, never writes; executor info goes to `{prefix}/experiments`, outside the inventoried trees; and the Uncheatable rerun's ready manifest lands in `.../delphi_tpp40_bridge_uncheatable_v1_20260830/ready_manifests/`, a disjoint prefix from the `.../east5/fit_002_run_00002/step-*` output roots (`launch_...:639-644`).

## Other changes verified

`--max-concurrent` default is now `len(BRIDGE_RUN_ORDERS) * len(CHECKPOINT_STEPS)` = 2 with a `1 <= x <= 2` bound (`launch_...:667, 679-680`); both frozen Uncheatable commands pass `--max-concurrent 2`, and the stale `8` would now hard-fail. The training-output count check is present (`analyze_...:299-303`). No stale four-row assumptions remain — every count derives from `BRIDGE_RUN_ORDERS`/`CHECKPOINT_STEPS`, the frozen manifest carries exactly 1/2/1 cells per side, and `run_orders` is digest-pinned (`launch_...:152-158`). With one pair, `_paired_threshold` makes mean == max == |Δ|, so 0.002 binds at phase 0, endpoint, and Table-9.

## Polish

1. `analyze_...:709-711` — the `unit_counts` check can never fire. `result_inventory` derives units purely from the path-manifest lists, and `_load_frozen_path_manifest:299-311` already pins those to 1/2/1. The handoff's "measured current unit counts" overstates it; the checks with teeth are `tree_payload_identity` raising on an empty tree and the per-output `STATUS_SUCCESS` scan in `collect_...:50-53`. No coverage hole, just dead weight that reads as a real check.
2. `collect_...:140-141, 163-174` — `state`/`exit_code`/`child_job_count` are literals the collector hardcodes after validating the raw Iris row, so the analyzer is re-checking the collector's constants. Acceptable only because the evidence bytes get pinned into `EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256`. The analyzer ignores fields it could independently cross-check: `bundle_id`, `entrypoint_sha256`, `submit_argv_sha256`, `submitted_at_ms`, `before_snapshot_sha256`.
3. `launch_...:546-556` — `_validate_completed_result` lacks the exact set-equality check its analyzer counterpart has; extra `component_bpb` keys would be silently ignored there.
4. `analyze_...:409-411` — `record["paths"]["europe"]` iterates *all* cache records, including non-`uncheatable_eval/` ones that were never validated. A record missing that key raises an uncaught `KeyError` rather than a clean message. Fail-closed, just ugly.
5. No test pins the `--max-concurrent` default to the cell count, so it can drift stale again.

## Two required steps before the gate can authorize

`EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256 = "UNFROZEN"` (`analyze_...:53`) means `production_launch_authorized` can never be True until a human pastes the collector's printed digest (`collect_...:273`) into source. That is the intended two-step freeze, but it is manual and absent from the handoff's sequence. And the final gate run must omit `--allow-incomplete` — with it, the residual case above applies.
