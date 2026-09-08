I read the handoff, the frozen contract/manifest/report, all four frozen command files, both cache audits, and the three modules that implement the gate (`launch_delphi_tpp40_bridge_uncheatable_eval.py`, `analyze_delphi_tpp40_bridge_acceptance.py`, `collect_delphi_tpp40_bridge_idempotence.py`) plus the launchers they call.

## Verdict: NO-GO as written

The scientific core (same-region paired evaluation of a mirrored canonical East checkpoint) is sound and I'd approve it. But proposal item 6 is not satisfiable against the current gate code, and the mirror — the one new trust assumption — is not enforced by the gate at all. Both are fixable without changing any threshold. Conditional GO once B1–B4 are closed.

## Launch blockers

**B1. East Table-9 has no home under this plan, and item 6's idempotence claim cannot pass.**
Item 6 says training idempotence "remains checked against the existing frozen East reference command." It can't:
- `collect_delphi_tpp40_bridge_idempotence.py:42-54` requires *all four* units per side — including `table9` — to be `STATUS_SUCCESS` before the before-snapshot is even takeable. East Table-9 is still unstarted (`bridge_acceptance_report_v1.json:12,36`), blocked on the same East5 v6e-8 capacity.
- Table-9's checkpoint is bound by executor reference to the training step: `launch_delphi_augmented_swarm_3e18.py:676` (`checkpoint=training_step / f"hf/step-{...}"`). There is no injection point for a mirrored path.
- You cannot simply repoint it: `launch_delphi_augmented_swarm_tpp40.py:74-77`, called at `:535`, rejects `--table9-tpu-zone europe-west4-a` whenever `--tpu-region us-east5`.
- So a rerun of the unchanged East reference command schedules the East Table-9 child job, and `validate_rerun_job` raises on any child (`collect_...:137-138`).

Required: decide explicitly where East Table-9 runs. If Europe, `build_launch_artifacts` and the tpp40 placement check need an explicit reference-evaluator region plus a mirrored checkpoint source; then reissue and re-freeze `east5_bridge_reference_run2_command_v3.txt`. Either way the rerun-role list must gain the new evaluator — both `collect_...:259` and `analyze_...:700` only iterate `("training", "uncheatable")`, so a separately-launched East Table-9 evaluator's idempotence would go unchecked, which is strictly weaker than v3.

**B2. The mirror is not covered by the gate, and it weakens the v3 inventory guarantee.**
Under v3 the East checkpoint lived inside `training_output_paths`, which `result_inventory` hashes with `tree_payload_identity` (`analyze_...:603-645`). A Europe mirror tree is in none of those lists. At analysis time the only mirror re-check is `metadata.json`'s sha for Uncheatable cells (`analyze_...:747-754`); Table-9 gets a bare string compare of `checkpoint_path` (`analyze_...:519-522`) and no content check at all. A mirror mutated between eval and analysis is undetectable on the endpoint macro.

Required: (a) persist the mirror audit as a source-frozen artifact with its own `EXPECTED_*_SHA256` and a loader that **re-verifies live at analysis time** against the canonical East source — the pattern `_load_evaluation_data_identity` already uses; (b) add the mirrored trees to `result_inventory` so before/after idempotence covers them. Use `tree_payload_identity` (`delphi_tpp40_evaluation_identity.py:91-112`) rather than a hand-rolled check — it already computes exactly relative-path + size + CRC32C into one canonical sha256, so mirror equality reduces to comparing two 64-hex digests. Note that `rsync --checksums-only` without `--delete-unmatched-destination-objects` does not preclude *extra* destination objects; make the audit require set equality, which `tree_payload_identity` gives you for free.

**B3. Contract v3 has no evaluator-placement fields, so the fallback cannot be frozen in the contract.**
`bridge_acceptance_contract_v3.json:4-22` pins training accelerators and zones but nothing about evaluators — placement lives only in source constants (`launch_delphi_tpp40_bridge_uncheatable_eval.py:65`, `BridgeSide.evaluator_zone`) and the command files. Under this fallback evaluator placement is the *only* thing that changes, so the contract would not describe the experiment it gates.

Required, all committed before the Europe evaluator is submitted: a contract v4 carrying `evaluator_tpu_type`, per-side evaluator region/zone, and the mirror manifest digest; a re-materialized path manifest v3 with `EXPECTED_PATH_MANIFEST_SHA256` re-frozen; the new mirror-audit digest; and all four command SHAs. Thresholds stay untouched — they're already absolute and independent of the noise estimate (contract `:72`, `:77-80`), so there's nothing outcome-contingent in the numerics themselves.

**B4. The gate's authorization scope must be narrowed before results exist.**
Item 5's claim that this "removes evaluator-region/hardware variation" is overstated: `EVALUATOR_TPU_TYPE = "v6e-8"` was already used on *both* sides (`launch_delphi_tpp40_bridge_uncheatable_eval.py:65`, and its docstring lines 7-9 already assert the v3 design isolates training deployment). The fallback removes region/instance only. The real justification here is capacity, not better science — and it should be recorded that way.

What the fallback actively *loses*: a common-mode Europe evaluator offset now cancels in the paired delta. Production sequencing step 4 submits disjoint East5 *and* Europe parents, whose numbers get pooled against an East5-derived historical SD (contract `historical_noise`, 0.00342). This gate can no longer bound that comparability.

Required before results: a contract scope note stating the gate bounds training-deployment drift only and does not authorize cross-region comparability of evaluated numbers; and a pre-registered threshold and consequence for the still-alive East5 eval parents kept as supplementary evidence in item 7, so their eventual outcome can't be reinterpreted after the fact.

## Implementation note (makes B1/B3 safe)

Keep `side.region = "us-east5"` for the East logical side and add a separate `evaluator_region`. The run-spec identity checks in `_run_specs` (`source_panel_sha256`, `source_coordinate_hash`, `fixed_identity_hash`, `scientific_identity_hash`) key off `side.region` via `_regional_inputs`, so leaving it alone preserves the frozen scientific identity unchanged. The cost is that "one prefix per side" is baked into five call sites that must be split: `_set_region_prefix:164-170`, `main:690`, `_require_uncheatable_caches:236`, `_eval_steps:470-487`, and `analyze_..._side_path_manifest:179-181`. Your proposal to keep the logical side names `east5`/`europe` is right — `analyze_...:290-292` hard-requires that tuple.

Also decide deliberately whether the Europe-placed East Table-9 reads the East or Europe request set; payload SHAs are identical (`evaluation_cache_audit.json:432-435`) so either passes validation, but it's the one place the East side would read a non-region-local input, and it should be recorded in the manifest rather than falling out of `MARIN_PREFIX`.

## Nonblocking polish

1. Fix item 5's wording — evaluator hardware type was already matched. The launch record should say the fallback trades a small attribution gain for availability.
2. `EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256 = "UNFROZEN"` (`analyze_...:53`) is the one legitimately post-hoc freeze. Commit it before running with `--idempotence-evidence`, and make the launch decision read `production_launch_authorized` from the JSON — `--allow-incomplete` (`analyze_...:992-998`) lets the script exit 0 with that field false.
3. Reconcile the stated 10.05 GB against `tree_payload_identity(...)["bytes"]` for the three trees before copying. It's the cheapest check that the right tree set was enumerated.
4. Once B2 lands, the copy mechanism stops mattering — drop `gcloud storage rsync --checksums-only` from the contract text so the gate depends on the audit, not a CLI flag.
5. Add an explicit `reference_checkpoint_source` field per manifest cell (canonical vs mirror) instead of overloading `checkpoint_path`, so the manifest reads unambiguously months from now.
