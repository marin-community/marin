I verified every round-1 blocker against the current files, plus the failure modes the repair could have introduced.

## Round-1 blocker: fixed and now guarded

**`save_interval` type.** `CheckpointerConfig.save_interval: Optional[timedelta]` (`lib/levanter/src/levanter/checkpoint.py:1171`), consumed as `last_save_time >= self.save_interval` at `checkpoint.py:587`. The launcher now passes `TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=10)` (`launch_...:71,263`), matching Marin's own `_RESUMPTION_INTERVAL` default (`lib/marin/src/marin/experiment/train.py:213`). `resolve_checkpointer_output_path` (`training.py:186-197`) replaces only `base_path`/`temporary_base_path`/`append_run_id_to_base_path`, so the value survives to the pod regardless of ordering. It is asserted twice: `isinstance(..., timedelta)` plus equality in the test (`tests/...:139-140`) and in the runtime audit (`launch_...:474`). The step-2 `TypeError` cannot recur.

I checked the adjacent trap that the same fix could have introduced: `keep` is genuinely `Optional[List[dict]]` (`checkpoint.py:1176`) and `create()` expands it via `CheckpointInterval(**k)` (`:1206`), whose fields are exactly `every`/`until`. The dict form is correct, not a second latent type error.

**Stage-conditional retention.** `_keep_policy` (`:220-224`) gives discovery a terminal-only policy and fixed-policy stages `[{22608, until 22608}, {28259, until None}]`. Against `_get_current_step_save_interval` (`checkpoint.py:693-699`, first policy with `until >= step`): steps ≤22608 select `every=22608` → one permanent save at the boundary; steps >22608 fall to `every=28259` → one permanent save at the endpoint. Sorted-by-`until` precondition holds. Boundary checkpoints drop from 644 rows to the 144 fixed-policy rows.

## Round-1 scientific issues 1–7

All seven are in the frozen payload and hash-consistent (`design_sha256` = `3de5ce73…`, matching `EXPECTED_DESIGN_SHA256`; `build_payload() == manifest` is asserted):

1. Log companion — `metrics.scale_invariant_secondary` and `p1_primary.scale_invariant_secondary`, sign convention consistent with the additive gain (both point toward untied).
2. Seeds — `CONFIRMATION_SEEDS` = 20260831–838, disjoint from 20260821–825; the generator hard-fails on overlap (`design_...:312`) and on historical-seed drift (`:112`); relabeled "independent cross-runtime reproducibility check" with an explicit failure action.
3. `completeness_contract` — exact endpoint definition (finite primary at step 28259), 125/125 per arm, 8/8 per block, retry-with-frozen-identity, no complete-case substitution, defer-don't-analyze.
4. `multiple_testing` — Holm at α=0.05 over the six additive main-arm contrasts; area-matched arm and log gain declared sensitivity-only.
5. Taxonomy collision — zero occurrences of `policy_role` remain in the JSON; `coordinates[]` carries `source_geometry_role` + `selection_class`, `runs[]` carries `selection_class`.
6. Replication scope wording is explicit that support and coordinates both change.
7. `p3.adaptive_inventory` states the ≤64-row second hashed design.

## Independent re-verification

Arm arithmetic re-derived: 22466.998 / 25293.0 / 26706.0 / 28118.5 total, phase-1 tails 1027.16 < 2826.5 < 4239.5 < 5652.0, area multiplier 0.888269 reproducing `decay_0p60`'s integral bitwise. `decay_0p80` (`decay=5652`, `min_lr_ratio=0.0`, lr 0.02/0.008, warmup 282) is *bitwise identical* to `base._optimizer(7_408_189_440)`, which is what the historical panel passed unmodified (`launch_...dense_support_surfaces.py:575`) — and that panel likewise left `train_holdout_*` unset and pinned `experiment_budget`/`target_budget`/`simulated_epoch_subset_seed` to `None`. The bridge is like-for-like on data and optimizer, differing only in runtime, exactly as `reuse_contract` now claims. Support arithmetic checks (1068×128×2048 = 279,969,792; 534 → 139,984,896). Provenance rows carry the historical discovery BPBs whose difference (0.78804 − 0.78197 = +0.0061) sits inside the frozen CI.

The reported evidence is self-consistent: 13 audited configs is exactly `|{(stage, support, arm)}|` = 4+5+4, and `--audit` runs `_validate_runtime_environment()` first (`:583`), so the uv.lock/jax/numpy pins are confirmed current in the real environment.

## Non-blocking findings

- **`metrics.broad_secondary` names a key that will never be emitted.** It is `eval/paloma/c4_en/bpb`, but handles are registered as `paloma/{subset}-llama3` (`experiments/datasets/paloma.py:46`) — the same convention that gives this panel's own primary its `-llama3` suffix. The real key is `eval/paloma/c4_en-llama3/bpb`. It is a descriptive secondary only: absent from `completeness_contract`, from every `analysis_contract` estimand, and from the Holm family, and the data is logged correctly regardless. Nothing is frozen yet (no `manifests/release.json`, CC review file empty), so this is cheap to correct in a regeneration if one happens for another reason.
- The contract does not say what to conclude if the Holm-significant additive contrast and the log-scale sensitivity contrast disagree. Pre-specifying both was round-1's own remedy and `claim_boundary` already limits interpretation; requiring agreement for a confirmatory claim would be stricter but is a judgment call.
- Carried over from round-1 polish, still unaddressed: `_validate_runtime_environment()` hard-fails on any `uv.lock` byte change (resumption risk across a multi-day launch), and `release["submitted"]` is write-only.

No launch-fatal defect and no science-invalidating defect remains.

VERDICT: PASS
