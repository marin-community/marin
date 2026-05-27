# Silent type-degradation in the Delphi cooldown checkpoint pipeline — risk register

**Bug class:** A model-config sub-dict decoded without an explicit `type:` discriminator
falls back to the default registry type (Llama). The Llama param tree is a proper
subset of Qwen3 (no `q_norm`/`k_norm` slots). When the loader pulls a Qwen3-native
checkpoint into the Llama tree, the on-disk q/k RMSNorm arrays are silently dropped.
Training proceeds for thousands of steps under the wrong forward pass. The written
checkpoint looks complete by file-presence checks (`manifest.ocdbt`, `metadata.json`,
`d/`) but its OCDBT kvstore is missing 12+ keys per layer. The failure only surfaces
in a downstream consumer that loads the result with the correct `type: qwen3`.

**Originating incident:** 2026-05-25 — three 70% prefix materializations (3e18, 3e19,
3e20) and six dependent cooldown30 cells (3e18/3e19 × three mixes). Fixed in commit
`5afac0bdf` at the materialization-helper level; this document covers the broader
audit and remaining defenses.

---

## Defenses landed in this branch

| Defense | Location | What it catches |
|---|---|---|
| **D1.** Source-config Qwen3 fail-closed in materialize helper | `scripts/materialize_delphi_prefix_checkpoint.py:386` (commit `5afac0bdf`) | Untyped `.executor_info` model payload silently decoded as Llama. Helper refuses unless the decoded source is `Qwen3Config`. |
| **D2.** Launcher renders explicit `type: qwen3` and asserts heuristic returns `Qwen3Config` | `experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py:248-252` (commit `5afac0bdf`) | Launcher path that builds the cooldown spec dictating a wrong type. |
| **D3.** `MidtrainSpec` validation requires `model_config["type"]` to be set | `lib/marin/src/marin/midtraining/spec.py:_assert_model_config_matches_base` | Any spec constructed without an explicit type discriminator. Fails at `validate_midtrain_spec` time — before submission. |
| **D4.** Preflight OCDBT-key schema check on the staged cooldown source | `lib/marin/src/marin/midtraining/preflight.py:_check_cooldown_stage` + `lib/marin/src/marin/midtraining/checkpoint_schema.py` | A staged cooldown source checkpoint whose OCDBT kvstore is missing class-specific arrays (currently: Qwen3 q_norm/k_norm). Blocks the launch with an actionable error naming the missing markers. |
| **D5.** Post-train OCDBT-key schema check on the saved final cooldown checkpoint | `experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py:_verify_final_checkpoint` | The (topologically possible but so-far unobserved) case where training silently degraded the param tree mid-run. Treats Iris success + bad checkpoint as a failed launch. |

All five defenses are exercised by `tests/midtraining/test_checkpoint_schema.py`
(15 tests) and `tests/midtraining/test_spec_validators.py` (2 new tests). Total
midtraining suite: 143 tests, all passing.

---

## Remaining risk areas (not yet defended in code; flag for future work)

These were surfaced by the audit but deliberately left for a follow-up because they
require touching vendored Levanter (`lib/levanter/`), have broader blast radius, or
are not exercised by the current cooldown-only flow. Listed in priority order.

### R1. `tree_deserialize_leaves_tensorstore` silently drops extra on-disk keys
- **Location:** `lib/levanter/src/levanter/tensorstore_serialization.py:117-300`
- **What:** The deserializer walks the in-memory exemplar pytree leaves and pulls
  matching arrays from OCDBT. On-disk keys with no exemplar slot are never enumerated.
- **Why it matters:** The root cause of the originating incident was this exact
  asymmetry. The Marin-layer defenses D3-D5 above catch it at the load and save
  boundaries, but a strict `allow_extra=False` mode in Levanter itself would catch it
  anywhere the loader is used, including non-cooldown call sites (eval, ad-hoc analysis,
  ablations).
- **Action:** Add a `strict: bool = False` parameter to
  `tree_deserialize_leaves_tensorstore`. When `True`, enumerate the full OCDBT kvstore
  and raise if any key has no matching exemplar leaf. Wire `strict=True` into the
  cooldown init path through `lib/marin/src/marin/midtraining/levanter_config.py`.
- **Owner:** unassigned. Estimated effort: half-day.

### R2. `decode_train_config_from_executor_info` only normalizes `model.type`; other ChoiceRegistry sub-dicts use heuristics
- **Location:** `scripts/materialize_delphi_prefix_checkpoint.py:474-543`
- **What:**
  - `normalize_optimizer_dict` infers `type = "adamH" if "adam_lr" in optimizer else "adam"` (line 514). No discriminator in stale configs.
  - `normalize_dataset_format` infers `type = "chat"|"prebuilt"|"text"` from field presence.
  - `ensure_choice_type` silently injects defaults without an audit trail.
- **Why it matters:** Different optimizer classes have different param-tree slots
  (e.g. AdamH has an Adam-scalar group with its own LR and moments). Same silent-drop
  risk as the original `model` bug, just less visible because optimizer state errors
  surface as subtle training-dynamics drift rather than a load-time crash.
- **Action:** Add per-field assertions analogous to D1: after decoding, verify each
  ChoiceRegistry-typed sub-config (`optimizer`, `data.format`, `tracker`) decoded to
  the expected concrete class for Delphi runs (e.g. optimizer must be `AdamHConfig`).
  Fail closed.
- **Owner:** unassigned. Estimated effort: 1-2 hours.

### R3. `_check_cpt_init` does not validate the CPT init checkpoint's model class
- **Location:** `lib/marin/src/marin/midtraining/preflight.py:_check_cpt_init`
- **What:** CPT verifies the init checkpoint *exists* but does not enumerate OCDBT keys
  or assert the checkpoint matches the declared model class.
- **Why it matters:** CPT is somewhat insulated (it uses `initialize_from_hf` for HF
  inits and resets optimizer state), but a CPT run from a `CheckpointSourceKind.NATIVE_LEVANTER`
  source still goes through the same Levanter load path. A Llama-degraded checkpoint
  used as a CPT init would silently produce a degraded CPT output.
- **Action:** Apply the same `assert_checkpoint_complete_for_model_type` call in
  `_check_cpt_init` when `mode.init.source_kind == NATIVE_LEVANTER`. The plumbing is
  already in place (preflight already takes `list_ocdbt_keys`).
- **Owner:** unassigned. Estimated effort: 30 minutes + tests.

### R4. Save-time invariant lives in the launcher, not in Levanter's checkpointer
- **Location:** `lib/levanter/src/levanter/checkpoint.py` (no model-class invariant
  at write time).
- **What:** D5 above catches a degraded final checkpoint at the launcher level, but
  any other Levanter caller writing checkpoints (continual-pretraining runs, eval-time
  re-saves) has no built-in guard.
- **Why it matters:** Defense-in-depth. If a future code path silently degrades a
  param tree and writes a checkpoint, the next consumer might not have a launcher
  wrapper to catch it.
- **Action:** Add an optional `expected_param_paths` field to `CheckpointerConfig`.
  When set, the checkpointer enforces that the in-memory pytree contains each path
  before serializing. Wire it into the cooldown launcher's rendered Levanter config.
- **Owner:** unassigned. Estimated effort: 1 day (touches vendored Levanter).

### R5. `experiments/delphi_models.py:DelphiModel` does not declare its expected model class
- **Location:** `experiments/delphi_models.py` and `lib/marin/src/marin/midtraining/spec.py:BaseModelRef`
- **What:** `DelphiModel` knows hidden_dim, num_layers, etc. but has no `model_type`
  field. The spec validator (`_assert_model_config_matches_base`) can verify dims but
  cannot cross-check that `model_config["type"]` matches what the base actually is.
- **Why it matters:** Today the only way a Delphi launcher gets the right type is
  by the launcher's own assertion that the heuristic produces `Qwen3Config`. A new
  Delphi launcher that bypasses that assertion would not be caught.
- **Action:** Add `model_type: str = "qwen3"` to `DelphiModel`. Extend `BaseModelRef`
  protocol. In `_assert_model_config_matches_base`, assert
  `model_config["type"] == base.model_type` for any base that declares one.
- **Owner:** unassigned. Estimated effort: 30 minutes + a couple of tests.

### R6. No negative test in vendored Levanter for the symmetric mismatch
- **Location:** `lib/levanter/tests/test_checkpoint.py`
- **What:** Levanter tests pass for the case where on-disk and exemplar match, and
  for the case where on-disk is missing keys (raises). They do not pin the behavior
  of "on-disk has extra keys not in exemplar". That asymmetry is the bug.
- **Why it matters:** If a future Levanter refactor changes the silent-drop behavior
  (e.g. starts logging a warning, or starts raising), we'd want a test that pins it
  intentionally and catches breaking changes. Either way, the silent-drop semantics
  should be documented and tested.
- **Action:** Add a test in Levanter that exercises an exemplar tree with fewer
  leaves than the on-disk OCDBT and pins the current behavior (silently ignores).
  Then add a `strict=True` test (paired with R1).
- **Owner:** unassigned. Estimated effort: 1-2 hours.

### R7. `list_delphi_checkpoints.py` does not surface q_norm/k_norm completeness
- **Location:** `scripts/list_delphi_checkpoints.py`
- **What:** The inventory helper reports which checkpoint steps exist and whether the
  required artifact files are present, but does not enumerate OCDBT keys or check for
  class-specific completeness.
- **Why it matters:** The 70% materialization candidate yaml was built from this
  helper's output. A "looks complete" listing in 2026-05-25 led to launching cooldown30
  against a degraded checkpoint. Surfacing the schema status in the inventory would
  have caught it before the candidate yaml was approved.
- **Action:** Add a `--check-model-type` flag that opens each listed checkpoint via
  `assert_checkpoint_complete_for_model_type` and reports a column. Default off (cheap
  listing remains cheap) but documented in the helper's usage.
- **Owner:** unassigned. Estimated effort: 1 hour.

---

## Operator playbook

If you see a `q_norm` or `k_norm` restore error during a cooldown launch:

1. Don't manually graft missing arrays into the bad checkpoint. The intervening weights
   were trained under the wrong forward pass; the graft would be scientifically wrong.
   This was considered and rejected at incident time — see
   `.agents/logbooks/midtraining_delphi.md` 2026-05-27T04:03Z.
2. Identify the bad checkpoint path in the launcher error.
3. Re-materialize the prefix checkpoint with the fixed helper (output to a new
   `-qwen3` discriminator path so the old root remains as audit trail).
4. Update the candidate row in
   `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml`
   to point at the new path, mark `materialized_checkpoint: true`, and leave the old
   path as `invalidated_materialized_checkpoint_path` with a reason.
5. Re-run the launcher. The preflight schema check (D4) should now pass.

If you see a `missing the 'type' discriminator` error from spec validation:

1. The launcher you're using is constructing `model_config` without an explicit type
   field. Fix at the call site by adding `"type": "qwen3"` (or whatever the base
   actually is) before passing to `MidtrainSpec`.
2. If you're decoding from a stale `.executor_info`, route through the materialize
   helper's normalization path (`normalize_executor_train_config`) rather than calling
   `draccus.decode` directly.

---

## Cross-references

- Originating fix commit: `5afac0bdf [midtrain] Fix Delphi prefix checkpoint materialization`
- Original incident postmortem: `.agents/logbooks/midtraining_delphi.md`
  2026-05-27T04:03Z — Invalidated materialized Delphi prefix checkpoints
- Related (distinct) trap, do not conflate: `.agents/projects/delphi_midtraining.md`
  Trap #1 (the 1e20 base substitution — different model entirely) and Trap #2 (the
  `-v5-` naming-only artifact on canonical 1e21/1e22/1e23 runs).
- Defense module: `lib/marin/src/marin/midtraining/checkpoint_schema.py`
- Tests: `tests/midtraining/test_checkpoint_schema.py`,
  `tests/midtraining/test_spec_validators.py` (`test_model_config_requires_type_discriminator`,
  `test_model_config_type_must_be_string`)
