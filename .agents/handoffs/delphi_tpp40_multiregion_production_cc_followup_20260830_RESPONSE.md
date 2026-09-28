## Verdict: **GO**

B1 and B3 are closed. The only remaining launch blockers are the already-frozen numerical/idempotence bridge gate and the post-gate byte-identical assignment re-materialization (B2, intentionally deferred). No new correctness blocker.

---

## B1 — exact East lineage proof: **closed**

The prior gap was that the materializer keyed on directory globs while the executor keys on the content-addressed version. The audit now resolves paths through the executor itself.

**The resolution path is the real one.** `audit_…resolved_paths.py:145-182` calls `bridge_eval._original_training_paths` (`launch_…bridge_uncheatable_eval.py:260-307`), which builds the graph via `base.build_launch_artifacts` and then runs `Executor(...).compute_version(step, is_pseudo_dep=False)` — the same call the executor uses at `executor.py:1289-1326`. This is resolution, not reconstruction.

**Every version-bearing input matches the pinned East command** (logbook `delphi-tpp40-multiregion.md:44`). `compute_version` hashes only `{name, config, dependencies}` (`executor.py:1313-1317`), and `DelphiSwarmTrainingConfig` has exactly eight fields (`launch_…_3e18.py:155-165`):

| Version input | East command | Audit |
|---|---|---|
| `experiment_name` (→ step name) | default `EXPERIMENT_NAME` | `side.training_experiment_name = tpp40.EXPERIMENT_NAME` (`:92`) |
| `run_spec` (incl. `tpu_type/region/zone`, `tensor_parallel_size`) | `v5p-8` / `us-east5` / `us-east5-a` | `BRIDGE_SIDES["east5"]` `:86-95` — identical |
| `analysis_output_path`, `source_panel` | `_regional_input_path(..., us-east5)` | same call, `:153-154` |
| `validation_configs` | `_default_validation_sets` + `step_to_lm_mixture_component(include_raw_paths=False)` (`:582-585`) | identical construction, `bridge_eval:217-228` |
| `wandb_tags` | `deployment=us-east5-v5p-8` (`:617-624`) | `deployment={side.region}-{side.training_tpu_type}` (`bridge_eval:283-290`) |
| `wandb_group` | default `delphi_tpp40_augmented_swarm` | `side.training_wandb_group = tpp40.DEFAULT_TRAINING_WANDB_GROUP` |
| `steps_per_eval` / `permanent_checkpoint_interval` | `STEPS_PER_EVAL` / `21855` (asserted `:572-578`) | same constants |

Three things I specifically checked for divergence and did not find:
- **`--table9-tpu-type v6e-8 --table9-tpu-zone us-east5-b` cannot perturb training paths.** `resources` is not in the version dict (`executor.py:1313-1317`), and `table9_eval_resources` reaches only eval steps (`launch_…_3e18.py:673-689`). The audit's Table-9 resources match anyway (`us-east5-b`, `bridge_eval:269-274`).
- **Row selection cannot perturb training paths.** `manifest_step` is standalone (`launch_…_3e18.py:691-703`); no training step depends on it, so the 153-row `run_specs_json` never enters a per-row version. The audit passes the identical 153 selected orders regardless (`:162-169`).
- **`MARIN_EXECUTOR_STRICT=1` in the command is version-neutral** — it only escalates the out-of-context warning (`execution/context.py:28,64`), and both sides build inside `executor_context()`.

**The audit detects drift, it doesn't assume its absence.** If any shared field drifted, all 153 rows would resolve to new hashes and the 27 completed rows would fail `:100-105` (`lacks its final marker` / `lacks SUCCESS status`). Per-row drift is separately impossible: `build_run_specs` enforces `EXPECTED_SOURCE_PANEL_SHA256`, `EXPECTED_SOURCE_COORDINATE_HASH`, and `EXPECTED_FIXED_IDENTITY_HASH` over all 280 specs (`:190-233`). Root and run-order/path binding are enforced at `:84-88`, order-set exhaustiveness and path uniqueness at `:74-79`.

**The artifact backs the claims.** `east5_resolved_path_audit_v1.json`: `passed: true`, `resolved_count: 153`, `completed_count: 27`, `resumable_count: 2`, `fresh_count: 124`, `resolved_path_payload_sha256: 0b9d6596…` (line 1696) — matching the handoff. Rows 27/29 resolve to `fit_027_…-565b70` and `fit_029_…-915223` with `phase0_checkpoint_present: true`, no final marker, non-SUCCESS status (lines 308-329); `_checkpoint_metadata` independently asserts `step == 21855`, `is_temporary == False`, and tensor payload presence (`bridge_eval:310-326`). I counted exactly 83 `true` values across the three artifact keys = 27×3 (completed) + 2 (resumable phase-0), which confirms all 124 fresh rows are clean on all three. Row 2 resolves to `fit_002_run_00002-29ef42` — the same path the frozen bridge artifact predicted, now confirmed against live GCS.

The audit also read the assignment from the **GCS** path the East command uses and recorded `assignment_file_sha256: 72f88de7…` with `assignment_sha256: 873959…` — so the uploaded copy is byte-identical to the pin, and B2's re-materialization has a concrete comparison target.

## B3 — production run record: **closed**

`.agents/logbooks/delphi-tpp40-multiregion.md` covers every field `manage-hero-run/SKILL.md:83-85` requires:

command `:44,:50` (plus SHA-pinned command files `:39-40`) · source SHA `88ad3e00…` `:28,:37` · dirty-tree status + diff identity `fff7cad1…` `:28` · DRI Calvin Xu `:53` · hardware/topology `:53` · tracker groups `:53` · output and checkpoint roots `:54` · retention and projected bytes `:30` · `initialize_from` `:55` · final step 27335 `:53` · monitoring owner `:53`.

The storage arithmetic reconciles: 112 × 4.3 GB = 481.6 ≈ 482 GB rolling; 2 × 4.3 + 1.45 = 10.05 ≈ 10.1 GB/row durable; 251 + 2 × (27335−21855)/27335 = 251.4009 full-run equivalents. Retention (one rolling temp checkpoint, region-local TTL prefix, permanent 21855/27335) is within `SKILL.md:141-146`. Issue and logbook links exist in frontmatter and Scope, satisfying `SKILL.md:79-81`; the issue body (`delphi_tpp40_multiregion_issue_body_20260830.md`) carries TL;DR, description, status, links, and decision log.

---

## Launch-time obligations (not blockers, but must happen at submit)

1. **Re-run the resolved-path audit on the exact submitted tree** and require `resolved_path_payload_sha256 == 0b9d6596…`. Nothing in the launcher pins `validation_cache_paths_sha256` — it is computed but never compared to a constant (`launch_…tpp40.py:586-590`) — so a change to `_default_validation_sets` or any of the 23 validation caches between now and submission would silently re-version all 153 rows. Cheap, read-only, and pairs naturally with the B2 re-materialization.
2. **Refresh the dirty-tree diff identity.** `fff7cad1…` is stamped 18:29 PDT 2026-08-30; the tree must still change (B2 re-materialization, freezing `EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256`). `SKILL.md:50` wants the identity of the *launched* instance. The bundle identity is legitimately post-submission.

## Nonblocking polish

- **Fresh rows assert nothing.** `audit_…:99-112` records `executor_status_succeeded` / `phase0_checkpoint_present` / `final_marker_present` for the 124 fresh rows but has no `else` branch, so `passed: true` does not depend on them. Today's data is clean, but a re-run would still pass if a fresh row acquired unexpected artifacts. Three asserts make the artifact self-certifying.
- **The audit is evidence, not a gate.** `east5_resolved_path_audit_v1.json` is referenced only from the logbook and handoff — never from the launcher. An East-only `--expect-resolved-path-payload-sha256`, mirroring the double-enforced assignment hash (`:325-333`), would close the drift window by construction rather than by procedure. Highest-value item here.
- **East deployment is hardcoded, not derived from the command.** The audit reads `BRIDGE_SIDES["east5"]`; I verified each field against the command by hand, but the link is manual. Editing `--tpu-zone` or `--training-wandb-group` would leave the audit certifying stale paths.
- **`_require_assignment_contract` still guards only `europe-west4`** (`launch_…tpp40.py:107-109`). An East production launch omitting `--assignment-file` still falls through to `_parse_run_orders("all")` at `:556` and trains all 280 rows. Carried over unfixed from the prior review; one-line symmetric fix.
- **`_resolve_east_training_paths` overwrites `MARIN_PREFIX` unconditionally** (`audit_…:152`), unlike `bridge_eval._set_region_prefix:164-170` and `launch_…tpp40.py:504-508`, which raise on a conflicting pre-set value. Benign, but it breaks the guard pattern used everywhere else in this family.
- **W&B resume policy not stated** in the logbook (`SKILL.md:54-55` asks for id/name and resume policy). Groups are recorded; per-row ids are derived as `basename(output_path)` (`training/training.py:236-239`). Worth one explicit line.

## Verification limits in this session

No shell and WebFetch denied, so three things rest on file evidence rather than recomputation: the logbook commit `3a5552e10d`; issue 8797 itself (assessed via the handoff body file, as the review instructions permit); and the audit file's byte SHA `3f9468e8…`. On the last — the file's embedded `payload_sha256` is `cae95b10…`, which is not a conflict: `audit_…:210-214` computes that digest over the payload *before* adding the key, so the whole-file hash is necessarily a different value. Test counts reconcile (6 in `test_delphi_tpp40_multiregion_assignment.py` + 2 in `test_audit_delphi_tpp40_multiregion_resolved_paths.py` = the claimed 8), but I could not execute them.
