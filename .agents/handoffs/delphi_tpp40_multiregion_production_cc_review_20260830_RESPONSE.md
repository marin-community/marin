I inspected all eight named files plus the launcher's dependencies (executor versioning, checkpointer path resolution, Table‑9 step construction, `marin_temp_bucket` routing) and the frozen bridge artifacts in the same reference tree.

# NO-GO — for "submit immediately after the gate"

The two command strings are, as far as I can verify, **correct**. I found no defect in the commands, the assignment, or the selection logic. The NO-GO is about three things that must land *before* submission, not about the commands themselves.

---

## Blockers

### B1. The East idempotent-replay claim is not backed by a resolved-path check

This is the one with real consequences. The materializer classifies rows by **directory glob** (`materialize_delphi_tpp40_multiregion_assignment.py:48-56`: `{east5_root}/*/hf/step-27335/model.safetensors`, parsed with `/fit_(\d{3})_`). The executor decides to skip by **content-addressed resolved path** (`lib/marin/src/marin/execution/executor.py:1270-1276`, path built at `:1313-1326` from `{name, config, dependencies}`).

Those are different keys. `DelphiSwarmTrainingConfig` carries `run_spec` (incl. `tpu_type`/`tpu_zone`/`tensor_parallel_size`), `analysis_output_path`, `validation_configs`, `wandb_tags`, `wandb_group`, `steps_per_eval`, `permanent_checkpoint_interval` — every one is versioned. If any drifted since the legacy parent ran, the East graph resolves to fresh `fit_NNN_…-<newhash>` directories and **silently retrains all 27 completed rows and restarts rows 27 and 29 from step 0**, discarding their phase‑0 checkpoints. No error, no warning — the exact failure the launch is designed to prevent.

The evidence set does not close this. The dry run doesn't resolve output paths (`_write_local_dry_run` only writes a local manifest + audit). The strongest available evidence is `bridge_acceptance_paths_v2.json:39`, which resolves East row 2 — a *completed* row, under the production experiment name, same v5p‑8/`us-east5-a` + Table‑9 v6e‑8/`us-east5-b` deployment — to `fit_002_run_00002-29ef42`. That transfers to the production graph (row selection and the 153‑spec manifest step do not enter any per-row training version), but it covers one row and has not itself been confirmed against live GCS.

**Fix:** before submitting, run the `Executor(...).compute_version` resolution the bridge already uses (`launch_delphi_tpp40_bridge_uncheatable_eval.py:299-307`) over the full 153‑row East graph and assert: all 27 completed rows resolve to a path whose `.executor_status` is SUCCESS, and rows 27/29 resolve to paths containing `checkpoints/step-21855/metadata.json`. Read-only, cheap, and it converts B1 from an assumption into evidence.

### B2. Freezing the assignment ahead of the gate breaches the preregistered contract

`bridge_acceptance_contract_v3.json:87` — the same file the launcher hash-pins as `be5398dc…` (`launch_delphi_tpp40_bridge_uncheatable_eval.py:63`) — states: *"No production assignment may be **frozen** or submitted until the run_order=2 bridge passes and its unchanged rerun is idempotent."* The assignment is frozen: byte-pinned at `72f88de7…` and already uploaded to both regional buckets.

Drift risk is genuinely closed (the East parent is killed; the Europe bridge writes under a disjoint experiment name so `materialize_…:186-189` cannot be tripped), so this is a preregistration breach rather than a stale-data hazard. **Fix:** after the gate passes, re-run the materializer and confirm the byte hash is still `72f88de7…` — it is deterministic given unchanged live state — or record an explicit, dated amendment to the scope note. Do not leave the contract silently violated.

### B3. No production run record exists

`.agents/logbooks/` has no TPP40 entry (only handoffs under `.agents/handoffs/`). This is a production run by every criterion in `manage-hero-run`: durable output contract, 253 remaining rows × ~3.1e19 FLOPs ≈ 7.8e21 total, two parents, two regions, a 7-day timeout, and two resume-sensitive rows. The workflow requires a dedicated experiment issue and an append-only `.agents/logbooks/<run>.md`, bootstrapped and pushed before launch.

This is load-bearing here, not ceremonial: nothing durable currently records which parent owns which 153/127 rows, the pinned output roots, the resume lineage for rows 27/29, the retention policy, or the monitoring owner — across a launch that spans two regions and two accelerator families.

---

## Answers

**1. Exhaustive, disjoint, deterministic, safe?** Yes, verified arithmetically. completed = 0–26 (27); east5 = odd 27–239 (107) + even 242–278 (19) = 126; europe = even 28–238 (106) + {240, 241} + odd 243–279 (19) = 127. Union = 280, pairwise disjoint. Strata reconcile exactly: `domain_deletion` 19+20 = 39, `qsplit_signal` 27+107+106+1 = 241 — matching `EXPECTED_PANEL_COUNTS` in the base launcher. Per-stratum imbalance ≤ 1 everywhere. `assign_remaining_rows` is deterministic (sorted stratum iteration, sorted order iteration, no RNG). The launcher re-derives every one of these invariants at load time (`launch_delphi_augmented_swarm_tpp40.py:340-367`), including `resumable_east5 ⊆ east5`. Table‑9 obligations are preserved because completed rows stay in the East graph and the executor recurses into a non-SUCCESS eval step's already-SUCCESS training dep without re-running it.

**2. Completed rows in East only?** Correct in design. `:370` selects `completed | east5` for East and only `europe` for Europe; the `.executor_status` SUCCESS check is what makes the training skip free, and a cached-success step writes nothing. Excluding them from Europe is right — including them there would train them a second time in a bucket with no status file. Subject to B1.

**3. Do rows 27/29 resume safely?** Mechanically yes. `resolve_checkpointer_output_path` (`lib/marin/src/marin/training/training.py:186-198`) puts permanent checkpoints at `{output_path}/checkpoints` — durable, not the 14-day TTL temp bucket — so `step-21855` survives regardless of TTL. The run id is `basename(output_path)` (`:236-239`), so W&B resumes into the same run. The executor never deletes output directories. Accelerator identity holds: v5p‑8/`us-east5-a` is unchanged from the legacy deployment, and `tensor_parallel_size` is 1 on both v5p‑8 and v6e‑8 for `hidden_dim=896`, so no mesh change even on the Europe side. The one unproven link is path identity — B1.

**4. Region locality?** Yes, and it is enforced in depth. `MARIN_PREFIX` equality guard at `:504-508`; regional-input coercion and re-validation at `:510-524`; 140 runtime caches and 23 validation caches each asserted region-local (`:387-435`); Table‑9 request set resolves prefix-relative (confirmed: `gs://marin-us-east5/raw/…` vs `gs://marin-eu-west4/raw/…`, identical payload SHA `7401f44e…`). Every GCS literal in each command is under its own bucket. Rolling checkpoints route through `marin_temp_bucket(source_prefix=output_path)` → `gs://marin-eu-west4/tmp/ttl=14d/…` for Europe; this is runtime-derived and *not* covered by `east5_launch_safety.py` (which only reads the command string), but it is correct unless a cluster-wide `MARIN_TEMP_PREFIX` is set — neither command sets one.

**5. Operationally sound?** Yes. `--max-concurrent 56` is at the hard ceiling (`:490-491`). Seven days is comfortable: East ≈ 124 fresh rows × ~12 h ÷ 56 ≈ 1–2 days if v5p‑8 capacity is there; Europe ≈ 127 × ~3 h ÷ 56 ≈ well under a day. Europe's CPU parent in `europe-west4-b` is proven — the bridge parent already ran there. Holding **Table‑9 on v6e‑8 in both regions** while training differs (v5p‑8 East, v6e‑8 Europe) is the right call: it keeps evaluation hardware constant across the whole panel. Two capacity notes: Europe production at 56 concurrent v6e‑8 in `europe-west4-a` will contend with the still-running bridge, and `--priority interactive` for a 7-day production sweep is worth a conscious decision rather than inheritance.

**6. Hash ignored / silent fallback to all rows / colliding output paths?**
- Hash: **enforced twice** (`:325-333`) — recomputed over the payload minus `assignment_sha256` with the materializer's exact canonicalization, checked against both the embedded value and the CLI `--expect-assignment-sha256`. A swapped-but-self-consistent file still fails. Float round-tripping is stable. Sound.
- Fallback: for these commands, no. **But `_require_assignment_contract` (`:107-109`) only guards `europe-west4`.** A production *East* launch that omits `--assignment-file` falls through to `_parse_run_orders("all")` at `:551-557` and trains all 280 rows. Not triggered here, but it is precisely the hazard the question names, and it's a one-line symmetric fix.
- Output paths: disjoint. Different `MARIN_PREFIX` buckets, disjoint row sets, and per-region versions that differ anyway.

**7. Blocker before submission?** Yes — B1, B2, B3 above. Separately, the gate cannot currently authorize at all: `analyze_delphi_tpp40_bridge_acceptance.py:53` still reads `EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256 = "UNFROZEN"`, a deliberate manual two-step freeze. The final gate run must also omit `--allow-incomplete`.

---

## Non-blocking findings

- **Table‑9 namespace contamination in eu-west4.** Table‑9 steps are named `evaluation/olmo_base_eval_table9/t9_{run_name}` (`lib/marin/src/marin/evaluation/olmo_base_eval/run.py:352`) — *not* namespaced by experiment. The Europe bridge writes `t9_fit_002_run_00002-768327` into that same directory, and row 2 is **not** in Europe's production assignment. A path-glob harvest of `gs://marin-eu-west4/evaluation/olmo_base_eval_table9/t9_fit_*` would silently absorb a v6e‑8 bridge model that is not part of the production panel. Harvest by W&B group or `provenance.panel`, not by path.
- **Sequence the East parents.** The East5 bridge reference command (`--run-orders 2`, production experiment name, same Table‑9 resources) produces the *identical* step `t9_fit_002_run_00002-f412ee`. Do not submit East production while that parent is active.
- **Rows with sub-21855 East progress assigned to Europe** restart from scratch; their partial work sits in a 14-day-TTL temp prefix. By design (only the permanent phase‑0 checkpoint is durable), and efficiency-only.
- **`east5_launch_safety.py` coverage is partial by design** — region/zone/bucket only. `--tpu-type`, `--table9-tpu-type`, priority, timeout, and `--max-concurrent` are covered instead by launcher-internal guards. Combined coverage is complete; just be aware it's split across two layers, so passing the safety check alone is not a full command audit.
- Test count reconciles: 5 + 22 = 27, matching the claimed result.

**What converts this to GO:** resolve and confirm the East graph's 153 output paths against live state (B1), re-mint or formally amend the frozen assignment after the gate (B2), and bootstrap the issue + logbook (B3). None of these requires touching the command text.
