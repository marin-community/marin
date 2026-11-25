# facet_v2 re-pin execution plan (quiesce-safe release → served declaration)

Worker: OpenCode ses_df1e36369ffeJibQG2Kd6P560U. Coordinator-approved judge spend (2026-11-24
~16:35 EST). Landing via repo-absolute `infra/ci/atlas-push` (NOT the PATH wrapper — it
deadlocked a nested lock on 2026-11-24). Report every job ID + SHA to the coordinator and the
lane operator (ses_dfec1e64affesTRF6vLcYkvXgy).

## The target

A fresh `OracleAdmittedReleaseReceipt` for `FWeindel:validated-tasks/facet_v2` bound to
`open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1@e9099dac47e02dbe53ab559583f67f5353857c3c`
(tasks.parquet sha256 `f026ee23f5eee69a5d191f974a3bd72ed681d66c1f5d75afa9cc85a343d7c34d`,
775 tasks), landed through the receipt-shard references so `all_sources()` binds
`current_release` to the repair release; `declaration_audit` passes; the next catalog load
projects it; the difficulty controller's TASK_SPEC_FACTORY_PENDING remediation builds the fresh
panel; then re-install the parked repair-verification
(`s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-ledger/v1/repair-verifications-parked/backend-repair-pending-FWeindel-validated-tasks-facet-v2-20261124.json`,
`reinstall_uri` inside).

## The validator's hard requirements (oracle_admitted_release.py `_qualified_population`)

Every measured section must name the NEW pin (`release = filtration.filtered_release`):
- `actor`: ActorCensusReceipt with input.dataset_id/revision = the repair release,
  converter_version/revision, measured_tasks 775, task_specs_sha256 = the NEW specs digest,
  measurement = the fresh DaytonaMeasurement, integrity_contracts covering required_shared=775
  with no missing contracts.
- `grading`: RetainedGrading with input == actor.input and required_counts shared=775.
- `lexical`: LexicalContaminationEvidence with input_dataset_id/revision = the new pin,
  measured_tasks 775, receipt.retained_tasks == 775.
- `output_dedup`: WithinSourceDedupEvidence with source_dataset_id/revision = the new pin,
  raw=unique=775, duplicates=0.
- `sample`: SampleQCReceipt whose review.panel binds the new pin + specs digest, seed 0,
  size min(100, 775), attempted_count == sample_size, ≥95% conclusive.
- `filtration`: **REVISED 2026-11-25 — a FRESH oracle-filter run over the repaired revision is
  REQUIRED.** The original plan said to reuse the 20261120 filter receipt (sha `2b603b1e…`);
  the FACET lane verified in code that `OracleFilterReceipt.admits`
  (`taskcompendium/pipeline/oracle_filter.py`) refuses every repaired TaskSpec:
  `entry.task_sha256 != task_spec_sha256(task)` → "Task changed after oracle filtering" — the
  receipt is hash-bound to the old revision by construction, so no receipt bound to old hashes
  can ever admit the new panel. The `RecollectionWaiver` route does NOT fit: it is ruling-bound
  to the campaign-era rebuilt-packages case and binds rebuilt packages to sealed trials by
  exact task/runtime content hashes, which changed bytes cannot satisfy. Re-run the maintained
  oracle filtration over `open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1` (775 repaired
  TaskSpecs) to produce a new receipt bound to the new hashes — faithful, not a bypass: the
  quiesce fix changed the grader's collect path, not task semantics (the A/B receipt proved
  collect[0] completes with the machine alive), so the selection semantics are unchanged and
  the fresh run reproduces the 775/780 selection under the new hashes while re-validating the
  repair through the maintained gate. The oracle trials run the shipped solutions through the
  repaired graders (facet: native_file_reward_grader → test.sh → reward file; no model judge
  spend). `oracle_filter_version` stays 4 (no static retention); the negative-control policy
  binds as in the original run. Cost: one oracle grading pass over the 775 repaired tasks.
- `repair_evidence`: ≥1 ContentEvidenceReference — use the repair artifacts
  (`.../facet-quiesce-repair/20261124/repair-summary.json`, `repair-ledger.json`,
  `ab-verification-receipt.json`, `census-measurement.json`) + the landing commit.
- `original_pipeline` / `original_info`: unchanged from the current receipt (retained pin
  `...ece97d02-retained-20261117-b-facet-v5@75d7e25d…`).
- `consumer`: the facet qualification receipt (`.../admission-v1/facet_v2/qualification.json`,
  sha 7fdb971a…) — verify it does not pin the release revision; if it does, re-issue.

## DONE

- **Actor + grading census measurement**: job `/benfeuer/benfeuer-facet-repin-census-v8-20261124`
  (repo checkout of the landed atlas-env-audit tip; script `facet_repin_census.py` in
  `/Users/benfeuer/Documents/marin-worktrees/facet-repin/`). 775/775 specs, 775 machine slots,
  405 distinct shapes (matches the 20261120 census exactly), 0 failed tasks, grading
  environments assessed, no Daytona API calls. Measurement document:
  `s3://marin-us-east-02a/marin/rl-data/atlas/facet-quiesce-repair/20261124/census-measurement.json`
  sha256 `3e53b6772073661bbd5d6e2828ddcf1226e6e11ba741cd154d17d0adc91a3ab2`.
- The quiesce-safe release itself (published, A/B verified): see STATUS.md.

## Stage 1 — the census artifact structure at the new pin

The 20261120 seal produced, per source, under
`s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-facet/2026.11.17.1/seal-qualified-census-20261120-v1/facet_v2/`:
`census/{actor,derived,grading,normalized,population,retained-grading}` + `uniqueness.json` +
`measurements.json`, each an artifact record (`.artifact.json`) written by
`marin.execution.artifact.write_record`. Reproduce at the new pin under
`s3://marin-us-east-02a/marin/rl-data/atlas/facet-quiesce-repair/20261124/census-artifacts/facet_v2/`:

1. `population` (SourceTextPopulation): read the release tasks.parquet; text rendered with
   `taskcompendium.models.format_conversation` (renderer `taskcompendium-format-conversation-v1`);
   `task_count` 775; `text_sha256` — the text is UNCHANGED from the parent (parent value
   `416073250f607d0661a59c3fff401f213cbb15b0ec1a2129fec65a05f7a30072` — verify by rendering);
   `task_specs_sha256` = canonical_sha256 of the specs map (NEW — compute); main_output_dir /
   specs_output_dir = the staged text/spec parquets.
2. `actor` (ActorCensus): from `census-measurement.json` (converted 775, slots 775, snapshot
   count 405, safety "unsafe" reason "too_many_snapshots" — 405 > DAYTONA_SAFE_SNAPSHOT_LIMIT,
   execution failures 0, roles/actor_environment_counts/empty_declared_environment_tasks from the
   measurement's `roles`, `integrity_contracts` from the SHARED-contract summary — compute from
   the release specs' RootDeliveryContract verifier files: all 775 shared, none missing).
   `input_artifact` = the population artifact above.
3. `grading` + `derived` + `uniqueness.json`: from the same measurement + the dedup run (stage 3).
4. Emit all as artifact records with `write_record` (name/version/fingerprint fields can be
   empty as the 20261120 originals did).

## Stage 2 — the SAMPLE panel + inventory

`sample_qc_artifacts.py`: SamplePanel{source_id, dataset_id/revision = the repair release,
converter_version = the facet recipe version, input_artifact = the population artifact,
task_specs_sha256 = NEW, retained_tasks 775, sample_seed 0, sample_size 100, panel_path,
panel_sha256} + SampleInventory whose `qc_panel()` equals the panel. Draw the panel with seed 0
from the 775-task population (the same sampler the 20261120 flow used — seed 0, first-100
ordering per the standard seeded design), stage it, write both artifacts.

## Stage 3 — the SAMPLE QC review (judge spend APPROVED)

`sample_qc.py review_panel(panel, reviewer, output_root)`:
- reviewer: `full_qc.py` `ReadyChatReviewer`/`local_reviewer` with `SAMPLE_REVIEWER_VERSION` +
  `qualify_reviewer`; the judge endpoint per `judges.py PinnedLocalJudgeEndpoint` (the QC judge
  service the 20261120 review attempt used — see its `reviewer` identity in
  `.../sample-v4-v1/reviews/aeba14787ab62f0ec04d/attempts/0f2c63e05bb34273801f24dd0f166e42`).
- Output: observations + SampleQCReview (attempted_count 100, ≥95% conclusive) under the
  evidence root; the review needs `REVIEW_RUBRIC` (unchanged) and the task contracts
  (`full_qc_inputs.task_contract`).
- Run as an Iris job from a clean checkout (the census job pattern works: workspace bundle
  15.8 MB, `uv run --prerelease=allow --with daytona==0.200.2`).

## Stage 4 — lexical + dedup at the new pin

- Lexical: `lexical_contamination.py --source-ids [facet_v2] --converter-revision …
  --artifact-version … --artifact-root … --output … --max-workers N --run` — it measures the
  population text against the Datakit eval blooms (`s3://marin-us-east-02a/marin/datakit/bloom/
  _combined_fixed_3a94b52b` + hash index, per the current receipt). Values will match the parent
  (0 contaminated — the text is unchanged) but the receipt must name the new pin.
- Dedup: `within_source_dedup` over the 775 rendered texts via
  `taskcompendium.pipeline.deduplication.unique_documents` (raw=unique=775, duplicates 0) +
  drop manifest staged under the census artifacts.

## Stage 5 — the receipt + registry publish

Assemble the OracleAdmittedReleaseReceipt (sections above). Publish as a NEW registry document
family in `open-athena/marin-rl-data-atlas-registry` via the receipt-shard publish flow
(`receipt_shards.write_receipt_shards` + the registry commit; the current facet receipt lives at
registry revision `9ddb7930ca19fd678496e1d81e6929107cc41535`, path
`.../oracle_admitted_release_receipts_.../part-*.json` — follow the sibling-shard rules in
`registry.py _require_complete_shard_family`). Then land the repo shard-reference update
(`datasets/oracle_admitted_release_receipts/part-*.json` → the new registry revision/path) via
repo-absolute atlas-push; `declaration_audit` runs as the landing gate.

## Stage 6 — load + panel + re-install

- The next catalog load projects the declaration (coordinate with the release-pipeline lane on
  the r40 cut in flight — facet_v2 is already in the r40 plan; the load must be cut AFTER this
  landing, or an r41 carries it).
- The controller's TASK_SPEC_FACTORY_PENDING remediation (or a driven FreezeSamplePanel) builds
  the fresh panel; verify the binding (declaration population revision == the receipt's
  population revision).
- Re-install the parked repair-verification verbatim (its `reinstall_uri`), confirm the
  controller dispatches facet_v2 against the revision, then verify the first scored rows land.

## Reference facts

- Repair release: `open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1@e9099dac…`, 775 rows,
  tasks.parquet sha256 `f026ee23f5eee69a5d191f974a3bd72ed681d66c1f5d75afa9cc85a343d7c34d`.
- Parent (old pin): `open-athena/fweindel-facet-v2-oracle-qualified-20261120-b-facet-v1@a3c1eeae…`.
- Retained original: `open-athena/fweindel-validated-tasks-facet-v2-ece97d02-retained-20261117-b-facet-v5@75d7e25d…` (780 tasks).
- Old actor census values (must match): 775 slots, 405 snapshots, 0 failures, unsafe
  (too_many_snapshots), task_specs_sha256 `e81c4eebaded7075e477484df5ba8311fd4e6950d0df1e7d7a41c9f4a3a6f245`.
- New task_specs_sha256: compute from the release (the grader repair changed the specs digest).
- Evidence root: `s3://marin-us-east-02a/marin/rl-data/atlas/facet-quiesce-repair/20261124/`.
- Generator fix: landed `12694eb511` (atlas-env-audit). Quiesce fix detail in STATUS.md.
