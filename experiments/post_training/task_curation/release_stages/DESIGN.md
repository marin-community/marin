# DESIGN — scripted Atlas release pipeline (R40 gate)

One entry point builds an Atlas release end to end; the agent reviews instead of
scheduling. This note defines the stage graph, the typed release manifest and the
submission guard. It accompanies the retirement inventory that
`inventory.py` writes to `~/Documents/experiments/active/atlas-preview-rebuild/INVENTORY.md`.

## 1. Shape

- Entry point: `atlas-release --version <release>` — module
  `experiments.post_training.task_curation.release_stages.__main__`, registered as a
  console script. Modes: `plan` (print the lowered graph, I/O-free), `preflight`
  (per-stage type-check + small-slice run on real inputs), `run` (full build),
  `status` (artifact records), `resume` (rerun; skips completed stages, restarts
  failed stages alone).
- The graph is a Marin lazy `ArtifactStep` graph (`marin.execution.lazy`): steps are
  content-free handles; `StepRunner` schedules them with `max_concurrent`, prunes
  subtrees whose artifact record already exists with a matching fingerprint
  (`step_is_built`), and supports `dry_run=True`. Parallel stages are sibling
  dependencies of their join step, so the runner runs them concurrently.
- Stage workers are one maintained module per stage in this package. All cluster
  dispatch goes through `marin.execution.remote(fn, resources=...)`; per-release
  launcher scripts, `sed`ed worker copies and `<release>`-named variants are retired
  (see INVENTORY.md). Stage graphs/bindings live under `experiments/**` per the
  pipeline placement policy.

## 2. Stages

Each stage module exposes `step(...) -> ArtifactStep[XArtifact]` with typed inputs and
outputs. `build_config(ctx)` reads only `ctx` values and typed manifest artifact
fields — never release-identifying literals (§4).

| Module | Stage | Consumes | Produces (Artifact) |
| --- | --- | --- | --- |
| `declarations.py` | Declarations | intake list, registry | `DeclarationsArtifact` (uri, sha256, source pins) |
| `catalog_stage.py` | Catalog staging | declarations, reviews | `CatalogPlanArtifact` (catalog uri, sha, counts) |
| `census.py` | Census | declarations, catalog plan | `CensusReceipt` (population digests, per-source counts) |
| `corpus_assembly.py` | Corpus assembly | census | `CorpusReceipt` (corpus uri, sha, inventory totals) |
| `embed.py` | Embedding | corpus, backend parity receipt | `EmbedReceipt` (vectors uri, sha, device) |
| `curriculum_compose.py` | Curriculum compose | embed, census, prior curriculum | `CurriculumPlanArtifact` (plan uri, plan hash) |
| `backend_validation.py` | Backend validation | census inputs | backend census manifest + `BackendValidationReceipt` (the release's snapshot-bound survey) |
| `identity_backend_compose.py` | Identity/backend compose | census, surveys, backend validation | `IdentitySurveyReceipt`, `BackendValidationReceipt` |
| `fresh_qc_index.py` | Fresh QC index | QC browse | `FreshQcIndexReceipt` (the staged SAMPLE panels bound to the target load) |
| `history_staging.py` | History staging | curriculum plan, prior release | `HistoryStagingReceipt` |
| `text_scans.py` | Text scans | curriculum plan | `TextScanReceipt` |
| `qc_browse.py` | QC browse | curriculum plan, history staging | `QcBrowseReceipt` (plan hash, browse uri) |
| `snapshot_publication.py` | Snapshot publication | census, compose, identity/backend | `SnapshotArtifact` (snapshot name, uri, sha256) |
| `catalog_load.py` | Catalog load | snapshot, prior pin | `CatalogLoadArtifact` (load id, uri, sha256) |
| `curriculum_load.py` | Curriculum load | snapshot, history staging | `CurriculumLoadArtifact` (load id, uri, sha256) |
| `strict_audit.py` | Strict audit | loads, snapshot | `StrictAuditReceipt` |
| `package_preview.py` | Package preview | audit, plans | `PreviewPackageArtifact` (package uri, sha) |
| `release_check.py` | Release check | package | `ReleaseCheckReceipt` (preflight/hosted/postpublish phases) |
| `marina_release.py` | Marina release | release check | `MarinaReleaseReceipt` |
| `hosted_check.py` | Hosted check | marina release | `HostedCheckReceipt` |
| `exposure_probe.py` | Exposure probe | hosted release | `ExposureProbeReceipt` |

Parallel structure (the runner enforces it; activation waits for all inputs):

```
declarations → census ─┬→ corpus assembly → embed → curriculum compose ─┐
                       ├→ identity compose ─┬→ identity/backend compose ┼→ snapshot publication
                       └→ backend validation ┘                          │
                                              catalog staging ──────────┘
snapshot publication → catalog load ─┬→ strict audit → package preview → release check
                     → curriculum load ┘        → marina release → hosted check → exposure probe
```

Exact edges are finalized with the r39 builder's review of each stage's quirks
(BRIEF.md reviewer clause); the manifest carries whatever prior pins a stage needs.

## 3. Typed release manifest (`manifest.py`)

`ReleaseManifest` is an `Artifact` (pydantic model) produced by an early step and
consumed by every stage. Fields are graph outputs; nothing here is ever a literal in
worker source:

- `release_id: str` — the only release-identifying value, supplied once at the CLI
  boundary (`--version`), never baked into modules.
- `prior: PriorPin` — prior release id, prior load id, prior snapshot uri + sha256,
  prior catalog uri (the "prior pin" for history staging and loads).
- `declarations: ArtifactRef` — uri + sha256 of the declaration set.
- `census: CensusRef` — receipt uri + sha256, `population_total`, per-source counts
  (sizing input).
- `plans: dict[str, PlanRef]` — corpus plan, curriculum compose plan, qc browse plan:
  uri + sha256 each (the "plan hashes" that flow through the graph).
- `identity: IdentityRef` — survey uri + sha256 + survey count.
- `backend: BackendRef` — runtime id, parity receipt uri, `gpu_eligible: bool`,
  validation receipt uri.
- `snapshot: SnapshotRef` — snapshot name, uri, sha256 (output of snapshot
  publication; input to loads).
- `loads: dict[str, LoadRef]` — catalog load id and curriculum load id with uris and
  sha256 (outputs of the load stages; the 32-hex load IDs live here and only here).
- `guard: GuardReceipt` — repo sha, atlas-env-audit sha, checked-at (§5).
- `stage_receipts: dict[str, ReceiptRef]` — per-stage receipt artifacts (resume,
  alerts, status).

## 4. No-literals rule (R40 gate §2)

Load IDs, snapshot IDs, prior and target pins and plan hashes are typed manifest /
artifact fields passed as step inputs. Stage worker source contains none of them. The
guard enforces this mechanically (§5), and the r39 history-stage failure (a `sed`ed
worker whose target load ID literal was never replaced) is the regression this
prevents.

## 5. Submission guard (`guard.py`, R40 gate §3)

The submission path (`atlas-release run`, and every `remote()` submission inside it)
refuses to proceed unless all of the following hold; failures are hard, with no
bypass flag:

1. **Landed module**: the entry point of every submitted stage resolves to a module
   file under `experiments/post_training/task_curation/release_stages/` in the current
   checkout, and that checkout HEAD is landed on `atlas-env-audit` — HEAD is an
   ancestor-or-equal of `origin/atlas-env-audit`, or appears in the atlas-push landing
   log (`ATLAS_PUSH_LOG`, default `~/.local/share/atlas-push/landings.jsonl`).
   Local `plan`/`preflight` runs from a worktree are allowed but record their sha in
   the preflight receipt; submission additionally requires receipt sha == landed sha.
2. **No release literals**: an AST scan of every stage module, `graph.py`,
   `manifest.py` and `preflight.py` rejects any string literal that is a 32-hex load
   ID or matches release/snapshot-name shapes (`r\d+[-_].*`, `current\d+[-_].*`).
   Scan failure = refusal before any submission.
3. **Fresh preflight receipt** (§6): every stage has an `ok` preflight receipt whose
   repo sha matches the landed sha and whose input digests match the manifest.

Test: `tests/test_release_stage_guard.py` covers the landed-sha check, the literal
scanner (synthetic offenders) and a self-scan of this package.

## 6. Per-stage preflight (`preflight.py`, BRIEF requirement 3)

Before any cluster submission, each stage gets: a pyrefly type-check of its module
plus a dry run of the lowered graph, then a **small-slice run on a slice of real
inputs through the real code path** (bounded declarations/census rows/corpus slice,
never synthetic fixtures). The result is a `PreflightReceipt` artifact (stage, mode,
repo sha, input digests, ok, timestamp) written under the release prefix. Submission
refuses a stage whose receipt is missing, failed, or stale for the current source sha.

## 7. Sizing from data (`sizing.py`, BRIEF requirement 4)

Workers, memory and timeouts derive from the census population size via per-stage
curves recorded in code with their provenance (measured r31–r39 runs); no stale
defaults (the old embed v1 fixed 4 h timeout is retired). GPU embedding is used only
when a valid parity receipt exists for the serving runtime; otherwise CPU. The chosen
resources are recorded in the manifest and the stage receipt.

## 8. Alerts (`alerts.py`, BRIEF requirement 5)

Every stage emits `start` / `finish` / `fail` / `stall` events **directly** with
evidence links to the ops feed and the job-watchdog `alerts.jsonl` — failure
detection never depends on a registry edit (coordinator amendment, 2026-11-23). Each
stage submission additionally proposes its own watchdog `RegistryEntry` with a probe
mapping (JSON-field probe on the stage receipt, log-regex probe on the job log); the
registry stays operator-owned — the pipeline proposes the entry, the coordinator
merges it.

## 8b. Pre-write checks for SQL-writing stages (coordinator + r39 builder, 2026-11-23)

History staging, catalog load and curriculum load write to the served database. Each
of them (1) runs in a dry-run mode that computes the planned writes without writing,
and (2) checks its typed target IDs — target load ID, snapshot name, prior pins —
against the manifest before any write. This is the mechanical fix for the r39
history-stage failure (a `sed`ed worker whose target load ID literal was never
replaced): the worker asserts `planned_target == manifest target` and refuses
otherwise. Preflight exercises the dry-run path on a small slice.

Identity coverage follows declared populations. An Excluded inventory row with no
actor-census input may omit its fingerprint-map entry; every populated or
non-Excluded row still requires an identity-survey entry. Backend surveys cover
all inventory rows, including Excluded rows. Snapshot publication, catalog staging,
catalog loading and strict/hosted audits use the same coverage rule. The
loader preserves an omitted metadata-only fingerprint as SQL NULL.

## 9. Resume (BRIEF requirement 6)

Reruns skip stages whose artifact record exists with matching input fingerprints
(`StepRunner` pruning); failed stages restart alone via `run(handle,
force_run_failed=True)`. Because manifests and plans are typed artifacts, a changed
pin changes fingerprints and invalidates exactly the downstream stages.

## 10. Acceptance

Unchanged from the brief: a fresh Codex cold-read subagent must dry-run the entire
build from a clean checkout using only repo docs and the single entry point (all
stages in dry-run / small-slice mode), and additionally verify that no step reads or
executes anything under `~/Documents/experiments`. Cold-read transcripts and verdicts
are recorded with the release evidence.
