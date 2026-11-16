# STATUS — atlas-repair-candidate-lane (ses_e1bde51e6ffeF2cW2kNjaGy1nP)

Branch: atlas-repair-candidate-lane (off origin/atlas-env-audit).

## Milestones

- 2026-11-16 ~11:35 EST — Bound on rhizome, read BRIEF.md, created STATUS.md.
- 2026-11-16 ~12:00 EST — Phase 1 design sent to coordinator.
- 2026-11-16 ~12:40 EST — USER APPROVED ("Repair lane is approved."). Phase 2 started: repair-match-v1 match cascade, comparative review (ComparativeVerdict, GLM-5.3 temp 0.7), replace rule (oracle+NoP pass, not worse, no introduced defects), dropped=advisory, new PRIVATE tracked-source version with per-task provenance. Behavior tests + FWeindel BugsInPy dry run (471) + task-curation.md "Repair candidates" docs + guide.html text to coordinator. Land via atlas-push.

## Overlap audit summary (from coordinator, 2026-11-16)

CalibForge-style semantic TB variants exist. Recommended folders: BugsInPy 471, PyMethods2Test 4,990, NL2Bash 1,042, CalibForge 3,799, ScaleSWE v2 2,655, SWE-Lego v2 1,245 (97 SWE-bench-repo tasks to drop). 0 exact-ID overlaps with SWE-bench, but same-repo exposure.

## Phase 2 plan

1. Typed records in lib/taskcompendium (new file, no shared-file collisions): RepairMatchClass, MatchReceipt (repair-match-v1), ComparativeVerdict, ComparativeRepairReceipt, CandidateDropAdvisory, RepairProvenance {original, replaced, added}.
2. Match cascade in experiments/post_training/task_curation/repair_match.py: exact task_identity join -> normalized (normalized_instruction + reference_test_hashes) join -> LSH NN (task_neighbors.PopulationIndex, 0.9) -> classes per approved design.
3. Comparative review in repair_review.py: rubric atlas-repair-comparative v1, original+candidate payload (reuse originals= plumbing), GLM-5.3 V3 ReviewerSpec unchanged.
4. repair_lane.py CLI: match -> review -> decide -> RepairMergePlan + provenance manifest; --dry-run mode reports breakdown without writing.
5. Behavior tests with synthetic tracked+candidate sources (fake judge), covering matched-better, matched-worse, matched-identical, new, dropped, conflicting.
6. BugsInPy dry run on Iris (batch priority): match breakdown, comparative outcomes, GPU-h/tokens.
7. Docs: task-curation.md "Repair candidates"; guide.html text to coordinator.
8. Provenance column shape agreed directly with qc-audit (ses_e1be7f0bbffeCpBu5OYd0WKMkb).

## Phase 2 progress (2026-11-16 ~13:45 EST)

- **Build green.** New code: `lib/taskcompendium/src/taskcompendium/pipeline/repair.py` (MatchReceipt repair-match-v1, ComparativeVerdict BETTER/TIE/WORSE/INCOMPARABLE + defects fixed/introduced, ComparativeRepairReceipt, CandidateDropAdvisory, RepairProvenance, `repair_decision` replace rule); `repair_match.py` (content-only cascade: exact identity join -> equivalent join -> LSH NN; multi-claim conflicts; dropped advisories); `repair_review.py` (atlas-repair-comparative v1 rubric, original+candidate payload, ComparativePairReviewer with retries + token usage); `repair_candidates.py` (FWeindel folder loader + preview lowering, explicitly not an admission converter); `repair_lane.py` (dry-run CLI: registry resolution, match, review with pinned GLM-5.3 V3 judge, DryRunReport with GPU-h/tokens).
- Tests: `tests/test_repair_lane.py` — 26 behavior tests passing (identical/equivalent/neighbor/multi-claim-conflict/new/dropped cascade; threshold band pinned; replace-rule gates; comparative reviewer with local judge stub incl. retries/usage/identity checks; matched-pair workload). pre-commit + ruff clean.
- Provenance agreed with qc-audit: they landed RepairProvenance + RepairAdvisory in finestore, AssessmentItem fields, loader columns, /assessment + /evidence payloads. My lane constructs AssessmentItems with those fields when publishing; ping them before touching server/app.py, server/schema.py or lib/finestore.
- Dry run needs in-region execution (S3 credentials are region-only): the `repair_lane.py` dry-run CLI is ready to launch on Iris at batch priority.

## Next

- Launch BugsInPy 471 dry run on Iris (match + comparative review) at batch priority.
- Docs: task-curation.md "Repair candidates" stage + guide.html text to coordinator.
- Land via atlas-push; ping coordinator when worker-ready.
