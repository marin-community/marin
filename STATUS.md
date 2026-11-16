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

## Phase 2 progress (2026-11-16 ~14:15 EST)

- **LANDED on atlas-env-audit as 9c40911e70** (atlas-push; tests + lint green on rebased head; advisory review findings addressed in a follow-up commit).
- Build: repair-match-v1 records + replace rule (lib/taskcompendium/pipeline/repair.py), match cascade (repair_match.py), comparative reviewer (repair_review.py, atlas-repair-comparative v1), FWeindel candidate loader with preview lowering (repair_candidates.py), dry-run CLI (repair_lane.py) with registry resolution + diagnostics. 27 behavior tests passing.
- Provenance agreed with qc-audit: finestore RepairProvenance/RepairAdvisory + AssessmentItem fields + loader columns landed by them; my publish path constructs AssessmentItems with those fields.

## Dry-run findings (in-region cw-us-east-02a, batch priority, CPU match only; 0 GPU spent)

- **BugsInPy 471 vs TaskTrove:laion__exp_rpt_bugsinpy-v4 (479 tracked): 0 overlap.** Best similarity ~0.018 both directions. Previews show the tracked source is laion's rewritten single-file katas ("fix the bug in /app/solution.py" from the sanic/pandas/black repos); FWeindel tasks are the original repo-level BugsInPy issues ("Work in the black project…# Issue…"). Same upstream corpus, entirely different derivations — NOT a cleaned subset. Recommendation: treat FWeindel BugsInPy as a NEW source through normal admission (its duplicate review vs the tracked source then records the upstream relationship), not the repair lane. 479 drop advisories suppressed by the new no-overlap guard.
- **NL2Bash 1,042 vs TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2 (prepared population): premise confirmed.** 1,034/1,042 candidates are the same "Terminal Automation Request" template family (best similarities 0.69–0.72), 8 new, 260 tracked absent. All 1,034 land CONFLICTING because FWeindel's repaired instructions dropped the tracked "## Environment Setup" preamble (~30% of text) — cross-conversion boilerplate, below the 0.9 repaired threshold by a hair. Policy question sent to the coordinator: (a) add a lane normalization that strips shared template setup sections before LSH (expected: most pairs → REPAIRED → FULL comparative review ≈1,034 pairs ≈ 25–40M tokens ≈ 4–8 GPU-h on 4xGB200), (b) keep 0.9/0.5 and accept ~1,034 manual conflict rulings (infeasible), or (c) rule the 0.5–0.9 same-template band as matched-repaired directly (user ruling).
- Report artifacts: s3://marin-us-east-02a/marin/rl-data/atlas/repair-candidates/20261116/{bugsinpy,nl2bash}-dry-run.json (bugsinpy report is pre-suppression-guard; census unchanged, advisories field stale — rerun value noted here).

## Next

- LANDED 9c40911e70 (lane build) and 4f674d012a (docs + STATUS) on atlas-env-audit.
- Awaiting coordinator/user ruling on the NL2Bash normalization policy; then the GPU comparative pass for whichever folder the user picks.
- Production merge/publish (new PRIVATE version + provenance) after a folder passes review + oracle/NoP.
