# STALE BRANCH ANALYSIS - FINAL REPORT

**Generated:** 2025-11-15
**Total Branches Analyzed:** 370 (excluding main and rw/main)

## EXECUTIVE SUMMARY

- **93 branches** are fully merged and safe to delete
- **277 branches** have unmerged changes
- **347 branches** have merge-base with old main (origin/rw/main)
- **31 branches** have merge-base with current main

### Key Finding

The repository's main branch history was rewritten at some point, causing most branches to lose their merge-base with current `origin/main`. However, `origin/rw/main` preserves the old history, allowing proper merge detection.

---

## PRIORITY 1: FULLY MERGED BRANCHES (93 branches)

These branches are fully merged into the old main and can be safely deleted:

### Sample of Merged Branches:

- `ksalahi/openqa` - 1787 ahead, 59 behind - 2024-11-18 - Kamyar Salahi
- `marin-config` - 163 ahead, 59 behind - 2024-08-10 - Abhinav Garg
- `tokenize` - 221 ahead, 59 behind - 2024-08-14 - David Hall
- `single_script` - 654 ahead, 59 behind - 2024-09-22 - Percy Liang
- `alapca` - 673 ahead, 59 behind - 2024-09-24 - Tony Lee
- `bert-consistency` - 689 ahead, 59 behind - 2024-09-25 - Tony Lee
- `nikil/testing_docs` - 1429 ahead, 59 behind - 2024-10-28 - Nikil Ravi
- `chris/fix-alpaca` - 3102 ahead, 59 behind - 2024-09-25 - Christopher Chou
- `prune-fineweb-dataset` - 1556 ahead, 59 behind - 2024-11-03 - Herumb Shandilya
- `arc_challenge_eval` - 1653 ahead, 59 behind - 2024-11-05 - chiheem

**Note:** These branches show as "ahead" in commit count because they contain commits that were integrated via merge/squash into the old main before it was rewritten. The content is already in main.

---

## PRIORITY 2: OLD BRANCHES LIKELY STALE (104 branches)

Branches older than 180 days that were NOT merged:

### Minimal Changes (1-50 commits ahead):
- `download-hf-dataset` - 40 ahead, 549 days old - Joel Niklaus
- `wip_abhinav_fineweb` - 47 ahead, 559 days old - Abhinav Garg
- `ksalahi/pg` - 47 ahead, 550 days old - Kamyar Salahi

### Medium Changes (51-100 commits ahead):
- `matt-pes2o` - 70 ahead, 520 days old - mattyding
- `ksalahi/server-fix` - 71 ahead, 515 days old - Kamyar Salahi
- `pubmed` - 72 ahead, 502 days old - mattyding
- `ksalahi/no-min` - 82 ahead, 507 days old - Kamyar Salahi
- `wikipedia` - 82 ahead, 503 days old - J38

### Large Changes (101+ commits ahead):
- `wip-fineweb` - 102 ahead, 488 days old
- `matt/data_analysis` - 107 ahead, 495 days old
- `ksalahi/comrak` - 128 ahead, 491 days old
- ... and 96 more

---

## STATISTICS

### By Merge Status:
- Fully merged: **93 branches** (25%)
- Not merged: **277 branches** (75%)

### By Age (Not Merged Only):
- Recent (<180 days): **173 branches**
- Old (180-365 days): **72 branches**
- Very old (>365 days): **32 branches**

### Merge-Base Analysis:
- With merge-base to current main: **31 branches** (8%)
- With merge-base to old main (rw/main): **347 branches** (94%)
- No merge-base to either: **23 branches** (6%)

---

## DELETION SCRIPT

### Option 1: Delete All 93 Merged Branches

```bash
#!/bin/bash
# WARNING: Review this list carefully before running!
# These branches are identified as fully merged into old main

git push origin --delete marin-config
git push origin --delete tokenize
git push origin --delete single_script
git push origin --delete alapca
git push origin --delete bert-consistency
git push origin --delete rohith-move-scripts
git push origin --delete nikil/change-ray-init
git push origin --delete migrate-parq-proc-to-draccus
git push origin --delete draccus_eval
git push origin --delete evalquickstart
git push origin --delete traf-error-handling-patch
git push origin --delete nikil/data_browser
git push origin --delete add-html-fineweb
git push origin --delete nikil/tokenize_parquet
git push origin --delete ksalahi/mmlu-data
git push origin --delete chris/oh2.5-transform
git push origin --delete nikil/testing_docs
git push origin --delete nikil/dedup_fix
git push origin --delete runtime
git push origin --delete abhi/exec_info
git push origin --delete prune-fineweb-dataset
git push origin --delete rohith-classifier-refactor
git push origin --delete nikil/dev-metrics
git push origin --delete ksalahi/mmlu
git push origin --delete ksalahi/openqa
git push origin --delete ksalahi/winogrande
git push origin --delete ksalahi/arc-easy
git push origin --delete ksalahi/boolq
git push origin --delete chris/102_classifier_ablations
git push origin --delete bug_ray_run
git push origin --delete changelog
git push origin --delete pip-deps-fix
git push origin --delete bug_fix_runtime
git push origin --delete add_ngram_features
git push origin --delete chris/mmlu-dataset
git push origin --delete chris/fix-torchxla
git push origin --delete nikil/ruff_fix
git push origin --delete tpu_monitoring_wandb
git push origin --delete quickstart-test-fix
git push origin --delete status_actor_fix
git push origin --delete nikil/scaling_laws
git push origin --delete cathy/readme-nit
git push origin --delete chris/download-model
git push origin --delete nemotron-cc-import
git push origin --delete wip_ab/status_actor
git push origin --delete partial_checkpoint
git push origin --delete nikil/scaling_law_exps
git push origin --delete chris/fix-alpaca
git push origin --delete oops
git push origin --delete ray_run_pyproject

# ... See /tmp/final_branch_analysis.json for complete list
```

### Option 2: Interactive Review

Use the analysis JSON file to review branches individually:

```bash
# View the full analysis
cat /tmp/final_branch_analysis.json | jq '.[] | select(.is_merged == true) | {branch, age_days, author}'

# Delete branches interactively
python3 scripts/analyze_stale_branches_final.py
```

---

## METHODOLOGY

This analysis uses the following approach:

1. **Merge Detection**: Checks if branches are ancestors of `origin/rw/main` (the old main before history rewrite)
2. **Content Comparison**: Verifies no unique commits exist on the branch
3. **Age Analysis**: Categorizes branches by last commit date
4. **Merge-Base Validation**: Confirms common ancestry with historical main

### Why origin/rw/main?

The repository's main branch history was rewritten, orphaning most branches. The `origin/rw/main` ref preserves the original history, making it the correct base for merge detection.

---

## RECOMMENDATIONS

1. **Immediate Action**: Delete the 93 merged branches (after team review)
2. **Follow-up Review**: Contact owners of old unmerged branches (180+ days)
3. **Regular Cleanup**: Implement automated branch cleanup for merged PRs
4. **Documentation**: Document the history rewrite and use of rw/main for future reference

---

## FILES GENERATED

- `/tmp/final_branch_analysis.json` - Complete analysis data
- `/home/user/marin/scripts/analyze_stale_branches_final.py` - Analysis script
- `/home/user/marin/STALE_BRANCHES_FINAL_REPORT.md` - This report
