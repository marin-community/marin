# Stale Branch Cleanup Report for marin-community/marin

**Generated:** November 15, 2025
**Analysis Date Cutoffs:**
- Very Old: Before May 15, 2024 (>1 year)
- Old: May 15, 2024 - November 15, 2024 (6-12 months)
- Recent: After November 15, 2024 (<6 months)

---

## Executive Summary

**Total Branches:** 372 (excluding main)

### Priority Breakdown:
- 🔴 **PRIORITY 1 - DELETE NOW:** 64 branches (>1 year old)
- 🟡 **PRIORITY 2 - REVIEW FIRST:** 116 branches (6-12 months old)
- 🟢 **KEEP FOR NOW:** 190 branches (<6 months old)

### Special Categories:
- 🤖 Bot/Automated branches (codex/, claude/, gh/): 49 total
- 🔧 Temp/Test/WIP/Fix branches: 47 total

---

## Priority 1: Very Old Branches (>1 year) - SAFE TO DELETE

### Summary
All 64 branches in this category are over 1 year old and have not been merged. These are prime candidates for immediate deletion.

### Complete List (64 branches):

1. `wip_abhinav_fineweb` - 2024-05-05 (by Abhinav Garg)
2. `ksalahi/pg` - 2024-05-14 (by Kamyar Salahi)
3. `download-hf-dataset` - 2024-05-15 (by Joel Niklaus)
4. `matt-pes2o` - 2024-06-13 (by mattyding)
5. `ksalahi/server-fix` - 2024-06-17 (by Kamyar Salahi)
6. `ksalahi/no-min` - 2024-06-26 (by Kamyar Salahi)
7. `wikipedia` - 2024-06-30 (by J38)
8. `pubmed` - 2024-07-01 (by mattyding)
9. `matt/data_analysis` - 2024-07-07 (by mattyding)
10. `ksalahi/comrak` - 2024-07-11 (by Kamyar Salahi)
11. `change_law_root_dir_dolma` - 2024-07-13 (by Joel Niklaus)
12. `ksalahi/ar5iv` - 2024-07-15 (by Kamyar Salahi)
13. `wip-fineweb` - 2024-07-15 (by Abhinav Garg)
14. `pubmed_v2` - 2024-07-16 (by mattyding)
15. `rohith-reddit` - 2024-08-01 (by RohithKuditipudi)
16. `ksalahi/evals` - 2024-08-05 (by Kamyar Salahi)
17. `marin-config` - 2024-08-10 (by Abhinav Garg)
18. `tokenize` - 2024-08-14 (by David Hall)
19. `ksalahi/evals-helm` - 2024-08-19 (by Kamyar Salahi)
20. `ray_config_transformers` - 2024-08-24 (by Abhinav Garg)
21. `annotate_fix` - 2024-08-25 (by Abhinav Garg)
22. `ray-job-launcher` - 2024-08-25 (by Ivan-Zhou)
23. `gh-actions-demo` - 2024-08-28 (by J38)
24. `classifier_model_types` - 2024-09-05 (by David Hall)
25. `ksalahi/eleuther-evaluator` - 2024-09-09 (by Kamyar Salahi)
26. `revert-248-trafilatura-extraction-support` - 2024-09-11 (by Abhinav Garg)
27. `helm-summarize` - 2024-09-22 (by Abhinav Garg)
28. `single_script` - 2024-09-22 (by Percy Liang)
29. `add-ngram-to-dedupe` - 2024-09-24 (by J38)
30. `alapca` - 2024-09-24 (by Tony Lee)
31. `mmlu-data-draccus` - 2024-09-24 (by J38)
32. `bert-consistency` - 2024-09-25 (by Tony Lee)
33. `rohith-move-scripts` - 2024-09-25 (by RohithKuditipudi)
34. `nikil/change-ray-init` - 2024-09-26 (by Nikil Ravi)
35. `migrate-parq-proc-to-draccus` - 2024-09-27 (by Herumb Shandilya)
36. `draccus_eval` - 2024-09-28 (by Tony Lee)
37. `evalquickstart` - 2024-09-28 (by Tony Lee)
38. `quickstart_ci` - 2024-09-29 (by Abhinav Garg)
39. `web-content-extraction-executor` - 2024-09-29 (by Herumb Shandilya)
40. `traf-error-handling-patch` - 2024-09-30 (by Herumb Shandilya)
41. `revert-356-content-extraction-executor` - 2024-10-01 (by Herumb Shandilya)
42. `dclm_1b_1x` - 2024-10-08 (by Abhinav Garg)
43. `nikil/data_browser` - 2024-10-14 (by Nikil Ravi)
44. `add-html-fineweb` - 2024-10-16 (by Herumb Shandilya)
45. `independent_tokenize` - 2024-10-16 (by David Hall)
46. `nikil/evals-ci` - 2024-10-16 (by Nikil Ravi)
47. `nikil/workflow_triggers` - 2024-10-18 (by Nikil Ravi)
48. `even_faster_recovery` - 2024-10-18 (by David Hall)
49. `nikil/tokenize_parquet` - 2024-10-21 (by Nikil Ravi)
50. `ksalahi/mmlu-data` - 2024-10-23 (by J38)
51. `chris/oh2.5-transform` - 2024-10-28 (by Christopher Chou)
52. `nikil/testing_docs` - 2024-10-28 (by Nikil Ravi)
53. `ksalahi/internal-eval` - 2024-10-28 (by Kamyar Salahi)
54. `exp_prefix` - 2024-10-29 (by David Hall)
55. `nikil/dedup_fix` - 2024-10-29 (by Nikil Ravi)
56. `runtime` - 2024-10-30 (by Abhinav Garg)
57. `abhi/exec_info` - 2024-10-30 (by Abhinav Garg)
58. `fix_train` - 2024-11-02 (by David Hall)
59. `prune-fineweb-dataset` - 2024-11-03 (by Herumb Shandilya)
60. `rohith-classifier-refactor` - 2024-11-04 (by RohithKuditipudi)
61. `arc_challenge_eval` - 2024-11-05 (by chiheem)
62. `rohith-bert-training-experiment` - 2024-11-11 (by RohithKuditipudi)
63. `abhi/hb` - 2024-11-12 (by Abhinav Garg)
64. `nikil/dev-metrics` - 2024-11-13 (by Nikil Ravi)

---

## Priority 2: Old Branches (6-12 months) - REVIEW BEFORE DELETING

### Summary
116 branches that are 6-12 months old. Recommend contacting owners before deletion.

### Complete List (116 branches):

1. `nikil/dclm-run` - 2024-11-16 (by Nikil Ravi)
2. `ksalahi/mmlu` - 2024-11-17 (by Kamyar Salahi)
3. `ksalahi/piqa` - 2024-11-18 (by Kamyar Salahi)
4. `ksalahi/openqa` - 2024-11-18 (by Kamyar Salahi)
5. `ksalahi/arc-easy` - 2024-11-18 (by Kamyar Salahi)
6. `ksalahi/winogrande` - 2024-11-18 (by Kamyar Salahi)
7. `ksalahi/hellaswag` - 2024-11-18 (by Kamyar Salahi)
8. `ksalahi/boolq` - 2024-11-18 (by Kamyar Salahi)
9. `bug_ray_run` - 2024-11-19 (by Abhinav Garg)
10. `chris/102_classifier_ablations` - 2024-11-19 (by Christopher Chou)
11. `nikil/eval_device` - 2024-11-19 (by Nikil Ravi)
12. `changelog` - 2024-11-19 (by Abhinav Garg)
13. `bug_fix_runtime` - 2024-11-19 (by Christopher Chou)
14. `pip-deps-fix` - 2024-11-19 (by Abhinav Garg)
15. `add_ngram_features` - 2024-11-20 (by J38)
16. `chris/mmlu-dataset` - 2024-11-20 (by Christopher Chou)
17. `rohith-bert-experiment` - 2024-12-01 (by Rohith Kuditipudi)
18. `compel` - 2024-12-04 (by Brando Miranda)
19. `test_branch_quickstart` - 2024-12-20 (by Abhinav Garg)
20. `decontaminate_exp` - 2025-01-06 (by J38)
21. `chris/dclm-reddit` - 2025-01-08 (by Christopher Chou)
22. `chris/fix-torchxla` - 2025-01-08 (by Abhinav Garg)
23. `nikil/ruff_fix` - 2025-01-09 (by Nikil Ravi)
24. `chris/eval-past` - 2025-01-10 (by Christopher Chou)
25. `tpu_monitoring_wandb` - 2025-01-13 (by Abhinav Garg)
26. `chris/vllm-label` - 2025-01-14 (by Christopher Chou)
27. `quickstart-test-fix` - 2025-01-15 (by Abhinav Garg)
28. `status_actor_fix` - 2025-01-16 (by Abhinav Garg)
29. `cathy/readme-nit` - 2025-01-23 (by Cathy Zhou)
30. `tokenizer_retry` - 2025-01-23 (by Herumb Shandilya)
31. `chris/download-model` - 2025-01-24 (by Christopher Chou)
32. `nikil/scaling_laws` - 2025-01-24 (by Nikil Ravi)
33. `launch_vm` - 2025-01-27 (by Nelson Liu)
34. `nemotron-cc-import` - 2025-01-28 (by Herumb Shandilya)
35. `cluster_switcher` - 2025-01-31 (by David Hall)
36. `wip_ab/status_actor` - 2025-02-03 (by Abhinav Garg)
37. `suhas/curriculum-actor-revert` - 2025-02-03 (by Suhas Kotha)
38. `jasonw/mup` - 2025-02-07 (by Jason Wang)
39. `data_browser_url_fix` - 2025-02-08 (by Nikil Ravi)
40. `partial_checkpoint` - 2025-02-12 (by David Hall)
41. `wip_ab/docker_changes` - 2025-02-12 (by Abhinav Garg)
42. `kaiyue/debug_tokenize` - 2025-02-13 (by Kaiyue Wen)
43. `suhas/cooldown-config` - 2025-02-13 (by William Held)
44. `nikil/scaling_law_exps` - 2025-02-16 (by Nikil Ravi)
45. `nikil/levanter-lm-eval` - 2025-02-17 (by Nikil Ravi)
46. `chris/mind` - 2025-02-22 (by Christopher Chou)
47. `chris/fix-alpaca` - 2025-02-25 (by Ahmed Ahmed)
48. `oops` - 2025-02-27 (by David Hall)
49. `nikil/eval` - 2025-02-28 (by Nikil Ravi)
50. `ray_run_pyproject` - 2025-02-28 (by David Hall)
51. `viz` - 2025-03-06 (by David Hall)
52. `metrics_wandb` - 2025-03-06 (by Abhinav Garg)
53. `suhas/curriculum` - 2025-03-10 (by Suhas Kotha)
54. `suhas/curriculum-merge` - 2025-03-10 (by Suhas Kotha)
55. `elyas/compel` - 2025-03-11 (by Elyas Obbad)
56. `rohith-reasoning-classifier` - 2025-03-13 (by RohithKuditipudi)
57. `chris/medu` - 2025-03-13 (by Christopher Chou)
58. `suhas/synthetic` - 2025-03-13 (by Suhas Kotha)
59. `nemotron_tokenize` - 2025-03-18 (by David Hall)
60. `vinference-integration` - 2025-03-18 (by erfanzar)
61. `cathy/babysit_tpu` - 2025-03-21 (by Abhinav Garg)
62. `nikil/add-nemotron` - 2025-03-22 (by Nikil Ravi)
63. `chris/eval-phase3` - 2025-03-24 (by Ahmed Ahmed)
64. `fix_alp_eval` - 2025-03-24 (by Ahmed Ahmed)
65. `nikil/add-nemotron-data` - 2025-03-30 (by Nikil Ravi)
66. `histo_spoon` - 2025-04-02 (by David Hall)
67. `will/vibe_checker` - 2025-04-03 (by William Held)
68. `refactor_train_on_pod` - 2025-04-03 (by David Hall)
69. `chris/top-urls` - 2025-04-03 (by Christopher Chou)
70. `chris/medu-mmlu` - 2025-04-03 (by Christopher Chou)
71. `add_legalbench_eval` - 2025-04-04 (by Joel Niklaus)
72. `experiments/exp943_fp32_attn` - 2025-04-04 (by David Hall)
73. `auto_ip` - 2025-04-05 (by David Hall)
74. `single_key_cluster` - 2025-04-07 (by David Hall)
75. `fix_cluster_again` - 2025-04-07 (by David Hall)
76. `nikil/quickstart_fix` - 2025-04-10 (by Nikil Ravi)
77. `nikil/dev` - 2025-04-11 (by Nikil Ravi)
78. `chris/perplexity` - 2025-04-12 (by Christopher Chou)
79. `kaiyue/exp961` - 2025-04-12 (by Kaiyue Wen)
80. `chris/rubric-qa` - 2025-04-12 (by Christopher Chou)
81. `og_alpaca` - 2025-04-13 (by Ahmed Ahmed)
82. `medu-sciences-crawling` - 2025-04-15 (by Herumb Shandilya)
83. `more_spoonbill` - 2025-04-15 (by David Hall)
84. `bert_url_classification` - 2025-04-15 (by Nelson Liu)
85. `nikil/megamath` - 2025-04-17 (by Nikil Ravi)
86. `elyas/compel-8b` - 2025-04-20 (by Elyas Obbad)
87. `nikil/scaling_proj` - 2025-04-23 (by Nikil Ravi)
88. `autopm` - 2025-04-24 (by David Hall)
89. `suhas/curriculum-3-10` - 2025-04-26 (by Suhas Kotha)
90. `chris/update-configs` - 2025-04-27 (by Christopher Chou)
91. `update-bert-training-infra` - 2025-04-28 (by Rohith Kuditipudi)
92. `fix_vllm_cluster` - 2025-04-28 (by Ahmed Ahmed)
93. `nikil/ci` - 2025-04-29 (by Nikil Ravi)
94. `test_tpu_test` - 2025-04-30 (by Abhinav Garg)
95. `test_tpu_2` - 2025-04-30 (by Abhinav Garg)
96. `joel/fix-broken-links` - 2025-05-01 (by Joel Niklaus)
97. `will/dclm_1b1x` - 2025-05-01 (by Helw150)
98. `crawl-annealing-runs` - 2025-05-02 (by Herumb Shandilya)
99. `nikil/docs` - 2025-05-02 (by Nikil Ravi)
100. `vsurge` - 2025-05-03 (by erfanzar)
101. `suhas/train-val-split` - 2025-05-07 (by Suhas Kotha)
102. `refactor-synthetic-rohith` - 2025-05-08 (by RohithKuditipudi)
103. `sft_doc` - 2025-05-08 (by Sherry Yang)
104. `readme_tweaks` - 2025-05-08 (by David Hall)
105. `rtd_fix` - 2025-05-08 (by Abhinav Garg)
106. `nikil/add-docs` - 2025-05-08 (by Nikil Ravi)
107. `docs-revamp` - 2025-05-09 (by Herumb Shandilya)
108. `rohith-distillation` - 2025-05-09 (by RohithKuditipudi)
109. `dclm-hd-extraction` - 2025-05-10 (by Herumb Shandilya)
110. `crawl-quality-ablations` - 2025-05-10 (by Herumb Shandilya)
111. `bump_cluster` - 2025-05-10 (by David Hall)
112. `update-filtering-data-rohith` - 2025-05-11 (by RohithKuditipudi)
113. `experiments/exp977_phoenix_cooldown` - 2025-05-12 (by David Hall)
114. `nikil/scaling-speedrun` - 2025-05-12 (by Nikil Ravi)
115. `wandb_entity` - 2025-05-12 (by David Hall)
116. `docs-review` - 2025-05-13 (by Herumb Shandilya)

---

## Action Plan

### Immediate Actions (Priority 1):
1. Review the 64 very old branches listed above
2. Verify none are referenced in active PRs or documentation
3. Delete using the script below

### Follow-up Actions (Priority 2):
1. Contact owners of the 116 old branches
2. Ask if they're still needed
3. Set a 30-day deadline for response
4. Delete unresponsive branches after deadline

### Ongoing Maintenance:
1. Set up branch protection rules
2. Implement automatic stale branch notifications (90 days)
3. Consider GitHub branch cleanup policies

---

## Deletion Scripts

### Script 1: Delete All Very Old Branches (>1 year)

Save this to a file like `delete_stale_branches.sh`:

```bash
#!/bin/bash
# Delete very old branches (>1 year old)
# REVIEW THIS LIST BEFORE RUNNING!

# You may want to run this in batches and verify each batch

# Batch 1 (First 20):
git push origin --delete wip_abhinav_fineweb
git push origin --delete ksalahi/pg
git push origin --delete download-hf-dataset
git push origin --delete matt-pes2o
git push origin --delete ksalahi/server-fix
git push origin --delete ksalahi/no-min
git push origin --delete wikipedia
git push origin --delete pubmed
git push origin --delete matt/data_analysis
git push origin --delete ksalahi/comrak
git push origin --delete change_law_root_dir_dolma
git push origin --delete ksalahi/ar5iv
git push origin --delete wip-fineweb
git push origin --delete pubmed_v2
git push origin --delete rohith-reddit
git push origin --delete ksalahi/evals
git push origin --delete marin-config
git push origin --delete tokenize
git push origin --delete ksalahi/evals-helm
git push origin --delete ray_config_transformers

# Batch 2 (Next 20):
git push origin --delete annotate_fix
git push origin --delete ray-job-launcher
git push origin --delete gh-actions-demo
git push origin --delete classifier_model_types
git push origin --delete ksalahi/eleuther-evaluator
git push origin --delete revert-248-trafilatura-extraction-support
git push origin --delete helm-summarize
git push origin --delete single_script
git push origin --delete add-ngram-to-dedupe
git push origin --delete alapca
git push origin --delete mmlu-data-draccus
git push origin --delete bert-consistency
git push origin --delete rohith-move-scripts
git push origin --delete nikil/change-ray-init
git push origin --delete migrate-parq-proc-to-draccus
git push origin --delete draccus_eval
git push origin --delete evalquickstart
git push origin --delete quickstart_ci
git push origin --delete web-content-extraction-executor
git push origin --delete traf-error-handling-patch

# Batch 3 (Next 20):
git push origin --delete revert-356-content-extraction-executor
git push origin --delete dclm_1b_1x
git push origin --delete nikil/data_browser
git push origin --delete add-html-fineweb
git push origin --delete independent_tokenize
git push origin --delete nikil/evals-ci
git push origin --delete nikil/workflow_triggers
git push origin --delete even_faster_recovery
git push origin --delete nikil/tokenize_parquet
git push origin --delete ksalahi/mmlu-data
git push origin --delete chris/oh2.5-transform
git push origin --delete nikil/testing_docs
git push origin --delete ksalahi/internal-eval
git push origin --delete exp_prefix
git push origin --delete nikil/dedup_fix
git push origin --delete runtime
git push origin --delete abhi/exec_info
git push origin --delete fix_train
git push origin --delete prune-fineweb-dataset
git push origin --delete rohith-classifier-refactor

# Batch 4 (Remaining):
git push origin --delete arc_challenge_eval
git push origin --delete rohith-bert-training-experiment
git push origin --delete abhi/hb
git push origin --delete nikil/dev-metrics
```

### Script 2: Safe Deletion (with confirmation)

```bash
#!/bin/bash
# Safe deletion script with confirmation for each branch

BRANCHES=(
  "wip_abhinav_fineweb"
  "ksalahi/pg"
  "download-hf-dataset"
  "matt-pes2o"
  "ksalahi/server-fix"
  "ksalahi/no-min"
  "wikipedia"
  "pubmed"
  "matt/data_analysis"
  "ksalahi/comrak"
  "change_law_root_dir_dolma"
  "ksalahi/ar5iv"
  "wip-fineweb"
  "pubmed_v2"
  "rohith-reddit"
  "ksalahi/evals"
  "marin-config"
  "tokenize"
  "ksalahi/evals-helm"
  "ray_config_transformers"
  "annotate_fix"
  "ray-job-launcher"
  "gh-actions-demo"
  "classifier_model_types"
  "ksalahi/eleuther-evaluator"
  "revert-248-trafilatura-extraction-support"
  "helm-summarize"
  "single_script"
  "add-ngram-to-dedupe"
  "alapca"
  "mmlu-data-draccus"
  "bert-consistency"
  "rohith-move-scripts"
  "nikil/change-ray-init"
  "migrate-parq-proc-to-draccus"
  "draccus_eval"
  "evalquickstart"
  "quickstart_ci"
  "web-content-extraction-executor"
  "traf-error-handling-patch"
  "revert-356-content-extraction-executor"
  "dclm_1b_1x"
  "nikil/data_browser"
  "add-html-fineweb"
  "independent_tokenize"
  "nikil/evals-ci"
  "nikil/workflow_triggers"
  "even_faster_recovery"
  "nikil/tokenize_parquet"
  "ksalahi/mmlu-data"
  "chris/oh2.5-transform"
  "nikil/testing_docs"
  "ksalahi/internal-eval"
  "exp_prefix"
  "nikil/dedup_fix"
  "runtime"
  "abhi/exec_info"
  "fix_train"
  "prune-fineweb-dataset"
  "rohith-classifier-refactor"
  "arc_challenge_eval"
  "rohith-bert-training-experiment"
  "abhi/hb"
  "nikil/dev-metrics"
)

for branch in "${BRANCHES[@]}"; do
  echo "About to delete: $branch"
  read -p "Delete this branch? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin --delete "$branch"
    echo "✓ Deleted: $branch"
  else
    echo "✗ Skipped: $branch"
  fi
done
```

---

## Notes

- **No branches are fully merged:** All 370 branches are unmerged into main
- This suggests the repository might benefit from a rebase/merge cleanup policy
- Consider implementing a "squash and merge" policy for feature branches
- Bot branches (codex/, claude/, gh/) should be cleaned up automatically

---

**Report Generated By:** Claude Code Analysis
**Date:** November 15, 2025
