# Mini840 and Micro210 experiment archive

This branch preserves the code behind [Mini840/Micro210, Marin issue #10804](https://github.com/marin-community/marin/issues/10804), compact validation subsets of [Eval Policy #10197](https://github.com/marin-community/marin/issues/10197) for the RL feedback loop. Its pull request is intentionally closed without merging. This is a historical research snapshot, not a maintained main-branch evaluation interface.

The directory contains task inventory and identity audits, random and learned selection methods, historical prediction and uncertainty analyses, timing launch/config materialization, TaskCompendium exporters, runtime profiles, original-policy index materialization, and release verification scripts. `../campaigns/mini_eval_cohort/` contains the earlier aggregate benchmark study. Explicit `selected_source_keys` support and its tests are preserved in the same branch because the original-policy subset experiments depend on it.

## Published inputs and outputs

- [Mini840](https://huggingface.co/datasets/open-athena/Marin-Mini-Eval-840): 840 unique tasks, 1,626 expanded rollout rows.
- [Micro210](https://huggingface.co/datasets/open-athena/Marin-Micro-Eval-210): the nested first 210 tasks, 450 expanded rollout rows.
- [Reproduction index](https://huggingface.co/datasets/open-athena/Marin-Mini-Eval-Policy-Index): frozen selections, original benchmark inputs/configs, source patches, pinned grading wheels, reports and public-readback receipts. Its README provides the dataset-loading and original-policy reproduction commands.
- [Frozen 21-to-21 study report](https://huggingface.co/datasets/open-athena/Marin-Mini-Eval-Policy-Index/blob/main/analysis/study-20261030-v6-21x21/report/COST_COMPARISON.md).

RL validation uses 65,536 total context tokens, 16,384 output tokens per turn, thinking enabled, Pi 0.87.0 for all agentic benchmarks, and generated-choice TruthfulQA. The separate original-policy index retains canonical benchmark-specific execution and MC2 TruthfulQA. The historical random-probe rank correlations use a fitted ridge predictor, which is not bundled in the validation datasets. Raw reward aggregation and the simplified execution profile have not been prospectively validated against the full policy.

Frozen published revisions: Mini840 `bae388a44fc0ba9ffa551a4c6347fbe03a9fefd7`; Micro210 `4564ed081c316bc3231ae5c3a53a46b39248c819`; index data/configs `d5d2f92ce4866c431063bd123ea31a3865530809`. Index metadata and public-readback verification are preserved at `f37d583e157163e4d1f43651447689f3a1339aef`. Pin these commits when downloading; unqualified links above are for browsing.

## Historical runtime and commands

The exact dated launch manifests/configs are preserved compactly, without formatting changes, in `reproduction/runtime-configs-20261030.tar.gz`. Restore them before running historical timing/collection entrypoints:

```bash
tar -xzf experiments/evaluation/mini_policy/reproduction/runtime-configs-20261030.tar.gz \
  -C experiments/evaluation/mini_policy
```


The published release was built from Marin `815c4b2c8197bcf1613b405d3cad144dbe0b9f81` plus the immutable source patch in the reproduction index. This archival branch starts from a later main revision; use the published source snapshot and pinned wheels when exact runtime reproduction is required. `reproduction/historical-grading.patch` separately preserves the study's NUPA/GSM8K grading changes without altering the graders on this branch. NUPA also has an independent implementation in PR #10733. Do not assume the historical patch and that PR have identical semantics.

Entrypoints run from the repository root with `uv run python -m experiments.evaluation.mini_policy.<module>`. Each analysis module's CLI documents its input paths and output destination. Collection/export scripts retain fixed, dated S3 staging roots and require the corresponding access credentials. The local completion watcher is operational glue and must be explicitly configured; importing this directory does not launch jobs or monitors.

To rebuild the original-policy subset YAMLs from a downloaded index repository:

```bash
uv run python -m experiments.evaluation.mini_policy.reproduce_release_policy \
  --index /path/to/Marin-Mini-Eval-Policy-Index --budget 840 --output /tmp/mini840-policy
```

`collect_release_sources`, `collect_release_harbor`, `build_release_runtime`, `build_release_dataset`, `verify_release_dataset` and `verify_published_release` preserve the release pipeline. The index contains the immutable runtime wheel revision and verification receipts. Generations and substantive analysis run on Iris; local scripts orchestrate jobs and publish bounded metadata.
