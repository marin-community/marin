# TaskTrove `.8` subset inventory and historical sampling plan

**Superseded for current release work by the `.9` inventory and sample set:**
[tasktrove-release-09-inventory-2026-09-14.md](tasktrove-release-09-inventory-2026-09-14.md)
and [tasktrove-clean-09-sample-metadata.json](tasktrove-clean-09-sample-metadata.json).
This file remains as provenance for the earlier `.8` fixture sweep.

This is a read-only inventory of the pinned raw TaskTrove catalog and the
current TaskCompendium TaskTrove archive importers. It does not change task
semantics or claim that a raw source directory survived clean release `.8`.

## Evidence and revision boundary

- Raw dataset: [`open-thoughts/TaskTrove`](https://huggingface.co/datasets/open-thoughts/TaskTrove)
  at revision
  [`02923004846e4e73862c20962f823a6d05100e7a`](https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/02923004846e4e73862c20962f823a6d05100e7a).
  The catalog was read from the [revision API](https://huggingface.co/api/datasets/open-thoughts/TaskTrove/revision/02923004846e4e73862c20962f823a6d05100e7a).
- `.8` producer: [`pipeline.py` at `ccc5ff24`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/pipeline.py),
  with `PIPELINE_VERSION = 2026.09.10.8`, raw `open-thoughts/TaskTrove`,
  `TASKTROVE_REVISION = 0292300`, and `RAW_VERSION = 2026.09.09`.
- The producer's verifier revision recorded in the existing inventory is
  [`b2b68d8b`](https://github.com/marin-community/marin/tree/b2b68d8b0a770cdc0ab3903780172c4b3eea81b1)
  (`lib/tasktrove-verify`).
- Producer source policy: [`source_verdicts.json` at `ccc5ff24`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/source_verdicts.json).
  Its `keep`/`drop` values are policy evidence, not a substitute for clean
  Parquet membership.
- Requested clean release: `s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8`.
  The release `manifest.json` and clean Parquet were not readable from this
  host, so `actual_clean_subset_names`, per-subset row counts, and the final
  sample rows remain unknown.
- The [historical clean report](https://github.com/marin-community/marin/blob/de412954e99488cf004b54104a7eeb4b670d10a2/.agents/projects/2026-09-09_tasktrove_clean.md)
  is useful for understanding the producer workflow, but its `.7` counts must
  not be reused as `.8` counts.

The raw catalog contains 133 source directories: 93 nondeprecated
`<source>/tasks.parquet` subsets and 40 under `deprecated/`. The source
directory lists below are exact for the pinned revision. The producer policy
contains 43 keep candidates and 50 drops among the 93 nondeprecated subsets. A
clean-release manifest is still required to know which keep candidates have at
least one retained row after conversion, deduplication, and verifier filtering.

## Current TaskCompendium representation

The classification below is for cleaned TaskTrove archives consumed through
`read_archive()` and the TaskTrove importers. Direct NeMo fixture importers do
not establish support for a TaskTrove subset with a different archive contract.

The following 15 producer-keep subsets have a matching current importer:

| importer | producer-keep subsets | boundary |
|---|---|---|
| [`tasktrove_shell`](../../lib/taskcompendium/src/taskcompendium/importers/tasktrove_shell.py) | `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` | `nl2bash` / `shell-cmd`; image and verifier files remain explicit inputs |
| [`tasktrove_coding`](../../lib/taskcompendium/src/taskcompendium/importers/tasktrove_coding.py) | `DCAgent__exp_rpt_curriculum-easy`, `DCAgent__exp_rpt_curriculum-medium-v2`, `DCAgent__exp_rpt_e2egit-large`, `DCAgent__exp_rpt_e2egit-v2`, `DCAgent__exp_rpt_multifile-v3`, `DCAgent__exp_rpt_pymethods2test-large-v2`, `DCAgent__exp_rpt_pymethods2test-v3`, `DCAgent__exp_rpt_stack-pytest-v2`, `DCAgent__exp_rpt_unitsyn-python-large-v2`, `DCAgent__exp_rpt_unitsyn-python-v4` | `python_unit_tests` / `pytest` only; caller supplies an immutable Python image |
| [`tasktrove_coding`](../../lib/taskcompendium/src/taskcompendium/importers/tasktrove_coding.py) | `laion__codeforces-v3` | `codeforces` / `stdio` only; caller supplies an immutable native image |
| [`tasktrove_answers`](../../lib/taskcompendium/src/taskcompendium/importers/tasktrove_answers.py) | `laion__nemotron-gym-knowledge-mcqa-v2` | `qa-short-answer` / `mcq` |
| [`tasktrove_judge`](../../lib/taskcompendium/src/taskcompendium/importers/tasktrove_judge.py) | `laion__nemotron-gym-knowledge-openqa-v4` | `qa-short-answer` / `nemotron_openqa`; a judge endpoint/configuration is still required at execution |
| [`tasktrove_math`](../../lib/taskcompendium/src/taskcompendium/importers/tasktrove_math.py) | `laion__all-puzzles-v2` | `all_puzzles` / `math` only |

One producer-keep subset is only partially represented:

- `laion__nemotron-gym-structured-outputs-v4` matches
  [`tasktrove_structured`](../../lib/taskcompendium/src/taskcompendium/importers/tasktrove_structured.py)
  for its supported JSON and XML contracts. The importer does not claim the
  producer's other schema variants, so sampling must retain the row's mode and
  schema type and report unsupported rows separately.

The remaining 27 producer-keep subsets are currently unsupported as
TaskTrove archive imports:

```text
DCAgent__code-contests-noblock
DCAgent__swe_rebench_v2_patched_oracle-v2
SankalpKJ__nemotron-math-oracle-filtered-v2
laion__exp_rpt_taco-v2
laion__glaive-code-assistant-sandboxes-verified-v2
laion__nemo-prism-math-v3
laion__nemotron-gym-agent-calendar-v2
laion__nemotron-gym-arc-agi-python-inductive-v2
laion__nemotron-gym-arc-agi-transductive-v3
laion__nemotron-gym-competitive-coding-v2
laion__nemotron-gym-instruction-following-calendar-v3
laion__nemotron-gym-instruction-following-structured-v3
laion__nemotron-gym-instruction-following-v3
laion__nemotron-gym-math-openmathreasoning-v2
laion__nemotron-gym-math-stack-overflow-v3
laion__nemotron-gym-math-v5
laion__nemotron-gym-multichallenge-advanced-v4
laion__nemotron-gym-reasoning-gym-v2
laion__nemotron-gym-safety-v3
laion__nemotron-gym-science-so-openq-v3
laion__stackexchange-codereview-sandboxes-verified-v2
laion__stackexchange-overflow-sandboxes-verified-v2
laion__stackexchange-superuser-sandboxes-verified-v2
laion__stackexchange-tezos-sandboxes-verified-v2
laion__stackexchange-unix-sandboxes-verified-v2
laion__swesmith-oracle-filtered-v2
laion__wizardlm-orca-v4
```

These unsupported statuses are contract statements, not quality judgments.
For example, `nemotron_competitive` is a producer converter for the direct
NeMo coding subset, while the current TaskTrove coding importer accepts only
`python_unit_tests` and `codeforces`; a direct NeMo importer does not make the
clean TaskTrove archive compatible.

## Full raw subset inventory

The following 50 nondeprecated subsets are present in the pinned raw catalog
but are producer drops, so they are not candidates for clean-release sampling
unless a later release manifest says otherwise:

```text
DCAgent__exp_rle_adversarial-v6
DCAgent__exp_rpt_nemotron-cpp
DCAgent__inferredbugs-sandboxes-verifier
DCAgent__mix_h4_binary_easy
DCAgent__selfinstruct-naive-sandboxes-2-verified-v3
SankalpKJ__nemotron-code-oracle-filtered
laion__codeelo-v2
laion__exp_rpt_bugsinpy-v4
laion__exp_rpt_codenet-python-v4
laion__exp_rpt_crosscodeeval-csharp-v4
laion__exp_rpt_crosscodeeval-java-v3
laion__exp_rpt_crosscodeeval-python-v2
laion__exp_rpt_crosscodeeval-typescript-v2
laion__exp_rpt_ghactions-v3
laion__exp_rpt_methods2test-large-v4
laion__exp_rpt_nemotron-junit-v6
laion__exp_rpt_scaffold-v3
laion__exp_rpt_stack-cpp-v4
laion__exp_rpt_stack-dockerfile-gpt5mini-v7
laion__exp_rpt_stack-go-v5
laion__exp_rpt_stack-jest-v5
laion__exp_rpt_stack-junit-v6
laion__exp_rpt_stack-php-large-v9
laion__exp_rpt_stack-pytest-large-v3
laion__exp_rpt_stack-rspec-v4
laion__magicoder-v4
laion__mix_h10_reward_proportional-v2
laion__mix_h11_single_skill_only-v2
laion__mix_h8_original_tests-v2
laion__nemotron-gym-agentic-function-calling-pivot-v3
laion__nemotron-gym-agentic-indirect-prompt-injection-v3
laion__nemotron-gym-agentic-swe-pivot-v4
laion__nemotron-gym-cfbench-v4
laion__nemotron-gym-identity-following-v4
laion__nemotron-gym-instruction-following-adversarial-v5
laion__nemotron-gym-instruction-following-citation-v2
laion__nemotron-gym-instruction-following-freeform-v2
laion__nemotron-gym-instruction-following-multiturnchat-v4
laion__nemotron-gym-inverse-ifeval-v4
laion__nemotron-gym-knowledge-web-search-mcqa-v2
laion__nemotron-gym-litmus-bench-v2
laion__nemotron-gym-math-advanced-calculations-v4
laion__nemotron-gym-multichallenge-vanilla-v3
laion__nemotron-gym-qa-abstention-v4
laion__nemotron-gym-sysbench-v4
laion__openswe-tasks-patched-v7-oracle-success
laion__r2egym-patched-full-oracle-v3
laion__swegym-tasks-patched-validated-v5
laion__toolscale-v4
laion__tulu3-sft-personas-math-sandboxes-verified-v3
```

For the producer's family and reason for every row, use the pinned
`source_verdicts.json` link above. This artifact intentionally does not copy
row counts from the historical `.7` report.

The 40 deprecated raw subsets are catalog evidence only and are excluded from
the `.8` candidate queue:

```text
DCAgent2__Toolscale-tasks
DCAgent2__Toolscale-tasks-cleaned
DCAgent2__nl2bash-tasks-cleaned-oracle
DCAgent__exp_rpt_codenet-python
DCAgent__exp_rpt_stack-dockerfile-v3
laion__exp_rle_detailed-v2
laion__exp_rle_error_report-v3
laion__exp_rle_github_issue-v3
laion__exp_rle_minimal_instructions-v3
laion__exp_rpt_codenet-python-v3
laion__exp_rpt_crosscodeeval-csharp-v2
laion__exp_rpt_crosscodeeval-csharp-v3
laion__exp_rpt_defects4j-v3-v4
laion__exp_rpt_exercism-python-v2
laion__exp_rpt_ghactions-v2
laion__exp_rpt_stack-csharp-v2
laion__exp_rpt_stack-csharp-v3
laion__exp_rpt_stack-csharp-v4
laion__exp_rpt_stack-junit-v2
laion__exp_rpt_stack-junit-v3
laion__exp_rpt_stack-junit-v4
laion__exp_rpt_stack-junit-v5
laion__exp_rpt_stack-php-large-v2
laion__exp_rpt_stack-php-large-v3
laion__exp_rpt_stack-php-large-v4
laion__exp_rpt_stack-php-large-v5
laion__exp_rpt_stack-ruby-v3
laion__glaive-code-assistant-sandboxes-verified
laion__nemotron-gym-agent-workplace-v2
laion__nemotron-gym-instruction-following-adversarial
laion__nemotron-gym-instruction-following-adversarial-v2
laion__nemotron-gym-math-advanced-calculations
laion__nemotron-gym-math-advanced-calculations-v2
laion__qasper-v3
laion__stackexchange-codereview-sandboxes-verified
laion__stackexchange-overflow-sandboxes-verified
laion__stackexchange-superuser-sandboxes-verified
laion__stackexchange-tezos-sandboxes-verified
laion__stackexchange-unix-sandboxes-verified
laion__staqc-v4
```

## Concrete next-step sampling manifest

The execution manifest should be materialized only after the `.8` release
manifest or clean Parquet is readable. It has this fixed shape:

```text
release = s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8
raw_dataset = open-thoughts/TaskTrove
raw_revision = 02923004846e4e73862c20962f823a6d05100e7a
seed = taskcompendium-tasktrove-clean-2026.09.10.8-v1
```

1. Read the clean Parquet's `source` and stable `path` columns. Resolve
   `eligible_sources = sorted(unique(source))`. If the manifest has source
   counts, require every eligible source to have a positive count and compare
   the sum with the Parquet row count.
2. For each eligible source `s`, compute
   `selector_sha256 = SHA256(seed + NUL + s + NUL + path)` for every row in that
   source. Sort by `(selector_sha256, path)` and select ranks 1 and 2.
3. Emit two records per eligible source with
   `source`, `path`, `rank`, `selector_sha256`, `release`, and `raw_revision`.
   Require two distinct nonempty paths for every source; a source with fewer
   than two rows is an explicit coverage exception, never a fabricated
   duplicate.
4. Before importing a selected archive, verify its `source` and `path` are
   unchanged, retain the archive checksum, and record importer status. Keep
   unsupported importer outcomes in the manifest rather than silently
   replacing them with another source.

The target row count is `2 * len(eligible_sources)`, which is intentionally
unknown until `.8` membership is read. If all 43 producer-keep candidates are
confirmed by the release manifest, the target would be 86 rows; that is a
conditional calculation, not a current release count.

## Limitations

- Raw directory presence does not prove `.8` clean membership.
- Producer `keep` does not prove a row survived conversion, deduplication, and
  verifier checks.
- The old `.7` report and the current raw HF Viewer aggregate are not valid
  substitutes for the `.8` release manifest.
- Current importer support is based on archive family/converter contracts; it
  does not claim that every row within a partially supported source will pass
  schema or verifier validation.
- No task archive was rewritten, no code was changed, and no live model,
  grader, or external environment was executed for this inventory.
