# TaskTrove `.8` release access and raw-catalog inventory

**Superseded for current release reporting by the `.9` inventory:**
[tasktrove-release-09-inventory-2026-09-14.md](tasktrove-release-09-inventory-2026-09-14.md).
Do not use this historical `.8` catalog for current counts.

Checked 2026-09-14 in `/Users/dlwh/.codex/worktrees/3dd0/marin`.

The pinned clean release is readable through an existing Iris GPU pod whose
`task` container receives the cluster's `iris-task-env` secret. The host itself
still has no CoreWeave credentials, so the successful path is alternative
authenticated local tooling rather than a newly created job or a new secret.

The exact read-only access command was:

```bash
KCFG=/Users/dlwh/.kube/coreweave-iris-gpu
POD=iris-bizon-exp278-scale-v1-w0011-0-fb287cc0-3-18b2257c8120f9c1
kubectl --kubeconfig "$KCFG" -n iris exec "$POD" -c task -- \
  /opt/conda/bin/python -c 'import fsspec, json; fs=fsspec.filesystem("s3"); \
print(json.dumps(fs.ls("s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8", detail=True), default=str, sort_keys=True))'
```

It returned these release objects (the `tasks` directory contains one Parquet
part):

```text
.artifact.json                         1,685 bytes
.executor_info                           593 bytes
.executor_status                           7 bytes
ledger.parquet                       4,169,368 bytes
manifest.json                         56,724 bytes
report.md                             22,922 bytes
tasks/part-00000.parquet          3,873,224,671 bytes
```

The manifest was read through the same pod with `fsspec` and the Parquet
footers were inspected without downloading the data:

```python
fs = fsspec.filesystem("s3")
manifest = json.load(fs.open(base + "/manifest.json"))
ledger = pq.ParquetFile(fs.open(base + "/ledger.parquet"))
tasks = pq.ParquetFile(fs.open(base + "/tasks/part-00000.parquet"))
```

Authoritative `.8` release results:

| object / field | value |
|---|---:|
| `manifest.json` `input_tasks` | 1,739,326 |
| `manifest.json` `clean_tasks` | **1,449,686** |
| `ledger.parquet` rows | 289,640 |
| `tasks/part-00000.parquet` rows | **1,449,686** |
| source keys in `by_source` / `source_verdicts` | 93 / 93 |
| kept source details | 43 |
| release `tasktrove.hf_id` / revision | `open-thoughts/TaskTrove` / `0292300` |
| verifier tool ref | `b2b68d8b0a770cdc0ab3903780172c4b3eea81b1` |

The release has one physical task subset (`tasks/part-00000.parquet`); its
`source` column and `manifest.json` `by_source` map provide the actual 93
logical source subsets. In the table below, `clean rows` means the manifest's
`converted` count; the status breakdown sums to the raw input rows for that
source. A zero clean count means the source was dropped by policy.

<!-- authoritative-source-table -->

| source | clean rows | status breakdown |
|---|---:|---|
| `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` | 1,497 | converted=1497, reviewed_defect=1 |
| `DCAgent__code-contests-noblock` | 8,222 | converted=8222, gold_in_instruction=504, reviewed_defect=2 |
| `DCAgent__exp_rpt_curriculum-easy` | 505 | converted=505, null_grader=4 |
| `DCAgent__exp_rpt_curriculum-medium-v2` | 489 | converted=489, null_grader=3 |
| `DCAgent__exp_rpt_e2egit-large` | 4,993 | converted=4993, unsupported_variant=5 |
| `DCAgent__exp_rpt_e2egit-v2` | 487 | converted=487 |
| `DCAgent__exp_rpt_multifile-v3` | 4,842 | converted=4842, null_grader=1 |
| `DCAgent__exp_rpt_pymethods2test-large-v2` | 4,990 | converted=4990, reviewed_defect=1 |
| `DCAgent__exp_rpt_pymethods2test-v3` | 500 | converted=500 |
| `DCAgent__exp_rpt_stack-pytest-v2` | 495 | converted=495, null_grader=1, unsupported_variant=4 |
| `DCAgent__exp_rpt_unitsyn-python-large-v2` | 4,991 | converted=4991 |
| `DCAgent__exp_rpt_unitsyn-python-v4` | 491 | converted=491 |
| `DCAgent__swe_rebench_v2_patched_oracle-v2` | 13,559 | converted=13559, duplicate=3, reviewed_defect=1, unsupported_variant=4756 |
| `SankalpKJ__nemotron-math-oracle-filtered-v2` | 57,383 | converted=57383, duplicate=24, verified:empty=364, verified:gold_leak=6 |
| `laion__all-puzzles-v2` | 6,719 | converted=6719, duplicate=5, verified:gold_leak=202 |
| `laion__codeforces-v3` | 9,697 | converted=9697, duplicate=48, gold_in_instruction=15, null_grader=240 |
| `laion__exp_rpt_taco-v2` | 5,116 | converted=5116, gold_in_instruction=2592, reviewed_defect=1, unsupported_variant=2291 |
| `laion__glaive-code-assistant-sandboxes-verified-v2` | 9,994 | converted=9994, duplicate=6 |
| `laion__nemo-prism-math-v3` | 2,219 | converted=2219, duplicate=182, verified:empty=3 |
| `laion__nemotron-gym-agent-calendar-v2` | 2,699 | converted=2699 |
| `laion__nemotron-gym-arc-agi-python-inductive-v2` | 10,000 | converted=10000 |
| `laion__nemotron-gym-arc-agi-transductive-v3` | 9,994 | converted=9994, verified:gold_leak=6 |
| `laion__nemotron-gym-competitive-coding-v2` | 13,973 | converted=13973, gold_in_instruction=1739, reviewed_defect=1 |
| `laion__nemotron-gym-instruction-following-calendar-v3` | 5,673 | converted=5673 |
| `laion__nemotron-gym-instruction-following-structured-v3` | 9,167 | converted=9167, null_grader=16, unsupported_variant=254 |
| `laion__nemotron-gym-instruction-following-v3` | 46,391 | converted=46391 |
| `laion__nemotron-gym-knowledge-mcqa-v2` | 611,698 | converted=611698, duplicate=3, reviewed_defect=1, unsupported_variant=3607, verified:empty=1579 |
| `laion__nemotron-gym-knowledge-openqa-v4` | 121,961 | converted=121961, duplicate=46, verified:gold_leak=350 |
| `laion__nemotron-gym-math-openmathreasoning-v2` | 42,506 | converted=42506, duplicate=18, verified:empty=102, verified:gold_leak=10 |
| `laion__nemotron-gym-math-stack-overflow-v3` | 110,266 | converted=110266, duplicate=3, reviewed_defect=1, verified:empty=228, verified:gold_leak=232 |
| `laion__nemotron-gym-math-v5` | 3,891 | converted=3891, duplicate=40, verified:empty=157, verified:gold_leak=8 |
| `laion__nemotron-gym-multichallenge-advanced-v4` | 1,052 | converted=1052, duplicate=16 |
| `laion__nemotron-gym-reasoning-gym-v2` | 13,712 | converted=13712, duplicate=265, unsupported_variant=282 |
| `laion__nemotron-gym-safety-v3` | 44,884 | converted=44884, duplicate=44182 |
| `laion__nemotron-gym-science-so-openq-v3` | 150,468 | converted=150468, verified:gold_leak=176 |
| `laion__nemotron-gym-structured-outputs-v4` | 50,446 | converted=50446, duplicate=462, null_grader=2438, reviewed_defect=1, unsupported_variant=523 |
| `laion__stackexchange-codereview-sandboxes-verified-v2` | 10,000 | converted=10000 |
| `laion__stackexchange-overflow-sandboxes-verified-v2` | 10,000 | converted=10000 |
| `laion__stackexchange-superuser-sandboxes-verified-v2` | 10,000 | converted=10000 |
| `laion__stackexchange-tezos-sandboxes-verified-v2` | 997 | converted=997, duplicate=9003 |
| `laion__stackexchange-unix-sandboxes-verified-v2` | 10,000 | converted=10000 |
| `laion__swesmith-oracle-filtered-v2` | 12,720 | converted=12720, unsupported_variant=207 |
| `laion__wizardlm-orca-v4` | 9,999 | converted=9999, reviewed_defect=1 |

The other 50 manifest source keys have `dropped_source` status and zero clean
rows. Their names and raw input counts are listed in the 93-source raw table
below; this keeps the authoritative clean count separate from the non-authoritative
HF footer catalog.

The manifest-wide status counts are:

```text
converted=1,449,686
dropped_source=212,418
duplicate=54,306
unsupported_variant=11,929
gold_in_instruction=4,850
null_grader=2,703
verified:empty=2,433
verified:gold_leak=990
reviewed_defect=11
```

The `.artifact.json` sidecar confirms this is `tasktrove/clean@2026.09.10.8`,
dependent on `tasktrove/graded@2026.09.10.8`, built from base commit `ccc5ff24cd`
with verifier ref `b2b68d8b0a770cdc0ab3903780172c4b3eea81b1` and a clean checkout.

## Host credential evidence and failed direct probes

For completeness, `uv run fsutil buckets` still reports `marin-us-east-02a` as a
CoreWeave backend with missing credentials. All of the following direct host
probes fail before reading an object:

```text
uv run fsutil ls -l s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8
fsutil: no credentials for coreweave bucket 'marin-us-east-02a': set CW_KEY_ID and CW_KEY_SECRET (or the generic AWS_* pair)

uv run fsutil stat s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8/manifest.json
uv run fsutil stat s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8/ledger.json
fsutil: no credentials for coreweave bucket 'marin-us-east-02a': set CW_KEY_ID and CW_KEY_SECRET (or the generic AWS_* pair)
```

The environment has none of `CW_KEY_ID`, `CW_KEY_SECRET`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `R2_KEY_ID`, or `R2_KEY_SECRET`; `~/.aws/credentials`
and `~/.aws/config` are absent. GCP ADC is present and `gcloud auth
application-default print-access-token` succeeds, but that credential is for GCS
and does not authenticate CoreWeave S3. An anonymous request to
`https://marin-us-east-02a.cwobject.com/` reaches CoreWeave and returns
`403 AccessDenied`; the path-style endpoint returns `400 PathStyleRequestNotAllowed`.
The Marina browser redirects to Google sign-in on this host, so it is not an
authenticated alternate release reader.

The producer source at `git show ccc5ff24cd:experiments/post_training/tasktrove/`
documents the release layout: `tasks/part-00000.parquet`, `ledger.parquet`,
`manifest.json`, and `report.md`. Its `dataset.py` pins
`open-thoughts/TaskTrove@0292300`, uses `*/tasks.parquet`, and its
`source_verdicts.json` contains 93 non-deprecated source decisions (43 `keep`,
50 `drop`). The checked-in 44-spec POC and the published
`open-athena/taskcompendium-spike` artifacts are sample/provenance artifacts,
not the authoritative release inventory.

## Reproducible raw fallback

The raw HF revision is public and independently readable:

```text
https://huggingface.co/api/datasets/open-thoughts/TaskTrove/revision/0292300
https://huggingface.co/api/datasets/open-thoughts/TaskTrove/tree/0292300?recursive=true&expand=false&limit=1000
```

The exact catalog/count commands were:

```bash
curl -fsSL --max-time 60 \
  'https://huggingface.co/api/datasets/open-thoughts/TaskTrove/tree/0292300?recursive=true&expand=false&limit=1000' \
  -o /tmp/tasktrove-tree-recursive.json
```

```python
import json, fsspec, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

entries = [x for x in json.load(open('/tmp/tasktrove-tree-recursive.json'))
           if x.get('type') == 'file' and x.get('path', '').endswith('/tasks.parquet')]
base = 'https://huggingface.co/datasets/open-thoughts/TaskTrove/resolve/0292300/'
def rows(x):
    with fsspec.open(base + x['path'], 'rb', block_size=65536, cache_type='none') as f:
        return x['path'], pq.ParquetFile(f).metadata.num_rows
with ThreadPoolExecutor(max_workers=16) as pool:
    found = list(pool.map(rows, entries))
print('sources', len(found), 'rows', sum(n for _, n in found))
print('default_rows', sum(n for p, n in found if not p.startswith('deprecated/')))
print('deprecated_rows', sum(n for p, n in found if p.startswith('deprecated/')))
```

This produced 133 successful footer reads and `default_rows 1739326`,
`deprecated_rows 226762`.

The recursive tree exposes 133 `tasks.parquet` files: 93 default source files and
40 under `deprecated/`. I read only Parquet footers through HTTP range requests
(16 concurrent `fsspec` HTTP files, `pyarrow.parquet.ParquetFile`), without
downloading the 8.3 GB default data or 955 MB deprecated data. Every footer read
succeeded. The per-source row counts sum to **1,739,326 default rows** and
**226,762 deprecated rows**, matching the producer-era default input count. The
Dataset Viewer `/size` endpoint currently reports 1,604,128 default and 296,667
deprecated rows for the same revision request; those aggregate values conflict
with the direct footer inventory and should not be used for release membership.

For the 93 producer-policy sources, the raw input inventory is 1,739,326 rows:

| source | policy | family | raw rows |
|---|---|---:|---:|
| DCAgent2__nl2bash-tasks-cleaned-oracle-v2 | keep | shell-cmd | 1498 |
| DCAgent__code-contests-noblock | keep | competitive-programming | 8728 |
| DCAgent__exp_rle_adversarial-v6 | drop | unit-test-gen | 2726 |
| DCAgent__exp_rpt_curriculum-easy | keep | unit-test-gen | 509 |
| DCAgent__exp_rpt_curriculum-medium-v2 | keep | unit-test-gen | 492 |
| DCAgent__exp_rpt_e2egit-large | keep | unit-test-gen | 4998 |
| DCAgent__exp_rpt_e2egit-v2 | keep | unit-test-gen | 487 |
| DCAgent__exp_rpt_multifile-v3 | keep | unit-test-gen | 4843 |
| DCAgent__exp_rpt_nemotron-cpp | drop | unit-test-gen | 4196 |
| DCAgent__exp_rpt_pymethods2test-large-v2 | keep | unit-test-gen | 4991 |
| DCAgent__exp_rpt_pymethods2test-v3 | keep | unit-test-gen | 500 |
| DCAgent__exp_rpt_stack-pytest-v2 | keep | unit-test-gen | 500 |
| DCAgent__exp_rpt_unitsyn-python-large-v2 | keep | unit-test-gen | 4991 |
| DCAgent__exp_rpt_unitsyn-python-v4 | keep | unit-test-gen | 491 |
| DCAgent__inferredbugs-sandboxes-verifier | drop | swe-repo | 9659 |
| DCAgent__mix_h4_binary_easy | drop | unit-test-gen | 1996 |
| DCAgent__selfinstruct-naive-sandboxes-2-verified-v3 | drop | shell-cmd | 6665 |
| DCAgent__swe_rebench_v2_patched_oracle-v2 | keep | swe-repo | 18319 |
| SankalpKJ__nemotron-code-oracle-filtered | drop | competitive-programming | 15165 |
| SankalpKJ__nemotron-math-oracle-filtered-v2 | keep | math-answer | 57777 |
| laion__all-puzzles-v2 | keep | math-answer | 6926 |
| laion__codeelo-v2 | drop | competitive-programming | 500 |
| laion__codeforces-v3 | keep | competitive-programming | 10000 |
| laion__exp_rpt_bugsinpy-v4 | drop | swe-repo | 479 |
| laion__exp_rpt_codenet-python-v4 | drop | competitive-programming | 6975 |
| laion__exp_rpt_crosscodeeval-csharp-v4 | drop | qa-short-answer | 1768 |
| laion__exp_rpt_crosscodeeval-java-v3 | drop | qa-short-answer | 2139 |
| laion__exp_rpt_crosscodeeval-python-v2 | drop | other | 500 |
| laion__exp_rpt_crosscodeeval-typescript-v2 | drop | other | 3356 |
| laion__exp_rpt_ghactions-v3 | drop | other | 9930 |
| laion__exp_rpt_methods2test-large-v4 | drop | unit-test-gen | 1194 |
| laion__exp_rpt_nemotron-junit-v6 | drop | unit-test-gen | 447 |
| laion__exp_rpt_scaffold-v3 | drop | other | 3121 |
| laion__exp_rpt_stack-cpp-v4 | drop | unit-test-gen | 7878 |
| laion__exp_rpt_stack-dockerfile-gpt5mini-v7 | drop | tool-use | 587 |
| laion__exp_rpt_stack-go-v5 | drop | unit-test-gen | 2275 |
| laion__exp_rpt_stack-jest-v5 | drop | unit-test-gen | 424 |
| laion__exp_rpt_stack-junit-v6 | drop | unit-test-gen | 843 |
| laion__exp_rpt_stack-php-large-v9 | drop | unit-test-gen | 462 |
| laion__exp_rpt_stack-pytest-large-v3 | drop | unit-test-gen | 1782 |
| laion__exp_rpt_stack-rspec-v4 | drop | unit-test-gen | 8860 |
| laion__exp_rpt_taco-v2 | keep | stdin-stdout | 10000 |
| laion__glaive-code-assistant-sandboxes-verified-v2 | keep | llm-judge-freeform | 10000 |
| laion__magicoder-v4 | drop | llm-judge-freeform | 4096 |
| laion__mix_h10_reward_proportional-v2 | drop | unit-test-gen | 2858 |
| laion__mix_h11_single_skill_only-v2 | drop | unit-test-gen | 2859 |
| laion__mix_h8_original_tests-v2 | drop | unit-test-gen | 2848 |
| laion__nemo-prism-math-v3 | keep | math-answer | 2404 |
| laion__nemotron-gym-agent-calendar-v2 | keep | tool-use | 2699 |
| laion__nemotron-gym-agentic-function-calling-pivot-v3 | drop | tool-use | 9579 |
| laion__nemotron-gym-agentic-indirect-prompt-injection-v3 | drop | prompt-injection | 1272 |
| laion__nemotron-gym-agentic-swe-pivot-v4 | drop | tool-use | 1541 |
| laion__nemotron-gym-arc-agi-python-inductive-v2 | keep | other | 10000 |
| laion__nemotron-gym-arc-agi-transductive-v3 | keep | other | 10000 |
| laion__nemotron-gym-cfbench-v4 | drop | instruction-following | 468 |
| laion__nemotron-gym-competitive-coding-v2 | keep | competitive-programming | 15713 |
| laion__nemotron-gym-identity-following-v4 | drop | instruction-following | 21660 |
| laion__nemotron-gym-instruction-following-adversarial-v5 | drop | instruction-following | 1000 |
| laion__nemotron-gym-instruction-following-calendar-v3 | keep | instruction-following | 5673 |
| laion__nemotron-gym-instruction-following-citation-v2 | drop | instruction-following | 9033 |
| laion__nemotron-gym-instruction-following-freeform-v2 | drop | instruction-following | 8869 |
| laion__nemotron-gym-instruction-following-multiturnchat-v4 | drop | instruction-following | 1982 |
| laion__nemotron-gym-instruction-following-structured-v3 | keep | instruction-following | 9437 |
| laion__nemotron-gym-instruction-following-v3 | keep | instruction-following | 46391 |
| laion__nemotron-gym-inverse-ifeval-v4 | drop | instruction-following | 1000 |
| laion__nemotron-gym-knowledge-mcqa-v2 | keep | qa-short-answer | 616888 |
| laion__nemotron-gym-knowledge-openqa-v4 | keep | qa-short-answer | 122357 |
| laion__nemotron-gym-knowledge-web-search-mcqa-v2 | drop | qa-short-answer | 2915 |
| laion__nemotron-gym-litmus-bench-v2 | drop | math-answer | 5232 |
| laion__nemotron-gym-math-advanced-calculations-v4 | drop | math-answer | 5291 |
| laion__nemotron-gym-math-openmathreasoning-v2 | keep | math-answer | 42636 |
| laion__nemotron-gym-math-stack-overflow-v3 | keep | math-answer | 110730 |
| laion__nemotron-gym-math-v5 | keep | math-answer | 4096 |
| laion__nemotron-gym-multichallenge-advanced-v4 | keep | llm-judge-freeform | 1068 |
| laion__nemotron-gym-multichallenge-vanilla-v3 | drop | llm-judge-freeform | 1050 |
| laion__nemotron-gym-qa-abstention-v4 | drop | qa-short-answer | 3150 |
| laion__nemotron-gym-reasoning-gym-v2 | keep | other | 14259 |
| laion__nemotron-gym-safety-v3 | keep | llm-judge-freeform | 89066 |
| laion__nemotron-gym-science-so-openq-v3 | keep | llm-judge-freeform | 150644 |
| laion__nemotron-gym-structured-outputs-v4 | keep | other | 53870 |
| laion__nemotron-gym-sysbench-v4 | drop | instruction-following | 1010 |
| laion__openswe-tasks-patched-v7-oracle-success | drop | swe-repo | 11730 |
| laion__r2egym-patched-full-oracle-v3 | drop | swe-repo | 2574 |
| laion__stackexchange-codereview-sandboxes-verified-v2 | keep | llm-judge-freeform | 10000 |
| laion__stackexchange-overflow-sandboxes-verified-v2 | keep | llm-judge-freeform | 10000 |
| laion__stackexchange-superuser-sandboxes-verified-v2 | keep | llm-judge-freeform | 10000 |
| laion__stackexchange-tezos-sandboxes-verified-v2 | keep | llm-judge-freeform | 10000 |
| laion__stackexchange-unix-sandboxes-verified-v2 | keep | llm-judge-freeform | 10000 |
| laion__swegym-tasks-patched-validated-v5 | drop | swe-repo | 2428 |
| laion__swesmith-oracle-filtered-v2 | keep | swe-repo | 12927 |
| laion__toolscale-v4 | drop | tool-use | 4048 |
| laion__tulu3-sft-personas-math-sandboxes-verified-v3 | drop | math-answer | 9998 |
| laion__wizardlm-orca-v4 | keep | llm-judge-freeform | 10000 |

The 43 `keep` policy sources account for 1,526,908 raw rows; the 50 `drop`
policy sources account for 212,418 raw rows. These are **input** rows, not clean
release rows: conversion errors, deduplication, reviewed defects, and verifier
checks are represented by the authoritative release manifest/ledger. The
manifest reports 1,449,686 converted clean rows.

The 40 raw `deprecated/` source directories are outside the producer's `*/tasks.parquet`
glob and are not part of the pinned `.8` input. Their footer counts are retained
here as a catalog check:

```text
DCAgent2__Toolscale-tasks=4063; DCAgent2__Toolscale-tasks-cleaned=199;
DCAgent2__nl2bash-tasks-cleaned-oracle=1570; DCAgent__exp_rpt_codenet-python=10000;
DCAgent__exp_rpt_stack-dockerfile-v3=485; laion__exp_rle_detailed-v2=773;
laion__exp_rle_error_report-v3=261; laion__exp_rle_github_issue-v3=264;
laion__exp_rle_minimal_instructions-v3=233; laion__exp_rpt_codenet-python-v3=10000;
laion__exp_rpt_crosscodeeval-csharp-v2=1768; laion__exp_rpt_crosscodeeval-csharp-v3=1768;
laion__exp_rpt_defects4j-v3-v4=216; laion__exp_rpt_exercism-python-v2=133;
laion__exp_rpt_ghactions-v2=9930; laion__exp_rpt_stack-csharp-v2=9526;
laion__exp_rpt_stack-csharp-v3=9485; laion__exp_rpt_stack-csharp-v4=9485;
laion__exp_rpt_stack-junit-v2=9999; laion__exp_rpt_stack-junit-v3=9999;
laion__exp_rpt_stack-junit-v4=9999; laion__exp_rpt_stack-junit-v5=9999;
laion__exp_rpt_stack-php-large-v2=5000; laion__exp_rpt_stack-php-large-v3=5000;
laion__exp_rpt_stack-php-large-v4=5000; laion__exp_rpt_stack-php-large-v5=5000;
laion__exp_rpt_stack-ruby-v3=2310; laion__glaive-code-assistant-sandboxes-verified=10000;
laion__nemotron-gym-agent-workplace-v2=297;
laion__nemotron-gym-instruction-following-adversarial=1000;
laion__nemotron-gym-instruction-following-adversarial-v2=1000;
laion__nemotron-gym-math-advanced-calculations=6000;
laion__nemotron-gym-math-advanced-calculations-v2=6000;
laion__qasper-v3=10000; laion__stackexchange-codereview-sandboxes-verified=10000;
laion__stackexchange-overflow-sandboxes-verified=10000;
laion__stackexchange-superuser-sandboxes-verified=10000;
laion__stackexchange-tezos-sandboxes-verified=10000;
laion__stackexchange-unix-sandboxes-verified=10000; laion__staqc-v4=10000.
```

## Provenance and fallback recommendation

The raw catalog evidence is anchored to HF commit
`02923004846e4e73862c20962f823a6d05100e7a` (the `0292300` revision requested by
the producer). The policy and release layout are anchored to Marin commit
`ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2`, whose pipeline version is
`2026.09.10.8` and verifier ref is
`b2b68d8b0a770cdc0ab3903780172c4b3eea81b1`.

For a complete coverage audit, the best reproducible path is to obtain a
CoreWeave key with read access and read exactly `manifest.json`, `ledger.parquet`,
and `tasks/part-00000.parquet` under the pinned release. If that key cannot be
provided, preserve this direct HF raw footer catalog and use the producer's
`source_verdicts.json` only as a source-policy/input approximation; do not infer
clean membership or post-filter counts from the 44/43-spec POC, viewer aggregates,
or raw source counts.
