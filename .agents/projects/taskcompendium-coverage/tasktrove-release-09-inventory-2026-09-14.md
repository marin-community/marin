# TaskTrove clean `.9` release inventory

Checked 2026-09-14 in `/Users/dlwh/.codex/worktrees/3dd0/marin`.

The authoritative current release is the HF mirror `open-athena/task-trove`,
backed by the clean S3 artifact
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9`. The downloaded
manifest at `/tmp/tasktrove-clean-09/manifest.json` reports:

| manifest field | value |
|---|---:|
| `input_tasks` | 1,739,326 |
| `clean_tasks` | **1,449,686** |
| source keys in `by_source` / `source_verdicts` | 93 / 93 |
| kept source details | 43 |
| `tasktrove.hf_id` / revision | `open-thoughts/TaskTrove` / `0292300` |
| `verify_tool_ref` | `b76d03131c` |

The canonical S3 prefix is readable from an existing Iris GPU pod using the
cluster-injected CoreWeave credentials. This was a read-only `kubectl exec`; no
pod was created, restarted, or modified:

```bash
KCFG=/Users/dlwh/.kube/coreweave-iris-gpu
POD=iris-bizon-exp278-scale-v1-w0011-0-fb287cc0-3-18b2257c8120f9c1
kubectl --kubeconfig "$KCFG" -n iris exec "$POD" -c task -- \
  /opt/conda/bin/python -c 'import fsspec, json; fs=fsspec.filesystem("s3"); \
print(json.dumps(fs.ls("s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9", detail=True), default=str, sort_keys=True))'
```

The prefix contains `manifest.json` (56,694 bytes), `ledger.parquet`
(4,170,372 bytes), `report.md` (22,892 bytes), `.artifact.json` (1,662 bytes),
executor metadata, and one physical task directory. The HF mirror's Parquet
file is the readable train export of that release.

The HF Parquet URL is directly readable:

```text
https://huggingface.co/datasets/open-athena/task-trove/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet
```

Exact read-only verification used HTTP range requests for the Parquet footer:

```python
import fsspec
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/open-athena/task-trove/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
with fsspec.open(url, "rb", block_size=65536, cache_type="none") as f:
    parquet = pq.ParquetFile(f)
    print(parquet.metadata.num_rows, parquet.metadata.num_row_groups)
```

It returned `1,449,686` rows, 66 row groups, and columns
`path, source, family, template_id, converter, mode, dockerfile_id, language,
element, has_solution, task_binary, solution_binary`. Reading only the first
row group metadata columns also succeeded:

```python
with fsspec.open(url, "rb", block_size=65536, cache_type="none") as f:
    sample = pq.ParquetFile(f).read_row_group(
        0, columns=["path", "source", "family", "template_id"], use_threads=False
    )
print(sample.slice(0, 1).to_pylist())
```

Observed sample:

```text
{'path': 'codeforces-06236', 'source': 'laion__codeforces-v3', 'family': 'competitive-programming', 'template_id': '9a29baf583bf'}
```

The mirror README says the `train` split is generated from the clean release's
manifest and `ledger.parquet`. The manifest status totals are:

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

There is one physical train Parquet file; the 93 source subsets are logical
partitions represented by the `source` column and manifest `by_source`. The 43
kept sources and their clean row counts are:

| source | clean rows | remaining statuses |
|---|---:|---|
| `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` | 1,497 | reviewed_defect=1 |
| `DCAgent__code-contests-noblock` | 8,222 | gold_in_instruction=504, reviewed_defect=2 |
| `DCAgent__exp_rpt_curriculum-easy` | 505 | null_grader=4 |
| `DCAgent__exp_rpt_curriculum-medium-v2` | 489 | null_grader=3 |
| `DCAgent__exp_rpt_e2egit-large` | 4,993 | unsupported_variant=5 |
| `DCAgent__exp_rpt_e2egit-v2` | 487 | — |
| `DCAgent__exp_rpt_multifile-v3` | 4,842 | null_grader=1 |
| `DCAgent__exp_rpt_pymethods2test-large-v2` | 4,990 | reviewed_defect=1 |
| `DCAgent__exp_rpt_pymethods2test-v3` | 500 | — |
| `DCAgent__exp_rpt_stack-pytest-v2` | 495 | null_grader=1, unsupported_variant=4 |
| `DCAgent__exp_rpt_unitsyn-python-large-v2` | 4,991 | — |
| `DCAgent__exp_rpt_unitsyn-python-v4` | 491 | — |
| `DCAgent__swe_rebench_v2_patched_oracle-v2` | 13,559 | duplicate=3, reviewed_defect=1, unsupported_variant=4,756 |
| `SankalpKJ__nemotron-math-oracle-filtered-v2` | 57,383 | duplicate=24, verified:empty=364, verified:gold_leak=6 |
| `laion__all-puzzles-v2` | 6,719 | duplicate=5, verified:gold_leak=202 |
| `laion__codeforces-v3` | 9,697 | duplicate=48, gold_in_instruction=15, null_grader=240 |
| `laion__exp_rpt_taco-v2` | 5,116 | gold_in_instruction=2,592, reviewed_defect=1, unsupported_variant=2,291 |
| `laion__glaive-code-assistant-sandboxes-verified-v2` | 9,994 | duplicate=6 |
| `laion__nemo-prism-math-v3` | 2,219 | duplicate=182, verified:empty=3 |
| `laion__nemotron-gym-agent-calendar-v2` | 2,699 | — |
| `laion__nemotron-gym-arc-agi-python-inductive-v2` | 10,000 | — |
| `laion__nemotron-gym-arc-agi-transductive-v3` | 9,994 | verified:gold_leak=6 |
| `laion__nemotron-gym-competitive-coding-v2` | 13,973 | gold_in_instruction=1,739, reviewed_defect=1 |
| `laion__nemotron-gym-instruction-following-calendar-v3` | 5,673 | — |
| `laion__nemotron-gym-instruction-following-structured-v3` | 9,167 | null_grader=16, unsupported_variant=254 |
| `laion__nemotron-gym-instruction-following-v3` | 46,391 | — |
| `laion__nemotron-gym-knowledge-mcqa-v2` | 611,698 | duplicate=3, reviewed_defect=1, unsupported_variant=3,607, verified:empty=1,579 |
| `laion__nemotron-gym-knowledge-openqa-v4` | 121,961 | duplicate=46, verified:gold_leak=350 |
| `laion__nemotron-gym-math-openmathreasoning-v2` | 42,506 | duplicate=18, verified:empty=102, verified:gold_leak=10 |
| `laion__nemotron-gym-math-stack-overflow-v3` | 110,266 | duplicate=3, reviewed_defect=1, verified:empty=228, verified:gold_leak=232 |
| `laion__nemotron-gym-math-v5` | 3,891 | duplicate=40, verified:empty=157, verified:gold_leak=8 |
| `laion__nemotron-gym-multichallenge-advanced-v4` | 1,052 | duplicate=16 |
| `laion__nemotron-gym-reasoning-gym-v2` | 13,712 | duplicate=265, unsupported_variant=282 |
| `laion__nemotron-gym-safety-v3` | 44,884 | duplicate=44,182 |
| `laion__nemotron-gym-science-so-openq-v3` | 150,468 | verified:gold_leak=176 |
| `laion__nemotron-gym-structured-outputs-v4` | 50,446 | duplicate=462, null_grader=2,438, reviewed_defect=1, unsupported_variant=523 |
| `laion__stackexchange-codereview-sandboxes-verified-v2` | 10,000 | — |
| `laion__stackexchange-overflow-sandboxes-verified-v2` | 10,000 | — |
| `laion__stackexchange-superuser-sandboxes-verified-v2` | 10,000 | — |
| `laion__stackexchange-tezos-sandboxes-verified-v2` | 997 | duplicate=9,003 |
| `laion__stackexchange-unix-sandboxes-verified-v2` | 10,000 | — |
| `laion__swesmith-oracle-filtered-v2` | 12,720 | unsupported_variant=207 |
| `laion__wizardlm-orca-v4` | 9,999 | reviewed_defect=1 |

The 50 dropped source keys have zero clean rows and collectively account for
212,418 input rows under `dropped_source`. The old `open-thoughts/TaskTrove`
raw catalog and the checked-in 44-spec POC are not used for this `.9` inventory;
they are historical/source context only.

Provenance: `/tmp/tasktrove-clean-09/manifest.json`,
`/tmp/tasktrove-clean-09/ledger.parquet`, `/tmp/tasktrove-clean-09/README.md`,
the direct HF Parquet URL above, and the producer's pinned upstream revision
`open-thoughts/TaskTrove@0292300`. No task semantics or source policy were
modified.
