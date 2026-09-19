# TaskTrove Clean `.9` independent representation review (group A)

Reviewed 2026-09-14 against the authoritative release:

- HF mirror: `open-athena/task-trove`, Parquet `default/train`
- Canonical S3 root: `s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9`
- Manifest: `/tmp/tasktrove-clean-09/manifest.json`
- Deterministic samples: `/tmp/tasktrove-clean-09/sample-metadata.json`

I resolved and fetched exactly the two metadata-selected task archives for each
of the 22 requested sources (44 archives). The Parquet metadata was queried by
exact `(source, path)` and the task BLOBs were fetched from the HF Dataset
Viewer rows API at
`https://datasets-server.huggingface.co/rows?dataset=open-athena/task-trove&config=default&split=train&offset={row_idx}&length=1`.
Each BLOB was read as a tar stream. I inspected `instruction.md`,
`task.toml`, `tests/verifier.toml`, the Dockerfile, schemas, checker scripts,
expected files, and test resources. I did not run source scripts, checkers, or
submitted programs, and I did not use old `.8` fixtures or the old raw HF
catalog.

The current checkout still pins `RELEASE = "2026.09.10.8"` and verifier
revision `b2b68d8b...`; every sampled Clean `.9` Dockerfile pins verifier
revision `b76d03131c` (full commit
`b76d03131cd88bd9fc711dba206659027edba3a8`). Thus a direct current importer
run rejects `.9` archives at the revision guard. The judgments below separate
that release integration blocker from the source representation. The `.9`
release and verifier pins must be updated before importer results are claimed
for this release.

## Disposition by sampled task

`keep` means the sampled source contract is preserved by an existing importer
once the release pin is corrected. `repair` means the source is coherent but
the current importer, prompt cleaning, or lowering needs a source-backed
change. `reject` means the sampled source checker or contract is defective.
`unsupported` means the samples are coherent but there is no current importer
or supported output protocol.

| source | exact sampled task paths | disposition | representation findings |
| --- | --- | --- | --- |
| `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` | `task_1139`; `task_6529` | `reject` | Both public goals require sorted/reverse-sorted output. `tests/nl2bash_check.py` scores `Counter(_records(...))`, ignoring order and accepting extra non-error records. Expected ordered outputs are `4 apple / 2 banana / 1 orange` and `honeydew:5 / fig:4 / egg:3 / carrot:2 / apple:1`. The public “verifier inspects them” line is also delivery leakage. Confirmed checker defect; both paths are in the problematic ledger. |
| `DCAgent__code-contests-noblock` | `code_contests-8689`; `code_contests-10155` | `repair` | Complete Codeforces-style Python exact-stdio tasks with ten cases and `/app/solution.py`; no special judge. The current coding importer only supports `codeforces`/`python_unit_tests` and requires a C++17 build, so it cannot preserve this Python batch lowering. |
| `DCAgent__exp_rpt_curriculum-easy` | `curriculum-easy-0005`; `curriculum-easy-0103` | `repair` | Pytest resources and core behavior are present, but prompts expose test scaffolding (“supporting the tests”, “For tests, return 0”) and generated test alignment. Prompt cleaning and release pin integration are needed. |
| `DCAgent__exp_rpt_curriculum-medium-v2` | `curriculum-medium-0289`; `curriculum-medium-0098` | `repair` | Complete pytest tasks with required resources. Instructions expose test paths, patching, and generated verifier-file details. Remove that evaluation machinery while retaining the semantic API contract. |
| `DCAgent__exp_rpt_e2egit-large` | `e2egit-3191`; `e2egit-1722` | `repair` | Small Python API tasks have `/tests/test_solution.py`, but the current importer hard-codes `/tests/test_curriculum.py`, causing rejection. Prompts also mention provided tests; scrub those references. |
| `DCAgent__exp_rpt_e2egit-v2` | `e2egit-0014`; `e2egit-0137` | `repair` | Same absolute-test-path importer gap and test-specific prompt leakage as the large variant. Required module/API facts are present. |
| `DCAgent__exp_rpt_multifile-v3` | `multifile-3520`; `multifile-2374` | `repair` | Multi-module behavior and files are present, with tests at `/tests/test_multifile.py`; current importer requires `/tests/test_curriculum.py`. Prompts expose `/setup_files/test_multifile.py` as a generated verifier test. |
| `DCAgent__exp_rpt_pymethods2test-large-v2` | `pymethods2test-2173`; `pymethods2test-0362` | `repair` | Python method tasks have `/tests/test_solution.py`; the current importer rejects the path and some instructions reveal tests/expected outputs. |
| `DCAgent__exp_rpt_pymethods2test-v3` | `pymethods2test-0325`; `pymethods2test-0011` | `repair` | Same test-path gap. Public prompts contain test-oriented expected-output guidance; semantic method contracts are otherwise available. |
| `DCAgent__exp_rpt_stack-pytest-v2` | `stack-pytest-0454`; `stack-pytest-0496` | `repair` | Pytest tasks use `/tests/test_solution.py`, not the current importer’s fixed curriculum path. Prompt text directly describes test cases and expected import behavior. |
| `DCAgent__exp_rpt_unitsyn-python-large-v2` | `unitsyn-python-2001`; `unitsyn-python-1010` | `repair` | Required Python files and tests are present, but the importer’s absolute test-path requirement rejects them. Some prompt templates expose test expected outputs. |
| `DCAgent__exp_rpt_unitsyn-python-v4` | `unitsyn-python-0358`; `unitsyn-python-0401` | `repair` | Same `/tests/test_solution.py` importer gap and test-template leakage; no missing source resource was found. |
| `DCAgent__swe_rebench_v2_patched_oracle-v2` | `agnostiqhq__covalent-1599`; `kinto__kinto-http.py-384` | `unsupported` | Tasks require cloning repositories into `/testbed`, installing requirements, applying patches, and running private must-pass/must-not-break paths. No current TaskCompendium SWE lowering preserves this isolated repo protocol. |
| `SankalpKJ__nemotron-math-oracle-filtered-v2` | `nemotron-121758`; `nemotron-025508` | `repair` | Self-contained scalar math tasks with exact private answers (`99/2`, `6`) and `/app/answer.txt`. Current math importer only accepts `all_puzzles`, so a source-backed `nemotron_math` path is needed. |
| `laion__all-puzzles-v2` | `all_puzzles-0880`; `all_puzzles-35346` | `repair` | Exact answer tasks are coherent, but the public wrapper mentions verifier/gold. Row `0880` was falsely rejected by a generic “ascending or descending” regex despite ascending expected output; fixed with a direction-specific match and regression test. Row `35346` is a choice answer that the math importer rejects, though the exact-answer importer can preserve it. |
| `laion__codeforces-v3` | `codeforces-08311`; `codeforces-07764` | `repair` / `unsupported` | `08311` is a complete exact-stdio task whose prompt exposes verifier/compile/special-judge delivery text and needs cleaning. `07764` uses a private special judge and constructive protocol; the current importer explicitly rejects special judges, requiring a dedicated lowering. |
| `laion__exp_rpt_taco-v2` | `taco-9046`; `taco-3045` | `repair` | Complete Python batch tasks with many exact token-comparison cases and `/app/solution.py`. Current importer does not support converter `taco` (and assumes exact/C++17), while the prompt exposes that the verifier can run the answer. |
| `laion__glaive-code-assistant-sandboxes-verified-v2` | `glaive-code-assistant-1622`; `glaive-code-assistant-3166` | `repair` | Complete JavaScript/Java coding questions with private checklist criteria and no missing context. Current judge importer supports only `qa-short-answer`/reference judges; these rubric checklist tasks need a generic rubric lowering. Prompt says the grader reads `response.txt`, which must be removed. |
| `laion__nemo-prism-math-v3` | `nemo-math-v2-14721`; `nemo-math-v2-12431` | `repair` | Self-contained scalar math with exact private answers (`6434`, `126`) and boxed-answer contracts. They use unsupported `nemotron_math`; add a source-backed importer path. |
| `laion__nemotron-gym-agent-calendar-v2` | `agent-calendar-7e7daa6b642b.tar.gz`; `agent-calendar-ef447f15b98b.tar.gz` | `repair` | Custom checker validates JSON events, exact IDs/names/durations, windows, before/after/at/between constraints, uniqueness, missing/extra events, and overlap. Public text exposes verifier details. The static final-state contract is representable, but no current `agent_calendar` importer exists. |
| `laion__nemotron-gym-arc-agi-python-inductive-v2` | `arc-induct-a02d14f53427.tar.gz`; `arc-induct-9f0822b88753.tar.gz` | `repair` | Private `cases.json` and `run_transform.py` execute candidate Python and compare exact grids. Public text exposes verifier/gold/held-out-test machinery. A dedicated code-answer lowering is needed to preserve this private runtime. |
| `laion__nemotron-gym-arc-agi-transductive-v3` | `arc-trans-62e4d5061bf3.tar.gz`; `arc-trans-f53c1ec0e12c.tar.gz` | `repair` | Exact private grid-string verifier and complete puzzle context are present, but public text exposes grader/gold/held-out/output machinery. A generic exact-answer reasoning lowering is needed. |

## Confirmed defect and code status

The two nl2bash paths above are recorded in
`.agents/logbooks/taskcompendium-problematic-tasks.md`. The checker’s
`Counter` comparison is order-insensitive even though the task contract is
ordered; I did not weaken or rewrite the source checker because a repair must
be source-backed and preserve the intended shell output semantics.

I fixed the current importer’s false positive for all-puzzles direction
matching in `lib/taskcompendium/src/taskcompendium/importers/tasktrove_answers.py`:
the rejection now matches only an instruction that actually says “sort these
words in descending order”, rather than any sentence mentioning ascending or
descending order. A regression test was added to
`lib/taskcompendium/tests/test_tasktrove_answers.py` for an ascending exact
source row. Focused validation passed: `10 passed`.

No release-pin migration was made in this review because the checkout’s
`.8` fixtures and current parent work are shared; changing the global pin
would make those fixtures and unrelated work invalid. The release integration
must update the verifier revision and Clean `.9` root together before claiming
that `.9` archives import successfully.

## Access limits

The HF mirror and Dataset Viewer rows API were available and sufficient to
inspect all 44 selected archives. Direct host credentials for the canonical S3
root were unavailable, so I did not independently read S3 objects. The report
therefore records HF-mirror evidence for the exact `.9` samples. No source
checker, submitted program, repository clone, or source-provided script was
executed.
