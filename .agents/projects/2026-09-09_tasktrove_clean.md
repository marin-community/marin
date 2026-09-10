# TaskTrove Clean

Convert `open-thoughts/TaskTrove` (revision 0292300, 1,739,326 tasks, 93 sources) into
"TaskTrove Clean": every kept task graded by `tasktrove-verify` in one declared mode, with its own
edited Dockerfile, deduplicated, statically and in-process verified, and tagged for selection.
Plan artifact: https://claude.ai/code/artifact/adcc779a-78d3-4b19-8daf-11e1ec1e8fbc

Branch `tasktrove-conversion-pipeline`, one PR (#9061 is absorbed into it).

## Pipeline (`experiments/post_training/tasktrove/pipeline.py`)

raw → summaries → templates → converted → graded → clean

- Agents write `source_verdicts.json` (done) and `converters/<name>.py` (one per converter key).
- Everything else is mechanical and reruns from the checked-in files.

## Tasks

1. [x] Spec in `tests/verifier.toml`, flat modes, `--verify-tool-ref` (experiments module)
2. [x] `lib/tasktrove-verify`: 13 modes, CLI, library entry point, fixture tests
3. [x] Converter registry keyed by (family, tests/ code files), coverage.json, tags columns
4. [x] 18 converters (one per key, each with a fixture test and a local Docker sampling report under `converters/reports/`); every kept source is covered, every other source is dropped with a reason
5. [x] Dedup and cap step
6. [x] Verified step: spec, dockerfile, gold leak, empty, expected, perturbed, shape
7. [x] Clean step: tasks/ per source, ledger/, manifest.json, report.md, export CLI, README
8. [x] Full run on Iris `cw-us-east-02a` with the pushed SHA as tool ref; measured counts in the PR body; 100-task random sample inspected
9. [x] Run 2026.09.10.1 (`/power/iris-run-job-20260910-030245`, tool ref 105b4541): its Docker sample exposed doctest-graded swesmith tasks and an uncompilable TACO oracle; both fixed in the converters and rerun as 2026.09.10.2, whose sample was clean; the reward-file fix reran it as 2026.09.10.3 (`/power/iris-run-job-20260910-055608`, tool ref 5062aaa3)
10. [x] Delete superseded bucket artifacts: `tasktrove/*/{2026.09.09,2026.09.10,2026.09.10.1,2026.09.10.2,2026.09.10.3,2026.09.10.4}`, `raw/tasktrove/2026.09.10` (duplicate download; the pipeline pins `raw/tasktrove/2026.09.09`) and the first two smoke attempts' exports and checkpoints are gone; `raw/tasktrove/2026.09.09`, `tasktrove/clean/2026.09.10.5` and the third smoke's export and checkpoint remain
11. [x] Update this logbook's run section and the PR body with the final numbers; verified with `gh pr view --json title,body`
12. [x] Publish an artifact with the detailed analysis: https://claude.ai/code/artifact/22018f48-ca1f-4a55-9f5c-12108557d2ba (funnel, kept and dropped sources with reasons, per-check rejections, normalization, the four Docker samples, the smoke run)
13. [x] Smoke-train Qwen 0.6B on the clean dataset through marin skyrl, configured after the curriculum experiment; config and result in the smoke section below (third attempt succeeded end to end, reward 0.0)
14. [x] Final report to the user and PR monitoring per the commit skill
15. [x] Reinstate the rubric-only judge sources with a `judge_rubric` converter and `rubric`/`no-reference` tags; run 2026.09.10.4 (`/power/iris-run-job-20260910-164051`), then 2026.09.10.5 (`/power/iris-run-job-20260910-180939`, tool ref d28b440f) with the validity-sample fixes
16. [x] Report and manifest: distinct Dockerfiles with base image and per-converter/source counts, tags, dropped sources by reason (`clean summary` regenerates them)
17. [x] End-to-end validity sample (`validity.py` + `validity_daytona.py`): stratified sample, Sonnet solve scripts, empty/oracle/candidate checks in Daytona; pilot of 54 tasks and full sample of 190 in the validity section below
18. [x] Fixes from the validity sample: pytest node ids rebased onto the workspace in the tool (a `tests/pytest.ini` rootdir made every id miss), failing pytest grades keep the output tail, truncated parametrized ids are dropped from PASS_TO_PASS and reject FAIL_TO_PASS (394 swe_rebench tasks); whole-directory uploads and per-task agent timeouts in the validity tooling
19. [x] M1 cleanup: recover non-null TOML/XML/CSV structured-output tasks, add end-to-end converter tests, fix the root `reasoning-gym` dependency, merge current `origin/main`, and run the format-parity audit
20. [x] M2 cleanup: leave non-letter MCQA golds, literal-newline prompt failures, and the broken `arc_agi`/`rearc` scorer rows rejected; their small recovery does not justify format-specific parsing or grading paths
21. [ ] M3 cleanup: audit the 25 dropped test sources and reinstate only tasks supported by existing graders or small direct adapters; tag usable kata tasks instead of dropping them for ease
22. [ ] M4 cleanup: validate kept Python SWE tasks and decide non-Python repository tasks without adding a new grading family or repairing repositories/toolchains
23. [ ] M5 cleanup: sample kept and dropped judge sources against the answerability, rubric, leakage, triviality, and persona checklist
24. [ ] M6 cleanup: measure near duplicates over instructions and hidden grading text before changing the exact within-source key
25. [ ] M7 cleanup: rerun the full pipeline once after the cleanup decisions, regenerate the report and artifact, run the final Docker sample, and update PR #9061

## Cleanup extension

The remaining cleanup lands in PR #9061. A row may be recovered only by a small deterministic
normalization with one interpretation, such as trimming surrounding whitespace, decoding literal
newlines, or matching answer text to exactly one option. Rows stay dropped when recovery needs
heuristic answer extraction, generated tests, repository reconstruction, a new language-specific
grader, dependency archaeology, or a subjective guess.

M1 added TOML parsing to `json-schema` and the structural `xml-elements` and `csv-columns` modes.
The complete structured-output source contains 4,166 TOML, 14,546 XML, and 4,151 CSV rows. The
converter recovers 3,520 TOML, 14,135 XML, and 2,802 CSV rows (20,457 total) and rejects the other
2,406 as null graders: the empty TOML table satisfies the schema, or an arbitrary well-formed XML
or CSV document would pass because the schema names no checkable field.

A seeded audit (`20260910`) sampled 100 converted rows per format from the local source parquet.
For each row, the bundled grader and `tasktrove-verify` graded an empty answer, a constructed
conforming answer, and the same answer with a required field removed. All 900 grader pairs agreed:
empty and perturbed answers scored 0, and conforming answers scored 1. A full-source check also
confirmed that every one of the 14,135 converted XML specs has representable element names. After
merging `origin/main` at `9f3cc8a80d`, the TaskTrove suites reported 411 passed and 1 skipped,
Pyrefly reported zero errors, the diff-scoped repository lint passed, and the safe affected-test
runner reported 1,683 passed, 4 skipped, and 5 expected failures.

M2 makes no converter change. The 3,607 non-letter MCQA golds include numeric and formatted
answer fragments that do not identify one displayed option without stripping units, LaTeX, or
other content-specific syntax. The 1,463 literal-newline prompt failures and 282 reasoning-gym
`arc_agi`/`rearc` rows could be special-cased, but together do not justify extra prompt rewriting
or grid-grading behavior. These rows remain rejected rather than adding recovery heuristics.

## Run 2026.09.10.5

Job `/power/iris-run-job-20260910-180939` on `cw-us-east-02a`, tool ref `d28b440fde1abf0e027e79baad057547f03f6372`,
25 minutes end to end (summaries 4 min, templates 4 min, converted 4 min, graded 4 min, clean 4 min; the raw
download is pinned to `raw/tasktrove/2026.09.09` and was reused). Output at
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.5`: `tasks/part-*.parquet` (1,025 shards, 3.2 GiB),
`ledger.parquet` (one row per rejected task with its reason), `manifest.json`, `report.md`. Run 2026.09.10.4
(`/power/iris-run-job-20260910-164051`, tool ref 5062aaa3) added the `judge_rubric` converter, which reinstates the eight rubric-only judge sources (safety, the five Stack Exchange
sites, glaive, wizardlm) that earlier runs dropped for lacking a reference answer; their tasks carry the tags
`judge`, `rubric`, `no-reference` and a domain tag so they can be selected or deselected as a block, and the
openqa/science/multichallenge judge tasks now carry `reference`. The manifest gained `by_tag` and a `dockerfiles`
table (base image, tasks, converters and sources per distinct Dockerfile) and `report.md` gained the same
sections plus a dropped-sources-by-reason table. This run carries the two fixes from the validity sample drawn on
2026.09.10.4 (pytest node ids rebased onto the workspace in the tool, truncated parametrized test ids dropped at
conversion) and is otherwise identical. The bucket directories of runs 2026.09.09 through 2026.09.10.4
and the duplicate download `raw/tasktrove/2026.09.10` were deleted, as were the first two smoke attempts' exports
and checkpoints; the third smoke's export and checkpoint (built on 2026.09.10.3) remain.

| status | tasks | meaning |
|---|---:|---|
| converted | 1,399,813 | in the clean dataset |
| dropped_source | 233,948 | source rejected in `source_verdicts.json` |
| duplicate | 54,133 | same instruction as an earlier task in the source |
| unsupported_variant | 42,871 | converter has no grading for this task shape (xml/toml/csv schemas, non-Python SWE repositories or test files, doctest-graded SWE tasks, non-letter MCQ gold, TACO oracles that do not compile) |
| gold_in_instruction | 4,850 | every hidden stdio case is a sample printed in the prompt |
| verified:empty | 2,433 | grader cannot score even an empty output (unparseable expected value, options not detected) |
| verified:gold_leak | 990 | expected value is visible in the instruction |
| null_grader | 288 | no stdio cases or an empty schema |

Clean tasks by mode:

| mode | tasks |
|---|---:|
| mcq | 611,699 |
| judge | 379,356 |
| math | 219,316 |
| ifeval | 46,391 |
| json-schema | 39,329 |
| stdio | 37,012 |
| script | 21,142 |
| pytest | 18,193 |
| reasoning-gym | 13,712 |
| exact | 13,663 |

Kept sources (34):

| source | family | input | clean | kept % | largest loss |
|---|---|---:|---:|---:|---|
| laion__nemotron-gym-knowledge-mcqa-v2 | qa-short-answer | 616,888 | 611,699 | 99% | unsupported_variant (3,607) |
| laion__nemotron-gym-science-so-openq-v3 | llm-judge-freeform | 150,644 | 150,468 | 100% | verified:gold_leak (176) |
| laion__nemotron-gym-knowledge-openqa-v4 | qa-short-answer | 122,357 | 121,961 | 100% | verified:gold_leak (350) |
| laion__nemotron-gym-math-stack-overflow-v3 | math-answer | 110,730 | 110,267 | 100% | verified:gold_leak (232) |
| laion__nemotron-gym-safety-v3 | llm-judge-freeform | 89,066 | 44,884 | 50% | duplicate (44,182) |
| SankalpKJ__nemotron-math-oracle-filtered-v2 | math-answer | 57,777 | 57,383 | 99% | verified:empty (364) |
| laion__nemotron-gym-structured-outputs-v4 | other | 53,870 | 30,162 | 56% | unsupported_variant (23,386) |
| laion__nemotron-gym-instruction-following-v3 | instruction-following | 46,391 | 46,391 | 100% |  (0) |
| laion__nemotron-gym-math-openmathreasoning-v2 | math-answer | 42,636 | 42,506 | 100% | verified:empty (102) |
| DCAgent__swe_rebench_v2_patched_oracle-v2 | swe-repo | 18,319 | 5,473 | 30% | unsupported_variant (12,844) |
| laion__nemotron-gym-competitive-coding-v2 | competitive-programming | 15,713 | 13,974 | 89% | gold_in_instruction (1,739) |
| laion__nemotron-gym-reasoning-gym-v2 | other | 14,259 | 13,712 | 96% | unsupported_variant (282) |
| laion__swesmith-oracle-filtered-v2 | swe-repo | 12,927 | 12,720 | 98% | unsupported_variant (207) |
| laion__codeforces-v3 | competitive-programming | 10,000 | 9,697 | 97% | null_grader (240) |
| laion__exp_rpt_taco-v2 | stdin-stdout | 10,000 | 5,117 | 51% | gold_in_instruction (2,592) |
| laion__glaive-code-assistant-sandboxes-verified-v2 | llm-judge-freeform | 10,000 | 9,994 | 100% | duplicate (6) |
| laion__nemotron-gym-arc-agi-python-inductive-v2 | other | 10,000 | 10,000 | 100% |  (0) |
| laion__nemotron-gym-arc-agi-transductive-v3 | other | 10,000 | 9,994 | 100% | verified:gold_leak (6) |
| laion__stackexchange-codereview-sandboxes-verified-v2 | llm-judge-freeform | 10,000 | 10,000 | 100% |  (0) |
| laion__stackexchange-overflow-sandboxes-verified-v2 | llm-judge-freeform | 10,000 | 10,000 | 100% |  (0) |
| laion__stackexchange-superuser-sandboxes-verified-v2 | llm-judge-freeform | 10,000 | 10,000 | 100% |  (0) |
| laion__stackexchange-tezos-sandboxes-verified-v2 | llm-judge-freeform | 10,000 | 997 | 10% | duplicate (9,003) |
| laion__stackexchange-unix-sandboxes-verified-v2 | llm-judge-freeform | 10,000 | 10,000 | 100% |  (0) |
| laion__wizardlm-orca-v4 | llm-judge-freeform | 10,000 | 10,000 | 100% |  (0) |
| laion__nemotron-gym-instruction-following-structured-v3 | instruction-following | 9,437 | 9,167 | 97% | unsupported_variant (254) |
| DCAgent__code-contests-noblock | competitive-programming | 8,728 | 8,224 | 94% | gold_in_instruction (504) |
| laion__all-puzzles-v2 | math-answer | 6,926 | 6,719 | 97% | verified:gold_leak (202) |
| laion__nemotron-gym-instruction-following-calendar-v3 | instruction-following | 5,673 | 5,673 | 100% |  (0) |
| laion__nemotron-gym-math-v5 | math-answer | 4,096 | 3,891 | 95% | verified:empty (157) |
| laion__nemotron-gym-agent-calendar-v2 | tool-use | 2,699 | 2,699 | 100% |  (0) |
| laion__nemo-prism-math-v3 | math-answer | 2,404 | 2,219 | 92% | duplicate (182) |
| DCAgent2__nl2bash-tasks-cleaned-oracle-v2 | shell-cmd | 1,498 | 1,498 | 100% |  (0) |
| laion__nemotron-gym-agentic-indirect-prompt-injection-v3 | prompt-injection | 1,272 | 1,272 | 100% |  (0) |
| laion__nemotron-gym-multichallenge-advanced-v4 | llm-judge-freeform | 1,068 | 1,052 | 99% | duplicate (16) |

Dropped sources (59, 233,948 tasks); the full reason per source is in `source_verdicts.json`:

| source | tasks | reason |
|---|---:|---|
| laion__nemotron-gym-identity-following-v4 | 21,660 | Persona is NVIDIA's; judge-only. Rewrite with our identity and deterministic name/language checks, or skip. |
| SankalpKJ__nemotron-code-oracle-filtered | 15,165 | Only test is the example shown in the prompt. Oracle solutions exist, so generate hidden cases by fuzzing inputs through the oracle. |
| laion__openswe-tasks-patched-v7-oracle-success | 11,730 | No FAIL_TO_PASS ids: the v7 verifier scores whichever tests its custom pytest guard plugin saw execute, and the repository is cloned by a root-level setup script at agent time. Needs its own converter. |
| laion__tulu3-sft-personas-math-sandboxes-verified-v3 | 9,998 | Easy SFT persona math, gold in plaintext, carries the terminal-bench canary. |
| laion__exp_rpt_ghactions-v3 | 9,930 | Instruction lists every job and step verbatim; workflow is never executed. Transcription. |
| DCAgent__inferredbugs-sandboxes-verifier | 9,659 | Never compiles or runs; regex on the rewritten method body with guards that accept either polarity. |
| laion__nemotron-gym-agentic-function-calling-pivot-v3 | 9,579 | Predict-the-next-call from a frozen transcript with exact key-set match. The only tool-call data here; rewrite into executable mock-tool envs built from the transcripts. |
| laion__nemotron-gym-instruction-following-citation-v2 | 9,033 | Grades presence of literal marker substrings; never checks the cited content. |
| laion__nemotron-gym-instruction-following-freeform-v2 | 8,869 | Counts markdown tables and bullets; no content check. |
| laion__exp_rpt_stack-rspec-v4 | 8,860 | Real Ruby test files but gems are never installed and some tasks are unsolvable offline. Bake gems and drop tasks that fail the oracle gate. |
| laion__exp_rpt_stack-cpp-v4 | 7,878 | Tests are lifted from real repositories with the repository stripped: sampled tasks include headers and data files that do not exist in the image, and one pastes the reference Solution class inside the test. |
| laion__exp_rpt_codenet-python-v4 | 6,975 | Only 3 hidden cases and whitespace-collapsing compare. Oracle present; regenerate 20+ cases per task. |
| DCAgent__selfinstruct-naive-sandboxes-2-verified-v3 | 6,665 | Per-task LLM-written test_state.py with loose file discovery and dead code. Task ideas are usable; regenerate verifiers with an oracle/no-op gate. |
| laion__nemotron-gym-math-advanced-calculations-v4 | 5,291 | Instruction refers to tools that do not exist and only the last number is graded. Ground-truth expression tree is present, so rewrite with a calculator tool and grade every subexpression. |
| laion__nemotron-gym-litmus-bench-v2 | 5,232 | Instruction asks for ((answer)), verifier greps boxed or last number; SMILES tasks with no RDKit. Fix format contract and install cheminformatics. |
| DCAgent__exp_rpt_e2egit-large | 4,998 | 8 of 10 sampled tasks are the same Calculator class; metadata is inert boilerplate. |
| DCAgent__exp_rpt_pymethods2test-large-v2 | 4,991 | Single pure-function katas whose examples mirror the tests. Too easy to spend RL on. |
| DCAgent__exp_rpt_unitsyn-python-large-v2 | 4,991 | Single-function implementations from synthesized unittest; first-.py-file fallback. Kata-grade. |
| DCAgent__exp_rpt_multifile-v3 | 4,843 | LLM-synthesized three-module toys (text normalizer, 1D Kalman update). Kata-grade, the tier already dropped for pymethods2test and unitsyn. |
| DCAgent__exp_rpt_nemotron-cpp | 4,196 | GoogleTest grading, but the reference implementation is pasted inside the test file and some tests need doctest, which is not installed. |
| laion__magicoder-v4 | 4,096 | Judge-only over a bundle of every file under /app collected by tests/collect_submission.py; the judge mode grades one answer file, so the bundle shape needs its own converter. Vague refactor prompts, nothing executed. |
| laion__toolscale-v4 | 4,048 | Good design (offline tool service) but the CLI script embeds the gold calls and answer, and the prompt states the conclusion. Move the fixture behind a server and strip the success criteria. |
| laion__exp_rpt_crosscodeeval-typescript-v2 | 3,356 | Re-skin of the Python variant with the same free 0.25 tier; metadata still says python. |
| laion__nemotron-gym-qa-abstention-v4 | 3,150 | Abstention is never rewarded so the framing is dead, reference leaks into judge text, and it duplicates openqa. |
| laion__exp_rpt_scaffold-v3 | 3,121 | LLM-synthesized stub-filling toys (TypeScript formatter shim, Flask hello page). Kata-grade. |
| laion__nemotron-gym-knowledge-web-search-mcqa-v2 | 2,915 | Promises web search but ships no tool. Worth rewriting as a real search-tool env; otherwise it is a 3k duplicate of mcqa. |
| laion__mix_h11_single_skill_only-v2 | 2,859 | Mixture of other sources, including content-free crosscodeeval slices and a syntactically broken test file. |
| laion__mix_h10_reward_proportional-v2 | 2,858 | Mixture; codereval slice tests a local mock rather than the solution. |
| laion__mix_h8_original_tests-v2 | 2,848 | Mixture; import-only test files for three of ten sampled tasks. |
| DCAgent__exp_rle_adversarial-v6 | 2,726 | Same generator as stack-pytest: implement a fake ddtrace/godot/mssqlcli module so LLM-written tests pass. Kata-grade fakes of real libraries. |
| laion__r2egym-patched-full-oracle-v3 | 2,574 | Grades by overlap between test_info.json and expected_output_json rather than by pytest node id; not the trusted-paths shape the swe converters handle. |
| laion__swegym-tasks-patched-validated-v5 | 2,428 | Image ships no repository: the instruction clones it and runs make init at agent time, and the old grader pip-installed requirements again at grading time. Sampled oracles fail on missing dependencies and the empty check cannot start. |
| laion__exp_rpt_stack-go-v5 | 2,275 | Tests import packages from the original repository (bridgr/internal/..., gosnowflake internals) that are not in the task, so most tasks are unsolvable as specified. |
| laion__exp_rpt_crosscodeeval-java-v3 | 2,139 | Exact string match on a single line completion. Not agentic, no execution. |
| DCAgent__mix_h4_binary_easy | 1,996 | Mixture; crosscodeeval slice only checks import succeeds. |
| laion__nemotron-gym-instruction-following-multiturnchat-v4 | 1,982 | Required literal format contradicts the demonstrated turns; judge-only. |
| laion__exp_rpt_stack-pytest-large-v3 | 1,782 | Sampled tests only assert truthiness of return values; a stub that returns a dict passes. Same stripped-repository shape as stack-go. |
| laion__exp_rpt_crosscodeeval-csharp-v4 | 1,768 | 0.25 reward for any identifier-shaped output; instruction coaches the hack. |
| laion__nemotron-gym-agentic-swe-pivot-v4 | 1,541 | No repo in the container; a 9B judge rates one predicted next action. |
| laion__exp_rpt_methods2test-large-v4 | 1,194 | Single @Test pasted verbatim into the prompt; implement one known assertion. |
| laion__nemotron-gym-multichallenge-vanilla-v3 | 1,050 | Single subjective criterion with 'Expected answer: YES' embedded in the judge prompt. |
| laion__nemotron-gym-sysbench-v4 | 1,010 | Deterministic gate before the judge uses 31 constraint ids outside the IFEval registry (tables, heading depth, numbered lists, unique words, ...); only 328 of 1,478 tasks are gate-able today. Port the gate checks before converting. |
| laion__nemotron-gym-instruction-following-adversarial-v5 | 1,000 | Asks an LLM judge to count exactly five spelling errors. |
| laion__nemotron-gym-inverse-ifeval-v4 | 1,000 | Gate matches against a deliberately broken synthetic reference; then judge. |
| laion__exp_rpt_stack-junit-v6 | 843 | JUnit grading is real but every instruction cites a test path that does not exist and scan-class-path counts any test class. |
| laion__exp_rpt_stack-dockerfile-gpt5mini-v7 | 587 | 587 rows of gpt-5-mini-written per-task test scripts whose instructions describe containers that do not exist. |
| DCAgent__exp_rpt_curriculum-easy | 509 | Starter skeleton is a complete working solution. |
| DCAgent__exp_rpt_pymethods2test-v3 | 500 | 500-row early cut of the same katas. |
| DCAgent__exp_rpt_stack-pytest-v2 | 500 | Pre-hardening version superseded by adversarial-v6 and pytest-large-v3. |
| laion__codeelo-v2 | 500 | Byte-identical generator to codeforces-v3 at 500 rows; merge, do not keep separately. |
| laion__exp_rpt_crosscodeeval-python-v2 | 500 | 0.25 for any non-empty output; instruction discloses the tiers. |
| DCAgent__exp_rpt_curriculum-medium-v2 | 492 | LLM-synthesized "implement a mock cluster" toys. Kata-grade. |
| DCAgent__exp_rpt_unitsyn-python-v4 | 491 | Small cut of the same. |
| DCAgent__exp_rpt_e2egit-v2 | 487 | Same generator as e2egit-large at 487 rows. |
| laion__exp_rpt_bugsinpy-v4 | 479 | LLM-synthesized tests against a single-file stub, with assert True placeholders. Rewrite against the real BugsInPy project suites. |
| laion__nemotron-gym-cfbench-v4 | 468 | Deterministic gate before the judge uses 31 constraint ids outside the IFEval registry (tables, heading depth, numbered lists, unique words, ...); only 328 of 1,478 tasks are gate-able today. Port the gate checks before converting. |
| laion__exp_rpt_stack-php-large-v9 | 462 | Fail-open exit paths, regex class discovery, 462 rows. |
| laion__exp_rpt_nemotron-junit-v6 | 447 | 20% of sampled tasks contain unconditional fail() stubs the verifier restores. |
| laion__exp_rpt_stack-jest-v5 | 424 | Spy-call contracts against a 90-package global npm image; 424 rows. |

kept sources: 34, dropped sources: 59, dropped tasks: 233,948

By converter (clean, rejected inside the converter, distinct Dockerfiles):

| converter | clean | rejected | dockerfiles |
|---|---:|---:|---:|
| agent_calendar | 2,699 | 0 | 1 |
| all_puzzles | 6,719 | 207 | 2 |
| code_contests | 8,224 | 0 | 1 |
| codeforces | 9,697 | 48 | 1 |
| judge_rubric | 105,875 | 53,191 | 4 |
| nemotron_competitive | 13,974 | 0 | 1 |
| nemotron_if_structured | 14,840 | 0 | 2 |
| nemotron_ifeval | 46,391 | 0 | 1 |
| nemotron_math | 216,266 | 1,377 | 2 |
| nemotron_mcqa | 611,699 | 1,582 | 1 |
| nemotron_multichallenge | 1,052 | 16 | 1 |
| nemotron_openqa | 272,429 | 572 | 1 |
| nemotron_reasoning | 33,706 | 271 | 3 |
| nemotron_structured_outputs | 30,162 | 290 | 1 |
| nl2bash | 1,498 | 0 | 1 |
| prompt_injection | 1,272 | 0 | 1 |
| swe_patched | 5,473 | 2 | 2 |
| swe_trusted_paths | 12,720 | 0 | 2 |
| taco | 5,117 | 0 | 1 |

kept sources: 34, dropped sources: 59, dropped tasks: 233,948

Distinct Dockerfiles (21):

| id | base image | tasks | converters | sources |
|---|---|---:|---|---:|
| a376fa78ab92 | python:3.11-slim-bookworm | 691,702 | nemotron_mcqa, nemotron_ifeval, nemotron_competitive, nemotron_reasoning, nemotron_if_structured, agent_calendar, prompt_injection | 7 |
| 43c753bdead4 | python:3.12-slim-bookworm | 318,365 | nemotron_openqa, judge_rubric, nemotron_multichallenge | 4 |
| bd9a35225592 | python:3.11-slim-bookworm | 212,375 | nemotron_math | 4 |
| 8f2aaecc08aa | ubuntu:24.04 | 40,997 | judge_rubric | 5 |
| 326242f77269 | python:3.11-slim-bookworm | 30,162 | nemotron_structured_outputs | 1 |
| ebab3261fe80 | python:3.11-slim-bookworm | 13,712 | nemotron_reasoning | 1 |
| d9ce27d56609 | python:3.10-bookworm | 12,191 | swe_trusted_paths | 1 |
| e4f12d5b654f | ubuntu:24.04 | 10,000 | judge_rubric | 1 |
| c69561d60b92 | python:3.11-slim-bookworm | 10,000 | nemotron_reasoning | 1 |
| 6f72aae93950 | ubuntu:24.04 | 9,994 | judge_rubric | 1 |
| c25a635c54d7 | python:3.10-slim | 9,697 | codeforces | 1 |
| 0dd7761c6cc9 | python:3.11-slim-bookworm | 9,167 | nemotron_if_structured | 1 |
| eae3b0d61489 | ubuntu:24.04 | 8,224 | code_contests | 1 |
| 755a683907d9 | python:3.10-slim | 5,117 | taco | 1 |
| 05f00c703407 | python:3.10-bookworm | 4,835 | swe_patched | 1 |
| 7e104dcbdc83 | python:3.11-slim-bookworm | 3,891 | nemotron_math | 1 |
| fc59aacf80c1 | python:3.10-slim | 3,669 | all_puzzles | 1 |
| 5dfaae492bf8 | python:3.10-slim | 3,050 | all_puzzles | 1 |
| cc509280253f | ubuntu:24.04 | 1,498 | nl2bash | 1 |
| 5bb293ce54d2 | python:3.9-bookworm | 638 | swe_patched | 1 |
| 2b2d3c8071a4 | python:3.11-bookworm | 529 | swe_trusted_paths | 1 |

Most common tags:

| tag | tasks |
|---|---:|
| nemotron | 1,244,490 |
| qa | 884,128 |
| mcq | 611,699 |
| judge | 379,356 |
| openqa | 272,429 |
| reference | 272,429 |
| math | 216,266 |
| science | 150,468 |
| knowledge | 121,961 |
| rubric | 105,875 |
| no-reference | 105,875 |
| code | 95,199 |
| instruction-following | 56,610 |
| ifeval | 46,391 |
| safety | 46,156 |
| stackexchange | 40,997 |
| json-schema | 39,329 |
| competitive-programming | 37,012 |
| stdio | 37,012 |
| reasoning | 33,706 |
| structured-outputs | 30,162 |
| arc-agi | 19,994 |
| swe | 18,193 |
| swe-repo | 18,193 |
| json | 15,652 |
| yaml | 14,510 |
| reasoning-gym | 13,712 |
| trusted-test-paths | 12,720 |
| shell | 11,498 |
| general-assistant | 10,000 |
| grid-transform | 10,000 |
| unix | 10,000 |
| stackoverflow | 10,000 |
| superuser | 10,000 |
| codereview | 10,000 |
| grid-match | 9,994 |
| code-assistant | 9,994 |
| codeforces | 9,697 |
| tool-use | 9,644 |
| structured-output | 9,167 |

Largest rejection reasons inside kept sources: 22,863 structured-outputs tasks ask for xml, toml, or csv
documents that the json-schema mode cannot validate; 12,453 swe_rebench tasks are Go, JavaScript, TypeScript, or
Rust repositories the pytest converter does not handle; 4,846 TACO, competitive-coding, and code-contests tasks
have no hidden input beyond the prompt's samples; 3,607 MCQA tasks have a numeric gold instead of an option
letter; 1,463 MCQA prompts carry literal `\n` sequences instead of newlines so only option A is detected;
854 math tasks have an expected value math-verify cannot parse (`floor(n^2/4)`, `(1, ∞)`, `f(x) = x + c`,
`Symmetric`); 240 codeforces tasks ship no cases.

Sample: 100 tasks drawn at random (seed 20260915): 52 mcq, 24 judge, 14 math, 2 stdio, 2 pytest, 2 script, 2 ifeval, 1 json-schema, 1 reasoning-gym. Every instruction is answerable as written and
names the file the grader reads; every Dockerfile pins the tool ref above. Under Docker the empty workspace must
score 0 and the oracle 1: 169 of 169 checks over 11 images. The earlier samples exposed six defects that this run fixes: half of
the MCQA prompts ask for `Answer: \boxed{X}`, which the mcq mode rejected; nemotron-math-oracle prompts named
`/app/solution.txt` while the grader read `/app/answer.txt`; some openqa references carried a leading `**`; the
voluptuous swesmith tasks failed pytest collection because the pytest mode clears `addopts` (dropping the repo's
`--doctest-glob=*.md`), so FAIL_TO_PASS ids outside `.py` files are rejected at conversion; parso's swesmith
tasks graded doctest items (`parso/__init__.py::parso`) that the same cleared `addopts` never collects, so
doctest node ids are rejected too (143 tasks) and dropped from PASS_TO_PASS (637 tasks); and 65 TACO oracles
are function bodies pasted at module level (`return` or `nonlocal` outside a function), now rejected because
they cannot compile. The sample of run 2026.09.10.2 was clean; the RL smoke on it then found that Harbor
rejects the tool's reward file, fixed in the tool for run 2026.09.10.3, whose sample was also clean. Known quirks left in place: one MCQA prompt lists its options twice (once as `A)` and once
as `A:`); SankalpKJ expected values are sometimes 30-digit decimals rather than closed forms; swesmith and
swe_rebench tasks clone their repository at agent time, so they need network access.

## Validity sample (tasks 17 and 18)

`validity.py sample` drew ten tasks per converter from clean 2026.09.10.4 (190 tasks, seed 20260915) and
exported them as Harbor task directories; `validity.py solve` asked Claude Sonnet (headless `claude -p`, no
tools, one attempt) for a `solve.sh` per task ($20.30 for 185 replies; four calls hung past the 600 s cap and
one instruction-following prompt was refused); `validity_daytona.py` then ran three checks per task in a
Daytona sandbox built from the task's Dockerfile (one snapshot per distinct Dockerfile, 16 concurrent
sandboxes, 1 CPU / 2 GB): the empty workspace, the shipped oracle, and the candidate. `validity.py report`
folds the verdicts into `report.md`. A 54-task pilot (three per converter) ran first and shaped the runner
(whole-directory uploads, snapshot quota handling, per-task agent timeouts).

Results (full table in the artifact): 190 of 190 empty workspaces scored 0; 90 of 94 oracles scored 1; of the
155 scorable candidates 117 scored 1 and 38 scored 0, none partial; the 30 judge tasks (openqa,
multichallenge, judge_rubric) are `infra_error` because no judge endpoint exists for the project (no Anthropic
key anywhere, the stored OpenAI key is invalid), which is the tool's intended masking path. Per-group solve
rates: prompt_injection and codeforces 1.0; taco 0.9; code_contests, nemotron_competitive and
nemotron_reasoning 0.89; agent_calendar, nemotron_if_structured, nemotron_math and nemotron_structured_outputs
0.8; nemotron_ifeval 0.78; all_puzzles and nemotron_mcqa 0.7; nl2bash, swe_patched and swe_trusted_paths 0.4.

The zeros are of three kinds. Model errors (wrong MCQA letter, wrong math answer, a code-contests solution
passing 1 of 5 cases, calendar event names, ifeval constraint misses). Format ambiguity the instruction does not
resolve (three all-puzzles answers given as a comma list instead of the expected form; six nl2bash tasks whose
expected output is the oracle command's exact record stream). Task defects, visible in the oracle column:
nl2bash task_2493 seeds a 200,000-byte file with `truncate` but expects 16,666 lines of `sample_line`;
swe_rebench cloud-sql-python-connector-906 has a `tests/pytest.ini`, so pytest reported node ids relative to
`tests/` and the tool matched none although the test passed (tool bug, fixed in d28b440f: ids are rebased onto
the workspace, and a failing grade keeps the output tail in the verdict); pennylane-2591 has the same
layout and with the fixed tool passes 129 of 132 (the other three fail in the patch's own test file); qiskit-terra-8924's test package imports
`ddt`, which the image does not ship and the oracle's own pip step installs only sometimes, so it passes 13 of
13 under local Docker and fails collection in the Daytona sandbox. The pilot found meltano's PASS_TO_PASS
ids cut at a space inside a parameter (now uncollectable at conversion) and pytensor grading macOS-parametrized
ids. Conclusion: the deterministic groups are solvable as written; nl2bash needs its instructions to pin the
output format; SWE tasks need an in-container oracle gate at conversion time; the judge groups are unmeasured.

## Smoke RL run (task 13)

`experiments/post_training/tasktrove/rl_smoke.py` trains Qwen3-0.6B for two GRPO steps on a sample of the
first clean shard through MarinSkyRL's `terminal_bench` entrypoint. The sample step reads shard 0 of the clean
output (the output is hash-sharded, so one shard is a uniform 1/1024 sample of about 1,270 tasks over every
kept converter), picks tasks round-robin across converters, and writes each as a Harbor task directory under
`users/power/tasktrove/rl-smoke-tasks/<version>/tasks/<source>__<path>/`, solutions excluded. The RL step
reuses the curriculum experiment's mirrored Qwen3-0.6B (`models/curriculum-rl-qwen3-0.6b/2026.08.29`) and runs
on `cw-rno2a` with two H100 nodes: one FSDP2 policy node and eight single-GPU vLLM engines
(`train_batch_size` 32, 4 samples per prompt, `micro_train_batch_size_per_gpu` 4, non-thinking chat template,
16,384-token window, 2,048 new tokens per turn, 10 turns). Harbor's terminus-2 agent drives each task in a
Daytona sandbox (1 CPU, 2 GB, 64 concurrent trials, `auto_snapshot` keyed by Dockerfile hash, well under the
40-snapshot org quota) and `tasktrove-verify` inside the image produces the reward.

Submitted from a CPU coordinator on `cw-rno2a`; the Daytona key is read from Secret Manager on the submit host
and forwarded with `-e DAYTONA_API_KEY` because coordinator pods carry no GCP credentials (the launcher then
skips its snapshot purge, since the daytona SDK is absent there). The judge tasks in the sample score 0 in this
run because no `TASKTROVE_JUDGE_*` endpoint is configured in the sandbox; a training run that wants judge
rewards has to set those three variables.

Attempt 1 (`/power/iris-run-job-20260910-032906`, clean 2026.09.10.1, the whole shard of 1,273 tasks): MarinSkyRL
materializes a task-directory data source one object at a time, so the 8,265 exported files held both GPU
nodes for 58 minutes before Ray started. Both training steps then completed (128 trajectories each, checkpoint
and HF export written), but every one of the 2,375 trials failed on its second turn with
`404 Not Found` from `http://<head>:8000/tokenize`: Harbor's exact-token continuation (harbor#111/#117) asks
vLLM's `/tokenize` for the next prompt, and the SkyRL inference HTTP endpoint only serves the chat and
completion routes. The first turn worked in every trial (sandbox started, ~180 output tokens), so the
Daytona, vLLM, trainer and export paths were exercised while the reward path was not; `reward/avg_raw_reward`
was 0.0. Filed as marin-community/MarinSkyRL#538.

Attempt 2 (`/power/iris-run-job-20260910-045146`, clean 2026.09.10.2): the export is capped at 160 tasks drawn
round-robin across converters (2,685 objects, 18 minutes of staging) and `collect_rollout_details: false`
makes Harbor count tokens locally instead of calling `/tokenize` (allowed because the objective needs no
behavior logprobs; it drops TIS/TITO evidence, which a real run would want back once #538 is fixed). The
reward path now ran and failed in Harbor's verifier: `tasktrove-verify` wrote `reward.json` as
`{"reward", "status", "detail"}` and Harbor's `VerifierResult` only accepts a name-to-number map, so all 2,334
trials raised a pydantic `ValidationError` and the run sat at step 0 until it was cancelled. Fixed in
5062aaa3: the tool writes `verdict.json` with the status and detail, `reward.json` as `{"reward": x}` and
`reward.txt` only for a scored grade, so an invalid task or grader crash becomes Harbor's missing-reward
error and is masked. Because the tool ref is baked into every Dockerfile, the pipeline reran as 2026.09.10.3
(`/power/iris-run-job-20260910-055608`, tool ref 5062aaa3, converters unchanged).

Attempt 3 (`/power/iris-run-job-20260910-061854`, clean 2026.09.10.3): succeeded end to end. 160 tasks over 16
converters (2,685 objects, staging 06:21 to 06:42), then two steps of 128 trajectories (190 s and 91 s), 726
trials, 0 failed trials, 0 masked, 166 `TurnCapExhaustedError`, 328 to 3,865 output tokens per trajectory
(mean 1,219). Every trial ran `tasktrove-verify` and produced `reward.json` with `{"reward": 0.0}` and a
`verdict.json` with status `scored`, so the reward path is exercised and every task in the sample is
gradable; `reward/avg_raw_reward` was 0.0 on both steps because Qwen3-0.6B solved nothing. The verdict
details say why: the agent almost never wrote the answer file (`no_output` dominates every mode; the
trajectories show the model repeating `ls -la` / `cd project` without a newline until the turn cap),
swesmith trials hit `setup_failed` (34, the clone-at-agent-time repositories), codeforces `build_failed`
(24), stdio tasks scored `passed=0/N`, arc-agi "no transform() found", and the 14 judge tasks scored 0 for
lack of a `TASKTROVE_JUDGE_*` endpoint. Checkpoint and HF export at
`users/power/checkpoints/tasktrove-rl-smoke/2026.09.10.3`, W&B run
https://wandb.ai/marin-community/marin-tasktrove/runs/ugi33mqj.

## Not in this PR

Filed as marin-community/marin#9083 (in-container oracle gate for repository tasks, nl2bash output formats and
seeded fixtures, judge sampling once an endpoint exists, the recoverable pools rejected by shape, dedup policy
for rubric sources, PyPI release and Hugging Face upload) and marin-community/MarinSkyRL#540 (read tasks from
the clean parquet instead of staged task directories, snapshot caching across runs under the Daytona quota,
`/tokenize` (#538), judge endpoint plumbing, sandbox environment and egress for repository tasks, installing
the tool without re-versioning the dataset).
