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
9. [ ] Run 2026.09.10.1 (`/power/iris-run-job-20260910-030245`, tool ref 105b4541): regenerate the tables, draw a fresh 100-task sample, Docker-check it, fix anything obvious it exposes and rerun if needed
10. [ ] Delete superseded bucket artifacts: `tasktrove/{fingerprints,verified,deduped,template_summaries,templates,converted,graded,clean}/2026.09.09`, `tasktrove/*/2026.09.10`, `raw/tasktrove/2026.09.10` (duplicate download; the pipeline pins `raw/tasktrove/2026.09.09`); keep `raw/tasktrove/2026.09.09` and `tasktrove/*/2026.09.10.1`
11. [ ] Update this logbook's run section and the PR body with the final numbers; verify with `gh pr view --json title,body`
12. [ ] Publish an artifact with the detailed analysis: source corpus before and after, kept and dropped sources with reasons, per-check rejections, how tasks were normalized (spec, Dockerfile, instruction edits, dedup), sample findings
13. [ ] Smoke-train Qwen 0.6B on the clean dataset through marin skyrl, configured after the curriculum experiment; record the config and result here
14. [ ] Final report to the user and PR monitoring per the commit skill

## Run 2026.09.10.1

Job `/power/iris-run-job-20260910-030245` on `cw-us-east-02a`, tool ref `105b4541b630c60750967cc736dc7bc3a781bfd4`,
25 minutes end to end (summaries 4 min, templates 4 min, converted 4 min, graded 4 min, clean 4 min; the raw
download is pinned to `raw/tasktrove/2026.09.09` and was reused). Output at
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.1`: `tasks/part-*.parquet` (1,025 shards, 3.17 GiB),
`ledger.parquet` (one row per rejected task with its reason), `manifest.json`, `report.md`.

| status | tasks | meaning |
|---|---:|---|
| converted | 1,294,537 | in the clean dataset |
| dropped_source | 393,014 | source rejected in `source_verdicts.json` |
| unsupported_variant | 42,261 | converter has no grading for this task shape (xml/toml/csv schemas, non-Python SWE repositories or test files, non-letter MCQ gold) |
| gold_in_instruction | 4,861 | every hidden stdio case is a sample printed in the prompt |
| verified:empty | 2,433 | grader cannot score even an empty output (unparseable expected value, options not detected) |
| verified:gold_leak | 990 | expected value is visible in the instruction |
| duplicate | 942 | same instruction as an earlier task in the source |
| null_grader | 288 | no stdio cases or an empty schema |

Clean tasks by mode:

| mode | tasks |
|---|---:|
| mcq | 611,699 |
| judge | 273,481 |
| math | 219,316 |
| ifeval | 46,391 |
| json-schema | 39,329 |
| stdio | 37,077 |
| script | 21,142 |
| pytest | 18,727 |
| reasoning-gym | 13,712 |
| exact | 13,663 |

Kept sources (26):

| source | family | input | clean | kept % | largest loss |
|---|---|---:|---:|---:|---|
| laion__nemotron-gym-knowledge-mcqa-v2 | qa-short-answer | 616,888 | 611,699 | 99% | unsupported_variant (3,607) |
| laion__nemotron-gym-science-so-openq-v3 | llm-judge-freeform | 150,644 | 150,468 | 100% | verified:gold_leak (176) |
| laion__nemotron-gym-knowledge-openqa-v4 | qa-short-answer | 122,357 | 121,961 | 100% | verified:gold_leak (350) |
| laion__nemotron-gym-math-stack-overflow-v3 | math-answer | 110,730 | 110,267 | 100% | verified:gold_leak (232) |
| SankalpKJ__nemotron-math-oracle-filtered-v2 | math-answer | 57,777 | 57,383 | 99% | verified:empty (364) |
| laion__nemotron-gym-structured-outputs-v4 | other | 53,870 | 30,162 | 56% | unsupported_variant (23,386) |
| laion__nemotron-gym-instruction-following-v3 | instruction-following | 46,391 | 46,391 | 100% |  (0) |
| laion__nemotron-gym-math-openmathreasoning-v2 | math-answer | 42,636 | 42,506 | 100% | verified:empty (102) |
| DCAgent__swe_rebench_v2_patched_oracle-v2 | swe-repo | 18,319 | 5,864 | 32% | unsupported_variant (12,453) |
| laion__nemotron-gym-competitive-coding-v2 | competitive-programming | 15,713 | 13,974 | 89% | gold_in_instruction (1,739) |
| laion__nemotron-gym-reasoning-gym-v2 | other | 14,259 | 13,712 | 96% | unsupported_variant (282) |
| laion__swesmith-oracle-filtered-v2 | swe-repo | 12,927 | 12,863 | 100% | unsupported_variant (64) |
| laion__codeforces-v3 | competitive-programming | 10,000 | 9,697 | 97% | null_grader (240) |
| laion__exp_rpt_taco-v2 | stdin-stdout | 10,000 | 5,182 | 52% | gold_in_instruction (2,603) |
| laion__nemotron-gym-arc-agi-python-inductive-v2 | other | 10,000 | 10,000 | 100% |  (0) |
| laion__nemotron-gym-arc-agi-transductive-v3 | other | 10,000 | 9,994 | 100% | verified:gold_leak (6) |
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

Dropped sources (67, 393,014 tasks); the full reason per source is in `source_verdicts.json`:

| source | tasks | reason |
|---|---:|---|
| laion__nemotron-gym-safety-v3 | 89,066 | Judge-only with a generic per-family rubric and no reference; reward is judge noise. |
| laion__nemotron-gym-identity-following-v4 | 21,660 | Persona is NVIDIA's; judge-only. Rewrite with our identity and deterministic name/language checks, or skip. |
| SankalpKJ__nemotron-code-oracle-filtered | 15,165 | Only test is the example shown in the prompt. Oracle solutions exist, so generate hidden cases by fuzzing inputs through the oracle. |
| laion__openswe-tasks-patched-v7-oracle-success | 11,730 | No FAIL_TO_PASS ids: the v7 verifier scores whichever tests its custom pytest guard plugin saw execute, and the repository is cloned by a root-level setup script at agent time. Needs its own converter. |
| laion__glaive-code-assistant-sandboxes-verified-v2 | 10,000 | Judge-only over code that is never run; boilerplate 4-criterion rubric. |
| laion__stackexchange-codereview-sandboxes-verified-v2 | 10,000 | Judge-only, no gold, gpt-4o-mini, rubric readable by the agent. |
| laion__stackexchange-overflow-sandboxes-verified-v2 | 10,000 | Same harness as codereview; no ground truth. |
| laion__stackexchange-superuser-sandboxes-verified-v2 | 10,000 | Same harness. If shell coverage is wanted, rewrite a small slice into sandboxed tasks instead. |
| laion__stackexchange-tezos-sandboxes-verified-v2 | 10,000 | Same harness and every sampled id is a _copyN duplicate. |
| laion__stackexchange-unix-sandboxes-verified-v2 | 10,000 | Same harness. Candidate for a shell-task rewrite, not for keeping. |
| laion__wizardlm-orca-v4 | 10,000 | Judge-only over stilted Orca paraphrases. |
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
| laion__magicoder-v4 | 4,096 | Judge-only, nothing executed, boilerplate rubric, vague prompts. |
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

kept sources: 26, dropped sources: 67, dropped tasks: 393,014

Largest rejection reasons inside kept sources: 22,863 structured-outputs tasks ask for xml, toml, or csv
documents that the json-schema mode cannot validate; 12,453 swe_rebench tasks are Go, JavaScript, TypeScript, or
Rust repositories the pytest converter does not handle; 4,846 TACO, competitive-coding, and code-contests tasks
have no hidden input beyond the prompt's samples; 3,607 MCQA tasks have a numeric gold instead of an option
letter; 1,463 MCQA prompts carry literal `\n` sequences instead of newlines so only option A is detected;
854 math tasks have an expected value math-verify cannot parse (`floor(n^2/4)`, `(1, ∞)`, `f(x) = x + c`,
`Symmetric`); 240 codeforces tasks ship no cases.

Sample: 100 tasks drawn at random (seed 20260911), SAMPLE_MODES. Every instruction is answerable as written and names the file the grader reads; every Dockerfile
pins the tool ref above. Under Docker, every task's empty workspace scored 0 and every oracle scored 1
(DOCKER_RESULT). Earlier samples exposed four defects that this run fixes: half of the MCQA prompts ask for
`Answer: \boxed{X}`, which the mcq mode rejected; nemotron-math-oracle prompts named `/app/solution.txt` while
the grader read `/app/answer.txt`; some openqa references carried a leading `**`; and the voluptuous swesmith
tasks failed pytest collection because the pytest mode clears `addopts` (dropping the repo's
`--doctest-glob=*.md`), so FAIL_TO_PASS ids outside `.py` files are now rejected at conversion (64 tasks).
Known quirks left in place: one MCQA prompt in the sample lists its options twice (once as `A)` and once as
`A:`); SankalpKJ expected values are sometimes 30-digit decimals rather than closed forms; swesmith and
swe_rebench tasks clone their repository at agent time, so they need network access.

## Not in this PR

Container-based verification of every task, judge sampling against a model, PyPI release,
Hugging Face upload.
