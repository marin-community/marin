# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sources retained in the inventory that do not yet have a conversion recipe."""


from experiments.post_training.task_curation.datasets.tasktrove.archives import TASKTROVE_REPO, TASKTROVE_REVISION
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference

MIMO_VERIFIER = SourceReference("Harbor", "a2ad9f6160b03ff2d47e59832bfb6b289f37c917", "")

TASKTROVE_INPUT = SourceReference(
    TASKTROVE_REPO,
    TASKTROVE_REVISION,
    f"https://huggingface.co/datasets/{TASKTROVE_REPO}/tree/{TASKTROVE_REVISION}",
)
TASKTROVE_RELEASE = SourceReference(
    "open-athena/task-trove",
    "ec049a4fb541ffbe5bbccb803e826563f5718dbf",
    ("https://huggingface.co/datasets/open-athena/task-trove/tree/" "ec049a4fb541ffbe5bbccb803e826563f5718dbf"),
)


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:AweAI-Team__CalibForge",
                title="AweAI-Team/CalibForge",
                origin="Task Trove",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "license:cc-by-4.0"),
                notes=(
                    "Native Harbor tasks retained with original graders; snapshot-unsafe dependencies "
                    "explicitly accepted."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=SourceReference("Harbor", "fb1e75441a94b8bb0ced08acd6b59e711704d70a", ""),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:DCAgent__exp_rle_adversarial-v6",
                title="DCAgent/exp_rle_adversarial-v6",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=2726,
                notes=(
                    "The legacy pytest grader performs source-specific Django discovery and loads extra "
                    "plugins. Ten sampled empty and trivial submissions failed, but the source has no "
                    "oracle and needs a custom environment adapter."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:DCAgent__exp_rpt_nemotron-cpp",
                title="DCAgent/exp_rpt_nemotron-cpp",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=4196,
                notes=(
                    "GoogleTest tasks without shipped oracles. Sampled empty and trivial submissions "
                    "failed, but some tests contain the reference implementation and others require an "
                    "uninstalled doctest dependency."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:DCAgent__inferredbugs-sandboxes-verifier",
                title="DCAgent/inferredbugs-sandboxes-verifier",
                origin="Task Trove",
                family="swe-repo",
                tags=("agentic", "multi-turn", "excluded"),
                count=9659,
                notes=(
                    "Never compiles or runs; regex on the rewritten method body with guards that accept "
                    "either polarity."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:DCAgent__mix_h4_binary_easy",
                title="DCAgent/mix_h4_binary_easy",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=1996,
                notes=(
                    "Mixed verifier shapes: eight of 10 sampled empty and trivial checks timed out, and "
                    "the crosscodeeval slice only checks that an import succeeds."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:DCAgent__selfinstruct-naive-sandboxes-2-verified-v3",
                title="DCAgent/selfinstruct-naive-sandboxes-2-verified-v3",
                origin="Task Trove",
                family="shell-cmd",
                tags=("agentic", "multi-turn", "excluded"),
                count=6665,
                notes=(
                    "Per-task LLM-written test_state.py with loose file discovery and dead code. Task "
                    "ideas are usable; regenerate verifiers with an oracle/no-op gate."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:GAIR__OpenSWE__openswe_oss",
                title="GAIR/OpenSWE__openswe_oss",
                origin="Task Trove",
                family="swe",
                tags=("agentic", "multi-turn", "license:other", "language:unknown"),
                notes=(
                    "Complete canonical openswe_oss population with native build/evaluation commands and "
                    "private full-harness grading. Shipped OSS gold remains in a separate solution "
                    "archive; the other configuration ships no oracle. Native failures are retained for "
                    "quality review. Quality review pending at this release. Independent quality review "
                    "is incomplete. All 36,884 archives passed the static audit; native runtime controls "
                    "cover one selected OSS task. A separate OTHER-configuration control reproduced a "
                    "native grading false positive in the shared harness. That is cross-configuration "
                    "evidence, not an executed OSS bypass or a prevalence estimate. Check the Atlas for "
                    "current review status before training use."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=SourceReference(
                    "Harbor", "f332529660d81fd64eb0ecf73b2bbf75241c36792fef7215de07c98e8ebeded3", ""
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:GAIR__OpenSWE__openswe_other",
                title="GAIR/OpenSWE__openswe_other",
                origin="Task Trove",
                family="swe",
                tags=("agentic", "multi-turn", "license:other", "language:unknown"),
                notes=(
                    "Complete canonical openswe_other population with native build/evaluation commands "
                    "and private full-harness grading. Shipped OSS gold remains in a separate solution "
                    "archive; the other configuration ships no oracle. Native failures are retained for "
                    "quality review. Quality review pending at this release. Independent quality review "
                    "is incomplete. All 8,436 archives passed the static audit; native runtime controls "
                    "cover one selected OTHER task. An incorrect control received full native reward "
                    "despite a canonical test failure. The unchanged native result is preserved; this "
                    "does not estimate source-wide prevalence. Check the Atlas for current review status "
                    "before training use."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=SourceReference(
                    "Harbor", "d8923833284d2018e63fe33775411487dde8635092d8f20a28c0f8b624917f25", ""
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__codeelo-v2",
                title="laion/codeelo-v2",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "excluded"),
                count=500,
                notes="Byte-identical generator to codeforces-v3 at 500 rows; merge, do not keep separately.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_bugsinpy-v4",
                title="laion/exp_rpt_bugsinpy-v4",
                origin="Task Trove",
                family="swe-repo",
                tags=("agentic", "multi-turn", "excluded"),
                count=479,
                notes=(
                    "LLM-synthesized tests against a single-file stub, with assert True placeholders. "
                    "Rewrite against the real BugsInPy project suites."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_codenet-python-v4",
                title="laion/exp_rpt_codenet-python-v4",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "excluded"),
                count=6975,
                notes=(
                    "Only 3 hidden cases and whitespace-collapsing compare. Oracle present; regenerate "
                    "20+ cases per task."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_crosscodeeval-csharp-v4",
                title="laion/exp_rpt_crosscodeeval-csharp-v4",
                origin="Task Trove",
                family="code-completion",
                tags=("agentic", "multi-turn", "excluded"),
                count=1768,
                notes="0.25 reward for any identifier-shaped output; instruction coaches the hack.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_crosscodeeval-java-v3",
                title="laion/exp_rpt_crosscodeeval-java-v3",
                origin="Task Trove",
                family="code-completion",
                tags=("agentic", "multi-turn", "excluded"),
                count=2139,
                notes="Exact string match on a single line completion. Not agentic, no execution.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_crosscodeeval-python-v2",
                title="laion/exp_rpt_crosscodeeval-python-v2",
                origin="Task Trove",
                family="code-completion",
                tags=("agentic", "multi-turn", "excluded"),
                count=500,
                notes="0.25 for any non-empty output; instruction discloses the tiers.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_crosscodeeval-typescript-v2",
                title="laion/exp_rpt_crosscodeeval-typescript-v2",
                origin="Task Trove",
                family="code-completion",
                tags=("agentic", "multi-turn", "excluded"),
                count=3356,
                notes=("Re-skin of the Python variant with the same free 0.25 tier; metadata still says " "python."),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_ghactions-v3",
                title="laion/exp_rpt_ghactions-v3",
                origin="Task Trove",
                family="ci-workflow",
                tags=("agentic", "multi-turn", "excluded"),
                count=9930,
                notes=("Instruction lists every job and step verbatim; workflow is never executed. " "Transcription."),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_methods2test-large-v4",
                title="laion/exp_rpt_methods2test-large-v4",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=1194,
                notes=(
                    "All 10 sampled shipped Java oracles failed because Maven could not resolve its "
                    "plugins offline. Recovering the source requires rebuilding the Java grading "
                    "environment."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_nemotron-junit-v6",
                title="laion/exp_rpt_nemotron-junit-v6",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=447,
                notes="20% of sampled tasks contain unconditional fail() stubs the verifier restores.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_scaffold-v3",
                title="laion/exp_rpt_scaffold-v3",
                origin="Task Trove",
                family="code-generation",
                tags=("agentic", "multi-turn", "excluded"),
                count=3121,
                notes=(
                    "LLM-synthesized stub-filling toys (TypeScript formatter shim, Flask hello page). " "Kata-grade."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-cpp-v4",
                title="laion/exp_rpt_stack-cpp-v4",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=7878,
                notes=(
                    "Tests are lifted from real repositories with the repository stripped: sampled tasks "
                    "include headers and data files that do not exist in the image, and one pastes the "
                    "reference Solution class inside the test."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-dockerfile-gpt5mini-v7",
                title="laion/exp_rpt_stack-dockerfile-gpt5mini-v7",
                origin="Task Trove",
                family="tool-use",
                tags=("agentic", "multi-turn", "excluded"),
                count=587,
                notes=(
                    "587 rows of gpt-5-mini-written per-task test scripts whose instructions describe "
                    "containers that do not exist."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-go-v5",
                title="laion/exp_rpt_stack-go-v5",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=2275,
                notes=(
                    "Tests import packages from the original repository (bridgr/internal/..., gosnowflake "
                    "internals) that are not in the task, so most tasks are unsolvable as specified."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-jest-v5",
                title="laion/exp_rpt_stack-jest-v5",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=424,
                notes="Spy-call contracts against a 90-package global npm image; 424 rows.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-junit-v6",
                title="laion/exp_rpt_stack-junit-v6",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=843,
                notes=(
                    "JUnit grading is real but every instruction cites a test path that does not exist "
                    "and scan-class-path counts any test class."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-php-large-v9",
                title="laion/exp_rpt_stack-php-large-v9",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=462,
                notes="Fail-open exit paths, regex class discovery, 462 rows.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-pytest-large-v3",
                title="laion/exp_rpt_stack-pytest-large-v3",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=1782,
                notes=(
                    "Stripped-repository pytest tasks without shipped oracles. Two of 10 sampled empty "
                    "and trivial checks timed out, and sampled tests include truthiness-only assertions "
                    "that weak stubs can satisfy."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_stack-rspec-v4",
                title="laion/exp_rpt_stack-rspec-v4",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=8860,
                notes=(
                    "Real Ruby test files but gems are never installed and some tasks are unsolvable "
                    "offline. Bake gems and drop tasks that fail the oracle gate."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__magicoder-v4",
                title="laion/magicoder-v4",
                origin="Task Trove",
                family="llm-judge-freeform",
                tags=("agentic", "multi-turn", "excluded"),
                count=4096,
                notes=(
                    "Judge-only over a bundle of every file under /app collected by tests/"
                    "collect_submission.py; the judge mode grades one answer file, so the bundle shape "
                    "needs its own converter. Vague refactor prompts, nothing executed."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__mix_h10_reward_proportional-v2",
                title="laion/mix_h10_reward_proportional-v2",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=2858,
                notes=(
                    "Four of 10 sampled trivial submissions received full credit because the codereval "
                    "slice tests a local mock rather than the solution."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__mix_h11_single_skill_only-v2",
                title="laion/mix_h11_single_skill_only-v2",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=2859,
                notes=(
                    "Mixed verifier shapes: eight of 10 sampled empty and trivial checks timed out; the "
                    "source also contains content-free crosscodeeval slices and syntactically invalid "
                    "tests."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__mix_h8_original_tests-v2",
                title="laion/mix_h8_original_tests-v2",
                origin="Task Trove",
                family="unit-test-gen",
                tags=("agentic", "multi-turn", "excluded"),
                count=2848,
                notes=(
                    "All 10 sampled empty and trivial checks timed out, and prior inspection found import-"
                    "only test files in three of 10 tasks."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-agentic-function-calling-pivot-v3",
                title="laion/nemotron-gym-agentic-function-calling-pivot-v3",
                origin="Task Trove",
                family="tool-use",
                tags=("agentic", "multi-turn", "excluded"),
                count=9579,
                notes=(
                    "Predict-the-next-call from a frozen transcript with exact key-set match. The only "
                    "tool-call data here; rewrite into executable mock-tool envs built from the "
                    "transcripts."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-agentic-indirect-prompt-injection-v3",
                title="laion/nemotron-gym-agentic-indirect-prompt-injection-v3",
                origin="Task Trove",
                family="prompt-injection",
                tags=("agentic", "multi-turn", "excluded"),
                count=1272,
                notes=(
                    "Five of five reviewed rows use the same negative-only checker, which rewards a "
                    "generic reply or unrelated action without validating the required safe continuation."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-agentic-swe-pivot-v4",
                title="laion/nemotron-gym-agentic-swe-pivot-v4",
                origin="Task Trove",
                family="tool-use",
                tags=("agentic", "multi-turn", "excluded"),
                count=1541,
                notes="No repo in the container; a 9B judge rates one predicted next action.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-cfbench-v4",
                title="laion/nemotron-gym-cfbench-v4",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=468,
                notes=(
                    "Deterministic gate before the judge uses 31 constraint ids outside the IFEval "
                    "registry (tables, heading depth, numbered lists, unique words, ...); only 328 of "
                    "1,478 tasks are gate-able today. Port the gate checks before converting."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-identity-following-v4",
                title="laion/nemotron-gym-identity-following-v4",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=21660,
                notes=(
                    "Persona is NVIDIA's; judge-only. Rewrite with our identity and deterministic name/"
                    "language checks, or skip."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-instruction-following-adversarial-v5",
                title="laion/nemotron-gym-instruction-following-adversarial-v5",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=1000,
                notes="Asks an LLM judge to count exactly five spelling errors.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-instruction-following-citation-v2",
                title="laion/nemotron-gym-instruction-following-citation-v2",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=9033,
                notes="Grades presence of literal marker substrings; never checks the cited content.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-instruction-following-freeform-v2",
                title="laion/nemotron-gym-instruction-following-freeform-v2",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=8869,
                notes="Counts markdown tables and bullets; no content check.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-instruction-following-multiturnchat-v4",
                title="laion/nemotron-gym-instruction-following-multiturnchat-v4",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=1982,
                notes="Required literal format contradicts the demonstrated turns; judge-only.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-inverse-ifeval-v4",
                title="laion/nemotron-gym-inverse-ifeval-v4",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=1000,
                notes="Gate matches against a deliberately broken synthetic reference; then judge.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-knowledge-web-search-mcqa-v2",
                title="laion/nemotron-gym-knowledge-web-search-mcqa-v2",
                origin="Task Trove",
                family="qa-short-answer",
                tags=("agentic", "multi-turn", "excluded"),
                count=2915,
                notes=(
                    "Promises web search but ships no tool. Worth rewriting as a real search-tool env; "
                    "otherwise it is a 3k duplicate of mcqa."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-litmus-bench-v2",
                title="laion/nemotron-gym-litmus-bench-v2",
                origin="Task Trove",
                family="math-answer",
                tags=("agentic", "multi-turn", "excluded"),
                count=5232,
                notes=(
                    "Instruction asks for ((answer)), verifier greps boxed or last number; SMILES tasks "
                    "with no RDKit. Fix format contract and install cheminformatics."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-math-advanced-calculations-v4",
                title="laion/nemotron-gym-math-advanced-calculations-v4",
                origin="Task Trove",
                family="math-answer",
                tags=("agentic", "multi-turn", "excluded"),
                count=5291,
                notes=(
                    "Instruction refers to tools that do not exist and only the last number is graded. "
                    "Ground-truth expression tree is present, so rewrite with a calculator tool and grade "
                    "every subexpression."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-multichallenge-vanilla-v3",
                title="laion/nemotron-gym-multichallenge-vanilla-v3",
                origin="Task Trove",
                family="llm-judge-freeform",
                tags=("agentic", "multi-turn", "excluded"),
                count=1050,
                notes="Single subjective criterion with 'Expected answer: YES' embedded in the judge prompt.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-qa-abstention-v4",
                title="laion/nemotron-gym-qa-abstention-v4",
                origin="Task Trove",
                family="qa-short-answer",
                tags=("agentic", "multi-turn", "excluded"),
                count=3150,
                notes=(
                    "Abstention is never rewarded so the framing is dead, reference leaks into judge "
                    "text, and it duplicates openqa."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-sysbench-v4",
                title="laion/nemotron-gym-sysbench-v4",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn", "excluded"),
                count=1010,
                notes=(
                    "Deterministic gate before the judge uses 31 constraint ids outside the IFEval "
                    "registry (tables, heading depth, numbered lists, unique words, ...); only 328 of "
                    "1,478 tasks are gate-able today. Port the gate checks before converting."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__openswe-tasks-patched-v7-oracle-success",
                title="laion/openswe-tasks-patched-v7-oracle-success",
                origin="Task Trove",
                family="swe-repo",
                tags=("agentic", "multi-turn", "excluded"),
                count=11730,
                notes=(
                    "No FAIL_TO_PASS ids: the v7 verifier scores whichever tests its custom pytest guard "
                    "plugin saw execute, and the repository is cloned by a root-level setup script at "
                    "agent time. Needs its own converter."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__r2egym-patched-full-oracle-v3",
                title="laion/r2egym-patched-full-oracle-v3",
                origin="Task Trove",
                family="swe-repo",
                tags=("agentic", "multi-turn", "excluded"),
                count=2574,
                notes=(
                    "Grades by overlap between test_info.json and expected_output_json rather than by "
                    "pytest node id; not the trusted-paths shape the swe converters handle."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__swegym-tasks-patched-validated-v5",
                title="laion/swegym-tasks-patched-validated-v5",
                origin="Task Trove",
                family="swe-repo",
                tags=("agentic", "multi-turn", "excluded"),
                count=2428,
                notes=(
                    "Image ships no repository: the instruction clones it and runs make init at agent "
                    "time, and the old grader pip-installed requirements again at grading time. Sampled "
                    "oracles fail on missing dependencies and the empty check cannot start."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__toolscale-v4",
                title="laion/toolscale-v4",
                origin="Task Trove",
                family="tool-use",
                tags=("agentic", "multi-turn", "excluded"),
                count=4048,
                notes=(
                    "Good design (offline tool service) but the CLI script embeds the gold calls and "
                    "answer, and the prompt states the conclusion. Move the fixture behind a server and "
                    "strip the success criteria."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__tulu3-sft-personas-math-sandboxes-verified-v3",
                title="laion/tulu3-sft-personas-math-sandboxes-verified-v3",
                origin="Task Trove",
                family="math-answer",
                tags=("agentic", "multi-turn", "excluded"),
                count=9998,
                notes="Easy SFT persona math, gold in plaintext, carries the terminal-bench canary.",
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:R2E-Gym__R2E-Gym-V1",
                title="R2E-Gym/R2E-Gym-V1",
                origin="Task Trove",
                family="swe",
                tags=("agentic", "multi-turn", "license:apache-2.0", "language:python"),
                notes=(
                    "Complete canonical R2E-Gym population with private native image tests and original "
                    "expected-status grading; oracle patches remain separate."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=SourceReference(
                    "Harbor", "ba712d3ff6ade9d6b1cd594af2e81185a5c23a99ccc2115b9b38552c2d4d6ed3", ""
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:SankalpKJ__nemotron-code-oracle-filtered",
                title="SankalpKJ/nemotron-code-oracle-filtered",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "excluded"),
                count=15165,
                notes=(
                    "Only test is the example shown in the prompt. Oracle solutions exist, so generate "
                    "hidden cases by fuzzing inputs through the oracle."
                ),
                dataset=TASKTROVE_INPUT,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:SWE-Gym__SWE-Gym",
                title="SWE-Gym/SWE-Gym",
                origin="Task Trove",
                family="swe",
                tags=("agentic", "multi-turn", "license:mit", "language:python"),
                notes=(
                    "Canonical SWE-Gym tasks with native test selection and grading; oracle patches " "remain separate."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=SourceReference(
                    "Harbor", "33e77858f09c2f6ed51db9fc3d2518afffdc54582d1672b91aa3ae579b84dfdd", ""
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__code",
                title="XiaomiMiMo/MiMo-V2.6-RL-oss__code",
                origin="Task Trove",
                family="swe",
                tags=("agentic", "multi-turn", "license:apache-2.0", "language:unknown"),
                notes=(
                    "Full canonical code population with the complete native setup and grading lifecycle; "
                    "no shipped solutions. Quality review pending at this release. Source-wide quality "
                    "review is pending. Canonical no-op and empty-commit controls on one task matched "
                    "separate direct native runs with reward 0. Four synthetic controls validated adapter "
                    "behavior. These controls do not establish quality across all 2,698 tasks. Image tags "
                    "and dependency resolution remain mutable."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=MIMO_VERIFIER,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__cyber",
                title="XiaomiMiMo/MiMo-V2.6-RL-oss__cyber",
                origin="Task Trove",
                family="cyber",
                tags=("agentic", "multi-turn", "license:apache-2.0", "language:unknown"),
                notes=(
                    "All 1,000 canonical ARVO tasks with the full native SDK, recipes and restricted-user "
                    "runtime. Quality review pending at this release. The audit matches every task "
                    "archive to its original input and pinned native packages, covering all 1,000 tasks. "
                    "A no-op control and a control that only wrote a note on canonical task arvo_35858 "
                    "matched the direct native grader at zero without execution errors. Those controls "
                    "submitted no input and executed no target; positive reproduction and independent "
                    "benchmark-quality review are pending. See [pinned runtime provenance](provenance/"
                    "XiaomiMiMo__MiMo-V2.6-RL-oss__cyber/source.json) for runtime revisions and file "
                    "hashes."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=MIMO_VERIFIER,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__general",
                title="XiaomiMiMo/MiMo-V2.6-RL-oss__general",
                origin="Task Trove",
                family="general",
                tags=("agentic", "multi-turn", "license:apache-2.0", "language:unknown"),
                notes=(
                    "All 989 canonical general rows; 925 private MCP environments and 64 native-registry-"
                    "unavailable terminal tasks. Quality review pending at this release. The audit "
                    "matches every task archive and immutable asset reference to the original inputs and "
                    "pinned packages, covering all 989 tasks. Both no-delivery controls on one canonical "
                    "MCP task measured zero without errors; a benign GLM-5.3 Low endpoint check passed "
                    "separately. Nonempty task deliveries and independent quality review are pending. The "
                    "pinned native registry cannot execute the 64 terminal_bench rows; all are retained "
                    "with that limitation. See [pinned runtime provenance](provenance/XiaomiMiMo__MiMo-"
                    "V2.6-RL-oss__general/source.json) for runtime revisions and file hashes."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=MIMO_VERIFIER,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__music",
                title="XiaomiMiMo/MiMo-V2.6-RL-oss__music",
                origin="Task Trove",
                family="music",
                tags=("agentic", "multi-turn", "license:apache-2.0", "language:zh", "language:en"),
                notes=(
                    "Canonical prompts with the complete native ABC-to-MIDI continuous scorer; no shipped " "solutions."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=MIMO_VERIFIER,
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__webdev",
                title="XiaomiMiMo/MiMo-V2.6-RL-oss__webdev",
                origin="Task Trove",
                family="webdev",
                tags=("agentic", "multi-turn", "license:apache-2.0", "language:unknown"),
                notes=(
                    "All 2,093 canonical webdev tasks with complete native rendering, query and group "
                    "grading sources. Quality review pending at this release. The audit matches every "
                    "task archive to its original input and pinned native packages, covering all 2,093 "
                    "tasks. Canonical-image no-delivery controls measured zero. Four synthetic pages "
                    "completed native rendering, query grading and group finalization using Kimi K3; "
                    "query-fit scores below the upstream threshold of 0.2 produced four measured zeros "
                    "without invalid rows or dropped slices. These controls do not establish benchmark "
                    "quality; independent review is pending. See [pinned runtime provenance](provenance/"
                    "XiaomiMiMo__MiMo-V2.6-RL-oss__webdev/source.json) for runtime revisions and file "
                    "hashes."
                ),
                dataset=TASKTROVE_RELEASE,
                verifier=MIMO_VERIFIER,
            )
        ),
    ]
