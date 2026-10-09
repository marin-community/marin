# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sources retained in the inventory that do not yet have a conversion recipe."""

from dataclasses import replace

from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

TASKTROVE_METADATA = DataSourceMetadata(
    id="",
    name="",
    origin="Task Trove",
    url="https://huggingface.co/datasets/open-athena/task-trove",
    dataset_id="open-athena/task-trove",
    revision="ec049a4fb541ffbe5bbccb803e826563f5718dbf",
    revised_at="2026-10-08T09:34:47.000Z",
    environment="Harbor",
    type="Agentic",
    turns="Multi-turn",
    count_basis="Released Harbor tasks: manifest by_source.converted",
    count_precision="exact",
    count_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb803"
        "e826563f5718dbf/manifest.json"
    ),
    benchmark_basis="Release manifest does not designate benchmarks",
    family_basis="Task Trove release manifest source_verdicts.family",
    family_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb803"
        "e826563f5718dbf/manifest.json"
    ),
    classification_basis="Task Trove tasks run as Agentic interactions in Harbor",
    canonical_url="https://huggingface.co/datasets/open-athena/task-trove",
    provenance_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb8"
        "03e826563f5718dbf/manifest.json"
    ),
    recorded_at="2026-10-08",
)


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:AweAI-Team__CalibForge",
                name="AweAI-Team__CalibForge",
                display_name="AweAI-Team/CalibForge",
                dataset_revision="fb1e75441a94b8bb0ced08acd6b59e711704d70a",
                verifier_revision="fb1e75441a94b8bb0ced08acd6b59e711704d70a",
                family="terminal-agent",
                task_count=5431,
                notes=(
                    "Native Harbor tasks retained with original graders; snapshot-unsafe dependencies "
                    "explicitly accepted."
                ),
                canonical_source="AweAI-Team/CalibForge",
                license=("cc-by-4.0",),
                verification="native-harbor",
                snapshot_safety_basis=(
                    "Native environments reference mutable Docker image tags; native test "
                    "scripts download bootstrap tools and dependencies at execution time. "
                    "Accepted without making the source snapshot-safe."
                ),
                upstream_repository="AweAI-Team/CalibForge",
                upstream_url="https://huggingface.co/datasets/AweAI-Team/CalibForge",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=5431,
                modes=("native-harbor",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:DCAgent__exp_rle_adversarial-v6",
                name="DCAgent__exp_rle_adversarial-v6",
                display_name="DCAgent/exp_rle_adversarial-v6",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "The legacy pytest grader performs source-specific Django discovery and loads "
                    "extra plugins. Ten sampled empty and trivial submissions failed, but the source "
                    "has no oracle and needs a custom environment adapter."
                ),
                canonical_source="DCAgent/exp_rle_adversarial-v6",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="DCAgent/exp_rle_adversarial-v6",
                upstream_url="https://huggingface.co/datasets/DCAgent/exp_rle_adversarial-v6",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2726,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:DCAgent__exp_rpt_nemotron-cpp",
                name="DCAgent__exp_rpt_nemotron-cpp",
                display_name="DCAgent/exp_rpt_nemotron-cpp",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "GoogleTest tasks without shipped oracles. Sampled empty and trivial submissions "
                    "failed, but some tests contain the reference implementation and others require "
                    "an uninstalled doctest dependency."
                ),
                canonical_source="DCAgent/exp_rpt_nemotron-cpp",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="DCAgent/exp_rpt_nemotron-cpp",
                upstream_url="https://huggingface.co/datasets/DCAgent/exp_rpt_nemotron-cpp",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=4196,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:DCAgent__inferredbugs-sandboxes-verifier",
                name="DCAgent__inferredbugs-sandboxes-verifier",
                display_name="DCAgent/inferredbugs-sandboxes-verifier",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="swe-repo",
                task_count=0,
                status="Excluded",
                notes=(
                    "Never compiles or runs; regex on the rewritten method body with guards that "
                    "accept either polarity."
                ),
                canonical_source="DCAgent/inferredbugs-sandboxes-verifier",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="DCAgent/inferredbugs-sandboxes-verifier",
                upstream_url="https://huggingface.co/datasets/DCAgent/inferredbugs-sandboxes-verifier",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=9659,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:DCAgent__mix_h4_binary_easy",
                name="DCAgent__mix_h4_binary_easy",
                display_name="DCAgent/mix_h4_binary_easy",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "Mixed verifier shapes: eight of 10 sampled empty and trivial checks timed out, "
                    "and the crosscodeeval slice only checks that an import succeeds."
                ),
                canonical_source="DCAgent/mix_h4_binary_easy",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="DCAgent/mix_h4_binary_easy",
                upstream_url="https://huggingface.co/datasets/DCAgent/mix_h4_binary_easy",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1996,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:DCAgent__selfinstruct-naive-sandboxes-2-verified-v3",
                name="DCAgent__selfinstruct-naive-sandboxes-2-verified-v3",
                display_name="DCAgent/selfinstruct-naive-sandboxes-2-verified-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="shell-cmd",
                task_count=0,
                status="Excluded",
                notes=(
                    "Per-task LLM-written test_state.py with loose file discovery and dead code. Task "
                    "ideas are usable; regenerate verifiers with an oracle/no-op gate."
                ),
                canonical_source="DCAgent/selfinstruct-naive-sandboxes-2-verified-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="DCAgent/selfinstruct-naive-sandboxes-2-verified-v3",
                upstream_url="https://huggingface.co/datasets/DCAgent/selfinstruct-naive-sandboxes-2-verified-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=6665,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:GAIR__OpenSWE__openswe_oss",
                name="GAIR__OpenSWE__openswe_oss",
                display_name="GAIR/OpenSWE__openswe_oss",
                dataset_revision="a8db93af5335df2c8baac0cd1ff367e4d475d3d7",
                verifier_revision="f332529660d81fd64eb0ecf73b2bbf75241c36792fef7215de07c98e8ebeded3",
                family="swe",
                task_count=36884,
                notes=(
                    "Complete canonical openswe_oss population with native build/evaluation commands "
                    "and private full-harness grading. Shipped OSS gold remains in a separate "
                    "solution archive; the other configuration ships no oracle. Native failures are "
                    "retained for quality review. Quality review pending at this release. Independent "
                    "quality review is incomplete. All 36,884 archives passed the static audit; "
                    "native runtime controls cover one selected OSS task. A separate "
                    "OTHER-configuration control reproduced a native grading false positive in the "
                    "shared harness. That is cross-configuration evidence, not an executed OSS bypass "
                    "or a prevalence estimate. Check the Atlas for current review status before "
                    "training use."
                ),
                canonical_source="GAIR/OpenSWE__openswe_oss",
                license=("other",),
                verification="native-openswe",
                snapshot_safety_basis=(
                    "Canonical Dockerfiles, the dataset and native grader code are pinned. "
                    "Native builds use mutable container tags and unpinned package downloads. "
                    "Repository context and grader images also use mutable tags."
                ),
                upstream_repository="GAIR/OpenSWE",
                upstream_configuration="openswe_oss",
                upstream_url="https://huggingface.co/datasets/GAIR/OpenSWE",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=36884,
                languages=("unknown",),
                modes=("native-openswe",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:GAIR__OpenSWE__openswe_other",
                name="GAIR__OpenSWE__openswe_other",
                display_name="GAIR/OpenSWE__openswe_other",
                dataset_revision="a8db93af5335df2c8baac0cd1ff367e4d475d3d7",
                verifier_revision="d8923833284d2018e63fe33775411487dde8635092d8f20a28c0f8b624917f25",
                family="swe",
                task_count=8436,
                notes=(
                    "Complete canonical openswe_other population with native build/evaluation "
                    "commands and private full-harness grading. Shipped OSS gold remains in a "
                    "separate solution archive; the other configuration ships no oracle. Native "
                    "failures are retained for quality review. Quality review pending at this "
                    "release. Independent quality review is incomplete. All 8,436 archives passed the "
                    "static audit; native runtime controls cover one selected OTHER task. An "
                    "incorrect control received full native reward despite a canonical test failure. "
                    "The unchanged native result is preserved; this does not estimate source-wide "
                    "prevalence. Check the Atlas for current review status before training use."
                ),
                canonical_source="GAIR/OpenSWE__openswe_other",
                license=("other",),
                verification="native-openswe",
                snapshot_safety_basis=(
                    "Canonical Dockerfiles, the dataset and native grader code are pinned. "
                    "Native builds use mutable container tags and unpinned package downloads. "
                    "Repository context and grader images also use mutable tags."
                ),
                upstream_repository="GAIR/OpenSWE",
                upstream_configuration="openswe_other",
                upstream_url="https://huggingface.co/datasets/GAIR/OpenSWE",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=8436,
                languages=("unknown",),
                modes=("native-openswe",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__codeelo-v2",
                name="laion__codeelo-v2",
                display_name="laion/codeelo-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="competitive-programming",
                task_count=0,
                status="Excluded",
                notes="Byte-identical generator to codeforces-v3 at 500 rows; merge, do not keep separately.",
                canonical_source="laion/codeelo-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/codeelo-v2",
                upstream_url="https://huggingface.co/datasets/laion/codeelo-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=500,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_bugsinpy-v4",
                name="laion__exp_rpt_bugsinpy-v4",
                display_name="laion/exp_rpt_bugsinpy-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="swe-repo",
                task_count=0,
                status="Excluded",
                notes=(
                    "LLM-synthesized tests against a single-file stub, with assert True placeholders. "
                    "Rewrite against the real BugsInPy project suites."
                ),
                canonical_source="laion/exp_rpt_bugsinpy-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_bugsinpy-v4",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_bugsinpy-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=479,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_codenet-python-v4",
                name="laion__exp_rpt_codenet-python-v4",
                display_name="laion/exp_rpt_codenet-python-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="competitive-programming",
                task_count=0,
                status="Excluded",
                notes=(
                    "Only 3 hidden cases and whitespace-collapsing compare. Oracle present; "
                    "regenerate 20+ cases per task."
                ),
                canonical_source="laion/exp_rpt_codenet-python-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_codenet-python-v4",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_codenet-python-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=6975,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_crosscodeeval-csharp-v4",
                name="laion__exp_rpt_crosscodeeval-csharp-v4",
                display_name="laion/exp_rpt_crosscodeeval-csharp-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="code-completion",
                task_count=0,
                status="Excluded",
                notes="0.25 reward for any identifier-shaped output; instruction coaches the hack.",
                canonical_source="laion/exp_rpt_crosscodeeval-csharp-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_crosscodeeval-csharp-v4",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_crosscodeeval-csharp-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1768,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_crosscodeeval-java-v3",
                name="laion__exp_rpt_crosscodeeval-java-v3",
                display_name="laion/exp_rpt_crosscodeeval-java-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="code-completion",
                task_count=0,
                status="Excluded",
                notes="Exact string match on a single line completion. Not agentic, no execution.",
                canonical_source="laion/exp_rpt_crosscodeeval-java-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_crosscodeeval-java-v3",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_crosscodeeval-java-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2139,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_crosscodeeval-python-v2",
                name="laion__exp_rpt_crosscodeeval-python-v2",
                display_name="laion/exp_rpt_crosscodeeval-python-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="code-completion",
                task_count=0,
                status="Excluded",
                notes="0.25 for any non-empty output; instruction discloses the tiers.",
                canonical_source="laion/exp_rpt_crosscodeeval-python-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_crosscodeeval-python-v2",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_crosscodeeval-python-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=500,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_crosscodeeval-typescript-v2",
                name="laion__exp_rpt_crosscodeeval-typescript-v2",
                display_name="laion/exp_rpt_crosscodeeval-typescript-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="code-completion",
                task_count=0,
                status="Excluded",
                notes="Re-skin of the Python variant with the same free 0.25 tier; metadata still says python.",
                canonical_source="laion/exp_rpt_crosscodeeval-typescript-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_crosscodeeval-typescript-v2",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_crosscodeeval-typescript-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=3356,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_ghactions-v3",
                name="laion__exp_rpt_ghactions-v3",
                display_name="laion/exp_rpt_ghactions-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="ci-workflow",
                task_count=0,
                status="Excluded",
                notes="Instruction lists every job and step verbatim; workflow is never executed. Transcription.",
                canonical_source="laion/exp_rpt_ghactions-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_ghactions-v3",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_ghactions-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=9930,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_methods2test-large-v4",
                name="laion__exp_rpt_methods2test-large-v4",
                display_name="laion/exp_rpt_methods2test-large-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "All 10 sampled shipped Java oracles failed because Maven could not resolve its "
                    "plugins offline. Recovering the source requires rebuilding the Java grading "
                    "environment."
                ),
                canonical_source="laion/exp_rpt_methods2test-large-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_methods2test-large-v4",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_methods2test-large-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1194,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_nemotron-junit-v6",
                name="laion__exp_rpt_nemotron-junit-v6",
                display_name="laion/exp_rpt_nemotron-junit-v6",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes="20% of sampled tasks contain unconditional fail() stubs the verifier restores.",
                canonical_source="laion/exp_rpt_nemotron-junit-v6",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_nemotron-junit-v6",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_nemotron-junit-v6",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=447,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_scaffold-v3",
                name="laion__exp_rpt_scaffold-v3",
                display_name="laion/exp_rpt_scaffold-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="code-generation",
                task_count=0,
                status="Excluded",
                notes="LLM-synthesized stub-filling toys (TypeScript formatter shim, Flask hello page). Kata-grade.",
                canonical_source="laion/exp_rpt_scaffold-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_scaffold-v3",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_scaffold-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=3121,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-cpp-v4",
                name="laion__exp_rpt_stack-cpp-v4",
                display_name="laion/exp_rpt_stack-cpp-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "Tests are lifted from real repositories with the repository stripped: sampled "
                    "tasks include headers and data files that do not exist in the image, and one "
                    "pastes the reference Solution class inside the test."
                ),
                canonical_source="laion/exp_rpt_stack-cpp-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-cpp-v4",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-cpp-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=7878,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-dockerfile-gpt5mini-v7",
                name="laion__exp_rpt_stack-dockerfile-gpt5mini-v7",
                display_name="laion/exp_rpt_stack-dockerfile-gpt5mini-v7",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="tool-use",
                task_count=0,
                status="Excluded",
                notes=(
                    "587 rows of gpt-5-mini-written per-task test scripts whose instructions describe "
                    "containers that do not exist."
                ),
                canonical_source="laion/exp_rpt_stack-dockerfile-gpt5mini-v7",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-dockerfile-gpt5mini-v7",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-dockerfile-gpt5mini-v7",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=587,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-go-v5",
                name="laion__exp_rpt_stack-go-v5",
                display_name="laion/exp_rpt_stack-go-v5",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "Tests import packages from the original repository (bridgr/internal/..., "
                    "gosnowflake internals) that are not in the task, so most tasks are unsolvable as "
                    "specified."
                ),
                canonical_source="laion/exp_rpt_stack-go-v5",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-go-v5",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-go-v5",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2275,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-jest-v5",
                name="laion__exp_rpt_stack-jest-v5",
                display_name="laion/exp_rpt_stack-jest-v5",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes="Spy-call contracts against a 90-package global npm image; 424 rows.",
                canonical_source="laion/exp_rpt_stack-jest-v5",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-jest-v5",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-jest-v5",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=424,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-junit-v6",
                name="laion__exp_rpt_stack-junit-v6",
                display_name="laion/exp_rpt_stack-junit-v6",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "JUnit grading is real but every instruction cites a test path that does not "
                    "exist and scan-class-path counts any test class."
                ),
                canonical_source="laion/exp_rpt_stack-junit-v6",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-junit-v6",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-junit-v6",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=843,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-php-large-v9",
                name="laion__exp_rpt_stack-php-large-v9",
                display_name="laion/exp_rpt_stack-php-large-v9",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes="Fail-open exit paths, regex class discovery, 462 rows.",
                canonical_source="laion/exp_rpt_stack-php-large-v9",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-php-large-v9",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-php-large-v9",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=462,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-pytest-large-v3",
                name="laion__exp_rpt_stack-pytest-large-v3",
                display_name="laion/exp_rpt_stack-pytest-large-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "Stripped-repository pytest tasks without shipped oracles. Two of 10 sampled "
                    "empty and trivial checks timed out, and sampled tests include truthiness-only "
                    "assertions that weak stubs can satisfy."
                ),
                canonical_source="laion/exp_rpt_stack-pytest-large-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-pytest-large-v3",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-pytest-large-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1782,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__exp_rpt_stack-rspec-v4",
                name="laion__exp_rpt_stack-rspec-v4",
                display_name="laion/exp_rpt_stack-rspec-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "Real Ruby test files but gems are never installed and some tasks are unsolvable "
                    "offline. Bake gems and drop tasks that fail the oracle gate."
                ),
                canonical_source="laion/exp_rpt_stack-rspec-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/exp_rpt_stack-rspec-v4",
                upstream_url="https://huggingface.co/datasets/laion/exp_rpt_stack-rspec-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=8860,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__magicoder-v4",
                name="laion__magicoder-v4",
                display_name="laion/magicoder-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="llm-judge-freeform",
                task_count=0,
                status="Excluded",
                notes=(
                    "Judge-only over a bundle of every file under /app collected by "
                    "tests/collect_submission.py; the judge mode grades one answer file, so the "
                    "bundle shape needs its own converter. Vague refactor prompts, nothing executed."
                ),
                canonical_source="laion/magicoder-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/magicoder-v4",
                upstream_url="https://huggingface.co/datasets/laion/magicoder-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=4096,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__mix_h10_reward_proportional-v2",
                name="laion__mix_h10_reward_proportional-v2",
                display_name="laion/mix_h10_reward_proportional-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "Four of 10 sampled trivial submissions received full credit because the "
                    "codereval slice tests a local mock rather than the solution."
                ),
                canonical_source="laion/mix_h10_reward_proportional-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/mix_h10_reward_proportional-v2",
                upstream_url="https://huggingface.co/datasets/laion/mix_h10_reward_proportional-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2858,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__mix_h11_single_skill_only-v2",
                name="laion__mix_h11_single_skill_only-v2",
                display_name="laion/mix_h11_single_skill_only-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "Mixed verifier shapes: eight of 10 sampled empty and trivial checks timed out; "
                    "the source also contains content-free crosscodeeval slices and syntactically "
                    "invalid tests."
                ),
                canonical_source="laion/mix_h11_single_skill_only-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/mix_h11_single_skill_only-v2",
                upstream_url="https://huggingface.co/datasets/laion/mix_h11_single_skill_only-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2859,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__mix_h8_original_tests-v2",
                name="laion__mix_h8_original_tests-v2",
                display_name="laion/mix_h8_original_tests-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="unit-test-gen",
                task_count=0,
                status="Excluded",
                notes=(
                    "All 10 sampled empty and trivial checks timed out, and prior inspection found "
                    "import-only test files in three of 10 tasks."
                ),
                canonical_source="laion/mix_h8_original_tests-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/mix_h8_original_tests-v2",
                upstream_url="https://huggingface.co/datasets/laion/mix_h8_original_tests-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2848,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-agentic-function-calling-pivot-v3",
                name="laion__nemotron-gym-agentic-function-calling-pivot-v3",
                display_name="laion/nemotron-gym-agentic-function-calling-pivot-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="tool-use",
                task_count=0,
                status="Excluded",
                notes=(
                    "Predict-the-next-call from a frozen transcript with exact key-set match. The "
                    "only tool-call data here; rewrite into executable mock-tool envs built from the "
                    "transcripts."
                ),
                canonical_source="laion/nemotron-gym-agentic-function-calling-pivot-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-agentic-function-calling-pivot-v3",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-agentic-function-calling-pivot-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=9579,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-agentic-indirect-prompt-injection-v3",
                name="laion__nemotron-gym-agentic-indirect-prompt-injection-v3",
                display_name="laion/nemotron-gym-agentic-indirect-prompt-injection-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="prompt-injection",
                task_count=0,
                status="Excluded",
                notes=(
                    "Five of five reviewed rows use the same negative-only checker, which rewards a "
                    "generic reply or unrelated action without validating the required safe "
                    "continuation."
                ),
                canonical_source="laion/nemotron-gym-agentic-indirect-prompt-injection-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-agentic-indirect-prompt-injection-v3",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-agentic-indirect-prompt-injection-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1272,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-agentic-swe-pivot-v4",
                name="laion__nemotron-gym-agentic-swe-pivot-v4",
                display_name="laion/nemotron-gym-agentic-swe-pivot-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="tool-use",
                task_count=0,
                status="Excluded",
                notes="No repo in the container; a 9B judge rates one predicted next action.",
                canonical_source="laion/nemotron-gym-agentic-swe-pivot-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-agentic-swe-pivot-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-agentic-swe-pivot-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1541,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-cfbench-v4",
                name="laion__nemotron-gym-cfbench-v4",
                display_name="laion/nemotron-gym-cfbench-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes=(
                    "Deterministic gate before the judge uses 31 constraint ids outside the IFEval "
                    "registry (tables, heading depth, numbered lists, unique words, ...); only 328 of "
                    "1,478 tasks are gate-able today. Port the gate checks before converting."
                ),
                canonical_source="laion/nemotron-gym-cfbench-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-cfbench-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-cfbench-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=468,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-identity-following-v4",
                name="laion__nemotron-gym-identity-following-v4",
                display_name="laion/nemotron-gym-identity-following-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes=(
                    "Persona is NVIDIA's; judge-only. Rewrite with our identity and deterministic "
                    "name/language checks, or skip."
                ),
                canonical_source="laion/nemotron-gym-identity-following-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-identity-following-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-identity-following-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=21660,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-instruction-following-adversarial-v5",
                name="laion__nemotron-gym-instruction-following-adversarial-v5",
                display_name="laion/nemotron-gym-instruction-following-adversarial-v5",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes="Asks an LLM judge to count exactly five spelling errors.",
                canonical_source="laion/nemotron-gym-instruction-following-adversarial-v5",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-instruction-following-adversarial-v5",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-instruction-following-adversarial-v5",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1000,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-instruction-following-citation-v2",
                name="laion__nemotron-gym-instruction-following-citation-v2",
                display_name="laion/nemotron-gym-instruction-following-citation-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes="Grades presence of literal marker substrings; never checks the cited content.",
                canonical_source="laion/nemotron-gym-instruction-following-citation-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-instruction-following-citation-v2",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-instruction-following-citation-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=9033,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-instruction-following-freeform-v2",
                name="laion__nemotron-gym-instruction-following-freeform-v2",
                display_name="laion/nemotron-gym-instruction-following-freeform-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes="Counts markdown tables and bullets; no content check.",
                canonical_source="laion/nemotron-gym-instruction-following-freeform-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-instruction-following-freeform-v2",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-instruction-following-freeform-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=8869,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-instruction-following-multiturnchat-v4",
                name="laion__nemotron-gym-instruction-following-multiturnchat-v4",
                display_name="laion/nemotron-gym-instruction-following-multiturnchat-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes="Required literal format contradicts the demonstrated turns; judge-only.",
                canonical_source="laion/nemotron-gym-instruction-following-multiturnchat-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-instruction-following-multiturnchat-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-instruction-following-multiturnchat-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1982,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-inverse-ifeval-v4",
                name="laion__nemotron-gym-inverse-ifeval-v4",
                display_name="laion/nemotron-gym-inverse-ifeval-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes="Gate matches against a deliberately broken synthetic reference; then judge.",
                canonical_source="laion/nemotron-gym-inverse-ifeval-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-inverse-ifeval-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-inverse-ifeval-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1000,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-knowledge-web-search-mcqa-v2",
                name="laion__nemotron-gym-knowledge-web-search-mcqa-v2",
                display_name="laion/nemotron-gym-knowledge-web-search-mcqa-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="qa-short-answer",
                task_count=0,
                status="Excluded",
                notes=(
                    "Promises web search but ships no tool. Worth rewriting as a real search-tool "
                    "env; otherwise it is a 3k duplicate of mcqa."
                ),
                canonical_source="laion/nemotron-gym-knowledge-web-search-mcqa-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-knowledge-web-search-mcqa-v2",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-knowledge-web-search-mcqa-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2915,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-litmus-bench-v2",
                name="laion__nemotron-gym-litmus-bench-v2",
                display_name="laion/nemotron-gym-litmus-bench-v2",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="math-answer",
                task_count=0,
                status="Excluded",
                notes=(
                    "Instruction asks for ((answer)), verifier greps boxed or last number; SMILES "
                    "tasks with no RDKit. Fix format contract and install cheminformatics."
                ),
                canonical_source="laion/nemotron-gym-litmus-bench-v2",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-litmus-bench-v2",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-litmus-bench-v2",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=5232,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-math-advanced-calculations-v4",
                name="laion__nemotron-gym-math-advanced-calculations-v4",
                display_name="laion/nemotron-gym-math-advanced-calculations-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="math-answer",
                task_count=0,
                status="Excluded",
                notes=(
                    "Instruction refers to tools that do not exist and only the last number is "
                    "graded. Ground-truth expression tree is present, so rewrite with a calculator "
                    "tool and grade every subexpression."
                ),
                canonical_source="laion/nemotron-gym-math-advanced-calculations-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-math-advanced-calculations-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-math-advanced-calculations-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=5291,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-multichallenge-vanilla-v3",
                name="laion__nemotron-gym-multichallenge-vanilla-v3",
                display_name="laion/nemotron-gym-multichallenge-vanilla-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="llm-judge-freeform",
                task_count=0,
                status="Excluded",
                notes="Single subjective criterion with 'Expected answer: YES' embedded in the judge prompt.",
                canonical_source="laion/nemotron-gym-multichallenge-vanilla-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-multichallenge-vanilla-v3",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-multichallenge-vanilla-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1050,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-qa-abstention-v4",
                name="laion__nemotron-gym-qa-abstention-v4",
                display_name="laion/nemotron-gym-qa-abstention-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="qa-short-answer",
                task_count=0,
                status="Excluded",
                notes=(
                    "Abstention is never rewarded so the framing is dead, reference leaks into judge "
                    "text, and it duplicates openqa."
                ),
                canonical_source="laion/nemotron-gym-qa-abstention-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-qa-abstention-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-qa-abstention-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=3150,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-sysbench-v4",
                name="laion__nemotron-gym-sysbench-v4",
                display_name="laion/nemotron-gym-sysbench-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="instruction-following",
                task_count=0,
                status="Excluded",
                notes=(
                    "Deterministic gate before the judge uses 31 constraint ids outside the IFEval "
                    "registry (tables, heading depth, numbered lists, unique words, ...); only 328 of "
                    "1,478 tasks are gate-able today. Port the gate checks before converting."
                ),
                canonical_source="laion/nemotron-gym-sysbench-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/nemotron-gym-sysbench-v4",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-sysbench-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=1010,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__openswe-tasks-patched-v7-oracle-success",
                name="laion__openswe-tasks-patched-v7-oracle-success",
                display_name="laion/openswe-tasks-patched-v7-oracle-success",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="swe-repo",
                task_count=0,
                status="Excluded",
                notes=(
                    "No FAIL_TO_PASS ids: the v7 verifier scores whichever tests its custom pytest "
                    "guard plugin saw execute, and the repository is cloned by a root-level setup "
                    "script at agent time. Needs its own converter."
                ),
                canonical_source="laion/openswe-tasks-patched-v7-oracle-success",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/openswe-tasks-patched-v7-oracle-success",
                upstream_url="https://huggingface.co/datasets/laion/openswe-tasks-patched-v7-oracle-success",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=11730,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__r2egym-patched-full-oracle-v3",
                name="laion__r2egym-patched-full-oracle-v3",
                display_name="laion/r2egym-patched-full-oracle-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="swe-repo",
                task_count=0,
                status="Excluded",
                notes=(
                    "Grades by overlap between test_info.json and expected_output_json rather than by "
                    "pytest node id; not the trusted-paths shape the swe converters handle."
                ),
                canonical_source="laion/r2egym-patched-full-oracle-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/r2egym-patched-full-oracle-v3",
                upstream_url="https://huggingface.co/datasets/laion/r2egym-patched-full-oracle-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2574,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__swegym-tasks-patched-validated-v5",
                name="laion__swegym-tasks-patched-validated-v5",
                display_name="laion/swegym-tasks-patched-validated-v5",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="swe-repo",
                task_count=0,
                status="Excluded",
                notes=(
                    "Image ships no repository: the instruction clones it and runs make init at agent "
                    "time, and the old grader pip-installed requirements again at grading time. "
                    "Sampled oracles fail on missing dependencies and the empty check cannot start."
                ),
                canonical_source="laion/swegym-tasks-patched-validated-v5",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/swegym-tasks-patched-validated-v5",
                upstream_url="https://huggingface.co/datasets/laion/swegym-tasks-patched-validated-v5",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=2428,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__toolscale-v4",
                name="laion__toolscale-v4",
                display_name="laion/toolscale-v4",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="tool-use",
                task_count=0,
                status="Excluded",
                notes=(
                    "Good design (offline tool service) but the CLI script embeds the gold calls and "
                    "answer, and the prompt states the conclusion. Move the fixture behind a server "
                    "and strip the success criteria."
                ),
                canonical_source="laion/toolscale-v4",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/toolscale-v4",
                upstream_url="https://huggingface.co/datasets/laion/toolscale-v4",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=4048,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__tulu3-sft-personas-math-sandboxes-verified-v3",
                name="laion__tulu3-sft-personas-math-sandboxes-verified-v3",
                display_name="laion/tulu3-sft-personas-math-sandboxes-verified-v3",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="math-answer",
                task_count=0,
                status="Excluded",
                notes="Easy SFT persona math, gold in plaintext, carries the terminal-bench canary.",
                canonical_source="laion/tulu3-sft-personas-math-sandboxes-verified-v3",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="laion/tulu3-sft-personas-math-sandboxes-verified-v3",
                upstream_url="https://huggingface.co/datasets/laion/tulu3-sft-personas-math-sandboxes-verified-v3",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=9998,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:R2E-Gym__R2E-Gym-V1",
                name="R2E-Gym__R2E-Gym-V1",
                display_name="R2E-Gym/R2E-Gym-V1",
                dataset_revision="903d405799ac435061c41e72260c81ca5100f964",
                verifier_revision="ba712d3ff6ade9d6b1cd594af2e81185a5c23a99ccc2115b9b38552c2d4d6ed3",
                family="swe",
                task_count=8101,
                notes=(
                    "Complete canonical R2E-Gym population with private native image tests and "
                    "original expected-status grading; oracle patches remain separate."
                ),
                canonical_source="R2E-Gym/R2E-Gym-V1",
                license=("apache-2.0",),
                verification="native-r2egym",
                snapshot_safety_basis=(
                    "Native per-task Docker image tags are mutable, and native setup installs "
                    "chardet. The private bridge installs a separate Python 3.12.11 "
                    "interpreter through a mutable uv image tag. Dataset, upstream "
                    "parser/reward method and bridge code identities are recorded."
                ),
                upstream_repository="R2E-Gym/R2E-Gym-V1",
                upstream_url="https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=8101,
                languages=("python",),
                modes=("native-r2egym",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:SankalpKJ__nemotron-code-oracle-filtered",
                name="SankalpKJ__nemotron-code-oracle-filtered",
                display_name="SankalpKJ/nemotron-code-oracle-filtered",
                dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
                verifier_revision=None,
                family="competitive-programming",
                task_count=0,
                status="Excluded",
                notes=(
                    "Only test is the example shown in the prompt. Oracle solutions exist, so "
                    "generate hidden cases by fuzzing inputs through the oracle."
                ),
                canonical_source="SankalpKJ/nemotron-code-oracle-filtered",
                snapshot_safe=True,
                snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
                upstream_repository="SankalpKJ/nemotron-code-oracle-filtered",
                upstream_url="https://huggingface.co/datasets/SankalpKJ/nemotron-code-oracle-filtered",
                upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
                input_count=15165,
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:SWE-Gym__SWE-Gym",
                name="SWE-Gym__SWE-Gym",
                display_name="SWE-Gym/SWE-Gym",
                dataset_revision="bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
                verifier_revision="33e77858f09c2f6ed51db9fc3d2518afffdc54582d1672b91aa3ae579b84dfdd",
                family="swe",
                task_count=2438,
                notes="Canonical SWE-Gym tasks with native test selection and grading; oracle patches remain separate.",
                canonical_source="SWE-Gym/SWE-Gym",
                license=("mit",),
                verification="native-swegym",
                snapshot_safety_basis=(
                    "Native environment builds download unpinned packages and use mutable "
                    "Ubuntu and uv image tags. The native evaluation script can reinstall "
                    "project dependencies. Dataset and harness code are pinned."
                ),
                upstream_repository="SWE-Gym/SWE-Gym",
                upstream_url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=2438,
                languages=("python",),
                modes=("native-swegym",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__code",
                name="XiaomiMiMo__MiMo-V2.6-RL-oss__code",
                display_name="XiaomiMiMo/MiMo-V2.6-RL-oss__code",
                dataset_revision="639865fd3374018d6cb29b9fb82dd531406fcf5f",
                verifier_revision="a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
                family="swe",
                task_count=2698,
                notes=(
                    "Full canonical code population with the complete native setup and grading "
                    "lifecycle; no shipped solutions. Quality review pending at this release. "
                    "Source-wide quality review is pending. Canonical no-op and empty-commit controls "
                    "on one task matched separate direct native runs with reward 0. Four synthetic "
                    "controls validated adapter behavior. These controls do not establish quality "
                    "across all 2,698 tasks. Image tags and dependency resolution remain mutable."
                ),
                canonical_source="XiaomiMiMo/MiMo-V2.6-RL-oss__code",
                license=("apache-2.0",),
                verification="native-mimo-code",
                snapshot_safety_basis=(
                    "The canonical image mapping uses mutable Docker Hub tags; runtime " "dependencies are unpinned."
                ),
                upstream_repository="XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_configuration="code",
                upstream_url="https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=2698,
                languages=("unknown",),
                modes=("native-mimo-code",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__cyber",
                name="XiaomiMiMo__MiMo-V2.6-RL-oss__cyber",
                display_name="XiaomiMiMo/MiMo-V2.6-RL-oss__cyber",
                dataset_revision="639865fd3374018d6cb29b9fb82dd531406fcf5f",
                verifier_revision="a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
                family="cyber",
                task_count=1000,
                notes=(
                    "All 1,000 canonical ARVO tasks with the full native SDK, recipes and "
                    "restricted-user runtime. Quality review pending at this release. The audit "
                    "matches every task archive to its original input and pinned native packages, "
                    "covering all 1,000 tasks. A no-op control and a control that only wrote a note "
                    "on canonical task arvo_35858 matched the direct native grader at zero without "
                    "execution errors. Those controls submitted no input and executed no target; "
                    "positive reproduction and independent benchmark-quality review are pending. See "
                    "[pinned runtime provenance](provenance/XiaomiMiMo__MiMo-V2.6-RL-oss__cyber/source"
                    ".json) for runtime revisions and file hashes."
                ),
                canonical_source="XiaomiMiMo/MiMo-V2.6-RL-oss__cyber",
                license=("apache-2.0",),
                verification="native-mimo-cyber",
                snapshot_safety_basis=(
                    "The canonical Docker Hub tags and native runtime dependency resolution " "are mutable."
                ),
                upstream_repository="XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_configuration="cyber",
                upstream_url="https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=1000,
                languages=("unknown",),
                modes=("native-mimo-cyber",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__general",
                name="XiaomiMiMo__MiMo-V2.6-RL-oss__general",
                display_name="XiaomiMiMo/MiMo-V2.6-RL-oss__general",
                dataset_revision="639865fd3374018d6cb29b9fb82dd531406fcf5f",
                verifier_revision="a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
                family="general",
                task_count=989,
                notes=(
                    "All 989 canonical general rows; 925 private MCP environments and 64 "
                    "native-registry-unavailable terminal tasks. Quality review pending at this "
                    "release. The audit matches every task archive and immutable asset reference to "
                    "the original inputs and pinned packages, covering all 989 tasks. Both "
                    "no-delivery controls on one canonical MCP task measured zero without errors; a "
                    "benign GLM-5.3 Low endpoint check passed separately. Nonempty task deliveries "
                    "and independent quality review are pending. The pinned native registry cannot "
                    "execute the 64 terminal_bench rows; all are retained with that limitation. See "
                    "[pinned runtime provenance](provenance/XiaomiMiMo__MiMo-V2.6-RL-oss__general/sour"
                    "ce.json) for runtime revisions and file hashes."
                ),
                canonical_source="XiaomiMiMo/MiMo-V2.6-RL-oss__general",
                license=("apache-2.0",),
                verification="native-mimo-general",
                snapshot_safety_basis=(
                    "Task assets bind immutable upstream file hashes. Canonical image tags, "
                    "dependency resolution and the required hosted judge's identity remain "
                    "mutable."
                ),
                upstream_repository="XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_configuration="general",
                upstream_url="https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=989,
                languages=("unknown",),
                modes=("native-mimo-general",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__music",
                name="XiaomiMiMo__MiMo-V2.6-RL-oss__music",
                display_name="XiaomiMiMo/MiMo-V2.6-RL-oss__music",
                dataset_revision="639865fd3374018d6cb29b9fb82dd531406fcf5f",
                verifier_revision="a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
                family="music",
                task_count=1000,
                notes="Canonical prompts with the complete native ABC-to-MIDI continuous scorer; no shipped solutions.",
                canonical_source="XiaomiMiMo/MiMo-V2.6-RL-oss__music",
                license=("apache-2.0",),
                verification="native-mimo-music",
                snapshot_safety_basis="The bridge uses a mutable Python image tag and unpinned Debian abcmidi packages.",
                upstream_repository="XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_configuration="music",
                upstream_url="https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=1000,
                languages=("zh", "en"),
                modes=("native-mimo-music",),
            )
        ),
        RlDataSource(
            metadata=replace(
                TASKTROVE_METADATA,
                id="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__webdev",
                name="XiaomiMiMo__MiMo-V2.6-RL-oss__webdev",
                display_name="XiaomiMiMo/MiMo-V2.6-RL-oss__webdev",
                dataset_revision="639865fd3374018d6cb29b9fb82dd531406fcf5f",
                verifier_revision="a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
                family="webdev",
                task_count=2093,
                notes=(
                    "All 2,093 canonical webdev tasks with complete native rendering, query and group "
                    "grading sources. Quality review pending at this release. The audit matches every "
                    "task archive to its original input and pinned native packages, covering all "
                    "2,093 tasks. Canonical-image no-delivery controls measured zero. Four synthetic "
                    "pages completed native rendering, query grading and group finalization using "
                    "Kimi K3; query-fit scores below the upstream threshold of 0.2 produced four "
                    "measured zeros without invalid rows or dropped slices. These controls do not "
                    "establish benchmark quality; independent review is pending. See [pinned runtime "
                    "provenance](provenance/XiaomiMiMo__MiMo-V2.6-RL-oss__webdev/source.json) for "
                    "runtime revisions and file hashes."
                ),
                canonical_source="XiaomiMiMo/MiMo-V2.6-RL-oss__webdev",
                license=("apache-2.0",),
                verification="native-mimo-webdev",
                snapshot_safety_basis=(
                    "The canonical image tag, runtime dependency resolution and externally "
                    "configured vision judges are mutable."
                ),
                upstream_repository="XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_configuration="webdev",
                upstream_url="https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss",
                upstream_link_basis="Task Trove release manifest source_details.upstream_repository",
                input_count=2093,
                languages=("unknown",),
                modes=("native-mimo-webdev",),
            )
        ),
    ]
