# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published campaign cohorts and runnable rebuilt releases.

Each entry pins an immutable Hugging Face revision in the RL Data Atlas.
Rebuilt TaskCompendium cohorts use ``prebuilt.import_prebuilt`` to validate and
admit their published tasks without repeating conversion. Their qualification
evidence and oversight levels describe the checks behind admission.
Native Harbor cohorts use ``prebuilt.import_harbor_prebuilt`` and require
exact-package control receipts for admission.

Retired legacy pointers are tracked in retired-inventory-only.json. Their
underlying artifacts and prior evidence remain archived.
"""

from dataclasses import replace

from finestore.rl.curation import (
    IndependentQC,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    Verdict,
)

from experiments.post_training.task_curation.prebuilt import (
    PrebuiltCohort,
    import_harbor_prebuilt,
    import_prebuilt,
)
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    TaskFormat,
)

TASKTROVE_UPSTREAM_URL = (
    "https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/" "02923004846e4e73862c20962f823a6d05100e7a"
)


def _rebuilt_cohort(config: PrebuiltCohort, info: SourceInfo) -> RlDataSource[PrebuiltCohort]:
    return RlDataSource(
        pipeline=import_prebuilt,
        config=config,
        info=replace(info, count=config.expected_rows, dataset=config.dataset),
    )


def sources() -> list[RlDataSource]:
    swesmith_repaired = PrebuiltCohort(
        name="swesmith-verifier-repaired-v5",
        version="2026.11.12.2",
        repo="open-athena/swesmith-verifier-repaired-v5-20261112",
        revision="26bb8c8ffba53adecd1c2a5e9d64b47b9f7e52fd",
        expected_rows=7016,
        repo_type="dataset",
    )
    swesmith_evidence = f"https://huggingface.co/datasets/{swesmith_repaired.repo}/blob/{swesmith_repaired.revision}/"
    return [
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="code-contests-built-image",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-code-contests-taskcompendium-20261112",
                revision="5b43c6f4c2322a5808e4dbdfaf4f0052037d3607",
                expected_rows=8222,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-code-contests-built-image",
                upstream_dataset="tasktrove-code-contests-built-image",
                upstream_name="tasktrove-code-contests-built-image",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production daytona_identity, Harbor machine_spec and Shellbox snapshot_identity over "
                    "all 8222 specs: one identity across actor and grader, at 2 CPU / 4096 MB / 10240 MB. "
                    "Zero identity failures and execution-requirement refusals; within the ten-identity budget"
                ),
                daytona_execution_failures=0,
                daytona_run_evidence=(
                    "https://huggingface.co/datasets/open-athena/tasktrove-code-contests-taskcompendium-20261112"
                    "/blob/5b43c6f4c2322a5808e4dbdfaf4f0052037d3607/liveness.jsonl"
                ),
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py version 3; frozen VerifyIT @ d8cdda47f4",
                    verification=(
                        "Withdrawn qualification: numeric-only rewards did not preserve unscored verifier errors. "
                        "all 8222 rows match their native archive's built context, including file bytes "
                        "and modes; all rendered stdio configurations equal their original typed graders. "
                        "The declared script invokes the image-installed VerifyIT CLI, preserving the "
                        "candidate interpreter. The seed-0 Daytona RL sample graded 20 empty submissions: "
                        "20 negative verdicts, reward 0.0, no infrastructure errors or timeouts. "
                        "Evidence accompanies tasks.parquet at the pinned private revision"
                    ),
                    reviewed=8222,
                    admitted=8222,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "Excluded: the pinned payload reads only numeric rewards and can hide verifier failures. "
                    "Republish with converter version 4 and robust SAMPLE/FULL controls before admission. "
                    "8222 tasks retain the exact built output of their archived environment for both "
                    "actor and grader. Registry manifests were read back and pulled image IDs matched "
                    "the local builds. The private release includes setup fidelity, stdio parity, "
                    "sampled liveness, retained exclusions and frozen build contexts. "
                    "No oracle solution or model-quality assessment ran"
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="taco-built-image",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-taco-taskcompendium-20261112",
                revision="c96a95bc94e4824fae425811f246501f64abfadd",
                expected_rows=5116,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-taco-built-image",
                upstream_dataset="tasktrove-taco-built-image",
                upstream_name="tasktrove-taco-built-image",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production daytona_identity, Harbor machine_spec and Shellbox snapshot_identity over "
                    "all 5116 specs: one identity across actor and grader, at 2 CPU / 4096 MB / 10240 MB. "
                    "Zero identity failures and execution-requirement refusals; within the ten-identity budget"
                ),
                daytona_execution_failures=0,
                daytona_run_evidence=(
                    "https://huggingface.co/datasets/open-athena/tasktrove-taco-taskcompendium-20261112"
                    "/blob/c96a95bc94e4824fae425811f246501f64abfadd/liveness.jsonl"
                ),
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py version 3; frozen VerifyIT @ d8cdda47f4",
                    verification=(
                        "Withdrawn qualification: numeric-only rewards did not preserve unscored verifier errors. "
                        "all 5116 rows match their native archive's built context, including file bytes "
                        "and modes; all rendered stdio configurations equal their original typed graders. "
                        "The declared script invokes the image-installed VerifyIT CLI, preserving the "
                        "candidate interpreter. The seed-0 Daytona RL sample graded 20 empty submissions: "
                        "20 negative verdicts, reward 0.0, no infrastructure errors or timeouts. "
                        "Evidence accompanies tasks.parquet at the pinned private revision"
                    ),
                    reviewed=5116,
                    admitted=5116,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "Excluded: the pinned payload reads only numeric rewards and can hide verifier failures. "
                    "Republish with converter version 4 and robust SAMPLE/FULL controls before admission. "
                    "5116 tasks retain the exact built output of their archived environment for both "
                    "actor and grader. Registry manifests were read back and pulled image IDs matched "
                    "the local builds. The private release includes setup fidelity, stdio parity, "
                    "sampled liveness, retained exclusions and frozen build contexts. "
                    "No oracle solution or model-quality assessment ran"
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="calibforge-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/calibforge-v1-rebuilt-taskcompendium-20261111",
                revision="8f00c12884d6a68107b245e7c98711b6bd64f4af",
                expected_rows=1357,
            ),
            info=SourceInfo(
                id="Campaign Release:smartie-calibforge-rebuilt-v1",
                upstream_dataset="smartie-calibforge-rebuilt-v1",
                upstream_name="smartie-calibforge-rebuilt-v1",
                subset=None,
                build_path="Snowball Smartie RL / Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="terminal-agent",
                tags=("agentic", "multi-turn", "license:cc-by-4.0", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=7,
                daytona_count_basis=(
                    "production snapshot_identity with Harbor's machine_spec over the lowered task specs "
                    "at the pinned revision: 1,357 specs, seven distinct identities, zero failures"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="Snowball Smartie RL",
                    recipe_revision="calibforge_rebuild.py @ 817f7dbd00",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 1,357 tasks (1,357 pass) and verifier liveness on "
                        "76 sampled tasks drawn across all seven bases (76 negative verdicts, reward 0.0, "
                        "no infrastructure errors)"
                    ),
                    reviewed=1357,
                    admitted=1357,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "Excluded pending file-delivery repair and qualification with verifier execution receipts. "
                    "1,357 CalibForge V1 tasks rebuilt onto seven digest-pinned base images, with fixtures "
                    "staged through task resources and environment_requirements.setup_commands. Upstream: "
                    "smartie-continuation-20261016/calibforge @ 6a6429ccba098e11ea9f6297a70cb43a34670c0f, "
                    "minus the curated decontamination list. 732 tasks deferred because their recipe needs a "
                    "step the lowering cannot express; one excluded because its own setup segfaults. "
                    "Withdrawn reward-only claim: 1,357 of 1,357 reproduce the native copies at their "
                    "destinations, and 76 of 76 sampled verifiers return a negative verdict on the "
                    "untouched workspace. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="nl2bash-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/nl2bash-v1-rebuilt-taskcompendium-20261111",
                revision="1be6be13a89ab3ef30c7390ca2f29d7b9eeb40dd",
                expected_rows=1498,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-nl2bash-rebuilt-v1",
                upstream_dataset="tasktrove-nl2bash-rebuilt-v1",
                upstream_name="tasktrove-nl2bash-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="terminal-agent",
                tags=("agentic", "multi-turn", "language:bash", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 1,498 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=1498,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/nl2bash.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 1,498 converted specs at this revision (all reproduce "
                        "their archive's lowered base, working directory, steps and environment variables) and "
                        "verifier liveness on 20 sampled tasks drawn across the cohort's single environment "
                        "(20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=1498,
                    admitted=1498,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "1,498 nl2bash tasks converted to run in the environment their archives specify: the "
                    "ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned base and its RUN "
                    "steps replay as setup commands, one distinct environment across the cohort. Measured "
                    "at the pinned upstream revision "
                    "DCAgent2__nl2bash-tasks-cleaned-oracle-v2 @ 02923004846e4e73862c20962f823a6d05100e7a: "
                    "zero refusals, zero substitutions. Withdrawn reward-only claim: setup fidelity matches on "
                    "all 1,498 rows and 20 of 20 sampled verifiers return a negative verdict on an empty "
                    "submission at this revision. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="codeforces-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-codeforces-v1-taskcompendium-20261112",
                revision="05167566303b67a1adea08743d26c39101c5c30a",
                expected_rows=9745,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-codeforces-rebuilt-v1",
                upstream_dataset="tasktrove-codeforces-rebuilt-v1",
                upstream_name="tasktrove-codeforces-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 9,745 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=9745,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 9,745 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks drawn across the cohort's "
                        "single environment (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=9745,
                    admitted=9745,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "9,745 codeforces tasks converted to run in the environment their archives specify: "
                    "the ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned base and "
                    "its RUN steps replay as setup commands, one distinct environment across the cohort, "
                    "measured at the pinned upstream revision laion__codeforces-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. 240 rows are "
                    "refused as null graders and 15 for gold leaked into the instruction; refused rows "
                    "are the conversion's own reviewed outcome classes. Withdrawn reward-only claim: setup "
                    "fidelity matches on every converted row and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="competitive-coding-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-competitive-coding-v1-taskcompendium-20261112",
                revision="5a7b7a75eb32901947157e39516a15d894db332b",
                expected_rows=13973,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-competitive-coding-rebuilt-v1",
                upstream_dataset="tasktrove-competitive-coding-rebuilt-v1",
                upstream_name="tasktrove-competitive-coding-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 13,973 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=13973,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 13,973 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks drawn across the cohort's "
                        "single environment (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=13973,
                    admitted=13973,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "13,973 competitive-coding tasks converted to run in the environment their archives "
                    "specify: the ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned "
                    "base and its RUN steps replay as setup commands, one distinct environment across the "
                    "cohort, measured at the pinned upstream revision "
                    "laion__nemotron-gym-competitive-coding-v2 @ 02923004846e4e73862c20962f823a6d05100e7a "
                    "with zero substitutions. 1,739 rows are refused for gold leaked into the instruction "
                    "and one as a reviewed defect; refused rows are the conversion's own reviewed outcome "
                    "classes. Withdrawn reward-only claim: setup fidelity matches on every converted row "
                    "and 20 of 20 sampled verifiers return a negative verdict on an empty submission. "
                    "The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="curriculum-easy-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-curriculum-easy-v1-taskcompendium-20261112",
                revision="4151b071dbb38df532b38c7de1becb12ca1e0f73",
                expected_rows=505,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-curriculum-easy-rebuilt-v1",
                upstream_dataset="tasktrove-curriculum-easy-rebuilt-v1",
                upstream_name="tasktrove-curriculum-easy-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=2,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 2 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 505 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=505,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 505 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=505,
                    admitted=505,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "505 curriculum-easy tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_curriculum-easy @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="curriculum-medium-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-curriculum-medium-v1-taskcompendium-20261112",
                revision="db17ab9b01d960d01f7a75c3d1d896504ceaf9fa",
                expected_rows=488,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-curriculum-medium-rebuilt-v1",
                upstream_dataset="tasktrove-curriculum-medium-rebuilt-v1",
                upstream_name="tasktrove-curriculum-medium-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 488 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=488,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 488 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=488,
                    admitted=488,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "488 curriculum-medium tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_curriculum-medium-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="e2egit-v2-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-e2egit-v2-taskcompendium-20261112",
                revision="dd34672091d917c96c922fd2ebf0d583a1dfc5b9",
                expected_rows=477,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-e2egit-v2-rebuilt",
                upstream_dataset="tasktrove-e2egit-v2-rebuilt",
                upstream_name="tasktrove-e2egit-v2-rebuilt",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 477 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=477,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py version 8; verifyit @ f0155c1143",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity compared on all 477 converted specs against their native archives; "
                        "the sole shared setup recipe executed successfully with real package installation. "
                        "Production grading with COMPILER_GRADER_PACKAGES, build identity "
                        "42bb937b102dd758b4d1210769750d7e259a036169ef1458d11821809118888b, "
                        "returned 20 negative verdicts (reward 0.0, no infrastructure errors) on the seed-0 sample. "
                        "Evidence is published beside tasks.parquet at the pinned revision"
                    ),
                    reviewed=477,
                    admitted=477,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "477 e2egit-v2 tasks preserve their archived environment, lowered onto its digest-pinned base. "
                    "The upstream is DCAgent__exp_rpt_e2egit-v2 at 02923004846e4e73862c20962f823a6d05100e7a; "
                    "ten reviewed source defects remain excluded. The private release includes setup and liveness "
                    "evidence, retained rejections, and an explicit durable compiler grader lock. "
                    "Qualification uses an empty submission; no oracle or model-quality run is claimed. "
                    "Daytona execution remains untested"
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="e2egit-large-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-e2egit-large-v1-taskcompendium-20261112",
                revision="4bdaeaaf069eb6ad7b0a716d445e9e789f37a758",
                expected_rows=4993,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-e2egit-large-rebuilt-v1",
                upstream_dataset="tasktrove-e2egit-large-rebuilt-v1",
                upstream_name="tasktrove-e2egit-large-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,993 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4993,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,993 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4993,
                    admitted=4993,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "4,993 e2egit-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision DCAgent__exp_rpt_e2egit-large @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="multifile-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-multifile-v1-taskcompendium-20261112",
                revision="5accbb5851dc5774a05424aa7d22f995cbfd7e34",
                expected_rows=4842,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-multifile-rebuilt-v1",
                upstream_dataset="tasktrove-multifile-rebuilt-v1",
                upstream_name="tasktrove-multifile-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,842 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4842,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,842 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4842,
                    admitted=4842,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "4,842 multifile tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision DCAgent__exp_rpt_multifile-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="pymethods2test-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-pymethods2test-v1-taskcompendium-20261112",
                revision="f6a20442763e152ec662483dc0ee1476caa6d541",
                expected_rows=500,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-pymethods2test-rebuilt-v1",
                upstream_dataset="tasktrove-pymethods2test-rebuilt-v1",
                upstream_name="tasktrove-pymethods2test-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 500 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=500,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 500 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=500,
                    admitted=500,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "500 pymethods2test tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_pymethods2test-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="pymethods2test-large-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-pymethods2test-large-v1-taskcompendium-20261112",
                revision="a9ed5c72ade6670853fc28af8bf1b59541d3827c",
                expected_rows=4990,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-pymethods2test-large-rebuilt-v1",
                upstream_dataset="tasktrove-pymethods2test-large-rebuilt-v1",
                upstream_name="tasktrove-pymethods2test-large-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,990 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4990,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,990 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4990,
                    admitted=4990,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "4,990 pymethods2test-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_pymethods2test-large-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="stack-pytest-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-stack-pytest-v1-taskcompendium-20261112",
                revision="f6909b5b5fe8d207fb442f24d4140556431d02a3",
                expected_rows=492,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-stack-pytest-rebuilt-v1",
                upstream_dataset="tasktrove-stack-pytest-rebuilt-v1",
                upstream_name="tasktrove-stack-pytest-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 492 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=492,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 492 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=492,
                    admitted=492,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "492 stack-pytest tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_stack-pytest-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="unitsyn-python-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-unitsyn-python-v1-taskcompendium-20261112",
                revision="3ed40d26a51c8ca0cf26c15cc84316409534a937",
                expected_rows=4991,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-unitsyn-python-rebuilt-v1",
                upstream_dataset="tasktrove-unitsyn-python-rebuilt-v1",
                upstream_name="tasktrove-unitsyn-python-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,991 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4991,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,991 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4991,
                    admitted=4991,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "Excluded: this release contains the large Unitsyn population rather than the 491-row small cohort. "
                    "4,991 tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_unitsyn-python-v4 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        _rebuilt_cohort(
            config=PrebuiltCohort(
                name="unitsyn-python-large-v1-rebuilt",
                version="2026.11.12.1",
                repo="open-athena/tasktrove-unitsyn-python-large-v1-taskcompendium-20261112",
                revision="3a64ac3ed258bd184c24c24e7c7578f64142c180",
                expected_rows=4991,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-unitsyn-python-large-rebuilt-v1",
                upstream_dataset="tasktrove-unitsyn-python-large-rebuilt-v1",
                upstream_name="tasktrove-unitsyn-python-large-rebuilt-v1",
                subset=None,
                build_path="Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "Production snapshot_identity over every pinned published TaskSpec: 1 distinct agent identities. "
                    "Offline Daytona factory validation refuses all 4,991 declared package-lock graders. The declared "
                    "grader environments require local execution; this cohort is Daytona-unsafe."
                ),
                daytona_execution_failures=4991,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "Unverified historical reward-only sample; does not certify test execution. "
                        "setup fidelity executed on all 4,991 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4991,
                    admitted=4991,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "4,991 unitsyn-python-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_unitsyn-python-large-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_harbor_prebuilt,
            config=swesmith_repaired,
            info=SourceInfo(
                id="Campaign Release:swesmith-verifier-repaired-v5",
                upstream_dataset="swesmith-verifier-repaired-v5",
                upstream_name="swesmith-verifier-repaired-v5",
                subset=None,
                build_path="Snowball Smartie RL / Campaign Release",
                origin=SourceOrigin.TASKTROVE,
                upstream_url=TASKTROVE_UPSTREAM_URL,
                family="swe-repo",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "verified"),
                task_format=TaskFormat.HARBOR,
                count=swesmith_repaired.expected_rows,
                dataset=swesmith_repaired.dataset,
                daytona_snapshot_count=5,
                daytona_execution_failures=0,
                daytona_count_basis=(
                    "production Harbor machine_spec and Shellbox snapshot_identity over every native "
                    "Harbor package at the pinned revision, retaining each task.toml environment shape; "
                    "zero mapping failures. Evidence: " + swesmith_evidence + "daytona-identity.json"
                ),
                oracle_validation=OracleValidation.EXECUTED,
                oversight=OversightEvidence(
                    campaign="SWE-smith verifier repair v5",
                    recipe_revision="43a40938875b4fbc4286a0784f6a7454110a0453",
                    verification=(
                        "Every admitted task has scored bug=0 and pinned-reference=1 controls, matched "
                        "to its exact TaskSpec SHA256; infrastructure errors cannot admit a task. "
                        "All 5,628 admitted v4 package strings remain byte-identical. Evidence: "
                        + swesmith_evidence
                        + "controls.jsonl"
                    ),
                    reviewed=swesmith_repaired.expected_rows,
                    admitted=swesmith_repaired.expected_rows,
                    level=OversightLevel.FULL,
                ),
                independent_qc=IndependentQC(
                    verdict=Verdict.GOOD,
                    method=(
                        "admission requires exact-package bug-zero/reference-one controls; "
                        "residual exclusions retained"
                    ),
                    reviewed=8045,
                    admitted=swesmith_repaired.expected_rows,
                    evidence_url=swesmith_evidence + "manifest.json",
                ),
                notes=(
                    "7,016 controlled SWE-smith tasks: 5,628 admitted v4 packages preserved byte-for-byte "
                    "and 1,388 newly admitted repairs. 1,029 of 8,045 candidates remain excluded with "
                    "per-task reasons and receipts. Native Harbor packages retain their declared "
                    "environment, bootstrap and hidden verifier; import validates the complete cohort "
                    "without rewriting packages or rerunning controls. Oversight is full and oracle "
                    "validation is executed. Preserved upstream recipes include floating image tags "
                    "and dependency resolution; control receipts identify the tested builds. "
                    "The repair-campaign repository is private. Controls do not measure model-solving accuracy."
                ),
            ),
        ),
    ]
