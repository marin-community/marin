# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published campaign cohorts and runnable rebuilt releases.

Each entry pins an immutable Hugging Face revision in the RL Data Atlas.
Rebuilt TaskCompendium cohorts use ``prebuilt.import_prebuilt`` to validate and
admit their published tasks without repeating conversion. Their qualification
evidence and oversight levels describe the checks behind admission.
Native Harbor cohorts use ``prebuilt.import_harbor_prebuilt`` and require
exact-package control receipts for admission.

Legacy Snowball Smartie releases remain inventory pointers: their host-local
image locators, absent image bytes and omitted verifier material cannot be used
to execute tasks from the screened public release alone.
"""

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
    SourceReference,
    TaskFormat,
)

TASK_TROVE_PUBLIC = "open-athena/task-trove"
PUBLIC_STAGED_REVISION = "5d6a5c9cbbf1541ae7325cbdba290a102ab05ca0"
CAMPAIGN_ISSUE = "https://github.com/marin-community/marin/issues/10305"

STAGE3_IMAGE_CAVEAT = (
    "Container image material is host-local and excluded from the screened release, so this "
    "release is data, not a runnable task environment."
)
QUALIFICATION_IS_NOT_ADMISSION = (
    "Campaign qualification is not curation admission: use the release's own admission "
    "manifest rather than treating archived rows as approved training examples."
)


def _public_release(release: str, revision: str) -> SourceReference:
    return SourceReference(
        TASK_TROVE_PUBLIC,
        revision,
        f"https://huggingface.co/datasets/{TASK_TROVE_PUBLIC}/tree/{revision}/releases/{release}",
    )


def _pinned_blob(reference: SourceReference, member: str = "") -> str:
    """Blob URL for a pinned tree reference itself or one of its members."""
    url = reference.url.replace("/tree/", "/blob/", 1)
    return f"{url}/{member}" if member else url


def sources() -> list[RlDataSource]:
    calibforge = _public_release("smartie-continuation-20261016/calibforge", "6a6429ccba098e11ea9f6297a70cb43a34670c0f")
    shared_v2 = _public_release("calibforge-shared-20261017-v2", "3b6c5b5533f1b702163f365b7ce869b834cb2994")
    swesmith = _public_release("smartie-continuation-20261016/swesmith", "0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c")
    staged_v1 = _public_release("smartie-swesmith-staged-v1", PUBLIC_STAGED_REVISION)
    staged_v2 = _public_release("smartie-swesmith-staged-v2", PUBLIC_STAGED_REVISION)
    swesmith_repaired = PrebuiltCohort(
        name="swesmith-verifier-repaired-v5",
        version="2026.11.12.1",
        repo="open-athena/swesmith-verifier-repaired-v5-20261112",
        revision="26bb8c8ffba53adecd1c2a5e9d64b47b9f7e52fd",
        expected_rows=7016,
        repo_type="dataset",
    )
    swesmith_evidence = f"https://huggingface.co/datasets/{swesmith_repaired.repo}/blob/{swesmith_repaired.revision}/"
    return [
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-calibforge-admitted",
                title="Snowball Smartie CalibForge admitted cohort",
                origin="Snowball Smartie RL",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "license:cc-by-4.0", "campaign-release", "deprecated"),
                count=2205,
                daytona_snapshot_count=2205,
                daytona_count_basis=(
                    "production snapshot_identity over the pinned payloads; each task names its own "
                    "registry image (2,205 distinct docker_image references)"
                ),
                notes=(
                    "2,205 quality-admitted CalibForge tasks. Deprecated: every task names its own "
                    "prebuilt registry image in task.toml, so running the cohort provisions one "
                    "Daytona snapshot per task — 2,205 distinct images across 2,205 tasks on the "
                    "pinned release payloads, far beyond the ten-snapshot budget. Oracle solution "
                    "execution was also skipped, so successful execution is not certified, and the "
                    "pinned release manifest records the cohort as not published. A snapshot-safe "
                    "rebuild from these candidates is planned. See the campaign report: " + CAMPAIGN_ISSUE
                ),
                dataset=calibforge,
                independent_qc=IndependentQC(
                    verdict=Verdict.SOME_ISSUES,
                    method="release quality review accepted the task text; oracle execution skipped and "
                    "privacy, split and overlap checks remain before publication",
                    reviewed=2205,
                    admitted=2205,
                    evidence_url=_pinned_blob(calibforge, "manifest.json"),
                ),
                oracle_validation=OracleValidation.SKIPPED,
            )
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="calibforge-v1-rebuilt",
                version="1",
                repo="open-athena/calibforge-v1-rebuilt-taskcompendium-20261111",
                revision="8f00c12884d6a68107b245e7c98711b6bd64f4af",
                expected_rows=1357,
            ),
            info=SourceInfo(
                id="Campaign Release:smartie-calibforge-rebuilt-v1",
                title="Snowball Smartie CalibForge V1 rebuilt cohort (TaskCompendium)",
                origin="Snowball Smartie RL",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "license:cc-by-4.0", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=1357,
                daytona_snapshot_count=7,
                daytona_count_basis=(
                    "production snapshot_identity with Harbor's machine_spec over the lowered task specs "
                    "at the pinned revision: 1,357 specs, seven distinct identities, zero failures"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="Snowball Smartie RL",
                    recipe_revision="calibforge_rebuild.py @ 817f7dbd00",
                    verification=(
                        "setup fidelity executed on all 1,357 tasks (1,357 pass) and verifier liveness on "
                        "76 sampled tasks drawn across all seven bases (76 negative verdicts, reward 0.0, "
                        "no infrastructure errors)"
                    ),
                    reviewed=1357,
                    admitted=1357,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "1,357 CalibForge V1 tasks rebuilt onto seven digest-pinned base images, with fixtures "
                    "staged through task resources and environment_requirements.setup_commands. Upstream: "
                    "smartie-continuation-20261016/calibforge @ 6a6429ccba098e11ea9f6297a70cb43a34670c0f, "
                    "minus the curated decontamination list. 732 tasks deferred because their recipe needs a "
                    "step the lowering cannot express; one excluded because its own setup segfaults. "
                    "Qualified without oracles: 1,357 of 1,357 reproduce the native copies at their "
                    "destinations, and 76 of 76 sampled verifiers return a negative verdict on the "
                    "untouched workspace. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/calibforge-v1-rebuilt-taskcompendium-20261111",
                    "8f00c12884d6a68107b245e7c98711b6bd64f4af",
                    "https://huggingface.co/datasets/open-athena/calibforge-v1-rebuilt-taskcompendium-20261111"
                    "/tree/8f00c12884d6a68107b245e7c98711b6bd64f4af",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="nl2bash-v1-rebuilt",
                version="1",
                repo="open-athena/nl2bash-v1-rebuilt-taskcompendium-20261111",
                revision="1be6be13a89ab3ef30c7390ca2f29d7b9eeb40dd",
                expected_rows=1498,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-nl2bash-rebuilt-v1",
                title="TaskTrove nl2bash rebuilt cohort (TaskCompendium)",
                origin="TaskTrove",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "language:bash", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=1498,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1,498 specs, one distinct agent identity (each task's archived ubuntu:24.04 recipe "
                    "lowered onto the pinned base), zero execution-requirement failures. Graders are "
                    "lock-placed and grade in the worker's local runtime, contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/nl2bash.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 1,498 converted specs at this revision (all reproduce "
                        "their archive's lowered base, working directory, steps and environment variables) and "
                        "verifier liveness on 20 sampled tasks drawn across the cohort's single environment "
                        "(20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=1498,
                    admitted=1498,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "1,498 nl2bash tasks converted to run in the environment their archives specify: the "
                    "ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned base and its RUN "
                    "steps replay as setup commands, one distinct environment across the cohort. Measured "
                    "at the pinned upstream revision "
                    "DCAgent2__nl2bash-tasks-cleaned-oracle-v2 @ 02923004846e4e73862c20962f823a6d05100e7a: "
                    "zero refusals, zero substitutions. Qualified without oracles: setup fidelity matches on "
                    "all 1,498 rows and 20 of 20 sampled verifiers return a negative verdict on an empty "
                    "submission at this revision. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/nl2bash-v1-rebuilt-taskcompendium-20261111",
                    "1be6be13a89ab3ef30c7390ca2f29d7b9eeb40dd",
                    "https://huggingface.co/datasets/open-athena/nl2bash-v1-rebuilt-taskcompendium-20261111"
                    "/tree/1be6be13a89ab3ef30c7390ca2f29d7b9eeb40dd",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="codeforces-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-codeforces-v1-taskcompendium-20261112",
                revision="05167566303b67a1adea08743d26c39101c5c30a",
                expected_rows=9745,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-codeforces-rebuilt-v1",
                title="TaskTrove codeforces rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=9745,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "one distinct agent identity (each task's archived recipe lowered onto the pinned "
                    "base), zero execution-requirement failures. Graders are lock-placed and grade in "
                    "the worker's local runtime, contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 9,745 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks drawn across the cohort's "
                        "single environment (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=9745,
                    admitted=9745,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "9,745 codeforces tasks converted to run in the environment their archives specify: "
                    "the ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned base and "
                    "its RUN steps replay as setup commands, one distinct environment across the cohort, "
                    "measured at the pinned upstream revision laion__codeforces-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. 240 rows are "
                    "refused as null graders and 15 for gold leaked into the instruction; refused rows "
                    "are the conversion's own reviewed outcome classes. Qualified without oracles: setup "
                    "fidelity matches on every converted row and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-codeforces-v1-taskcompendium-20261112",
                    "05167566303b67a1adea08743d26c39101c5c30a",
                    "https://huggingface.co/datasets/open-athena/tasktrove-codeforces-v1-taskcompendium-20261112"
                    "/tree/05167566303b67a1adea08743d26c39101c5c30a",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="competitive-coding-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-competitive-coding-v1-taskcompendium-20261112",
                revision="5a7b7a75eb32901947157e39516a15d894db332b",
                expected_rows=13973,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-competitive-coding-rebuilt-v1",
                title="TaskTrove competitive-coding rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=13973,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "one distinct agent identity (each task's archived recipe lowered onto the pinned "
                    "base), zero execution-requirement failures. Graders are lock-placed and grade in "
                    "the worker's local runtime, contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 13,973 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks drawn across the cohort's "
                        "single environment (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=13973,
                    admitted=13973,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "13,973 competitive-coding tasks converted to run in the environment their archives "
                    "specify: the ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned "
                    "base and its RUN steps replay as setup commands, one distinct environment across the "
                    "cohort, measured at the pinned upstream revision "
                    "laion__nemotron-gym-competitive-coding-v2 @ 02923004846e4e73862c20962f823a6d05100e7a "
                    "with zero substitutions. 1,739 rows are refused for gold leaked into the instruction "
                    "and one as a reviewed defect; refused rows are the conversion's own reviewed outcome "
                    "classes. Qualified without oracles: setup fidelity matches on every converted row "
                    "and 20 of 20 sampled verifiers return a negative verdict on an empty submission. "
                    "The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-competitive-coding-v1-taskcompendium-20261112",
                    "5a7b7a75eb32901947157e39516a15d894db332b",
                    "https://huggingface.co/datasets/open-athena/tasktrove-competitive-coding-v1-taskcompendium-20261112"
                    "/tree/5a7b7a75eb32901947157e39516a15d894db332b",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="curriculum-easy-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-curriculum-easy-v1-taskcompendium-20261112",
                revision="4151b071dbb38df532b38c7de1becb12ca1e0f73",
                expected_rows=505,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-curriculum-easy-rebuilt-v1",
                title="TaskTrove curriculum-easy rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=505,
                daytona_snapshot_count=2,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "2 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 505 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=505,
                    admitted=505,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "505 curriculum-easy tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_curriculum-easy @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-curriculum-easy-v1-taskcompendium-20261112",
                    "4151b071dbb38df532b38c7de1becb12ca1e0f73",
                    "https://huggingface.co/datasets/open-athena/tasktrove-curriculum-easy-v1-taskcompendium-20261112"
                    "/tree/4151b071dbb38df532b38c7de1becb12ca1e0f73",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="curriculum-medium-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-curriculum-medium-v1-taskcompendium-20261112",
                revision="db17ab9b01d960d01f7a75c3d1d896504ceaf9fa",
                expected_rows=488,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-curriculum-medium-rebuilt-v1",
                title="TaskTrove curriculum-medium rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=488,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 488 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=488,
                    admitted=488,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "488 curriculum-medium tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_curriculum-medium-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-curriculum-medium-v1-taskcompendium-20261112",
                    "db17ab9b01d960d01f7a75c3d1d896504ceaf9fa",
                    "https://huggingface.co/datasets/open-athena/tasktrove-curriculum-medium-v1-taskcompendium-20261112"
                    "/tree/db17ab9b01d960d01f7a75c3d1d896504ceaf9fa",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="e2egit-v2-rebuilt",
                version="1",
                repo="open-athena/tasktrove-e2egit-v2-taskcompendium-20261112",
                revision="dd34672091d917c96c922fd2ebf0d583a1dfc5b9",
                expected_rows=477,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-e2egit-v2-rebuilt",
                title="TaskTrove e2egit-v2 rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=477,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over all 477 lowered specs at the pinned revision: "
                    "one distinct agent identity at 1 CPU / 4096 MB, zero identity or execution failures. "
                    "The compiler package-lock grader runs in the worker's local runtime and adds no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py version 8; verifyit @ f0155c1143",
                    verification=(
                        "setup fidelity compared on all 477 converted specs against their native archives; "
                        "the sole shared setup recipe executed successfully with real package installation. "
                        "Production grading with COMPILER_GRADER_PACKAGES, build identity "
                        "42bb937b102dd758b4d1210769750d7e259a036169ef1458d11821809118888b, "
                        "returned 20 negative verdicts (reward 0.0, no infrastructure errors) on the seed-0 sample. "
                        "Evidence is published beside tasks.parquet at the pinned revision"
                    ),
                    reviewed=477,
                    admitted=477,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "477 e2egit-v2 tasks preserve their archived environment, lowered onto its digest-pinned base. "
                    "The upstream is DCAgent__exp_rpt_e2egit-v2 at 02923004846e4e73862c20962f823a6d05100e7a; "
                    "ten reviewed source defects remain excluded. The private release includes setup and liveness "
                    "evidence, retained rejections, and an explicit durable compiler grader lock. "
                    "Qualification uses an empty submission; no oracle or model-quality run is claimed. "
                    "Daytona execution remains untested"
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-e2egit-v2-taskcompendium-20261112",
                    "dd34672091d917c96c922fd2ebf0d583a1dfc5b9",
                    "https://huggingface.co/datasets/open-athena/tasktrove-e2egit-v2-taskcompendium-20261112"
                    "/tree/dd34672091d917c96c922fd2ebf0d583a1dfc5b9",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="e2egit-large-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-e2egit-large-v1-taskcompendium-20261112",
                revision="4bdaeaaf069eb6ad7b0a716d445e9e789f37a758",
                expected_rows=4993,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-e2egit-large-rebuilt-v1",
                title="TaskTrove e2egit-large rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=4993,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 4,993 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4993,
                    admitted=4993,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "4,993 e2egit-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision DCAgent__exp_rpt_e2egit-large @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-e2egit-large-v1-taskcompendium-20261112",
                    "4bdaeaaf069eb6ad7b0a716d445e9e789f37a758",
                    "https://huggingface.co/datasets/open-athena/tasktrove-e2egit-large-v1-taskcompendium-20261112"
                    "/tree/4bdaeaaf069eb6ad7b0a716d445e9e789f37a758",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="multifile-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-multifile-v1-taskcompendium-20261112",
                revision="5accbb5851dc5774a05424aa7d22f995cbfd7e34",
                expected_rows=4842,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-multifile-rebuilt-v1",
                title="TaskTrove multifile rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=4842,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 4,842 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4842,
                    admitted=4842,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "4,842 multifile tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision DCAgent__exp_rpt_multifile-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-multifile-v1-taskcompendium-20261112",
                    "5accbb5851dc5774a05424aa7d22f995cbfd7e34",
                    "https://huggingface.co/datasets/open-athena/tasktrove-multifile-v1-taskcompendium-20261112"
                    "/tree/5accbb5851dc5774a05424aa7d22f995cbfd7e34",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="pymethods2test-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-pymethods2test-v1-taskcompendium-20261112",
                revision="f6a20442763e152ec662483dc0ee1476caa6d541",
                expected_rows=500,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-pymethods2test-rebuilt-v1",
                title="TaskTrove pymethods2test rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=500,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 500 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=500,
                    admitted=500,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "500 pymethods2test tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_pymethods2test-v3 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-pymethods2test-v1-taskcompendium-20261112",
                    "f6a20442763e152ec662483dc0ee1476caa6d541",
                    "https://huggingface.co/datasets/open-athena/tasktrove-pymethods2test-v1-taskcompendium-20261112"
                    "/tree/f6a20442763e152ec662483dc0ee1476caa6d541",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="pymethods2test-large-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-pymethods2test-large-v1-taskcompendium-20261112",
                revision="a9ed5c72ade6670853fc28af8bf1b59541d3827c",
                expected_rows=4990,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-pymethods2test-large-rebuilt-v1",
                title="TaskTrove pymethods2test-large rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=4990,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 4,990 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4990,
                    admitted=4990,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "4,990 pymethods2test-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_pymethods2test-large-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-pymethods2test-large-v1-taskcompendium-20261112",
                    "a9ed5c72ade6670853fc28af8bf1b59541d3827c",
                    "https://huggingface.co/datasets/open-athena/tasktrove-pymethods2test-large-v1-taskcompendium-20261112"
                    "/tree/a9ed5c72ade6670853fc28af8bf1b59541d3827c",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="stack-pytest-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-stack-pytest-v1-taskcompendium-20261112",
                revision="f6909b5b5fe8d207fb442f24d4140556431d02a3",
                expected_rows=492,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-stack-pytest-rebuilt-v1",
                title="TaskTrove stack-pytest rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=492,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 492 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=492,
                    admitted=492,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "492 stack-pytest tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_stack-pytest-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-stack-pytest-v1-taskcompendium-20261112",
                    "f6909b5b5fe8d207fb442f24d4140556431d02a3",
                    "https://huggingface.co/datasets/open-athena/tasktrove-stack-pytest-v1-taskcompendium-20261112"
                    "/tree/f6909b5b5fe8d207fb442f24d4140556431d02a3",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="unitsyn-python-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-unitsyn-python-v1-taskcompendium-20261112",
                revision="3ed40d26a51c8ca0cf26c15cc84316409534a937",
                expected_rows=4991,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-unitsyn-python-rebuilt-v1",
                title="TaskTrove unitsyn-python rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=4991,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 4,991 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4991,
                    admitted=4991,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "4,991 unitsyn-python tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_unitsyn-python-v4 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-unitsyn-python-v1-taskcompendium-20261112",
                    "3ed40d26a51c8ca0cf26c15cc84316409534a937",
                    "https://huggingface.co/datasets/open-athena/tasktrove-unitsyn-python-v1-taskcompendium-20261112"
                    "/tree/3ed40d26a51c8ca0cf26c15cc84316409534a937",
                ),
            ),
        ),
        RlDataSource(
            pipeline=import_prebuilt,
            config=PrebuiltCohort(
                name="unitsyn-python-large-v1-rebuilt",
                version="1",
                repo="open-athena/tasktrove-unitsyn-python-large-v1-taskcompendium-20261112",
                revision="3a64ac3ed258bd184c24c24e7c7578f64142c180",
                expected_rows=4991,
            ),
            info=SourceInfo(
                id="Campaign Release:tasktrove-unitsyn-python-large-rebuilt-v1",
                title="TaskTrove unitsyn-python-large rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                count=4991,
                daytona_snapshot_count=1,
                daytona_count_basis=(
                    "production snapshot_identity over the lowered task specs at the pinned revision: "
                    "1 distinct agent identity, zero execution-requirement failures. Graders are "
                    "lock-placed (the compiler grader lock) and grade in the worker's local runtime, "
                    "contributing no Daytona identity"
                ),
                daytona_execution_failures=0,
                oracle_validation=OracleValidation.VERIFIER_EXECUTED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/python_tests.py @ e6be97168d",
                    verification=(
                        "setup fidelity executed on all 4,991 converted specs at this revision (all "
                        "reproduce their archive's lowered base, working directory, steps and environment "
                        "variables) and verifier liveness on 20 sampled tasks across the cohort's "
                        "environments (20 negative verdicts, reward 0.0, no infrastructure errors)"
                    ),
                    reviewed=4991,
                    admitted=4991,
                    level=OversightLevel.SAMPLE,
                ),
                notes=(
                    "4,991 unitsyn-python-large tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_unitsyn-python-large-v2 @ "
                    "02923004846e4e73862c20962f823a6d05100e7a with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
                ),
                dataset=SourceReference(
                    "open-athena/tasktrove-unitsyn-python-large-v1-taskcompendium-20261112",
                    "3a64ac3ed258bd184c24c24e7c7578f64142c180",
                    "https://huggingface.co/datasets/open-athena/tasktrove-unitsyn-python-large-v1-taskcompendium-20261112"
                    "/tree/3a64ac3ed258bd184c24c24e7c7578f64142c180",
                ),
            ),
        ),
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-calibforge-shared-v2",
                title="Snowball Smartie repaired CalibForge shared environment v2",
                origin="Snowball Smartie RL",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "license:cc-by-4.0", "campaign-release", "deprecated"),
                count=324,
                daytona_snapshot_count=23,
                daytona_count_basis=(
                    "campaign provisioning record snapshot-capacity-check.json (the public lowered "
                    "payloads carry no resource shapes and pin only two verifier contexts)"
                ),
                notes=(
                    "324 environment-admitted CalibForge tasks split 259 train / 65 validation. "
                    "Deprecated pending the snapshot-safe CalibForge rebuild: the campaign's "
                    "provisioning records (snapshot-capacity-check) report the cohort needing 23 "
                    "image/resource snapshot variants (19 on the retained training slice), over "
                    "the ten-snapshot budget. The public lowered payloads pin only two verifier "
                    "contexts and carry no per-task resource shapes, so the variants are invisible "
                    "there; admission checks cover fixture equivalence and setup and verifier "
                    "execution, and oracle solution execution was waived."
                ),
                dataset=shared_v2,
                independent_qc=IndependentQC(
                    verdict=Verdict.GOOD,
                    method="fixture equivalence plus setup and verifier execution admission checks; "
                    "oracle solution execution waived",
                    reviewed=2205,
                    admitted=324,
                    evidence_url=_pinned_blob(shared_v2, "manifest.json"),
                ),
            )
        ),
        RlDataSource(
            pipeline=import_harbor_prebuilt,
            config=swesmith_repaired,
            info=SourceInfo(
                id="Campaign Release:swesmith-verifier-repaired-v5",
                title="SWE-smith verifier repaired v5",
                origin="Snowball Smartie RL",
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
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-swesmith-admitted",
                title="Snowball Smartie SWE-Smith admitted cohort",
                origin="Snowball Smartie RL",
                family="swe-repo",
                tags=("agentic", "multi-turn", "campaign-release", "deprecated"),
                count=8045,
                daytona_snapshot_count=5,
                daytona_count_basis="shellbox-identity over the pinned tasks.parquet build contexts",
                notes=(
                    "Deprecated: the original 8,045-task admission did not enforce the hardened "
                    "bug-zero/reference-one controls. Re-control excluded 2,417 tasks, including "
                    "scratch permissions, generated metadata, verifier infrastructure failures, "
                    "timeouts, degenerate references and passing bug commits. Use the separately "
                    "pinned repaired cohort; its v4-admitted task packages retain their tested bytes. "
                    "The original campaign's quality and oracle records remain historical evidence."
                ),
                dataset=swesmith,
                independent_qc=IndependentQC(
                    verdict=Verdict.GOOD,
                    method="every retained task passed Daytona oracle execution and a Good quality verdict",
                    reviewed=8045,
                    admitted=8045,
                    evidence_url=_pinned_blob(swesmith, "manifest.json"),
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-swesmith-staged-v1",
                title="Snowball Smartie staged SWE-Smith Stage 1 cohort",
                origin="Snowball Smartie RL",
                family="swe-repo",
                tags=("agentic", "multi-turn", "campaign-release"),
                count=1699,
                notes=(
                    "1,683 setup-branch parents from qualified repositories, retained as 1,699 staged "
                    "variants. Stage 1 tests repository and workspace readiness, not issue repair. The "
                    "verifier inspects actual checkout state and rejects upstream defaults and wrong "
                    "branches. The public release is screened to agent-visible instructions; execution "
                    "and verifier materials remain host-local. "
                    + QUALIFICATION_IS_NOT_ADMISSION
                    + " "
                    + STAGE3_IMAGE_CAVEAT
                ),
                dataset=staged_v1,
                independent_qc=IndependentQC(
                    verdict=Verdict.UNKNOWN,
                    method="Stage 1 workspace-readiness verification; the release makes no task-quality claim",
                    reviewed=1699,
                    evidence_url=_pinned_blob(staged_v1, "manifest.json"),
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-swesmith-staged-v2",
                title="Snowball Smartie staged SWE-Smith Stage 3 handoffs",
                origin="Snowball Smartie RL",
                family="swe-repo",
                tags=("agentic", "multi-turn", "campaign-release"),
                count=778,
                notes=(
                    "778 repair handoffs from 600 parents: 600 complete and 178 partial, retained as "
                    "762 train / 16 validation. Complete handoffs start from repaired code that passed "
                    "the original required tests; partials start from partially repaired code with a "
                    "witnessed remaining failure that the known repair clears. Stage 3 was not used in "
                    "run 2026.10.19.57. The public release is screened to agent-visible instructions; "
                    "execution and verifier materials remain host-local. "
                    + QUALIFICATION_IS_NOT_ADMISSION
                    + " "
                    + STAGE3_IMAGE_CAVEAT
                ),
                dataset=staged_v2,
                independent_qc=IndependentQC(
                    verdict=Verdict.SOME_ISSUES,
                    method="Stage 3 handoff verification against required tests; 178 of 778 handoffs are partial "
                    "and none were used in run 2026.10.19.57",
                    reviewed=778,
                    evidence_url=_pinned_blob(staged_v2, "manifest.json"),
                ),
            )
        ),
    ]
