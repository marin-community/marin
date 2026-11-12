# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published campaign cohorts and runnable rebuilt releases.

Each entry pins an immutable Hugging Face revision in the RL Data Atlas.
Rebuilt TaskCompendium cohorts use ``prebuilt.import_prebuilt`` to validate and
admit their published tasks without repeating conversion. Their qualification
evidence and oversight levels describe the checks behind admission.

Legacy Snowball Smartie releases remain inventory pointers: their host-local
image locators, absent image bytes and omitted verifier material cannot be used
to execute tasks from the screened public release alone.
"""

from dataclasses import replace

from finestore.rl.curation import (
    IndependentQC,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    Verdict,
)

from experiments.post_training.task_curation.datasets.tasktrove.archives import TASKTROVE_REVISION
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_prebuilt
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


def _lock_grader_count_basis(identities: int, lock_graders: int) -> str:
    """The measured Daytona basis behind a rebuilt cohort whose graders need package locks."""
    return (
        "Production snapshot_identity over every pinned published TaskSpec: "
        f"{identities} distinct agent identities. "
        "Offline Daytona factory validation refuses all "
        f"{lock_graders:,} declared package-lock graders. "
        "The declared grader environments require local execution; this cohort is Daytona-unsafe."
    )


def _single_environment_verification(converted: int, sampled: int) -> str:
    """The oversight evidence a rebuilt single-environment cohort records."""
    return (
        "Unverified historical reward-only sample; does not certify test execution. "
        "setup fidelity executed on all "
        f"{converted:,} converted specs at this revision (all reproduce their archive's "
        "lowered base, working directory, steps and environment variables) and verifier "
        f"liveness on {sampled} sampled tasks drawn across the cohort's single environment "
        f"({sampled} negative verdicts, reward 0.0, no infrastructure errors)"
    )


def _rebuilt_cohort(config: PrebuiltCohort, info: SourceInfo) -> RlDataSource[PrebuiltCohort]:
    """Derive inventory identity and row counts from the pinned import configuration."""
    return RlDataSource(
        pipeline=import_prebuilt,
        config=config,
        info=replace(info, count=config.expected_rows, dataset=config.dataset),
    )


def sources() -> list[RlDataSource]:
    calibforge = _public_release("smartie-continuation-20261016/calibforge", "6a6429ccba098e11ea9f6297a70cb43a34670c0f")
    shared_v2 = _public_release("calibforge-shared-20261017-v2", "3b6c5b5533f1b702163f365b7ce869b834cb2994")
    swesmith = _public_release("smartie-continuation-20261016/swesmith", "0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c")
    staged_v1 = _public_release("smartie-swesmith-staged-v1", PUBLIC_STAGED_REVISION)
    staged_v2 = _public_release("smartie-swesmith-staged-v2", PUBLIC_STAGED_REVISION)
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
                title="Snowball Smartie CalibForge V1 rebuilt cohort (TaskCompendium)",
                origin="Snowball Smartie RL",
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
                title="TaskTrove nl2bash rebuilt cohort (TaskCompendium)",
                origin="TaskTrove",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "language:bash", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 1498),
                daytona_execution_failures=1498,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/nl2bash.py @ e6be97168d",
                    verification=_single_environment_verification(1498, 20),
                    reviewed=1498,
                    admitted=1498,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "1,498 nl2bash tasks converted to run in the environment their archives specify: the "
                    "ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned base and its RUN "
                    "steps replay as setup commands, one distinct environment across the cohort. Measured "
                    "at the pinned upstream revision "
                    f"DCAgent2__nl2bash-tasks-cleaned-oracle-v2 @ {TASKTROVE_REVISION}: "
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
                title="TaskTrove codeforces rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 9745),
                daytona_execution_failures=9745,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=_single_environment_verification(9745, 20),
                    reviewed=9745,
                    admitted=9745,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "9,745 codeforces tasks converted to run in the environment their archives specify: "
                    "the ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned base and "
                    "its RUN steps replay as setup commands, one distinct environment across the cohort, "
                    "measured at the pinned upstream revision laion__codeforces-v3 @ "
                    f"{TASKTROVE_REVISION} with zero substitutions. 240 rows are "
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
                title="TaskTrove competitive-coding rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 13973),
                daytona_execution_failures=13973,
                oracle_validation=OracleValidation.SKIPPED,
                oversight=OversightEvidence(
                    campaign="RL Data Atlas",
                    recipe_revision="datasets/tasktrove/code.py @ e6be97168d",
                    verification=_single_environment_verification(13973, 20),
                    reviewed=13973,
                    admitted=13973,
                    level=OversightLevel.NONE,
                ),
                notes=(
                    "13,973 competitive-coding tasks converted to run in the environment their archives "
                    "specify: the ubuntu:24.04 recipe each task ships is lowered onto the digest-pinned "
                    "base and its RUN steps replay as setup commands, one distinct environment across the "
                    "cohort, measured at the pinned upstream revision "
                    f"laion__nemotron-gym-competitive-coding-v2 @ {TASKTROVE_REVISION} "
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
                title="TaskTrove curriculum-easy rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=2,
                daytona_count_basis=_lock_grader_count_basis(2, 505),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove curriculum-medium rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 488),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove e2egit-v2 rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 477),
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
                    f"The upstream is DCAgent__exp_rpt_e2egit-v2 at {TASKTROVE_REVISION}; "
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
                title="TaskTrove e2egit-large rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 4993),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove multifile rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 4842),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove pymethods2test rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 500),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove pymethods2test-large rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 4990),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove stack-pytest rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 492),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove unitsyn-python rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 4991),
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
                    "4,991 unitsyn-python tasks converted to run in their archived environment, lowered onto "
                    "the digest-pinned base, measured at the pinned upstream revision "
                    "DCAgent__exp_rpt_unitsyn-python-v4 @ "
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
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
                title="TaskTrove unitsyn-python-large rebuilt cohort (TaskCompendium)",
                origin="Task Trove",
                family="python-tests",
                tags=("agentic", "multi-turn", "language:python", "campaign-release", "converted", "excluded"),
                task_format=TaskFormat.TASKCOMPENDIUM,
                daytona_snapshot_count=1,
                daytona_count_basis=_lock_grader_count_basis(1, 4991),
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
                    f"{TASKTROVE_REVISION} with zero substitutions. Qualified without "
                    "oracles: per-row setup fidelity matches and 20 of 20 sampled verifiers return a "
                    "negative verdict on an empty submission. The repository is private pending release."
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
            info=SourceInfo(
                id="Campaign Release:smartie-swesmith-admitted",
                title="Snowball Smartie SWE-Smith admitted cohort",
                origin="Snowball Smartie RL",
                family="swe-repo",
                tags=("agentic", "multi-turn", "campaign-release"),
                count=8045,
                daytona_snapshot_count=5,
                daytona_count_basis="shellbox-identity over the pinned tasks.parquet build contexts",
                notes=(
                    "8,045 quality-admitted SWE-Smith tasks. The release keeps only tasks that passed "
                    "the Daytona oracle and a Good quality verdict; quality review is not a "
                    "model-solving accuracy claim, and campaign-specific train and validation splits "
                    "remain in the private archive. Tasks share five build contexts (two FROM bases, "
                    "floating python:3.10/3.11-bookworm tags), so a run provisions five snapshots; "
                    "the campaign's capacity check budgets all five."
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
