# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory-only declarations for published campaign cohorts.

These entries make releases the Snowball Smartie RL campaign prepared and
published outside the curation driver visible in the RL Data Atlas. Each pins
one immutable Hugging Face revision. None has a conversion recipe: campaign
admission, split assignment and qualification evidence already live in the
release itself, so re-converting them here would invent a second provenance.

Container-backed cohorts additionally carry host-local Docker image locators.
Their screened public releases record instruction text and identity but no image
bytes or verifier material, so no machine can execute these tasks from the public
release alone. Treat every entry as a data pointer, not a runnable pipeline.
"""

from finestore.rl.curation import DaytonaSafety, IndependentQC, Verdict

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference

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
                daytona_safety=DaytonaSafety.UNSAFE,
            )
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
                daytona_safety=DaytonaSafety.UNSAFE,
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
                daytona_safety=DaytonaSafety.SAFE,
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
                daytona_safety=DaytonaSafety.UNSAFE,
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
                daytona_safety=DaytonaSafety.UNSAFE,
            )
        ),
    ]
