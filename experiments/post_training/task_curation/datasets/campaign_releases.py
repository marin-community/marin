# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory-only declarations for published campaign cohorts.

These entries make releases the Snowball Smartie RL campaign prepared and
published outside the curation driver visible in the RL Data Atlas. Each pins
one immutable Hugging Face revision. None has a conversion recipe: campaign
admission, split assignment and qualification evidence already live in the
release itself, so re-converting them here would invent a second provenance.

Container-backed cohorts additionally carry host-local Docker image locators.
Their archives record image IDs but not image bytes, so no machine can execute
these tasks until the images are rebuilt or separately distributed. Treat every
entry as a data pointer, not a runnable pipeline.
"""

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference

TASK_TROVE_PUBLIC = "open-athena/task-trove"
TASK_TROVE_PRIVATE = "open-athena/task-trove-smartie-continuation-private"
CAMPAIGN_ARCHIVE = "campaign-archive/2026.10.20"
PRIVATE_ARCHIVE_REVISION = "ab5173cc03cd4392d5b8cdf38a3392e1ff44a0bc"
CAMPAIGN_ISSUE = "https://github.com/marin-community/marin/issues/10305"

STAGE3_IMAGE_CAVEAT = (
    "Docker image locators are local IDs, not pullable registry references; the archive "
    "excludes container image bytes, so this release is not self-contained executable data."
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


def _private_archive(archive: str) -> SourceReference:
    return SourceReference(
        TASK_TROVE_PRIVATE,
        PRIVATE_ARCHIVE_REVISION,
        f"https://huggingface.co/datasets/{TASK_TROVE_PRIVATE}/tree/{PRIVATE_ARCHIVE_REVISION}/{CAMPAIGN_ARCHIVE}/{archive}",
    )


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-calibforge-admitted",
                title="Snowball Smartie CalibForge admitted cohort",
                origin="Snowball Smartie RL",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "license:cc-by-4.0", "campaign-release"),
                count=2205,
                notes=(
                    "2,205 quality-admitted CalibForge tasks. Quality review accepted the task text; "
                    "oracle solution execution was explicitly skipped, so successful execution is not "
                    "certified. The pinned release manifest records this cohort as not published: "
                    "source privacy, split and overlap checks remain. "
                    "smartie-calibforge-shared-v2, repaired from these candidates, is the usable "
                    "release. See the campaign report: " + CAMPAIGN_ISSUE
                ),
                dataset=_public_release(
                    "smartie-continuation-20261016/calibforge", "6a6429ccba098e11ea9f6297a70cb43a34670c0f"
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-calibforge-shared-v2",
                title="Snowball Smartie repaired CalibForge shared environment v2",
                origin="Snowball Smartie RL",
                family="terminal-agent",
                tags=("agentic", "multi-turn", "license:cc-by-4.0", "campaign-release"),
                count=324,
                notes=(
                    "324 environment-admitted CalibForge tasks split 259 train / 65 validation, with "
                    "fixtures, qualification evidence, source attribution and reproduction scripts. "
                    "Admission checks cover fixture equivalence and setup and verifier execution; "
                    "oracle solution execution was waived."
                ),
                dataset=_public_release("calibforge-shared-20261017-v2", "3b6c5b5533f1b702163f365b7ce869b834cb2994"),
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
                notes=(
                    "8,045 quality-admitted SWE-Smith tasks. The release keeps only tasks that passed "
                    "the Daytona oracle and a Good quality verdict; quality review is not a "
                    "model-solving accuracy claim, and campaign-specific train and validation splits "
                    "remain in the private archive."
                ),
                dataset=_public_release(
                    "smartie-continuation-20261016/swesmith", "0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c"
                ),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-swesmith-staged-v1",
                title="Snowball Smartie staged SWE-Smith Stage 1 cohort",
                origin="Snowball Smartie RL",
                family="swe-repo",
                tags=("agentic", "multi-turn", "campaign-release", "private"),
                count=1699,
                notes=(
                    "1,683 setup-branch parents from qualified repositories, retained as 1,699 staged "
                    "variants. Stage 1 tests repository and workspace readiness, not issue repair. The "
                    "verifier inspects actual checkout state and rejects upstream defaults and wrong "
                    "branches. Private: Open-Athena access required. "
                    + QUALIFICATION_IS_NOT_ADMISSION
                    + " "
                    + STAGE3_IMAGE_CAVEAT
                ),
                dataset=_private_archive("swesmith-staged-v1-20261017.tar.gz"),
            )
        ),
        RlDataSource(
            info=SourceInfo(
                id="Campaign Release:smartie-swesmith-staged-v2",
                title="Snowball Smartie staged SWE-Smith Stage 3 handoffs",
                origin="Snowball Smartie RL",
                family="swe-repo",
                tags=("agentic", "multi-turn", "campaign-release", "private"),
                count=778,
                notes=(
                    "778 repair handoffs from 600 parents: 600 complete and 178 partial, retained as "
                    "762 train / 16 validation. Complete handoffs start from repaired code that passed "
                    "the original required tests; partials start from partially repaired code with a "
                    "witnessed remaining failure that the known repair clears. Stage 3 was not used in "
                    "run 2026.10.19.57. Private: Open-Athena access required. "
                    + QUALIFICATION_IS_NOT_ADMISSION
                    + " "
                    + STAGE3_IMAGE_CAVEAT
                ),
                dataset=_private_archive("swesmith-staged-v2-20261017.tar.gz"),
            )
        ),
    ]
