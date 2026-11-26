# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 license receipts for the two repaired SWE sources the user ruled about-to-be-admitted.

Both receipts are the Atlas license resolver's resolutions at the exact pins the
declarations import (atlas license-v2), transcribed verbatim; the declaration tests
re-derive them from the audit record.
"""

from finestore.rl.licenses import (
    LicenseDecision,
    LicenseDecisionKind,
    LicenseEvidence,
    LicenseReceipt,
    LicenseSourcePin,
)

SWE_REBENCH_V2_1_REBENCHED_README = (
    "https://huggingface.co/datasets/ibragim-bad/SWE-rebench-V2.1/resolve/"
    "26c0145ded93168b0d5d095420155901ad2d5a2f/README.md"
)
SWE_REBENCH_README_SHA256 = "10e21fa86fa4aa9631d431db5fa2da96d9ca3f6c7f20c71ffb07713ecb72c77e"

SWE_REBENCH_V2_1_RECEIPT = LicenseReceipt(
    source_id="SWE-rebench:ibragim-bad/SWE-rebench-V2.1@repaired-20261119",
    dataset_id="ibragim-bad/SWE-rebench-V2.1",
    revision="26c0145ded93168b0d5d095420155901ad2d5a2f",
    resolved_at="2026-11-20T00:55:21.418874+00:00",
    resolver_version="atlas-license-v2",
    evidence=[
        LicenseEvidence(
            url=SWE_REBENCH_V2_1_REBENCHED_README,
            kind="dataset_card",
            sha256=SWE_REBENCH_README_SHA256,
            license_ids=["cc-by-4.0"],
            quoted_lines=[
                "L2: license: cc-by-4.0",
                "L1: ---",
                "L3: task_categories:",
                "L4: - text-generation",
                "L5: language:",
                "L6: - en",
                "L7: tags:",
                "L8: - code",
                "L9: - software-engineering",
                "L10: - swe-bench",
                "L11: - reinforcement-learning",
                "L12: - harbor",
            ],
            restrictive_terms=False,
            unidentified_third_party_terms=False,
        ),
        LicenseEvidence(
            url=SWE_REBENCH_V2_1_REBENCHED_README,
            kind="readme_license",
            sha256=SWE_REBENCH_README_SHA256,
            license_ids=["missing"],
            quoted_lines=["(no license statement in this document)"],
            restrictive_terms=False,
            unidentified_third_party_terms=False,
        ),
        LicenseEvidence(
            url=SWE_REBENCH_V2_1_REBENCHED_README,
            kind="readme_notice",
            sha256=SWE_REBENCH_README_SHA256,
            license_ids=[],
            quoted_lines=["(no license statement in this document)"],
            restrictive_terms=False,
            unidentified_third_party_terms=False,
        ),
    ],
    bundled_third_party=False,
    model_generated_outputs=False,
)

SWESMITH_V5_RELEASE = "open-athena/campaign-release-swesmith-verifier-repaired-v5-0bfcd894-retained-20261116-v1"
SWESMITH_V5_REVISION = "758b28bf4cc0d0e770329f2944c9451fd578566a"
TASKTROVE_CARD = "https://huggingface.co/datasets/open-athena/task-trove/resolve/0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c/README.md"
TASKTROVE_CARD_SHA256 = "e7b0794110fa2b1942f1c8d982ea0cfce0154ca36d517f7c2fde8db34806a1ee"
SWESMITH_ISSUE_URL = "https://github.com/marin-community/marin/issues/11497#issuecomment-6902843532"

SWESMITH_VERIFIER_REPAIRED_V5_RECEIPT = LicenseReceipt(
    source_id="TaskTrove:swesmith-verifier-repaired-v5",
    dataset_id=SWESMITH_V5_RELEASE,
    revision=SWESMITH_V5_REVISION,
    resolved_at="2026-11-16T16:29:08.721625+00:00",
    resolver_version="atlas-license-v2",
    evidence=[
        LicenseEvidence(
            url=TASKTROVE_CARD,
            kind="dataset_card",
            sha256=TASKTROVE_CARD_SHA256,
            license_ids=["apache-2.0", "cc-by-4.0", "mit", "other"],
            quoted_lines=[
                "L3: license:",
                "L5: - cc-by-4.0",
                "L1: ---",
                "L2: pretty_name: TaskTrove Clean",
                "L4: - apache-2.0",
                "L6: - mit",
                "L7: - other",
                "L8: language:",
                "L9: - en",
                "L10: task_categories:",
                "L11: - text-generation",
                "L12: tags:",
            ],
            restrictive_terms=False,
            unidentified_third_party_terms=False,
        ),
        LicenseEvidence(
            url=TASKTROVE_CARD,
            kind="readme_license",
            sha256=TASKTROVE_CARD_SHA256,
            license_ids=["Apache-2.0"],
            quoted_lines=[
                "L234: The pinned upstream TaskTrove dataset declares the Apache 2.0 license. The `source` column and",
                "L235: release manifest retain source-level provenance for reviewing the terms that apply to a selected",
                "L236: cohort.",
            ],
            restrictive_terms=False,
            unidentified_third_party_terms=False,
        ),
        LicenseEvidence(
            url=TASKTROVE_CARD,
            kind="readme_notice",
            sha256=TASKTROVE_CARD_SHA256,
            license_ids=[],
            quoted_lines=["(no license statement in this document)"],
            restrictive_terms=False,
            unidentified_third_party_terms=False,
        ),
    ],
    bundled_third_party=True,
    model_generated_outputs=True,
    decisions=[
        LicenseDecision(
            kind=LicenseDecisionKind.EXCEPTION,
            approved=True,
            who="user",
            when="2026-11-16T11:30:00-05:00",
            source_id="TaskTrove:swesmith-verifier-repaired-v5",
            dataset_id=SWESMITH_V5_RELEASE,
            revision=SWESMITH_V5_REVISION,
            scope=["train"],
            rationale=(
                'The user ruled (verbatim: "TaskTrove:swesmith-verifier-repaired-v5 is just swesmith data. '
                'It should share the same simple license.") that this remediated version shares the same '
                "license as the existing recorded decision for this source at "
                "26bb8c8ffba53adecd1c2a5e9d64b47b9f7e52fd, which records the SWE-bench/SWE-smith (MIT) grant "
                "of the SWE-smith source it repackages, with the checker-PASS upstream pin "
                "SWE-bench/SWE-smith@ea6d7173829c7ec8fa16c22055699ff2e9188091. The raw checker escalation at "
                "this pin is the open-athena/task-trove@0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c aggregate "
                "release card's multi-license list (E2), unchanged from the prior pin. Training use only at "
                "this exact pin; an exception does not transfer to a new revision."
            ),
            evidence_urls=[
                "https://huggingface.co/datasets/SWE-bench/SWE-smith/blob/ea6d7173829c7ec8fa16c22055699ff2e9188091/README.md",
                SWESMITH_ISSUE_URL,
                f"https://huggingface.co/datasets/{SWESMITH_V5_RELEASE}/tree/{SWESMITH_V5_REVISION}",
            ],
            issue_url=SWESMITH_ISSUE_URL,
        ),
    ],
    resolved_source=LicenseSourcePin(
        dataset_id="open-athena/task-trove",
        revision="0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c",
        url="https://huggingface.co/datasets/open-athena/task-trove/tree/0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c",
        declaration=LicenseEvidence(
            url=(
                f"https://huggingface.co/datasets/{SWESMITH_V5_RELEASE}/resolve/{SWESMITH_V5_REVISION}/README.md"
            ),
            kind="wrapper_source",
            sha256="22dfb6f2da39d0c7ac9b59c899a7a2683e3fd17b48252dc7f1a55e292add432c",
            license_ids=[],
            quoted_lines=[
                "L8: license_source:",
                "L9:   dataset_id: open-athena/task-trove",
                "L10:   revision: 0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c",
                "L11:   url: https://huggingface.co/datasets/open-athena/task-trove/tree/0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c",
                "L23: Source pin: `open-athena/swesmith-verifier-repaired-v5-20261112@26bb8c8ffba53adecd1c2a5e9d64b47b9f7e52fd`. "
                "Converter: `2026.11.12.2@1509fc8638f6be3840b5c82b53348f93152a9c31`.",
            ],
            restrictive_terms=False,
            unidentified_third_party_terms=False,
        ),
    ),
)
