# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete sample and unfinished full-population review panels."""

from finestore.rl.curation import QualifiedSourceDeclaration as SourceDeclaration

from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

# Waived FULL finality needs a retained population large enough that its unreviewed
# remainder stays within the 1% bound; the over-cap case exists to show it is not final.
FULL_WAIVER_RETAINED = {"full-waiver": 150, "full-waiver-unconclusive": 300, "full-waiver-over-cap": 150}


def pending_review_declaration() -> SourceDeclaration:
    base = contamination_declarations()[0][1].model_dump(mode="json")
    return SourceDeclaration.model_validate(
        {
            **base,
            "id": "Marin:review-pending",
            "license_receipt": fixture_license("Marin:review-pending", base["dataset_id"], base["dataset_revision"]),
            "name": "review-pending",
            "display_name": "review-pending",
            "upstream_dataset": "review-pending",
            "upstream_name": "review-pending",
            "oversight": "sample",
            "quality_result": None,
            "independent_qc": None,
        }
    )


def review_declarations() -> list[tuple[bool, SourceDeclaration]]:
    base = contamination_declarations()[0][1].model_dump(mode="json")
    cases = [
        ("sample-content", 2, 0, 0, "none", 2, False),
        ("sample-missing-five", 95, 0, 5, "sample", 95, False),
        ("sample-missing-six", 94, 0, 6, "sample", 94, True),
        ("full-missing-five", 95, 0, 5, "full", 95, True),
        ("sample-unconclusive", 0, 2, 0, "none", 2, True),
        ("full-waiver-unconclusive", 0, 299, 1, "full", 299, False),
        ("sample-complete", 2, 0, 0, "sample", None, False),
        ("sample-uncertain", 1, 1, 0, "sample", None, True),
        ("full-inflight", 1, 0, 2, "full", 1, True),
        ("full-uncertain", 2, 1, 0, "full", 3, True),
        ("full-complete", 3, 0, 0, "full", 3, False),
        ("full-waiver", 149, 0, 1, "full", 149, False),
        ("full-waiver-over-cap", 148, 0, 2, "full", 148, True),
        ("full-false-complete", 1, 0, 2, "full", 1, True),
        ("sample-false-waiver", 1, 0, 1, "sample", 1, True),
    ]
    return [
        (
            provisional,
            SourceDeclaration.model_validate(
                {
                    **base,
                    "id": f"Marin:{name}",
                    "license_receipt": fixture_license(f"Marin:{name}", base["dataset_id"], base["dataset_revision"]),
                    "name": name,
                    "display_name": name,
                    "upstream_dataset": name,
                    "upstream_name": name,
                    "oversight": oversight,
                    **(
                        {
                            "task_count": retained,
                            "within_source_dedup": deduplication_evidence(
                                "fixture", base["dataset_revision"], "1", retained
                            ),
                        }
                        if (retained := FULL_WAIVER_RETAINED.get(name)) is not None
                        else {}
                    ),
                    "quality_result_finalization": (
                        "complete"
                        if name == "full-false-complete"
                        else (
                            "accepted_incomplete"
                            if name
                            in {"full-waiver", "sample-false-waiver", "full-waiver-unconclusive", "full-waiver-over-cap"}
                            else None
                        )
                    ),
                    "quality_result_waiver": (
                        {"uri": "https://example.org/waiver-with-exact-reasons.json", "sha256": "a" * 64}
                        if name
                        in {"full-waiver", "sample-false-waiver", "full-waiver-unconclusive", "full-waiver-over-cap"}
                        else None
                    ),
                    "quality_result": {
                        "reviewed": reviewed,
                        "uncertain_count": uncertain,
                        "unavailable_count": unavailable,
                        "issue_count": 0,
                        "issues": [],
                        "repair_recommendation": "Inspect review coverage",
                    },
                    "independent_qc": (
                        {
                            "verdict": "good",
                            "method": "Fixture content assembly",
                            "reviewed": qc,
                            "evidence_url": "https://example.org/assembly",
                        }
                        if qc is not None
                        else None
                    ),
                }
            ),
        )
        for name, reviewed, uncertain, unavailable, oversight, qc, provisional in cases
    ]
