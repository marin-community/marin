# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete sample and unfinished full-population review panels."""

from finestore.rl.curation import SourceDeclaration

from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations


def pending_review_declaration() -> SourceDeclaration:
    base = contamination_declarations()[0][1].model_dump(mode="json")
    return SourceDeclaration.model_validate(
        {
            **base,
            "id": "Marin:review-pending",
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
        ("sample-complete", 2, 0, 0, "sample", None, False),
        ("sample-uncertain", 1, 1, 0, "sample", None, True),
        ("full-inflight", 1, 0, 2, "full", 1, True),
        ("full-uncertain", 2, 1, 0, "full", 3, True),
        ("full-complete", 3, 0, 0, "full", 3, False),
        ("full-waiver", 1, 1, 1, "full", 2, False),
    ]
    return [
        (
            provisional,
            SourceDeclaration.model_validate(
                {
                    **base,
                    "id": f"Marin:{name}",
                    "name": name,
                    "display_name": name,
                    "upstream_dataset": name,
                    "upstream_name": name,
                    "oversight": oversight,
                    "quality_result_finalization": "accepted_incomplete" if name == "full-waiver" else None,
                    "quality_result_waiver": (
                        {"uri": "https://example.org/waiver-with-exact-reasons.json", "sha256": "a" * 64}
                        if name == "full-waiver"
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
