# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Primary license evidence for the derived HARDMath training population."""

from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseEvidence, LicenseReceipt

UPSTREAM_REVISION = "9ade062a80b7d475666b11ae13a2fb0d7b7f0341"
PROCESSED_REVISION = "937e9f10356e31e854f6efb9a2507f1e200c8b25"
UPSTREAM_LICENSE_URL = f"https://github.com/sarahmart/HARDMath/blob/{UPSTREAM_REVISION}/LICENSE"
LINEAGE_NOTES = (
    "MIT upstream grant: " + UPSTREAM_LICENSE_URL + ". "
    "All 702 processed questions exactly match upstream data/HARDMath.json; question_type and answer_type "
    "also match all 702. 135 solution strings match exactly; the other solutions are modified. "
    "Question identity establishes derivation, not answer correctness. Pending repair and qualification: "
    "check transformed solutions and ground_truths, then republish and measure a clean retained cohort. "
    "The raw processed pin is not admitted for training."
)

LICENSE_RECEIPT = LicenseReceipt(
    source_id="pafitis:hardmath",
    dataset_id="pafitis/HARDMath_processed_training",
    revision=PROCESSED_REVISION,
    resolved_at="2026-11-25T21:49:00+00:00",
    resolver_version=LICENSE_RESOLVER_VERSION,
    evidence=[
        LicenseEvidence(
            url=UPSTREAM_LICENSE_URL,
            kind="upstream_license_file",
            sha256="f3421821278074a4cd7e50c5bcf523648ce908f5a729e12e1a0bfde940569377",
            license_ids=["mit"],
            quoted_lines=["MIT License", "Permission is hereby granted, free of charge, to any person obtaining a copy"],
        ),
        LicenseEvidence(
            url="https://huggingface.co/datasets/open-athena/atlas-source-pins/resolve/eec74e0a1f6e7a8c81f382d193a82ae00ced069e/hardmath_lineage.json",
            kind="derived_dataset_row_match",
            sha256="0c0b2144b5c1c5de8337a85e56e7d6533afa12c3fe387765cf7b1759c67802b5",
            license_ids=["mit"],
            quoted_lines=['"total": 702,', '"matched": 702,'],
        ),
    ],
    bundled_third_party=False,
    model_generated_outputs=False,
)
