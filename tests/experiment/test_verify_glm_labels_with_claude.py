# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from verify_glm_labels_with_claude import comparison  # noqa: E402


def test_comparison_measures_blinded_label_agreement() -> None:
    package = {
        "taxonomy": [{"bucket_id": "SCIENCE"}, {"bucket_id": "CODE"}],
        "glm_assignments": [
            {
                "sample_index": 1,
                "primary_bucket_id": "SCIENCE",
                "secondary_bucket_ids": ["CODE"],
                "language": "English",
                "document_type": "Article",
                "confidence": 0.8,
                "rationale": "Science prose.",
            },
            {
                "sample_index": 2,
                "primary_bucket_id": "CODE",
                "secondary_bucket_ids": [],
                "language": "Python",
                "document_type": "Code",
                "confidence": 0.9,
                "rationale": "Python code.",
            },
        ],
    }
    claude_rows = [
        {
            "sample_index": 1,
            "primary_bucket_id": "SCIENCE",
            "secondary_bucket_ids": ["CODE"],
            "language": "english",
            "document_type": "article",
            "confidence": 0.9,
            "rationale": "Science.",
        },
        {
            "sample_index": 2,
            "primary_bucket_id": "SCIENCE",
            "secondary_bucket_ids": [],
            "language": "Python",
            "document_type": "Code",
            "confidence": 0.6,
            "rationale": "Technical text.",
        },
    ]

    result = comparison(package, claude_rows)

    assert result["primary_exact_agreement"] == 0.5
    assert result["language_exact_agreement"] == 1.0
    assert result["document_type_exact_agreement"] == 1.0
    assert result["secondary_overlap_fraction"] == 0.5
    assert [row["sample_index"] for row in result["disagreements"]] == [2]
