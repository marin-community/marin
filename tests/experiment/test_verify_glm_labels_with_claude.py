# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from export_glm_claude_review import CLAUDE_REVIEW_CHUNK_MARKER  # noqa: E402
from verify_glm_labels_with_claude import comparison, review_package_from_chunks  # noqa: E402


def test_review_package_from_chunks() -> None:
    package = {"taxonomy": [{"bucket_id": "CODE"}], "documents": []}
    encoded = base64.b64encode(gzip.compress(json.dumps(package).encode())).decode()
    split = len(encoded) // 2
    logs = "\n".join(
        [
            f"task prefix {CLAUDE_REVIEW_CHUNK_MARKER}0001/0002:{encoded[split:]}",
            f"task prefix {CLAUDE_REVIEW_CHUNK_MARKER}0000/0002:{encoded[:split]}",
        ]
    )

    assert review_package_from_chunks(logs) == package


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
            "secondary_bucket_ids": ["CODE"],
            "language": "Python",
            "document_type": "Code",
            "confidence": 0.6,
            "rationale": "Technical text.",
        },
    ]

    result = comparison(package, claude_rows)

    assert result["primary_exact_agreement"] == 0.5
    assert result["bucket_set_overlap_fraction"] == 1.0
    assert result["glm_primary_in_claude_set_fraction"] == 1.0
    assert result["claude_primary_in_glm_set_fraction"] == 0.5
    assert [row["sample_index"] for row in result["disagreements"]] == [2]
