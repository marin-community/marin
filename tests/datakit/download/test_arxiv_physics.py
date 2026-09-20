# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.arxiv_physics import arxiv_record_to_document


def _record(*, license_uri: str, categories: str, update_date: str) -> dict:
    return {
        "id": "2501.01234",
        "title": "A physics result",
        "abstract": "We derive the result.",
        "categories": categories,
        "license": license_uri,
        "update_date": update_date,
        "doi": "10.1000/example",
        "versions": [{"version": "v1", "created": "Wed, 1 Jan 2025 00:00:00 GMT"}],
    }


def test_arxiv_record_to_document_retains_post_snapshot_physics() -> None:
    document = arxiv_record_to_document(
        _record(
            license_uri="http://creativecommons.org/licenses/by/4.0/",
            categories="cs.LG quant-ph",
            update_date="2025-01-01",
        )
    )

    assert document is not None
    assert document["id"] == "arxiv:2501.01234v1"
    assert document["provenance"]["novelty_basis"] == "post_common_pile_snapshot"
    assert document["provenance"]["categories"] == ["cs.LG", "quant-ph"]


def test_arxiv_record_to_document_retains_older_noncommercial_physics() -> None:
    document = arxiv_record_to_document(
        _record(
            license_uri="http://creativecommons.org/licenses/by-nc-sa/4.0/",
            categories="hep-th",
            update_date="2020-01-01",
        )
    )

    assert document is not None
    assert document["provenance"]["novelty_basis"] == "noncommercial_license"


def test_arxiv_record_to_document_rejects_replay_or_incompatible_rows() -> None:
    assert (
        arxiv_record_to_document(
            _record(
                license_uri="http://creativecommons.org/licenses/by/4.0/",
                categories="physics.optics",
                update_date="2020-01-01",
            )
        )
        is None
    )
    assert (
        arxiv_record_to_document(
            _record(
                license_uri="http://creativecommons.org/licenses/by-nc-nd/4.0/",
                categories="physics.optics",
                update_date="2026-01-01",
            )
        )
        is None
    )
    assert (
        arxiv_record_to_document(
            _record(
                license_uri="http://creativecommons.org/licenses/by/4.0/",
                categories="cs.LG stat.ML",
                update_date="2026-01-01",
            )
        )
        is None
    )
