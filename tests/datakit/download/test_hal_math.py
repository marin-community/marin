# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.hal_math import hal_record_to_document


def test_hal_record_to_document_preserves_rights_and_identifiers() -> None:
    document = hal_record_to_document(
        {
            "docid": "123",
            "halId_s": "hal-00000123",
            "uri_s": "https://hal.science/hal-00000123v2",
            "title_s": ["A theorem"],
            "abstract_s": ["We prove it."],
            "fileLicenses_s": ["https://creativecommons.org/licenses/by-nc-sa/4.0/"],
            "arxivId_s": "2401.01234",
            "doiId_s": "10.1000/example",
            "domain_s": ["0.math", "1.math.math-ap"],
            "submittedDate_s": "2026-04-01 12:00:00",
        }
    )

    assert document == {
        "id": "hal:123",
        "text": "# A theorem\n\nWe prove it.",
        "source": "hal/mathematics-licensed-abstracts",
        "title": "A theorem",
        "license": ["CC BY-NC-SA 4.0"],
        "provenance": {
            "docid": "123",
            "hal_id": "hal-00000123",
            "source_url": "https://hal.science/hal-00000123v2",
            "license_uris": ["https://creativecommons.org/licenses/by-nc-sa/4.0/"],
            "arxiv_ids": ["2401.01234"],
            "doi_ids": ["10.1000/example"],
            "domains": ["0.math", "1.math.math-ap"],
            "submitted_date": "2026-04-01 12:00:00",
        },
    }


def test_hal_record_to_document_rejects_mixed_rights() -> None:
    document = hal_record_to_document(
        {
            "docid": "123",
            "halId_s": "hal-00000123",
            "abstract_s": ["We prove it."],
            "fileLicenses_s": [
                "https://creativecommons.org/licenses/by/4.0/",
                "https://creativecommons.org/licenses/by-nc-nd/4.0/",
            ],
        }
    )

    assert document is None
