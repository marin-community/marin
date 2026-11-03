# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pytest

from experiments.datakit.store.focus_provider_replay import (
    FocusTarget,
    WarcRecordRange,
    focus_record_ranges,
    focus_targets_from_index,
)


def test_focus_cdx_keeps_each_successful_html_occurrence_of_target_url():
    records = [
        {"url": "https://selected", "status": "200", "mime": "text/html", "offset": 10, "length": 20},
        {"url": "https://selected", "status": "200", "mime": "application/pdf", "offset": 30, "length": 100},
        {"url": "https://other", "status": "200", "mime": "text/html", "offset": 130, "length": 20},
        {"url": "https://selected", "status": "404", "mime": "text/html", "offset": 150, "length": 20},
        {"url": "https://selected", "status": "200", "mime-detected": "text/html", "offset": 170, "length": 30},
    ]
    cdx = gzip.compress("".join(f"key 20260701000000 {json.dumps(record)}\n" for record in records).encode())
    assert list(focus_record_ranges(cdx, {"https://selected"})) == [
        WarcRecordRange("https://selected", 10, 20),
        WarcRecordRange("https://selected", 170, 30),
    ]


def test_focus_locator_index_uses_selected_uuid_and_never_requires_cached_text():
    selected = [{"id": "selected-id", "source_id": json.dumps("selected-uuid")}]
    index = [
        {"id": "selected-id", "source_id": "other-uuid", "warc_filename": "wrong.warc.gz", "url": "https://wrong"},
        {
            "id": "selected-id",
            "source_id": "selected-uuid",
            "warc_filename": "selected.warc.gz",
            "url": "https://selected",
        },
    ]
    assert focus_targets_from_index(selected, iter(index)) == {
        "selected.warc.gz": [FocusTarget("selected-uuid", "https://selected", "selected-id")]
    }
    with pytest.raises(ValueError, match="lacks original WARC locators"):
        focus_targets_from_index(selected, iter(index[:1]))
