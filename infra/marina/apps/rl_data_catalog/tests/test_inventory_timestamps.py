# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native and gateway timestamp spellings preserve the same strict source comparison."""

from copy import deepcopy

import pytest

from infra.marina.apps.rl_data_catalog.release_check import inventory_timestamp_projection


def inventory(timestamp):
    return {
        "created_at": timestamp,
        "synced_at": timestamp,
        "sources": [{"id": "source", "updated_at": timestamp, "example_count": 7, "notes": timestamp}],
    }


def test_equivalent_metadata_timestamp_spellings_compare_equal_without_mutation():
    gateway = inventory("2026-11-17T17:46:28.54137+00:00")
    hosted = inventory("2026-11-17T12:46:28.541370-05:00")
    hosted["sources"][0]["notes"] = gateway["sources"][0]["notes"]
    original = deepcopy(gateway)
    assert inventory_timestamp_projection(gateway) == inventory_timestamp_projection(hosted)
    assert gateway == original


@pytest.mark.parametrize("changed", ["instant", "count", "content", "naive", "malformed"])
def test_a_different_instant_or_source_value_is_never_excused(changed):
    expected = inventory("2026-11-17T17:46:28.541370Z")
    actual = deepcopy(expected)
    if changed == "instant":
        actual["sources"][0]["updated_at"] = "2026-11-17T17:46:28.541371Z"
    elif changed == "count":
        actual["sources"][0]["example_count"] = 8
    elif changed == "content":
        actual["sources"][0]["notes"] = "2026-11-17T17:46:28.54137+00:00"
    elif changed == "naive":
        actual["created_at"] = "2026-11-17T17:46:28.541370"
    else:
        actual["synced_at"] = "invalid"
    assert inventory_timestamp_projection(expected) != inventory_timestamp_projection(actual)
