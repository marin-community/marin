# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A candidate's bounded SQL transport preserves every selected row and real failures."""

import pytest
from marina.client import MarinaRequestError

from infra.marina.apps.rl_data_catalog.release_gateway import bounded_read_query


def test_oversized_read_keeps_original_bounds_and_all_rows_in_order():
    original = {"limit": 10, "offset": 7}
    selected = [{"id": index} for index in range(7, 12)]
    calls = []

    def query(sql, parameters):
        calls.append((sql, parameters))
        assert parameters["limit"] == 10 and parameters["offset"] == 7
        size = parameters.get("__atlas_gate_limit", 100)
        if size > 2:
            raise MarinaRequestError(413, "query response exceeds byte budget")
        offset = parameters["__atlas_gate_offset"]
        rows = selected[offset : offset + size]
        return {"rows": rows, "row_count": len(rows)}

    result = bounded_read_query(query, "SELECT id FROM sample ORDER BY id LIMIT :limit OFFSET :offset", original)
    assert result["rows"] == selected
    assert result["row_count"] == len(selected)
    assert original == {"limit": 10, "offset": 7}
    assert all("ORDER BY id LIMIT :limit OFFSET :offset" in sql for sql, _ in calls)


@pytest.mark.parametrize("status", [401, 500, 504])
def test_candidate_gateway_never_pages_away_non_size_failures(status):
    calls = []

    def query(sql, parameters):
        calls.append(sql)
        raise MarinaRequestError(status, "real failure")

    with pytest.raises(MarinaRequestError) as error:
        bounded_read_query(query, "SELECT id FROM sample", {})
    assert error.value.status_code == status
    assert len(calls) == 1


def test_single_oversized_row_stays_a_failure():
    limits = []

    def query(sql, parameters):
        limits.append(parameters.get("__atlas_gate_limit"))
        raise MarinaRequestError(413, "one row exceeds transport")

    with pytest.raises(MarinaRequestError) as error:
        bounded_read_query(query, "SELECT large_payload FROM sample", {})
    assert error.value.status_code == 413
    assert limits[-1] == 1 and len(limits) == 8
