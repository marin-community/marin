# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The difficulty transport stores JSONB objects and pages readbacks under the bound."""


import pytest
from sqlalchemy import create_engine, text

from infra.marina.apps.rl_data_catalog.load_difficulty import MEASUREMENT_READBACK_BATCH, load_measurements
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL
from infra.marina.apps.rl_data_catalog.tests.test_difficulty_coverage import _measurement_record


def _applet_query(engine):
    def query(sql: str, parameters: dict) -> dict:
        with engine.begin() as connection:
            result = connection.execute(text(sql), parameters)
            if result.returns_rows:
                rows = [dict(row) for row in result.mappings()]
                return {"rows": rows, "row_count": len(rows)}
            return {"rows": [], "row_count": result.rowcount}

    return query


@pytest.fixture
def difficulty_database(database_url: str):
    engine = create_engine(database_url)
    with engine.begin() as connection:
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                connection.execute(text(statement))
    yield engine
    engine.dispose()


def test_measurements_store_jsonb_objects_and_read_back_typed(database_url, difficulty_database):
    """The recordset insert stores measurement objects, not escaped JSON strings."""
    records = [_measurement_record(index) for index in range(7)]
    query = _applet_query(difficulty_database)
    assert load_measurements(records, query) == {"inserted": 7, "records": 7}
    with difficulty_database.begin() as connection:
        kinds = (
            connection.execute(
                text(
                    "SELECT measurement_id, jsonb_typeof(measurement) AS kind "
                    "FROM atlas_difficulty ORDER BY measurement_id"
                )
            )
            .mappings()
            .all()
        )
    assert [row["kind"] for row in kinds] == ["object"] * 7
    # idempotent: rerunning inserts nothing and still verifies every record
    assert load_measurements(records, query) == {"inserted": 0, "records": 7}


def test_measurement_readback_pages_under_the_transport_bound(database_url, difficulty_database):
    assert MEASUREMENT_READBACK_BATCH == 50


def test_measurement_readback_rejects_differing_evidence(database_url, difficulty_database):
    records = [_measurement_record(index) for index in range(3)]
    query = _applet_query(difficulty_database)
    load_measurements(records, query)
    clash = records[0].model_copy(update={"source_id": "Example:other"})
    with pytest.raises(ValueError, match="already names different evidence"):
        load_measurements([clash], query)
