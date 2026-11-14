# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize item-level repeat statistics for historical catalog records."""

import sqlalchemy
from evalrecords.records import EvalRunRecord
from evalrecords.repeat_aggregates import normalize_repeat_record


def upgrade(conn: sqlalchemy.Connection) -> None:
    metadata = sqlalchemy.MetaData()
    catalog_runs = sqlalchemy.Table("eval_catalog_runs", metadata, autoload_with=conn)
    catalog_state = sqlalchemy.Table("eval_catalog_state", metadata, autoload_with=conn)
    state = sqlalchemy.select(catalog_state.c.singleton).where(catalog_state.c.singleton.is_(True))
    if conn.dialect.name == "postgresql":
        state = state.with_for_update()
    conn.execute(state).one()

    candidates = sqlalchemy.select(catalog_runs.c.run_id, catalog_runs.c.record).where(
        sqlalchemy.cast(catalog_runs.c.record, sqlalchemy.Text).like('%"num_repeat"%')
    )
    updates = []
    for row in conn.execute(candidates).mappings():
        record = EvalRunRecord.model_validate(row["record"])
        projected = normalize_repeat_record(record)
        if not projected.repeat_aggregates:
            continue
        updates.append(
            {"target_run_id": row["run_id"], "projected_record": projected.model_dump(mode="json", by_alias=True)}
        )
    if not updates:
        return
    update = (
        catalog_runs.update()
        .where(catalog_runs.c.run_id == sqlalchemy.bindparam("target_run_id"))
        .values(record=sqlalchemy.bindparam("projected_record"))
    )
    for start in range(0, len(updates), 128):
        conn.execute(update, updates[start : start + 128])
    conn.execute(
        catalog_state.update()
        .where(catalog_state.c.singleton.is_(True))
        .values(generation=catalog_state.c.generation + 1, updated_at=sqlalchemy.func.now())
    )
