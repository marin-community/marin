# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit the active catalog without fetching large declaration bodies."""

import json

from marina.applets import validate_query
from sqlalchemy import create_engine, text

from experiments.post_training.task_curation import ensure_current_projections
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL


def test_pin_audit_projects_active_identity_under_the_gateway_bound(database_url, monkeypatch):
    actor = {
        "dataset_id": "dataset",
        "dataset_revision": "revision",
        "converter_version": "converter",
        "converter_revision": "converter-revision",
        "measured_tasks": 3,
        "task_specs_sha256": "a" * 64,
        "evidence_url": "s3://example/actor.json",
    }
    pipeline = {"name": "example", "version": "converter"}
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            for statement in SCHEMA_SQL.split(";"):
                if statement.strip():
                    connection.execute(text(statement))
            connection.execute(text("UPDATE atlas_current SET load_id='active' WHERE name='latest'"))
            for load_id in ("historical", "active", "staged"):
                connection.execute(
                    text(
                        "INSERT INTO atlas_sources(load_id,id,declaration,latest_attempt_status,diagnostics) "
                        "VALUES (:load_id,'source',jsonb_build_object('dataset_id','dataset', "
                        "'dataset_revision',CAST(:revision AS TEXT),'pipeline',CAST(:pipeline AS JSONB), "
                        "'actor_census_input',CAST(:actor AS JSONB),'notes',repeat('x',6000000)),'complete','[]'::jsonb)"
                    ),
                    {
                        "load_id": load_id,
                        "revision": "revision" if load_id == "active" else load_id,
                        "actor": json.dumps(actor),
                        "pipeline": json.dumps(pipeline),
                    },
                )
            connection.execute(
                text(
                    "INSERT INTO atlas_sources(load_id,id,declaration,latest_attempt_status,diagnostics) "
                    "SELECT 'active','source-' || lpad(n::text,3,'0'),CAST(:declaration AS JSONB),"
                    "'complete','[]'::jsonb FROM generate_series(1,103) n"
                ),
                {
                    "declaration": json.dumps(
                        {"dataset_id": "dataset", "dataset_revision": "revision", "pipeline": pipeline}
                    )
                },
            )
            connection.execute(text("UPDATE curriculum_current SET load_id='active' WHERE name='latest'"))
            connection.execute(
                text(
                    "INSERT INTO curriculum_sources VALUES ('active','source',CAST(:declaration AS JSONB),"
                    "'example','{}'::jsonb)"
                ),
                {
                    "declaration": json.dumps(
                        {"dataset_id": "dataset", "dataset_revision": "revision", "pipeline": pipeline}
                    )
                },
            )

        pages = []

        def query(sql, parameters):
            validate_query(sql)
            with engine.begin() as connection:
                rows = [dict(row) for row in connection.execute(text(sql), parameters).mappings()]
            if "FROM atlas_sources" in sql:
                pages.append(len(rows))
                if len(pages) == 1:
                    with engine.begin() as connection:
                        connection.execute(text("UPDATE atlas_current SET load_id='staged' WHERE name='latest'"))
            assert len(json.dumps(rows).encode()) < 5_242_880
            return {"rows": rows}

        monkeypatch.setattr(ensure_current_projections, "atlas_applet_query", lambda: query)
        selection = ensure_current_projections.atlas_catalog_selection()
        pins = ensure_current_projections.served_declaration_pins(selection)
        assert [pin.source_id for pin in pins] == ["source"] + [f"source-{n:03d}" for n in range(1, 104)]
        assert pages == [50, 50, 4]
        projected = pins[0]
        assert projected.dataset_id == "dataset"
        assert projected.dataset_revision == "revision"
        assert projected.pipeline == pipeline
        assert projected.actor_binding.model_dump(mode="json", exclude_unset=True) == actor
        curriculum = ensure_current_projections.atlas_curriculum_rows()
        assert ensure_current_projections.curriculum_state(projected, curriculum) == ("current", None)
    finally:
        engine.dispose()
