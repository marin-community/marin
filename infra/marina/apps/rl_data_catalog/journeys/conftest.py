# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed lexical evidence in the installed app's disposable Postgres schema."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from finestore.rl.difficulty import Solver
from marina.applets import applet_schema
from marina.db import DatabaseSpec, engine_for
from marina.journeys import Kernel, running_kernel
from marina.preview import install_preview_sources
from marina.server import MarinaConfig
from sqlalchemy import text

from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from infra.marina.apps.rl_data_catalog.server.app import migrate
from infra.marina.apps.rl_data_catalog.server.quality import review_completeness
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.difficulty_fixtures import difficulty_measurement
from infra.marina.apps.rl_data_catalog.tests.review_fixtures import pending_review_declaration, review_declarations

APP_ID = UUID("fb11c931-5861-4878-8bb5-a964d652b45f")


@pytest.fixture(scope="session")
def marina_kernel(marina_database: DatabaseSpec, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Kernel]:
    root = tmp_path_factory.mktemp("atlas-browser")
    apps = root / "apps"
    shutil.copytree(Path(__file__).resolve().parents[1], apps / "rl_data_catalog")
    install_preview_sources(apps, marina_database)
    with running_kernel(MarinaConfig(data_root=str(root), iap_audience=None, database=marina_database)) as kernel:
        yield kernel


@pytest.fixture(scope="session", autouse=True)
def lexical_catalog(marina_kernel: Kernel, marina_database: DatabaseSpec) -> Iterator[None]:
    engine = engine_for(marina_database, applet_schema(APP_ID))
    try:
        with engine.begin() as connection:
            migrate(connection)
            connection.execute(text("INSERT INTO atlas_loads VALUES ('lexical','fixture','lexical',NOW(),NOW())"))
            for _, declaration in [
                *contamination_declarations(),
                *review_declarations(),
                (False, pending_review_declaration()),
            ]:
                if declaration.name in {"clean", "eval"}:
                    declaration = with_declaration_admission_census(declaration)
                connection.execute(
                    text(
                        "INSERT INTO atlas_sources (load_id,id,declaration,latest_attempt_status,"
                        "diagnostics,review_completeness) "
                        "VALUES ('lexical',:id,CAST(:declaration AS JSONB),'skipped','{}'::jsonb,"
                        "CAST(:completeness AS JSONB))"
                    ),
                    {
                        "id": declaration.id,
                        "declaration": declaration.model_dump_json(),
                        "completeness": json.dumps(review_completeness(declaration, [], [])),
                    },
                )
            connection.execute(text("UPDATE atlas_current SET load_id='lexical' WHERE name='latest'"))
            for _, declaration in contamination_declarations():
                if declaration.name not in {"clean", "eval"}:
                    continue
                declaration = with_declaration_admission_census(declaration)
                for solver in Solver:
                    solved = 3 if (declaration.name == "clean") == (solver == Solver.SMALL) else 1
                    measurement = difficulty_measurement(declaration, solver, solved)
                    connection.execute(
                        text(
                            "INSERT INTO atlas_difficulty VALUES (:id,:source,:protocol,:created,CAST(:value AS JSONB))"
                        ),
                        {
                            "id": measurement.measurement_id,
                            "source": declaration.id,
                            "protocol": measurement.protocol_id,
                            "created": measurement.created_at,
                            "value": measurement.model_dump_json(),
                        },
                    )
        yield
    finally:
        engine.dispose()
