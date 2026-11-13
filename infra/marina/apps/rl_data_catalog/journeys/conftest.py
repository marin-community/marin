# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed lexical evidence in the installed app's disposable Postgres schema."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from finestore.rl.curation import SourceDeclaration
from marina.applets import applet_schema
from marina.db import DatabaseSpec, engine_for
from marina.journeys import Kernel, running_kernel
from marina.preview import install_preview_sources
from marina.server import MarinaConfig
from sqlalchemy import text

from infra.marina.apps.rl_data_catalog.server.app import migrate
from infra.marina.apps.rl_data_catalog.server.quality import review_completeness
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
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
                if declaration.id == "Marin:clean":
                    declaration = SourceDeclaration.model_validate(
                        {
                            **declaration.model_dump(mode="json"),
                            "daytona_safety": "unsafe",
                            "daytona_snapshot_count": 11,
                            "daytona_machine_slots": 3,
                            "daytona_execution_failures": 1,
                            "daytona_count_basis": "Complete actor census",
                            "daytona_unsafe_reasons": ["too_many_snapshots", "unbuilt_image"],
                            "daytona_unsafe_details": {
                                "too_many_snapshots": "Eleven distinct actor snapshots",
                                "unbuilt_image": "One actor declares an unbuilt image",
                            },
                            "daytona_unsafe_counts": {"too_many_snapshots": 3, "unbuilt_image": 1},
                        }
                    )
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
        yield
    finally:
        engine.dispose()
