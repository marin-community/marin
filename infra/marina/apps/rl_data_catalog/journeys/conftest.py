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
from infra.marina.apps.rl_data_catalog.tests.backend_fixtures import backend_source_maps
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.difficulty_fixtures import difficulty_measurement
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import filtered_declaration
from infra.marina.apps.rl_data_catalog.tests.review_fixtures import pending_review_declaration, review_declarations

APP_ID = UUID("fb11c931-5861-4878-8bb5-a964d652b45f")
ACTOR_INPUT = {
    "dataset_id": "fixture",
    "dataset_revision": "a" * 40,
    "converter_version": "1",
    "converter_revision": "1",
    "measured_tasks": 3,
    "task_specs_sha256": "f" * 64,
    "evidence_url": "https://example.org/actor-census.json",
}


@pytest.fixture(scope="session")
def marina_kernel(marina_database: DatabaseSpec, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Kernel]:
    root = tmp_path_factory.mktemp("atlas-browser")
    apps = root / "apps"
    shutil.copytree(Path(__file__).resolve().parents[1], apps / "rl_data_catalog")
    (apps / "rl_data_catalog/dist/measurements/oracle-inventory.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": source_id,
                        "input": ACTOR_INPUT,
                        "oracle_tasks": 0,
                        "execution_qualification": "not_assessed",
                        "evidence_url": "https://example.org/oracle-inventory.json",
                    }
                    for source_id in ["Marin:clean", "Marin:stale-clean"]
                ]
            }
        )
    )
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
                (False, filtered_declaration(0)),
                (False, filtered_declaration(1)),
            ]:
                if declaration.id == "Marin:unmeasured":
                    declaration = declaration.model_copy(update={"pipeline": None})
                if declaration.name in {"clean", "eval"}:
                    declaration = with_declaration_admission_census(declaration)
                if declaration.id == "Marin:clean":
                    declaration = SourceDeclaration.model_validate(
                        {
                            **declaration.model_dump(mode="json"),
                            "daytona_safety": "unsafe",
                            "actor_census_input": ACTOR_INPUT,
                            "grading_environment": {
                                "assessed_tasks": 3,
                                "required_counts": {"separate": 3},
                                "declared_counts": {"separate": 3},
                                "status_counts": {"matched": 3},
                                "reason_counts": {"Independent grader": 3},
                                "evidence_url": "https://example.org/grading.json",
                            },
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
                if declaration.id == "Marin:stale-clean":
                    declaration = SourceDeclaration.model_validate(
                        {
                            **declaration.model_dump(mode="json"),
                            "actor_census_input": {**ACTOR_INPUT, "dataset_revision": declaration.dataset_revision},
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
            for source in backend_source_maps():
                connection.execute(
                    text(
                        "INSERT INTO atlas_backend_sources (load_id,source_id,capabilities) "
                        "VALUES ('lexical',:source,CAST(:capabilities AS JSONB))"
                    ),
                    {"source": source.source_id, "capabilities": source.model_dump_json()},
                )
            connection.execute(
                text("INSERT INTO atlas_backend_surveys VALUES ('lexical','survey-fixture',CAST(:survey AS JSONB))"),
                {
                    "survey": json.dumps(
                        {
                            "catalog_root": "fixture",
                            "snapshot_id": "lexical",
                            "generated_at": "2026-11-14T10:00:00Z",
                            "git_revision": "b" * 40,
                            "schema_sha256": "c" * 64,
                            "runtime_defaults": {"agent": {"backend": "daytona"}, "grader": {"backend": "local"}},
                            "deployment_limits": {},
                            "local_disk_pool": None,
                        }
                    )
                },
            )
            connection.execute(text("UPDATE atlas_current SET load_id='lexical' WHERE name='latest'"))
            for payload in connection.execute(
                text(
                    "SELECT declaration FROM atlas_sources WHERE load_id='lexical' "
                    "AND id IN ('Marin:clean','Marin:eval')"
                )
            ).scalars():
                declaration = SourceDeclaration.model_validate(payload)
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
        with engine.begin() as connection:
            nodes = [
                ("D05", "subject:D05", None, "subject", "Earth, Atmospheric, Ocean & Environmental Sciences", 7088),
                ("D05", "earth", "subject:D05", "group", "Earth, atmospheric, ocean, and environmental sciences", 10),
                ("D05", "climate", "earth", "capability", "Climate models", 10),
                ("D01", "subject:D01", None, "subject", "Mathematics & Statistics", 24),
                ("D01", "arithmetic", "subject:D01", "group", "Arithmetic", 12),
                ("D01", "add", "arithmetic", "capability", "Addition", 12),
                ("D01", "geometry", "subject:D01", "group", "Geometry", 8),
                ("D01", "angle", "geometry", "capability", "Angles", 8),
            ]
            manifest = {
                "created_at": "2026-11-14T00:00:00Z",
                "population": "converted",
                "sample_size": None,
                "curriculum_key": "fixture",
                "lineage": {},
                "coverage": {
                    "subject_distance": 0.45,
                    "capability_distance": 0.25,
                    "sample_task_count": 0,
                    "source_sample_size": 100,
                    "area_sample_size": 10,
                },
                "nodes": [
                    {
                        "subject_id": subject,
                        "node_id": node,
                        "parent_id": parent,
                        "kind": kind,
                        "name": name,
                        "description": name,
                    }
                    for subject, node, parent, kind, name, _ in nodes
                ],
            }
            connection.execute(
                text(
                    "INSERT INTO curriculum_loads (load_id,release_id,release_uri,manifest) "
                    "VALUES ('counts','counts','fixture',CAST(:manifest AS JSONB))"
                ),
                {"manifest": json.dumps(manifest)},
            )
            connection.execute(text("UPDATE curriculum_current SET load_id='counts' WHERE name='latest'"))
            connection.execute(
                text(
                    "INSERT INTO curriculum_sources VALUES "
                    "('counts','Marin:curriculum','{\"display_name\":\"Count fixture\"}'::jsonb,'fixture',"
                    "CAST(:alignment AS JSONB))"
                ),
                {
                    "alignment": json.dumps(
                        {
                            "basis": "matched",
                            "source_id": "Marin:curriculum",
                            "intended_use": "train",
                            "task_count": 7112,
                            "input_rows": None,
                        }
                    )
                },
            )
            for subject, node, _, _, _, count in nodes:
                connection.execute(
                    text(
                        "INSERT INTO curriculum_statistics VALUES "
                        "('counts','Marin:curriculum','placement',:subject,:node,:count)"
                    ),
                    {"subject": subject, "node": node, "count": count},
                )
            for metric, count in [("denominator", 7112), ("assigned", 30), ("subject_only", 7082)]:
                connection.execute(
                    text(
                        "INSERT INTO curriculum_statistics VALUES " "('counts','Marin:curriculum',:metric,'','',:count)"
                    ),
                    {"metric": metric, "count": count},
                )
        yield
    finally:
        engine.dispose()
