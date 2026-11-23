# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed lexical evidence in the installed app's disposable Postgres schema."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from finestore.rl.curation import ActorCensusInput, QualifiedSourceDeclaration
from finestore.rl.difficulty import Solver
from marina.applets import applet_schema
from marina.db import DatabaseSpec, engine_for
from marina.journeys import Kernel, running_kernel
from marina.preview import install_preview_sources
from marina.server import MarinaConfig
from sqlalchemy import text

from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from infra.marina.apps.rl_data_catalog.server.app import migrate
from infra.marina.apps.rl_data_catalog.server.evidence_content import pack_content
from infra.marina.apps.rl_data_catalog.server.methods import rl_variant
from infra.marina.apps.rl_data_catalog.server.quality import review_completeness
from infra.marina.apps.rl_data_catalog.tests.backend_fixtures import backend_source_maps
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from infra.marina.apps.rl_data_catalog.tests.difficulty_fixtures import difficulty_measurement
from infra.marina.apps.rl_data_catalog.tests.filtration_fixtures import count_evidence, filtered_declaration
from infra.marina.apps.rl_data_catalog.tests.method_fixtures import sft_variant
from infra.marina.apps.rl_data_catalog.tests.review_fixtures import pending_review_declaration, review_declarations

APP_ID = UUID("fb11c931-5861-4878-8bb5-a964d652b45f")


LONG_TOOL_OUTPUT = "\n".join(
    f"drwxr-xr-x  3 root root  4096 Nov 21 0{index % 10}:12 .cache-{index}" for index in range(120)
)


def rollout_fixtures(solver: str) -> list[tuple[dict, dict]]:
    """Synthetic attempt evidence: one tool-using trace, one plain chat, one exception.

    The conversation shapes are the ones the difficulty pipeline stages from Harbor
    trials: typed events (`message`, `assistant_tool_calls`, `tool_result`), not
    provider messages. Attempt 0 exercises every rendered feature; attempt 1 is the
    single-turn shape most of the catalog holds today; attempt 2 carries an exception
    instead of a transcript.
    """
    receipt = {"uri": "s3://fixture/attempt.json", "sha256": "a" * 64}
    archive = {"uri": "s3://fixture/trial.tar.gz", "sha256": "b" * 64}

    def summary(score, status="scored", exception=None):
        return {
            "solver": solver,
            "attempt": "00",
            "score": score,
            "status": status,
            "duration": 2.5,
            "exception": exception,
            "reason": None,
            "has_rollout": True,
        }

    def base_detail(conversation, exception=None):
        return {
            "conversation": conversation,
            "model_input": {"events": conversation["events"][:2]},
            "verifier": {"detail": {"expected": "45", "extracted": "45"}},
            "verifier_result": {"rewards": {"reward": 1.0}, "stderr": None, "stdout": None},
            "exception": exception,
            "exception_info": None,
            "receipt": receipt,
            "archive": archive,
        }

    agentic = base_detail(
        {
            "events": [
                {
                    "type": "message",
                    "role": "system",
                    "content": "You are a careful solver. Work in /workspace and answer precisely.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": (
                        "## Task\n\nCount the tracked files under /workspace and report the total.\n\n"
                        "Steps:\n\n1. list the tree\n2. count the entries\n\n"
                        "Return the count in a fenced block:\n\n```\ncount = <total>\n```\n\n"
                        "Be careful with hidden directories, and explain your reasoning before the final answer. "
                        + "The tree is large, so stream the listing rather than buffering it. " * 6
                        + "Mention the checksum of the listing explicitly in your explanation."
                    ),
                },
                {
                    "type": "assistant_tool_calls",
                    "content": "I'll list the workspace tree first.",
                    "reasoning": (
                        "The task asks for a file count. Listing the tree is the direct way; "
                        "I'll count lines afterwards instead of trusting a single tool."
                    ),
                    "calls": [
                        {
                            "call_id": "call-1",
                            "name": "terminal",
                            "arguments": {"command": "find /workspace -type f | head -200"},
                        },
                        {"call_id": "call-2", "name": "python", "arguments": {"code": "print(sum(range(10)))"}},
                    ],
                },
                {"type": "tool_result", "call_id": "call-1", "content": LONG_TOOL_OUTPUT},
                {"type": "tool_result", "call_id": "call-2", "content": "45"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        "The workspace holds 120 tracked files.\n\n"
                        "```\ncount = 120\n```\n\n"
                        "The remaining tool ran the checksum, which agreed with the listing."
                    ),
                },
            ]
        }
    )
    single_turn = base_detail(
        {
            "events": [
                {"type": "message", "role": "system", "content": "Answer the task directly."},
                {
                    "type": "message",
                    "role": "user",
                    "content": (
                        "Two quantum states have lifetimes $10^{-9}$ s and $10^{-8}$ s. "
                        "Which energy difference lets them be told apart?"
                    ),
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        "**D. $10^{-4}$ eV**\n\n"
                        "The energy uncertainty from finite lifetimes is roughly $\\Delta E \\approx \\hbar/\\tau$, "
                        "and the resolvable splitting is about $1/\\tau_1 - 1/\\tau_2$."
                    ),
                },
            ]
        }
    )
    exception_detail = {
        "conversation": {"events": []},
        "model_input": {"events": []},
        "verifier": None,
        "verifier_result": None,
        "exception": "AgentTimeoutError: the solver exceeded its 300-second budget",
        "exception_info": {"kind": "AgentTimeoutError", "stage": "solve"},
        "receipt": receipt,
        "archive": archive,
    }
    return [
        (summary(1.0), agentic),
        (summary(0.0), single_turn),
        (
            summary(None, status="exception", exception="AgentTimeoutError: the solver exceeded its 300-second budget"),
            exception_detail,
        ),
    ]


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
    # The status extras are a build-time output of the applet's build_command; the staged
    # app is packaged from this copy, so generate the artifact beside the inventory.
    # Imported here rather than at module level: the marina test environment does not
    # install the task-curation dependency chain (dupekit) that this import pulls in.
    from experiments.post_training.task_curation.source_status import status_extras  # noqa: PLC0415

    measurements = apps / "rl_data_catalog/dist/measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    (measurements / "source-status-extras.json").write_text(json.dumps(status_extras(), indent=2) + "\n")
    (apps / "rl_data_catalog/dist/measurements/oracle-inventory.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": source_id,
                        # Bind the census through the model so the measurement serializes
                        # the exact identity the declarations carry, including carry_forward.
                        "input": ActorCensusInput.model_validate(ACTOR_INPUT).model_dump(mode="json"),
                        "oracle_tasks": 0,
                        "execution_qualification": "not_assessed",
                        "evidence_url": "https://example.org/oracle-inventory.json",
                    }
                    for source_id in ["Marin:clean", "Marin:stale-clean"]
                ]
            }
        )
    )
    observations = []
    for _, declaration in contamination_declarations():
        if declaration.id.rpartition(":")[2] not in {"clean", "eval", "stale-clean", "required"}:
            continue
        declaration = with_declaration_admission_census(declaration)
        if declaration.id.rpartition(":")[2] == "clean":
            declaration = QualifiedSourceDeclaration.model_validate(
                {**declaration.model_dump(mode="json"), "actor_census_input": ACTOR_INPUT}
            )
        observation = count_evidence(
            declaration,
            1 if declaration.id.rpartition(":")[2] == "required" else 0,
            None if declaration.id.rpartition(":")[2] == "required" else 1,
        )
        if declaration.id.rpartition(":")[2] == "stale-clean":
            observation = observation.model_copy(update={"input": ActorCensusInput.model_validate(ACTOR_INPUT)})
        observations.append(observation.model_dump(mode="json"))
    payload = json.dumps({"version": 1, "sources": observations})
    (apps / "rl_data_catalog/server/count_evidence.py").write_text(f"SOURCE_COUNTS = {payload!r}\n")
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
                if declaration.id.rpartition(":")[2] in {"clean", "eval", "required"}:
                    declaration = with_declaration_admission_census(declaration)
                if declaration.id == "Marin:clean":
                    declaration = QualifiedSourceDeclaration.model_validate(
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
                    declaration = QualifiedSourceDeclaration.model_validate(
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
            connection.execute(
                text("INSERT INTO atlas_derived_contracts VALUES ('Marin:clean','fixture',CAST(:receipt AS JSONB))"),
                {
                    "receipt": json.dumps(
                        {
                            "source_id": "Marin:clean",
                            "input": ACTOR_INPUT,
                            "scanned_task_specs_sha256": "f" * 64,
                            "evidence_url": "https://example.org/derived-receipt.json",
                            "summary": {
                                "task_count": 3,
                                "grader_type_counts": {"script": 1, "llm-judge": 2},
                                "judge_rubric_counts": {"reference": 2},
                                "judge_reference_count": 2,
                                "judge_no_reference_count": 0,
                                "agentic_count": 3,
                                "single_turn_count": 0,
                                "reply_only_count": 0,
                                "workspace_graded_count": 3,
                            },
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
                declaration = QualifiedSourceDeclaration.model_validate(payload)
                for solver in Solver:
                    solved = 3 if (declaration.id.rpartition(":")[2] == "clean") == (solver == Solver.SMALL) else 1
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
            for measurement in connection.execute(text("SELECT measurement FROM atlas_difficulty")).scalars():
                identity = measurement["measurement_id"]
                solver = measurement["solver"]
                for index, attempt in enumerate(rollout_fixtures(solver)):
                    summary, detail = attempt
                    summary["attempt_id"] = str(index)
                    summary["task_id"] = str(index)
                    summary["trial_id"] = f"trial-{index}"
                    connection.execute(
                        text(
                            "INSERT INTO atlas_rollout_attempts VALUES (:id,:attempt,:task,"
                            "CAST(:summary AS JSONB),CAST(:detail AS JSONB))"
                        ),
                        {
                            "id": identity,
                            "attempt": str(index),
                            "task": str(index),
                            "summary": json.dumps(summary),
                            "detail": json.dumps(pack_content(detail)),
                        },
                    )
                connection.execute(
                    text("INSERT INTO atlas_rollout_sets VALUES (:id,CAST(:metadata AS JSONB))"),
                    {"id": identity, "metadata": json.dumps({"attempt_count": 3, "recorded_attempt_count": 3})},
                )
        with engine.begin() as connection:
            nodes = [
                ("D05", "subject:D05", None, "subject", "Earth, Atmospheric, Ocean & Environmental Sciences", 7088),
                ("D05", "earth", "subject:D05", "group", "Earth, atmospheric, ocean, and environmental sciences", 10),
                ("D05", "climate", "earth", "capability", "Climate models", 10),
                ("D05", "weather", "earth", "capability", "Weather prediction", 0),
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
                    "sample_task_count": 3,
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
                    "('counts','Marin:clean','{\"display_name\":\"Count fixture\"}'::jsonb,'fixture',"
                    "CAST(:alignment AS JSONB))"
                ),
                {
                    "alignment": json.dumps(
                        {
                            "basis": "matched",
                            "source_id": "Marin:clean",
                            "intended_use": "train",
                            "task_count": 7110,
                            "input_rows": None,
                        }
                    )
                },
            )
            connection.execute(
                text(
                    "INSERT INTO curriculum_sources VALUES "
                    "('counts','Marin:eval','{\"display_name\":\"Other fixture\"}'::jsonb,'fixture',"
                    "CAST(:alignment AS JSONB))"
                ),
                {
                    "alignment": json.dumps(
                        {
                            "basis": "matched",
                            "source_id": "Marin:eval",
                            "intended_use": "eval",
                            "task_count": 2,
                            "input_rows": None,
                        }
                    )
                },
            )
            connection.execute(
                text(
                    "INSERT INTO curriculum_statistics VALUES "
                    "('counts','Marin:eval','placement','D05','subject:D05',2),"
                    "('counts','Marin:eval','denominator','','',2),"
                    "('counts','Marin:eval','subject_only','','',2)"
                )
            )
            for subject, node, _, _, _, count in nodes:
                if subject == "D05" and node == "subject:D05":
                    count -= 2
                connection.execute(
                    text(
                        "INSERT INTO curriculum_statistics VALUES "
                        "('counts','Marin:clean','placement',:subject,:node,:count)"
                    ),
                    {"subject": subject, "node": node, "count": count},
                )
            for source_id, task_id, capability, similarity in [
                ("Marin:clean", "earth-only", "climate", 0.6),
                ("Marin:clean", "climate-example", "climate", 0.9),
                ("Marin:eval", "other-source-example", "climate", 0.6),
            ]:
                detail = {
                    "source_id": source_id,
                    "task_id": task_id,
                    "source_key": "fixture",
                    "task_hash": task_id,
                    "content_hash": task_id,
                    "intended_use": "eval" if source_id == "Marin:eval" else "train",
                    "population": "converted",
                    "description": "Earth observation",
                    "reason": "",
                    "task_preview": "Describe the Earth's atmosphere.",
                    "preview_truncated": False,
                    "instruction_characters": 32,
                    "matches": {
                        "subjects": [
                            {
                                "subject_id": "D05",
                                "membership_similarity": 0.9,
                                "candidates": [
                                    {"subject_id": "D05", "capability_id": capability, "similarity": similarity}
                                ],
                            }
                        ],
                        "capabilities": [],
                    },
                }
                connection.execute(
                    text(
                        "INSERT INTO curriculum_tasks VALUES "
                        "('counts',:source,:task,:task,'ready','Earth observation',CAST(:detail AS JSONB))"
                    ),
                    {"source": source_id, "task": task_id, "detail": json.dumps(detail)},
                )
                connection.execute(
                    text(
                        "INSERT INTO curriculum_matches VALUES "
                        "('counts',:source,:task,'D05',:capability,'subject_domain',0.9,:similarity)"
                    ),
                    {"source": source_id, "task": task_id, "capability": capability, "similarity": similarity},
                )
            for ancestor in ["climate", "earth", "subject:D05"]:
                connection.execute(
                    text("INSERT INTO curriculum_ancestors VALUES ('counts','D05','climate',:ancestor)"),
                    {"ancestor": ancestor},
                )
            for metric, count in [("denominator", 7110), ("assigned", 30), ("subject_only", 7080)]:
                connection.execute(
                    text("INSERT INTO curriculum_statistics VALUES " "('counts','Marin:clean',:metric,'','',:count)"),
                    {"metric": metric, "count": count},
                )
            confidence_levels = [
                {"label": label, "subject_min_similarity": subject, "capability_min_similarity": capability}
                for label, subject, capability in [
                    ("All (nearest)", 0.3, 0.3),
                    ("Broad", 0.55, 0.6),
                    ("Moderate", 0.55, 0.65),
                    ("Strict", 0.55, 0.7),
                    ("Original", 0.55, 0.75),
                ]
            ]
            confidence_manifest = {
                **manifest,
                "confidence": {"levels": confidence_levels, "reference": {"uri": "fixture", "sha256": "a" * 64}},
            }
            connection.execute(
                text(
                    "INSERT INTO curriculum_loads (load_id,release_id,release_uri,manifest) VALUES "
                    "('confidence','confidence','fixture',CAST(:manifest AS JSONB))"
                ),
                {"manifest": json.dumps(confidence_manifest)},
            )
            connection.execute(text("UPDATE curriculum_current SET previous_load_id='confidence' WHERE name='latest'"))
            for table in ("sources", "nodes", "ancestors", "tasks", "matches", "statistics"):
                columns = [
                    row[0]
                    for row in connection.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns WHERE "
                            "table_schema=current_schema() AND table_name=:table ORDER BY ordinal_position"
                        ),
                        {"table": "curriculum_" + table},
                    )
                    if row[0] != "load_id"
                ]
                names = ",".join(columns)
                connection.execute(
                    text(
                        f"INSERT INTO curriculum_{table} (load_id,{names}) SELECT 'confidence',{names} FROM "
                        f"curriculum_{table} WHERE load_id='counts'"
                    )
                )
            vectors = []
            for row in connection.execute(text("SELECT * FROM curriculum_statistics WHERE load_id='counts'")).mappings():
                broad = row["instances"]
                if row["metric"] == "assigned":
                    broad = 7110 if row["source_id"] == "Marin:clean" else 2
                elif row["metric"] == "subject_only":
                    broad = 0
                elif row["metric"] == "placement" and row["subject_id"] == "D05" and row["node_id"] != "weather":
                    broad = 7086 if row["source_id"] == "Marin:clean" else 2
                elif row["metric"] == "capability":
                    broad = 7086 if row["subject_id"] == "D05" else 24
                vector = [broad, broad, row["instances"], row["instances"], row["instances"]]
                vectors.append(
                    {
                        "scope": row["source_id"],
                        "metric": row["metric"],
                        "subject": row["subject_id"],
                        "node": row["node_id"],
                        "values": vector,
                    }
                )
            for metric, subject, node, values in [
                ("assigned", "", "", [2, 2, 0, 0, 0]),
                ("placement", "D05", "earth", [2, 2, 0, 0, 0]),
                ("placement", "D05", "climate", [2, 2, 0, 0, 0]),
            ]:
                vectors.append(
                    {"scope": "Marin:eval", "metric": metric, "subject": subject, "node": node, "values": values}
                )
            for scope, use in [("@all", None), ("@train", "Marin:clean"), ("@eval", "Marin:eval")]:
                combined = {}
                for row in vectors:
                    if use and row["scope"] != use:
                        continue
                    key = (row["metric"], row["subject"], row["node"])
                    combined[key] = [a + b for a, b in zip(combined.get(key, [0] * 5), row["values"], strict=True)]
                for (metric, subject, node), values in combined.items():
                    connection.execute(
                        text(
                            "INSERT INTO curriculum_confidence VALUES "
                            "('confidence',:scope,:metric,:subject,:node,:values)"
                        ),
                        {"scope": scope, "metric": metric, "subject": subject, "node": node, "values": values},
                    )
            for row in vectors:
                connection.execute(
                    text(
                        "INSERT INTO curriculum_confidence VALUES ('confidence',:scope,:metric,:subject,:node,:values)"
                    ),
                    row,
                )
            migrate(connection)
        yield
    finally:
        engine.dispose()


@pytest.fixture
def readability_catalog(lexical_catalog: None, marina_database: DatabaseSpec) -> Iterator[None]:
    """Include frozen review packets and an archive in the readability walk."""
    engine = engine_for(marina_database, applet_schema(APP_ID))
    with engine.begin() as connection:
        method_row = (
            connection.execute(
                text(
                    "SELECT declaration,method_variants FROM atlas_sources WHERE load_id='lexical' AND id='Marin:clean'"
                )
            )
            .mappings()
            .one()
        )
        prior_variants = method_row["method_variants"]
        variants = [
            rl_variant(method_row["declaration"]).model_dump(mode="json"),
            sft_variant("Marin:clean").model_dump(mode="json"),
        ]
        connection.execute(
            text(
                "UPDATE atlas_sources SET method_variants=CAST(:variants AS jsonb) "
                "WHERE load_id='lexical' AND id='Marin:clean'"
            ),
            {"variants": json.dumps(variants)},
        )
        connection.execute(
            text("UPDATE atlas_sources SET assessment_key='readability' WHERE load_id='lexical' AND id='Marin:clean'")
        )
        connection.execute(
            text(
                """
            INSERT INTO atlas_assessments VALUES ('lexical','readability','Marin:clean','sample',
                '{"model": "fixture","revision": "pinned"}','Assess instruction clarity.',
                '{"status": "complete","verification": null,"admission_counts": {"admitted": 1,"excluded": 1}}',
                'https://example.org/curation.json','[]')
        """
            )
        )
        for cohort, verdict in [("admitted", "good"), ("excluded", "bad")]:
            connection.execute(
                text("INSERT INTO atlas_populations VALUES ('lexical','readability',:cohort,1,1)"), {"cohort": cohort}
            )
            connection.execute(
                text(
                    """
                INSERT INTO atlas_items VALUES ('lexical','readability',:task,:cohort,'reviewed',:verdict,
                    'The instructions are complete; check the reference evidence.', '["instruction"]',
                    'https://example.org/packet.json',true,'[]',NULL)
            """
                ),
                {"task": cohort, "cohort": cohort, "verdict": verdict},
            )
            connection.execute(
                text(
                    """
                INSERT INTO atlas_packets VALUES ('lexical','readability',:task,
                    '{"instructions": "Describe the Earth atmosphere.",
                      "repair_recommendation": "Clarify the requested output."}')
            """
                ),
                {"task": cohort},
            )
        connection.execute(
            text(
                """
            INSERT INTO catalog_reviews VALUES ('readability','Marin:clean',
                '{"quality": "good","reviews": [],"repair_recommendation": "Keep instructions explicit."}',NOW())
        """
            )
        )
        connection.execute(
            text(
                """
            INSERT INTO review_artifacts VALUES ('readability','trace.txt','Stored readability evidence','fixture')
        """
            )
        )
    try:
        with engine.begin() as connection:
            migrate(connection)
        yield
    finally:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE atlas_sources SET method_variants=CAST(:variants AS jsonb) "
                    "WHERE load_id='lexical' AND id='Marin:clean'"
                ),
                {"variants": json.dumps(prior_variants) if prior_variants is not None else None},
            )
            connection.execute(
                text("UPDATE atlas_sources SET assessment_key=NULL WHERE load_id='lexical' AND id='Marin:clean'")
            )
            for table in ["atlas_packets", "atlas_items", "atlas_populations", "atlas_assessments"]:
                connection.execute(text(f"DELETE FROM {table} WHERE load_id='lexical' AND pipeline_key='readability'"))
            connection.execute(text("DELETE FROM review_artifacts WHERE review_id='readability'"))
            connection.execute(text("DELETE FROM catalog_reviews WHERE id='readability'"))
        engine.dispose()
