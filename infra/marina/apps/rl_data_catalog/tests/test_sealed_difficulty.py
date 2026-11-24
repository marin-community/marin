# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from finestore.rl.difficulty import MeasurementStatus

from experiments.post_training.task_curation.difficulty.importer import declaration_population_revision
from experiments.post_training.task_curation.difficulty.ledger import append_rows
from experiments.post_training.task_curation.difficulty.seal import seal_ledger
from experiments.post_training.task_curation.difficulty.state import CatalogSource
from experiments.post_training.task_curation.tests.test_difficulty_seal import attempt
from infra.marina.apps.rl_data_catalog.load_difficulty_seal import load_difficulty_seal
from infra.marina.apps.rl_data_catalog.server.difficulty import table_difficulty
from infra.marina.apps.rl_data_catalog.server.difficulty_coverage import difficulty_coverage
from infra.marina.apps.rl_data_catalog.server.sealed_difficulty import sealed_difficulty_summary
from infra.marina.apps.rl_data_catalog.tests.test_difficulty_coverage import declaration


def test_sealed_partial_and_stale_cells_never_gain_difficulty_values():
    source = declaration()
    summary = {
        "protocol_id": "atlas-difficulty-v3",
        "models": [
            {"solver": solver, "status": MeasurementStatus.NOT_MEASURED.value, "n": 0, "solved": 0, "solve_rate": None}
            for solver in ("small", "large", "hosted", "snowball")
        ],
    }
    cell = {
        "source_id": source["id"],
        "registry_revision": declaration_population_revision(source),
        "solver": "small",
        "protocol_id": "atlas-difficulty-v3",
        "measured": False,
        "difficulty": None,
        "pass_rate": None,
        "scored_attempts": 7,
        "solved_attempts": 6,
        "gaps": [{"reason": "GRADER_INFRASTRUCTURE_FAILURE"}],
    }
    record = {"snapshot_id": "seal", "uri": "s3://bucket/seal/manifest.json", "sha256": "a" * 64, "cells": [cell]}
    result = table_difficulty(sealed_difficulty_summary(summary, source, record))
    small = result["models"][0]
    assert small["status"] == MeasurementStatus.NOT_MEASURED.value and small["solve_rate"] is None
    assert small["n"] == 7 and small["sealed_snapshot_id"] == "seal"
    assert result["sealed_snapshot"]["sha256"] == "a" * 64
    coverage = difficulty_coverage({source["id"]: source}, {}, {source["id"]: record})
    assert coverage.counts["blocked"] == 1 and coverage.counts["solvers_measured"] == 0
    cell.update(measured=True, difficulty=0.25, pass_rate=0.75, gaps=[])
    assert sealed_difficulty_summary(summary, source, record)["models"][0]["solve_rate"] == 0.75
    cell["registry_revision"] = "stale-population"
    assert sealed_difficulty_summary(summary, source, record)["models"][0]["solve_rate"] is None


def test_release_loader_reads_back_cells_and_rejects_changed_snapshot_identity(tmp_path):

    ledger = str(tmp_path / "ledger")
    append_rows(ledger, [attempt("a", "a"), attempt("b", "b")])
    source = CatalogSource("source", "population", (("a", "a" * 64), ("b", "b" * 64)), "queue", "digest")
    reference = seal_ledger(ledger, [source], str(tmp_path / "sealed"), {}, "2026-11-24T18:00:00-05:00")
    snapshots, cells, loads = {}, {}, {}

    def query(sql, parameters):
        if sql.strip().startswith("CREATE TABLE"):
            return {"row_count": 0}
        if sql.startswith("INSERT INTO atlas_difficulty_seals("):
            snapshots.setdefault(
                parameters["snapshot_id"],
                {
                    "manifest_uri": parameters["uri"],
                    "manifest_sha256": parameters["sha256"],
                    "manifest": json.loads(parameters["manifest"]),
                },
            )
            return {"row_count": 1}
        if sql.startswith("SELECT manifest_uri"):
            return {"rows": [snapshots[parameters["snapshot_id"]]]}
        if sql.startswith("INSERT INTO atlas_difficulty_seal_cells("):
            for row in json.loads(parameters["rows"]):
                cells.setdefault((parameters["snapshot_id"], row["source_id"], row["solver"]), row["cell"])
            return {"row_count": 1}
        if sql.startswith("SELECT c.source_id,c.solver,c.cell"):
            return {
                "rows": [
                    {
                        "source_id": row["source_id"],
                        "solver": row["solver"],
                        "cell": cells[(parameters["snapshot_id"], row["source_id"], row["solver"])],
                    }
                    for row in json.loads(parameters["keys"])
                ]
            }
        if sql.startswith("INSERT INTO atlas_difficulty_seal_loads("):
            loads.setdefault(parameters["load_id"], parameters["snapshot_id"])
            return {"row_count": 1}
        if sql.startswith("SELECT snapshot_id FROM atlas_difficulty_seal_loads"):
            return {"rows": [{"snapshot_id": loads[parameters["load_id"]]}]}
        raise AssertionError(sql)

    result = load_difficulty_seal(reference, query, "r41-load")
    assert result["covered_cells"] == 1 and len(cells) == 4
    assert loads["r41-load"] == result["snapshot_id"]
    snapshots[result["snapshot_id"]]["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="different evidence"):
        load_difficulty_seal(reference, query, "r41-load")
