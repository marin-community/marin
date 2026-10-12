# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from finestore.reader import ReadView

from experiments.post_training.task_curation.run_catalog import ManifestEvidence, SourceDocuments, append_run_catalog


def test_catalog_preserves_evidence_and_dispositions_across_reused_paths(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = {
        "source": "numbers",
        "source_revision": "pinned",
        "mode": "full",
        "status": "incomplete",
        "raw_population_count": 1000,
        "processed_rows": 1000,
        "unprocessed_rows": 0,
        "admission_counts": {"admitted": 700, "rejected": 200, "deferred": 100},
        "verification": {"status": "inconclusive", "sample_count": 100, "counts": {"infra_error": 1}},
        "datasets": {"final": str(artifact / "final")},
    }
    original = json.dumps(manifest, indent=2).encode()
    (artifact / "manifest.json").write_bytes(original)
    (artifact / "telemetry.json").write_text(
        json.dumps(
            {
                "attempt_started_at": "2026-10-11T10:00:00+00:00",
                "source_wall_seconds": 20.5,
                "status": "completed",
                "phases": [{"phase": "verification", "status": "completed", "wall_seconds": 5.0, "executions": []}],
            }
        )
    )
    report = {
        "status": "failed",
        "mode": "full",
        "updated_at": "2026-10-11T10:01:00+00:00",
        "counts": {"failed": 1},
        "sources": [{"name": "numbers", "path": str(artifact), "status": "failed", "error": "controls incomplete"}],
    }
    root = str(tmp_path / "catalog")
    token = append_run_catalog(
        root,
        json.dumps(report).encode(),
        run_id="first",
        report_path="report.json",
        source_ids={"numbers": "fixture:numbers"},
        started_at="2026-10-11T10:00:00+00:00",
    )
    pinned = ReadView(root)
    assert pinned.token == token
    row = pinned.point("sources", run_id="first", name="numbers")
    assert row["processed_rows"] == 1000
    assert (row["admitted"], row["rejected"], row["deferred"], row["no_grader"]) == (700, 200, 100, 0)
    assert row["status"] == "incomplete"
    assert row["verification_status"] == "inconclusive"
    assert row["controls_infra_error"] == 1
    assert row["atlas_id"] == "fixture:numbers"
    assert row["manifest_sha256"] == hashlib.sha256(original).hexdigest()
    snapshot = json.loads(row["evidence_snapshots_json"])["manifest.json"]
    assert pinned.resolve(snapshot) == original
    assert pinned.point("phases", run_id="first", name="numbers", ordinal=0)["wall_seconds"] == 5.0

    (artifact / "manifest.json").write_text(json.dumps({**manifest, "admission_counts": {"admitted": 1000}}))
    report["status"] = "completed"
    report["sources"][0]["status"] = "completed"
    append_run_catalog(
        root,
        json.dumps(report).encode(),
        run_id="second",
        report_path="report.json",
        source_ids={"numbers": "fixture:numbers"},
        started_at="2026-10-11T11:00:00+00:00",
    )
    latest = ReadView(root)
    assert len(list(latest.iter_rows("runs"))) == 2
    assert pinned.point("sources", run_id="second", name="numbers") is None
    assert latest.point("sources", run_id="first", name="numbers")["admitted"] == 700
    assert latest.resolve(snapshot) == original
    reused = latest.point("sources", run_id="second", name="numbers")
    assert reused["admitted"] == 1000
    assert reused["attempt_started_at"] == "2026-10-11T10:00:00+00:00"
    assert reused["source_wall_seconds"] == 20.5


def test_catalog_imports_saved_attempt_without_loading_current_artifact(tmp_path):
    original = b'{"mode":"sample","processed_rows":100,"raw_population_count":9000,"status":"sampled"}'
    report = {
        "status": "completed",
        "mode": "sample",
        "updated_at": "2026-10-11T10:00:00+00:00",
        "counts": {"sampled": 1},
        "sources": [{"name": "historic", "path": "gs://missing/old-attempt", "status": "sampled"}],
    }
    root = str(tmp_path / "catalog")
    append_run_catalog(
        root,
        json.dumps(report).encode(),
        run_id="historic",
        report_path="saved-report.json",
        source_ids={"historic": "fixture:historic"},
        documents=(SourceDocuments("historic", {"manifest.json": original}, ManifestEvidence.SAVED_SNAPSHOT),),
    )
    view = ReadView(root)
    row = view.point("sources", run_id="historic", name="historic")
    assert row["mode"] == "sample"
    assert (row["raw_population_count"], row["processed_rows"]) == (9000, 100)
    assert row["admitted"] is None
    assert row["controls_passed"] is None
    assert row["source_wall_seconds"] is None
    assert view.point("runs", run_id="historic")["started_at"] is None
    with pytest.raises(ValueError, match="already exists"):
        append_run_catalog(
            root,
            json.dumps(report).encode(),
            run_id="historic",
            report_path="saved-report.json",
            source_ids={"historic": "fixture:historic"},
            documents=(SourceDocuments("historic", {"manifest.json": b"{}"}, ManifestEvidence.SAVED_SNAPSHOT),),
        )
    assert ReadView(root).point("sources", run_id="historic", name="historic")["processed_rows"] == 100


def test_failed_retry_retains_old_manifest_without_attributing_its_counts_to_retry(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    original = b'{"mode":"quick","input_rows":100,"status":"completed"}'
    (artifact / "manifest.json").write_bytes(original)
    partial = b'{"attempt_started_at":'
    (artifact / "telemetry.json").write_bytes(partial)
    report = {
        "status": "failed",
        "mode": "quick",
        "updated_at": "2026-10-11T11:00:01+00:00",
        "counts": {"failed": 1},
        "sources": [{"name": "retry", "path": str(artifact), "status": "failed", "error": "output already exists"}],
    }
    root = str(tmp_path / "catalog")
    append_run_catalog(
        root,
        json.dumps(report).encode(),
        run_id="retry",
        report_path="report.json",
        source_ids={"retry": "fixture:retry"},
        started_at="2026-10-11T11:00:00+00:00",
    )
    view = ReadView(root)
    row = view.point("sources", run_id="retry", name="retry")
    assert row["manifest_evidence"] == "unconfirmed"
    assert row["status"] is None
    assert row["processed_rows"] is None
    assert view.resolve(json.loads(row["evidence_snapshots_json"])["manifest.json"]) == original
    assert view.resolve(json.loads(row["evidence_snapshots_json"])["telemetry.json"]) == partial
    assert "telemetry.json" in json.loads(row["evidence_errors_json"])
