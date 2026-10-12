# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from finestore.reader import ReadView

from experiments.post_training.task_curation.curation_catalog import (
    ManifestEvidence,
    SourceDocuments,
    append_curation_catalog,
)


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
    token = append_curation_catalog(
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
    append_curation_catalog(
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
    append_curation_catalog(
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
        append_curation_catalog(
            root,
            json.dumps(report).encode(),
            run_id="historic",
            report_path="saved-report.json",
            source_ids={"historic": "fixture:historic"},
            documents=(SourceDocuments("historic", {"manifest.json": b"{}"}, ManifestEvidence.SAVED_SNAPSHOT),),
        )
    assert ReadView(root).point("sources", run_id="historic", name="historic")["processed_rows"] == 100


@pytest.mark.parametrize("status", ["failed", "running"])
def test_failed_retry_retains_old_manifest_without_attributing_its_counts_to_retry(tmp_path, status):
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
        "counts": {status: 1},
        "sources": [{"name": "retry", "path": str(artifact), "status": status, "error": "output already exists"}],
    }
    root = str(tmp_path / "catalog")
    append_curation_catalog(
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


def test_interrupted_running_source_retains_partial_phase_evidence(tmp_path):
    artifact = tmp_path / "interrupted"
    artifact.mkdir()
    telemetry = json.dumps(
        {
            "attempt_started_at": "2026-10-11T11:00:00+00:00",
            "status": "failed",
            "error_type": "KeyboardInterrupt",
            "source_wall_seconds": 1.0,
            "phases": [
                {
                    "phase": "full_prepare",
                    "status": "failed",
                    "wall_seconds": 1.0,
                    "error_type": "KeyboardInterrupt",
                    "executions": [],
                }
            ],
        }
    ).encode()
    (artifact / "telemetry.json").write_bytes(telemetry)
    report = {
        "status": "failed",
        "mode": "full",
        "updated_at": "2026-10-11T11:00:01+00:00",
        "counts": {"running": 1},
        "sources": [{"name": "interrupted", "path": str(artifact), "status": "running"}],
    }
    root = str(tmp_path / "catalog")
    append_curation_catalog(
        root,
        json.dumps(report).encode(),
        run_id="interrupted",
        report_path="report.json",
        source_ids={"interrupted": "fixture:interrupted"},
        started_at="2026-10-11T11:00:00+00:00",
    )
    view = ReadView(root)
    row = view.point("sources", run_id="interrupted", name="interrupted")
    assert row["processed_rows"] is None
    assert row["telemetry_error_type"] == "KeyboardInterrupt"
    assert view.resolve(json.loads(row["evidence_snapshots_json"])["telemetry.json"]) == telemetry
    phase = view.point("phases", run_id="interrupted", name="interrupted", ordinal=0)
    assert (phase["status"], phase["error_type"]) == ("failed", "KeyboardInterrupt")


@pytest.mark.parametrize("status", ["completed", "failed", "skipped"])
def test_catalog_retains_assessment_and_upstream_evidence_at_distinct_paths(tmp_path, status):
    curated, inputs, judgments, report_root = (tmp_path / name for name in ("curated", "inputs", "judgments", "report"))
    for path in (curated, inputs, judgments):
        path.mkdir()
    manifest = {"mode": "full", "status": "completed", "processed_rows": 30, "admission_counts": {"admitted": 28}}
    (curated / "manifest.json").write_text(json.dumps(manifest))
    (curated / ".artifact.json").write_text('{"result":{"status":"completed"}}')
    (inputs / "manifest.json").write_text('{"sample_count":5}')
    partial = b'{"counts":{"reviewed":4,"unavailable":1}}'
    (judgments / "manifest.json").write_bytes(partial)
    paths = {"manifest": str(curated / "manifest.json")}
    if status != "skipped":
        paths.update(
            assessment_inputs=str(inputs),
            assessment_judgments=str(judgments),
            assessment_report=str(report_root / "report.json"),
        )
    if status == "completed":
        report_root.mkdir()
        (report_root / "report.json").write_text('{"assessment":{"admitted":{"grade":"good","sample_count":3}}}')
    outcome = {
        "name": "aime24",
        "path": str(report_root),
        "status": "completed" if status == "skipped" else status,
        "pipeline_key": "rl-assessment/aime24/report@1:abc",
        "planned_evidence": paths,
    }
    report = {
        "status": "failed" if status == "failed" else "completed",
        "mode": "full",
        "updated_at": "2026-10-12T20:00:00+00:00",
        "counts": {outcome["status"]: 1},
        "sources": [outcome],
    }
    root = str(tmp_path / "catalog")
    append_curation_catalog(
        root,
        json.dumps(report).encode(),
        run_id="canary",
        report_path="campaign.json",
        source_ids={"aime24": "source:aime24"},
    )
    view = ReadView(root)
    source = view.point("sources", run_id="canary", name="aime24")
    assessment = view.point("assessment_runs", run_id="canary", name="aime24")
    assert source["manifest_path"] == str(curated / "manifest.json")
    assert source["admitted"] == 28
    assert assessment["status"] == {"completed": "complete", "failed": "failed", "skipped": "skipped"}[status]
    assert assessment["pipeline_key"] == outcome["pipeline_key"]
    if status == "completed":
        assert json.loads(assessment["report_json"])["assessment"]["admitted"]["sample_count"] == 3
    if status == "failed":
        assert assessment["report_json"] is None
        snapshot = json.loads(assessment["evidence_snapshots_json"])["assessment_judgments/manifest.json"]
        assert view.resolve(snapshot) == partial
