# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import tomllib
from pathlib import Path
from uuid import UUID

import pytest
from finestore.rl.rl_progress import AttemptSummary

APPLET_ID = tomllib.loads((Path(__file__).parents[1] / "apps/rl_progress/applet.toml").read_text())["applet_id"]


@pytest.fixture
def rl_progress_client(tmp_path, monkeypatch):
    from marina.applets import AppletStore, package_applet, read_applet_package  # noqa: PLC0415
    from marina.database_setup import ensure_applet_provisioning  # noqa: PLC0415
    from marina.db import UrlDatabase  # noqa: PLC0415
    from marina.server import MarinaConfig, create_app  # noqa: PLC0415
    from marina.testing import test_database  # noqa: PLC0415
    from starlette.testclient import TestClient  # noqa: PLC0415

    with test_database() as local_url:
        database = UrlDatabase(local_url)
        store = AppletStore(database)
        ensure_applet_provisioning(store.engine)
        store.migrate()
        source = Path(__file__).parents[1] / "apps/rl_progress"
        package = read_applet_package(package_applet(source))
        release = store.publish(package, "test", UUID(APPLET_ID))
        client = TestClient(
            create_app(MarinaConfig(data_root=str(tmp_path), iap_audience=None, database=database)),
            client=("127.0.0.1", 40000),
        )

        def request(origin, method, path, *, json_body=None, **kwargs):
            if method == "GET" and path == "/evaldash/api/runs":
                return []
            response = client.request(method, path, json=json_body)
            response.raise_for_status()
            return response.json()

        def publish_release():
            nonlocal release
            release = store.publish(package, "test", UUID(APPLET_ID), base_version=release.version)

        monkeypatch.setattr("marina.client.marina_request", request)
        try:
            yield client, request, publish_release
        finally:
            client.close()
            store.engine.dispose()


def progress_record() -> dict:
    record = {
        "run_id": "test/control",
        "attempt_id": "native-first",
        "label": "learning-medium lr=2e-06",
        "series": "learning-medium",
        "issue": 123,
        "gate": "test/gate",
        "run_date": "2026-11-02T00:00:00Z",
        "started_at": "2026-11-02T00:00:00Z",
        "completed_at": "2026-11-02T01:00:00Z",
        "elapsed_seconds": 3600,
        "timing_scope": "native_job_start_to_finish",
        "status": "succeeded",
        "optimizer_steps": 25,
        "source_revision": "source-1",
        "data_identity": "data-1",
        "initializer_identity": "model-1",
        "gate_result": {
            "comparisons": [
                {"metric": "score", "role": "objective", "run": 0.4, "control": 0.2, "n_run": 1, "n_control": 4}
            ],
            "invariants": [{"name": "planned_updates", "value": 25, "limit": 25, "holds": True, "replicate": 0}],
            "verdict": "pass",
            "effect": "improves",
            "reason": None,
            "attempt_ids": ["native-first"],
            "replicates": 1,
        },
        "artifact_uri": "s3://example/test",
        "evidence": [],
    }

    return AttemptSummary.model_validate_json(json.dumps(record), strict=True).model_dump(mode="json")


def grader_report() -> dict:
    return {
        "training": {},
        "training_tasks": {},
        "evaluation": {
            "eval_lite/0": [
                {
                    "subset_id": "test/source",
                    "data_sources": ["test/source"],
                    "attempted": 2,
                    "verified": 2,
                    "passed": 1,
                    "verification_coverage": 1.0,
                    "success_rate": 0.5,
                    "verification_statuses": {"verified": 2},
                    "grading_failures": {},
                    "rollout_failures": {},
                    "grading_latency_seconds": {"count": 2, "mean": 2.0, "p95": 2.0, "max": 2.0},
                }
            ]
        },
        "evaluation_tasks": {},
    }
